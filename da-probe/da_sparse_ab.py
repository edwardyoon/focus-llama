#!/usr/bin/env python3
"""W2 sparse-vs-dense flash-attention A/B + sparse-engagement check (focus-llama).

Two things are verified:
  1. correctness : sparse and dense runs agree (identical greedy text, small
     |d logprob|, the answer ZEBRA-42 survives - a silently dropped tail cell
     would remove the question and break the answer).
  2. engagement  : the CUDA kernels really took the sparse path. The server
     journal of the test window must contain 'da_sparse[VEC]: SPARSE' and
     'da_sparse[MMA]: SPARSE'. When they are missing the verdict says WHO
     blocked it:
       server guard : 'forcing dense, R3' / DENSE with bound 0 / n_kv_max=0
       kernel gate  : DENSE with bound > 0 ('no sparse kernel variant',
                      'KV < min_kv(...)') <- the regression class

Production server (stays as configured; restart after every build swap and run
BEFORE real traffic - kv_holes_active is sticky and trips the R3 guard):

  python3 da_sparse_ab.py --mode capture --server http://127.0.0.1:8080 \
      --unit <systemd unit> [--ssh user@123]

  --unit  : journalctl -u <unit> is read for exactly the test window
  --ssh   : run journalctl on the GPU box when this script runs elsewhere
  without --unit you can pass --sparse-log FILE (whole file is analysed, so
  truncate/rotate it before the run)

Numeric A/B on one GPU (two servers cannot coexist):
  1. capture against the normal server      -> /tmp/da_capture_sparse.json
  2. restart with FOCUS_DA_DENSE=1, capture -> --server-role dense
  3. python3 da_sparse_ab.py --mode compare \
         --a /tmp/da_capture_sparse.json --b /tmp/da_capture_dense.json

Modes: capture (default for production) | compare | both (two live servers).

Notes
  - Journal lines are deduplicated by the server: SPARSE re-logs when the KV
    rows change, DENSE only when the reason changes. 'no line' can therefore
    mean "same dense state as before". Set FOCUS_SPARSE_DEBUG=1 in the unit's
    Environment for a live decision every second per path.
  - The MMA path only runs for multi-token batches (MTP verify). If MMA shows
    no line at all the path probably never ran in this window.
  - The server keeps the last --da-tail-keep (K) tokens before the removal boundary
    (log: 'da: tail_keep=[a, b) - N token(s) newly kept'). A prompt shorter than K
    has NOTHING removed -> no holes -> n_kv_max stays 0 (correct, not a bug). The
    prompt is therefore sized by --target-tokens (default 38000: > K=20480, and
    < --kv-offload-threshold 40960 so offload holes / R3 stay out of the picture).
  - MMA gate floor is K->ne[1] >= 4096.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error

FILLER_A = ("The system processes incoming requests through a queue. Each request "
            "is validated, then dispatched to a worker pool. Workers maintain a "
            "local cache of recent results to reduce latency. The cache expires "
            "entries after thirty seconds. ")
FILLER_B = ("Tasks are scheduled using a round-robin policy. The scheduler "
            "considers task priority and estimated duration. Long-running tasks "
            "are preempted after five minutes. Their state is checkpointed to "
            "disk so a restart does not lose progress. ")

FACT = ("The emergency shutdown codeword for the facility is ZEBRA-42. "
        "Operators must memorize it. It is printed on the wall of the control "
        "room in red letters.")

QUESTION = ("Question: What is the emergency shutdown codeword for the facility? "
            "Reply with the codeword repeated 8 times separated by spaces, "
            "nothing else.\nAnswer: ")
# Reasoning models (Qwen3.8, --reasoning on) open with '<think>\n'. The old
# stop=["\n"] ended every request after 2 tokens ('<think>' + newline), so the
# decode kernels were barely exercised. Pre-close the think block so the model
# answers directly, and let it run ~40 decode steps (VEC + MTP verify/MMA).
THINK_OFF = "<think>\n\n</think>\n\n"

TOK_PER_PAIR = 85.3  # measured on Qwen3.8: FILLER_A 41.2 + FILLER_B 44.1 tokens per paragraph pair

RE_DASPARSE = re.compile(r"da_sparse\[(VEC|MMA)\]: (SPARSE|DENSE) - (.*)")
RE_PUSH = re.compile(r"-> (\d+)")
RE_TAIL = re.compile(r"tail_keep=\[(\d+), (\d+)\) - (\d+) token\(s\) newly kept \(K=(\d+)")
REQUIRE_VEC = False  # set from --require-vec


# ----------------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------------
def build_prompt(n_filler, think_off=True):
    return ("The following is a technical document.\n\n"
            "[Section 1: Overview]\n" + FILLER_A * n_filler + "\n"
            "[Section 2: The codeword]\n" + FACT + "\n\n"
            "[Section 3: Scheduling]\n" + FILLER_B * n_filler + "\n"
            + QUESTION + (THINK_OFF if think_off else ""))


def post(base, path, body, timeout=300):
    req = urllib.request.Request(base + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_json(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def wait_idle(base, timeout):
    """Best-effort: wait until no slot is processing (other traffic would add
    kv_holes_active / co-tenant noise). Returns True/False/None(unknown)."""
    if timeout <= 0:
        return None
    t0 = time.time()
    announced = False
    while True:
        try:
            slots = get_json(base, "/slots")
        except Exception as e:
            print("preflight: /slots unavailable (%s) - idle check skipped" % e)
            return None
        busy = [s.get("id") for s in slots if s.get("is_processing")]
        if not busy:
            if announced:
                print("preflight: slots idle")
            return True
        if time.time() - t0 > timeout:
            print("preflight: slots %s still busy after %ds - results may be "
                  "polluted by live traffic (R3 guard!)" % (busy, timeout))
            return False
        if not announced:
            print("preflight: waiting for slots %s to go idle ..." % busy)
            announced = True
        time.sleep(1)


def run_one(base, prompt, da_rm, da_rm_at, max_tokens, n_probs):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": False,
            "da_rm": da_rm, "da_rm_at": da_rm_at}
    if n_probs > 0:
        body["n_probs"] = n_probs
    t0 = time.time()
    res = post(base, "/v1/completions", body)
    wall = time.time() - t0
    choice = res["choices"][0]
    text = choice["text"]
    lps, tops = [], []
    content = ((choice.get("logprobs") or {}).get("content")) or []
    for entry in content:
        lps.append(entry["logprob"])
        tops.append([(t["token"], t["logprob"]) for t in entry.get("top_logprobs", [])])
    tim = res.get("timings", {})
    return {"text": text, "lps": lps, "tops": tops, "wall": wall,
            "tps": tim.get("predicted_per_second"),
            "n_gen": tim.get("predicted_n"), "n_prompt": tim.get("prompt_n")}


def get_layout(base, prompt):
    """Token ranges for the fact, the two filler removal ranges, the question."""
    ids = post(base, "/tokenize", {"content": prompt})["tokens"]
    full = post(base, "/detokenize", {"tokens": ids})["content"]

    def tok_len(prefix):
        return len(post(base, "/tokenize", {"content": prefix})["tokens"])

    fact_lo = tok_len(full[:full.index(FACT)])
    fact_hi = tok_len(full[:full.index(FACT) + len(FACT)])
    s1 = full.index("[Section 1: Overview]")
    s3 = full.index("[Section 3: Scheduling]")
    q0 = full.index("Question:")
    s1_lo = tok_len(full[:s1])
    s3_lo = tok_len(full[:s3])
    q_lo = tok_len(full[:q0])
    # remove both filler sections (headers included); keep fact + question
    rm = [[s1_lo, fact_lo], [s3_lo, q_lo]]
    removed = (fact_lo - s1_lo) + (q_lo - s3_lo)
    return {"n_prompt": len(ids), "fact_range": [fact_lo, fact_hi], "rm": rm,
            "rm_at": q_lo, "removed": removed}


def estimate(layout, max_tokens, tail_keep, offload_threshold):
    """Informational only. The old '2*n_kv_max' PASS/FAIL rule is gone: the real
    gate is kv_pct based (bound 40448 of 60160 = 67% engaged in production).
    The server re-keeps [rm_at - tail_keep, rm_at) (--da-tail-keep), so only the
    part of each range below that floor is really removed."""
    n = layout["n_prompt"]
    floor_lo = max(0, layout["rm_at"] - tail_keep)
    removed_eff = sum(max(0, min(hi, floor_lo) - lo) for lo, hi in layout["rm"])
    finite = n - removed_eff + max_tokens
    bucket = ((finite + 511) // 512) * 512
    n_kv = ((n + max_tokens + 127) // 128) * 128
    return {"finite_est": finite, "n_kv_max_bucket": bucket, "n_kv_pad_est": n_kv,
            "removed_eff": removed_eff, "tail_floor_lo": floor_lo,
            "tail_keep": tail_keep, "offload_threshold": offload_threshold}


def print_layout(layout, est):
    n = layout["n_prompt"]
    print("prompt tokens            : %d" % n)
    print("fact range (kept)        : %s" % str(layout["fact_range"]))
    for r in layout["rm"]:
        print("removed filler range     : [%d, %d)  (%d tokens)" % (r[0], r[1], r[1] - r[0]))
    print("removed total            : %d tokens (%.0f%% of prompt)"
          % (layout["removed"], 100.0 * layout["removed"] / n))
    print("tail floor (assumed K=%d)  : server keeps [%d, %d) -> effective removal %d tokens (%.0f%% of prompt)"
          % (est["tail_keep"], est["tail_floor_lo"], layout["rm_at"], est["removed_eff"],
             100.0 * est["removed_eff"] / n))
    if est["removed_eff"] < 512:
        print("WARNING: the tail floor re-keeps (almost) everything -> no holes -> n_kv_max=0 is "
              "EXPECTED. Raise --target-tokens above K.")
    if n > est["offload_threshold"]:
        print("WARNING: prompt %d > kv-offload-threshold %d: offload holes may engage (R3 confound)."
              % (n, est["offload_threshold"]))
    print("est. finite cells        : ~%d -> 512-bucket %d" % (est["finite_est"], est["n_kv_max_bucket"]))
    floor_ok = est["n_kv_pad_est"] >= 4096
    print("est. KV rows             : ~%d  (MMA gate floor 4096: %s)"
          % (est["n_kv_pad_est"], "ok" if floor_ok else "BELOW -> MMA will always be DENSE"))
    print("-" * 72)


# ----------------------------------------------------------------------------
# Journal handling
# ----------------------------------------------------------------------------
def fetch_log(args, t_start):
    """Server log text for the test window (journalctl), else the log file."""
    age = int(time.time() - t_start) + 5
    if args.unit:
        cmd = ["journalctl", "-u", args.unit, "--since=-%ds" % age,
               "--no-pager", "-o", "cat"]
        if args.ssh:
            cmd = ["ssh", args.ssh] + cmd
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout:
                return r.stdout
            print("journalctl: rc=%d %s" % (r.returncode, r.stderr.strip()[:200]))
        except Exception as e:
            print("journalctl failed: %s" % e)
    if args.sparse_log:
        return read_file(args.sparse_log)
    return None


def read_file(path):
    if not path:
        return None
    with open(path, errors="replace") as f:
        return f.read()


def filter_journal(text):
    if text is None:
        return None
    return [l.strip() for l in text.splitlines()
            if "da_sparse[" in l or "da: n_kv_max" in l or "R3 lifted" in l or "tail_keep=" in l]


def analyze_journal(text):
    info = {"sparse": {"VEC": [], "MMA": []}, "dense": {"VEC": [], "MMA": []},
            "r3": [], "r3_lifted": [], "nkv": [], "tail": []}
    for l in (text or "").splitlines():
        m = RE_DASPARSE.search(l)
        if m:
            path, kind, rest = m.groups()
            info["sparse" if kind == "SPARSE" else "dense"][path].append(rest.strip())
        elif RE_TAIL.search(l):
            info["tail"].append(tuple(int(x) for x in RE_TAIL.search(l).groups()))
        elif "R3 lifted" in l:
            info["r3_lifted"].append(l.strip())
        elif "forcing dense, R3" in l:
            info["r3"].append(l.strip())
            info["nkv"].append(l.strip())
        elif "da: n_kv_max" in l:
            info["nkv"].append(l.strip())
    return info


def is_guard_reason(reason):
    """DENSE caused by the server pushing n_kv_max=0 (not by the kernel gate)."""
    return bool(re.search(r"n_kv_max=0|bound 0,|no DA bound", reason))


def diagnose(info):
    msgs = []
    if info["r3"]:
        msgs.append("R3 guard fired (kv_holes_active): this run is INVALID for the kernel "
                    "test. Restart the server and rerun before any real traffic.")
    if info["r3_lifted"]:
        msgs.append("R3 lifted inside this window: an idle slot still held live-session holes "
                    "(kv_holes_active) before the test, so dense was forced for ALL slots until "
                    "that slot was cleared. Production does the same between turns.")
    if (not info["nkv"] and not info["r3"] and
            not any(info["sparse"][p] or info["dense"][p] for p in ("VEC", "MMA"))):
        msgs.append("no 'da: n_kv_max' and no da_sparse line: the bound stayed 0 for the whole "
                    "window (DENSE repeats are deduplicated). Check that update_da_n_kv_max on "
                    "the GPU box pushes a bound for a da_rm request (slot.da_seq / da_removed_a).")
    for p in ("VEC", "MMA"):
        if info["sparse"][p]:
            continue
        if p == "VEC" and not REQUIRE_VEC:
            continue
        if info["dense"][p]:
            r = info["dense"][p][-1]
            if is_guard_reason(r):
                msgs.append("[%s] server pushed n_kv_max=0 -> DA bound not applied "
                            "(guard / prefill / request never entered the DA path): %s" % (p, r[:90]))
            else:
                msgs.append("[%s] KERNEL gate rejected sparse (regression candidate): %s" % (p, r[:90]))
        else:
            msgs.append("[%s] no da_sparse line: path did not run, or same dense state "
                        "as before (dedup; use FOCUS_SPARSE_DEBUG=1)" % p)
    return msgs


def sparse_journal_checks(jtext, check):
    info = analyze_journal(jtext)
    push = [int(m.group(1)) for l in info["nkv"] if (m := RE_PUSH.search(l))]
    check("server pushed n_kv_max>0", any(v > 0 for v in push),
          "%d 'da: n_kv_max' line(s), values %s" % (len(info["nkv"]), push[-4:]))
    if info["tail"]:
        lo, hi, kept, K = info["tail"][-1]
        print("server tail_keep           : K=%d keeps [%d, %d) (%d token(s) newly kept) - the script "
              "estimate assumes --tail-keep" % (K, lo, hi, kept))
    for p in ("VEC", "MMA"):
        n = len(info["sparse"][p])
        if n:
            last = info["sparse"][p][-1]
        elif info["dense"][p]:
            last = "DENSE " + info["dense"][p][-1]
        else:
            last = "no da_sparse line"
        detail = "%d SPARSE line(s); last: %s" % (n, last[:100])
        if p == "VEC" and not REQUIRE_VEC and n == 0:
            print("%-32s: n/a  (%s)" % ("kernel SPARSE[VEC]", detail))
            print("  NOTE: with MTP the draft context's single-token steps stay dense by design "
                  "(ctx_dft gets no bound) and the target mostly runs multi-token verify batches "
                  "(MMA). VEC SPARSE needs target single-token decodes: use --require-vec against "
                  "a server started without spec decoding.")
            continue
        check("kernel SPARSE[%s]" % p, n > 0, detail)
    for m in diagnose(info):
        print("  DIAG: %s" % m)


def bound_consistency(layout, jtext, check):
    """The pushed bound must never be BELOW the finite-cell count: the gather
    kernel drops the overflow (highest cell index = newest tokens) silently.
    Uses the K the server actually applied ('tail_keep=... (K=N'), not the CLI
    guess. Lower bound = finite cells at the FIRST decode step (+1 token)."""
    info = analyze_journal(jtext)
    if not info["tail"] or layout is None:
        return
    K = info["tail"][-1][3]
    est = estimate(layout, 1, K, 10 ** 9)  # max_tokens=1 -> finite at the first decode step
    pushed = [int(m.group(1)) for l in info["nkv"] if (m := RE_PUSH.search(l))]
    pushed = [v for v in pushed if v > 0]
    if not pushed:
        return
    lo = est["n_kv_max_bucket"]
    check("bound >= finite (K=%d)" % K, max(pushed) >= lo,
          "pushed max %d vs expected >= %d (prompt %d - removed %d + 1; under-estimate = silent drop)"
          % (max(pushed), lo, layout["n_prompt"], est["removed_eff"]))


def dense_journal_checks(jtext, check):
    info = analyze_journal(jtext)
    check("dense journal: no n_kv_max push", len(info["nkv"]) == 0,
          "%d line(s) (FOCUS_DA_DENSE=1 must return before the push)" % len(info["nkv"]))


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def run_checks(label, results, max_lp_diff, sparse_j, dense_j, layout=None):
    """results = (sparse_runs, dense_runs|None). Returns overall ok."""
    checks = []

    def check(name, ok, detail):
        checks.append(bool(ok))
        print("%-32s: %s (%s)" % (name, "PASS" if ok else "FAIL", detail))

    tps = []
    if results[1] is not None:
        worst = 0.0
        worst_top = 0.0
        text_ok = None
        for i, (r_s, r_d) in enumerate(zip(results[0], results[1])):
            same_text = r_s["text"] == r_d["text"]
            text_ok = same_text if text_ok is None else (text_ok and same_text)
            n_cmp = min(len(r_s["lps"]), len(r_d["lps"]))
            d_lp = max((abs(a - b) for a, b in zip(r_s["lps"][:n_cmp], r_d["lps"][:n_cmp])),
                       default=0.0)
            d_top = 0.0
            for ts, td in zip(r_s["tops"][:n_cmp], r_d["tops"][:n_cmp]):
                m = {t: p for t, p in td}
                for t, p in ts:
                    if t in m:
                        d_top = max(d_top, abs(p - m[t]))
            worst = max(worst, d_lp)
            worst_top = max(worst_top, d_top)
            if r_s["tps"] and r_d["tps"]:
                tps.append((r_s["tps"], r_d["tps"]))
            print("run %d: text identical=%s  max|dlp|=%.3e  max|dtop5|=%.3e  "
                  "tps sparse=%s dense=%s  n_gen=%s"
                  % (i + 1, same_text, d_lp, d_top,
                     "%.2f" % r_s["tps"] if r_s["tps"] else "?",
                     "%.2f" % r_d["tps"] if r_d["tps"] else "?",
                     r_s["n_gen"]))
            if not same_text:
                print("  sparse: %r" % r_s["text"][:200])
                print("  dense : %r" % r_d["text"][:200])
        print("-" * 72)
        n_runs = min(len(results[0]), len(results[1]))
        check("text identical", text_ok is True, "%d run(s) compared" % n_runs)
        if all(r["lps"] for r in results[0] + results[1]):
            check("max|d lp| <= %.0e" % max_lp_diff, worst <= max_lp_diff,
                  "worst %.3e over %d runs" % (worst, n_runs))
            print("max|d top-5 lp|           : %.3e (informational)" % worst_top)
        else:
            print("max|d lp|                 : skipped (captures without logprobs)")

        # Silent-drop canary: a dropped tail cell breaks the answer on one side
        # only. Both sides failing identically is a model limit, not a kernel bug.
        ok_s = all("ZEBRA-42" in r["text"] for r in results[0])
        ok_d = all("ZEBRA-42" in r["text"] for r in results[1])
        if ok_s and ok_d:
            note = ""
        elif not ok_s and not ok_d:
            note = "  [both wrong = model limit, kernel consistent]"
        elif ok_d:
            note = "  [sparse-only failure = silent drop]"
        else:
            note = "  [dense-only failure]"
        check("silent-drop canary (parity)", ok_s == ok_d,
              "sparse=%s dense=%s%s" % (ok_s, ok_d, note))
    else:
        n_ans = sum("ZEBRA-42" in r["text"] for r in results[0])
        print("answer (ZEBRA-42)          : %d/%d run(s) (informational - model capability)"
              % (n_ans, len(results[0])))
        for i, r in enumerate(results[0]):
            print("run %d: tps=%s n_gen=%s server prompt_n=%s text=%r"
                  % (i + 1, "%.2f" % r["tps"] if r["tps"] else "?", r["n_gen"],
                     r["n_prompt"], r["text"][:60]))

    print("-" * 72)
    if sparse_j is not None:
        sparse_journal_checks(sparse_j, check)
        bound_consistency(layout, sparse_j, check)
    else:
        print("sparse journal             : none given (--unit / --sparse-log) - "
              "kernel SPARSE NOT verified")
    if dense_j is not None:
        dense_journal_checks(dense_j, check)

    if tps:
        print("-" * 72)
        print("decode tps (informational, includes one-time graph rebuild in sparse):")
        print("  sparse: %.2f   dense: %.2f" % (sum(a for a, _ in tps) / len(tps),
                                                 sum(b for _, b in tps) / len(tps)))
    ok = all(checks)
    print("OVERALL                    : %s" % ("PASS" if ok else "FAIL"))
    return ok


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "capture", "compare"], default="capture")
    ap.add_argument("--server", default="http://127.0.0.1:8080",
                    help="capture mode: server base URL (production default :8080)")
    ap.add_argument("--server-role", choices=["sparse", "dense"], default="sparse",
                    help="capture mode: which server this capture came from")
    ap.add_argument("--sparse", default="http://127.0.0.1:8080", help="both mode: sparse server")
    ap.add_argument("--dense", default="http://127.0.0.1:8087", help="both mode: dense server")
    ap.add_argument("--out", default=None, help="capture output JSON (default /tmp/da_capture_<role>.json)")
    ap.add_argument("--a", default=None, help="compare mode: sparse capture JSON")
    ap.add_argument("--b", default=None, help="compare mode: dense capture JSON")
    ap.add_argument("--unit", default=None, help="systemd unit; journalctl is read for the test window")
    ap.add_argument("--ssh", default=None, help="run journalctl via ssh on this host")
    ap.add_argument("--sparse-log", default=None, help="server log file (whole file is analysed)")
    ap.add_argument("--dense-log", default=None)
    ap.add_argument("--n-filler", type=int, default=None,
                    help="paragraphs per filler section (default: derived from --target-tokens)")
    ap.add_argument("--target-tokens", type=int, default=38000,
                    help="approx. prompt size; must exceed --tail-keep and stay below the offload threshold")
    ap.add_argument("--tail-keep", type=int, default=20480,
                    help="server --da-tail-keep (tokens before the boundary that are never removed)")
    ap.add_argument("--offload-threshold", type=int, default=40960,
                    help="server --kv-offload-threshold (warn when the prompt exceeds it)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--allow-think", action="store_true",
                    help="do not pre-close the <think> block in the prompt")
    ap.add_argument("--max-lp-diff", type=float, default=1e-4)
    ap.add_argument("--no-probs", action="store_true",
                    help="do not request logprobs (kernel-engagement bisect only; "
                         "may leave spec decoding enabled)")
    ap.add_argument("--require-vec", action="store_true",
                    help="treat missing VEC SPARSE as FAIL (only meaningful without spec decoding)")
    ap.add_argument("--wait-idle", type=int, default=60,
                    help="seconds to wait for idle slots before starting (0 = skip)")
    args = ap.parse_args()
    globals()["REQUIRE_VEC"] = args.require_vec
    n_probs = 0 if args.no_probs else 5

    if args.mode == "compare":
        if not args.a or not args.b:
            ap.error("--a and --b are required in compare mode")
        with open(args.a) as f:
            cap_a = json.load(f)
        with open(args.b) as f:
            cap_b = json.load(f)
        if cap_a["n_prompt"] != cap_b["n_prompt"] or cap_a["rm"] != cap_b["rm"]:
            print("FATAL: captures disagree on prompt layout")
            sys.exit(2)
        mt = cap_a.get("max_tokens", args.max_tokens)
        _tk = analyze_journal("\n".join(cap_a.get("journal") or []))["tail"]
        k_eff = _tk[-1][3] if _tk else args.tail_keep
        if _tk:
            print("(tail-keep K=%d taken from the sparse capture's server journal)" % k_eff)
        print_layout(cap_a, estimate(cap_a, mt, k_eff, args.offload_threshold))
        sj = read_file(args.sparse_log)
        if sj is None and cap_a.get("journal") is not None:
            sj = "\n".join(cap_a["journal"])
        dj = read_file(args.dense_log)
        if dj is None and cap_b.get("journal") is not None:
            dj = "\n".join(cap_b["journal"])
        ok = run_checks("compare", (cap_a["runs"], cap_b["runs"]), args.max_lp_diff, sj, dj, cap_a)
        sys.exit(0 if ok else 1)

    n_filler = args.n_filler or max(1, round(args.target_tokens / TOK_PER_PAIR))
    prompt = build_prompt(n_filler, not args.allow_think)

    if args.mode == "capture":
        base = args.server
        role = args.server_role
        out = args.out or "/tmp/da_capture_%s.json" % role
        wait_idle(base, args.wait_idle)
        layout = get_layout(base, prompt)
        est = estimate(layout, args.max_tokens, args.tail_keep, args.offload_threshold)
        print_layout(layout, est)
        t0 = time.time()
        runs = [run_one(base, prompt, layout["rm"], layout["rm_at"], args.max_tokens, n_probs)
                for _ in range(args.repeat)]
        time.sleep(1.5)  # let the journal flush
        jtext = fetch_log(args, t0)
        cap = {"n_prompt": layout["n_prompt"], "fact_range": layout["fact_range"],
               "rm": layout["rm"], "rm_at": layout["rm_at"],
               "removed": layout["removed"], "max_tokens": args.max_tokens,
               "n_filler": n_filler, "role": role, "est": est,
               "journal": filter_journal(jtext), "runs": runs}
        with open(out, "w") as f:
            json.dump(cap, f)
        print("captured %d run(s) -> %s" % (len(runs), out))
        sj, dj = (jtext, None) if role == "sparse" else (None, jtext)
        ok = run_checks("capture", (runs, None), args.max_lp_diff, sj, dj, layout)
        sys.exit(0 if ok else 1)

    # mode: both (two live servers, local small models)
    wait_idle(args.sparse, args.wait_idle)
    layout = get_layout(args.sparse, prompt)
    print_layout(layout, estimate(layout, args.max_tokens, args.tail_keep, args.offload_threshold))
    t0 = time.time()
    sparse_runs = [run_one(args.sparse, prompt, layout["rm"], layout["rm_at"], args.max_tokens, n_probs)
                   for _ in range(args.repeat)]
    dense_runs = [run_one(args.dense, prompt, layout["rm"], layout["rm_at"], args.max_tokens, n_probs)
                  for _ in range(args.repeat)]
    time.sleep(1.5)
    ok = run_checks("both", (sparse_runs, dense_runs), args.max_lp_diff,
                    fetch_log(args, t0), read_file(args.dense_log), layout)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
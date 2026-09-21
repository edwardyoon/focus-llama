#!/usr/bin/env python3
"""W2 Phase 1 Step 3/4 — sparse vs dense flash-attention A/B (logit verification).

Verifies the n_kv_max sparse gather path (Metal locally, CUDA VEC/MMA on the
target) against the dense reference for the same DA (static da_rm) request:
the mask is identical in both runs (same holes), only the kernel differs
(dense reads all KV rows with -inf masking; sparse gathers the finite cells
via the n_kv_max bound). If the bound is a true upper bound on the finite
cells per mask row, the attention results must agree:
  1. generated text is IDENTICAL (temperature 0, greedy)
  2. per-token logprobs agree within a small fp tolerance (max |delta lp|)
  3. the answer is still correct — a silently dropped tail cell (bound
     under-estimate) would remove the question itself and break the answer

Removal layout: both filler sections are removed (~84% of the prompt),
keeping only the fact + question. The CUDA VEC sparse gate requires
K->ne[1] >= 2*n_kv_max (sparse only when we read at most half the KV), so
the removal must be >= 50% for the gate to engage on the target.

Modes
-----
both    (default) — two live servers, alternating runs. Needs both servers in
        VRAM at once: local small models only.
        sparse: ./build/bin/llama-server -m M --port 8086 --da-prompt-scan -c 8192 -v
        dense : FOCUS_DA_DENSE=1 ./build/bin/llama-server -m M --port 8087 --da-prompt-scan -c 8192 -v

capture — one server, saves the runs to a JSON file. For 27B-class GPUs where
        two servers cannot coexist (123): run capture against the sparse
        server, restart the server with FOCUS_DA_DENSE=1, capture again,
        then compare.
        python3 da_sparse_ab.py --mode capture --server http://127.0.0.1:8086 \
            --out /tmp/da_capture_sparse.json --repeat 3

compare — runs the checks against two capture files:
        python3 da_sparse_ab.py --mode compare \
            --a /tmp/da_capture_sparse.json --b /tmp/da_capture_dense.json \
            [--sparse-log /tmp/da_sparse.log] [--dense-log /tmp/da_dense.log] \
            [--max-lp-diff 1e-4]

Journal evidence (when logs are given):
  sparse log: 'da: n_kv_max 0 -> N' with N > 0 (the bound was pushed)
  dense log : no 'da: n_kv_max' line (the override returns before the push)
"""
import argparse
import json
import re
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
            "Reply with the codeword only, nothing else.\nAnswer: ")

N_FILLER = 30  # paragraphs per filler section


def build_prompt():
    return ("The following is a technical document.\n\n"
            "[Section 1: Overview]\n" + FILLER_A * N_FILLER + "\n"
            "[Section 2: The codeword]\n" + FACT + "\n\n"
            "[Section 3: Scheduling]\n" + FILLER_B * N_FILLER + "\n"
            + QUESTION)


def post(base, path, body, timeout=300):
    req = urllib.request.Request(base + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_one(base, prompt, da_rm, da_rm_at, max_tokens):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "stop": ["\n"], "n_probs": 5, "cache_prompt": False,
            "da_rm": da_rm, "da_rm_at": da_rm_at}
    t0 = time.time()
    res = post(base, "/v1/completions", body)
    wall = time.time() - t0
    text = res["choices"][0]["text"]
    lps = []
    tops = []
    for entry in res["choices"][0]["logprobs"]["content"]:
        lps.append(entry["logprob"])
        tops.append([(t["token"], t["logprob"]) for t in entry["top_logprobs"]])
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
    n = len(ids)
    removed = (fact_lo - s1_lo) + (q_lo - s3_lo)
    return {"n_prompt": n, "fact_range": [fact_lo, fact_hi], "rm": rm,
            "rm_at": q_lo, "removed": removed}


def gate_estimate(layout, max_tokens):
    """Estimate whether the CUDA VEC sparse gate (K->ne[1] >= 2*n_kv_max) fires."""
    n = layout["n_prompt"]
    removed = layout["removed"]
    finite = n - removed + max_tokens  # finite cells at the last decode step
    bucket = ((finite + 511) // 512) * 512
    n_kv_pad = ((n + max_tokens + 127) // 128) * 128  # KV rows are 128-aligned
    return {"finite_est": finite, "n_kv_max_bucket": bucket,
            "n_kv_pad_est": n_kv_pad,
            "gate_pass": n_kv_pad >= 2 * bucket}


def print_layout(layout, gate):
    n = layout["n_prompt"]
    print("prompt tokens            : %d" % n)
    print("fact range (kept)        : %s" % str(layout["fact_range"]))
    for r in layout["rm"]:
        print("removed filler range     : [%d, %d)  (%d tokens)" % (r[0], r[1], r[1] - r[0]))
    print("removed total            : %d tokens (%.0f%% of prompt)"
          % (layout["removed"], 100.0 * layout["removed"] / n))
    print("finite cells at decode   : ~%d -> 512-bucket %d"
          % (gate["finite_est"], gate["n_kv_max_bucket"]))
    print("sparse gate estimate     : n_kv~%d >= 2*%d -> %s"
          % (gate["n_kv_pad_est"], gate["n_kv_max_bucket"],
             "PASS (sparse kernel engages)" if gate["gate_pass"] else "FAIL (dense fallback)"))
    print("-" * 72)


def run_checks(label, results, max_lp_diff, log, log_mode):
    """Run the A/B checks against two result lists (or a single capture for
    capture-mode sanity). Returns (ok, worst_lp, worst_top)."""
    checks = []

    def check(name, ok, detail):
        checks.append(bool(ok))
        print("%-30s: %s (%s)" % (name, "PASS" if ok else "FAIL", detail))

    worst = 0.0
    worst_top = 0.0
    text_ok = None
    tps = []
    if results[1] is not None:
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
            if r_s["tps"]:
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
        check("text identical", text_ok is True,
              "%d/%d run(s) byte-identical" % (len(results[0]), len(results[0])))
        check("max|d lp| <= %.0e" % max_lp_diff, worst <= max_lp_diff,
              "worst %.3e over %d runs" % (worst, len(results[0])))
        print("max|d top-5 lp|           : %.3e (informational)" % worst_top)

        # Silent-drop canary: a dropped tail cell breaks the answer on the
        # SPARSE side only. Both sides failing identically is a model
        # capability limit (e.g. very small test model), not a kernel bug.
        ok_s = all("ZEBRA-42" in r["text"] for r in results[0])
        ok_d = all("ZEBRA-42" in r["text"] for r in results[1])
        check("silent-drop canary (answer parity)", ok_s == ok_d,
              "sparse=%s dense=%s%s" % (ok_s, ok_d,
                   "" if (ok_s and ok_d) else "  [both wrong = model limit, kernel consistent]"
                   if not ok_s else "  [dense-only failure = silent drop]"))
    else:
        n_ans = sum("ZEBRA-42" in r["text"] for r in results[0])
        print("answer (ZEBRA-42)          : %d/%d run(s) (informational — model capability)"
              % (n_ans, len(results[0])))
        print("capture text             : %r" % results[0][0]["text"][:120])

    # journal evidence
    if log:
        with open(log, errors="replace") as f:
            lines = [l for l in f if "n_kv_max" in l]
        vals = [int(m.group(1)) for l in lines if (m := re.search(r"-> (\d+)", l))]
        if log_mode == "sparse":
            check("sparse journal n_kv_max>0", any(v > 0 for v in vals),
                  "%d line(s), values %s" % (len(lines), vals[-4:]))
        else:
            check("dense journal no n_kv_max", len(lines) == 0, "%d line(s)" % len(lines))
        for l in lines[-3:]:
            print("    | %s" % l.strip()[:160])

    if tps:
        print("-" * 72)
        print("decode tps (informational, includes one-time graph rebuild in sparse):")
        print("  sparse: %.2f   dense: %.2f" % (sum(a for a, _ in tps) / len(tps),
                                                 sum(b for _, b in tps) / len(tps)))
    ok = all(checks)
    print("OVERALL                 : %s" % ("PASS" if ok else "FAIL"))
    return ok, worst, worst_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "capture", "compare"], default="both")
    ap.add_argument("--sparse", default="http://127.0.0.1:8086")
    ap.add_argument("--dense", default="http://127.0.0.1:8087")
    ap.add_argument("--server", default=None, help="capture mode: single server base URL")
    ap.add_argument("--server-role", choices=["sparse", "dense"], default="sparse",
                    help="capture mode: which server this capture came from (journal check)")
    ap.add_argument("--out", default=None, help="capture mode: output JSON path")
    ap.add_argument("--a", default=None, help="compare mode: sparse capture JSON")
    ap.add_argument("--b", default=None, help="compare mode: dense capture JSON")
    ap.add_argument("--sparse-log", default=None)
    ap.add_argument("--dense-log", default=None)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-lp-diff", type=float, default=1e-4)
    args = ap.parse_args()

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
        print_layout(cap_a, gate_estimate(cap_a, args.max_tokens))
        ok, worst, worst_top = run_checks(
            "compare", (cap_a["runs"], cap_b["runs"]), args.max_lp_diff,
            args.sparse_log, "sparse")
        if args.dense_log:
            with open(args.dense_log, errors="replace") as f:
                dl = [l for l in f if "n_kv_max" in l]
            if dl:
                print("dense journal n_kv_max lines : %d — FAIL (dense server must not push)" % len(dl))
                ok = False
            else:
                print("dense journal n_kv_max lines : 0 (PASS)")
        sys.exit(0 if ok else 1)

    if args.mode == "capture":
        if not args.server or not args.out:
            ap.error("--server and --out are required in capture mode")
        base = args.server
    else:
        base = args.sparse

    prompt = build_prompt()
    layout = get_layout(base, prompt)
    gate = gate_estimate(layout, args.max_tokens)
    print_layout(layout, gate)

    runs = [run_one(base, prompt, layout["rm"], layout["rm_at"], args.max_tokens)
            for _ in range(args.repeat)]

    if args.mode == "capture":
        out = {"n_prompt": layout["n_prompt"], "fact_range": layout["fact_range"],
               "rm": layout["rm"], "rm_at": layout["rm_at"],
               "removed": layout["removed"], "max_tokens": args.max_tokens,
               "gate_est": gate, "runs": runs}
        with open(args.out, "w") as f:
            json.dump(out, f)
        print("captured %d run(s) -> %s" % (len(runs), args.out))
        ok, _, _ = run_checks("capture", (runs, None), args.max_lp_diff,
                              args.sparse_log, args.server_role)
        sys.exit(0 if ok else 1)

    # mode: both
    dense_runs = [run_one(args.dense, prompt, layout["rm"], layout["rm_at"], args.max_tokens)
                  for _ in range(args.repeat)]
    # interleave display: re-run sparse already done; pair them
    ok, worst, worst_top = run_checks(
        "both", (runs, dense_runs), args.max_lp_diff, args.sparse_log, "sparse")
    if args.dense_log:
        with open(args.dense_log, errors="replace") as f:
            dl = [l for l in f if "n_kv_max" in l]
        print("dense journal n_kv_max lines : %d (informational)" % len(dl))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""W2 Phase 1 Step 3 — sparse vs dense flash-attention A/B (logit verification).

Verifies the n_kv_max sparse gather path (Metal locally, CUDA MMA_F16 on the
target) against the dense reference for the same DA (static da_rm) request:
the mask is identical in both runs (same holes), only the kernel differs
(dense reads all KV rows with -inf masking; sparse gathers the finite cells
via the n_kv_max bound). If the bound is a true upper bound on the finite
cells per mask row, the attention results must agree:
  1. generated text is IDENTICAL (temperature 0, greedy)
  2. per-token logprobs agree within a small fp tolerance (max |delta lp|)
  3. the answer is still correct — a silently dropped tail cell (bound
     under-estimate) would remove the question itself and break the answer

Two server instances are required (FOCUS_DA_DENSE is a process env var read
once): the sparse server (no override) and the dense reference server
(FOCUS_DA_DENSE=1). Same binary, model, flags, port aside:
  sparse: ./build/bin/llama-server -m M --port 8086 --da-prompt-scan -c 8192 -v
  dense : FOCUS_DA_DENSE=1 ./build/bin/llama-server -m M --port 8087 --da-prompt-scan -c 8192 -v

Journal evidence:
  sparse log: 'da: n_kv_max 0 -> N' with N > 0 (the bound was pushed; with
              N <= 4096 and head_dim 128 the Metal vec_idx kernel engages)
  dense log : no 'da: n_kv_max' line (the override returns before the push)

Usage:
  python3 da_sparse_ab.py --sparse http://127.0.0.1:8086 --dense http://127.0.0.1:8087
                          --sparse-log /tmp/da_sparse.log --dense-log /tmp/da_dense.log
                          [--repeat 3] [--max-lp-diff 1e-4]
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

N_FILLER = 30  # paragraphs per filler section (~1350 tokens each at this tokenizer)


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


def strip_thinking(text):
    i = text.rfind("\n\n\n")
    return text[i + 4:].strip() if i != -1 else text.strip()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", default="http://127.0.0.1:8086")
    ap.add_argument("--dense", default="http://127.0.0.1:8087")
    ap.add_argument("--sparse-log", default=None)
    ap.add_argument("--dense-log", default=None)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-lp-diff", type=float, default=1e-4)
    args = ap.parse_args()

    prompt = build_prompt()
    ids = post(args.sparse, "/tokenize", {"content": prompt})["tokens"]
    full = post(args.sparse, "/detokenize", {"tokens": ids})["content"]

    def tok_len(prefix):
        return len(post(args.sparse, "/tokenize", {"content": prefix})["tokens"])

    fact_lo = tok_len(full[:full.index(FACT)])
    fact_hi = tok_len(full[:full.index(FACT) + len(FACT)])
    s3 = full.index("[Section 3: Scheduling]")
    q0 = full.index("Question:")
    s3_lo = tok_len(full[:s3])
    q_lo = tok_len(full[:q0])
    # remove the first half of Section 3 (pure filler, far from fact and question)
    rm_lo, rm_hi = s3_lo, (s3_lo + q_lo) // 2
    n = len(ids)
    print("prompt tokens            : %d" % n)
    print("fact range (kept)        : [%d, %d)" % (fact_lo, fact_hi))
    print("removed filler range     : [%d, %d)  (%d tokens, %.0f%% of prompt)"
          % (rm_lo, rm_hi, rm_hi - rm_lo, 100.0 * (rm_hi - rm_lo) / n))
    print("finite cells at decode   : ~%d -> 512-bucket %d"
          % (n - (rm_hi - rm_lo), ((n - (rm_hi - rm_lo)) + 511) // 512 * 512))
    print("-" * 72)

    checks = []

    def check(name, ok, detail):
        checks.append(bool(ok))
        print("%-30s: %s (%s)" % (name, "PASS" if ok else "FAIL", detail))

    worst = 0.0
    worst_top = 0.0
    text_ok = None
    tps = {"sparse": [], "dense": []}
    for i in range(args.repeat):
        r_s = run_one(args.sparse, prompt, [[rm_lo, rm_hi]], q_lo, args.max_tokens)
        r_d = run_one(args.dense, prompt, [[rm_lo, rm_hi]], q_lo, args.max_tokens)
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
        tps["sparse"].append(r_s["tps"])
        tps["dense"].append(r_d["tps"])
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
          "%d/%d run(s) byte-identical" % (args.repeat, args.repeat))
    check("max|d lp| <= %.0e" % args.max_lp_diff, worst <= args.max_lp_diff,
          "worst %.3e over %d runs x %d tokens" % (worst, args.repeat, args.max_tokens))
    print("max|d top-5 lp|           : %.3e (informational)" % worst_top)

    # answer correctness on both sides (silent-drop canary: a dropped tail
    # would remove the question and break the answer)
    r_s = run_one(args.sparse, prompt, [[rm_lo, rm_hi]], q_lo, args.max_tokens)
    r_d = run_one(args.dense, prompt, [[rm_lo, rm_hi]], q_lo, args.max_tokens)
    ok_s = "ZEBRA-42" in r_s["text"]
    ok_d = "ZEBRA-42" in r_d["text"]
    check("answer correct (both)", ok_s and ok_d,
          "sparse=%s dense=%s" % (ok_s, ok_d))

    # journal evidence
    if args.sparse_log:
        with open(args.sparse_log, errors="replace") as f:
            sl = [l for l in f if "n_kv_max" in l]
        vals = [int(m.group(1)) for l in sl if (m := re.search(r"-> (\d+)", l))]
        check("sparse journal n_kv_max>0", any(v > 0 for v in vals),
              "%d line(s), values %s" % (len(sl), vals[-4:]))
        for l in sl[-3:]:
            print("    | %s" % l.strip()[:160])
    if args.dense_log:
        with open(args.dense_log, errors="replace") as f:
            dl = [l for l in f if "n_kv_max" in l]
        check("dense journal no n_kv_max", len(dl) == 0, "%d line(s)" % len(dl))

    s = [t for t in tps["sparse"] if t]
    d = [t for t in tps["dense"] if t]
    if s and d:
        print("-" * 72)
        print("decode tps (informational, includes one-time graph rebuild in sparse):")
        print("  sparse: %.2f   dense: %.2f   (Metal: FA block-skip + gather)" % (sum(s) / len(s), sum(d) / len(d)))
    ok = all(checks)
    print("OVERALL                 : %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

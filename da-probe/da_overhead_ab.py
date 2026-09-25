#!/usr/bin/env python3
"""DA reasoning overhead (2-pass) hypothesis verification + measurement.

Hypothesis: the DA protocol splits inference into two passes -
  (1) global mode: "find where the answer is" (emits <focus magic_chunks="N">)
  (2) focus mode:  "re-reason on that chunk"
- so even though KV reads drop, a DA request generates MORE tokens than a
vanilla one-pass answer. Metric: DA efficiency = saved_KV_work /
additional_generation_work. The transition count (global<->focus<->local
bouncing) is the key loss signal.

Arms (same 5-chunk + filler document, reused from da_ab_harness.py):
  A  vanilla  - plain system text + direct question, no da_* fields
  C  DA sparse - harness tag instruction + da_chunks/da_filler/da_b, default
                 (sparse) kernel
  B  DA dense  - identical request to C, but the server was started with
                 FOCUS_DA_DENSE=1 (dense kernel with -inf masking)

B and C differ only by the kernel, and FOCUS_DA_DENSE is process-global, so
B must be captured against a separate server instance. A and C share a
server (A simply sends no da_* fields).

Modes
-----
capture - run the given arms against one live server, save per-run timings
(incl. the first-class counters da_n_transitions / da_tokens_global /
da_tokens_focus / da_tokens_local, present in timings only when
da_n_restricted_steps > 0) + generated text + correctness to a JSON file.

  python3 da_overhead_ab.py --mode capture --server http://127.0.0.1:8090 \
      --arms A,C --repeat 3 --filler-k 8 --out /tmp/da_overhead_ac.json

compare - load the A+C capture and (optionally) the B capture, print the
per-arm table, the 2-pass overhead deltas, and the DA efficiency.

  python3 da_overhead_ab.py --mode compare --ac /tmp/da_overhead_ac.json \
      --b /tmp/da_overhead_b.json

Server (local Metal, thinking model - primed with the think prefill):
  ./build/bin/llama-server -m /Users/edwardyoon/models/Bonsai-8B-Q1_0.gguf \
      --port 8090 --host 127.0.0.1 -c 32768 --kv-unified --parallel 2 \
      --da-prompt-scan --no-webui
  FOCUS_DA_DENSE=1 <same> --port 8091 ...
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from da_ab_harness import (  # noqa: E402
    CHUNKS, FILLER, DOC_HEADER, SYSTEM_TEXT, INSTRUCTION,
    THINK_PREFILL, ANSWER,
)

# Arm A (vanilla): direct one-pass answer, NO tag instructions.
SYSTEM_A = "You are a precise reading engine."
QUESTION_A = ("Question: What is the emergency shutdown codeword for the facility? "
              "Reply with the codeword only, nothing else.\n")


def post(base, path, body, timeout=300):
    req = urllib.request.Request(base + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(base, text):
    return post(base, "/tokenize", {"content": text})["tokens"]


def detokenize(base, ids):
    return post(base, "/detokenize", {"tokens": ids})["content"]


def range_of(base, ids, start_marker, end_marker):
    """Token [lo, hi) span of start_marker..end_marker inside ids
    (same approach as da_ab_harness.range_of, explicit base)."""
    full = detokenize(base, ids)
    b0 = full.index(start_marker)
    b1 = full.index(end_marker) + len(end_marker)
    lo = len(tokenize(base, full[:b0]))
    hi = len(tokenize(base, full[:b1]))
    got = detokenize(base, ids[lo:hi])
    if not (got.startswith(start_marker) and end_marker in got):
        for dlo in (-1, 1):
            for dhi in (-1, 1):
                g = detokenize(base, ids[lo + dlo:hi + dhi])
                if g.startswith(start_marker) and end_marker in g:
                    return lo + dlo, hi + dhi
        raise SystemExit("range mismatch for %r: %r" % (start_marker, got))
    return lo, hi


def build_prompt(arm, filler_k):
    """Arm A: vanilla one-pass scaffold. Arms C/B: harness tag scaffold."""
    parts = [DOC_HEADER]
    for c in range(1, 6):
        parts.append(CHUNKS[c][0])
    parts.append(FILLER * filler_k)
    if arm == "A":
        head, tail = SYSTEM_A + "\n\n", QUESTION_A
    else:
        head, tail = SYSTEM_TEXT + "\n\n", INSTRUCTION
    return head + "".join(parts) + tail + THINK_PREFILL


def get_layout(base, prompt, arm):
    """Prompt token count + (for C/B) the da_chunks/da_filler token ranges."""
    ids = tokenize(base, prompt)
    layout = {"n_prompt": len(ids)}
    if arm in ("C", "B"):
        chunks = {cnum: list(range_of(base, ids, "[Chunk %d]" % cnum, end_marker))
                  for cnum, (_text, end_marker) in CHUNKS.items()}
        # da_filler = the filler block only (between the last chunk and the
        # instruction). The instruction is scaffold (kept), not filler.
        last_hi = max(hi for _, hi in chunks.values())
        instr_lo = len(tokenize(base, prompt[:prompt.index("Instructions:")]))
        layout["da_chunks"] = [chunks[k] for k in range(1, 6)]
        layout["da_filler"] = [last_hi, instr_lo]
    return layout


def complete(base, prompt, extra, max_tokens):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": False, "stop": ["\n"]}
    body.update(extra)
    t0 = time.monotonic()
    res = post(base, "/v1/completions", body)
    wall = (time.monotonic() - t0) * 1000.0
    text = res["choices"][0]["text"]
    tim = res.get("timings", {})
    m = re.search(r'magic_chunks="(\d+)"', text)
    return {"timings": tim, "text": text, "wall_ms": round(wall, 1),
            "correct": ANSWER in text,
            "tag_chunk": int(m.group(1)) if m else None}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def arm_stats(arm_entry):
    ts = [r["timings"] for r in arm_entry["runs"]]
    return {
        "n": len(arm_entry["runs"]),
        "n_prompt": arm_entry["n_prompt"],
        "prompt_ms": mean([t.get("prompt_ms", 0) for t in ts]),
        "decode_ms": mean([t.get("predicted_ms", 0) for t in ts]),
        "gen": mean([t.get("predicted_n", 0) for t in ts]),
        "per_tok_ms": mean([t.get("predicted_per_token_ms", 0) for t in ts]),
        "tps": mean([t.get("predicted_per_second", 0) for t in ts]),
        "total_ms": mean([t.get("prompt_ms", 0) + t.get("predicted_ms", 0) for t in ts]),
        "wall_ms": mean([r["wall_ms"] for r in arm_entry["runs"]]),
        "ok": sum(1 for r in arm_entry["runs"] if r["correct"]),
        "restricted": mean([t.get("da_n_restricted_steps", 0) for t in ts]),
        "attended": mean([t.get("da_n_attended_tokens", 0) for t in ts]),
        "transitions": mean([t.get("da_n_transitions", 0) for t in ts]),
        "tok_global": mean([t.get("da_tokens_global", 0) for t in ts]),
        "tok_focus": mean([t.get("da_tokens_focus", 0) for t in ts]),
        "tok_local": mean([t.get("da_tokens_local", 0) for t in ts]),
    }


def fmt(v, spec=".1f", none="-"):
    return none if v is None else format(v, spec)


def print_arm_table(stats):
    print("%-4s %-8s %-10s %-10s %-8s %-9s %-7s %-10s %-11s %-9s %-6s %-12s %s"
          % ("arm", "n_prompt", "prompt_ms", "decode_ms", "gen_tok",
             "tok/ms", "tps", "total_ms", "restricted", "attended",
             "trans", "g/f/l", "answer"))
    for name, s in stats:
        gfl = "%d/%d/%d" % (round(s["tok_global"]), round(s["tok_focus"]),
                            round(s["tok_local"]))
        print("%-4s %-8d %-10.0f %-10.0f %-8.1f %-9.2f %-7.2f %-10.0f "
              "%-11.0f %-9.0f %-6.0f %-12s %d/%d"
              % (name, s["n_prompt"], s["prompt_ms"], s["decode_ms"], s["gen"],
                 s["per_tok_ms"], s["tps"], s["total_ms"], s["restricted"],
                 s["attended"], s["transitions"], gfl, s["ok"], s["n"]))


def cmd_capture(args):
    base = args.server
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("A", "B", "C"):
            raise SystemExit("unknown arm %r (use A,B,C)" % a)

    prompts = {a: build_prompt(a, args.filler_k) for a in arms}
    layouts = {a: get_layout(base, prompts[a], a) for a in arms}
    print("base        : %s" % base)
    print("filler x%d, repeat %d, max_tokens %d"
          % (args.filler_k, args.repeat, args.max_tokens))
    for a in arms:
        print("arm %s prompt: %d tokens" % (a, layouts[a]["n_prompt"]))
        if a in ("C", "B"):
            print("  da_chunks: %s" % layouts[a]["da_chunks"])
            print("  da_filler: %s" % layouts[a]["da_filler"])
    print("-" * 78)

    out = {"server": base, "filler_k": args.filler_k, "repeat": args.repeat,
           "max_tokens": args.max_tokens, "arms": {}, "da_layout": {}}
    for a in arms:
        extra = {}
        if a in ("C", "B"):
            extra = {"da_chunks": layouts[a]["da_chunks"],
                     "da_filler": layouts[a]["da_filler"],
                     "da_b": True}
        runs = []
        for r in range(1, args.repeat + 1):
            run = complete(base, prompts[a], extra, args.max_tokens)
            runs.append(run)
            t = run["timings"]
            print("%-3s r%d: %s  prompt=%.0fms decode=%.0fms n_gen=%s "
                  "restricted=%s attended=%s trans=%s g/f/l=%s/%s/%s "
                  "wall=%.0fms %r"
                  % (a, r, "OK " if run["correct"] else "BAD",
                     t.get("prompt_ms", -1), t.get("predicted_ms", -1),
                     t.get("predicted_n", "?"),
                     t.get("da_n_restricted_steps", "-"),
                     t.get("da_n_attended_tokens", "-"),
                     t.get("da_n_transitions", "-"),
                     t.get("da_tokens_global", "-"),
                     t.get("da_tokens_focus", "-"),
                     t.get("da_tokens_local", "-"),
                     run["wall_ms"], run["text"][:60]))
        out["arms"][a] = {"n_prompt": layouts[a]["n_prompt"], "runs": runs}
        if a in ("C", "B"):
            out["da_layout"][a] = {"da_chunks": layouts[a]["da_chunks"],
                                   "da_filler": layouts[a]["da_filler"]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("-" * 78)
    print("captured %d arm(s) -> %s" % (len(arms), args.out))


def cmd_compare(args):
    with open(args.ac) as f:
        cap = json.load(f)
    A = arm_stats(cap["arms"]["A"])
    C = arm_stats(cap["arms"]["C"])
    B = None
    A_dense = None
    if args.b:
        with open(args.b) as f:
            capb = json.load(f)
        B = arm_stats(capb["arms"]["C"])
        if "A" in capb["arms"]:
            A_dense = arm_stats(capb["arms"]["A"])
        if capb["filler_k"] != cap["filler_k"]:
            print("NOTE: captures use different filler_k (%d vs %d)"
                  % (cap["filler_k"], capb["filler_k"]))
    print("filler x%d, repeat %d  (A/C: %s%s)"
          % (cap["filler_k"], cap["repeat"], cap["server"],
             ("  B: %s" % capb["server"]) if B else ""))
    print("-" * 78)
    stats = [("A (vanilla)", A), ("C (DA sparse)", C)]
    if B:
        stats.append(("B (DA dense)", B))
    print_arm_table(stats)

    # sanity: A must be kernel-independent (no da fields -> same path)
    if A_dense:
        print("sanity: A on dense server decode=%.1fms vs sparse decode=%.1fms "
              "(should match)" % (A_dense["decode_ms"], A["decode_ms"]))
    print("-" * 78)

    # ---- 2-pass overhead: C vs A ----
    d_gen = C["gen"] - A["gen"]
    d_decode = C["decode_ms"] - A["decode_ms"]
    d_total = C["total_ms"] - A["total_ms"]
    print("2-pass overhead (C vs A)")
    print("  Δgen    = %.1f tokens  (C %.1f - A %.1f)" % (d_gen, C["gen"], A["gen"]))
    print("  Δdecode = %+.1f ms      (C %.0f - A %.0f)" % (d_decode, C["decode_ms"], A["decode_ms"]))
    print("  Δtotal  = %+.1f ms      (C %.0f - A %.0f)" % (d_total, C["total_ms"], A["total_ms"]))
    print("  C per-mode split: global %.1f (search) / focus %.1f (re-reason) / "
          "local %.1f  (of %.1f gen tokens)"
          % (C["tok_global"], C["tok_focus"], C["tok_local"], C["gen"]))
    print("  C transitions = %.1f  (expect 2: GLOBAL->FOCUS, FOCUS->GLOBAL)"
          % C["transitions"])

    # ---- DA efficiency ----
    print("-" * 78)
    if C["restricted"] == 0:
        print("DA arm never restricted (no tag fired?) - check the journal "
              "da_tag: lines; efficiency n/a")
        return
    # per-run saved KV work, then mean (token units: KV token reads)
    saved_runs = []
    for r in cap["arms"]["C"]["runs"]:
        t = r["timings"]
        if t.get("da_n_restricted_steps", 0):
            saved_runs.append(t["da_n_restricted_steps"] * C["n_prompt"]
                              - t["da_n_attended_tokens"])
    saved = mean(saved_runs) if saved_runs else None
    print("DA efficiency (C)")
    print("  saved_KV_work = restricted*n_prompt - attended = %.0f KV token reads"
          "  (restricted %.0f steps x %d prompt tok - attended %.0f)"
          % (saved if saved is not None else -1, C["restricted"], C["n_prompt"],
             C["attended"]))
    print("  additional_generation_work = Δgen = %.1f tokens" % d_gen)
    if saved is not None and d_gen > 0:
        ratio_tok = saved / d_gen
        print("  efficiency (token) = %.2f  (KV token reads saved per extra "
              "generated token)" % ratio_tok)
        # time-normalized: assume decode step time ~ proportional to attended
        # KV (KV-read-bound decode, Phase 1b finding)
        step_ms = C["decode_ms"] / C["gen"] if C["gen"] else 0.0
        att_per_step = C["attended"] / C["restricted"] if C["restricted"] else 0.0
        if att_per_step > 0:
            kv_tok_ms = step_ms / att_per_step
            saved_kv_ms = saved * kv_tok_ms
            extra_gen_ms = d_gen * C["per_tok_ms"]
            print("  efficiency (time)  = %.2f  (saved %.0f ms KV-read work / "
                  "extra %.0f ms generation; assumes step time ~ attended KV)"
                  % (saved_kv_ms / extra_gen_ms if extra_gen_ms else -1,
                     saved_kv_ms, extra_gen_ms))
    elif saved is not None:
        print("  Δgen <= 0: the DA request did NOT generate more tokens than "
              "vanilla - ratio n/a (overhead hypothesis not supported by "
              "generation count)")

    # ---- kernel delta: B vs C ----
    if B:
        print("-" * 78)
        print("kernel delta (B dense vs C sparse, identical request)")
        print("  decode tps: B %.2f vs C %.2f  (local Metal - the CUDA kernel "
              "delta is 123's job)" % (B["tps"], C["tps"]))
        print("  decode ms : B %.0f vs C %.0f" % (B["decode_ms"], C["decode_ms"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["capture", "compare"], required=True)
    ap.add_argument("--server", default=None, help="capture: server base URL")
    ap.add_argument("--arms", default="A,C", help="capture: comma list of A,B,C")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--filler-k", type=int, default=8,
                    help="filler repetition count (8 ~ 2.7K filler tokens)")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", default=None, help="capture: output JSON path")
    ap.add_argument("--ac", default=None, help="compare: A+C capture JSON")
    ap.add_argument("--b", default=None, help="compare: B (dense) capture JSON")
    args = ap.parse_args()

    if args.mode == "capture":
        if not args.server or not args.out:
            ap.error("--server and --out are required in capture mode")
        cmd_capture(args)
    else:
        if not args.ac:
            ap.error("--ac is required in compare mode")
        cmd_compare(args)


if __name__ == "__main__":
    main()

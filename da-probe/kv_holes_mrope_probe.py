#!/usr/bin/env python3
"""R1 probe — MROPE main-sequence mid-hole (seq_rm) physical-mechanism check.

This is the gate for Option B (seq_rm-based low-cost kv offload, see
plans/focus-offload-option-b.md). Option B's physical mechanism is identical to
the DA A path: a `seq_rm` of a mid-sequence range leaves a HOLE in the MAIN
sequence (cells pos=-1), which the generic KQ mask turns into -INFINITY
(llama-kv-cache.cpp:1631-1638). R1 asks: on a MROPE model (n_pos_per_embd()==4,
the production 27B), does that mid-hole behave like -inf masking — i.e. does the
model still read the KEPT chunk correctly and put high logprob on the answer —
rather than leaking stale K/V or breaking the 4-dim MROPE position handling?

Two arms, same server, same 5-chunk+filler document (da_ab_harness scaffold):
  baseline - no da_* fields, full attention (reference).
  hole     - DA A path: da_chunks (all 5) + da_filler, NO da_b. The model emits
             <focus magic_chunks="4">; at tag close the server seq_rm's every
             chunk except 4 plus the filler -> a MID-SEQUENCE HOLE on the main
             sequence. Generation then continues in FOCUS mode over the hole.

Checks (R1 passes iff all hold):
  1. both arms answer ZEBRA-42 (the kept chunk survives the hole).
  2. the removal actually ran mid-decode: the server's __verbose.timings shows
     da_n_restricted_steps > 0 and da_path == "A" (a main-seq mid-hole was
     created). NOTE: the <focus> tag is erased from the client-visible text by
     apply_da_tag, so it must NOT be looked for in the output - that is the
     false-negative this probe used to have (R1 09-25: tag=None yet the journal
     proved the hole was created and the answer read through it).
  3. the hole arm's answer-token logprob is within --lp-tol nats of the
     baseline's (the hole behaves like -inf masking, not garbage / not a
     position-handling break). A large negative delta = the hole corrupted the
     kept chunk's attention.

Why this validates Option B: Option B differs from the A path only in the
release() cache policy (keep vs clear) — the seq_rm mid-hole + generic -inf mask
is the SAME physical state. If the A path is logprob-consistent on MROPE, the
hole mechanism is safe; the remaining Option B risks (R2 cache persistence, R3
sparse n_kv_max) are tested separately.

NOTE: the A path (and hence this probe) only makes sense on a model where the
mid-hole is legal. Non-MROPE batch validation (llama-batch.cpp:281-320) REQUIRES
contiguous positions and would ERROR on a mid-hole, so this probe is MROPE-only
(the local qwen3 test model cannot run it — use the 123 27B MROPE node).

Usage:
  python3 kv_holes_mrope_probe.py [BASE_URL] [--runs N] [--filler-k M] [--lp-tol T]

Journal evidence (grep the server log):
  da_tag: <focus magic_chunks="4"> closed at generated token K - A removal, ...
  da_rm: applying N range(s) (at tag close) - bound=P, ...
  da: A path - ... restricted step(s), logical attended ... (-XX%)
"""
import json
import os
import re
import sys
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from da_ab_harness import (  # noqa: E402
    CHUNKS, FILLER, DOC_HEADER, SYSTEM_TEXT, INSTRUCTION,
    THINK_PREFILL, ANSWER,
)

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")

# Arm "baseline": direct one-pass answer, NO tag instructions (full attention).
SYSTEM_A = "You are a precise reading engine."
QUESTION_A = ("Question: What is the emergency shutdown codeword for the facility? "
              "Reply with the codeword only, nothing else.\n")

TAG_RE = re.compile(r'<focus[^>]*magic_chunks=["\']?(\d+)')


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
    parts = [DOC_HEADER]
    for c in range(1, 6):
        parts.append(CHUNKS[c][0])
    parts.append(FILLER * filler_k)
    if arm == "baseline":
        head, tail = SYSTEM_A + "\n\n", QUESTION_A
    else:
        head, tail = SYSTEM_TEXT + "\n\n", INSTRUCTION
    return head + "".join(parts) + tail + THINK_PREFILL


def complete(base, prompt, extra, max_tokens=48):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": False, "stop": ["\n"], "n_probs": 5, "verbose": True}
    body.update(extra)
    res = post(base, "/v1/completions", body)
    text = res["choices"][0]["text"]
    lps = []
    toks = []
    for entry in res["choices"][0]["logprobs"]["content"]:
        lps.append(entry["logprob"])
        toks.append(entry["token"])
    m = TAG_RE.search(text)
    # The server ERASES the <focus> tag from the client-visible text
    # (apply_da_tag -> generated_text.erase), so the tag is never in `text`
    # even when it was emitted. The reliable signal that the removal actually
    # ran (a mid-seq hole was created) is the server's own __verbose.timings:
    # da_n_restricted_steps > 0 and da_path ("A" = main-seq mid-hole).
    t = res.get("__verbose", {}).get("timings", {})
    return {"text": text, "lps": lps, "toks": toks,
            "tag_chunk": int(m.group(1)) if m else None,
            "restricted": t.get("da_n_restricted_steps", 0),
            "attended": t.get("da_n_attended_tokens", 0),
            "da_path": t.get("da_path")}


def answer_lp(run):
    """logprob of the first generated token that is part of the answer text.

    The answer 'ZEBRA-42' may span several tokens; we return the logprob at the
    position where the running detokenization first contains 'ZEBRA'. This is the
    token the model committed to the answer - the one that must stay high when the
    kept chunk is read through the hole."""
    acc = ""
    for i, t in enumerate(run["toks"]):
        acc += (t or "")
        if "ZEBRA" in acc and i < len(run["lps"]):
            return i, run["lps"][i]
    return None, None


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    global BASE
    args = sys.argv[1:]
    runs = 1
    filler_k = 8
    lp_tol = 0.5
    if "--runs" in args:
        i = args.index("--runs"); runs = int(args[i + 1]); del args[i:i + 2]
    if "--filler-k" in args:
        i = args.index("--filler-k"); filler_k = int(args[i + 1]); del args[i:i + 2]
    if "--lp-tol" in args:
        i = args.index("--lp-tol"); lp_tol = float(args[i + 1]); del args[i:i + 2]
    if args:
        BASE = args[0]

    prompts = {a: build_prompt(a, filler_k) for a in ("baseline", "hole")}
    # layout for the hole arm (da_chunks/da_filler token ranges)
    ids = tokenize(BASE, prompts["hole"])
    chunks = {c: list(range_of(BASE, ids, "[Chunk %d]" % c, em))
              for c, (_t, em) in CHUNKS.items()}
    last_hi = max(hi for _, hi in chunks.values())
    instr_lo = len(tokenize(BASE, prompts["hole"][:prompts["hole"].index("Instructions:")]))
    da_chunks = [chunks[k] for k in range(1, 6)]
    print("base        : %s" % BASE)
    print("filler x%d, runs %d, lp_tol %.2f nats" % (filler_k, runs, lp_tol))
    print("baseline prompt: %d tokens" % len(tokenize(BASE, prompts["baseline"])))
    print("hole prompt    : %d tokens  da_filler=[%d,%d)"
          % (len(ids), last_hi, instr_lo))
    print("-" * 78)

    base_runs = [complete(BASE, prompts["baseline"], {}) for _ in range(runs)]
    hole_extra = {"da_chunks": da_chunks, "da_filler": [last_hi, instr_lo]}  # NO da_b -> A path
    hole_runs = [complete(BASE, prompts["hole"], hole_extra) for _ in range(runs)]

    checks = []

    def check(name, ok, detail):
        checks.append(bool(ok))
        print("%-26s: %s (%s)" % (name, "PASS" if ok else "FAIL", detail))

    for r in range(runs):
        b, h = base_runs[r], hole_runs[r]
        print("run %d baseline: %r" % (r + 1, b["text"][:70]))
        print("run %d hole    : restricted=%s attended=%s path=%s %r"
              % (r + 1, h["restricted"], h["attended"], h["da_path"], h["text"][:70]))
        b_ok = ANSWER in b["text"]
        h_ok = ANSWER in h["text"]
        check("baseline answer", b_ok, "contains %s" % ANSWER)
        check("hole answer (kept chunk readable through mid-hole)", h_ok,
              "contains %s" % ANSWER)
        # the removal must have actually run mid-decode. The server erases the
        # <focus> tag from the client-visible text, so the tag is never in
        # h["text"] even when emitted - the reliable signal is the server's own
        # __verbose.timings: restricted steps > 0 on the A path means a
        # main-sequence mid-hole was created (exactly the Option B mechanism).
        removed_ok = (h["restricted"] or 0) > 0 and h["da_path"] == "A"
        check("hole removal ran mid-decode (server timings)", removed_ok,
              "restricted=%s attended=%s path=%s (tag erased from output)"
              % (h["restricted"], h["attended"], h["da_path"]))
        # answer-token logprob: hole must be within tol of baseline (=-inf masking, not garbage)
        bi, blp = answer_lp(b)
        hi_, hlp = answer_lp(h)
        if blp is not None and hlp is not None:
            delta = hlp - blp
            check("answer-token lp within %.2f nats" % lp_tol, delta >= -lp_tol,
                  "baseline lp=%.3f (tok %d)  hole lp=%.3f (tok %d)  delta=%+.3f"
                  % (blp, bi, hlp, hi_, delta))
        else:
            check("answer-token lp present", False,
                  "baseline=%s hole=%s (answer token not found in logprobs)"
                  % (blp, hlp))
        print("  baseline lps[:8] = %s" % [round(x, 3) for x in b["lps"][:8]])
        print("  hole     lps[:8] = %s" % [round(x, 3) for x in h["lps"][:8]])
        print("-" * 78)

    ok = all(checks) if checks else False
    print("R1 VERDICT       : %s" % ("PASS - MROPE mid-hole behaves like -inf masking"
                                      if ok else "FAIL - see the checks above"))
    if not ok:
        print("  -> if the hole answer is WRONG or the answer-token lp dropped a lot, the")
        print("     MROPE main-sequence mid-hole is NOT safe: Option B is rejected (R1).")
    print("journal checks (must match the hole arm):")
    print("  grep 'da_tag: <focus magic_chunks' -> 'A removal' (NOT 'B switch')")
    print("  grep 'da_rm: applying .* (at tag close)' -> N range(s)")
    print("  grep 'da: A path -' -> restricted step(s) + attended reduction %")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

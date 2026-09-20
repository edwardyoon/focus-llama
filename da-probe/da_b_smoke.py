#!/usr/bin/env python3
"""Smoke test for llama-server da_b (Declarative Attention 2-stream B) — server-side verification.

Usage:
  python3 da_b_smoke.py [BASE_URL] [--hybrid] [--runs k1,k2,...] [--skip-cache]

  BASE_URL     default: $DA_BASE or http://127.0.0.1:8080
  --hybrid     model has recurrent/linear-attention layers (e.g. qwen35):
               the fact-removed run's TEXT is informational (the fact can
               survive via the recurrent state); the logprob checks are
               architecture-independent.
  --runs       run only the given phase-1 keys: baseline,bkeep,bnofact,logical,bkeep2
  --skip-cache skip phase 2 (cache integrity; needs --parallel 2)

The server MUST be started with --kv-unified --parallel 2 (a partial-range
llama_memory_seq_cp aborts on a non-unified KV pool, and the second stream
needs the reserved extra sequence id). Without those flags the server falls
back to the logical removal (apply_da_rm) with a WARN in the journal — the
answers below would still pass, so the journal is the real evidence:
  grep 'da_b:' <journal>           # expect 'switched decode to seq', 0x 'falling back'
  grep 'falling back to logical removal' <journal>   # must be EMPTY

Phase 1 — computation checks (all on the ZEBRA prompt, stop=["\\n"],
max_tokens=24, n_probs=5, cache_prompt=false):
  baseline - no da_rm                                        -> exactly ZEBRA-42
  bkeep    - da_rm = everything EXCEPT the fact range,
             da_rm_at=Q, da_b=true                           -> exactly ZEBRA-42.
             Stream 1 keeps scaffold + fact + boundary token,
             so the answer must come from the COPIED ranges.
  bnofact  - da_rm = the fact range, da_rm_at=Q, da_b=true   -> clean (pure attn) /
             informational (hybrid).
  logical  - da_rm = the fact range, da_rm_at=Q (A path)     -> clean / informational.
  bkeep2   - same params as bkeep, sent LAST                 -> exactly ZEBRA-42.
             Consecutive da_b: the previous B slot is idle with its da_seq KV
             cleared; the reserved id must be reusable (no spurious fallback).

Core checks:
  1. mechanism   : |lp(baseline) - lp(bkeep)| > 1e-6 — bkeep's question was
                   prefilled against the copied KV, so the first-token logprob
                   must differ from the baseline's. Identical logprobs mean the
                   request was served from the untouched full prompt (da_b did
                   not run: old binary, missing flags, or fallback).
  2. B vs A equal: |lp(bnofact) - lp(logical)| < 5e-3 — same removal applied
                   by two different means (seq_cp copy vs seq_rm). The
                   attended token sets are identical, so the first-token
                   distributions must be near-equal (observed 0.0003 on local
                   Bonsai-8B). A large delta means B's keep-set computation is
                   wrong.
  3. answers     : baseline / bkeep / bkeep2 exactly ZEBRA-42; bnofact /
                   logical clean (pure attn) or informational (hybrid).

Phase 2 — cache integrity (--parallel 2, LRU slot selection is deterministic
for these prompts): C1 (P1, cache_prompt=true) lands on slot 0 and keeps its
prompt cache; C2 (ZEBRA + da_b) lands on slot 1 and switches to the reserved
da_seq; C3 (P1 again, cache_prompt=true) must hit slot 0's cache UNTOUCHED by
C2's B switch (journal: 'cache reuse: n_past = <full>' on slot 0). The old
bug let a B slot take an idle slot's id as da_seq, whose prompt_clear() then
destroyed the cached prompt.

After the run, the journal chain for a B request is:
  request start (da_b=1) -> batch fill stopped at boundary
  -> da_b: switched decode to seq 2 (at da_rm_at) - logical read set now N of M prefix token(s)
  -> da_b:   keep [lo, hi) n token(s)             (one line per kept range)
  -> da_b: original seq 0 untouched (M token(s)) - rollback/reference
  -> da_b: decode #p: logical read set N of M token(s) on seq 2 (...)  (per step, with -v)
  -> da_b: finished on seq 2 - final logical read set N of M token(s) (...)
"""
import json
import os
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")
HYBRID = False
SKIP_CACHE = False

# B vs A first-token logprob equality threshold. The two paths attend to the
# same token set, so the distributions are near-identical (observed 0.0003 on
# local Bonsai-8B); 5e-3 leaves headroom for backend nondeterminism while
# still being ~50x below the mechanism delta (~0.24).
BA_EQUAL_EPS = 5e-3

PROMPT = """The following is a technical document.

[Section 1: Overview]
The system processes incoming requests through a queue. Each request is validated, then dispatched to a worker pool. Workers maintain a local cache of recent results to reduce latency. The cache expires entries after thirty seconds.

[Section 2: The codeword]
The emergency shutdown codeword for the facility is ZEBRA-42. Operators must memorize it. It is printed on the wall of the control room in red letters.

[Section 3: Scheduling]
Tasks are scheduled using a round-robin policy. The scheduler considers task priority and estimated duration. Long-running tasks are preempted after five minutes.

Question: What is the emergency shutdown codeword for the facility? Reply with the codeword only, nothing else.
Answer:
"""

DOC_START = "[Section 1: Overview]"
FACT_START = "The emergency shutdown codeword"
FACT_END = "red letters."
QUESTION_START = "Question: What is the"
THINK_END = "\n</think>\n"

# Phase 2 (cache integrity): a short prompt unrelated to the ZEBRA prompt, so
# LRU slot selection is deterministic: C1 -> slot 0 (first free), C2 -> slot 1
# (least recently used), C3 -> slot 0 (cache hit, least recently used).
P1 = ("The capital of France is Paris. It is famous for the Eiffel Tower. "
      "What is the capital of France? Reply with the city name only.\n")
P1_ANSWER = "Paris"


def post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(text):
    return post("/tokenize", {"content": text})["tokens"]


def detokenize(ids):
    return post("/detokenize", {"tokens": ids})["content"]


def complete(prompt, extra, max_tokens=24, cache_prompt=False):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": cache_prompt, "stop": ["\n"], "n_probs": 5}
    body.update(extra)
    return post("/v1/completions", body)


def first_token_lp(res):
    """(token, logprob, top-3) of the first generated token, or (None, None, [])."""
    try:
        entry = res["choices"][0]["logprobs"]["content"][0]
        top = [(t["token"], round(t["logprob"], 4)) for t in entry["top_logprobs"][:3]]
        return entry["token"], round(entry["logprob"], 4), top
    except (KeyError, TypeError, IndexError):
        return None, None, []


def strip_thinking(text):
    i = text.rfind(THINK_END)
    return text[i + len(THINK_END):] if i != -1 else text


def range_of(ids, start_marker, end_marker):
    """Token range [lo, hi) whose detokenization covers start_marker..end_marker."""
    full = detokenize(ids)
    b0 = full.index(start_marker)
    b1 = full.index(end_marker) + len(end_marker)
    lo = len(tokenize(full[:b0]))
    hi = len(tokenize(full[:b1]))
    got = detokenize(ids[lo:hi])
    if not (got.startswith(start_marker) and end_marker in got):
        # tolerate +/-1 token boundary drift
        for dlo in (-1, 1):
            for dhi in (-1, 1):
                g = detokenize(ids[lo + dlo:hi + dhi])
                if g.startswith(start_marker) and end_marker in g:
                    return lo + dlo, hi + dhi
        raise SystemExit("range mismatch: %r" % got)
    return lo, hi


def main():
    global BASE, HYBRID, SKIP_CACHE
    args = sys.argv[1:]
    if "--hybrid" in args:
        HYBRID = True
        args.remove("--hybrid")
    if "--skip-cache" in args:
        SKIP_CACHE = True
        args.remove("--skip-cache")
    runs_filter = None
    if "--runs" in args:
        i = args.index("--runs")
        runs_filter = set(a for a in args[i + 1].split(",") if a)
        del args[i:i + 2]
    if args:
        BASE = args[0]
    print("base   : %s  (%s)" % (BASE,
                                 "hybrid expectations" if HYBRID
                                 else "pure-attention expectations"))

    ids = tokenize(PROMPT)
    n = len(ids)
    full = detokenize(ids)
    doc_lo = len(tokenize(full[:full.index(DOC_START)]))
    fact_lo, fact_hi = range_of(ids, FACT_START, FACT_END)
    q_pos = len(tokenize(full[:full.index(QUESTION_START)]))
    print("prompt tokens            : %d" % n)
    print("document start           : %d" % doc_lo)
    print("fact range (S2)          : [%d, %d)" % (fact_lo, fact_hi))
    print("question start (da_rm_at): %d" % q_pos)
    # what stream 1 will hold in the bkeep run: scaffold + fact + boundary token
    keep_n = doc_lo + (fact_hi - fact_lo) + 1
    print("bkeep stream-1 read set  : %d of %d token(s) at the boundary (-%.1f%%)"
          % (keep_n, q_pos, 100.0 * (q_pos - keep_n) / q_pos))
    print("-" * 72)

    # (key, label, extra) — --runs filters on key. bkeep2 repeats bkeep LAST:
    # the previous B slot is idle (da_seq KV cleared in release), so the
    # reserved id must be reusable without a spurious fallback.
    runs = [
        ("baseline", "baseline",
         {}),
        ("bkeep",    "bkeep    (keep fact, B)",
         {"da_rm": [[doc_lo, fact_lo], [fact_hi, q_pos]], "da_rm_at": q_pos, "da_b": True}),
        ("bnofact",  "bnofact  (fact rm, B)   ",
         {"da_rm": [[fact_lo, fact_hi]], "da_rm_at": q_pos, "da_b": True}),
        ("logical",  "logical  (fact rm, A)   ",
         {"da_rm": [[fact_lo, fact_hi]], "da_rm_at": q_pos}),
        ("bkeep2",   "bkeep2   (keep fact, B) ",
         {"da_rm": [[doc_lo, fact_lo], [fact_hi, q_pos]], "da_rm_at": q_pos, "da_b": True}),
    ]
    if runs_filter is not None:
        runs = [r for r in runs if r[0] in runs_filter]
    results = {}
    lps = {}
    machine_ok = True
    for key, name, extra in runs:
        try:
            res = complete(PROMPT, extra)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            results[name] = None
            machine_ok = False
            print("%s: HTTP ERROR %s  (mechanism FAIL)" % (name, e))
            continue
        text = strip_thinking(res["choices"][0]["text"]).strip()
        results[name] = text
        tok, lp, top = first_token_lp(res)
        lps[key] = (tok, lp, top)
        has = "ZEBRA-42" in text
        print("%s: %r  (ZEBRA-42: %s)" % (name, text, has))
        print("          prompt_tokens=%s completion_tokens=%s"
              % (res["usage"]["prompt_tokens"], res["usage"]["completion_tokens"]))
        if tok is not None:
            print("          first_token=%r lp=%s top=%s" % (tok, lp, top))
    print("-" * 72)

    ANSWER = "ZEBRA-42"
    L_BASE, L_BKEEP, L_BNOFACT, L_LOGICAL, L_BKEEP2 = (
        "baseline", "bkeep    (keep fact, B)", "bnofact  (fact rm, B)   ",
        "logical  (fact rm, A)   ", "bkeep2   (keep fact, B) ")

    def exact(name):
        return results.get(name) == ANSWER

    def leak_state(name):
        """Classify a fact-removed run: clean / PARTIAL-LEAK / FULL-LEAK / EMPTY."""
        t = results.get(name)
        if t is None:
            return "ERROR"
        if t == "":
            return "EMPTY"
        if t == ANSWER:
            return "FULL-LEAK"
        if t.startswith("Z"):
            return "PARTIAL-LEAK"
        return "clean"

    def delta(key_a, key_b):
        """First-token logprob distance between two runs, or None."""
        a, b = lps.get(key_a), lps.get(key_b)
        if not a or not b or a[1] is None or b[1] is None:
            return None
        return round(abs(a[1] - b[1]), 6)

    checks = []
    if L_BASE in results:
        ok = exact(L_BASE)
        checks.append(ok)
        print("baseline         : %s (exact %r)" % ("PASS" if ok else "FAIL", ANSWER))

    # bkeep: the answer must survive on the copied ranges (mechanism + keep
    # ranges are attended on the new sequence)
    if L_BKEEP in results:
        ok = exact(L_BKEEP)
        checks.append(ok)
        print("bkeep answer     : %s (exact %r — the fact must be readable from stream 1)"
              % ("PASS" if ok else "FAIL", ANSWER))

    # bkeep2: consecutive da_b — the reserved da_seq id must be reusable after
    # the previous B slot released (journal: a second 'da_b: switched' for the
    # same prompt, no 'falling back').
    if L_BKEEP2 in results:
        ok = exact(L_BKEEP2)
        checks.append(ok)
        print("bkeep2 answer    : %s (consecutive da_b — journal must show a 2nd "
              "'da_b: switched' and no fallback)" % ("PASS" if ok else "FAIL"))

    # mechanism proof: bkeep's question was prefilled against the copied KV,
    # so its first-token logprob must differ from the baseline's. Identical
    # logprobs mean the request was served from the untouched full prompt —
    # the B switch did not run (old binary ignoring da_b, or the A fallback).
    d_bkeep = delta("baseline", "bkeep")
    if "baseline" in lps and "bkeep" in lps:
        if d_bkeep is None:
            checks.append(False)
            print("mechanism (Δlp)  : FAIL (no logprobs: baseline=%s bkeep=%s) - check the "
                  "first_token=... lines above" % (lps["baseline"][1] is not None,
                                                   lps["bkeep"][1] is not None))
        else:
            ok = d_bkeep > 1e-6
            checks.append(ok)
            print("mechanism (Δlp)  : baseline vs bkeep first-token Δlp=%s — %s"
                  % (d_bkeep, "different prefill computation (the B switch ran; the "
                               "journal must show 'da_b: switched decode to seq')"
                     if ok else
                     "IDENTICAL - served from the untouched full prompt (da_b did not "
                     "run: old binary? missing --kv-unified/--parallel? journal: look "
                     "for 'falling back to logical removal')"))

    # CORE: B vs A equality — bnofact (B) and logical (A) apply the same
    # removal by different means; the attended token sets are identical, so
    # the first-token distributions must be near-equal.
    d_ba = delta("bnofact", "logical")
    if "bnofact" in lps and "logical" in lps:
        if d_ba is None:
            checks.append(False)
            print("B vs A equal     : FAIL (no logprobs for bnofact/logical)")
        else:
            ok = d_ba < BA_EQUAL_EPS
            checks.append(ok)
            print("B vs A equal     : %s (bnofact vs logical first-token Δlp=%s, eps=%g — "
                  "same removal, two paths; a large delta means B's keep-set is wrong)"
                  % ("PASS" if ok else "FAIL", d_ba, BA_EQUAL_EPS))

    # bnofact: the fact is absent from stream 1 — removal effective on decode
    if L_BNOFACT in results:
        st = leak_state(L_BNOFACT)
        if HYBRID:
            print("bnofact text     : %s (informational on hybrid — the fact can survive "
                  "via the recurrent state, which was updated while the fact was "
                  "visible)" % st)
        else:
            ok = st == "clean"
            checks.append(ok)
            print("bnofact text     : %s (must be clean — the fact is not in stream 1; %s)"
                  % ("PASS" if ok else "FAIL", st))

    # logical (A path) control: same removal, logical seq_rm
    if L_LOGICAL in results:
        st = leak_state(L_LOGICAL)
        if HYBRID:
            print("logical text     : %s (informational on hybrid — paper semantics, "
                  "surviving-KV leak)" % st)
        else:
            ok = st == "clean"
            checks.append(ok)
            print("logical text     : %s (%s)" % ("PASS" if ok else "FAIL", st))

    # ---------------- phase 2: cache integrity (--parallel 2) ----------------
    if not SKIP_CACHE:
        print("-" * 72)
        print("phase 2: cache integrity (C1 slot0 cache -> C2 slot1 da_b -> C3 slot0 hit)")
        c1 = c2 = c3 = None
        try:
            c1 = complete(P1, {}, cache_prompt=True)
            c2 = complete(PROMPT,
                          {"da_rm": [[fact_lo, fact_hi]], "da_rm_at": q_pos, "da_b": True})
            c3 = complete(P1, {}, cache_prompt=True)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            machine_ok = False
            print("cache phase      : HTTP ERROR %s" % e)
        if c1 is not None and c2 is not None and c3 is not None:
            t1 = strip_thinking(c1["choices"][0]["text"]).strip()
            t2 = strip_thinking(c2["choices"][0]["text"]).strip()
            t3 = strip_thinking(c3["choices"][0]["text"]).strip()
            print("C1 (P1, cache)   : %r" % t1)
            print("C2 (ZEBRA da_b)  : %r  (runs on slot 1, switches to da_seq)" % t2)
            print("C3 (P1, cache)   : %r  (must equal C1 — slot 0 cache intact)" % t3)
            print("          C1 prompt_tokens=%s  C3 prompt_tokens=%s"
                  % (c1["usage"]["prompt_tokens"], c3["usage"]["prompt_tokens"]))
            ok1 = (t1 == P1_ANSWER)
            ok3 = (t3 == t1)
            checks.append(ok1)
            checks.append(ok3)
            print("C1 answer        : %s (exact %r)" % ("PASS" if ok1 else "FAIL", P1_ANSWER))
            print("C3 == C1         : %s (journal: 'cache reuse: n_past = %d' on slot 0 "
                  "means the full cache survived C2's B switch)"
                  % ("PASS" if ok3 else "FAIL", c1["usage"]["prompt_tokens"]))
        else:
            checks.append(False)
            print("cache phase      : FAIL (incomplete responses)")

    ok = machine_ok and all(checks) if checks else machine_ok
    print("-" * 72)
    print("OVERALL          : %s" % ("PASS" if ok else "FAIL"))
    print("journal checks (must match):")
    print("  grep 'da_b: switched decode to seq'            -> 1 line per B run (bkeep, bnofact, bkeep2, C2)")
    print("  grep 'falling back to logical removal'         -> EMPTY")
    print("  grep 'cache reuse: n_past'                     -> C3's full prompt length on slot 0")
    print("  per-step 'logical read set' lines with -v; final 'finished on seq' summary always.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

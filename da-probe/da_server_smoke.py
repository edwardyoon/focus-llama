#!/usr/bin/env python3
"""Smoke test for llama-server da_rm (Declarative Attention) — server-side verification.

Usage:
  python3 da_server_smoke.py [BASE_URL] [--hybrid]

  BASE_URL    default: $DA_BASE or http://127.0.0.1:8080
  --hybrid    model has recurrent/linear-attention layers (e.g. qwen35):
              the strict (fact-only) run is expected to LEAK (paper semantics —
              the fact survives via the surviving full-attention KV), so it is
              informational. The mechanism proof is the first-token Δlp vs
              baseline; the masked-all TEXT result is informational on a
              hybrid (a 'Z' start there is a recurrent-state leak candidate).

Runs 5 completions against BASE/v1/completions. All requests use
stop=["\n"], max_tokens=24 and n_probs=5, and the verdict is EXACT match, so a
run that re-recites the document (or starts thinking) cannot pass by merely
containing ZEBRA-42 somewhere in the text:
  a. baseline   - no da_rm                          -> expect exactly ZEBRA-42
  b. strict     - da_rm=fact range, da_rm_at=Q      -> clean (pure attn) / informational (hybrid)
  c. paper      - da_rm=fact range, no boundary     -> decode-time restriction (informational)
  d. masked-all - da_rm=whole document, da_rm_at=Q  -> mechanism proof is the
              first-token Δlp vs baseline; on pure attention also expect clean
  e. control    - da_rm=other section, da_rm_at=Q   -> expect exactly ZEBRA-42 (selectivity)

Leak classification for removal runs (b/d):
  clean         - not the full answer and does not start with 'Z'
  PARTIAL-LEAK  - starts with 'Z' but is not the full answer: the first
                  token was sampled from pre-removal prefill logits (the
                  post-prefill path ran) or from surviving state. On a
                  hybrid, once the mid-prefill path is confirmed working
                  (mechanism Δlp PASS), a 'Z' start can only be
                  surviving/recurrent-state leak - whether it really is the
                  recurrent state is a separate observation, confirmed by
                  the codeword-swap control (da_probe QUOKKA-17).
  FULL-LEAK     - exactly ZEBRA-42

Why the first-token Δlp is the mechanism proof: the first generated token is
sampled from the logits at the last prompt position. masked-all removes the
WHOLE document before the question is prefilled, so if the mid-prefill path
ran its prefill computation differs from the baseline's and the first-token
logprob differs (|Δlp| > 1e-6); identical logprobs mean the removal only ran
after prefill. This works on both architectures and is immune to the
hybrid's surviving-KV leak, which makes the strict run behaviorally
identical to an old server binary that silently ignores the da_rm field.
On a pure-attention model the masked-all run must also answer cleanly
(end-to-end check, still a PASS/FAIL criterion); on a hybrid the text check
is informational — the recurrent state was updated while the document was
still visible, so a 'Z' start despite a Δlp divergence is a recurrent-state
leak (a real finding, confirmed by the codeword-swap control QUOKKA-17).
The masked-all range reaches the da_rm_at boundary, so apply_da_rm clamps its
end to bound-1 and emits a WARN line in the server log (visible at default
verbosity) — a second piece of evidence.

No server restart is required — provided the running binary was built from a
commit that includes the da_rm field. Check first (macOS: the code lives in
libllama-server-impl.dylib, not the thin server binary):
  strings build/bin/libllama-server-impl.dylib | grep -m1 da_rm
(run_probe.sh only rebuilds da_probe, not llama-server).

After the run, grep the server journal for the da_rm diagnostic chain:
  request start -> batch fill stopped at boundary -> applying (at da_rm_at)
  or: request start -> boundary reached but skipped (media present) ->
      mid-prefill boundary never applied -> applying (after prefill)
The mid-prefill guards test a media-presence check (da_rm_text_only: text
tokens == all tokens), not has_mtmd — has_mtmd is true whenever an mmproj is
loaded, even for text-only prompts, which silently pushed every text-only
da_rm request to the post-prefill path.

Note: chat_template_kwargs (e.g. enable_thinking:false) only applies to
/v1/chat/completions, not to /v1/completions. Thinking is neutralized
here by stop+exact match: a thinking-first response fails the baseline
visibly instead of passing via substring. A trailing thinking-end tag is
still tolerated (only text after the last one is checked).
"""
import json
import os
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")
HYBRID = False

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
OTHER_START = "Tasks are scheduled using"
OTHER_END = "after five minutes."
QUESTION_START = "Question: What is the"
THINK_END = "\n</think>\n"


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


def complete(prompt, extra, max_tokens=24):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": False, "stop": ["\n"], "n_probs": 5}
    body.update(extra)
    return post("/v1/completions", body)


def first_token_lp(res):
    """(token, logprob, top-3) of the first generated token, or (None, None, []).

    The first generated token is sampled from the logits at the last prompt
    position, so its logprob is a fingerprint of what the prefill actually
    computed: two runs that took different prefill paths (mid- vs post-prefill
    removal) produce different values even if the text looks similar.
    """
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
    global BASE, HYBRID
    args = sys.argv[1:]
    if "--hybrid" in args:
        HYBRID = True
        args.remove("--hybrid")
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
    oth_lo, oth_hi = range_of(ids, OTHER_START, OTHER_END)
    q_pos = len(tokenize(full[:full.index(QUESTION_START)]))
    print("prompt tokens            : %d" % n)
    print("document start           : %d" % doc_lo)
    print("fact range (S2)          : [%d, %d)" % (fact_lo, fact_hi))
    print("other range (S3)         : [%d, %d)" % (oth_lo, oth_hi))
    print("question start (da_rm_at): %d" % q_pos)
    print("-" * 72)

    # (key, label, extra) — --runs filters on key
    runs = [
        ("baseline",   "baseline",                 {}),
        ("strict",     "strict    (fact, at Q)",   {"da_rm": [[fact_lo, fact_hi]], "da_rm_at": q_pos}),
        ("paper",      "paper     (fact, end)",    {"da_rm": [[fact_lo, fact_hi]]}),
        ("masked-all", "masked-all (doc, at Q)",   {"da_rm": [[doc_lo, q_pos]], "da_rm_at": q_pos}),
        ("control",    "control   (S3, at Q)  ",   {"da_rm": [[oth_lo, oth_hi]], "da_rm_at": q_pos}),
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
    L_BASE, L_STRICT, L_PAPER, L_MASKED, L_CTRL = (
        "baseline", "strict    (fact, at Q)", "paper     (fact, end)",
        "masked-all (doc, at Q)", "control   (S3, at Q)  ")

    def exact(name):
        return results.get(name) == ANSWER

    def leak_state(name):
        """Classify a removal run: clean / PARTIAL-LEAK / FULL-LEAK / EMPTY."""
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

    checks = []
    if L_BASE in results:
        ok = exact(L_BASE)
        checks.append(ok)
        print("baseline         : %s (exact %r)" % ("PASS" if ok else "FAIL", ANSWER))

    # mechanism proof (both architectures): masked-all removes the whole
    # document before the question is prefilled, so the first generated token
    # (sampled from the last prompt position's logits) must come from a
    # different prefill computation than the baseline. Identical logprobs mean
    # the removal never reached the prefill (post-prefill path only).
    if "baseline" in lps and "masked-all" in lps:
        b_tok, b_lp, _ = lps["baseline"]
        m_tok, m_lp, _ = lps["masked-all"]
        if b_lp is None or m_lp is None:
            checks.append(False)
            print("mechanism (Δlp)  : FAIL (no logprobs: baseline=%s masked-all=%s) - "
                  "check the first_token=... lines above; if this server does not "
                  "return n_probs on /v1/completions, switch to native /completion "
                  "with completion_probabilities"
                  % (b_lp is not None, m_lp is not None))
        else:
            d = round(abs(b_lp - m_lp), 6)
            ok = d > 1e-6
            checks.append(ok)
            print("mechanism (Δlp)  : baseline vs masked-all first-token Δlp=%s — %s"
                  % (d, "different prefill computation (mid-prefill removal ran)"
                     if ok else
                     "IDENTICAL - the removal did not reach the prefill (mid-prefill "
                     "path did not run; journal: expect 'batch fill stopped at "
                     "boundary', look for 'skipped'/'never applied' instead)"))

    if L_MASKED in results:
        st = leak_state(L_MASKED)
        if HYBRID:
            print("mechanism (text) : %s (informational on hybrid — with a Δlp "
                  "divergence above, a leading 'Z' means the fact survives via the "
                  "recurrent state, not the full-attention KV; confirm with the "
                  "codeword-swap control QUOKKA-17)" % st)
        else:
            ok = st == "clean"
            checks.append(ok)
            print("mechanism (text) : %s (masked-all must be clean; %s)"
                  % ("PASS" if ok else "FAIL", st))
    if L_CTRL in results:
        ok = exact(L_CTRL)
        checks.append(ok)
        print("selectivity      : %s (exact %r)" % ("PASS" if ok else "FAIL", ANSWER))
    if L_STRICT in results:
        st = leak_state(L_STRICT)
        if HYBRID:
            print("strict leak      : %s (informational on hybrid — paper semantics, "
                  "surviving-KV leak; see da_probe masked-A)" % st)
        else:
            ok = st == "clean"
            checks.append(ok)
            print("strict isolation : %s (%s)" % ("PASS" if ok else "FAIL", st))

    # path divergence fingerprint: strict (da_rm_at=Q) and paper (no boundary)
    # sample the first token from different computations iff the mid-prefill
    # path ran for strict; identical logprobs mean both took the same path.
    # Applies whenever both runs executed, regardless of the --runs subset;
    # missing logprobs fail instead of slipping through as "n/a".
    if "strict" in lps and "paper" in lps:
        s_tok, s_lp, _ = lps["strict"]
        p_tok, p_lp, _ = lps["paper"]
        if s_lp is None or p_lp is None:
            checks.append(False)
            print("path divergence  : FAIL (no logprobs: strict=%s paper=%s) - the prefill "
                  "path cannot be proven. Check that the first_token=... lines printed "
                  "above; if this server does not return n_probs on /v1/completions, "
                  "switch to native /completion with completion_probabilities"
                  % (s_lp is not None, p_lp is not None))
        else:
            d = round(abs(s_lp - p_lp), 6)
            same = d < 1e-6
            checks.append(not same)
            print("path divergence  : strict vs paper first-token Δlp=%s — %s"
                  % (d, "IDENTICAL, same prefill path (mid-prefill did not run)"
                     if same else "different prefill paths"))

    ok = machine_ok and all(checks) if checks else machine_ok
    print("OVERALL          : %s" % ("PASS" if ok else "FAIL"))
    print("paper run is informational on both architectures (decode-time restriction).")
    print("also check the server terminal for a WARN 'clamping end' line from the masked-all run.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

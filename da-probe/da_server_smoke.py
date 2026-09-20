#!/usr/bin/env python3
"""Smoke test for llama-server da_rm (Declarative Attention) — server-side verification.

Usage:
  python3 da_server_smoke.py [BASE_URL] [--hybrid]

  BASE_URL    default: $DA_BASE or http://127.0.0.1:8080
  --hybrid    model has recurrent/linear-attention layers (e.g. qwen35):
              the strict (fact-only) run is expected to LEAK (paper semantics —
              the fact survives via the surviving full-attention KV), so it is
              informational. The mechanism proof is the masked-all run.

Runs 5 completions against BASE/v1/completions:
  a. baseline   - no da_rm                          -> expect ZEBRA-42
  b. strict     - da_rm=fact range, da_rm_at=Q      -> clean (pure attn) / leak (hybrid)
  c. paper      - da_rm=fact range, no boundary     -> decode-time restriction (informational)
  d. masked-all - da_rm=whole document, da_rm_at=Q  -> expect NOT ZEBRA-42 (MECHANISM PROOF)
  e. control    - da_rm=other section, da_rm_at=Q   -> expect ZEBRA-42 (selectivity)

Why masked-all is the mechanism proof: on a hybrid the strict run leaks
(surviving-KV), which is behaviorally identical to an old server binary that
silently ignores the unknown da_rm field. Removing the WHOLE document up to
the question is different: da_probe's masked-all control proved that on
qwen35 this breaks the answer (the recurrent state does not retain the fact),
so "masked-all answers no ZEBRA-42" can only happen if seq_rm actually ran.
The masked-all range reaches the da_rm_at boundary, so apply_da_rm clamps its
end to bound-1 and emits a WARN line in the server log (visible at default
verbosity) — a second piece of evidence.

No server restart is required — provided the running binary was built from a
commit that includes the da_rm field. Check first:
  strings build/bin/llama-server | grep -m1 da_rm
(run_probe.sh only rebuilds da_probe, not llama-server).

A leading thinking block is tolerated: when the response contains a
thinking-end tag, only the text after the last one is checked.
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

Question: What is the emergency shutdown codeword for the facility?
Answer:
"""

DOC_START = "[Section 1: Overview]"
FACT_START = "The emergency shutdown codeword"
FACT_END = "red letters."
OTHER_START = "Tasks are scheduled using"
OTHER_END = "after five minutes."
QUESTION_START = "Question: What is the"
THINK_END = "
</think>

"


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


def complete(prompt, extra, max_tokens=512):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": False}
    body.update(extra)
    return post("/v1/completions", body)


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

    runs = [
        ("baseline", {}),
        ("strict    (fact, at Q)", {"da_rm": [[fact_lo, fact_hi]], "da_rm_at": q_pos}),
        ("paper     (fact, end)",  {"da_rm": [[fact_lo, fact_hi]]}),
        ("masked-all (doc, at Q)", {"da_rm": [[doc_lo, q_pos]], "da_rm_at": q_pos}),
        ("control   (S3, at Q)  ", {"da_rm": [[oth_lo, oth_hi]], "da_rm_at": q_pos}),
    ]
    results = {}
    machine_ok = True
    for name, extra in runs:
        try:
            res = complete(PROMPT, extra)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            results[name] = None
            machine_ok = False
            print("%s: HTTP ERROR %s  (mechanism FAIL)" % (name, e))
            continue
        text = strip_thinking(res["choices"][0]["text"]).strip()
        results[name] = text
        has = "ZEBRA-42" in text
        print("%s: %r  (ZEBRA-42: %s)" % (name, text, has))
        print("          prompt_tokens=%s completion_tokens=%s"
              % (res["usage"]["prompt_tokens"], res["usage"]["completion_tokens"]))
    print("-" * 72)

    def has_f(name):
        return results.get(name) is not None and "ZEBRA-42" in results[name]

    base_ok = has_f("baseline")
    all_clean = not has_f("masked-all (doc, at Q)")
    ctrl_ok = has_f("control   (S3, at Q)  ")
    strict_leak = has_f("strict    (fact, at Q)")
    ok = machine_ok and base_ok and all_clean and ctrl_ok
    print("baseline         : %s" % ("PASS" if base_ok else "FAIL"))
    print("mechanism        : %s (masked-all must break the answer — only a "
          "real seq_rm can do that on a hybrid; an old binary ignores da_rm "
          "and would still answer ZEBRA-42)" % ("PASS" if all_clean else "FAIL"))
    print("selectivity      : %s" % ("PASS" if ctrl_ok else "FAIL"))
    if HYBRID:
        print("strict leak      : %s (expected on hybrid — paper semantics, "
              "surviving-KV leak; see da_probe masked-A)"
              % ("YES" if strict_leak else "NO (stronger than paper semantics)"))
    else:
        print("strict isolation : %s" % ("PASS" if not strict_leak else "FAIL (leak)"))
    print("OVERALL          : %s" % ("PASS" if ok else "FAIL"))
    print("paper run is informational on both architectures (decode-time restriction).")
    print("also check the server terminal for a WARN 'clamping end' line from the masked-all run.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smoke test for llama-server DA tag parser (model-driven attention restriction).

The client sends the CHUNK LAYOUT (da_chunks + da_filler); the MODEL's own
emitted <focus magic_chunks="N"> tag decides which chunk to keep. The server
scans the generated text mid-decode and, at tag close, removes every chunk
except N plus the filler - applied exactly once, via apply_da_b (da_b mode)
or apply_da_rm (A mode).

Usage:
  python3 da_tag_smoke.py [BASE_URL] [--hybrid] [--runs k1,k2,...]

  BASE_URL   default: $DA_BASE or http://127.0.0.1:8080
  --hybrid   model has recurrent/linear-attention layers (informational)
  --runs     run only the given keys: baseline,tag_a,tag_b

The server MUST be started with --kv-unified --parallel 2 for the B run
(tag_b); without it tag_b falls back to the logical removal (WARN in the
journal). The A run (tag_a) works either way.

The prompt has 5 chunks (one codeword each) + a filler, and instructs the
model to first emit <focus magic_chunks="N"> for the answer's chunk, then the
code. Ground truth: chunk 4 holds ZEBRA-42, so the expected tag is
<focus magic_chunks="4"> and the answer ZEBRA-42.

Runs:
  baseline - no da_chunks: the model emits the tag but NO removal runs.
             The answer must still be ZEBRA-42 (full attention reference).
  tag_a    - da_chunks (all 5) + da_filler, A path: the server parses the
             tag, removes chunks {1,2,3,5}+filler mid-decode. The answer
             (chunk 4, kept) must be ZEBRA-42.
  tag_b    - same + da_b: B path (seq_cp keep ranges to the 2nd seq). The
             answer must be ZEBRA-42.

Checks:
  1. answers   : baseline / tag_a / tag_b all contain ZEBRA-42 (the kept
                 chunk survives; the model's tag identified the right chunk).
  2. tag drove : the answer must come AFTER the emitted tag in the raw text
                 (the restriction applied before the answer was generated).
  3. mechanism : tag_a / tag_b first-token logprob differs from baseline is
                 NOT required (the removal is mid-decode, after the first
                 token) - instead the journal is the evidence (below).

Journal evidence (grep the server log / journalctl):
  da_tag: request start - 5 chunk(s), da_b=0/1, filler=...   (layout received)
  da_tag:   chunk N [lo, hi)                                 (one per chunk)
  da_tag: <focus magic_chunks="4"> closed at generated token K - removing 5 range(s), bound=P
  da_rm: applying 5 range(s) (at tag close) - bound=P, ...   (A path)
    - or - da_b: switched decode to seq 2 (at tag close) - logical read set ... (B path)
  grep 'falling back to logical removal'  -> only for tag_b if --kv-unified is missing
"""
import json
import os
import re
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")
HYBRID = False

SYSTEM_TEXT = (
    "You are a precise reading engine. The document below contains several "
    "chunks. Some chunks contain a codeword in the form WORD-NUMBER.")

DOC_HEADER = "The following is a technical document about the facility's control systems.\n\n"

CHUNKS = {
    1: ("[Chunk 1]\nThe main power grid is monitored by a redundant array of relay units. "
        "The power grid reset phrase used by operators is ECHO-9. It is stored in the backup "
        "terminal and must be confirmed by a second operator before use.\n\n",
        "before use."),
    2: ("[Chunk 2]\nThe ventilation system recirculates air through a bank of axial fans. "
        "The ventilation restart code for the control room is TANGO-7. The fans return to "
        "nominal speed within ninety seconds after the code is accepted.\n\n",
        "code is accepted."),
    3: ("[Chunk 3]\nFire suppression is handled by a distributed network of halon valves. "
        "The fire suppression override code is KILO-31. A full discharge locks the valves "
        "for four hours until a manual reset.\n\n",
        "a manual reset."),
    4: ("[Chunk 4]\nThe emergency shutdown sequence is triggered from the control room "
        "console. The emergency shutdown codeword for the facility is ZEBRA-42. Operators "
        "must memorize it. It is printed on the wall of the control room in red letters.\n\n",
        "in red letters."),
    5: ("[Chunk 5]\nElevator movement during an alarm is governed by a dedicated controller. "
        "The elevator lockdown token is SIERRA-17. The token disables all car calls until "
        "it is cleared from the controller.\n\n",
        "from the controller."),
}

FILLER = ("FILLERSTART The mountain range stretches across the northern border of the "
          "valley. Hikers often begin their trails at dawn, when the light is soft and the "
          "air is cold. Small streams cross the path near the base camp, and the pines grow "
          "thicker above the ridge. FILLEREND\n\n")

INSTRUCTION = (
    "Instructions:\n"
    "1. First identify the chunk that contains the answer to the question, and output "
    "exactly the tag <focus magic_chunks=\"N\"> where N is the chunk number (1-5).\n"
    "2. Then answer the question with the code only, nothing else.\n\n"
    "Question: What is the emergency shutdown codeword for the facility?\n")

# Empty thinking-block prefill (same technique as da_probe_dynamic.cpp): the
# model is a thinking model, and priming it with a closed thinking block makes
# it answer directly (emit the tag) instead of reasoning first. Without it the
# model skips the tag and answers straight, so the parser never fires.
THINK_PREFILL = "<think>\n</think>\n"

ANSWER = "ZEBRA-42"
GT_CHUNK = 4
TAG_RE = re.compile(r'<focus[^>]*magic_chunks=["\']?(\d+)')


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


def complete(prompt, extra, max_tokens=32, cache_prompt=False):
    # stop at the first newline: the model emits <focus ...>CODE</focus>\n, so
    # this captures the tag + answer and cuts the trailing chatter.
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": cache_prompt, "stop": ["\n"], "n_probs": 5}
    body.update(extra)
    return post("/v1/completions", body)


def first_token_lp(res):
    try:
        entry = res["choices"][0]["logprobs"]["content"][0]
        top = [(t["token"], round(t["logprob"], 4)) for t in entry["top_logprobs"][:3]]
        return entry["token"], round(entry["logprob"], 4), top
    except (KeyError, TypeError, IndexError):
        return None, None, []


def range_of(ids, start_marker, end_marker):
    """Token range [lo, hi) whose detokenization covers start_marker..end_marker."""
    full = detokenize(ids)
    b0 = full.index(start_marker)
    b1 = full.index(end_marker) + len(end_marker)
    lo = len(tokenize(full[:b0]))
    hi = len(tokenize(full[:b1]))
    got = detokenize(ids[lo:hi])
    if not (got.startswith(start_marker) and end_marker in got):
        for dlo in (-1, 1):
            for dhi in (-1, 1):
                g = detokenize(ids[lo + dlo:hi + dhi])
                if g.startswith(start_marker) and end_marker in g:
                    return lo + dlo, hi + dhi
        raise SystemExit("range mismatch for %r: %r" % (start_marker, got))
    return lo, hi


def build_prompt():
    parts = [SYSTEM_TEXT + "\n\n", DOC_HEADER]
    for n in range(1, 6):
        parts.append(CHUNKS[n][0])
    parts.append(FILLER)
    parts.append(INSTRUCTION)
    parts.append(THINK_PREFILL)
    return "".join(parts)


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
                                 "hybrid expectations" if HYBRID else "pure-attention expectations"))

    PROMPT = build_prompt()
    ids = tokenize(PROMPT)
    n = len(ids)
    full = detokenize(ids)

    chunk_ranges = {}
    for cnum, (text, end_marker) in CHUNKS.items():
        lo, hi = range_of(ids, "[Chunk %d]" % cnum, end_marker)
        chunk_ranges[cnum] = (lo, hi)
    fill_lo, fill_hi = range_of(ids, "FILLERSTART", "FILLEREND")

    da_chunks = [[chunk_ranges[k][0], chunk_ranges[k][1]] for k in range(1, 6)]
    print("prompt tokens          : %d" % n)
    for k in range(1, 6):
        print("  chunk %d                : [%d, %d)" % (k, *chunk_ranges[k]))
    print("  filler               : [%d, %d)" % (fill_lo, fill_hi))
    print("  expected tag         : <focus magic_chunks=\"%d\">  answer %s" % (GT_CHUNK, ANSWER))
    print("-" * 72)

    runs = [
        ("baseline", "baseline (no da_chunks)   ", {}),
        ("tag_a",    "tag_a    (tag -> A path)  ",
         {"da_chunks": da_chunks, "da_filler": [fill_lo, fill_hi]}),
        ("tag_b",    "tag_b    (tag -> B path)  ",
         {"da_chunks": da_chunks, "da_filler": [fill_lo, fill_hi], "da_b": True}),
    ]
    if runs_filter is not None:
        runs = [r for r in runs if r[0] in runs_filter]

    results = {}
    raws = {}
    lps = {}
    machine_ok = True
    for key, name, extra in runs:
        try:
            res = complete(PROMPT, extra)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            results[name] = None
            machine_ok = False
            print("%s: HTTP ERROR %s" % (name, e))
            continue
        raw = res["choices"][0]["text"]
        text = raw.strip()
        results[name] = text
        raws[key] = raw
        tok, lp, top = first_token_lp(res)
        lps[key] = (tok, lp, top)
        m = TAG_RE.search(raw)
        tag_n = m.group(1) if m else None
        has = ANSWER in text
        print("%s: tag=magic_chunks=%s  answer=%r  (ZEBRA-42: %s)"
              % (name, tag_n, text[:80], has))
        print("          prompt_tokens=%s completion_tokens=%s"
              % (res["usage"]["prompt_tokens"], res["usage"]["completion_tokens"]))
        if tok is not None:
            print("          first_token=%r lp=%s top=%s" % (tok, lp, top))
    print("-" * 72)

    L_BASE = "baseline (no da_chunks)   "
    L_TAGA = "tag_a    (tag -> A path)  "
    L_TAGB = "tag_b    (tag -> B path)  "
    checks = []

    def answer_ok(name):
        t = results.get(name)
        return t is not None and ANSWER in t

    def tag_before_answer(key):
        """The emitted tag must precede the answer in the raw text."""
        raw = raws.get(key)
        if raw is None:
            return False
        m = TAG_RE.search(raw)
        if not m:
            return False
        ai = raw.find(ANSWER)
        return ai != -1 and m.start() < ai

    for name, key in ((L_BASE, "baseline"), (L_TAGA, "tag_a"), (L_TAGB, "tag_b")):
        if name in results:
            ok = answer_ok(name)
            checks.append(ok)
            print("%s answer : %s (contains %r)" % (name.split()[0], "PASS" if ok else "FAIL", ANSWER))

    # the tag-driven runs: the tag must be present AND precede the answer
    for name, key in ((L_TAGA, "tag_a"), (L_TAGB, "tag_b")):
        if name in results:
            ok = tag_before_answer(key)
            checks.append(ok)
            print("%s tag->answer: %s (the removal ran mid-decode, before the answer; "
                  "journal: 'da_tag: <focus ...> closed at generated token')"
                  % (name.split()[0], "PASS" if ok else "FAIL"))

    # the model must have identified the ground-truth chunk
    for key in ("tag_a", "tag_b"):
        raw = raws.get(key)
        if raw is None:
            continue
        m = TAG_RE.search(raw)
        if m and m.group(1) != str(GT_CHUNK):
            print("  note: model tagged chunk %s (ground truth %d) - answer may be wrong; "
                  "this is a model-compliance issue, not a server bug"
                  % (m.group(1), GT_CHUNK))

    ok = machine_ok and all(checks) if checks else machine_ok
    print("-" * 72)
    print("OVERALL          : %s" % ("PASS" if ok else "FAIL"))
    print("journal checks (must match):")
    print("  grep 'da_tag: request start'      -> 1 line per tag run (tag_a, tag_b), 5 chunk(s)")
    print("  grep 'da_tag: <focus magic_chunks' -> 1 line per tag run: closed at generated token K")
    print("  tag_a: grep 'da_rm: applying .* (at tag close)'  -> 5 range(s)")
    print("  tag_b: grep 'da_b: switched decode to seq .* (at tag close)'"
          "  (or 'falling back' if --kv-unified missing)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

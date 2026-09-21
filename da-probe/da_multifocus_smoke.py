#!/usr/bin/env python3
"""P3 smoke test — multi-focus state machine (2+ focus calls in one response).

The paper calls focus 1.4~1.9 times per response, returning to global at the
closing tag. This test asks the model to answer two questions, each wrapped
in its own <focus magic_chunks="N"> ... </focus> pair:

  <focus magic_chunks="4">ZEBRA-42</focus> <focus magic_chunks="2">TANGO-7</focus>

Expected server behavior (B path, needs --kv-unified):
  1. first <focus ...> open  -> restriction 1 (keep chunk 4) mid-decode
  2. </focus>                -> return to global (tail seq_cp back, da_seq freed)
  3. second <focus ...> open -> restriction 2 (keep chunk 2) on the original seq
  4. </focus>                -> return to global again
  5. release()               -> B tail restore + prompt cache KEPT

Checks:
  1. both answers present and correct (ZEBRA-42 after tag 1, TANGO-7 after tag 2)
  2. each answer comes AFTER its own opening tag in the raw text
  3. journal: 2x 'closed at generated token' + 2x return-to-global evidence,
     no cell allocation error lines
  4. next turn (same prefix + new question) reuses the cache:
     'cache reuse: n_past' near the previous turn's total length

Usage:
  python3 da_multifocus_smoke.py [BASE_URL] --log /path/to/server.log

  The server must run with --kv-unified (B path) and --da-prompt-scan.
"""
import json
import os
import re
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")
LOG = None

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
    "Instructions: answer BOTH questions. For each question, first identify the chunk "
    "that contains the answer and output the tag <focus magic_chunks=\"N\"> where N is "
    "the chunk number (1-5), then the code only, then the closing tag </focus>. "
    "Output exactly two tag pairs in order, nothing else.\n"
    "Example of the exact output format (with the real codes, not these):\n"
    "<focus magic_chunks=\"1\">ECHO-9</focus> <focus magic_chunks=\"3\">KILO-31</focus>\n\n"
    "Question 1: What is the emergency shutdown codeword for the facility?\n"
    "Question 2: What is the ventilation restart code for the control room?\n")

THINK_PREFILL = "<think>\n</think>\n"

ANSWERS = [("ZEBRA-42", "4"), ("TANGO-7", "2")]


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


def complete(prompt, extra, max_tokens=64, cache_prompt=True):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": cache_prompt, "stop": ["\n"]}
    body.update(extra)
    return post("/v1/completions", body)


def range_of(ids, start_marker, end_marker):
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


def main():
    global BASE, LOG
    args = sys.argv[1:]
    if "--log" in args:
        i = args.index("--log")
        LOG = args[i + 1]
        del args[i:i + 2]
    if args:
        BASE = args[0]
    if LOG is None:
        raise SystemExit("--log /path/to/server.log is required (server must run with -v)")

    print("base   : %s" % BASE)
    print("journal: %s" % LOG)

    doc_parts = [SYSTEM_TEXT + "\n\n", DOC_HEADER]
    for c in range(1, 6):
        doc_parts.append(CHUNKS[c][0])
    doc_parts.append(FILLER)
    DOC_PREFIX = "".join(doc_parts)
    PROMPT = DOC_PREFIX + INSTRUCTION + THINK_PREFILL

    ids = tokenize(PROMPT)
    n = len(ids)
    doc_n = len(tokenize(DOC_PREFIX))
    full = detokenize(ids)
    chunk_ranges = {}
    for cnum, (text, end_marker) in CHUNKS.items():
        chunk_ranges[cnum] = range_of(ids, "[Chunk %d]" % cnum, end_marker)
    fill_lo, fill_hi = range_of(ids, "FILLERSTART", "FILLEREND")
    da_chunks = [list(chunk_ranges[k]) for k in range(1, 6)]

    print("prompt tokens          : %d" % n)
    for k in range(1, 6):
        print("  chunk %d                : %s" % (k, chunk_ranges[k]))
    print("  filler               : [%d, %d)" % (fill_lo, fill_hi))
    print("  expected             : <focus magic_chunks=\"4\">ZEBRA-42</focus> "
          "<focus magic_chunks=\"2\">TANGO-7</focus>")
    print("-" * 72)

    with open(LOG, "r", errors="replace") as f:
        log_off = f.tell()

    def journal():
        with open(LOG, "r", errors="replace") as f:
            f.seek(log_off)
            data = f.read()
        return [l.strip() for l in data.splitlines()]

    # da_b: True -> B path (2-stream): the original sequence stays intact, so
    # the next turn can reuse the prefix cache (P3 Step 6).
    res = complete(PROMPT, {"da_chunks": da_chunks, "da_filler": [fill_lo, fill_hi],
                            "da_b": True})
    raw = res["choices"][0]["text"]
    print("raw response           : %r" % raw)
    print("-" * 72)

    checks = []

    def check(name, ok, detail):
        checks.append(bool(ok))
        print("%-32s: %s (%s)" % (name, "PASS" if ok else "FAIL", detail))

    # Recognized tags are erased from generated_text, so the response holds
    # the answers only - the tag evidence (numbers, order, count) comes from
    # the journal's 'closed at generated token' lines.
    i1 = raw.find(ANSWERS[0][0])
    i2 = raw.find(ANSWERS[1][0])
    check("both answers, in order", i1 != -1 and i2 != -1 and i1 < i2,
          "%s@%d %s@%d" % (ANSWERS[0][0], i1, ANSWERS[1][0], i2))
    check("no tag leak in response", "<focus" not in raw and "</focus" not in raw,
          "response %r" % raw[:80])

    jl = journal()
    # The server's stdout is buffered, so lines from EARLIER requests that
    # share this log file can flush in after log_off. Anchor to this run's own
    # DA request: count only tag lines after its 'request start' (chronological
    # flush order makes this run's start the last one).
    start_idx = None
    for i, l in enumerate(jl):
        if "da_tag: request start" in l:
            start_idx = i
    if start_idx is not None:
        jl = jl[start_idx + 1:]
    opens = [l for l in jl if "closed at generated token" in l and "magic_chunks" in l]
    returns = [l for l in jl if "returned to GLOBAL" in l]
    # real failures only - startup lines like 'ggml_metal_init: allocating',
    # 'no_alloc' and 'set_abort_callback' are normal and must not match
    errs = [l for l in jl if re.search(
        r"(alloc|memory).{0,20}fail|fail.{0,20}(alloc|memory)|out of memory|"
        r"GGML_ASSERT|core dumped|segfault|abort\(\)", l, re.I)]
    nums = [m.group(1) for l in opens if (m := re.search(r'magic_chunks=["\']?([\d,]+)', l))]
    check("journal 2x focus open", len(opens) >= 2, "%d line(s)" % len(opens))
    check("tag numbers = GT (4, 2)", nums[:2] == ["4", "2"], "got %s" % nums)
    check("journal 2x return", len(returns) >= 2, "%d line(s)" % len(returns))
    check("no cell allocation errors", len(errs) == 0, "%d line(s)" % len(errs))
    for l in opens + returns:
        print("    | %s" % l[:170])

    # next turn: SAME document prefix + a fresh single-question instruction
    # (appending 'Question 3' to the two-pair instruction confused the model
    # into listing questions instead of answering). No da layout -> must
    # reuse the prefix cache (B tail restore on release).
    FOLLOWUP = (DOC_PREFIX +
                "Instructions: answer with the code only, nothing else.\n"
                "Question: What is the fire suppression override code?\n"
                + THINK_PREFILL)
    res2 = complete(FOLLOWUP, {}, max_tokens=32, cache_prompt=True)
    raw2 = res2["choices"][0]["text"]
    print("follow-up response     : %r" % raw2)
    jl2 = [l for l in journal() if "cache reuse" in l]
    reuse = None
    if jl2:
        m = re.search(r"n_past = (\d+)", jl2[-1])
        reuse = int(m.group(1)) if m else None
    check("follow-up answers KILO-31", "KILO-31" in raw2, "answer %r" % raw2[:60])
    check("next-turn cache reuse", reuse is not None and reuse >= doc_n - 64,
          "n_past=%s vs shared doc prefix %d" % (reuse, doc_n))
    if jl2:
        print("    | %s" % jl2[-1][:170])

    ok = all(checks)
    print("-" * 72)
    print("OVERALL                 : %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

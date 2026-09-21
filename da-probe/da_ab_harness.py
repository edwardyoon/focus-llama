#!/usr/bin/env python3
"""P5 Step 2 A/B harness - vanilla vs DA on the SAME request.

Runs the identical tag-driven prompt twice per round:
  vanilla - no da_* fields: the model emits the <focus> tag but NO removal
            runs (full attention, tags stay visible in the output)
  da_b    - da_chunks (all K) + da_filler + da_b: the server parses the tag
            and removes every chunk except N mid-decode (B path, 2-stream)

Per run it records from __verbose.timings:
  prompt_ms, predicted_ms (decode), predicted_n (decode steps),
  da_n_attended_tokens / da_n_restricted_steps / da_path (DA run only),
  answer correctness, and total_ms = prompt_ms + predicted_ms.

All runs use cache_prompt=false (cold prefill) so the comparison is not
contaminated by prefix-cache hits.

Spec baselines: this model (Bonsai-8B) has NO MTP head, so spec on/off is
not applicable locally - both arms run spec-less. On 123 (27B, MTP) the
same harness must be run against spec-on and spec-off servers; see the
TODO P5 Step 2 note for the user-execution steps.

Usage:
  python3 da_ab_harness.py [BASE_URL] [--runs N] [--filler-k M]

  --runs N     repeat rounds (default 3)
  --filler-k M filler repetition count (default 8, ~2.7K filler tokens)
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")

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
    "Instructions: identify the chunk that contains the answer and output the tag "
    "<focus magic_chunks=\"N\"> where N is the chunk number (1-5), then the code only, "
    "then the closing tag </focus>. Output exactly one tag pair, nothing else.\n\n"
    "Question: What is the emergency shutdown codeword for the facility?\n")

THINK_PREFILL = "<think>\n</think>\n"

ANSWER = "ZEBRA-42"
GT_CHUNK = 4


def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(text):
    return post("/tokenize", {"content": text})["tokens"]


def detokenize(ids):
    return post("/detokenize", {"tokens": ids})["content"]


def complete(prompt, extra):
    body = {"prompt": prompt, "max_tokens": 48, "temperature": 0,
            "cache_prompt": False, "stop": ["\n"]}
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


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    global BASE
    args = sys.argv[1:]
    runs = 3
    filler_k = 8
    if "--runs" in args:
        i = args.index("--runs")
        runs = int(args[i + 1])
        del args[i:i + 2]
    if "--filler-k" in args:
        i = args.index("--filler-k")
        filler_k = int(args[i + 1])
        del args[i:i + 2]
    if args:
        BASE = args[0]

    parts = [SYSTEM_TEXT + "\n\n", DOC_HEADER]
    for c in range(1, 6):
        parts.append(CHUNKS[c][0])
    parts.append(FILLER * filler_k)
    parts.append(INSTRUCTION)
    parts.append(THINK_PREFILL)
    PROMPT = "".join(parts)

    ids = tokenize(PROMPT)
    n = len(ids)
    chunk_ranges = {}
    for cnum, (text, end_marker) in CHUNKS.items():
        chunk_ranges[cnum] = range_of(ids, "[Chunk %d]" % cnum, end_marker)
    # da_filler = the filler block only (between the last chunk and the
    # instruction). The instruction is scaffold (kept), not filler (removed).
    last_hi = max(hi for _, hi in chunk_ranges.values())
    instr_lo = len(tokenize(PROMPT[:PROMPT.index("Instructions:")]))
    da_chunks = [list(chunk_ranges[k]) for k in range(1, 6)]

    print("base        : %s" % BASE)
    print("prompt tokens: %d  (filler x%d, da_filler=[%d, %d))"
          % (n, filler_k, last_hi, instr_lo))
    for k in range(1, 6):
        print("  chunk %d     : %s" % (k, chunk_ranges[k]))
    print("-" * 78)

    arms = [
        ("vanilla", {}),
        # da_filler = the filler block only; the instruction/question is
        # scaffold (kept) - covering it would remove the question in FOCUS
        # mode and the model regurgitates the document instead of answering
        ("da_b", {"da_chunks": da_chunks, "da_filler": [last_hi, instr_lo], "da_b": True}),
    ]

    rows = []  # (arm, round, timings-dict, ok)
    for arm, extra in arms:
        for r in range(1, runs + 1):
            t0 = time.monotonic()
            res = complete(PROMPT, extra)
            wall = (time.monotonic() - t0) * 1000.0
            text = res["choices"][0]["text"]
            t = res.get("__verbose", {}).get("timings", {})
            ok = ANSWER in text
            rows.append((arm, r, t, ok, wall, text))
            print("%-7s r%d: %s  prompt=%.0fms decode=%.0fms steps=%s "
                  "attended=%s restricted=%s path=%s wall=%.0fms  %r"
                  % (arm, r, "OK " if ok else "BAD",
                     t.get("prompt_ms", -1), t.get("predicted_ms", -1),
                     t.get("predicted_n", "?"),
                     t.get("da_n_attended_tokens", "-"),
                     t.get("da_n_restricted_steps", "-"),
                     t.get("da_path", "-"), wall, text[:60]))
    print("-" * 78)

    # summary table
    print("%-8s %-12s %-12s %-10s %-12s %-10s" %
          ("arm", "prompt_ms", "decode_ms", "steps", "attended/step", "total_ms"))
    for arm, _ in arms:
        ts = [t for a, r, t, ok, w, x in rows if a == arm]
        ok_n = sum(1 for a, r, t, ok, w, x in rows if a == arm and ok)
        attended = [t.get("da_n_attended_tokens", 0) for t in ts]
        restricted = [t.get("da_n_restricted_steps", 0) for t in ts]
        att_per = (sum(attended) / sum(restricted)) if sum(restricted) else None
        tot = mean([t.get("prompt_ms", 0) + t.get("predicted_ms", 0) for t in ts])
        print("%-8s %-12.0f %-12.0f %-10s %-12s %-10.0f   (answer OK %d/%d)"
              % (arm, mean([t.get("prompt_ms", 0) for t in ts]),
                 mean([t.get("predicted_ms", 0) for t in ts]),
                 mean([t.get("predicted_n", 0) for t in ts]),
                 ("%.0f" % att_per) if att_per is not None else "n/a (no removal)",
                 tot, ok_n, runs))

    # attended reduction (logical): DA attended vs vanilla full prompt
    ts_da = [t for a, r, t, ok, w, x in rows if a == "da_b"]
    restricted_tot = sum(t.get("da_n_restricted_steps", 0) for t in ts_da)
    attended_tot = sum(t.get("da_n_attended_tokens", 0) for t in ts_da)
    if restricted_tot:
        full_baseline = restricted_tot * n
        print("-" * 78)
        print("logical attended reduction: DA %d tok over %d restricted step(s) vs "
              "vanilla baseline %d -> %.1f%% attended"
              % (attended_tot, restricted_tot, full_baseline,
                 100.0 * attended_tot / full_baseline))
    else:
        print("-" * 78)
        print("NOTE: DA arm never restricted (no tag fired?) - check the journal")
    print("NOTE: 'attended' is a LOGICAL counter (mask cells), not measured bytes.")
    print("NOTE: local model has no MTP head -> spec on/off baselines are 123-only.")


if __name__ == "__main__":
    main()

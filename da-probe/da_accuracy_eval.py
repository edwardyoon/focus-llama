#!/usr/bin/env python3
"""P5 Step 3 accuracy eval - vanilla vs DA (B path) on synthetic tasks.

Goal: prove DA (attention restriction) does NOT degrade answer accuracy.
Both arms receive the SAME prompt (including the tag protocol instruction);
only the DA arm carries the chunk layout, so its emitted <focus> tags
actually restrict attention mid-decode (B path). The vanilla arm's tags are
inert (no layout). A correct answer under DA means the model kept the chunk(s)
it depended on; a wrong one means it mislabeled a chunk and removed the
answer - the core DA accuracy risk.

Tasks (synthetic, exact ground truth):
  needle          - retrieve one codeword from a 5-chunk document
  multi-key       - answer 3 questions, each from a different chunk (3 tag pairs)
  variable-track  - a counter mutated across 4 chunks; report the final value
                    (the model must keep ALL 4 chunks to get it right)

Context length is set by --target (the server must run with -c >= target).
Locally feasible: 8192 / 32768. 100K is 123-only (see the TODO note).

Per (task, arm) it reports correct/N, a Wilson 95% CI, mean prompt/decode ms,
and (DA arm) the tag-firing rate (da_n_restricted_steps > 0) plus mean
attended tokens per restricted step.

Each run is cold-prefill (cache_prompt=false), temperature 0, stop=["\\n"].
Per-run results are appended to --out (JSONL) so a long background run keeps
partial data if interrupted.

Usage:
  python3 da_accuracy_eval.py [BASE_URL] --target 8192 --runs 10 \
      --log /path/to/server.log --out /tmp/da_acc_8k.jsonl

  --target N    context length in tokens to pad the filler up to (default 8192)
  --runs N      repeats per (task, arm) (default 10)
  --tasks a,b   comma subset of needle,multi-key,variable-track (default all)
"""
import json
import math
import os
import re
import sys
import time
import urllib.request

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")
LOG = None
OUT = None

SYSTEM = (
    "You are a precise reading engine. The document below contains several "
    "chunks. Some chunks contain a codeword in the form WORD-NUMBER.")

CODE_HEADER = ("The following is a technical document about the facility's "
               "control systems.\n\n")

# 5-chunk codeword document (same proven chunks as the P3/P5 smokes)
CODE_CHUNKS = {
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

COUNTER_HEADER = ("The following is a technical document describing a sequence of "
                  "operations on a single variable.\n\n")

# 4-chunk counter document. Final value: ((5 + 3) * 2) - 1 = 15.
COUNTER_CHUNKS = {
    1: ("[Chunk 1]\nThe counter variable is initialized to the value 5.\n\n",
        "the value 5."),
    2: ("[Chunk 2]\nThe counter is increased by 3.\n\n",
        "increased by 3."),
    3: ("[Chunk 3]\nThe counter is doubled.\n\n",
        "is doubled."),
    4: ("[Chunk 4]\nThe counter is decreased by 1.\n\n",
        "decreased by 1."),
}

FILLER_UNIT = ("The mountain range stretches across the northern border of the "
               "valley. Hikers often begin their trails at dawn, when the light is "
               "soft and the air is cold. Small streams cross the path near the base "
               "camp, and the pines grow thicker above the ridge. ")

TAG_RULE = (
    "For each question, first identify the chunk(s) that contain the answer and "
    "output the tag <focus magic_chunks=\"N\"> where N is the chunk number "
    "(use a comma list if several chunks are needed), then the answer, then the "
    "closing tag </focus>.\n")

THINK_PREFILL = "<think>\n</think>\n"

TASKS = {
    "needle": {
        "field": "code",
        "instruction": (
            "Instructions: " + TAG_RULE +
            "Output exactly one tag pair, nothing else.\n\n"
            "Question: What is the emergency shutdown codeword for the facility?\n"),
        "answers": ["ZEBRA-42"],
        "check": lambda t: "ZEBRA-42" in t,
    },
    "multi-key": {
        "field": "code",
        "instruction": (
            "Instructions: " + TAG_RULE +
            "Output exactly three tag pairs in order, nothing else.\n"
            "Example of the exact output format (with the real codes, not these):\n"
            "<focus magic_chunks=\"1\">ECHO-9</focus> "
            "<focus magic_chunks=\"3\">KILO-31</focus> "
            "<focus magic_chunks=\"5\">SIERRA-17</focus>\n\n"
            "Question 1: What is the emergency shutdown codeword for the facility?\n"
            "Question 2: What is the ventilation restart code for the control room?\n"
            "Question 3: What is the fire suppression override code?\n"),
        "answers": ["ZEBRA-42", "TANGO-7", "KILO-31"],
        "check": lambda t: all(a in t for a in ["ZEBRA-42", "TANGO-7", "KILO-31"]),
    },
    "variable-track": {
        "field": "counter",
        "instruction": (
            "Instructions: " + TAG_RULE +
            "Output exactly one tag pair, nothing else.\n\n"
            "Question: What is the final value of the counter after all operations? "
            "Reply with the number only.\n"),
        "answers": ["15"],
        "check": lambda t: re.search(r"\b15\b", t) is not None,
    },
}


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


def complete(prompt, extra, max_tokens=48):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "cache_prompt": False, "stop": ["\n"]}
    body.update(extra)
    return post("/v1/completions", body)


def range_of(ids, start_marker, end_marker):
    """Token range [lo, hi) whose detokenization covers start..end marker."""
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


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - m) / d, (c + m) / d)


def build_prompt(task_name, target):
    """Assemble the padded prompt and the DA layout for one task."""
    spec = TASKS[task_name]
    if spec["field"] == "code":
        header, chunks = CODE_HEADER, CODE_CHUNKS
    else:
        header, chunks = COUNTER_HEADER, COUNTER_CHUNKS

    scaffold = SYSTEM + "\n\n" + header + "".join(chunks[i][0] for i in sorted(chunks))
    instr_block = spec["instruction"] + THINK_PREFILL

    # pad the filler (between the last chunk and the instruction) up to target
    filler = FILLER_UNIT
    n = len(tokenize(scaffold + filler + instr_block))
    while n < target - 256:
        filler += FILLER_UNIT
        n = len(tokenize(scaffold + filler + instr_block))
    prompt = scaffold + filler + instr_block
    ids = tokenize(prompt)

    chunk_ranges = {}
    for i in sorted(chunks):
        chunk_ranges[i] = range_of(ids, "[Chunk %d]" % i, chunks[i][1])
    filler_lo = max(hi for _, hi in chunk_ranges.values())
    filler_hi = len(tokenize(prompt[:prompt.index("Instructions:")]))
    da = {"da_chunks": [list(chunk_ranges[i]) for i in sorted(chunk_ranges)],
          "da_filler": [filler_lo, filler_hi], "da_b": True}
    return prompt, da, len(ids), chunk_ranges, (filler_lo, filler_hi)


def main():
    global BASE, LOG, OUT
    args = sys.argv[1:]
    target = 8192
    runs = 10
    task_sel = None
    if "--log" in args:
        i = args.index("--log")
        LOG = args[i + 1]
        del args[i:i + 2]
    if "--out" in args:
        i = args.index("--out")
        OUT = args[i + 1]
        del args[i:i + 2]
    if "--target" in args:
        i = args.index("--target")
        target = int(args[i + 1])
        del args[i:i + 2]
    if "--runs" in args:
        i = args.index("--runs")
        runs = int(args[i + 1])
        del args[i:i + 2]
    if "--tasks" in args:
        i = args.index("--tasks")
        task_sel = [t.strip() for t in args[i + 1].split(",") if t.strip()]
        del args[i:i + 2]
    if args:
        BASE = args[0]

    tasks = task_sel or list(TASKS)
    print("base    : %s" % BASE)
    print("target  : %d   runs: %d   tasks: %s" % (target, runs, ",".join(tasks)))
    print("log     : %s" % LOG)
    print("out     : %s" % OUT)
    print("-" * 96)

    out_f = open(OUT, "a") if OUT else None
    records = []  # (task, arm, run, ok, t-dict, text)
    for task in tasks:
        prompt, da, n, cr, fr = build_prompt(task, target)
        print("[%s] prompt=%d tok  chunks=%s  filler=[%d,%d)"
              % (task, n, {k: list(v) for k, v in cr.items()}, fr[0], fr[1]))
        for arm, extra in (("vanilla", {}), ("da_b", da)):
            for r in range(1, runs + 1):
                t0 = time.monotonic()
                res = complete(prompt, extra)
                wall = (time.monotonic() - t0) * 1000.0
                text = res["choices"][0]["text"]
                t = res.get("__verbose", {}).get("timings", {})
                ok = TASKS[task]["check"](text)
                rec = {"task": task, "arm": arm, "run": r, "ok": ok,
                       "n": n, "prompt_ms": t.get("prompt_ms"),
                       "decode_ms": t.get("predicted_ms"),
                       "steps": t.get("predicted_n"),
                       "restricted": t.get("da_n_restricted_steps", 0),
                       "attended": t.get("da_n_attended_tokens", 0),
                       "path": t.get("da_path", "-"),
                       "wall_ms": round(wall), "text": text[:120]}
                records.append(rec)
                if out_f:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                print("  %-13s %-7s r%02d %s prompt=%s decode=%s steps=%s "
                      "restricted=%s attended=%s %r"
                      % (task, arm, r, "OK " if ok else "BAD",
                         t.get("prompt_ms"), t.get("predicted_ms"),
                         t.get("predicted_n"), t.get("da_n_restricted_steps", 0),
                         t.get("da_n_attended_tokens", 0), text[:50]))
        print("-" * 96)

    if out_f:
        out_f.close()

    # summary
    print("%-15s %-8s %-10s %-22s %-12s %-12s %-14s"
          % ("task", "arm", "acc", "wilson95", "prompt_ms", "decode_ms",
             "fired/attended"))
    summary = {}
    for task in tasks:
        for arm in ("vanilla", "da_b"):
            rs = [x for x in records if x["task"] == task and x["arm"] == arm]
            if not rs:
                continue
            k = sum(1 for x in rs if x["ok"])
            n = len(rs)
            p, lo, hi = wilson(k, n)
            prompt_ms = sum(x["prompt_ms"] or 0 for x in rs) / n
            decode_ms = sum(x["decode_ms"] or 0 for x in rs) / n
            if arm == "da_b":
                fired = sum(1 for x in rs if x["restricted"] > 0)
                restr = sum(x["restricted"] for x in rs)
                att = sum(x["attended"] for x in rs)
                att_s = ("%.0f" % (att / restr)) if restr else "n/a"
                fired_s = "%d/%d f, %s/step" % (fired, n, att_s)
            else:
                fired_s = "-"
            summary[(task, arm)] = (k, n, p, lo, hi, prompt_ms, decode_ms, fired_s)
            print("%-15s %-8s %d/%d (%.0f%%)  [%.2f,%.2f]  %-12.0f %-12.0f %-14s"
                  % (task, arm, k, n, 100 * p, lo, hi, prompt_ms, decode_ms, fired_s))
    print("-" * 96)

    # parity verdict per task: DA accuracy CI should overlap vanilla
    for task in tasks:
        v = summary.get((task, "vanilla"))
        d = summary.get((task, "da_b"))
        if not v or not d:
            continue
        parity = d[2] >= v[3] - 0.02  # DA point est within/at vanilla CI lower
        print("%-15s vanilla %d/%d  da_b %d/%d  -> %s"
              % (task, v[0], v[1], d[0], d[1],
                 "PARITY" if parity else "DIVERGENT (inspect)"))
    print("NOTE: 'fired' = DA runs where a tag restricted attention; 'attended/step'")
    print("      is the logical attended-token counter (masked cells), not bytes.")
    print("NOTE: 100K depth is 123-only (local -c caps at 8K/32K for this model).")


if __name__ == "__main__":
    main()

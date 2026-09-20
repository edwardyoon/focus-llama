#!/usr/bin/env python3
"""Phase 0 — DA-nm compliance probe (mask-free): does the model follow the
magic-chunk protocol zero-shot?

Protocol under test (DA paper "DA-nm" ablation, no mask):
  - the document is presented as numbered chunks [Chunk 1]..[Chunk 5]
  - the model is instructed to first emit <focus magic_chunks="N"> for the
    chunk that contains the answer, then answer the question

Measured per question (no pass/fail verdict — Phase 0 is a measurement):
  tag     - the model emitted a <focus magic_chunks=...> tag at all
  correct - the tag names the chunk that actually contains the fact
  answer  - the expected code appears in the reply

Why: before Phase 1b invests in dynamic tag parsing, this separates
"the model cannot follow the protocol" (needs fine-tuning) from
"the engine path is broken" (mask/seq_rm bug). Paper: 8B 1-bit class
shows near-zero zero-shot compliance; 27B class is the meaningful target.

Usage:
  python3 da_phase0.py [BASE_URL] [--repeat N]

  BASE_URL    default: $DA_BASE or http://127.0.0.1:8080
  --repeat N  repeat each question N times (default 1)

Uses /v1/chat/completions with chat_template_kwargs enable_thinking=false
(a thinking model would otherwise bury the tag in a reasoning block). If
the server rejects the parameter (HTTP 400) it retries without it.
"""
import json
import os
import re
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")
REPEAT = 1

DOC = """The following is a technical document about the facility's control systems.

[Chunk 1]
The main power grid is monitored by a redundant array of relay units. The power grid reset phrase used by operators is ECHO-9. It is stored in the backup terminal and must be confirmed by a second operator before use.

[Chunk 2]
The ventilation system recirculates air through a bank of axial fans. The ventilation restart code for the control room is TANGO-7. The fans return to nominal speed within ninety seconds after the code is accepted.

[Chunk 3]
Fire suppression is handled by a distributed network of halon valves. The fire suppression override code is KILO-31. A full discharge locks the valves for four hours until a manual reset.

[Chunk 4]
The emergency shutdown sequence is triggered from the control room console. The emergency shutdown codeword for the facility is ZEBRA-42. Operators must memorize it. It is printed on the wall of the control room in red letters.

[Chunk 5]
Elevator movement during an alarm is governed by a dedicated controller. The elevator lockdown token is SIERRA-17. The token disables all car calls until it is cleared from the controller."""

INSTRUCTION = """Instructions:
1. First identify the chunk that contains the answer to the question, and output exactly the tag <focus magic_chunks="N"> where N is the chunk number (1-5).
2. Then answer the question with the code only, nothing else."""

# (question, expected chunk, expected code)
QUESTIONS = [
    ("What is the power grid reset phrase used by operators?", 1, "ECHO-9"),
    ("What is the ventilation restart code for the control room?", 2, "TANGO-7"),
    ("What is the fire suppression override code?", 3, "KILO-31"),
    ("What is the emergency shutdown codeword for the facility?", 4, "ZEBRA-42"),
    ("What is the elevator lockdown token?", 5, "SIERRA-17"),
]

TAG_RE = re.compile(
    r'<focus\b[^>]*magic_chunks\s*=\s*["\']?(\d+(?:\s*,\s*\d+)*)', re.IGNORECASE)


def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(question):
    body = {
        "messages": [
            {"role": "user",
             "content": DOC + "\n\n" + INSTRUCTION + "\n\nQuestion: " + question + "\n"},
        ],
        "temperature": 0,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        res = post("/v1/chat/completions", body)
    except urllib.error.HTTPError as e:
        if e.code == 400:
            # server without chat_template_kwargs support
            body.pop("chat_template_kwargs")
            res = post("/v1/chat/completions", body)
        else:
            raise
    return res["choices"][0]["message"]["content"]


def parse_reply(text, expected_chunk, expected_code):
    m = TAG_RE.search(text)
    tag_emitted = m is not None
    tag_correct = False
    if m:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        tag_correct = expected_chunk in nums
    answer_ok = expected_code.lower() in text.lower()
    return tag_emitted, tag_correct, answer_ok


def main():
    global BASE, REPEAT
    args = sys.argv[1:]
    if "--repeat" in args:
        i = args.index("--repeat")
        REPEAT = int(args[i + 1])
        del args[i:i + 2]
    if args:
        BASE = args[0]
    print("base   : %s" % BASE)
    print("repeat : %d" % REPEAT)
    print("-" * 78)

    machine_ok = True
    n_tag = n_correct = n_answer = n_total = 0
    for q, chunk, code in QUESTIONS:
        for r in range(REPEAT):
            try:
                text = chat(q).strip()
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                machine_ok = False
                print("%-52s HTTP ERROR %s" % (q, e))
                continue
            n_total += 1
            emitted, correct, answer = parse_reply(text, chunk, code)
            n_tag += emitted
            n_correct += correct
            n_answer += answer
            first_line = text.splitlines()[0] if text else ""
            print("%-52s chunk=%d -> %s | tag=%s correct=%s answer=%s"
                  % (q, chunk, (first_line[:60] + "...") if len(first_line) > 60 else first_line,
                     emitted, correct, answer))
            if REPEAT > 1:
                for extra in text.splitlines()[1:]:
                    if extra.strip():
                        print("    | " + extra[:100])
    print("-" * 78)
    if n_total:
        print("questions      : %d" % n_total)
        print("tag rate       : %d/%d  (%.0f%%)" % (n_tag, n_total, 100.0 * n_tag / n_total))
        print("tag accuracy   : %d/%d  (%.0f%% of emitted)"
              % (n_correct, n_tag, 100.0 * n_correct / n_tag) if n_tag else
              "tag accuracy   : n/a (no tags emitted)")
        print("answer rate    : %d/%d  (%.0f%%)" % (n_answer, n_total, 100.0 * n_answer / n_total))
    print("Phase 0 is a measurement, not a verdict: low compliance here means")
    print("the protocol needs fine-tuning; it says nothing about the engine path.")
    sys.exit(0 if machine_ok else 1)


if __name__ == "__main__":
    main()

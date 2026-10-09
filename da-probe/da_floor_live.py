#!/usr/bin/env python3
"""Live test for the DA tail floor (plans/continous_work2.md) — the amnesia fix.

Scenario: the model's selection (da_rm) nominally removes a region that
INCLUDES the fact the question asks about. The fact sits within the most
recent K tokens before the removal boundary, so the tail floor
[bound-K, bound) must protect it.

  Without the floor (server --da-tail-keep 0):
      the fact is removed -> the model cannot answer ZEBRA-42 (amnesia).
  With the floor (server default --da-tail-keep 8192):
      the floor keeps the fact -> the model answers exactly ZEBRA-42.

The server MUST be started with --kv-unified --parallel 2 (B path).
The journal is the real evidence:
  grep 'tail_keep=' <journal>     # expect '... token(s) newly kept'
  grep 'da_b: switched' <journal> # expect a switch (not a fallback/skip)

Usage:
  python3 da_floor_live.py [BASE_URL] [--expect floor|legacy]
    --expect floor   (default) answer must be exactly ZEBRA-42
    --expect legacy  answer must NOT be ZEBRA-42 (fact was removed)
"""
import json
import os
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else os.environ.get("DA_BASE", "http://127.0.0.1:8085")
EXPECT = "floor"
if "--expect" in sys.argv:
    EXPECT = sys.argv[sys.argv.index("--expect") + 1]

# A fact the question asks about. It will be placed INSIDE the da_rm range,
# within the last K tokens of the prompt, so only the tail floor saves it.
FACT = ("The emergency shutdown codeword for the facility is ZEBRA-42. "
        "Operators must memorize it. It is printed on the wall in red letters.\n")
QUESTION = ("Question: What is the emergency shutdown codeword for the "
            "facility? Reply with the codeword only, nothing else.\nAnswer:\n")

# Repeating padding to push the prompt well past K=8192 tokens.
FILLER_UNIT = ("The scheduler dispatches tasks to a worker pool in round-robin "
               "order and records each completion in a durable log entry. ")


def post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"])


def build_prompt(target_filler_tokens):
    # grow the filler until it is at least target_filler_tokens long
    reps = max(1, target_filler_tokens // max(1, ntok(FILLER_UNIT)))
    filler = FILLER_UNIT * reps
    return filler, ntok(filler)


def main():
    # target: filler ~ 10000 tokens so bound (end of prompt) is ~10600 and
    # [bound-8192, bound) reaches well into the fact region.
    filler, f_len = build_prompt(10000)
    fact_len = ntok(FACT)
    q_len = ntok(QUESTION)

    prompt = filler + FACT + QUESTION
    n_prompt = ntok(prompt)
    bound = n_prompt  # remove boundary = end of prompt (question position)
    # da_rm covers the filler AND the fact (the fact is deliberately inside
    # the removal range); the question stays (it is at/after bound).
    rm_hi = f_len + fact_len
    print(f"prompt_tokens={n_prompt} filler={f_len} fact_len={fact_len} "
          f"q_len={q_len} bound={bound} rm=[0, {rm_hi})")
    print(f"floor [bound-8192, bound) = [{bound - 8192}, {bound}); "
          f"fact at [{f_len}, {rm_hi}) -> "
          f"{'inside floor (protected)' if f_len >= bound - 8192 else 'OUTSIDE floor (removed)'}")
    if f_len < bound - 8192:
        print("FATAL: fact is outside the floor - the test would not exercise "
              "the floor. Increase the filler target.")
        return 2

    body = {"prompt": prompt, "max_tokens": 24, "temperature": 0,
            "cache_prompt": False, "stop": ["\n"], "n_probs": 5,
            "da_rm": [[0, rm_hi]], "da_rm_at": bound, "da_b": True}
    res = post("/v1/completions", body)
    text = res["choices"][0]["text"].strip()
    print(f"answer: {text!r}")

    ok = (text == "ZEBRA-42") if EXPECT == "floor" else (text != "ZEBRA-42")
    print(f"expect={EXPECT}: {'PASS' if ok else 'FAIL'} "
          f"({'fact protected by floor' if EXPECT == 'floor' else 'fact removed (amnesia reproduced)'}: "
          f"answer={text!r})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

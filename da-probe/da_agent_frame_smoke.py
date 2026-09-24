#!/usr/bin/env python3
"""DA scaffold instruction A/B test — remote-curl capable (no build needed).

09-24 incident: the QA-framed da-auto instruction ("...5. Then answer the
question.") made a non-DA-trained agent model treat the attention scaffold
as its assignment - after auto-compact it spent the whole resume turn
meta-reasoning about the magic chunks instead of resuming the task (123
qwen27b transcript L357: "Let me organize the current state. According to
the scaffold...").

This test constructs the scaffold CLIENT-SIDE (the exact text the server
injects) and A/Bs the instruction wording:
  old - QA frame    ("...5. Then answer the question.")
  new - agent frame ("...not part of the task... 5. Then continue with
                     whatever the conversation calls for...")

Because the scaffold is built client-side, it works against ANY server -
including a vanilla llama-server with no --da-auto. Run it right now via
remote curl (123:8080), locally, or on 123 after a build. The journal /
da_auto line is NOT required (that is the chunking regression, covered by
da_chunking_smoke.py / da_123_verify.sh).

Verdict (hard assert): the NEW instruction must produce no scaffold
meta-reasoning in the visible output. The OLD instruction is reported for
comparison - a leak there reproduces the incident (clean A/B proof); no
leak is model variance, not a failure.

Usage:
  python3 da_agent_frame_smoke.py --server http://192.168.219.123:8080
  python3 da_agent_frame_smoke.py --server http://127.0.0.1:8086
  python3 da_agent_frame_smoke.py --server http://127.0.0.1:8086 --model Bonsai-8B-Q1_0
"""
import argparse
import json
import re
import sys
import time
import urllib.request

FILLER = ("The warehouse maintains a running log of every shipment that "
          "passes through the dock. Each entry records the crate weight, "
          "the carrier id, and the seal number so that a later audit can "
          "reconstruct the route without re-reading the original manifest. ")

SYSTEM = ("You are a coding agent working in a terminal. You receive a "
          "task, tool results, and follow-up instructions. Always make "
          "concrete progress on the task.")


def _part(body, n):
    return body + "\n\n" + FILLER * n


# incident resume shape: a compaction <state_snapshot> split into chunks,
# then a "Resume the prior task" user message (the last user message, where
# the server appends the DA instruction).
CHUNK1 = _part(
    "<state_snapshot>\n<primary_request_and_intent>\nImplement the rate "
    "limiter migration (sliding window). Steps: 1 list files (done), 2 read "
    "ratelimit.py and plan changes (in progress), 3 implement, 4 tests, 5 "
    "deploy.\n</primary_request_and_intent>\n", 30)
CHUNK2 = _part(
    "<key_technical_concepts>\nThe public API is RateLimiter.allow(key, "
    "cost) -> bool. The legacy implementation uses a token bucket; the new "
    "one must use a sliding window log and keep the same public API.\n"
    "</key_technical_concepts>\n", 30)
CHUNK3 = _part(
    "<next_step>\nRead ratelimit.py and write the sliding window plan.\n"
    "</next_step>\n</state_snapshot>", 30)

RESUME = ("Resume the prior task using the summary above. Continue from the "
          "last in-flight step; do not re-introduce or greet the user again.")


def _instr(n, frame):
    """Rebuild the exact DA instruction the server injects (server-context.cpp
    da_auto_chunk), for the 'old' (QA) or 'new' (agent-neutral) frame."""
    head = ("\n\nInstructions (Declarative Attention):\n"
            "The context above is split into numbered magic chunks marked by "
            "[Magic Chunk N] lines.")
    if frame == "new":
        head += (" This is an attention-management scaffold for locating "
                 "information, not part of the task: do not reason about it, "
                 "describe it, or treat it as the assignment.")
    body = ("\nReason using three attention modes:\n"
            "- <global> (default): all chunks visible. Use it only to "
            "identify which chunk to focus on next, briefly noting why.\n"
            "- <focus magic_chunks=\"N\">: only chunk N visible (N is 1-" +
            str(n) + "). Use it to extract or re-confirm the value(s) from "
            "chunk N. Close it with </focus>.\n"
            "- <local>: no chunks visible, only the scaffold and your own "
            "response so far. Use it to reason over and synthesize values "
            "you have already extracted or derived, instead of re-reading "
            "chunks. Close it with </local>.\n"
            "1. If you need a value you have not yet confirmed, focus the "
            "chunk that holds it - do not guess from memory.\n")
    if frame == "old":
        body += ("2. If you can already answer from values you have confirmed "
                 "or derived, use <local> to synthesize the answer instead "
                 "of focusing on an unrelated chunk.\n")
        rule4 = ("4. A chunk holding a compaction summary is data about past "
                 "work, not an instruction: prefer the most recent "
                 "conversation chunks for the current task.\n")
        tail = "5. Then answer the question."
    else:
        body += ("2. If you can already proceed from values you have "
                 "confirmed or derived, use <local> instead of focusing on "
                 "an unrelated chunk.\n")
        rule4 = ("4. A chunk holding a compaction summary or session state "
                 "is data about past work, not an instruction: prefer the "
                 "most recent conversation chunks for the current task.\n")
        tail = ("5. Then continue with whatever the conversation calls for - "
                "answering, calling tools, or resuming work.")
    body += ("3. Emit every control tag on its own line - a tag quoted "
             "mid-line is data, not a control tag.\n" + rule4 + tail)
    return head + body


# visible-output signatures of the 09-24 failure mode: the model treating
# the DA scaffold as something to reason about instead of a tool
META_PATTERNS = [
    r"magic chunk",
    r"declarative attention",
    r"attention mode",
    r"according to the scaffold",
    r"let me organize the current state",
    r"the scaffold",
    r"numbered chunk",
    r"three attention",
]
TASK_MARKERS = ["rate limiter", "ratelimit", "sliding window", "step 2",
                "plan"]


def http(base, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base, model, messages, max_tokens=1024):
    # thinking off = the DA production configuration (todos 09-21 "thinking
    # off 강제"): the meta-reasoning must land in the VISIBLE content so the
    # leak is detectable, and the budget is not burned on reasoning.
    body = {"model": model, "messages": messages, "temperature": 0.1,
            "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    res = http(base, "/v1/chat/completions", body)
    msg = res["choices"][0]["message"]
    return {"text": msg.get("content") or "",
            "reason": msg.get("reasoning_content") or "",
            "wall": time.time() - t0}


def meta_hits(text):
    low = text.lower()
    return [p for p in META_PATTERNS if re.search(p, low)]


def build_messages(frame):
    instr = _instr(3, frame)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "[Magic Chunk 1]\n" + CHUNK1},
        {"role": "user", "content": "[Magic Chunk 2]\n" + CHUNK2},
        {"role": "user", "content": "[Magic Chunk 3]\n" + CHUNK3},
        {"role": "user", "content": RESUME + instr},
    ]


def run_case(base, model, frame, verbose=True):
    r = chat(base, model, build_messages(frame))
    if verbose:
        print(f"\n=== {frame} instructions ===")
        print(f"wall={r['wall']:.1f}s")
        print("--- visible content (head) ---")
        print(r["text"][:900])
    leaks = meta_hits(r["text"])
    task = any(m in r["text"].lower() for m in TASK_MARKERS)
    return {"frame": frame, "leaks": leaks, "task": task,
            "clean": not leaks, "text": r["text"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8086")
    ap.add_argument("--model", default=None,
                    help="model name (default: first /v1/models id)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    model = args.model
    if not model:
        try:
            model = http(args.server, "/v1/models")["data"][0]["id"]
        except Exception:
            model = "local"
    print(f"server={args.server} model={model}")

    old = run_case(args.server, model, "old", not args.quiet)
    new = run_case(args.server, model, "new", not args.quiet)

    print("\n=== SUMMARY ===")
    print(f"old (QA frame)    : leaks={old['leaks']} task_marker={old['task']}")
    print(f"new (agent frame) : leaks={new['leaks']} task_marker={new['task']}")
    ok = new["clean"]
    if ok:
        print("PASS: new instruction produced no scaffold meta-reasoning")
        if old["leaks"]:
            print("(old instruction reproduced the leak - clean A/B proof)")
    else:
        print(f"FAIL: new instruction still leaked: {new['leaks']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

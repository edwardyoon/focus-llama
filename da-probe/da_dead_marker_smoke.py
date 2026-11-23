#!/usr/bin/env python3
"""Smoke test for the DA scanner's post-compaction dead-marker defense (P1).

A native compaction replaces the conversation with a summary (one user
message) that may have copied [[da:N]] marker text from pre-compaction DA
blocks verbatim. The scanner (da_scan_prompt, tail anchoring) must then:
  - accept the LIVE block (appended by the client hook to the CURRENT user
    prompt = the last user message), even when dead chunk markers float in
    the summary (dead_partial), and
  - FAIL OPEN when only a dead block exists (a summary copy, no live block
    this turn) instead of accepting it (dead_complete_only - the hijack
    that pins attention to summary fragments).

The prompts are sent as chat completions, so the server renders them with
the model's own chat template and the "
</think>

"
boundaries the scanner anchors on are the real ones. Acceptance is read
from the server log: a "da_scan: N chunk(s) numbered ..." line means the
layout was accepted; no such line means fail-open. The verdict is
model-independent (no tag emission required).

Usage:
  python3 da_dead_marker_smoke.py [BASE_URL] [--log /path/to/server.log]

  BASE_URL  default: $DA_BASE or http://127.0.0.1:8081
  --log     server log file (server stdout redirected to a file) - required
            for a definite verdict; without it the response timings are
            checked instead (weaker: needs the model to emit a tag)

Server launch (local, A path):
  ./build/bin/llama-server -m <model>.gguf --port 8081 --parallel 1 \
      --da-prompt-scan > /tmp/da_dead_marker.log 2>&1 &

Runs:
  baseline            live block only (regression guard)
  dead_partial        summary carries dead chunk markers (no footer) + live block
  dead_complete_only  summary carries a complete dead block, no live block

Expected:
  baseline            accepted (da_scan: numbered 5..6)
  dead_partial        accepted (da_scan: numbered 5..6)
  dead_complete_only  fail-open (no da_scan: line)
"""
import json
import os
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8081")
LOG = None

SYSTEM = ("You are a precise reading engine. The memory entries contain "
          "codewords in the form WORD-NUMBER.")

LIVE_ENTRIES = {
    5: ("The emergency shutdown codeword for the facility is ZEBRA-42. "
        "It is printed in red letters on the wall of the control room."),
    6: ("The ventilation restart code for the control room is TANGO-7. "
        "The fans return to nominal speed within ninety seconds after the "
        "code is accepted."),
}

DEAD_ENTRIES = {
    3: ("The main power grid reset phrase used by operators is ECHO-9. "
        "It is stored in the backup terminal and must be confirmed by a "
        "second operator before use."),
    4: ("The fire suppression override code is KILO-31. A full discharge "
        "locks the valves for four hours until a manual reset."),
}

QUESTION = ("Question: What is the emergency shutdown codeword for the "
            "facility? Answer with the code only.")


def build_block(start, order, entries):
    """A DA marker block in the exact shape the FocusMemory hook emits
    (buildDaBlock in FocusMemory/index.js)."""
    n = len(order)
    parts = []
    for num in order:
        parts.append("[[da:%d]]%s" % (num, entries[num]))
    filler = (
        "\n\nInstructions (Declarative Attention): "
        "The memory entries above are numbered magic chunks (%d-%d). "
        "First identify the chunk that contains the answer to the question, "
        "and output the tag <focus magic_chunks=\"N\"> on its own line, "
        "where N is the chunk number (%d-%d). Then answer the question."
        % (start, start + n - 1, start, start + n - 1))
    parts.append("[[da:filler]]" + filler)
    parts.append("[[da:layout:%d]]" % n)
    return "\n".join(parts)


SUMMARY_PARTIAL = (
    "This is a summary of the earlier conversation. The user was reviewing "
    "the facility's control systems. Memory entries discussed earlier: "
    "[[da:3]]%s [[da:4]]%s The user then moved on to a new question."
    % (DEAD_ENTRIES[3], DEAD_ENTRIES[4]))

SUMMARY_COMPLETE = (
    "This is a summary of the earlier conversation. The user was reviewing "
    "the facility's control systems. The memory context from that turn:\n"
    + build_block(3, [3, 4], DEAD_ENTRIES) + "\n"
    "The user then moved on to a new question.")


def chat(messages, max_tokens=48):
    body = {"model": "local", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def log_new_lines():
    """New server-log lines since the previous call (or start of file)."""
    global LOG
    if not LOG or not os.path.exists(LOG):
        return None
    prev = getattr(log_new_lines, "offset", 0)
    with open(LOG, "r", errors="replace") as f:
        f.seek(prev)
        data = f.read()
        log_new_lines.offset = f.tell()
    return data


def main():
    global BASE, LOG
    args = sys.argv[1:]
    if "--log" in args:
        i = args.index("--log")
        LOG = args[i + 1]
        del args[i:i + 2]
    if args:
        BASE = args[0]
    print("base   : %s" % BASE)
    print("log    : %s" % (LOG or "(none - falling back to response timings)"))
    print("-" * 72)

    runs = [
        ("baseline", "accept", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION + "\n\n" + build_block(5, [5, 6], LIVE_ENTRIES)},
        ]),
        ("dead_partial", "accept", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": SUMMARY_PARTIAL},
            {"role": "user", "content": QUESTION + "\n\n" + build_block(5, [5, 6], LIVE_ENTRIES)},
        ]),
        ("dead_complete_only", "fail_open", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": SUMMARY_COMPLETE},
            {"role": "user", "content": QUESTION},
        ]),
    ]

    all_ok = True
    for name, expect, messages in runs:
        if LOG:
            log_new_lines()  # reset the offset before this request
        try:
            res = chat(messages)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print("%-20s: HTTP ERROR %s" % (name, e))
            all_ok = False
            continue
        answer = res["choices"][0]["message"]["content"].strip()
        da_timings = {k: v for k, v in (res.get("timings") or {}).items()
                      if k.startswith("da_")}

        if LOG:
            seg = log_new_lines() or ""
            scan_lines = [l.strip() for l in seg.splitlines() if "da_scan:" in l]
            accepted = any("chunk(s) numbered" in l for l in scan_lines)
        else:
            accepted = bool(da_timings)  # weak: needs a model tag
            scan_lines = []

        ok = (accepted == (expect == "accept"))
        all_ok = all_ok and ok
        print("%-20s: %s  (expected %s)"
              % (name, "PASS" if ok else "FAIL", expect))
        print("          answer      : %r" % answer[:80])
        print("          da_timings  : %s" % (da_timings or "{}"))
        for l in scan_lines:
            print("          journal     : %s" % l[:110])
        if not ok and LOG:
            print("          NOTE: check the journal for 'da_scan:' / 'failing open' lines")
    print("-" * 72)
    print("OVERALL          : %s" % ("PASS" if all_ok else "FAIL"))
    print("journal cross-check (server log):")
    print("  grep 'da_scan:'              -> baseline + dead_partial: 'numbered 5..6'")
    print("  grep 'failing open to vanilla' -> dead_complete_only only (if SRV_DBG visible)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

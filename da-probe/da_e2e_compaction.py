#!/usr/bin/env python3
"""E2E: real DA block + auto-compact on the 123 GPU server (A path).

This is the faithful reproduction of the PRODUCTION failure mode: qwen-code's
autoCompact summarizes+erases a session, and the DA session must keep working.
Production (123:8080) runs the A path (--da-prompt-scan): the FocusMemory hook
appends [[da:N]] blocks to the CURRENT user message, and the server's scanner
(da_scan_prompt) tail-anchors to the last user message, validates the live
block, and sets up declarative attention.

The driver reproduces that end to end on the production model:

  Phase A  pre-compact   - 3 turns, each user message carries the REAL
                           FocusMemory hook block (buildDaBlock copied
                           verbatim from FocusMemory/index.js:1894-1947).
                           Numbers are monotonic per session (1-5, 6-10,
                           11-15) exactly as the hook's counter does. Each
                           answer must be CORRECT.
  Phase B  compact       - the conversation is replaced by:
                             [system]
                             [user] summary (verbatim copy of the turn-1
                                     DA block = DEAD markers 1-5 + dead
                                     filler + dead footer, exactly what a
                                     summarizer would reproduce)
                             [user] NEW question + a NEW live DA block
                                     (numbers continue: 16-20)
                           This is the money line: the scanner must
                           tail-anchor to the last user message, IGNORE the
                           dead block in the summary, validate the live
                           block (16-20), and the answer must be CORRECT.
                           (If it pinned attention to the dead summary
                           block, that's the original hijack bug.)
  Phase C  post-compact  - one more turn in the compacted session (live
                           block 21-25). Answer must stay CORRECT - proves
                           the DA path persists across compaction.

Expected journal (read from the server log, full-file order matching -
buffering independent, same design as da_chunking_smoke.py):
  da_scan: 5 chunk(s) numbered  1..5  + filler, ...   (phase A turn 1)
  da_scan: 5 chunk(s) numbered  6..10 + filler, ...   (phase A turn 2)
  da_scan: 5 chunk(s) numbered 11..15 + filler, ...   (phase A turn 3)
  da_scan: 5 chunk(s) numbered 16..20 + filler, ...   (phase B - money line)
  da_scan: 5 chunk(s) numbered 21..25 + filler, ...   (phase C)
  and NO "failing open" line anywhere.

Usage:
  python3 da_e2e_compaction.py [BASE_URL] --log /path/to/server.log

Server (123, the GPU box - the production model, A path):
  stdbuf -o0 -e0 ./build/bin/llama-server \
      -m /home/edwardyoon/my_model/qwen3.8/Qwen3.8-27B-AD-Q6_K.gguf \
      -ctk q4_0 -ctv q4_0 --port 8086 \
      --da-prompt-scan --kv-unified --parallel 1 -c 65536 -v \
      > /tmp/da_e2e.log 2>&1 &
"""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8086")
LOG = None

SYSTEM = ("You are a precise retrieval assistant. Answer each question using "
          "only the conversation context. When asked for a codeword, reply "
          "with the codeword alone on one line, nothing else.")

# ── The memory entries (what a FocusMemory search would return) ────────────
# 5 entries; each carries one codeword so every turn targets a different
# chunk. Text shape = daChunkText output (summary_text + "\n" + detail).
ENTRIES = [
    {"payload": {"summary_text": "Alpha project codeword",
                 "detail": "The alpha project codeword is TANGO-7. It is "
                           "stored in the vault under the red drawer and "
                           "rotates quarterly."}},
    {"payload": {"summary_text": "Beta project codeword",
                 "detail": "The beta project codeword is ZEBRA-42. It is "
                           "stored in the vault under the blue drawer and "
                           "rotates quarterly."}},
    {"payload": {"summary_text": "Gamma project codeword",
                 "detail": "The gamma project codeword is KILO-99. It is "
                           "stored in the vault under the green drawer and "
                           "rotates quarterly."}},
    {"payload": {"summary_text": "Delta project codeword",
                 "detail": "The delta project codeword is NOVEMBER-15. It "
                           "is stored in the vault under the yellow drawer "
                           "and rotates quarterly."}},
    {"payload": {"summary_text": "Epsilon project codeword",
                 "detail": "The epsilon project codeword is SIERRA-88. It "
                           "is stored in the vault under the purple drawer "
                           "and rotates quarterly."}},
]

# ── VERBATIM copy of the FocusMemory hook block builder ────────────────────
# Source: FocusMemory/index.js buildDaBlock + daSafeSlice + daChunkText
# (lines 1894-1947, committed 53f12ad). Kept in sync by hand; if the hook
# changes, re-copy. The block structure is what the server scans.
DA_MAX_ENTRIES = 5
DA_ENTRY_CHARS = 300


def da_safe_slice(s, n):
    if len(s) <= n:
        return s
    return s[:n] + "…"


def da_chunk_text(r):
    p = r.get("payload") or {}
    text = ""
    if p.get("summary_text"):
        text += p["summary_text"] + "\n"
    body = p.get("detail") or p.get("content") or ""
    if body:
        text += str(body)
    if not text.strip():
        return ""
    return da_safe_slice(text.strip(), DA_ENTRY_CHARS)


def build_da_block(entries, start_num):
    """FocusMemory/index.js buildDaBlock - verbatim port."""
    texts = [da_chunk_text(r) for r in entries]
    if any("[[da:" in t or "<da:" in t for t in texts):
        return None
    n = len(texts)
    block = ""
    for i, t in enumerate(texts):
        block += "\n[[da:%d]]%s" % (start_num + i, t)
    instruction = (
        "\n\nInstructions (Declarative Attention): "
        "The memory entries above are numbered magic chunks (%d-%d). "
        "First identify the chunk that contains the answer to the question, "
        "and output the tag <focus magic_chunks=\"N\"> on its own line, "
        "where N is the chunk number (%d-%d). Then answer the question."
        % (start_num, start_num + n - 1, start_num, start_num + n - 1))
    block += "\n[[da:filler]]%s\n[[da:layout:%d]]" % (instruction, n)
    return block


# ── HTTP ───────────────────────────────────────────────────────────────────
def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(messages, max_tokens=256):
    body = {"model": "local", "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    res = post("/v1/chat/completions", body)
    wall = time.time() - t0
    msg = res["choices"][0]["message"]
    return {"text": msg.get("content") or "",
            "reason": msg.get("reasoning_content") or "",
            "n_gen": (res.get("timings") or {}).get("predicted_n"),
            "n_prompt": (res.get("timings") or {}).get("prompt_n"),
            "wall": wall}


# ── Journal (full-file order matching, buffering independent) ─────────────
# A-path success line (server-context.cpp da_scan_prompt):
#   da_scan: N chunk(s) numbered X..Y[ + filler], M prompt token(s) - A|B path
# Fail-open lines are da_scan lines containing "failing open".
DA_SCAN_RE = re.compile(
    r"da_scan: (\d+) chunk\(s\) numbered (\d+)\.\.(\d+)(?: \+ filler)?, (\d+) prompt token\(s\)")


def read_lines():
    with open(LOG, "r", errors="replace") as f:
        return f.read().splitlines()


def da_scan_lines():
    out = []
    for line in read_lines():
        m = DA_SCAN_RE.search(line)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        int(m.group(4)), line.strip()))
    return out


def fail_open_lines():
    out = []
    for line in read_lines():
        if "failing open" in line or "fail-open" in line:
            out.append(line.strip())
    return out


# ── Scenario ───────────────────────────────────────────────────────────────
def question(name):
    return ("Question: What is the %s project codeword? "
            "Reply with the codeword only, nothing else." % name)


def run():
    global LOG
    args = sys.argv[1:]
    if "--log" in args:
        i = args.index("--log")
        LOG = args[i + 1]
        del args[i:i + 2]
    if args:
        BASE = args[0]
    if not LOG:
        print("ERROR: --log is required")
        sys.exit(2)
    print("base   : %s" % BASE)
    print("log    : %s" % LOG)
    print("-" * 72)

    base_success = len(da_scan_lines())
    base_fo = len(fail_open_lines())

    # ── Phase A: 3 turns, DA block appended each turn (hook behavior) ────
    # The hook appends the block to the CURRENT user message. Chunk numbers
    # are monotonic per session: 1-5, 6-10, 11-15... Each request's scanner
    # tail-anchors to the LAST user message, so it sees only the current
    # turn's live block; earlier turns' blocks sit in history (inert).
    msgs = [{"role": "system", "content": SYSTEM}]
    counter = 1  # the hook's daSessionCounters value
    results = []

    phase_a = [
        ("alpha", "TANGO-7"),
        ("beta", "ZEBRA-42"),
        ("gamma", "KILO-99"),
    ]
    print("Phase A: pre-compact (3 turns with live DA blocks)")
    for name, codeword in phase_a:
        user_msg = question(name) + "\n\n" + build_da_block(ENTRIES, counter)
        counter += len(ENTRIES)
        msgs.append({"role": "user", "content": user_msg})
        r = chat(msgs)
        ok = codeword in r["text"]
        results.append(("A-%s" % name, codeword, ok, r))
        print("  turn %-5s expect=%-10s %s  n_prompt=%s n_gen=%s"
              % (name, codeword, "CORRECT" if ok else "WRONG",
                 r["n_prompt"], r["n_gen"]))
        if not ok:
            print("          content: %r" % r["text"][:200])
            print("          reason : %r" % r["reason"][-300:])
        # feed back only the visible answer (as a real client would)
        msgs.append({"role": "assistant", "content": r["text"]})

    # ── Phase B: compact - history replaced by a summary ─────────────────
    # The summarizer copied the turn-1 DA block verbatim (dead markers 1-5
    # + dead filler + dead footer). A new turn arrives with a NEW live block
    # (numbers continue: 16-20). The scanner must ignore the dead block and
    # validate the live one.
    dead_block = build_da_block(ENTRIES, 1)  # the turn-1 block, now dead
    summary = ("This is a summary of the earlier conversation. The user was "
               "reviewing project codewords from the memory context below. "
               "%s The user then asked for the alpha, beta and gamma "
               "codewords and received them." % dead_block)
    compacted = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": summary},
    ]
    print("Phase B: post-compact (summary carries dead block 1-5, "
          "live block 16-20)")
    live_b = build_da_block(ENTRIES, 16)
    user_b = question("delta") + "\n\n" + live_b
    compacted.append({"role": "user", "content": user_b})
    r = chat(compacted)
    ok_b = "NOVEMBER-15" in r["text"]
    results.append(("B-delta", "NOVEMBER-15", ok_b, r))
    print("  turn B-delta expect=NOVEMBER-15  %s  n_prompt=%s n_gen=%s"
          % ("CORRECT" if ok_b else "WRONG", r["n_prompt"], r["n_gen"]))
    if not ok_b:
        print("          content: %r" % r["text"][:200])
        print("          reason : %r" % r["reason"][-300:])
    compacted.append({"role": "assistant", "content": r["text"]})

    # ── Phase C: another turn in the compacted session ───────────────────
    print("Phase C: second post-compact turn")
    live_c = build_da_block(ENTRIES, 21)
    user_c = question("epsilon") + "\n\n" + live_c
    compacted.append({"role": "user", "content": user_c})
    r = chat(compacted)
    ok_c = "SIERRA-88" in r["text"]
    results.append(("C-epsilon", "SIERRA-88", ok_c, r))
    print("  turn C-epsilon expect=SIERRA-88  %s  n_prompt=%s n_gen=%s"
          % ("CORRECT" if ok_c else "WRONG", r["n_prompt"], r["n_gen"]))
    if not ok_c:
        print("          content: %r" % r["text"][:200])
        print("          reason : %r" % r["reason"][-300:])

    # ── Journal verification ──────────────────────────────────────────────
    # 5 requests were sent; the journal must show 5 da_scan success lines,
    # in order, with the expected chunk numbers. Phase B's line (16..20) is
    # the money line: the compacted prompt was scanned (not failed open) and
    # the live block won over the dead summary block.
    deadline = time.time() + 60.0
    lines = da_scan_lines()[base_success:]
    while len(lines) < 5 and time.time() < deadline:
        time.sleep(1.0)
        lines = da_scan_lines()[base_success:]
    fo = fail_open_lines()[base_fo:]

    expected = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)]
    print("-" * 72)
    print("journal: %d/5 da_scan line(s), %d fail-open line(s)"
          % (len(lines), len(fo)))
    for i, (n_c, s, e, n_t, raw) in enumerate(lines[:10]):
        if i < len(expected):
            exp = expected[i]
            match = ("ok" if (s == exp[0] and e == exp[1])
                     else "MISMATCH (expected %d..%d)" % exp)
        else:
            match = "UNEXPECTED extra line"
        print("  [%d] %d chunk(s) numbered %d..%d, %d tok  %s"
              % (i + 1, n_c, s, e, n_t, match))
    for l in fo[:10]:
        print("  FAIL-OPEN: %s" % l[:110])

    num_ok = (len(lines) == 5 and
              all(lines[i][1] == expected[i][0] and lines[i][2] == expected[i][1]
                  for i in range(5)))
    all_ok = all(ok for _, _, ok, _ in results) and num_ok and not fo
    print("-" * 72)
    print("OVERALL : %s" % ("PASS" if all_ok else "FAIL"))
    if not all_ok:
        bad = [n for n, _, ok, _ in results if not ok]
        if bad:
            print("  wrong answers: %s" % "+".join(bad))
        if not num_ok:
            print("  da_scan lines missing or chunk numbers mismatched "
                  "(check server has --da-prompt-scan and the log path)")
        if fo:
            print("  fail-open present - a DA block was REJECTED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    run()

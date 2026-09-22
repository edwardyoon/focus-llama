#!/usr/bin/env python3
"""Smoke test for the da_auto_chunk paper alignment (P2).

Verifies the chunking behaviors via the server journal line
"da_auto: N chunk(s) from M source message(s), T token(s) - B path":

  packing     - many tiny messages are packed across message boundaries
                into target-sized chunks (300 x ~15-token messages must NOT
                become 300 one-message chunks; expect >= 2 chunks, avg
                <= hard cap)
  hard_cap    - a single boundary-free message over the hard cap
                (2048*5/4 = 2560) is force-cut into pieces <= the cap
                (a ~12K-token whitespace-free blob must become >= 2 chunks;
                the pre-fix code kept it as one uncapped chunk)
  prose       - a single ~25K-token prose message (140 paras) is split at
                sentence boundaries and packed into target-sized chunks
                (regression guard for the existing levels 0-2)
  perf_50k    - a ~50K-token single message: report the chunking wall time
                (request start -> da_auto journal line) as the performance
                baseline (plan section 5)

Journal reading design (robust to stdout buffering):
  The server log may be block-buffered (through `tee` or the stdio layer),
  so a da_auto line can reach the file seconds after the HTTP response.
  Per-request log reads therefore CANNOT be trusted (verified on 123:
  all 4 journal lines were in the log, yet per-request reads saw none).
  Instead: run all cases first, then wait until the file contains one
  da_auto line per case and match them in order. The cases run
  sequentially on one server and the log is append-only, so the i-th new
  da_auto line is the i-th case. The chunking wall time pairs each line
  with the nearest preceding 'Using specialized template' line (the
  completion handler logs it right before da_auto_chunk runs).

Usage:
  python3 da_chunking_smoke.py [BASE_URL] --log /path/to/server.log
      [--target 2048]

  BASE_URL  default: http://127.0.0.1:8087
  --log     server log file (required: the verdict is journal-based).
            Must be the CURRENT server's stdout target - the server should
            have been (re)started with `> /path/to/server.log 2>&1`.
  --target  the server's --da-chunk-tokens value (hard cap = 5/4 x target)

Server launch (local, B path, default chunk target 2048 / cap 2560):
  ./build/bin/llama-server -m <model>.gguf --port 8087 --parallel 1 \
      --kv-unified --da-auto -c 65536 > /tmp/da_chunking.log 2>&1 &
"""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8087")
LOG = None

SYSTEM = ("You are a precise retrieval assistant. Answer using only the "
          "conversation context.")

DA_AUTO_RE = re.compile(
    r"da_auto: (\d+) chunk\(s\) from (\d+) source message\(s\), (\d+) token\(s\)")

REQ_START_MARKER = "Using specialized template"


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(messages, max_tokens=16):
    body = {"model": "local", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    t0 = time.time()
    res = post("/v1/chat/completions", body)
    return res, time.time() - t0


def read_lines():
    with open(LOG, "r", errors="replace") as f:
        return f.read().splitlines()


def ts_seconds(line):
    """llama.cpp log timestamp 'M.SS.mmm.uuu' -> seconds (float)."""
    m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)\.(\d+)", line)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 \
        + int(m.group(4)) / 1e6


def chunking_times(lines, base):
    """The da_auto lines after the first `base`, in log order. Each is
    paired with its chunking wall time (nearest preceding request-start
    marker). Returns [(n_chunks, n_msgs, n_tokens, dt_or_None), ...]."""
    out = []
    seen = 0
    for i, line in enumerate(lines):
        m = DA_AUTO_RE.search(line)
        if not m:
            continue
        seen += 1
        if seen <= base:
            continue
        t_auto = ts_seconds(line)
        dt = None
        if t_auto is not None:
            for j in range(i - 1, max(i - 20, -1), -1):
                if REQ_START_MARKER in lines[j]:
                    t_start = ts_seconds(lines[j])
                    if t_start is not None:
                        dt = t_auto - t_start
                    break
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), dt))
    return out


def make_blob(n_chars):
    """A whitespace/punctuation-free blob (no cut boundary at any level)."""
    alpha = "abcdefghijklmnopqrstuvwxyz0123456789"
    out = []
    i = 0
    while sum(len(x) for x in out) < n_chars:
        out.append(alpha[(i * 7) % 26] + alpha[(i * 13) % 26]
                   + alpha[(i * 31) % 36][:10] + str(i % 97))
        i += 1
    return "".join(out)[:n_chars]


def main():
    global BASE, LOG
    args = sys.argv[1:]
    if "--log" in args:
        i = args.index("--log")
        LOG = args[i + 1]
        del args[i:i + 2]
    if args:
        BASE = args[0]
    if not LOG:
        print("ERROR: --log is required (the verdict is journal-based)")
        sys.exit(2)
    print("base   : %s" % BASE)
    print("log    : %s" % LOG)
    print("-" * 72)

    # case 1: packing - 300 tiny user messages (~15 tokens each)
    tiny = ["Note %03d: the valve pressure reading is %.1f bar nominal."
            % (i, 10 + (i % 50) / 4.0) for i in range(300)]
    msgs_packing = [{"role": "system", "content": SYSTEM}]
    for t in tiny:
        msgs_packing.append({"role": "user", "content": t})
    msgs_packing.append({"role": "user", "content":
                         "Question: How many notes were listed?"})

    # case 2: hard cap - one ~12K-token boundary-free blob
    blob = make_blob(15000)
    msgs_blob = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Archive payload:\n" + blob},
        {"role": "user", "content": "Question: How long is the payload?"},
    ]

    # case 3: prose - one ~25K-token sentence-rich message (140 paras)
    sentence = ("The relay unit in sector %02d reports a nominal temperature "
                "of 41 degrees and a current draw of 12 amps. ")
    prose = "\n\n".join(" ".join(sentence % (i * 10 + j)
                                 for j in range(6))
                        for i in range(140))
    msgs_prose = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prose},
        {"role": "user", "content": "Question: Summarize in one word."},
    ]

    # case 4: perf baseline - one ~50K-token prose message (280 paras x
    # ~184 tok; 2100 paras was ~385K tok and blew the 65536 context)
    big = "\n\n".join(" ".join(sentence % (i * 10 + j) for j in range(6))
                      for i in range(280))
    msgs_50k = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": big},
        {"role": "user", "content": "Question: Summarize in one word."},
    ]

    # hard cap = 5/4 of the chunk target (default target 2048 -> cap 2560).
    # The server flag is --da-chunk-tokens; pass --target to match a custom
    # server (e.g. a smoke server running target 4096 -> cap 5120).
    target = 2048
    if "--target" in sys.argv[1:]:
        i = sys.argv.index("--target")
        target = int(sys.argv[i + 1])
    cap = int(target * 5 / 4)

    cases = [
        ("packing",  msgs_packing,
         lambda n_c, n_m, n_t, dt: n_c >= 2 and n_m == 300
         and n_t // n_c <= cap,
         ">=2 packed chunks from 300 messages, avg <= cap %d (pre-fix: 300)"
         % cap),
        ("hard_cap", msgs_blob,
         lambda n_c, n_m, n_t, dt: n_c >= 2 and n_m == 1
         and n_t // n_c <= cap,
         ">=2 chunks from 1 boundary-free message, avg <= cap %d"
         " (pre-fix: 1 uncapped chunk)" % cap),
        ("prose",    msgs_prose,
         lambda n_c, n_m, n_t, dt: n_c >= 2 and n_m == 1
         and n_t // n_c <= cap,
         ">=2 sentence-packed chunks from 1 ~25K-token prose message, "
         "avg <= cap %d" % cap),
        ("perf_50k", msgs_50k,
         lambda n_c, n_m, n_t, dt: n_c >= 2 and n_t // n_c <= cap,
         ">=2 chunks, avg <= cap %d; report chunking wall time" % cap),
    ]

    # Baseline: da_auto lines already in the log (0 for a fresh server log -
    # the server truncates it at startup).
    base = sum(1 for l in read_lines() if DA_AUTO_RE.search(l))
    if base:
        print("note   : %d pre-existing da_auto line(s) in the log - "
              "restart the server (fresh log) for a clean run" % base)
    print("(running %d cases...)" % len(cases))

    results = []
    for name, messages, check, expect in cases:
        try:
            res, wall = chat(messages)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print("%-10s: HTTP ERROR %s" % (name, e))
            results.append((wall, None))
            continue
        n_prompt = (res.get("timings") or {}).get("prompt_n")
        results.append((wall, n_prompt))

    # Wait until every case has a journal line (buffered stdout can delay
    # them well past the HTTP response), then match in order.
    deadline = time.time() + 60.0
    times = chunking_times(read_lines(), base)
    while len(times) < len(cases) and time.time() < deadline:
        time.sleep(1.0)
        times = chunking_times(read_lines(), base)

    all_ok = True
    for idx, (name, messages, check, expect) in enumerate(cases):
        wall, n_prompt = results[idx]
        if idx >= len(times):
            print("%-10s: FAIL  (no da_auto journal line after 60s wait)" % name)
            print("          NOTE: is the running server writing to %s?" % LOG)
            print("          check: readlink /proc/$(pgrep -f llama-server)/fd/1")
            all_ok = False
            continue
        n_c, n_m, n_t, dt = times[idx]
        ok = check(n_c, n_m, n_t, dt)
        all_ok = all_ok and ok
        dt_s = ("chunking=%.3fs" % dt) if dt is not None else "chunking=n/a"
        print("%-10s: %s  (expected: %s)" % (name, "PASS" if ok else "FAIL", expect))
        print("          chunks=%d msgs=%d tokens=%d  prompt_n=%s  wall=%.2fs  %s"
              % (n_c, n_m, n_t, n_prompt, wall, dt_s))
    print("-" * 72)
    print("OVERALL : %s" % ("PASS" if all_ok else "FAIL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

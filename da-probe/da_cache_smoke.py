#!/usr/bin/env python3
"""P2 cache policy smoke test — release() cache policy for DA (static da_rm) requests.

Verifies the P2 acceptance criteria against a running llama-server with DA
support (any binary that includes the release() DA cache policy chain):
  (a) two consecutive untagged requests -> the second reuses the prefix
      cache (journal: 'cache reuse: n_past = N' with N near the prompt length)
  (b) the request after a tagged (da_rm) request is not broken: the A path
      clears the holed KV on release, so the follow-up is a full prefill and
      must still answer the question correctly
  (c) cache hit rate identical to vanilla: the n_past of the post-DA untagged
      pair (R5) equals the vanilla pair's n_past (R2)

Runs 5 completions (temperature 0, stop=["\n"], exact-match verdict on the
ZEBRA-42 codeword document — same prompt as da_server_smoke.py):
  R1 baseline        - no da_rm          -> exactly ZEBRA-42
  R2 baseline again  - no da_rm          -> exactly ZEBRA-42, journal 'cache reuse: n_past'
  R3 strict removal  - da_rm=fact, at Q  -> clean (removal ran: A path)
  R4 baseline again  - no da_rm          -> exactly ZEBRA-42 (not broken by R3's holes)
  R5 baseline again  - no da_rm          -> exactly ZEBRA-42, journal 'cache reuse: n_past' == R2

Usage:
  python3 da_cache_smoke.py [BASE_URL] --log /path/to/server.log

  BASE_URL   default: $DA_BASE or http://127.0.0.1:8080
  --log      server log file to grep for the journal lines (required for the
             cache-reuse verdicts; the server must run with -v)
"""
import json
import os
import re
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")
LOG = None

PROMPT = """The following is a technical document.

[Section 1: Overview]
The system processes incoming requests through a queue. Each request is validated, then dispatched to a worker pool. Workers maintain a local cache of recent results to reduce latency. The cache expires entries after thirty seconds.

[Section 2: The codeword]
The emergency shutdown codeword for the facility is ZEBRA-42. Operators must memorize it. It is printed on the wall of the control room in red letters.

[Section 3: Scheduling]
Tasks are scheduled using a round-robin policy. The scheduler considers task priority and estimated duration. Long-running tasks are preempted after five minutes.

Question: What is the emergency shutdown codeword for the facility? Reply with the codeword only, nothing else.
Answer:
"""

FACT_START = "The emergency shutdown codeword"
FACT_END = "red letters."
QUESTION_START = "Question: What is the"
THINK_END = "\n</think>\n"


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


def complete(prompt, extra):
    body = {"prompt": prompt, "max_tokens": 24, "temperature": 0,
            "stop": ["\n"]}
    body.update(extra)
    return post("/v1/completions", body)


def strip_thinking(text):
    i = text.rfind(THINK_END)
    return text[i + len(THINK_END):] if i != -1 else text


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
        raise SystemExit("range mismatch: %r" % got)
    return lo, hi


class Journal:
    """Tail of the server log since the last mark; greps for DA/cache lines."""

    def __init__(self, path):
        self.path = path
        self.off = 0

    def since(self, from_start=False):
        if self.path is None:
            return []
        with open(self.path, "r", errors="replace") as f:
            f.seek(0 if from_start else self.off)
            data = f.read()
            self.off = f.tell()
        return [l.strip() for l in data.splitlines()
                if any(k in l for k in
                       ("cache reuse", "clearing slot", "da_rm", "stop processing"))]


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

    ids = tokenize(PROMPT)
    n = len(ids)
    full = detokenize(ids)
    fact_lo, fact_hi = range_of(ids, FACT_START, FACT_END)
    q_pos = len(tokenize(full[:full.index(QUESTION_START)]))
    print("prompt tokens            : %d" % n)
    print("fact range               : [%d, %d)" % (fact_lo, fact_hi))
    print("question start (da_rm_at): %d" % q_pos)
    print("-" * 72)

    j = Journal(LOG)
    j.since()  # drop startup lines
    results = {}
    reuse = {}  # key -> n_past from 'cache reuse: n_past = N'

    def run(key, extra):
        res = complete(PROMPT, extra)
        text = strip_thinking(res["choices"][0]["text"]).strip()
        results[key] = text
        print("%-16s: %r" % (key, text))
        lines = j.since()
        m = [l for l in lines if "cache reuse: n_past" in l]
        if m:
            v = re.search(r"n_past = (\d+)", m[-1])
            reuse[key] = int(v.group(1)) if v else None
            print("                   cache reuse: n_past = %s" % reuse[key])
        else:
            print("                   cache reuse: (none — full prefill)")
        for l in lines:
            if "clearing slot" in l or "da_rm" in l:
                print("                   | %s" % l[:150])
        return res

    run("R1 baseline", {})
    run("R2 baseline", {})
    run("R3 strict", {"da_rm": [[fact_lo, fact_hi]], "da_rm_at": q_pos})
    run("R4 baseline", {})
    run("R5 baseline", {})
    print("-" * 72)

    ANSWER = "ZEBRA-42"
    checks = []

    def check(name, ok, detail):
        checks.append(bool(ok))
        print("%-28s: %s (%s)" % (name, "PASS" if ok else "FAIL", detail))

    # (a) untagged pair: second request reuses the prefix cache
    r2 = reuse.get("R2 baseline")
    check("(a) R2 cache reuse", r2 is not None and r2 >= n - 32,
          "n_past=%s vs prompt %d" % (r2, n))

    # R3: the removal actually ran — with the fact range removed the model
    # cannot answer the codeword
    check("R3 removal ran", results.get("R3 strict") != ANSWER and len(results.get("R3 strict", "")) > 0,
          "answer %r (must not be the codeword)" % results.get("R3 strict"))

    # (b) the request after the tagged one is not broken
    check("(b) R4 not broken", results.get("R4 baseline") == ANSWER,
          "answer %r" % results.get("R4 baseline"))

    # (c) post-DA untagged pair reuses like vanilla
    r5 = reuse.get("R5 baseline")
    check("(c) R5 reuse == R2", r5 is not None and r2 is not None and r5 == r2,
          "R5 n_past=%s vs R2 n_past=%s" % (r5, r2))

    # journal: R3's release cleared the slot (A path), R4 was a full prefill
    all_lines = j.since(from_start=True)
    cleared = [l for l in all_lines if "clearing slot after da_rm request" in l]
    check("journal R3 clear", len(cleared) >= 1, "%d 'clearing slot after da_rm' line(s)" % len(cleared))
    r4_no_reuse = "R4 baseline" not in reuse
    check("journal R4 full prefill", r4_no_reuse,
          "R4 %s" % ("has no cache-reuse line" if r4_no_reuse else "unexpectedly reused n_past=%s" % reuse.get("R4 baseline")))

    ok = all(checks)
    print("-" * 72)
    print("OVERALL          : %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

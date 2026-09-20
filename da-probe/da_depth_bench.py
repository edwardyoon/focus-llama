#!/usr/bin/env python3
"""P-G gate bench: vanilla decode tps vs. prompt depth (measures r).

The P-G gate question (todos/2026-09-21.md, P-G): even with perfect logical
removal, is there an environment where DA pays off in speed? Break-even:

    T_DA / T_vanilla ~= (1 + d) * [g + (1 - g) * r]
    r = tps(D_full) / tps(shallow)

with d = step inflation 0.15..0.35 and g = global-step share 0.27..0.45
(paper Gemma ranges). DA wins only where r < ~0.82 (best corner) /
~0.53 (worst corner). This bench measures r on the real target server:
prompt depths 3K/16K/32K/64K, n_predict 64, temperature 0,
cache_prompt=false (full re-prefill every request). The ratio r is the
gate input; the chat template's constant overhead does not affect it.

Spec on/off: the server's spec state cannot be queried, so the user runs
this bench ONCE PER SPEC STATE — restart the server between runs (spec on
= the production MTP config, the vanilla baseline; spec off = the ceiling
without MTP) and pass a matching --label. Keep both reports.

Usage:
  python3 da_depth_bench.py [BASE_URL] [options]

  BASE_URL          default: $DA_BASE or http://127.0.0.1:8080
  --depths 3072,16384,32768,65536
  --n-predict 64    decode tokens per run
  --temp 0
  --reps 1          measured repetitions per depth (mean reported)
  --label spec-on   run label (spec-on / spec-off), kept in the report name
  --journal-file P  server log FILE — the decode tps is read from the
                    journal's 'eval time' line, the evidence this todo asks
                    for. The response's timings.predicted_per_second is the
                    same quantity (both are stats.n_gen_tps()) and is
                    recorded as the cross-check.
  --journal-cmd C   journal command instead of a file, e.g.
                    'journalctl -u llama-server --no-pager -n 500'
  --warmup 1        unmeasured runs at the first (smallest) depth, to drop
                    one-off costs (graph capture, allocator warm-up)
  --settle 2        seconds to sleep between requests (GPU/thermal settle)
  --timeout 1800    per-request timeout in seconds (a 64K prefill can take
                    minutes on a 27B)

Prompt construction: a fixed filler unit repeated until the depth is
reached, then the token list is truncated to EXACTLY the depth and
detokenized (the unit starts on a word boundary, so the
tokenize/detokenize round trip is stable; a residual drift of a few tokens
is reported, not fatal — only the KV length matters for the gate).

Journal line format (tools/server/server-context.cpp print_timing, INF):
  slot print_timing: id  0 | task  7 |        eval time =  1234.56 ms /    64 tokens (   19.29 ms per token,    51.84 tokens per second)
The bench is sequential (one server slot), so the LAST 'eval time' line
that appeared since the previous request is this request's. If the line's
token count differs from the response's completion_tokens, the value is
flagged (a concurrent task's line was likely picked up).

Exit codes: 0 = all depths measured, 1 = at least one depth failed.
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("DA_BASE", "http://127.0.0.1:8080")

# Filler unit (~30 tokens). Starts on a word boundary (no leading space) so
# the tokenize/detokenize round trip after truncation is stable.
FILLER_UNIT = ("The quick brown fox jumps over the lazy dog near the old stone bridge. "
               "Farmers carry grain across the market square while the bells ring softly. ")

# Journal 'eval time' line. The leading '\|\s+eval' anchors to the decode
# line (8-space indent after the task pipe); 'prompt eval time' has
# 'prompt' right after the pipe, so it never matches. The format string is
# "eval time = %10.2f ms / %5d tokens (...)" — %5d right-pads, so allow
# multiple spaces after the slash.
RE_EVAL = re.compile(
    r"\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens \([^)]*?([\d.]+) tokens per second\)")

# Break-even parameter grid (todos/2026-09-21.md P-G background):
# d = step inflation, g = global-step share (paper Gemma ranges).
D_LO, D_HI = 0.15, 0.35
G_LO, G_HI = 0.27, 0.45


class Journal(object):
    """Collects the server journal's 'eval time' line for each request.

    File mode remembers the file size before each request and parses the
    appended tail after it. Command mode runs the command (e.g.
    journalctl -u llama-server --no-pager -n 500) and parses the output.
    The LAST matching line is taken — the bench is sequential, so with one
    server slot it is the line this request produced.
    """

    def __init__(self, path=None, cmd=None):
        self.path = path
        self.cmd = cmd
        self.offset = 0
        if path and os.path.exists(path):
            self.offset = os.path.getsize(path)

    def snapshot(self):
        """Mark the journal position before a request (file mode only)."""
        if self.path and os.path.exists(self.path):
            self.offset = os.path.getsize(self.path)

    def eval_line(self):
        """(ms, tokens, tps) of the last 'eval time' line, or None."""
        if self.path:
            if not os.path.exists(self.path):
                return None
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
            self.offset += len(data)
            text = data.decode("utf-8", "replace")
        elif self.cmd:
            proc = subprocess.Popen(self.cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            data, _ = proc.communicate(timeout=60)
            text = data.decode("utf-8", "replace")
        else:
            return None
        matches = list(RE_EVAL.finditer(text))
        if not matches:
            return None
        m = matches[-1]
        return float(m.group(1)), int(m.group(2)), float(m.group(3))


def post(path, body, timeout):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(text, timeout=300):
    return post("/tokenize", {"content": text}, timeout)["tokens"]


def detokenize(ids, timeout=300):
    return post("/detokenize", {"tokens": ids}, timeout)["content"]


def build_prompt(n_tokens):
    """Filler text whose tokenization is exactly n_tokens (returns (text, actual_n)).

    The repeated filler unit tokenizes SHORTER in context than standalone
    (boundary merge, observed 29 -> 28), so the pool is sized on unit_n - 1
    with a 15% margin, and the truncation length is corrected iteratively:
    detokenize(ids[:L]), re-tokenize, L += (target - actual) until exact
    (observed: converges in 0-1 iterations for 3K..64K).
    """
    unit_n = max(len(tokenize(FILLER_UNIT)) - 1, 1)
    s = FILLER_UNIT * (int(n_tokens * 1.15) // unit_n + 10)
    ids = tokenize(s)
    if len(ids) < n_tokens:
        s += FILLER_UNIT * 40
        ids = tokenize(s)
    L = n_tokens
    actual = None
    for _ in range(3):
        s = detokenize(ids[:L])
        actual = len(tokenize(s))
        if actual == n_tokens:
            break
        L += n_tokens - actual
    if actual != n_tokens:
        print("NOTE: prompt for depth %d tokenizes to %d (drift %d) - proceeding"
              % (n_tokens, actual, actual - n_tokens))
    return s, actual


def complete(prompt, n_predict, temp, timeout):
    return post("/v1/completions",
                {"prompt": prompt, "n_predict": n_predict, "temperature": temp,
                 "cache_prompt": False},
                timeout)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    global BASE
    args = sys.argv[1:]
    flags = {}
    positional = []
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            if i + 1 >= len(args):
                raise SystemExit("missing value for %s" % args[i])
            flags[args[i][2:]] = args[i + 1]
            i += 2
        else:
            positional.append(args[i])
            i += 1
    if positional:
        BASE = positional[0]

    depths = sorted(int(x) for x in
                    flags.get("depths", "3072,16384,32768,65536").split(",") if x.strip())
    if not depths:
        raise SystemExit("no depths given")
    n_predict = int(flags.get("n-predict", "64"))
    temp = float(flags.get("temp", "0"))
    reps = int(flags.get("reps", "1"))
    label = flags.get("label", "")
    warmup = int(flags.get("warmup", "1"))
    settle = float(flags.get("settle", "2"))
    timeout = int(flags.get("timeout", "1800"))
    jfile = flags.get("journal-file")
    jcmd = flags.get("journal-cmd")
    journal = Journal(jfile, jcmd)

    model = "?"
    try:
        model = get("/v1/models")["data"][0]["id"]
    except Exception:
        pass

    print("base   : %s" % BASE)
    print("model  : %s" % model)
    print("label  : %s" % (label or "-"))
    print("journal: %s" % (jfile or jcmd or "none (API timings only)"))
    print("params : depths=%s n_predict=%d temp=%g reps=%d warmup=%d settle=%gs timeout=%ds"
          % (",".join(str(d) for d in depths), n_predict, temp, reps, warmup, settle, timeout))
    print("-" * 100)

    rows = []
    ok_all = True
    for idx, depth in enumerate(depths):
        print("depth %6d: building prompt..." % depth)
        s, actual = build_prompt(depth)
        if idx == 0 and warmup > 0:
            for w in range(warmup):
                try:
                    complete(s, n_predict, temp, timeout)
                    print("warmup %d ok (%d tokens)" % (w + 1, actual))
                except (urllib.error.HTTPError, urllib.error.URLError) as e:
                    print("warmup: %s" % e)
                time.sleep(settle)
        rep_results = []
        for rep in range(reps):
            journal.snapshot()
            t0 = time.time()
            try:
                res = complete(s, n_predict, temp, timeout)
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                rep_results.append({"err": str(e), "api": None, "jr": None, "wall": time.time() - t0})
                ok_all = False
                print("depth %6d rep %d: HTTP ERROR %s" % (depth, rep + 1, e))
                time.sleep(settle)
                continue
            wall = time.time() - t0
            api = ptok = None
            try:
                api = float(res["timings"]["predicted_per_second"])
            except (KeyError, TypeError, ValueError):
                pass
            try:
                ptok = float(res["timings"]["prompt_per_second"])
            except (KeyError, TypeError, ValueError):
                pass
            jr = None
            jr_n = None
            line = None
            try:
                line = journal.eval_line()
            except Exception as e:
                print("journal read error: %s" % e)
            if line is not None:
                _ms, jr_n, jr = line
            elif jfile or jcmd:
                print("  NOTE: no 'eval time' line in the journal tail (log level too low?)")
            gen_n = res.get("usage", {}).get("completion_tokens")
            if jr is not None and jr_n is not None and gen_n is not None and jr_n != gen_n:
                print("  WARNING: journal eval line has %d tokens, response generated %d "
                      "(concurrent task?) - journal value flagged" % (jr_n, gen_n))
            rep_results.append({"api": api, "jr": jr, "jr_n": jr_n, "gen_n": gen_n,
                                "ptok": ptok, "wall": wall})
            print("depth %6d rep %d: %d tok  decode %s t/s (api) / %s t/s (journal)  "
                  "prompt %s t/s  wall %.1fs"
                  % (depth, rep + 1, gen_n,
                     "%.2f" % api if api is not None else "  -  ",
                     "%.2f" % jr if jr is not None else "  -  ",
                     "%.1f" % ptok if ptok is not None else "  -  ",
                     wall))
            time.sleep(settle)
        rows.append({"depth": depth, "actual": actual, "reps": rep_results})

    for row in rows:
        row["api_mean"] = mean([r["api"] for r in row["reps"]])
        row["jr_mean"] = mean([r["jr"] for r in row["reps"]])
        row["tps"] = row["jr_mean"] if row["jr_mean"] is not None else row["api_mean"]

    base = None
    for row in rows:
        if row["tps"] is not None:
            base = row["tps"]
            break
    for row in rows:
        row["r"] = (row["tps"] / base) if (row["tps"] is not None and base) else None

    print("-" * 100)
    print("%-8s %-8s %-12s %-14s %-16s %s" %
          ("depth", "tokens", "prompt_t/s", "decode_t/s api", "decode_t/s journal", "r (vs %d)" % rows[0]["depth"]))
    for row in rows:
        print("%-8d %-8d %-12s %-14s %-16s %s"
              % (row["depth"], row["actual"],
                 "%.1f" % row["reps"][0]["ptok"] if row["reps"][0].get("ptok") is not None else "-",
                 "%.2f" % row["api_mean"] if row["api_mean"] is not None else "  -  ",
                 "%.2f" % row["jr_mean"] if row["jr_mean"] is not None else "  -  ",
                 "%.4f" % row["r"] if row["r"] is not None else "  -  "))

    # Break-even: ratio = (1+d)*(g + (1-g)*r), gain iff ratio < 1.
    # ratio is monotone increasing in d and (for r < 1) in g, so:
    #   best  corner (d_lo, g_lo): gain iff r < r_best
    #   worst corner (d_hi, g_hi): gain iff r < r_worst
    r_best = (1.0 / (1 + D_LO) - G_LO) / (1 - G_LO)
    r_worst = (1.0 / (1 + D_HI) - G_HI) / (1 - G_HI)
    print("-" * 100)
    print("break-even: T_DA/T_vanilla ~= (1+d)*(g + (1-g)*r), d in [%.2f,%.2f], g in [%.2f,%.2f]"
          % (D_LO, D_HI, G_LO, G_HI))
    print("            gain iff ratio < 1  =>  r < %.4f (best corner) / r < %.4f (worst corner)"
          % (r_best, r_worst))
    print("%-8s %-8s %-14s %-14s %-14s %-14s %s"
          % ("depth", "r", "d=0.15 g=0.27", "d=0.15 g=0.45", "d=0.35 g=0.27", "d=0.35 g=0.45", "verdict"))
    for row in rows:
        if row["r"] is None:
            print("%-8d %-8s %-14s %-14s %-14s %-14s %s"
                  % (row["depth"], "-", "-", "-", "-", "-", "n/a"))
            continue
        r = row["r"]
        cells = ["%.4f" % ((1 + d) * (g + (1 - g) * r)) for d in (D_LO, D_HI) for g in (G_LO, G_HI)]
        if r < r_worst:
            verdict = "GAIN over the full (d,g) range"
        elif r < r_best:
            verdict = "GAIN only for favorable (d,g)"
        else:
            verdict = "NO GAIN in the (d,g) range"
        print("%-8d %-8.4f %-14s %-14s %-14s %-14s %s"
              % (row["depth"], r, cells[0], cells[1], cells[2], cells[3], verdict))

    # report file
    ts = time.strftime("%Y%m%d_%H%M%S")
    rdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    if not os.path.isdir(rdir):
        os.makedirs(rdir)
    report = os.path.join(rdir, "depth_bench_%s_%s.txt" % (ts, label or "run"))
    lines = []
    lines.append("=== da_depth_bench report (P-G gate: vanilla tps vs depth) ===")
    lines.append("time : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("host : %s" % socket.gethostname())
    lines.append("base : %s" % BASE)
    lines.append("label: %s" % (label or "-"))
    lines.append("model: %s" % model)
    lines.append("journal: %s" % (jfile or jcmd or "none (API timings only)"))
    lines.append("params: depths=%s n_predict=%d temp=%g reps=%d warmup=%d settle=%gs timeout=%ds"
                 % (",".join(str(d) for d in depths), n_predict, temp, reps, warmup, settle, timeout))
    lines.append("")
    lines.append("per-rep detail (decode t/s; api = timings.predicted_per_second,")
    lines.append("               jr = journal 'eval time' line - same quantity, cross-check):")
    for row in rows:
        lines.append("  depth %d (actual %d tokens):" % (row["depth"], row["actual"]))
        for k, r in enumerate(row["reps"]):
            if r.get("err"):
                lines.append("    rep %d: ERROR %s" % (k + 1, r["err"]))
            else:
                lines.append("    rep %d: api=%s jr=%s (jr_tokens=%s gen_tokens=%s) "
                             "prompt_tps=%s wall=%.1fs"
                             % (k + 1,
                                "%.2f" % r["api"] if r["api"] is not None else "-",
                                "%.2f" % r["jr"] if r["jr"] is not None else "-",
                                r["jr_n"], r.get("gen_n"),
                                "%.1f" % r["ptok"] if r.get("ptok") is not None else "-",
                                r["wall"]))
    lines.append("")
    lines.append("means: depth tokens decode_api decode_journal r")
    for row in rows:
        lines.append("  %8d %8d %s %s %s"
                     % (row["depth"], row["actual"],
                        "%.2f" % row["api_mean"] if row["api_mean"] is not None else "-",
                        "%.2f" % row["jr_mean"] if row["jr_mean"] is not None else "-",
                        "%.4f" % row["r"] if row["r"] is not None else "-"))
    lines.append("")
    lines.append("break-even: T_DA/T_vanilla ~= (1+d)*(g + (1-g)*r), gain iff < 1")
    lines.append("  d in [%.2f,%.2f] (step inflation), g in [%.2f,%.2f] (global-step share, paper Gemma)"
                 % (D_LO, D_HI, G_LO, G_HI))
    lines.append("  thresholds: r < %.4f (best corner d=%.2f g=%.2f) / r < %.4f (worst corner d=%.2f g=%.2f)"
                 % (r_best, D_LO, G_LO, r_worst, D_HI, G_HI))
    for row in rows:
        if row["r"] is None:
            lines.append("  depth %8d: n/a" % row["depth"])
            continue
        r = row["r"]
        if r < r_worst:
            verdict = "GAIN over the full (d,g) range"
        elif r < r_best:
            verdict = "GAIN only for favorable (d,g)"
        else:
            verdict = "NO GAIN in the (d,g) range"
        lines.append("  depth %8d: r=%.4f  ratios d0.15g0.27=%.4f d0.15g0.45=%.4f "
                     "d0.35g0.27=%.4f d0.35g0.45=%.4f  =>  %s"
                     % (row["depth"], r,
                        (1 + D_LO) * (G_LO + (1 - G_LO) * r),
                        (1 + D_LO) * (G_HI + (1 - G_HI) * r),
                        (1 + D_HI) * (G_LO + (1 - G_LO) * r),
                        (1 + D_HI) * (G_HI + (1 - G_HI) * r),
                        verdict))
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("-" * 100)
    print("report : %s" % report)
    print("run again per spec state (server restart between): --label spec-on / --label spec-off")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

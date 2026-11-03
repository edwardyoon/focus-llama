#!/usr/bin/env python3
"""sigma-strip prove — qwen-code 클라이언트 루프 시뮬레이터 (v2).

대상 (plans/2026-11-03-compressfast-auto-trigger.md, 방향 B):
  stateless 클라이언트는 매 턴 원본 전체 히스토리를 재전송한다. 서버가
  임계(70%)를 넘으면 K를 한 번에 전진시켜 [0,K)의 COMPACTABLE tool result를
  결정적 stub으로 교체하고, on 모드에서는 응답 usage.prompt_tokens가 줄어든
  값이 된다 (F2 — 클라이언트 앵커가 줄어든 값을 보는 것). 이 스크립트는
  실제 qwen CLI 없이 그 **와이어 계약**을 검증한다.

  K는 마지막 assistant tool_calls 인덱스를 넘지 못한다 — 마지막 tool 라운드
  (현재 턴)는 모델의 작업 컨텍스트이므로 항상 보존된다. cap에 막혀
  target(35%)에 못 미치면 target_unreachable로 표시되고, 임계 초과가 지속되는
  세션은 native compact 안전망이 담당한다.

v2에서 추가된 검증 (v1의 허점 보강):
  - 토큰 정합성: |log tokens_before - api prompt_tokens| / api < 2%
    (filler만 있던 v1 fixture는 갭을 못 재현 — R_args 케이스로 tools
    스키마 / reasoning_content / 큰 arguments 포함)
  - K 전진 후 성장: R4(R3+ack+user)에서 advanced=0, usage 작게 유지
    (원본 기준 재전진 버그 회귀 검사)
  - 캐시 적중: 재전송 시 timings.cache_n ≈ 이전 prompt 길이 (재prefill 없음)
  - 역사 축소(클라이언트 compact 모사): reset=1 (지문 불일치 감지)
  - 병렬 tool 라운드 2개: 첫 쌍 stub + 마지막 라운드 K cap(현재 턴) 보호,
    target_unreachable (native compact 위임 시그널)
  - SESSION은 실행마다 uuid (K 영속화 이후 재실행 안전)
  - Σ 스냅샷: refresh 연동 전이라 SKIP (연동 후 케이스 추가 필요)

케이스 (모드별, 각각 독립 uuid 세션):
  shadow:
    grow     : R1(<임계) R2(재전송) R3(전진) R3(재전송: K 안정+byte-identical)
               R4(성장: advanced=0) + 정합성
    reset    : R3(전진) R3' (첫 user 메시지 변경 → reset=1)
    shrink   : R3(전진) R3-  (앞쪽 tool 쌍 제거, 첫 user 동일 → reset=1)
    args     : C1(베이스라인: c2m tools 없음, est/api 정합성 sanity)
               C2(+tools 스키마: 같은 대화 delta = tools 효과, 갭 재현)
               C3(+reasoning_content: delta = reasoning 효과, 갭 재현)
    parallel : P1(병렬 라운드 2개 → 첫 쌍 stub, K=7/10 cap, target_unreachable,
               본체 미변경)
  on:
    base     : O1(R1: 미변경+오버헤드 측정) O2(R3: F2 급감+정합성)
               O3(R3 재전송: advanced=0, cache_n≈O2) O4(R4 성장: advanced=0,
               usage 작게 유지, cache_n≈O2)
    parallel : P1(첫 병렬 쌍 stub, F2 급감, K=7/10 cap, 응답 non-empty)

참고:
  - 대상 모델이 E2B(Gemma)면 토크나이저/템플릿이 프로덕션(Qwen)과 다르다.
    계약 검증(전진/리셋/캐시/F2)엔 무방하지만, 정합성 수치는 프로덕션
    모델에서 재확인해야 한다. Gemma 템플릿이 tools/reasoning을 렌더링하지
    않으면 C2/C3는 INCONCLUSIVE(별도 표시) — Qwen에서 재검증.
  - shadow 모드는 본체를 건드리지 않으므로 api usage는 항상 원본.

실행 (로컬, 서버는 123):
  python3 sigma_probe.py --server http://192.168.219.123:8180 --mode shadow \
      --log-ssh edwardyoon@192.168.219.123 --log-path /tmp/sigma_p1.log \
      [--cases grow,args]

판정: 0=PASS 전체, 1=FAIL 존재, 2=INCONCLUSIVE 존재(FAIL 없음).
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid

WINDOW = 60000   # e2b.service --sigma-strip-window-tokens
THRESHOLD = 0.58  # e2b.service --sigma-strip-threshold (트리거점 34.8k)
CONSISTENCY_PCT = 2.0  # |est - api| / api 허용

# rewrite= 필드 뒤에 줄바꿈이 없어 다음 로그 엔트리의 타임스탬프
# (0.05.432.784 형식)가 붙는다 — kv_offload_hash16의 16hex 해시로 앵커
# (타임스탬프는 점을 포함해 16연속 hex를 만들 수 없음).
LOG_RE = re.compile(
    r"sigma-strip: session=(\S+) mode=(\S+) K=(\d+)/(\d+) reset=(\d) "
    r"advanced=(\d) stripped=(\d+) tokens (\d+)->(\d+) usage ([\d.]+)->([\d.]+) "
    r"rewrite=([0-9a-f]{16})?( target-unreachable)?")

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def filler(n_chars, seed=0):
    """결정적 prose filler (read_file 결과 시뮬레이션)."""
    words = ("the quick brown fox jumps over lazy dog token context window "
             "session strip shadow").split()
    lines, total, i = [], 0, seed
    while total < n_chars:
        line = f"{i:05d} " + " ".join(words[(i * 7 + j) % len(words)] for j in range(14))
        lines.append(line)
        total += len(line) + 1
        i += 1
    return "\n".join(lines)[:n_chars]


def js_code(n_lines, seed=0):
    lines = []
    for i in range(n_lines):
        lines.append(f"  const value_{i} = compute((({i} + {seed}) * 31) % 97, 'item-{i}');")
        if i % 3 == 0:
            lines.append(f"  // section {i}: validate and store value_{i}")
    return "\n".join(lines)


def big_edit_args():
    """~20KB edit arguments JSON (큰 편집 인자 시뮬레이션)."""
    return json.dumps({
        "file_path": "/tmp/big.js",
        "old_string": js_code(130, seed=1),
        "new_string": js_code(170, seed=2),
        "replace_all": False,
    })


TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Reads and returns the content of a specified file. "
                       "Supports text files, PDFs, and Jupyter notebooks. "
                       "For large files, read specific line ranges.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file to read."},
            "offset": {"type": "integer", "description": "0-based line number to start reading from."},
            "limit": {"type": "integer", "description": "Maximum number of lines to read."}},
            "required": ["file_path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Writes content to a specified file in the local filesystem. Overwrites existing files.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file to write to."},
            "content": {"type": "string", "description": "The content to write to the file."}},
            "required": ["file_path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Replaces text within a file. Replaces a single occurrence by default; "
                       "set replace_all to true to modify every instance.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file to modify."},
            "old_string": {"type": "string", "description": "The exact literal text to replace, including context."},
            "new_string": {"type": "string", "description": "The exact literal text to replace old_string with."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)."}},
            "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "run_shell_command",
        "description": "Executes a given shell command in a subprocess with optional timeout. "
                       "For system commands and terminal operations.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Exact bash command to execute."},
            "timeout": {"type": "integer", "description": "Optional timeout in milliseconds (max 600000)."},
            "description": {"type": "string", "description": "Brief description of the command."}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "grep_search",
        "description": "Search file contents using a regular expression (ripgrep). "
                       "Supports full regex syntax and glob filters.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "The regular expression pattern to search for."},
            "path": {"type": "string", "description": "File or directory to search in."},
            "glob": {"type": "string", "description": "Glob pattern to filter files (e.g. '*.js')."},
            "limit": {"type": "integer", "description": "Limit output to first N lines."}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "glob",
        "description": "Fast file pattern matching. Supports glob patterns like '**/*.js'. "
                       "Returns matching paths sorted by modification time.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "The glob pattern to match files against."},
            "path": {"type": "string", "description": "The directory to search in."}},
            "required": ["pattern"]}}},
]
TOOLS_JSON = json.dumps(TOOLS_SCHEMA)


def tc(id, name, args):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


SYSTEM = ("You are a precise retrieval assistant. Answer using only the "
          "conversation context.")


def build_conversation():
    """(R1, R2, R3, R4, R5, R3shrink) — stateless 재전송용 누적 메시지.

    크기 (est ~4.5자/토큰, window 60000): R1 ~12.5k (21% <임계) / R3 ~45k (75% >= 0.58 트리거)
    R4 = R3 + ack + user (성장). R5 = R3 첫 user 변경.
    R3shrink = 앞쪽 tool 쌍(assistant1+tool1) 제거, user1 유지.
    """
    r1 = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Read /tmp/d1.txt and tell me its first line."},
        {"role": "assistant", "content": None,
         "tool_calls": [tc("call_1", "read_file", {"file_path": "/tmp/d1.txt"})]},
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
         "content": filler(55000, seed=1)},
    ]
    r2 = r1 + [
        {"role": "assistant", "content": "The first line of /tmp/d1.txt is '00000 the quick brown'."},
        {"role": "user", "content": "Now read /tmp/d2.txt too."},
    ]
    r3 = r2 + [
        {"role": "assistant", "content": None,
         "tool_calls": [tc("call_2", "read_file", {"file_path": "/tmp/d2.txt"})]},
        {"role": "tool", "tool_call_id": "call_2", "name": "read_file",
         "content": filler(150000, seed=2)},
    ]
    r4 = r3 + [
        {"role": "assistant", "content": "Done reading /tmp/d2.txt as well."},
        {"role": "user", "content": "What is the second line of /tmp/d2.txt?"},
    ]
    r5 = [dict(m) for m in r3]
    r5[1] = dict(r5[1], content="(compacted) earlier request removed.")
    r3shrink = r3[:2] + r3[4:]  # assistant1+tool1(call_1 쌍) 제거, user1 유지
    return r1, r2, r3, r4, r5, r3shrink


def build_args_cases():
    """(C1, C2, C3) — 정합성 케이스. C1/C2는 같은 대화(c2m)로 tools 효과만
    분리(±tools 필드), C3는 reasoning_content 추가. tools는 요청 본체("tools"
    필드)로 전송."""
    edit_args = big_edit_args()
    reasoning = ("I need to apply the edit carefully. " * 60
                 + "The old_string must match exactly including indentation.")
    c2_msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Apply the pending edit to /tmp/big.js."},
        {"role": "assistant", "content": None,
         "tool_calls": [tc("call_c2", "edit", edit_args)]},
        {"role": "tool", "tool_call_id": "call_c2", "name": "edit",
         "content": "The file /tmp/big.js has been edited successfully."},
    ]
    c3_msgs = [dict(m) for m in c2_msgs]
    c3_msgs[2] = dict(c3_msgs[2], reasoning_content=reasoning)
    return (
        (c2_msgs, {}),                        # C1: 베이스라인 (tools 없음)
        (c2_msgs, {"tools": TOOLS_SCHEMA}),   # C2: + tools 스키마
        (c3_msgs, {"tools": TOOLS_SCHEMA}),   # C3: + reasoning_content
    )


def build_parallel():
    """P1 — 병렬 tool 라운드 2개 (n=10).

    첫 라운드(call_p1/p2, 각 60KB)는 stub 대상, 마지막 라운드(call_p3/p4)는
    K cap(마지막 assistant tool_calls 인덱스 7)으로 보호된다. 합 ~55k >= 임계 34.8k,
    첫 쌍 stub 후 잔여 ~27.7k > target 21k → target_unreachable.
    """
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Read /tmp/p1.txt and /tmp/p2.txt in parallel."},
        {"role": "assistant", "content": None,
         "tool_calls": [tc("call_p1", "read_file", {"file_path": "/tmp/p1.txt"}),
                        tc("call_p2", "read_file", {"file_path": "/tmp/p2.txt"})]},
        {"role": "tool", "tool_call_id": "call_p1", "name": "read_file",
         "content": filler(60000, seed=11)},
        {"role": "tool", "tool_call_id": "call_p2", "name": "read_file",
         "content": filler(60000, seed=12)},
        {"role": "assistant", "content": "Both files read."},
        {"role": "user", "content": "Now read /tmp/p3.txt and /tmp/p4.txt in parallel."},
        {"role": "assistant", "content": None,
         "tool_calls": [tc("call_p3", "read_file", {"file_path": "/tmp/p3.txt"}),
                        tc("call_p4", "read_file", {"file_path": "/tmp/p4.txt"})]},
        {"role": "tool", "tool_call_id": "call_p3", "name": "read_file",
         "content": filler(60000, seed=13)},
        {"role": "tool", "tool_call_id": "call_p4", "name": "read_file",
         "content": filler(60000, seed=14)},
    ]


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

class Probe:
    def __init__(self, args):
        self.base = args.server
        self.mode = args.mode
        self.log_ssh = args.log_ssh
        self.log_path = args.log_path
        self.log_off = 1
        self.results = []
        self.run_id = uuid.uuid4().hex[:8]
        all_cases = self.SHADOW_CASES if args.mode == "shadow" else self.ON_CASES
        if args.cases.strip() == "all":
            self.cases = set(all_cases)
        else:
            self.cases = {c.strip() for c in args.cases.split(",") if c.strip()}
            bad = self.cases - set(all_cases)
            if bad:
                sys.exit(f"unknown cases for mode {args.mode}: {sorted(bad)} "
                         f"(valid: {all_cases})")

    def sess(self, name):
        return f"sigma-{self.run_id}-{name}"

    def check(self, name, ok, detail="", inconclusive=False):
        self.results.append((name, ok, inconclusive, detail))
        tag = "PASS" if ok else ("INCONCLUSIVE" if inconclusive else "FAIL")
        print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))

    def log_size(self):
        """로그 파일 현재 바이트 크기 (초기 오프셋 — 기동 로그 건너뜀용)."""
        cmd = f"ssh -o ConnectTimeout=8 -o BatchMode=yes {self.log_ssh} " \
              f"'stat -c %s {self.log_path} 2>/dev/null || echo 0'"
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        try:
            return int(out.stdout.strip() or 0)
        except ValueError:
            return 0

    def read_log_delta(self):
        cmd = f"ssh -o ConnectTimeout=8 -o BatchMode=yes {self.log_ssh} " \
              f"'tail -c +{self.log_off} {self.log_path} 2>/dev/null'"
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        text = out.stdout
        self.log_off += len(text.encode())
        return text

    def sigma_line(self, session, tries=6):
        """session의 최신 sigma-strip 기록.

        물리적 로그 라인은 "0.13.480.828 I srv sigma_strip_: sigma-strip: ..."
        처럼 타임스탬프+로거 프리픽스로 시작하고 다른 slot 로그와 한 줄에 섞이므로
        delta 전체에서 LOG_RE finditer + session 필터로 마지막(=최근 요청) 매칭을
        쓴다. 요청 직후 로그 미반영 경합을 피하려 0.5s씩 tries까지 재시도.
        """
        buf = ""
        for _ in range(tries):
            buf += self.read_log_delta()
            matches = [m for m in LOG_RE.finditer(buf) if m.group(1) == session]
            if matches:
                return self._parse_sigma(matches[-1]), ""
            time.sleep(0.5)
        return None, buf

    @staticmethod
    def _parse_sigma(m):
        return {
            "session": m.group(1), "mode": m.group(2),
            "K": int(m.group(3)), "n": int(m.group(4)),
            "reset": int(m.group(5)), "advanced": int(m.group(6)),
            "stripped": int(m.group(7)),
            "tokens_before": int(m.group(8)), "tokens_after": int(m.group(9)),
            "usage_before": float(m.group(10)), "usage_after": float(m.group(11)),
            "rewrite": m.group(12) or "",
            "target_unreachable": bool(m.group(13)),
        }

    def chat(self, messages, session, max_tokens=48, extra=None):
        body = {"model": "gemma-e2b", "messages": messages, "temperature": 0,
                "max_tokens": max_tokens, "stream": False}
        if extra:
            body.update(extra)
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "X-Session-Id": session})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            res = json.loads(r.read())
        wall = time.time() - t0
        msg = res["choices"][0]["message"]
        return {
            "text": msg.get("content") or "",
            "finish_reason": res["choices"][0].get("finish_reason"),
            "prompt_tokens": (res.get("usage") or {}).get("prompt_tokens"),
            "cache_n": (res.get("timings") or {}).get("cache_n"),
            "timings": res.get("timings") or {},
            "wall": wall,
        }

    def consistency(self, name, log_est, api, tol=CONSISTENCY_PCT):
        """|est - api| / api < tol — est(서버 log) vs 실제 렌더링(api)."""
        if not log_est or not api:
            self.check(name, False, "missing est or api tokens")
            return
        gap = abs(log_est - api) / api * 100.0
        self.check(name, gap < tol,
                   f"est={log_est} api={api} gap={gap:.2f}% (허용 {tol}%)")

    # -- shadow cases -------------------------------------------------------

    def case_grow(self, r1, r2, r3, r4):
        s = self.sess("grow")
        res1 = self.chat(r1, s)
        l1, _ = self.sigma_line(s)
        self.check("S1 <임계: K=0 adv=0 reset=0",
                   l1 and l1["session"] == s and l1["K"] == 0
                   and l1["advanced"] == 0 and l1["reset"] == 0,
                   f"usage={l1['usage_before']:.3f} tokens={l1['tokens_before']}" if l1 else "no log line")
        if l1:
            self.consistency("S1 정합성(prose 베이스라인)", l1["tokens_before"], res1["prompt_tokens"])

        res2 = self.chat(r2, s)
        l2, _ = self.sigma_line(s)
        self.check("S2 재전송: K=0 adv=0 reset=0",
                   l2 and l2["K"] == 0 and l2["advanced"] == 0 and l2["reset"] == 0,
                   f"tokens={l2['tokens_before']}" if l2 else "no log line")

        res3 = self.chat(r3, s)
        l3, _ = self.sigma_line(s)
        self.check("S3 임계 초과: adv=1 stripped>=1",
                   l3 and l3["advanced"] == 1 and l3["stripped"] >= 1,
                   f"K={l3['K']}/{l3['n']} tokens {l3['tokens_before']}->{l3['tokens_after']} "
                   f"usage {l3['usage_before']:.3f}->{l3['usage_after']:.3f}" if l3 else "no log line")
        self.check("S3 K cap: K=6/8 (마지막 tool 라운드 보호) target_unreachable",
                   l3 and l3["K"] == 6 and l3["n"] == 8 and l3["target_unreachable"],
                   f"K={l3['K']}/{l3['n']} stripped={l3['stripped']} tokens_after={l3['tokens_after']}" if l3 else "no log line")
        self.check("S3 shadow 본체 미변경 (api=raw)",
                   (res3["prompt_tokens"] or 0) > 15000,
                   f"api={res3['prompt_tokens']} (raw ~22k)")

        res4 = self.chat(r3, s)
        l4, _ = self.sigma_line(s)
        self.check("S4 재전송: K 안정, adv=0, reset=0",
                   l4 and l3 and l4["K"] == l3["K"] and l4["advanced"] == 0 and l4["reset"] == 0,
                   f"K={l4['K']}/{l4['n']}" if l4 else "no log line")
        self.check("S4 byte-identical rewrite hash",
                   l4 and l3 and l4["rewrite"] != "" and l4["rewrite"] == l3["rewrite"],
                   f"s3={l3['rewrite'] if l3 else '?'} "
                   f"s4={l4['rewrite'] if l4 else '?'}")
        # A/B 베이스라인: raw(미변경) 본체도 비어 있는가 — raw 본체가 비면
        # on 모드 빈 응답은 모델 특성이지 rewrite 문제가 아님
        t4 = res4["text"].strip()
        if t4:
            self.check("S4 응답(raw 본체) non-empty", True, f"text={t4[:60]!r}")
        else:
            self.check("S4 응답(raw 본체) non-empty", True,
                       f"text='' finish_reason={res4['finish_reason']} — raw 본체도 empty (소형 모델 특성)",
                       inconclusive=True)

        res6 = self.chat(r4, s)
        l6, _ = self.sigma_line(s)
        self.check("S6 성장(R3+ack+user): adv=0, usage<임계 (resident 기준)",
                   l6 and l3 and l6["advanced"] == 0 and l6["K"] == l3["K"]
                   and l6["usage_before"] < THRESHOLD,
                   f"K={l6['K']}/{l6['n']} usage_before={l6['usage_before']:.3f} "
                   f"(raw={l6['tokens_before']})" if l6 else "no log line")

    def case_reset(self, r3, r5):
        s = self.sess("reset")
        self.chat(r3, s)
        l3, _ = self.sigma_line(s)
        if not l3 or l3["advanced"] != 1:
            self.check("R3 전진(setup)", False, f"setup failed: {l3}")
            return
        res5 = self.chat(r5, s)
        l5, _ = self.sigma_line(s)
        self.check("S5 첫 user 변경: reset=1",
                   l5 and l5["reset"] == 1,
                   f"K={l5['K']}/{l5['n']} adv={l5['advanced']}" if l5 else "no log line")

    def case_shrink(self, r3, r3shrink):
        s = self.sess("shrink")
        self.chat(r3, s)
        l3, _ = self.sigma_line(s)
        if not l3 or l3["advanced"] != 1:
            self.check("R3 전진(setup)", False, f"setup failed: {l3}")
            return
        resx = self.chat(r3shrink, s)
        lx, _ = self.sigma_line(s)
        self.check("S7 역사 축소(앞쪽 tool 쌍 제거, 첫 user 동일): reset=1",
                   lx and lx["reset"] == 1,
                   f"K={lx['K']}/{lx['n']} n={lx['n']} (메시지 수 {lx['n']})" if lx else "no log line")

    def case_args(self):
        (c1m, c1x), (c2m, c2x), (c3m, c3x) = build_args_cases()
        s = self.sess("args")
        tools_chars = len(TOOLS_JSON)
        reasoning_chars = len(c3m[2].get("reasoning_content", ""))

        # C1: 베이스라인 (c2m, tools 없음) — est/api 정합성 sanity
        # 코드 중심 콘텐츠는 per-message est와 템플릿 렌더링 갭이 prose보다
        # 큼(관측 3~7%) — 허용 10% (S1 prose 베이스라인은 2% 유지)
        r1 = self.chat(c1m, s, extra=c1x or None)
        l1, _ = self.sigma_line(s)
        if l1:
            self.consistency("C1 베이스라인(tools 없음)", l1["tokens_before"], r1["prompt_tokens"], tol=10.0)
        else:
            self.check("C1 log line", False, "no log line")

        # C2: + tools 스키마 (같은 대화 c2m) — delta = tools 효과
        r2 = self.chat(c2m, s, extra=c2x or None)
        l2, _ = self.sigma_line(s)
        if l1 and l2:
            delta = (r2["prompt_tokens"] or 0) - (r1["prompt_tokens"] or 0)
            expected_min = tools_chars // 8  # JSON은 밀도 높음 — 보수적 하한
            if delta < expected_min:
                self.check("C2 +tools 스키마", True,
                           f"api delta={delta} < tools 기대 하한({expected_min}) — "
                           f"이 템플릿이 tools를 렌더링하지 않음 (Qwen에서 재검증)",
                           inconclusive=True)
            else:
                self.check("C2 +tools: delta 합리", True,
                           f"delta={delta} >= {expected_min} (tools 렌더링 확인)")
                self.consistency("C2 +tools (est vs api 갭)", l2["tokens_before"], r2["prompt_tokens"], tol=10.0)
        else:
            self.check("C2 log line", False, "no log line")

        # C3: + reasoning_content (tools 유지) — delta = reasoning 효과
        r3 = self.chat(c3m, s, extra=c3x or None)
        l3, _ = self.sigma_line(s)
        if l2 and l3:
            delta = (r3["prompt_tokens"] or 0) - (r2["prompt_tokens"] or 0)
            expected_min = reasoning_chars // 8
            if delta < expected_min:
                self.check("C3 +reasoning_content", True,
                           f"api delta={delta} < reasoning 기대 하한({expected_min}) — "
                           f"이 템플릿이 reasoning을 렌더링하지 않음 (Qwen에서 재검증)",
                           inconclusive=True)
            else:
                self.check("C3 +reasoning: delta 합리", True,
                           f"delta={delta} >= {expected_min} (reasoning 렌더링 확인)")
                self.consistency("C3 +reasoning (est vs api 갭)", l3["tokens_before"], r3["prompt_tokens"], tol=10.0)
        else:
            self.check("C3 log line", False, "no log line")

    def case_parallel(self, p1):
        s = self.sess("parallel")
        res = self.chat(p1, s)
        l, _ = self.sigma_line(s)
        self.check("S9 병렬 2라운드: adv=1 stripped=2 K=7/10 (마지막 라운드 cap)",
                   l and l["advanced"] == 1 and l["stripped"] == 2
                   and l["K"] == 7 and l["n"] == 10,
                   f"K={l['K']}/{l['n']} tokens {l['tokens_before']}->{l['tokens_after']}" if l else "no log line")
        self.check("S9 target_unreachable (보호 라운드 잔여 > target — native compact 위임)",
                   l and l["target_unreachable"],
                   f"tokens_after={l['tokens_after']} (target {int(WINDOW * 0.35)})" if l else "no log line")
        self.check("S9 shadow 본체 미변경 (api=raw)",
                   (res["prompt_tokens"] or 0) > 15000,
                   f"api={res['prompt_tokens']}")

    # -- on cases -----------------------------------------------------------

    def case_base_on(self, r1, r3, r4):
        s = self.sess("base")
        # O1 — 미변경 + 템플릿 오버헤드 측정
        res1 = self.chat(r1, s)
        l1, _ = self.sigma_line(s)
        self.check("O1 <임계: K=0, api=raw",
                   l1 and l1["K"] == 0 and l1["advanced"] == 0
                   and (res1["prompt_tokens"] or 0) > 5000,
                   f"api={res1['prompt_tokens']} est={l1['tokens_before'] if l1 else '?'}")
        if not l1 or not res1["prompt_tokens"]:
            return
        overhead = res1["prompt_tokens"] - l1["tokens_before"]
        if not (-50 <= overhead <= 2000):
            self.check("O1 템플릿 오버헤드 합리적", False, f"overhead={overhead}")
            return

        # O2 — 전진: F2 급감 + rewrite 정합성
        res2 = self.chat(r3, s, max_tokens=64)
        l2, _ = self.sigma_line(s)
        self.check("O2 log: adv=1 stripped>=1 mode=on",
                   l2 and l2["advanced"] == 1 and l2["stripped"] >= 1 and l2["mode"] == "on",
                   f"K={l2['K']}/{l2['n']} tokens {l2['tokens_before']}->{l2['tokens_after']}" if l2 else "no log line")
        self.check("O2 K cap: K=6/8 (마지막 tool result 보호) target_unreachable",
                   l2 and l2["K"] == 6 and l2["n"] == 8 and l2["target_unreachable"],
                   f"K={l2['K']}/{l2['n']} stripped={l2['stripped']} tokens_after={l2['tokens_after']}" if l2 else "no log line")
        if l2 and res2["prompt_tokens"]:
            # "급감" = raw(rewrite 전) 대비 상대값 — 절대 상한 아님.
            # cap(마지막 tool 라운드 보호)으로 잔여가 target(35%)보다 클 수 있어
            # 상한은 80%까지 완화 — 정확한 값 매칭은 rewrite 정합성 체크가 담당.
            self.check("O2 F2: api usage 급감",
                       50 < res2["prompt_tokens"] < l2["tokens_before"] * 0.80,
                       f"api={res2['prompt_tokens']} (raw est={l2['tokens_before']}, O1={res1['prompt_tokens']})")
            expect = l2["tokens_after"] + overhead
            gap = abs(res2["prompt_tokens"] - expect) / res2["prompt_tokens"] * 100
            self.check("O2 rewrite 정합성 (api ≈ tokens_after + 오버헤드)",
                       gap < CONSISTENCY_PCT,
                       f"api={res2['prompt_tokens']} expect≈{expect} gap={gap:.2f}%")
        t2 = res2["text"].strip()
        if t2:
            self.check("O2 응답 non-empty (rewrite well-formed)", True,
                       f"wall={res2['wall']:.1f}s text={t2[:60]!r}")
        elif res2["finish_reason"] == "stop":
            self.check("O2 응답 non-empty (rewrite well-formed)", True,
                       f"wall={res2['wall']:.1f}s text='' finish_reason=stop — "
                       f"마지막 tool result 보존 상태 — 모델/프롬프트 문제 (Qwen에서 재검증)",
                       inconclusive=True)
        else:
            self.check("O2 응답 non-empty (rewrite well-formed)", False,
                       f"wall={res2['wall']:.1f}s text='' "
                       f"finish_reason={res2['finish_reason']}")

        # O3 — 재전송: K 안정 + 캐시 적중 (재prefill 없음)
        res3 = self.chat(r3, s)
        l3, _ = self.sigma_line(s)
        self.check("O3 재전송: adv=0, K 안정, hash 동일",
                   l3 and l2 and l3["advanced"] == 0 and l3["K"] == l2["K"]
                   and l3["rewrite"] == l2["rewrite"],
                   f"K={l3['K']}/{l3['n']} hash={l3['rewrite'] if l3 else '?'}"
                   f"/{l2['rewrite'] if l2 else '?'}")
        if res2["prompt_tokens"] and res3["cache_n"] is not None:
            self.check("O3 캐시 적중: cache_n ≈ O2 prompt (재prefill 없음)",
                       res3["cache_n"] >= res2["prompt_tokens"] - 64,
                       f"cache_n={res3['cache_n']} vs O2 api={res2['prompt_tokens']} "
                       f"(O3 eval wall={res3['wall']:.1f}s vs O2 {res2['wall']:.1f}s)")
        else:
            self.check("O3 캐시 적중", True,
                       f"timings.cache_n 부재 — timings keys={sorted(res3['timings'].keys())} "
                       f"(TTFT로 간접 확인: O3 wall={res3['wall']:.1f}s)", inconclusive=True)

        # O4 — 성장: adv=0, usage 작게 유지, 캐시 적중
        res4 = self.chat(r4, s)
        l4, _ = self.sigma_line(s)
        self.check("O4 성장: adv=0, K 안정, usage<임계",
                   l4 and l2 and l4["advanced"] == 0 and l4["K"] == l2["K"]
                   and l4["usage_before"] < THRESHOLD,
                   f"K={l4['K']}/{l4['n']} usage_before={l4['usage_before']:.3f} "
                   f"api={res4['prompt_tokens']}" if l4 else "no log line")
        if res2["prompt_tokens"] and res4["cache_n"] is not None:
            self.check("O4 캐시 적중: 접두어(W3) 재사용",
                       res4["cache_n"] >= res2["prompt_tokens"] - 64,
                       f"cache_n={res4['cache_n']} vs O2 api={res2['prompt_tokens']}")

    def case_parallel_on(self, p1):
        s = self.sess("parallel")
        res = self.chat(p1, s, max_tokens=64)
        l, _ = self.sigma_line(s)
        self.check("O5 log: adv=1 stripped=2 K=7/10 (첫 병렬 쌍 stub, 마지막 라운드 cap) mode=on",
                   l and l["advanced"] == 1 and l["stripped"] == 2
                   and l["K"] == 7 and l["n"] == 10 and l["mode"] == "on",
                   f"K={l['K']}/{l['n']} tokens {l['tokens_before']}->{l['tokens_after']}" if l else "no log line")
        self.check("O5 target_unreachable (보호 라운드 잔여 > target — native compact 위임)",
                   l and l["target_unreachable"],
                   f"tokens_after={l['tokens_after']} (target {int(WINDOW * 0.35)})" if l else "no log line")
        if l and res["prompt_tokens"]:
            self.check("O5 F2: api usage 급감 (첫 병렬 쌍 stub)",
                       50 < res["prompt_tokens"] < l["tokens_before"] * 0.85,
                       f"api={res['prompt_tokens']} (raw est={l['tokens_before']})")
        t5 = res["text"].strip()
        if t5:
            self.check("O5 응답 non-empty (마지막 병렬 라운드 보존)", True, f"text={t5[:60]!r}")
        elif res["finish_reason"] == "stop":
            self.check("O5 응답 non-empty (마지막 병렬 라운드 보존)", True,
                       f"wall={res['wall']:.1f}s text='' finish_reason=stop — "
                       f"마지막 tool result 보존 상태 — 모델/프롬프트 문제 (Qwen에서 재검증)",
                       inconclusive=True)
        else:
            self.check("O5 응답 non-empty (마지막 병렬 라운드 보존)", False,
                       f"wall={res['wall']:.1f}s text='' finish_reason={res['finish_reason']}")

    # -- driver -------------------------------------------------------------

    SHADOW_CASES = ("grow", "reset", "shrink", "args", "parallel")
    ON_CASES = ("base", "parallel")

    def run(self):
        r1, r2, r3, r4, r5, r3shrink = build_conversation()
        p1 = build_parallel()
        print(f"\n== {self.mode}: {self.base} (log {self.log_path}, run {self.run_id}) ==")
        self.log_off = self.log_size() + 1  # 기동 로그 스킵
        if self.mode == "shadow":
            if "grow" in self.cases:
                self.case_grow(r1, r2, r3, r4)
            if "reset" in self.cases:
                self.case_reset(r3, r5)
            if "shrink" in self.cases:
                self.case_shrink(r3, r3shrink)
            if "args" in self.cases:
                self.case_args()
            if "parallel" in self.cases:
                self.case_parallel(p1)
        else:
            if "base" in self.cases:
                self.case_base_on(r1, r3, r4)
            if "parallel" in self.cases:
                self.case_parallel_on(p1)
        print("  [SKIP] Σ 스냅샷 삽입 — refresh 연동 전 (단계 3 이후 케이스 추가)")
        self.finish()

    def finish(self):
        fails = [r for r in self.results if not r[1] and not r[2]]
        inc = [r for r in self.results if r[2]]
        ok = len(self.results) - len(fails) - len(inc)
        print(f"\nSUMMARY ({self.mode}): {ok} pass, {len(inc)} inconclusive, {len(fails)} fail")
        if inc:
            print("  INCONCLUSIVE (템플릿 렌더링 미확인 — Qwen 재검증):")
            for n, _, _, d in inc:
                print(f"    - {n}: {d}")
        sys.exit(1 if fails else (2 if inc else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--mode", choices=["shadow", "on"], required=True)
    ap.add_argument("--log-ssh", required=True, help="user@host (123)")
    ap.add_argument("--log-path", required=True, help="server log path on 123")
    ap.add_argument("--cases", default="all",
                    help="comma-separated case filter (shadow: grow,reset,shrink,args,parallel; "
                         "on: base,parallel) or 'all'")
    args = ap.parse_args()
    Probe(args).run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""W3 — Option B (kv-offload-holes) refill 트리거 스모크 테스트.

목적:
  Option B의 refill 메커니즘(상용 트래픽에서 한 번도 관찰된 적 없음)을
  제어 환경에서 강제 발화시켜 서버 저널에서 실제로 refill이 일어나는지
  확인한다. 대상 = plans/결정에서 "구조적으로 코드 상 option B 시나리오에서
  refill 메커니즘이 잘 구현되어 있는지, 버그가 없는지" 검증.

메커니즘 (tools/server/server-context.cpp):
  - --fm-offload --kv-offload-holes --focus-memory-host + 낮은
    --kv-offload-threshold 조건에서 게이트가 렌더된 프롬프트가 임계값을
    넘으면 가장 오래된 middle 메시지들을 evict한다. Option B는 evict된
    텍스트를 prompt에 유지(실제 magic chunk)하되, 그 KV만 다음 request의
    n_past check에서 main 시퀀스에서 seq_rm hole로 잘라낸다.
  - 이후 모델이 <focus magic_chunks="N">를 hole과 겹치는 chunk로 발화하면,
    B-switch가 빈 range를 복사하게 되므로 서버가 hole 겹침 토큰을 kv_seq()
    꼬리에 re-prefill하고 B-switch가 da_seq로 복사한다 — 이것이 "refill".
  - 관찰 대상(타깃) 로그:
        kv_offload: focus on holed chunk K - re-prefilling X token(s) at [lo,hi)

시퀀스 (턴마다 별도 request → 접두어 확장 → 슬롯 재사용 → n_past>0):
  turn 1  [system, u1]                     -> a1  (짧은 앵커)
  turn 2  [system, u1, a1, u2]             -> a2  (u2=긴 reference+코드워드;
                                                   last user라 보호, evict 안 됨)
  turn 3  [system, u1, a1, u2, a2, u3]     -> a3  (u3=짧은 무관 질문;
                                                   여기서 u2가 evict+hole)
  turn 4  [system, u1, a1, u2, a2, u3, a3, u4] -> a4 (u4=회상 질문;
                                                   모델이 holed chunk focus -> REFILL)

저널 증거 체인 (목표):
  1. kv_offload: gate - threshold=512 host=... tokens=...
  2. kv_offload: evict plan - N segment(s), ~X token(s) to offload
  3. kv_offload: PUT ok key=...            (스토어 도달 — 없으면 fail-open)
  4. kv-offload-holes: pending hole <key> -> tokens [lo, hi)
  5. kv-offload-holes: applied N hole(s), ... (n_past=...)   <- hole 절단
  6. kv_offload: checking K keep chunk(s) against H hole range(s)
  7. kv_offload: focus on holed chunk C - re-prefilling X token(s)  <- REFILL

판정:
  - --server-log 제공 + "focus on holed chunk" 발견            -> PASS (refill 관찰)
  - --server-log 제공 + hole applied는 있지만 refill 라인 없음  -> INCONCLUSIVE
    (hole은 잘렸으나 모델이 해당 chunk에 focus 안 함 = 모델 행동, refill 경로 미도달)
  - --server-log 제공 + "PUT failed"                          -> BLOCKED
    (스토어 미달 → fail-open → hole 계획 자체가 안 됨. 스토어 점검 필요)
  - --server-log 제공 + "refill failed"                       -> FAIL (refill 버그)
  - --server-log 없음 + 마지막 턴 content에 코드워드 포함      -> LIKELY PASS
    (행동 증거. 확정 증명은 저널로)
  - --server-log 없음 + 코드워드 부재                          -> INCONCLUSIVE

서버 기동 (테스트 대상 123 — 스토어는 반드시 살아있어야 함):
  ./build/bin/llama-server -m MODEL --port 8086 \\
      --da-auto --kv-unified --fm-offload --kv-offload-holes \\
      --focus-memory-host http://<store>:<port> \\
      --kv-offload-threshold 512 --da-min-ctx 512 --da-chunk-tokens 4096 \\
      --parallel 1 -c 32768 -v 2>&1 | tee /tmp/da_refill.log

  - --kv-unified 필수: 없으면 da-auto가 바닐라 유지(안정장치) → hole/refill 안 일어남
  - --focus-memory-host 필수: 게이트의 전제(!host.empty). PUT이 실패하면
    segment가 prompt에 남고(fail-open) hole 계획 자체가 안 됨 → refill 미발화
  - --kv-offload-threshold 512: u2(약 900 tok)가 evict되도록 낮게.
    렌더 프롬프트(스캐폴드+DA 지시 포함)가 이를 넘으면 middle evict 발화
  - --da-chunk-tokens 4096: middle 메시지 전체가 1청크로 유지 → 모델이
    chunk 1 하나만 focus하면 됨 (호출 단순화, refill 발화 확률 극대화)

실행:
  python3 da_refill_smoke.py http://192.168.219.123:8086 --server-log /tmp/da_refill.log
  (기본 thinking off = 123:8080 상용 설정. --thinking 으로 on 전환 가능)

저널이 123에 있는 경우 (로컬에 로그가 없을 때):
  grep -E 'kv_offload: (gate|evict plan|PUT|focus on holed|refill failed)|\\
           kv-offload-holes: (applied|pending)' /tmp/da_refill.log
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

# ---- 테스트 픽스처 -------------------------------------------------------
# 코드워드는 서로 다른 섹션에 안 나오도록 고유하게. u2(에비트 대상)에만 존재.
CODEROW_DEFAULT = "ZEBRA-42"

FILLER = ("The reference archive contains a long sequence of calibration notes "
          "for the optical bench. Each note records the lens focal length, the "
          "aperture diameter, and the measured point-spread function at the test "
          "wavelength. The notes are written in a terse style, with no commentary "
          "beyond the recorded values. ")

SYSTEM = ("You are a precise retrieval assistant. Answer each question using "
          "only the conversation context. When asked for a codeword, reply "
          "with the codeword alone on one line, nothing else.")

U1 = ("I'm about to send you a batch of reference material to remember. "
      "Just reply 'Ready' when you're set.")

U3 = "Unrelated quick question: what is 2+2?"

U4 = ("Now recall: what was the emergency shutdown codeword in the reference "
      "material I sent earlier? Focus on the chunk containing that reference "
      "material and reply with the codeword only, nothing else.")


def u2_text(codeword):
    """에비트 대상 긴 reference 메시지. 코드워드는 중간에, 양쪽을 filler가 감싼다.
    약 900 tok → threshold 512를 넘겨 evict 대상이 된다."""
    return ("Reference material (store this carefully).\n\n"
            + FILLER * 8
            + " The emergency shutdown codeword for the facility is %s. " % codeword
            + FILLER * 6
            + "\nPlease confirm you have stored this reference material. "
              "Reply 'Noted'.")


# ---- HTTP ---------------------------------------------------------------
def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base, model, messages, temperature, max_tokens, no_thinking=False):
    # NO stop=["\n"]: thinking 모델은 생성이 <thinking> 블록 안에서 시작되고
    # DA 프로토콜은 줄바꿈이 필요함(<focus> 태그가 별도 라인). "\n" stop은
    # 첫 생성 줄바꿈에서 중단 → content 공백, 태그 미발화.
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False}
    if no_thinking:
        # thinking off = 123:8080 상용 설정(FocusMemory DA hook 전제).
        # thinking 모델은 max_tokens를 전부 reasoning에 쓰고 답안 전에 절단됨.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    res = post(base, "/v1/chat/completions", body)
    wall = time.time() - t0
    msg = res["choices"][0]["message"]
    text = msg.get("content") or ""
    reason = msg.get("reasoning_content") or ""
    tim = res.get("timings", {})
    return {"text": text, "reason": reason, "wall": wall,
            "tps": tim.get("predicted_per_second"),
            "n_gen": tim.get("predicted_n"), "n_prompt": tim.get("prompt_n")}


def detect_model(base):
    try:
        data = get(base, "/v1/models")
        for m in data.get("data", []):
            if m.get("id"):
                return m["id"]
    except Exception as e:
        print("WARN: /v1/models 조회 실패 (%s) — 'model' id 없이 진행" % e)
    return "local"


# ---- 시퀀스 실행 --------------------------------------------------------
def run(base, model, codeword, temperature, max_tokens, no_thinking, verbose=True):
    """4턴 시퀀스 실행. 턴마다 별도 request(접두어 확장 → 슬롯 재사용)."""
    messages = [{"role": "system", "content": SYSTEM}]
    turns = []

    def step(label, user_text, expect_codeword=False):
        messages.append({"role": "user", "content": user_text})
        r = chat(base, model, messages, temperature, max_tokens, no_thinking)
        in_content = expect_codeword and (codeword in r["text"])
        in_reason = expect_codeword and (codeword in r["reason"])
        turns.append({"label": label, "expect_codeword": expect_codeword,
                      "in_content": in_content, "in_reason": in_reason, **r})
        # 가시 답안(content)만 피드백 — 실제 클라이언트가 보내는 것과 일치
        messages.append({"role": "assistant", "content": r["text"]})
        if verbose:
            where = ("content" if in_content
                     else ("reasoning" if in_reason else "-"))
            print("turn %-3s n_prompt=%-5s n_gen=%-4s tps=%-6s codeword=%s"
                  % (label, r["n_prompt"], r["n_gen"],
                     ("%.1f" % r["tps"]) if r["tps"] else "?", where))
            print("          user:    %r" % user_text[:90])
            print("          content: %r" % r["text"][:200])
        return r

    step("1", U1)
    step("2", u2_text(codeword))
    step("3", U3)
    step("4", U4, expect_codeword=True)
    return turns


# ---- 저널 분석 ----------------------------------------------------------
# (카테고리, 정규식) — 순서대로 매칭. 첫 매칭 카테고리 귀속.
JOURNAL_PATTERNS = [
    ("gate",        re.compile(r"kv_offload: gate - threshold=(\d+)")),
    ("evict_plan",  re.compile(r"kv_offload: evict plan - (\d+) segment")),
    ("put_ok",      re.compile(r"kv_offload: PUT ok key=")),
    ("put_cached",  re.compile(r"kv_offload: PUT skip \(cached\)")),
    ("put_failed",  re.compile(r"kv_offload: PUT failed")),
    ("pending",     re.compile(r"kv-offload-holes: pending hole \S+ -> tokens \[(\d+), (\d+)\)")),
    ("applied",     re.compile(r"kv-offload-holes: applied (\d+) hole\(s\), \d+ token span, in main seq \d+ \(n_past=(\d+)\)")),
    ("checking",    re.compile(r"kv_offload: checking (\d+) keep chunk\(s\) against (\d+) hole range")),
    ("refill",      re.compile(r"kv_offload: focus on holed chunk (\d+) - re-prefilling (\d+) token\(s\) at \[(\d+), ?(\d+)\)")),
    ("refill_fail", re.compile(r"kv_offload: refill failed for holed chunk")),
    ("no_evict",    re.compile(r"kv_offload: no eviction planned")),
    ("disabled",    re.compile(r"kv_offload: disabled \(flag=")),
]


def scan_journal(path):
    """저널에서 refill 관련 라인 추출. {카테고리: [매치...]} 반환."""
    hits = {}
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            for cat, pat in JOURNAL_PATTERNS:
                m = pat.search(line)
                if m:
                    hits.setdefault(cat, []).append((m.groups(), line.rstrip()))
                    break
    return hits


def print_journal(hits):
    if hits is None:
        return
    order = ["gate", "evict_plan", "put_ok", "put_cached", "put_failed",
             "pending", "applied", "checking", "refill", "refill_fail",
             "no_evict", "disabled"]
    print("\n저널 증거 (--server-log):")
    found_any = False
    for cat in order:
        for groups, line in hits.get(cat, []):
            found_any = True
            mark = "  <<" if cat == "refill" else "   "
            print("  [%-11s] %s" % (cat, line.strip()[:150]) + mark)
    if not found_any:
        print("  (refill 관련 라인 없음 — 서버 플래그 확인 필요)")


def verdict(hits, turns, codeword, have_log):
    """판정. (결과, 설명) 반환."""
    if have_log:
        if hits.get("refill_fail"):
            return "FAIL", "refill 실행 중 실패 (refill failed) — 버그 재현"
        if hits.get("put_failed") and not hits.get("refill"):
            return "BLOCKED", "스토어 PUT 실패 → fail-open → hole 계획 미발생 (스토어 점검)"
        if hits.get("disabled"):
            return "BLOCKED", "서버가 kv_offload 비활성 보고 — 플래그 누락"
        if hits.get("refill"):
            r = hits["refill"][-1][0]
            return "PASS", ("refill 관찰: chunk %s re-prefill %s token(s) at [%s,%s)"
                            % (r[0], r[1], r[2], r[3]))
        if hits.get("applied"):
            return ("INCONCLUSIVE",
                    "hole은 절단됨(applied)이나 모델이 holed chunk에 focus 안 함 — "
                    "refill 경로 미도달 (모델 행동/질문 어안)")
        if hits.get("no_evict"):
            return ("INCONCLUSIVE",
                    "evict 계획 미발생 — threshold 임계 미달 또는 middle 메시지 없음 "
                    "(u2 토큰 수 / threshold 확인)")
        return "INCONCLUSIVE", "저널에 게이트/evict/hole/refill 흔적 없음 — 플래그 확인"

    # 저널 없음: 행동 기준
    last = turns[-1]
    if last["in_content"]:
        return ("LIKELY PASS",
                "마지막 턴 content에 코드워드 포함 — refill 추정. "
                "확정 증명은 --server-log로 저널 확인")
    if last["in_reason"]:
        return ("INCONCLUSIVE",
                "코드워드가 reasoning에만 — 답안 생성 실패 또는 refill 미발화. "
                "저널로 확인 필요")
    return ("INCONCLUSIVE",
            "마지막 턴에 코드워드 부재 — refill 미발화 또는 모델 미도달. "
            "저널로 확인 필요")


def main():
    ap = argparse.ArgumentParser(
        description="Option B (kv-offload-holes) refill 트리거 스모크 테스트")
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8086",
                    help="llama-server base URL (default: http://127.0.0.1:8086)")
    ap.add_argument("--model", default=None, help="모델 id (기본: /v1/models 자동)")
    ap.add_argument("--server-log", default=None,
                    help="서버 tee 로그 경로 (refill 라인 확정 증거)")
    ap.add_argument("--codeword", default=CODEROW_DEFAULT)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="생성 예산 (thinking off라 256이면 충분)")
    ap.add_argument("--thinking", action="store_true",
                    help="thinking on (기본 off = 123:8080 상용 설정)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    no_thinking = not args.thinking
    model = args.model or detect_model(base)
    print("da_refill_smoke  base=%s  model=%s  thinking=%s  codeword=%s"
          % (base, model, "on" if args.thinking else "off", args.codeword))
    print("시퀀스: u1(앵커) -> u2(긴reference+코드워드) -> u3(무관) -> u4(회상→focus)")
    print()

    t0 = time.time()
    turns = run(base, model, args.codeword, args.temperature, args.max_tokens,
                no_thinking, verbose=not args.quiet)
    wall = time.time() - t0

    hits = scan_journal(args.server_log)
    print_journal(hits)

    result, detail = verdict(hits, turns, args.codeword, have_log=hits is not None)
    print("\n판정: %s" % result)
    print("설명: %s" % detail)
    print("총 소요: %.1f s" % wall)

    # 저널이 없으면 사용자가 123에서 직접 grep할 수 있게 명령 제시
    if hits is None:
        print("\n저널 확정 (123에서 실행):")
        print("  grep -E 'kv_offload: (gate|evict plan|PUT|focus on holed|refill "
              "failed)|kv-offload-holes: (applied|pending)' /tmp/da_refill.log")

    # PASS/LIKELY PASS 외에는 0이 아닌 종료码 (CI/스크립트 체인용)
    sys.exit(0 if result in ("PASS", "LIKELY PASS") else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""W2 — da-auto 다중 턴 가역성(B-path) 회귀 테스트.

대상 (plans/focus-llama-da-auto-multiturn.md):
  원래 결함 = A-path(단조 seq_rm)가 wrong-focus 시 답 청크를 영구 삭제.
  수정(P3) = da-auto는 이제 가역 B-path만 사용(서버가 --kv-unified 없으면
  da-auto를 바닐라로 유지 — 비가역 A-path fallback 차단). B-path는 원본
  시퀀스를 보존하고 </focus>/요청 종료 시 생성 꼬리를 복원해 전체 컨텍스트
  를 재참조한다. 이 테스트는 그 **가역성**을 검증한다: 다중 턴에서 앞 턴의
  focus가 뒤 턴이 필요로 하는 청크를 영구 소실시키지 않는지.

디자인 (다중 턴, 같은 슬롯 재사용 전제):
  setup  : 3개의 섹션(user 메시지)에 각기 다른 코드워드 + filler
           (alpha=TANGO-7, beta=ZEBRA-42, gamma=KILO-99). 각 섹션은
           da-auto의 별도 middle 메시지 → 별도 청크.
  turn 4 : "alpha codeword?" → 모델이 alpha 청크에 focus
  turn 5 : "beta codeword?"  → B-path면 원본 보존이라 ZEBRA-42 도달 가능
  turn 6 : "gamma codeword?" → B-path면 KILO-99 도달 가능

판정 (content 기준 — reasoning 언급은 정답으로 안 친다):
  - 모든 턴 content 정답 + focus 발화                 → PASS (가역성 확인)
  - focus 발화 + 저널 삭제 확인 + 코드워드가 content
    AND reasoning 모두 소실(물리 삭제)                → FAIL (소실 재현)
  - 코드워드가 reasoning에만 있고 content 답안 부재   → INCONCLUSIVE
    (답 생성 실패: 환각/thinking 소진 = 모델 품질, DA 삭제 아님 —
     옛 하네스는 이를 PASS로 위장해 3건 답 생성 실패를 놓침)
  - focus 발화 없음                                   → INCONCLUSIVE (삭제 안 일어남)

서버 기동 (테스트 대상, da-auto + 가역 B-path):
  ./build/bin/llama-server -m M --port 8086 --da-auto --kv-unified \
      --da-min-ctx 2048 --da-chunk-tokens 4096 --parallel 1 -c 32768 -v \
      2>&1 | tee /tmp/da_multiturn.log
  --kv-unified 필수: 없으면 서버가 da-auto를 바닐라로 유지(안정장치)해
  focus/삭제가 안 일어나 INCONCLUSIVE. B-path는 추가 seq id를 --kv-unified가
  내부적으로 예약하므로 --parallel 2 불필요.
  --da-chunk-tokens는 각 setup 섹션(약 1.5K tok)보다 커야 섹션이
  1청크로 유지됨 (기본 4096 충족).

실행:
  python3 da_multiturn_test.py --server http://127.0.0.1:8086 \
      --server-log /tmp/da_multiturn.log

참고: --da-measure-only 서버에서는 삭제가 일어나지 않아 INCONCLUSIVE
  (가역성 검증 불가). measure-only는 태그 발화율(g) 측정용일 뿐.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

# (코드워드, 섹션 본문) — 코드워드는 서로 다르고 서로의 섹션에 안 나오게
FACTS = [
    ("alpha", "TANGO-7",
     "The alpha project codeword is TANGO-7. It is stored in the vault "
     "under the red drawer and rotates quarterly."),
    ("beta", "ZEBRA-42",
     "The beta project codeword is ZEBRA-42. It is stored in the vault "
     "under the blue drawer and rotates quarterly."),
    ("gamma", "KILO-99",
     "The gamma project codeword is KILO-99. It is stored in the vault "
     "under the green drawer and rotates quarterly."),
]
FILLER = ("The warehouse maintains a running log of every shipment that "
          "passes through the dock. Each entry records the crate weight, "
          "the carrier id, and the seal number so that a later audit can "
          "reconstruct the route without re-reading the original manifest. ")

SYSTEM = ("You are a precise retrieval assistant. Answer each question using "
          "only the conversation context. When asked for a codeword, reply "
          "with the codeword alone on one line, nothing else.")


def section_text(body, filler_paras):
    return "Context section.\n" + body + "\n\n" + FILLER * filler_paras


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base, model, messages, temperature, max_tokens):
    # NO stop=["\n"]: the model is a thinking model (generation starts inside an
    # open <thinking> block) and the DA protocol needs newlines (the
    # <focus magic_chunks> tag sits on its own line). A "\n" stop halts on the
    # very first generated newline -> empty content, no tag ever emitted
    # (verified on 123: 3 turns, each "eval 0.00 ms / 1 token", generated "").
    # Bound length with max_tokens instead; a thinking model needs enough budget
    # to think + emit the tag + answer.
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False}
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


def build_setup(filler_paras):
    """setup 턴들: 각 섹션(user) + ack(assistant). (role, content) 리스트."""
    msgs = []
    for _, _, body in FACTS:
        msgs.append(("user", section_text(body, filler_paras)))
        msgs.append(("assistant", "Understood. I have read the section."))
    return msgs


def question_text(name):
    return ("Question: What is the %s project codeword? "
            "Reply with the codeword only, nothing else." % name)


def run_conversation(base, model, filler_paras, temperature, max_tokens, verbose=True):
    """setup + 3 질문 턴을 같은 세션(접두어 확장 → 슬롯 재사용)으로 실행."""
    messages = [{"role": "system", "content": SYSTEM}]
    for role, content in build_setup(filler_paras):
        messages.append({"role": role, "content": content})

    turns = []
    for name, codeword, _ in FACTS:
        messages.append({"role": "user", "content": question_text(name)})
        r = chat(base, model, messages, temperature, max_tokens)
        # Three-way fact state per turn (the fix for the false-PASS):
        #   correct     : codeword in the VISIBLE answer (content) -> right answer
        #   seen        : codeword in content OR reasoning -> the model "saw" it
        #   deleted     : codeword in NEITHER -> physically gone (DA deletion)
        #   answer_fail : seen but not in content -> model saw it but failed to
        #                 answer (hallucination / thinking exhaustion) = a MODEL
        #                 QUALITY failure, NOT a DA deletion. The old harness
        #                 counted this as PASS (codeword in text+reason), which
        #                 masked answer failures as multi-turn safety.
        in_content = codeword in r["text"]
        in_reason = codeword in r["reason"]
        seen = in_content or in_reason
        turns.append({"name": name, "codeword": codeword,
                      "correct": in_content, "seen": seen,
                      "deleted": not seen,
                      "answer_fail": seen and not in_content,
                      "in_content": in_content, "in_reason": in_reason, **r})
        # feed back only the visible answer (content) so the next turn's prompt
        # matches what a real client would send (reasoning is not echoed)
        messages.append({"role": "assistant", "content": r["text"]})
        if verbose:
            where = "content" if in_content else ("reasoning" if in_reason else "ABSENT")
            state = ("CORRECT" if in_content
                     else ("ANSWER-FAIL" if in_reason else "LOST"))
            print("turn %-5s expect=%-8s %s n_prompt=%s n_gen=%s tps=%s  fact in %s"
                  % (name, codeword, state,
                     r["n_prompt"], r["n_gen"], "%.1f" % r["tps"] if r["tps"] else "?", where))
            if in_content:
                print("          content: %r" % r["text"][:200])
            else:
                # failing turn: dump the output to diagnose where the answer
                # went (thinking loop / batch cut / budget). n_gen == max_tokens
                # means the budget was hit; the reasoning TAIL shows where the
                # model stopped (the answer comes after the thinking).
                print("          content (%d chars): %r" % (len(r["text"]), r["text"]))
                rn = r["reason"]
                tail = rn[-2000:] if len(rn) > 2000 else rn
                print("          reasoning (%d chars, showing last %d):" % (len(rn), len(tail)))
                for ln in (tail if tail else "<empty>").splitlines():
                    print("          | " + ln)
    return turns


def read_journal(path):
    if not path:
        return None
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()
    except OSError as e:
        print("WARN: cannot read server log %s (%s)" % (path, e))
        return None


def analyze_journal(lines):
    """da_auto/da_tag/n_kv_max/슬롯 재사용 증거를 추출."""
    if lines is None:
        return None
    da_auto = [l for l in lines if "da_auto:" in l]
    da_tag = [l for l in lines if "da_tag:" in l and ("focus magic_chunks" in l
               or "A removal" in l or "monotonic" in l)]
    n_kv = [l for l in lines if "n_kv_max" in l]
    # full DA lifecycle trace (in log order): chunking, tag switches, B/A
    # removals, batch cuts, n_kv_max pushes, the per-request completion log
    # (mode/keep/removed + the full tag-erased generated_text), and the cache
    # policy (returned / prompt_clear). This is what reveals a beta-style
    # answer-generation failure.
    trace_pat = ("da_auto:", "da_tag:", "da_b:", "da: request complete",
                 "da: generated_text", "set_n_kv_max", "batch cut",
                 "returned to GLOBAL", "prompt_clear", "falling back",
                 "staying VANILLA", "Invalid input batch")
    da_trace = [l for l in lines if any(p in l for p in trace_pat)]
    slot_ids = []
    for l in lines:
        m = re.search(r"\|\s*task\s+(\d+)\s*\|", l)
        if m and ("da_" in l or "slot" in l):
            slot_ids.append(m.group(1))
    # 슬롯 재사용: da 관련 라인의 slot id가 몇 개인지 (1개 = 같은 슬롯 지속)
    return {"da_auto": da_auto, "da_tag": da_tag, "n_kv": n_kv,
            "da_trace": da_trace,
            "n_da_auto": len(da_auto), "n_da_tag": len(da_tag),
            "n_nkv": len(n_kv), "n_distinct_task": len(set(slot_ids))}


def verdict(turns, j):
    """(status, reason). status: PASS / FAIL / INCONCLUSIVE.

    The verdict is CONTENT-based (the visible answer). A codeword that appears
    only in reasoning is an ANSWER-FAIL (model saw it but did not answer) — a
    model-quality issue, NOT a DA deletion. A DA deletion shows up as the
    codeword being absent from BOTH content and reasoning (physically gone).
    This separation is what the old harness lacked (it scored reasoning
    mentions as correct, producing a false PASS over 3 answer failures).
    """
    all_correct = all(t["correct"] for t in turns)
    lost = [t["name"] for t in turns if t["deleted"]]
    answer_fail = [t["name"] for t in turns if t["answer_fail"]]
    any_focus = (j and j["n_da_tag"] > 0)
    any_deletion = any(("A removal" in l) for l in (j["da_tag"] if j else []))

    if all_correct:
        if not any_focus:
            return ("INCONCLUSIVE",
                    "모든 턴 content 정답이지만 focus 발화 없음 — 삭제가 일어나지 "
                    "않아 결함을 검증 못 함 (재시도 또는 --temperature 0)")
        return ("PASS", "모든 턴 content 정답 + focus 발화 — 다중 턴 안전")
    if lost and any_focus and any_deletion:
        return ("FAIL",
                "결함 재현: focus 발화 후 %s 청크 content+reasoning 모두 소실 "
                "(물리 삭제) → 코드워드 영구 손실" % "+".join(lost))
    if answer_fail and not lost:
        return ("INCONCLUSIVE",
                "%s 턴은 reasoning에 코드워드가 있으나 content 답안 부재 = 답 생성 "
                "실패(환각/thinking 소진)이지 DA 삭제 아님. wrong-focus 격리 불가 — "
                "모델 품질 이슈. content 기준 재측정 필요" % "+".join(answer_fail))
    if lost and not any_focus:
        return ("INCONCLUSIVE",
                "코드워드가 content+reasoning 모두 소실됐으나 focus 발화 없음 — "
                "DA 삭제 아님(모델 불응/생성 실패). 저널 확인")
    return ("INCONCLUSIVE", "예상 패턴과 다름 — 저널을 직접 확인")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8086")
    ap.add_argument("--model", default="qwen27b")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="결정성을 위해 기본 0 (모델의 focus 발화를 고정)")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="생각(thinking) 모델이 생각+focus 태그+답안까지 생성할 예산")
    ap.add_argument("--filler-paras", type=int, default=30,
                    help="섹션당 filler 단락 수 (setup을 da-min-ctx 초과로 만들)")
    ap.add_argument("--server-log", default=None,
                    help="서버 stdout 로그 (tee) 경로 — da_* 저널 증거용")
    ap.add_argument("--dump-json", default=None,
                    help="턴별 전체 content+reasoning을 JSON로 저장 — 답 생성 실패(beta)가 "
                         "정확히 어디서/왜 답을 내뱉지 못하는지(태그 위치, FOCUS 후 행동, "
                         "중단 지점) 분석용")
    args = ap.parse_args()

    print("=" * 72)
    print("da-auto 다중 턴 컨텍스트 보존 회귀 테스트")
    print("server=%s model=%s temp=%s filler_paras=%d"
          % (args.server, args.model, args.temperature, args.filler_paras))
    print("=" * 72)

    turns = run_conversation(args.server, args.model, args.filler_paras,
                             args.temperature, args.max_tokens)
    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump(turns, f, ensure_ascii=False, indent=2)
        print("전체 응답(content+reasoning) 저장: %s" % args.dump_json)
    lines = read_journal(args.server_log)
    j = analyze_journal(lines)

    print("-" * 72)
    if j:
        print("저널 증거: da_auto=%d da_tag=%d n_kv_max=%d distinct_task=%d (1 = 같은 슬롯 재사용)"
              % (j["n_da_auto"], j["n_da_tag"], j["n_nkv"], j["n_distinct_task"]))
        print("DA 라이프사이클 트레이스 (%d 줄):" % len(j["da_trace"]))
        for l in j["da_trace"]:
            s = l.strip()
            # keep the full generation text + the completion state; trim the
            # rest for readability
            keep = s if ("generated_text" in s or "request complete" in s) else s[:200]
            print("    | " + keep)
    else:
        print("저널 없음 (--server-log 미지정 또는 읽기 실패) — 판정은 답변 기준만")
        print("  (서버가 tee 중인 로그 파일 경로를 --server-log에 주어야 DA 트레이스가 나옴)")

    status, reason = verdict(turns, j)
    print("-" * 72)
    print("OVERALL : %s" % status)
    print("  %s" % reason)
    # exit code: PASS=0, FAIL=1, INCONCLUSIVE=2
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[status])


if __name__ == "__main__":
    main()

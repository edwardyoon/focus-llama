#!/usr/bin/env python3
"""W2 — da-auto 다중 턴 컨텍스트 보존 회귀 테스트.

재현 대상 결함 (plans/focus-llama-da-auto-multiturn.md):
  --da-auto는 매 턴 전체 회화를 다시 청킹하고, 모델이 매 턴 발화하는
  <focus magic_chunks="N"> 태그가 A-path 단조 seq_rm으로 해당 턴 질문에만
  필요한 청크를 영구 삭제한다. 다중 턴(같은 슬롯 재사용)에서 이후 턴이
  삭제된 청크의 내용을 필요로 하면 답이 영구 소실된다.

디자인 (다중 턴, 같은 슬롯 재사용 전제):
  setup  : 3개의 섹션(user 메시지)에 각기 다른 코드워드 + filler
           (alpha=TANGO-7, beta=ZEBRA-42, gamma=KILO-99). 각 섹션은
           da-auto의 별도 middle 메시지 → 별도 청크.
  turn 4 : "alpha codeword?" → 모델이 alpha 청크에 focus → beta/gamma 청크 삭제
  turn 5 : "beta codeword?"  → ZEBRA-42여야 하지만 beta 청크 삭제됨 → 소실
  turn 6 : "gamma codeword?" → KILO-99여야 하지만 gamma 청크 삭제됨 → 소실

판정:
  - turn 4 정답(TANGO-7) + turn 5/6 정답            → PASS (다중 턴 안전)
  - turn 4 정답 + (turn 5 또는 6 실패) + 저널에
    focus 발화 + 삭제 확인                            → FAIL (결함 재현)
  - turn 4/5/6 전부 정답 + 저널에 focus 발화 없음    → INCONCLUSIVE
    (모델이 태그를 안 내어 삭제가 안 일어남 — 재시도 또는 temp 하향)

서버 기동 (테스트 대상, da-auto + 실제 삭제):
  ./build/bin/llama-server -m M --port 8086 --da-auto \
      --da-min-ctx 2048 --da-chunk-tokens 4096 --parallel 1 -c 32768 -v \
      2>&1 | tee /tmp/da_multiturn.log
  --parallel 1 필수: path A(단조 seq_rm) 검증 경로 (kv_unified 없이).
  --da-chunk-tokens는 각 setup 섹션(약 1.5K tok)보다 커야 섹션이
  1청크로 유지됨 (기본 4096 충족).

실행:
  python3 da_multiturn_test.py --server http://127.0.0.1:8086 \
      --server-log /tmp/da_multiturn.log

참고: --da-measure-only 서버에서는 삭제가 일어나지 않아 이 테스트는
  항상 PASS (결함 검증 불가). measure-only는 태그 발화율(g) 측정용일 뿐.
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
        # "fact accessible" = codeword appears anywhere the model generated
        # (content answer OR reasoning). A physically-removed KV chunk cannot be
        # reproduced in either, so absence from both = the fact is gone.
        ok = codeword in (r["text"] + r["reason"])
        in_content = codeword in r["text"]
        in_reason = (not in_content) and (codeword in r["reason"])
        turns.append({"name": name, "codeword": codeword, "ok": ok,
                      "in_content": in_content, "in_reason": in_reason, **r})
        # feed back only the visible answer (content) so the next turn's prompt
        # matches what a real client would send (reasoning is not echoed)
        messages.append({"role": "assistant", "content": r["text"]})
        if verbose:
            where = "content" if in_content else ("reasoning" if in_reason else "ABSENT")
            print("turn %-5s expect=%-8s ok=%-5s n_prompt=%s tps=%s  fact in %s"
                  % (name, codeword, "PASS" if ok else "FAIL",
                     r["n_prompt"], "%.1f" % r["tps"] if r["tps"] else "?", where))
            print("          content: %r" % r["text"][:120])
            print("          reason : %r" % r["reason"][:120])
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
    slot_ids = []
    for l in lines:
        m = re.search(r"\|\s*task\s+(\d+)\s*\|", l)
        if m and ("da_" in l or "slot" in l):
            slot_ids.append(m.group(1))
    # 슬롯 재사용: da 관련 라인의 slot id가 몇 개인지 (1개 = 같은 슬롯 지속)
    return {"da_auto": da_auto, "da_tag": da_tag, "n_kv": n_kv,
            "n_da_auto": len(da_auto), "n_da_tag": len(da_tag),
            "n_nkv": len(n_kv), "n_distinct_task": len(set(slot_ids))}


def verdict(turns, j):
    """(status, reason). status: PASS / FAIL / INCONCLUSIVE."""
    t4 = turns[0]["ok"]   # alpha (focus 발화 턴)
    t5 = turns[1]["ok"]   # beta
    t6 = turns[2]["ok"]   # gamma
    any_focus = (j and j["n_da_tag"] > 0)
    any_deletion = any(("A removal" in l) for l in (j["da_tag"] if j else []))

    if t4 and t5 and t6:
        if not any_focus:
            return ("INCONCLUSIVE",
                    "모두 정답이지만 focus 태그 발화 없음 — 삭제가 일어나지 않아 "
                    "결함을 검증하지 못함 (재시도 또는 --temperature 0)")
        return ("PASS", "모든 턴 정답 + focus 발화 — 다중 턴 안전")
    if t4 and not (t5 and t6) and any_focus and any_deletion:
        lost = [t["name"] for t in turns[1:] if not t["ok"]]
        return ("FAIL",
                "결함 재현: focus 발화 후 %s 청크 영구 삭제 → 해당 코드워드 소실"
                % "+".join(lost))
    if not t4:
        return ("FAIL", "alpha 턴(turn 4)조차 실패 — focus가 alpha 청크를 지움 "
                        "또는 모델 불응 (저널 확인)")
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
    args = ap.parse_args()

    print("=" * 72)
    print("da-auto 다중 턴 컨텍스트 보존 회귀 테스트")
    print("server=%s model=%s temp=%s filler_paras=%d"
          % (args.server, args.model, args.temperature, args.filler_paras))
    print("=" * 72)

    turns = run_conversation(args.server, args.model, args.filler_paras,
                             args.temperature, args.max_tokens)
    lines = read_journal(args.server_log)
    j = analyze_journal(lines)

    print("-" * 72)
    if j:
        print("저널 증거:")
        print("  da_auto 라인      : %d" % j["n_da_auto"])
        print("  da_tag(focus) 라인: %d" % j["n_da_tag"])
        print("  n_kv_max 라인     : %d" % j["n_nkv"])
        print("  da 관련 distinct task(슬롯) 수: %d (1 = 같은 슬롯 재사용 확인)"
              % j["n_distinct_task"])
        for l in j["da_auto"][-2:]:
            print("    | %s" % l.strip()[:150])
        for l in j["da_tag"][-3:]:
            print("    | %s" % l.strip()[:150])
        for l in j["n_kv"][-3:]:
            print("    | %s" % l.strip()[:150])
    else:
        print("저널 없음 (--server-log 미지정 또는 읽기 실패) — 판정은 답변 기준만")

    status, reason = verdict(turns, j)
    print("-" * 72)
    print("OVERALL : %s" % status)
    print("  %s" % reason)
    # exit code: PASS=0, FAIL=1, INCONCLUSIVE=2
    sys.exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[status])


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# sigma-strip prove driver — 123의 e2b.service(systemd, port 8180)를
# shadow → on 순으로 재기동하고, 로컬에서 sigma_probe.py(클라이언트 루프
# 시뮬레이터)를 실행한다.
#
# 서버 기동은 systemd(e2b.service)가 담당. 이 스크립트는 mode 스위치
# (env 파일) + restart + health 대기 + probe만 한다:
#
#   Phase 1  shadow -> K 기계 검증 (전진/안정/리셋/byte-identical,
#                본체 미변경 = 응답 usage 원본 유지)
#   Phase 2  on     -> F2 검증 (응답 usage.prompt_tokens 급감)
#
# 로그: /tmp/e2b-test.log (systemd append). probe는 기동 시점 로그
# 오프셋부터 읽으므로 페이즈마다 자동 클린.
#
# 상용 서버(8080/8087/8088)는 건드리지 않음 — e2b.service만 port 8180.
# 종료 후 서비스는 활성 유지 (테스트 서버 상시 띄워두기).
#
# 사용 (로컬, focus-llama/da-probe/ 기준):
#   bash sigma_123_verify.sh
#
# 선택 env:
#   HOST=edwardyoon@192.168.219.123
#   PORT=8180
#
# 전제 (1회성 sudo 설치 — 123에서):
#   sudo cp ~/e2b.service /etc/systemd/system/e2b.service
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now e2b
#   # 스크립트의 비대화형 restart용 (권장):
#   echo 'edwardyoon ALL=(root) NOPASSWD: /usr/bin/systemctl' | \
#       sudo tee /etc/sudoers.d/e2b-test
set -u
cd "$(dirname "$0")"

HOST=${HOST:-edwardyoon@192.168.219.123}
PORT=${PORT:-8180}
SRV_IP=${HOST#*@}  # probe(로컬)가 서버를 부르는 LAN IP
LOG=/tmp/e2b-test.log
SSHOPTS="-o ConnectTimeout=8 -o BatchMode=yes"

setup_hint() {
    cat <<'EOF'
[e2b.service 설치 필요] 123 터미널에서 (sudo 비밀번호 입력):
  sudo cp ~/e2b.service /etc/systemd/system/e2b.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now e2b
  # 스크립트가 비대화형으로 restart하려면 (권장):
  echo 'edwardyoon ALL=(root) NOPASSWD: /usr/bin/systemctl' | sudo tee /etc/sudoers.d/e2b-test
EOF
}

gpu1_mem() {
    ssh $SSHOPTS "$HOST" \
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | awk -F', ' '\$1==1'"
}

set_mode() {  # $1 = shadow|on
    echo "  SIGMA_MODE=$1 -> systemctl restart e2b ..."
    # 1) mode env 파일 작성 (비sudo — home dir)
    ssh $SSHOPTS "$HOST" \
        "mkdir -p ~/.config && echo SIGMA_MODE=$1 > ~/.config/e2b-test.env" || return 1
    # 2) restart (sudo 필요)
    local out
    if out=$(ssh $SSHOPTS "$HOST" "sudo -n systemctl restart e2b 2>&1"); then
        :
    else
        echo "  [FAIL] systemctl restart: $out"
        if ssh $SSHOPTS "$HOST" "systemctl cat e2b >/dev/null 2>&1"; then
            echo "  (유닛은 있음 — 비비밀 sudo 규칙 추가 필요:)"
            echo "   echo 'edwardyoon ALL=(root) NOPASSWD: /usr/bin/systemctl' | sudo tee /etc/sudoers.d/e2b-test"
        else
            setup_hint
        fi
        return 1
    fi
    # 3) health 대기
    local code
    for i in $(seq 1 90); do
        code=$(ssh $SSHOPTS "$HOST" \
            "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health" 2>/dev/null)
        if [ "$code" = "200" ]; then
            echo "  server READY after ${i}s  log=$LOG (append)"
            echo "  GPU1 mem: $(gpu1_mem)"
            return 0
        fi
        if [ $((i % 10)) -eq 0 ]; then echo "  ... still starting (${i}s)"; fi
        sleep 1
    done
    echo "TIMEOUT: not healthy after 90s - last log lines:"
    ssh $SSHOPTS "$HOST" "tail -20 $LOG"
    return 1
}

echo "===================================================================="
echo "sigma-strip prove  (123:$PORT, E2B, e2b.service)"
echo "host : $HOST"
echo "log  : $LOG (systemd append)"
echo "===================================================================="

R1=1; S1="fail"
echo
echo "===================================================================="
echo "Phase 1: shadow - K 기계 검증 (본체 미변경)"
echo "===================================================================="
if set_mode shadow; then
    python3 sigma_probe.py --server "http://$SRV_IP:$PORT" --mode shadow \
        --log-ssh "$HOST" --log-path "$LOG"
    R1=$?
    [ "$R1" -eq 0 ] && S1="pass"
    [ "$R1" -eq 2 ] && S1="warn(INCONCLUSIVE)"
fi

echo
echo "===================================================================="
echo "Phase 2: on - F2 검증 (응답 usage 급감)"
echo "===================================================================="
R2=1; S2="fail"
if set_mode on; then
    python3 sigma_probe.py --server "http://$SRV_IP:$PORT" --mode on \
        --log-ssh "$HOST" --log-path "$LOG"
    R2=$?
    [ "$R2" -eq 0 ] && S2="pass"
    [ "$R2" -eq 2 ] && S2="warn(INCONCLUSIVE)"
fi

echo
echo "===================================================================="
echo "SUMMARY  (pass / warn=INCONCLUSIVE / fail)"
echo "  P1 shadow K-machinery : $S1  (rc=$R1)"
echo "  P2 on F2 usage        : $S2  (rc=$R2)"
echo "  (e2b.service는 on mode로 활성 유지 — GPU1: $(gpu1_mem))"
echo "===================================================================="
FINAL=0
for rc in "$R1" "$R2"; do
    if [ "$rc" -eq 1 ]; then
        FINAL=1
    elif [ "$rc" -eq 2 ] && [ "$FINAL" -ne 1 ]; then
        FINAL=2
    fi
done
exit "$FINAL"

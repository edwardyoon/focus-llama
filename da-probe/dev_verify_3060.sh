#!/usr/bin/env bash
# 123 focus-llama-dev (rebase 브랜치) 검증 드라이버 — 3060 (CUDA_VISIBLE_DEVICES=1)
#
# 페이즈별 클린 서버 + 클린 로그 (smoke는 서버 로그를 file offset 기준으로
# per-request 읽으므로 페이즈 간 로그 섞임 금지):
#
#   P1a DA core  (--da-auto --kv-unified)   -> da_multiturn_test.py  (user-start
#        checkpoint + 캐시 재사용 — message_delimiters 수정의 직접 검증)
#   P1b DA core  (fresh server/log)         -> da_chunking_smoke.py
#   P2  DA marker (--da-prompt-scan)        -> da_dead_marker_smoke.py
#   P3a sigma    (--sigma-strip shadow)     -> sigma_probe.py shadow
#   P3b sigma    (--sigma-strip on)         -> sigma_probe.py on
#   P4  kv offload (--fm-offload --kv-offload-holes --focus-memory-host)
#                                                -> da_refill_smoke.py (offloaded
#        kv fetch = refill)
#   P5  sparse vec (--flash-attn on --sparse-gate-threshold, FOCUS_SPARSE_DEBUG=1)
#                                                -> da_sparse_ab.py capture
#
# 사용 (123에서):
#   bash /home/edwardyoon/focus-llama-dev/da-probe/dev_verify_3060.sh
#
# stdbuf -o0 -e0 필수: smoke가 저널을 per-request 읽는데 block-buffered stdout이
# da_scan:/da_auto: 라인을 버퍼에 갇힘 (da_123_verify.sh 동일).
set -u
cd "$(dirname "$0")/.."

MODEL=/home/edwardyoon/my_model/gemma4/gemma-4-12b-it-UD-Q4_K_XL.gguf
PORT=8086
export CUDA_VISIBLE_DEVICES=1
TS=$(date +%Y%m%d_%H%M%S)
SRV=
TAIL=
SRV_ENV=

cleanup() {
    [ -n "${TAIL:-}" ] && kill "$TAIL" 2>/dev/null
    [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null
}
trap cleanup INT TERM

if [ ! -x ./build/bin/llama-server ]; then
    echo "FATAL: ./build/bin/llama-server not found - build first:"
    echo "  cmake --build build -j \$(nproc) --target llama-server"
    exit 1
fi

start_server() {  # $1 = log, rest = args (SRV_ENV env prefix 허용)
    local log=$1; shift
    echo "  starting server on port $PORT (env: ${SRV_ENV:-none}) ..."
    pkill -f "llama-server .*--port $PORT" 2>/dev/null || true
    sleep 3
    env ${SRV_ENV:-} stdbuf -o0 -e0 ./build/bin/llama-server -m "$MODEL" -ngl 99 \
        --port "$PORT" "$@" > "$log" 2>&1 &
    SRV=$!
    tail -f "$log" &
    TAIL=$!
    local code
    for i in $(seq 1 240); do
        if ! kill -0 "$SRV" 2>/dev/null; then
            kill "$TAIL" 2>/dev/null; TAIL=
            echo "FATAL: server process died during startup - last log lines:"
            tail -20 "$log"
            return 1
        fi
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
        if [ "$code" = "200" ]; then
            echo "server pid=$SRV READY after ${i}s  log=$log"
            return 0
        fi
        [ $((i % 10 == 0)) ] && echo "  ... still loading model (${i}s)"
        sleep 1
    done
    echo "TIMEOUT: server not healthy after 240s - last log lines:"
    tail -20 "$log"
    return 1
}

stop_server() {
    [ -n "${TAIL:-}" ] && kill "$TAIL" 2>/dev/null
    TAIL=
    [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null
    [ -n "${SRV:-}" ] && wait "$SRV" 2>/dev/null
    SRV=
}

echo "===================================================================="
echo "repo head : $(git log --oneline -1 2>/dev/null || echo '(not a git repo)')"
echo "model     : $MODEL"
echo "port      : $PORT   GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "===================================================================="

# ================= P1a: DA core — multiturn =================
LOG1A=/tmp/dev_p1a_multiturn_$TS.log
R1A=1
echo; echo "===================================================================="
echo "P1a: DA core (--da-auto --kv-unified) - multiturn (delimiters fix 검증)"
echo "===================================================================="
if start_server "$LOG1A" --da-auto --kv-unified --da-min-ctx 2048 --da-chunk-tokens 4096 \
    --parallel 1 -c 32768 -ctk q4_0 -ctv q4_0 --jinja --no-webui -v; then
    if grep -q 'da_auto: requires --kv-unified' "$LOG1A"; then
        echo "FATAL: server stayed VANILLA (kv-unified not effective) - see $LOG1A"
    else
        python3 da-probe/da_multiturn_test.py --server "http://127.0.0.1:$PORT" \
            --model local --server-log "$LOG1A"
        R1A=$?
    fi
fi
echo "--- checkpoint evidence (delimiters fix) ---"
grep -E "checkpoint" "$LOG1A" | head -8
stop_server

# ================= P1b: DA core — chunking (fresh log) =================
LOG1B=/tmp/dev_p1b_chunking_$TS.log
R1B=1
echo; echo "===================================================================="
echo "P1b: DA core (fresh server/log) - chunking smoke"
echo "===================================================================="
if start_server "$LOG1B" --da-auto --kv-unified --da-min-ctx 2048 --da-chunk-tokens 4096 \
    --parallel 1 -c 32768 -ctk q4_0 -ctv q4_0 --jinja --no-webui -v; then
    if grep -q 'da_auto: requires --kv-unified' "$LOG1B"; then
        echo "FATAL: server stayed VANILLA (kv-unified not effective) - see $LOG1B"
    else
        python3 da-probe/da_chunking_smoke.py "http://127.0.0.1:$PORT" --log "$LOG1B"
        R1B=$?
    fi
fi
stop_server

# ================= P2: DA marker =================
LOG2=/tmp/dev_p2_marker_$TS.log
R2=1
echo; echo "===================================================================="
echo "P2: DA marker path (--da-prompt-scan) - dead-marker smoke"
echo "===================================================================="
if start_server "$LOG2" --da-prompt-scan --parallel 1 -c 8192 -ctk q4_0 -ctv q4_0 \
    --jinja --no-webui -v; then
    python3 da-probe/da_dead_marker_smoke.py "http://127.0.0.1:$PORT" --log "$LOG2"
    R2=$?
fi
stop_server

# ================= P3a: sigma shadow =================
LOG3A=/tmp/dev_p3a_sigma_shadow_$TS.log
R3A=1
echo; echo "===================================================================="
echo "P3a: sigma-strip shadow"
echo "===================================================================="
if start_server "$LOG3A" --sigma-strip shadow --sigma-strip-window-tokens 30000 \
    --sigma-strip-threshold 0.58 --sigma-strip-target 0.35 \
    -c 32768 -b 2048 -ub 2048 --jinja --no-webui --reasoning off -v; then
    python3 da-probe/sigma_probe.py --server "http://127.0.0.1:$PORT" --mode shadow \
        --log-ssh edwardyoon@192.168.219.123 --log-path "$LOG3A"
    R3A=$?
fi
stop_server

# ================= P3b: sigma on =================
LOG3B=/tmp/dev_p3b_sigma_on_$TS.log
R3B=1
echo; echo "===================================================================="
echo "P3b: sigma-strip on"
echo "===================================================================="
if start_server "$LOG3B" --sigma-strip on --sigma-strip-window-tokens 30000 \
    --sigma-strip-threshold 0.58 --sigma-strip-target 0.35 \
    -c 32768 -b 2048 -ub 2048 --jinja --no-webui --reasoning off -v; then
    python3 da-probe/sigma_probe.py --server "http://127.0.0.1:$PORT" --mode on \
        --log-ssh edwardyoon@192.168.219.123 --log-path "$LOG3B"
    R3B=$?
fi
stop_server

# ================= P4: kv offload + refill =================
LOG4=/tmp/dev_p4_refill_$TS.log
R4=1
echo; echo "===================================================================="
echo "P4: kv offload + offloaded kv fetch (refill) - store: 127.0.0.1:3900"
echo "===================================================================="
if start_server "$LOG4" --da-auto --kv-unified --fm-offload --kv-offload-holes \
    --focus-memory-host http://127.0.0.1:3900 --kv-offload-threshold 512 \
    --da-min-ctx 512 --da-chunk-tokens 4096 \
    --parallel 1 -c 32768 -ctk q4_0 -ctv q4_0 --jinja --no-webui -v; then
    python3 da-probe/da_refill_smoke.py "http://127.0.0.1:$PORT" --server-log "$LOG4"
    R4=$?
fi
stop_server

# ================= P5: sparse vec =================
LOG5=/tmp/dev_p5_sparse_$TS.log
R5=1
echo; echo "===================================================================="
echo "P5: sparse vec (flash-attn, FOCUS_SPARSE_DEBUG=1)"
echo "===================================================================="
SRV_ENV="FOCUS_SPARSE_DEBUG=1"
if start_server "$LOG5" --da-auto --kv-unified \
    --da-scaffold-file tools/server/da-scaffold.txt \
    --da-min-ctx 2048 --da-chunk-tokens 4096 --da-tail-keep 20480 \
    --sparse-gate-threshold 80 --flash-attn on \
    --parallel 1 -c 49152 -ctk q4_0 -ctv q4_0 --jinja --no-webui -v; then
    python3 da-probe/da_sparse_ab.py --mode capture --server "http://127.0.0.1:$PORT" \
        --sparse-log "$LOG5" --target-tokens 38000 --tail-keep 20480
    R5=$?
fi
SRV_ENV=
stop_server

echo
echo "===================================================================="
echo "SUMMARY  (0=pass, 1=fail, 2=inconclusive)"
echo "  P1a DA core multiturn : $R1A"
echo "  P1b DA core chunking  : $R1B"
echo "  P2  DA marker         : $R2"
echo "  P3a sigma shadow      : $R3A"
echo "  P3b sigma on          : $R3B"
echo "  P4  kv offload/refill : $R4"
echo "  P5  sparse vec        : $R5"
echo "logs: $LOG1A $LOG1B $LOG2 $LOG3A $LOG3B $LOG4 $LOG5"
echo "===================================================================="
echo "send me: this script's stdout + the log files"

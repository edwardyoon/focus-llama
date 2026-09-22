#!/usr/bin/env bash
# 123 verification driver - DA x auto-compact (P1 scanner + P2 chunking + placement)
#
# Three phases, each with a clean server + clean log (journal evidence must
# not mix: the smokes read the server log per request by file offset):
#
#   Phase 1  marker path  (--da-prompt-scan)
#            -> da_dead_marker_smoke.py            (P1 dead-marker defense)
#   Phase 2a auto path    (--da-auto --kv-unified, default 2048 target)
#            -> da_multiturn_test.py               (P2 placement fix, 3 turns)
#   Phase 2b auto path    (same flags, fresh log)
#            -> da_chunking_smoke.py               (P2 packing/hard_cap/prose/perf)
#
# stdbuf -o0 -e0 is REQUIRED: the smokes read the journal per request and
# block-buffered stdout (8KB) hides the da_scan:/da_auto: lines until the
# buffer flushes -> false "no journal line" failures.
#
# The server log is streamed live to the terminal (tail -f) and progress is
# printed every 10s while the model loads, so startup is never silent. A
# dead server process is detected immediately instead of polling 240s.
# Ctrl-C cleans up the server + tail (trap).
#
# Prereq: git pull + cmake --build build --target llama-server (on 123)
#
# Usage (123, anywhere):
#   bash /path/to/focus-llama/da-probe/da_123_verify.sh [model.gguf]
#
# Optional env:
#   PORT=8086                          (default 8086)
#   SPEC_ARGS="--draft-model X --draft-max 4"   # include spec MTP = the 8080
#                                               # production decode path
#
# Takes over $PORT for the duration; kills all its own servers at the end.
set -u
cd "$(dirname "$0")/.."

MODEL=${1:-/home/edwardyoon/my_model/qwen3.8/Qwen3.8-27B-AD-Q6_K.gguf}
PORT=${PORT:-8086}
TS=$(date +%Y%m%d_%H%M%S)
LOG1=/tmp/da_verify_p1_$TS.log
LOG2A=/tmp/da_verify_p2a_$TS.log
LOG2B=/tmp/da_verify_p2b_$TS.log
SPEC_ARGS=${SPEC_ARGS:-}
SRV=
TAIL=

cleanup() {
    [ -n "${TAIL:-}" ] && kill "$TAIL" 2>/dev/null
    [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null
}
trap cleanup INT TERM

if [ ! -x ./build/bin/llama-server ]; then
    echo "FATAL: ./build/bin/llama-server not found - build first:"
    echo "  cmake --build build --target llama-server"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "FATAL: model not found: $MODEL"
    exit 1
fi

start_server() {  # $1 = log, rest = extra args
    echo "  killing any existing server on port $PORT, then starting..."
    pkill -f "llama-server .*--port $PORT" 2>/dev/null || true
    sleep 3
    stdbuf -o0 -e0 ./build/bin/llama-server -m "$MODEL" -ctk q4_0 -ctv q4_0 \
        --port "$PORT" $SPEC_ARGS "$@" > "$1" 2>&1 &
    SRV=$!
    tail -f "$1" &
    TAIL=$!
    local code
    for i in $(seq 1 240); do
        if ! kill -0 "$SRV" 2>/dev/null; then
            kill "$TAIL" 2>/dev/null; TAIL=
            echo "FATAL: server process died during startup - last log lines:"
            tail -20 "$1"
            return 1
        fi
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
        if [ "$code" = "200" ]; then
            echo "server pid=$SRV READY after ${i}s  log=$1"
            return 0
        fi
        [ $((i % 10 == 0)) ] && echo "  ... still loading model (${i}s)"
        sleep 1
    done
    echo "TIMEOUT: server not healthy after 240s - last log lines:"
    tail -20 "$1"
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
echo "repo head: $(git log --oneline -1 2>/dev/null || echo '(not a git repo)')"
echo "model : $MODEL"
echo "port  : $PORT   spec args: ${SPEC_ARGS:-(none)}"
echo "===================================================================="
echo
echo "===================================================================="
echo "Phase 1: marker path (--da-prompt-scan) - P1 dead-marker smoke"
echo "===================================================================="
R1=1
if start_server "$LOG1" --da-prompt-scan --parallel 1 -c 8192 -v; then
    python3 da-probe/da_dead_marker_smoke.py "http://127.0.0.1:$PORT" --log "$LOG1"
    R1=$?
fi
stop_server

echo
echo "===================================================================="
echo "Phase 2a: auto path (--da-auto --kv-unified) - P2 multiturn"
echo "===================================================================="
R2=1
if start_server "$LOG2A" --da-auto --kv-unified --da-min-ctx 2048 --parallel 1 -c 65536 -v; then
    if grep -q 'da_auto: requires --kv-unified' "$LOG2A"; then
        echo "FATAL: server stayed VANILLA (kv-unified not effective) - see $LOG2A"
    else
        python3 da-probe/da_multiturn_test.py --server "http://127.0.0.1:$PORT" \
            --model local --server-log "$LOG2A"
        R2=$?
    fi
fi
stop_server

echo
echo "===================================================================="
echo "Phase 2b: auto path (fresh log) - P2 chunking smoke"
echo "===================================================================="
R3=1
if start_server "$LOG2B" --da-auto --kv-unified --da-min-ctx 2048 --parallel 1 -c 65536 -v; then
    if grep -q 'da_auto: requires --kv-unified' "$LOG2B"; then
        echo "FATAL: server stayed VANILLA (kv-unified not effective) - see $LOG2B"
    else
        python3 da-probe/da_chunking_smoke.py "http://127.0.0.1:$PORT" --log "$LOG2B"
        R3=$?
    fi
fi
stop_server

echo
echo "===================================================================="
echo "SUMMARY  (0=pass, 1=fail, 2=inconclusive)"
echo "  P1 dead-marker : $R1"
echo "  P2 multiturn   : $R2"
echo "  P2 chunking    : $R3"
echo "logs: $LOG1"
echo "      $LOG2A"
echo "      $LOG2B"
echo "===================================================================="
echo "send me: this script's stdout + the three log files"

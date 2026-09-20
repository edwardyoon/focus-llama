#!/usr/bin/env bash
# run_probe.sh — build + test + report for da_probe (Phase 1a static-mask probe)
#
# Usage:
#   da-probe/run_probe.sh <model.gguf> [max_tokens] [--ctx N]
#
# What it does:
#   1. build da_probe against the repo's libllama (cmake Release if build/ is missing)
#   2. run diagnostics: --render (model's own chat template), --thinkscan (thinking tags)
#   3. run the full probe: baseline / masked-A / masked-C / keep-A vs rm-A
#   4. write a consolidated report to da-probe/reports/report_<ts>.txt
#
# da_probe exit codes:
#   0 = OVERALL PASS (pure-attention model, removal works, all checks green)
#   1 = FAIL (a behavioral/logits check failed)
#   2 = usage/build error
#   3 = SEQ_RM REJECTED — the memory backend does not support middle-range
#       removal (expected on hybrid/SSM models such as qwen35). This is a
#       valid architecture-gate result, not a crash.
#
# Works on macOS (Metal, libllama.dylib) and Linux (libllama.so).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${1:-}"
[ $# -gt 0 ] && shift
EXTRA_ARGS=("$@")

# --- model resolution: explicit arg, else a single *.gguf in common locations ---
if [ -z "$MODEL" ]; then
    FOUND=()
    for d in "$HOME/models" /root/models /data/models; do
        [ -d "$d" ] || continue
        while IFS= read -r f; do FOUND+=("$f"); done \
            < <(find "$d" -maxdepth 2 -name '*.gguf' 2>/dev/null)
    done
    if [ "${#FOUND[@]}" -eq 1 ]; then
        MODEL="${FOUND[0]}"
        echo "[model] auto-detected: $MODEL"
    elif [ "${#FOUND[@]}" -gt 1 ]; then
        echo "ERROR: multiple GGUF files found, specify one explicitly:"
        printf '  %s\n' "${FOUND[@]}"
        exit 2
    else
        echo "ERROR: no model given and no *.gguf found in ~/models /root/models /data/models"
        echo "usage: $0 <model.gguf> [max_tokens] [--ctx N]"
        exit 2
    fi
fi
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; exit 2; }

# --- build ---
# a C++ driver is required (a C driver like `cc` compiles .cpp but does not
# link the C++ standard library)
if [ -z "${CXX:-}" ]; then
    for c in c++ g++ clang++; do
        command -v "$c" >/dev/null 2>&1 && { CXX="$c"; break; }
    done
fi
[ -n "${CXX:-}" ] || { echo "FATAL: no C++ compiler found (c++/g++/clang++)"; exit 2; }
echo "[build] compiler: $CXX"
LIB=""
for c in "$REPO/build/bin/libllama.so" "$REPO/build/bin/libllama.dylib"; do
    [ -f "$c" ] && { LIB="$c"; break; }
done
if [ -z "$LIB" ]; then
    echo "[build] libllama not found — cmake configure + build (Release)..."
    cmake -S "$REPO" -B "$REPO/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF || exit 2
    cmake --build "$REPO/build" --target llama -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" || exit 2
    for c in "$REPO/build/bin/libllama.so" "$REPO/build/bin/libllama.dylib"; do
        [ -f "$c" ] && { LIB="$c"; break; }
    done
fi
[ -n "$LIB" ] || { echo "FATAL: libllama build failed"; exit 2; }
echo "[build] libllama: $LIB"

echo "[build] compiling da_probe..."
"$CXX" -O2 -std=c++17 "$SCRIPT_DIR/da_probe.cpp" -o "$SCRIPT_DIR/da_probe" \
    -I "$REPO/include" -I "$REPO/ggml/include" "$LIB" \
    -Wl,-rpath,"$(dirname "$LIB")" || exit 2

# --- run + report ---
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SCRIPT_DIR/reports"
REPORT="$SCRIPT_DIR/reports/report_${TS}.txt"

{
    echo "=== da_probe report ==="
    echo "time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "host : $(hostname)"
    echo "repo : $REPO @ $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "model: $MODEL"
    echo "args : ${EXTRA_ARGS[*]:-}"
    echo
    echo "----- --render (model's own chat template) -----"
    "$SCRIPT_DIR/da_probe" "$MODEL" --render --quiet
    echo
    echo "----- --thinkscan (thinking tag detection) -----"
    "$SCRIPT_DIR/da_probe" "$MODEL" --thinkscan --quiet
    echo
    echo "----- main run -----"
    "$SCRIPT_DIR/da_probe" "$MODEL" --quiet ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    RC=$?
    echo
    echo "exit code : $RC"
    case $RC in
        0) echo "VERDICT   : OVERALL PASS" ;;
        1) echo "VERDICT   : FAIL — see log above" ;;
        3) echo "VERDICT   : SEQ_RM REJECTED (hybrid/SSM gate: middle-range removal unsupported -> KQ mask-injection path required)" ;;
        *) echo "VERDICT   : ERROR (exit $RC) — see log above" ;;
    esac
} > "$REPORT" 2>&1

echo "----------------------------------------------------------------"
sed -n '/^----- main run -----/,$p' "$REPORT"
echo "----------------------------------------------------------------"
echo "report : $REPORT"
echo "Send this file back (or paste the section above)."
exit "$RC"

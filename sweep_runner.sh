#!/usr/bin/env bash
# Drive the full dual-stream multimodal benchmark sweep.
#
#   5 scenarios × { none × 1, static × {1,2,4,8,16}, continuous × {1,2,4,8,16} } ≈ 55 runs.
#   Reuses ONE running vLLM server (started separately via run_server.sh).
#
# Outputs:
#   results/run_<scenario>_<mode>_B<n>.json
#   results/timeline_<scenario>_3way.png   (3-panel timeline)
#   results/sweep_<scenario>_<metric>.png  (batch-size sweep curves)
#
# Use:
#   bash run_server.sh   # in another shell — must already be up before this runs
#   bash sweep_runner.sh
#
# Optional env vars:
#   SCENARIOS="mixed_3agent pure_proactive"   # subset (default: all 5)
#   MODES="continuous"                         # subset (default: none static continuous)
#   BATCH_SIZES="1 2 4 8 16"                   # subset (default: 1 2 4 8 16)
#   HOST=localhost  PORT=8901  MODEL=qwen3-omni
#   MAX_NUM_SEQS=32   # for the record only; the server's true cap
#   DRY_RUN=1                                  # print commands, don't execute
#
set -uo pipefail
cd "$(dirname "$0")"

HOST="${HOST:-localhost}"
PORT="${PORT:-8901}"
MODEL="${MODEL:-qwen3-omni}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
SEED="${SEED:-7}"

# Resolve the same python as the conda env that ships vllm_omni; fall back to PATH.
PY="${PY:-/home/davidchang/miniconda3/envs/vllm_omni/bin/python}"
if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi

SCENARIOS="${SCENARIOS:-pure_proactive pure_agent mixed_1agent mixed_3agent burst24}"
MODES="${MODES:-none static continuous}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"

mkdir -p results logs

run() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry] $*"
        return 0
    fi
    "$@"
}

# Server up?
if ! curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM server is not responding at http://${HOST}:${PORT}/v1/models"
    echo "       Start it first: bash run_server.sh   (then wait for 'Application startup complete')"
    exit 1
fi
echo "[sweep] server OK at http://${HOST}:${PORT}"

# Warmup: send a couple of multimodal requests so the JIT compile cost for the
# audio encoder + image embed (~6-8 s on first call) doesn't get charged to the
# first measured run. A single text-only warmup is NOT sufficient — Qwen-Omni's
# mm encoder triggers a separate set of JIT compilations.
echo "[sweep] warmup (multimodal) ..."
run "$PY" cabin_demo.py --scenario pure_proactive --mode continuous --batch-size 4 \
    --max-num-seqs "$MAX_NUM_SEQS" --seed 999 \
    --out results/_warmup.json >/dev/null 2>&1
rm -f results/_warmup.json

run_one() {
    local scenario="$1" mode="$2" B="$3"
    local out="results/run_${scenario}_${mode}_B${B}.json"
    if [ -e "$out" ] && [ "${OVERWRITE:-0}" != "1" ]; then
        echo "[sweep] skip (exists)  $out"
        return 0
    fi
    echo "[sweep] -> $out"
    run "$PY" cabin_demo.py \
        --scenario "$scenario" \
        --mode "$mode" \
        --batch-size "$B" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --host "$HOST" --port "$PORT" --model "$MODEL" \
        --seed "$SEED" \
        --out "$out" \
        >> "logs/sweep_${scenario}.log" 2>&1
}

echo "[sweep] running matrix ..."
for SC in $SCENARIOS; do
    : > "logs/sweep_${SC}.log"
    for MODE in $MODES; do
        if [ "$MODE" = "none" ]; then
            run_one "$SC" "$MODE" 1
        else
            for B in $BATCH_SIZES; do
                run_one "$SC" "$MODE" "$B"
            done
        fi
    done
done

echo "[sweep] rendering 3-panel timelines (one per scenario) ..."
for SC in $SCENARIOS; do
    # pick representative B for each mode: none B=1, static B=8, continuous B=8 (fall back if missing)
    P_NONE="results/run_${SC}_none_B1.json"
    P_STATIC="results/run_${SC}_static_B8.json"
    [ -f "$P_STATIC" ] || P_STATIC="results/run_${SC}_static_B4.json"
    [ -f "$P_STATIC" ] || P_STATIC="results/run_${SC}_static_B2.json"
    P_CONT="results/run_${SC}_continuous_B8.json"
    [ -f "$P_CONT" ] || P_CONT="results/run_${SC}_continuous_B4.json"
    panels=""
    [ -f "$P_NONE" ]   && panels="$panels $P_NONE"
    [ -f "$P_STATIC" ] && panels="$panels $P_STATIC"
    [ -f "$P_CONT" ]   && panels="$panels $P_CONT"
    if [ -n "$panels" ]; then
        run "$PY" plot_timeline.py --panels $panels \
            --out "results/timeline_${SC}_3way.png" \
            --title "Cockpit dual-stream — ${SC} — none vs static vs continuous"
    fi
done

echo "[sweep] rendering batch-size sweep curves (per scenario per metric) ..."
for SC in $SCENARIOS; do
    for METRIC in ttft e2e queue_wait throughput; do
        files=$(ls results/run_${SC}_*.json 2>/dev/null || true)
        if [ -n "$files" ]; then
            run "$PY" plot_sweep.py --scenario "$SC" --metric "$METRIC" \
                --in $files \
                --out "results/sweep_${SC}_${METRIC}.png"
        fi
    done
done

echo "[sweep] done."

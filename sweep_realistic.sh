#!/usr/bin/env bash
# Drive cabin_solo_prod + cabin_family_prod (production sensor rates) × 5 modes
# on a CX1-throttled 5090, plus the conservative variants as comparison.
#
# Matrix:
#   First batch (PROD = ~3.8 req/s baseline):
#     cabin_solo_prod   × {none, static, static_vip, continuous, continuous_pri}  = 5 runs
#     cabin_family_prod × same                                                       = 5 runs
#   Second batch (CONS = ~2.0 req/s baseline):
#     cabin_solo        × 5 modes = 5
#     cabin_family      × 5 modes = 5
#
# All runs at the CX1-throttle clock (do `bash throttle_cx1.sh` first).
# Total estimated wallclock: ~60-90 min depending on saturation.
set -uo pipefail
cd "$(dirname "$0")"

HOST="${HOST:-localhost}"
PORT="${PORT:-8901}"
MODEL="${MODEL:-qwen3-omni}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
SEED="${SEED:-7}"
PY="${PY:-/home/davidchang/miniconda3/envs/vllm_omni/bin/python}"
mkdir -p results logs

SCENARIOS="${SCENARIOS:-cabin_solo_prod cabin_family_prod cabin_solo cabin_family}"
MODES="${MODES:-none static static_vip continuous continuous_pri}"

run() {
    if [ "${DRY_RUN:-0}" = "1" ]; then echo "[dry] $*"; return 0; fi
    "$@"
}

run_one() {
    local scenario="$1" mode="$2" B="$3"
    local out="results/${scenario}_${mode}.json"
    if [ -e "$out" ] && [ "${OVERWRITE:-0}" != "1" ]; then
        echo "[sweep] skip (exists): $out"
        return 0
    fi
    echo ""
    echo "================ $scenario × $mode (B=$B) ================"
    local start=$(date +%s)
    run "$PY" cabin_demo.py \
        --scenario "$scenario" \
        --mode "$mode" \
        --batch-size "$B" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --host "$HOST" --port "$PORT" --model "$MODEL" \
        --seed "$SEED" \
        --out "$out" \
        > "logs/${scenario}_${mode}.log" 2>&1
    local ec=$?
    local elapsed=$(( $(date +%s) - start ))
    if [ $ec -ne 0 ]; then echo "[sweep] FAILED $scenario/$mode (exit=$ec)"; return $ec; fi
    "$PY" -c "
import json
d=json.load(open('${out}'))
s=d['summary']
def ms(k):
    v=s.get(k); return ('—' if v is None or v!=v else f'{v*1000:.0f}ms')
print(f'  ({${elapsed}}s) reqs={s[\"n_requests_total\"]} errors={s[\"n_errors\"]} busy={s.get(\"busy_span_s\",0):.0f}s')
print(f'  TTFT  p50: inter={ms(\"interactive_ttft_p50_s\")} agent={ms(\"agent_ttft_p50_s\")} pro={ms(\"proactive_ttft_p50_s\")}')
print(f'  TTFT  p95: inter={ms(\"interactive_ttft_p95_s\")} agent={ms(\"agent_ttft_p95_s\")}')
print(f'  e2e   p50: inter={ms(\"interactive_e2e_p50_s\")} pro={ms(\"proactive_e2e_p50_s\")}')
"
}

if ! curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM server not at http://${HOST}:${PORT}"; exit 1
fi
echo "[sweep] server OK"
echo "[sweep] current throttle state:"
bash throttle_cx1.sh status 2>/dev/null | tail -2 || true

# Warmup
run "$PY" cabin_demo.py --scenario pure_proactive --mode continuous --batch-size 4 \
    --max-num-seqs "$MAX_NUM_SEQS" --seed 999 --out results/_warmup.json > /dev/null 2>&1
rm -f results/_warmup.json

for SC in $SCENARIOS; do
    # Scenario implies total_cap (read from spec via cabin_demo). For the
    # CLI --batch-size we set 6 for solo, 8 for family; scenario override
    # supersedes via total_in_flight_cap. None mode forces B=1 internally.
    case "$SC" in
        *family*) B=8 ;;
        *)        B=6 ;;
    esac
    for MODE in $MODES; do
        run_one "$SC" "$MODE" "$B"
    done
done

echo ""
echo "[sweep] done"

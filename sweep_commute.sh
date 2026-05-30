#!/usr/bin/env bash
# Drive the commute_run × 5 batching modes benchmark on a CX1-throttled 5090.
#
# Plan v2 (REPORT_CX1_EQUIV) matrix:
#   R1 = 5090 native + continuous_pri  (best-case headroom)
#   R2 = CX1 throttle + none           (worst-case serial)
#   R3 = CX1 throttle + static
#   R4 = CX1 throttle + static_vip
#   R5 = CX1 throttle + continuous
#   R6 = CX1 throttle + continuous_pri  (target system)
#
# Throttle = sudo nvidia-smi -lmc 810 -lgc 745,745 → measured BW ≈ 68 GB/s
# (CX1 spec is 154 GB/s; our throttle is 2.26× stricter so any continuous-batching
#  gain we measure is a CONSERVATIVE lower bound for real CX1).
#
# Each run ≈ 3 min. Total wallclock ~ 20-25 min including server warmup.
set -uo pipefail
cd "$(dirname "$0")"

HOST="${HOST:-localhost}"
PORT="${PORT:-8901}"
MODEL="${MODEL:-qwen3-omni}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
SEED="${SEED:-7}"
B="${B:-6}"                                  # client-side batch cap (user spec)
PY="${PY:-/home/davidchang/miniconda3/envs/vllm_omni/bin/python}"
mkdir -p results logs

run() {
    if [ "${DRY_RUN:-0}" = "1" ]; then echo "[dry] $*"; return 0; fi
    "$@"
}

run_one() {
    local label="$1" mode="$2" Bcur="$3"
    local out="results/commute_${label}.json"
    if [ -e "$out" ] && [ "${OVERWRITE:-0}" != "1" ]; then
        echo "[commute] skip (exists): $out"
        return 0
    fi
    echo ""
    echo "==============================================="
    echo "[commute] $label  (mode=$mode, B=$Bcur)"
    echo "==============================================="
    run "$PY" cabin_demo.py \
        --scenario commute_run \
        --mode "$mode" \
        --batch-size "$Bcur" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --host "$HOST" --port "$PORT" --model "$MODEL" \
        --seed "$SEED" \
        --out "$out" \
        > "logs/commute_${label}.log" 2>&1
    local ec=$?
    if [ $ec -ne 0 ]; then
        echo "[commute] FAILED label=$label exit=$ec"; return $ec
    fi
    # quick summary print
    "$PY" -c "
import json,sys
d=json.load(open('${out}'))
s=d['summary']
def ms(k):
    v=s.get(k); return ('—' if v is None or v!=v else f'{v*1000:.0f}ms')
print(f'  TTFT p50 inter={ms(\"interactive_ttft_p50_s\")} agent={ms(\"agent_ttft_p50_s\")} pro={ms(\"proactive_ttft_p50_s\")}')
print(f'  e2e  p50 inter={ms(\"interactive_e2e_p50_s\")} agent={ms(\"agent_e2e_p50_s\")} pro={ms(\"proactive_e2e_p50_s\")}')
print(f'  reqs total={s[\"n_requests_total\"]} errors={s[\"n_errors\"]} busy={s.get(\"busy_span_s\",0):.1f}s thru={s.get(\"busy_output_tok_per_s\",0):.0f}tok/s decode={s.get(\"mean_decode_tok_per_s\",0):.1f}tok/s')
"
}

# Sanity
if ! curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM server not at http://${HOST}:${PORT}"; exit 1
fi
echo "[commute] server OK"

# ---- R1: native 5090 (release throttle first) ----
if [ "${SKIP_R1:-0}" != "1" ]; then
    echo "[commute] releasing GPU clocks for R1 (native 5090) ..."
    bash throttle_cx1.sh release > /dev/null 2>&1 || true
    # warmup at full speed
    run "$PY" cabin_demo.py --scenario pure_proactive --mode continuous --batch-size 4 \
        --max-num-seqs "$MAX_NUM_SEQS" --seed 999 --out results/_warmup.json > /dev/null 2>&1
    rm -f results/_warmup.json
    run_one "R1_native_continuous_pri" continuous_pri "$B"
fi

# ---- R2..R6: throttle on ----
echo ""
echo "[commute] applying CX1 throttle ..."
bash throttle_cx1.sh > logs/throttle.log 2>&1
tail -4 logs/throttle.log
# warmup at throttled rate
run "$PY" cabin_demo.py --scenario pure_proactive --mode continuous --batch-size 4 \
    --max-num-seqs "$MAX_NUM_SEQS" --seed 999 --out results/_warmup.json > /dev/null 2>&1
rm -f results/_warmup.json

run_one "R2_throttle_none"            none            1
run_one "R3_throttle_static"          static          "$B"
run_one "R4_throttle_static_vip"      static_vip      "$B"
run_one "R5_throttle_continuous"      continuous      "$B"
run_one "R6_throttle_continuous_pri"  continuous_pri  "$B"

echo ""
echo "[commute] rendering 5-panel timeline (throttle runs) ..."
run "$PY" plot_timeline.py --panels \
    results/commute_R2_throttle_none.json \
    results/commute_R3_throttle_static.json \
    results/commute_R4_throttle_static_vip.json \
    results/commute_R5_throttle_continuous.json \
    results/commute_R6_throttle_continuous_pri.json \
    --out results/timeline_commute_throttle_5way.png \
    --title "commute_run on CX1-throttled 5090 (BW≈68 GB/s) — 5 scheduling modes" \
    --share-x

echo "[commute] done. all 6 JSONs in results/commute_*.json"

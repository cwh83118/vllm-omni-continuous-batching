#!/usr/bin/env bash
# OPTIONAL: a more rigorous throughput/latency sweep using vLLM's built-in
# `vllm bench serve`, against the SAME running server, at several concurrency levels.
#
# The cabin_demo.py / burst runs already tell the story; this is here if you want
# clean TTFT/TPOT/throughput curves vs. concurrency from the standard benchmark.
#
# Run the server first (e.g. `MAX_NUM_SEQS=16 bash run_server.sh`), then:
#   bash bench_sweep.sh
# Repeat with MAX_NUM_SEQS=1 to get the "no continuous batching" baseline.
#
# Output: results/bench_sweep_seqs<N>.csv

# Run with the project's Python env active (e.g. `conda activate vllm_omni`).
set -uo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8901}"
MODEL="${MODEL:-qwen3-omni}"
HOST="${HOST:-localhost}"
# detect server's max_num_seqs from its log if present, else "unknown"
SEQS="${MAX_NUM_SEQS:-$(grep -oE "'max_num_seqs': [0-9]+" logs/server_seqs*.log 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo unknown)}"
OUT="results/bench_sweep_seqs${SEQS}.csv"
mkdir -p results
echo "concurrency,num_prompts,req_throughput,output_throughput,mean_ttft_ms,p99_ttft_ms,mean_tpot_ms,p99_tpot_ms" > "$OUT"

for CC in 1 2 4 8 16 32; do
  NP=$(( CC * 8 ))
  echo "=== concurrency=$CC  num_prompts=$NP ==="
  JSON=$(vllm bench serve \
      --backend openai-chat --endpoint /v1/chat/completions \
      --host "$HOST" --port "$PORT" --model "$MODEL" \
      --dataset-name random --random-input-len 64 --random-output-len 160 \
      --num-prompts "$NP" --max-concurrency "$CC" \
      --percentile-metrics ttft,tpot --metric-percentiles 99 \
      --save-result --result-dir /tmp --result-filename bench_tmp.json 2>&1 | tee /dev/stderr | tail -1)
  python3 - "$CC" "$NP" "$OUT" <<'PY'
import json, sys
cc, np_, out = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open("/tmp/bench_tmp.json"))
    row = [cc, np_,
           round(d.get("request_throughput", 0), 2),
           round(d.get("output_throughput", 0), 1),
           round(d.get("mean_ttft_ms", 0), 1),
           round(d.get("p99_ttft_ms", 0), 1),
           round(d.get("mean_tpot_ms", 0), 2),
           round(d.get("p99_tpot_ms", 0), 2)]
    open(out, "a").write(",".join(map(str, row)) + "\n")
    print("  ->", row)
except Exception as e:
    print("  (could not parse benchmark result:", e, ")")
PY
done
echo "wrote $OUT"
column -t -s, "$OUT"

#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for the continuous-batching demo.
#
# We serve the *Thinker* of Qwen3-Omni-30B-A3B (the autoregressive text-generation
# core that vLLM-Omni's stage-0 wraps). On a single 32 GB RTX 5090 the full
# Thinker+Talker+Code2Wav omni pipeline does not fit (official deploy config is
# verified on 2x H100-80G); the continuous-batching behaviour we want to show lives
# entirely in the AR scheduler, which is identical whether driven by `vllm serve`
# here or by vLLM-Omni's stage-0 engine.
#
# Plain `vllm serve <Qwen3-Omni model>` maps the arch
# `Qwen3OmniMoeForConditionalGeneration` -> `Qwen3OmniMoeThinkerForConditionalGeneration`
# (text out only), so only ~20 GB of thinker weights load, leaving room for KV cache.
#
# Run with the project's Python env active (e.g. `conda activate vllm_omni`); this script
# just calls `vllm` from PATH.
#
# Env vars:
#   MAX_NUM_SEQS   thinker max concurrent sequences. 1 = "no continuous batching"
#                  (requests serialize, wait for each other). >=2 = continuous batching.
#                  Default: 8
#   MODEL          model id/path. Default: cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
#                  (compressed-tensors int4 — ~20 GB thinker weights, fits a single 32 GB GPU).
#                  For bf16 use Qwen/Qwen3-Omni-30B-A3B-Instruct (needs more VRAM).
#   PORT           server port. Default: 8901
#   MAX_MODEL_LEN  context length. Default: 8192
#   GMU            gpu_memory_utilization. Default: 0.92
#   SERVED_NAME    served-model-name. Default: qwen3-omni
#   EXTRA          extra args appended to the command.
#
# Logs -> logs/server_seqs${MAX_NUM_SEQS}.log

set -uo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit}"
PORT="${PORT:-8901}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GMU="${GMU:-0.92}"
SERVED_NAME="${SERVED_NAME:-qwen3-omni}"
EXTRA="${EXTRA:-}"

LOG="logs/server_seqs${MAX_NUM_SEQS}.log"
mkdir -p logs

echo "[run_server] model=$MODEL port=$PORT MAX_NUM_SEQS=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN gmu=$GMU -> $LOG"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GMU" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --skip-mm-profiling \
  $EXTRA \
  > "$LOG" 2>&1

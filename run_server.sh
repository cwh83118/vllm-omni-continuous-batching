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
# Multimodal: this server is now configured to accept 1 audio + up to 2 images per
# prompt (the dual-stream cockpit benchmark needs them). The text-only baseline
# results (results/run_{none,static,continuous}.json) were produced with
# `--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' --skip-mm-profiling`;
# remove those flags here so vLLM sizes the mm encoder workspace correctly.
#
# Logs -> logs/server_seqs${MAX_NUM_SEQS}.log

set -uo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# vLLM-Omni 0.20.0 ships CUDA 13 libs as pip packages under nvidia/cu13/lib,
# but they aren't on the default loader path. PyTorch's eager-mode CUDA JIT
# (used by the multimodal profiler for tensor ops like grid_thw.prod()) fails
# with "libnvrtc-builtins.so.13.0: cannot open" unless we point ld at them.
# Same for the cuda_nvrtc 12.x libs used by older code paths.
NVRTC_PATHS="$(/home/davidchang/miniconda3/envs/vllm_omni/bin/python -c '
import os, sysconfig
site = sysconfig.get_paths()["purelib"]
roots = []
for sub in ("nvidia/cu13/lib", "nvidia/cuda_nvrtc/lib"):
    p = os.path.join(site, sub)
    if os.path.isdir(p):
        roots.append(p)
print(":".join(roots))
' 2>/dev/null)"
if [ -n "$NVRTC_PATHS" ]; then
    export LD_LIBRARY_PATH="${NVRTC_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

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
  --limit-mm-per-prompt '{"image":2,"video":0,"audio":1}' \
  $EXTRA \
  > "$LOG" 2>&1

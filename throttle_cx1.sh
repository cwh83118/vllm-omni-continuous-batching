#!/usr/bin/env bash
# Throttle this RTX 5090 to approximate CX1's effective compute + memory BW.
#
# CX1 targets (paper / spec):
#   Memory BW    : 154 GB/s   (vs 5090 native ~1.8 TB/s   → ratio ~1/12)
#   BF16 compute : ~50 TFLOPS (vs 5090 native ~210 TFLOPS → ratio ~1/4)
#
# 5090 GDDR7 only exposes 5 supported memory clocks: {14001, 13801, 7001, 810, 405} MHz
# (queried via `nvidia-smi --query-supported-clocks=mem`). There is no in-between
# step. Locking to 810 MHz produces measured D2D copy bw ≈ 68 GB/s — this is
# ~2× stricter than CX1's 154 GB/s.
#
# We deliberately err on the strict side. Real CX1 will be a bit FASTER than what
# we measure here (~154 vs our ~68 GB/s), so any latency advantage of continuous
# batching observed here is a CONSERVATIVE lower-bound for CX1.
#
# Graphics clock 745 MHz ≈ 24% of 5090's 3105 MHz max → ~50 TFLOPS effective.
#
# Usage:
#   bash throttle_cx1.sh           # apply throttle + calibration probe
#   bash throttle_cx1.sh release   # restore full clocks
#   bash throttle_cx1.sh status    # show current clocks + a fresh BW probe
#
# Env:  SUDO_PASS (default "qwer6789"),  PY (default vllm_omni env python)
set -uo pipefail

SUDO_PASS="${SUDO_PASS:-qwer6789}"
PY="${PY:-/home/davidchang/miniconda3/envs/vllm_omni/bin/python}"

# Fixed lock values (chosen from supported clock list above)
LOCK_MEM_MHZ=810
LOCK_GFX_MHZ=745

sudo_run() {
    echo "$SUDO_PASS" | sudo -S -p '' "$@"
}

probe_bw() {
    "$PY" - <<'PYEOF'
import torch, time, sys
torch.cuda.set_device(0)
n = 1 << 28          # 256M floats = 1 GiB > L2 cache
a = torch.empty(n, dtype=torch.float32, device='cuda')
b = torch.empty_like(a)
torch.cuda.synchronize()
for _ in range(5): b.copy_(a)
torch.cuda.synchronize()
N = 50
t0 = time.monotonic()
for _ in range(N): b.copy_(a)
torch.cuda.synchronize()
dt = (time.monotonic() - t0) / N
bw_gb = (a.numel() * a.element_size() * 2) / dt / 1e9  # read+write
print(f"{bw_gb:.0f}")
PYEOF
}

case "${1:-throttle}" in
  release)
    echo "[throttle] releasing memory + graphics clock locks"
    sudo_run nvidia-smi -rmc >/dev/null 2>&1 || true
    sudo_run nvidia-smi -rgc >/dev/null 2>&1 || true
    nvidia-smi --query-gpu=clocks.current.memory,clocks.current.graphics --format=csv
    echo "[throttle] BW probe at released clocks: $(probe_bw) GB/s"
    exit 0
    ;;
  status)
    nvidia-smi --query-gpu=clocks.current.memory,clocks.current.graphics,clocks.max.memory,clocks.max.graphics,power.draw,temperature.gpu --format=csv
    echo "[throttle] BW probe: $(probe_bw) GB/s"
    exit 0
    ;;
  throttle|*) ;;
esac

echo "[throttle] enabling persistence mode"
sudo_run nvidia-smi -pm 1 >/dev/null

MAX_MEM=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits)
MAX_GFX=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)
echo "[throttle] 5090 max clocks: mem=${MAX_MEM} MHz, gfx=${MAX_GFX} MHz"
echo "[throttle] locking to    : mem=${LOCK_MEM_MHZ} MHz, gfx=${LOCK_GFX_MHZ} MHz"
echo "[throttle] CX1 reference  : BW 154 GB/s, BF16 ~50 TFLOPS"
echo "[throttle] note: 5090 GDDR7 only supports 5 discrete mem clocks; 810 MHz is"
echo "[throttle]       the lowest non-idle one. Measured BW ≈ 68 GB/s is ~2× stricter"
echo "[throttle]       than CX1's 154 — real CX1 will be slightly faster than this,"
echo "[throttle]       so any continuous-batching advantage we measure is a LOWER BOUND."
echo ""

sudo_run nvidia-smi -lmc "${LOCK_MEM_MHZ}" 2>&1 | tail -2
sudo_run nvidia-smi -lgc "${LOCK_GFX_MHZ},${LOCK_GFX_MHZ}" 2>&1 | tail -2

# Show what landed and probe
nvidia-smi --query-gpu=clocks.current.memory,clocks.current.graphics --format=csv

BW=$(probe_bw)
echo ""
echo "[throttle] D2D copy bw probe at locked clocks: ${BW} GB/s"
RATIO=$(awk "BEGIN{printf \"%.2f\", 154.0/${BW}}")
echo "[throttle] CX1 spec target = 154 GB/s. Real CX1 is ${RATIO}× faster than our throttled 5090."
echo "[throttle] Any continuous-batching latency advantage measured here is a conservative LOWER bound for CX1."

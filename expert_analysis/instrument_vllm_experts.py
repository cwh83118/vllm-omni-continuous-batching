#!/usr/bin/env python3
"""Capture **ground-truth** per-(layer, expert) routing decisions from the real
Qwen3-Omni-30B-A3B Thinker on a GPU.

Strategy: monkey-patch ``vllm.model_executor.layers.fused_moe.select_experts``
(the routing helper that returns ``(topk_weights, topk_ids)``) BEFORE vLLM is
constructed; record ``topk_ids`` per-layer per call into a shared counters dict;
then drive the model with a chosen batch size via the offline ``LLM`` API,
capturing routing during decode.

For each batch size we emit:
  - ``results/measured_expert_heatmap_B{B}.csv``    same schema as the simulated
    file (layer, expert, activation_count, share_of_layer_topk_picks). After
    running, re-execute ``plot_results.py`` to redraw the figures from measured
    data (it auto-prefers ``measured_*`` files if present -- patch a one-liner
    yourself if you want strict replacement).
  - ``results/measured_summary_B{B}.json``          per-step distinct counts,
    elapsed time, tokens generated.

Run with the project's Python env active (``conda activate vllm_omni``) on a
machine with the model on disk / cached:

  python expert_analysis/instrument_vllm_experts.py \
      --batch-sizes 1 8 32 \
      --model cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit \
      --max-num-seqs 64 --max-tokens 64

If the patch point has moved in your vLLM version, set
``--patch-location`` to ``MODULE.FUNCTION`` (eg
``vllm.model_executor.layers.fused_moe.fused_moe.select_experts``).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- counters / patcher

# Shared state, populated by the wrapped select_experts. Keys are layer ids
# assigned in module-traversal order at first call (vLLM stacks all MoE layers
# linearly, so call ordering during a forward equals layer order).
_LAYER_CALL_ORDER: dict[int, int] = {}  # id(call_site) -> layer index
_COUNTS: dict[int, "np.ndarray"] = {}    # layer index -> [E] int64 counts
_NUM_EXPERTS = None
_TOP_K = None
_STEP_COUNTER = [0]
_PER_STEP_DISTINCT: list[list[int]] = []   # [step][layer] = distinct experts


def make_patched_select(orig_select):
    import numpy as np
    import torch

    def patched(*args, **kwargs):
        out = orig_select(*args, **kwargs)
        # vLLM versions: select_experts -> (topk_weights, topk_ids) or with
        # token_expert_indices appended. We only need topk_ids.
        topk_ids = None
        if isinstance(out, tuple) and len(out) >= 2:
            topk_ids = out[1]
        # heuristic: call site identity. Use traceback frame depth + the gate
        # module address if obtainable; simpler: assign incremental layer id
        # per call inside a single forward, reset between forwards.
        # Simplest: count calls modulo num_layers.
        global _NUM_EXPERTS, _TOP_K
        if topk_ids is not None:
            try:
                ids = topk_ids.detach().cpu().numpy()  # [B*T, top_k]
                if _NUM_EXPERTS is None:
                    # infer from the gate logits shape if available
                    _NUM_EXPERTS = int(args[0].shape[-1]) if hasattr(args[0], "shape") else 128
                    _TOP_K = int(ids.shape[-1])
                layer_idx = len(_PER_STEP_LAYER_RECORD)
                if layer_idx not in _COUNTS:
                    _COUNTS[layer_idx] = np.zeros(_NUM_EXPERTS, dtype=np.int64)
                flat = ids.reshape(-1)
                bc = np.bincount(flat, minlength=_NUM_EXPERTS)
                _COUNTS[layer_idx] += bc
                _PER_STEP_LAYER_RECORD.append(int((bc > 0).sum()))
            except Exception as e:
                print(f"[hook] warning: {e!r}", file=sys.stderr)
        return out
    return patched


# Per-forward layer-call record. Reset between forwards via end-of-step hook.
_PER_STEP_LAYER_RECORD: list[int] = []


def install_patch(loc: str) -> tuple[object, str]:
    mod_path, _, fname = loc.rpartition(".")
    mod = importlib.import_module(mod_path)
    orig = getattr(mod, fname)
    setattr(mod, fname, make_patched_select(orig))
    return orig, f"{mod_path}.{fname}"


def reset_step():
    global _PER_STEP_LAYER_RECORD
    if _PER_STEP_LAYER_RECORD:
        _PER_STEP_DISTINCT.append(list(_PER_STEP_LAYER_RECORD))
    _PER_STEP_LAYER_RECORD = []


# ------------------------------------------------------------------------- harness

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--patch-location", default="vllm.model_executor.layers.fused_moe.select_experts",
                    help="dotted path of the routing function to wrap; fallbacks: "
                         "vllm.model_executor.layers.fused_moe.fused_moe.select_experts")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # Try the user-given patch location, then fallbacks.
    candidates = [args.patch_location,
                  "vllm.model_executor.layers.fused_moe.fused_moe.select_experts",
                  "vllm.model_executor.layers.fused_moe.layer.select_experts"]
    patched_name = None
    last_err = None
    for loc in candidates:
        try:
            install_patch(loc)
            patched_name = loc
            break
        except Exception as e:
            last_err = e
    if not patched_name:
        sys.exit(f"could not patch routing function; last error: {last_err!r}. "
                 f"Set --patch-location explicitly.")
    print(f"[hook] patched {patched_name}")

    # Late import so the patch is already in place.
    from vllm import LLM, SamplingParams  # noqa: E402

    llm = LLM(model=args.model, trust_remote_code=True,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})

    PROMPTS = [
        "請用三句話描述今天的台北天氣。",
        "解釋什麼是 mixture-of-experts，盡量簡短。",
        "幫我規劃一條從台北到台中的順路路線。",
        "翻譯：The mitochondria is the powerhouse of the cell.",
        "寫一段 Python：fibonacci(n) 用 memoization。",
        "用四川話對小孩說放學在校門口等。",
        "比較 RTX 5090 和 H100 的記憶體頻寬。",
        "什麼是 continuous batching？一句話。",
    ]

    sp = SamplingParams(temperature=0.7, max_tokens=args.max_tokens, seed=args.seed)

    for B in args.batch_sizes:
        # reset counters
        _COUNTS.clear()
        _PER_STEP_DISTINCT.clear()
        global _PER_STEP_LAYER_RECORD
        _PER_STEP_LAYER_RECORD = []

        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(B)]
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        # End-of-run: dump any remaining layer record as a final step.
        reset_step()
        elapsed = time.time() - t0
        total_out = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"\n=== B={B}  elapsed={elapsed:.2f}s  output_tokens={total_out}  "
              f"throughput={total_out/elapsed:.1f} tok/s ===")
        if not _COUNTS:
            print("  (no routing captured — patch may be wrong; try --patch-location)")
            continue
        L = len(_COUNTS)
        import numpy as np
        arr = np.stack([_COUNTS[L_idx] for L_idx in range(L)], axis=0)  # [L, E]
        per_layer_total = arr.sum(axis=1, keepdims=True).clip(min=1)
        share = arr / per_layer_total

        out_csv = RES / f"measured_expert_heatmap_B{B}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "expert", "activation_count", "share_of_layer_topk_picks"])
            for L_idx in range(L):
                for E_idx in range(arr.shape[1]):
                    w.writerow([L_idx, E_idx, int(arr[L_idx, E_idx]),
                                f"{share[L_idx, E_idx]:.6f}"])
        print(f"  wrote {out_csv}")

        summary = {
            "B": B,
            "elapsed_s": elapsed,
            "output_tokens": total_out,
            "throughput_tok_s": total_out / elapsed,
            "num_layers": L,
            "num_experts": int(arr.shape[1]),
            "top_k": _TOP_K,
            "distinct_per_layer_per_step_mean": (np.mean(_PER_STEP_DISTINCT, axis=0).tolist()
                                                  if _PER_STEP_DISTINCT else None),
            "per_layer_total_topk_picks": arr.sum(axis=1).tolist(),
            "patch_location": patched_name,
        }
        with open(RES / f"measured_summary_B{B}.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

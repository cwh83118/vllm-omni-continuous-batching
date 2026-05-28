#!/usr/bin/env python3
"""Batch-size sweep: TTFT / throughput / per-layer expert activation / expert DRAM.

What this script DOES on a CPU (no GPU needed):
  1. Exact DRAM accounting for the Qwen3-Omni-30B-A3B Thinker (MoE) per component,
     at bf16 / fp8 / AWQ int4-g32. Tells you how many GiB are spent on experts.
  2. Simulates the MoE routing decision per decode step for a sweep of batch sizes,
     logs per-(layer, expert) activation counts, and computes:
       - distinct experts activated per layer per step (mean over the run)
       - dynamic working-set expert DRAM at batch B (the part that actually
         needs to be touched per step -- relevant for offloading systems)
  3. A calibrated roofline performance model (TTFT, decode throughput) tied to the
     repo's real RTX 5090 measurements at B=1 and B=8 (from REPORT.md).

What this script does NOT do: run the real 30B model. The expert routing here is a
*model*, not measured; the analytic uniform-routing curve is rigorous; the skewed
simulation is illustrative. For ground-truth per-layer routing on real hardware,
use ``instrument_vllm_experts.py`` on a machine with a 5090 + the running server.

Outputs go under ``expert_analysis/results/``:
  dram_accounting.csv          -- bytes per component per precision
  dram_per_layer.csv           -- per-layer split (expert vs non-expert)
  expert_saturation.csv        -- distinct experts per layer per step, vs B
  expert_heatmap_B{B}.csv      -- (layer, expert) cumulative activation, for B in {1,8,32,128}
  batch_sweep.csv              -- per-B: working-set GiB, weight traffic, modeled TTFT/throughput
  summary.json                 -- everything above + headline numbers
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------- architecture

@dataclass
class Arch:
    """Qwen3-Omni-30B-A3B Thinker (Qwen3-MoE family).

    Values match the public Qwen3-30B-A3B config (the Thinker uses the same MoE
    backbone): 48 decoder layers, every layer is MoE, 128 experts/layer, top-8
    routing, moe_intermediate_size 768, hidden 2048, GQA (32 Q heads / 4 KV heads,
    head_dim 128), no shared expert, ``tie_word_embeddings=false``. Override any
    field if your specific checkpoint differs.
    """
    name: str = "Qwen3-Omni-30B-A3B-Thinker"
    num_layers: int = 48
    hidden: int = 2048
    num_q_heads: int = 32
    num_kv_heads: int = 4
    head_dim: int = 128
    moe_intermediate: int = 768
    num_experts: int = 128
    top_k: int = 8
    has_shared_expert: bool = False
    vocab: int = 151936
    tie_word_embeddings: bool = False

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def per_expert_params(self) -> int:
        return 3 * self.hidden * self.moe_intermediate

    @property
    def per_layer_attn_params(self) -> int:
        return (self.hidden * self.q_dim + self.hidden * self.kv_dim * 2 + self.q_dim * self.hidden)

    @property
    def per_layer_router_params(self) -> int:
        return self.hidden * self.num_experts

    @property
    def embed_params(self) -> int:
        return self.vocab * self.hidden

    def total_expert_params(self) -> int:
        return self.num_layers * self.num_experts * self.per_expert_params

    def total_attn_params(self) -> int:
        return self.num_layers * self.per_layer_attn_params

    def total_router_params(self) -> int:
        return self.num_layers * self.per_layer_router_params

    def total_embed_lmhead_params(self) -> int:
        return self.embed_params * (1 if self.tie_word_embeddings else 2)

    def total_params(self) -> int:
        return (self.total_expert_params() + self.total_attn_params()
                + self.total_router_params() + self.total_embed_lmhead_params())

    def active_params_per_token(self) -> int:
        return (self.per_layer_attn_params * self.num_layers
                + self.top_k * self.per_expert_params * self.num_layers
                + self.per_layer_router_params * self.num_layers
                + self.embed_params  # lm_head projection
                )


# bytes-per-parameter for different storage formats
BPP = {
    "bf16": 2.0,
    "fp8":  1.0,
    # AWQ int4 with group_size 32: 4 bits weight + 16-bit scale per 32 weights +
    # 4-bit zero point per 32 weights -> 4 + 16/32 + 4/32 = 4.625 bits = 0.578125 B/param.
    "int4_awq_g32": 4.625 / 8,
}

def bytes_for(params: int, precision: str, *, fp16_embed_lmhead: bool = False,
              n_embed_lmhead: int = 0) -> float:
    """Bytes for ``params`` weights at ``precision``; for AWQ configs the
    embedding/lm_head are often kept in fp16 -- caller passes ``n_embed_lmhead``
    to subtract those from ``params`` and add them back at fp16."""
    if fp16_embed_lmhead:
        return (params - n_embed_lmhead) * BPP[precision] + n_embed_lmhead * BPP["bf16"]
    return params * BPP[precision]

GiB = 1024 ** 3


# --------------------------------------------------------------------- DRAM accounting

def dram_accounting(arch: Arch) -> dict:
    exp_p = arch.total_expert_params()
    att_p = arch.total_attn_params()
    rtr_p = arch.total_router_params()
    emb_p = arch.total_embed_lmhead_params()
    tot_p = exp_p + att_p + rtr_p + emb_p

    rows = []
    for prec in ["bf16", "fp8", "int4_awq_g32"]:
        fp16_el = (prec == "int4_awq_g32")  # AWQ commonly keeps embed/lm_head fp16
        comp = {
            "experts":      bytes_for(exp_p, prec),
            "attention":    bytes_for(att_p, prec),
            "router":       bytes_for(rtr_p, prec),
            "embed_lmhead": bytes_for(emb_p, prec if not fp16_el else "bf16"),
        }
        total = sum(comp.values())
        rows.append({
            "precision": prec,
            "experts_GiB":      round(comp["experts"]      / GiB, 3),
            "attention_GiB":    round(comp["attention"]    / GiB, 3),
            "router_GiB":       round(comp["router"]       / GiB, 3),
            "embed_lmhead_GiB": round(comp["embed_lmhead"] / GiB, 3),
            "total_weights_GiB": round(total / GiB, 3),
            "experts_pct_of_params":  round(100 * exp_p / tot_p, 2),
            "experts_pct_of_bytes":   round(100 * comp["experts"] / total, 2),
        })
    return {"rows": rows, "params": {
        "experts": exp_p, "attention": att_p, "router": rtr_p,
        "embed_lmhead": emb_p, "total": tot_p,
        "active_per_token": arch.active_params_per_token(),
    }}


def per_layer_dram(arch: Arch, precision: str = "int4_awq_g32") -> list[dict]:
    """One row per layer; constant across layers in the homogeneous Qwen3-MoE, but
    we emit per-layer rows so the format is friendly for replacing with measured
    per-layer numbers (e.g. some layers dense / others MoE in other models)."""
    out = []
    for L in range(arch.num_layers):
        exp_p = arch.num_experts * arch.per_expert_params
        att_p = arch.per_layer_attn_params
        rtr_p = arch.per_layer_router_params
        out.append({
            "layer": L,
            "expert_params": exp_p,
            "attn_params": att_p,
            "router_params": rtr_p,
            "expert_bytes": bytes_for(exp_p, precision),
            "attn_bytes":   bytes_for(att_p, precision),
            "expert_GiB":   round(bytes_for(exp_p, precision) / GiB, 4),
        })
    return out


# ------------------------------------------------------- expert-activation simulation

def make_layer_popularity(arch: Arch, alpha: float, seed: int) -> np.ndarray:
    """Per-layer expert popularity distributions (shape ``[L, E]``, rows sum to 1).

    Drawn from Dirichlet(alpha). alpha large -> uniform; alpha small -> very skewed.
    The same draws are used across all batch sizes so the popularity *identity* of
    hot experts is stable, only batch size changes."""
    rng = np.random.default_rng(seed)
    if math.isinf(alpha):
        return np.full((arch.num_layers, arch.num_experts), 1.0 / arch.num_experts)
    return rng.dirichlet(np.full(arch.num_experts, alpha), size=arch.num_layers)


def topk_per_token(prob_layer: np.ndarray, B: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``k`` distinct experts per token via the Gumbel top-k trick on
    weighted probabilities. ``prob_layer`` is ``[E]``; returns ``[B, k]`` indices."""
    E = prob_layer.shape[0]
    log_p = np.log(np.maximum(prob_layer, 1e-30))
    g = -np.log(-np.log(rng.uniform(size=(B, E))))
    scores = log_p[None, :] + g
    # argpartition for top-k
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return idx  # [B, k]


def simulate_activations(arch: Arch, batch_sizes: list[int], *,
                         alpha: float, n_steps: int, seed: int,
                         detail_for: list[int]) -> dict:
    """For each B, simulate ``n_steps`` decode steps. Each step has B tokens; each
    token picks top-k experts per layer according to that layer's popularity.

    Returns:
      saturation[B][layer]    : mean number of *distinct* experts activated per step
      cumulative[B]           : [L, E] total activation counts over the whole run
      per_step_distinct[B]    : [n_steps, L] distinct experts per step per layer
    """
    L, E, k = arch.num_layers, arch.num_experts, arch.top_k
    pop = make_layer_popularity(arch, alpha, seed)
    rng = np.random.default_rng(seed + 1)

    saturation = {}
    cumulative = {}
    distinct_per_step = {}

    for B in batch_sizes:
        cum = np.zeros((L, E), dtype=np.int64)
        distinct_steps = np.zeros((n_steps, L), dtype=np.int32)
        for s in range(n_steps):
            for L_idx in range(L):
                chosen = topk_per_token(pop[L_idx], B, k, rng)  # [B, k]
                flat = chosen.reshape(-1)
                counts = np.bincount(flat, minlength=E)
                cum[L_idx] += counts
                distinct_steps[s, L_idx] = int((counts > 0).sum())
        saturation[B] = distinct_steps.mean(axis=0)  # [L]
        cumulative[B] = cum
        distinct_per_step[B] = distinct_steps

    return {
        "popularity": pop,
        "saturation": saturation,
        "cumulative": cumulative,
        "per_step_distinct": distinct_per_step,
        "alpha": alpha,
        "n_steps": n_steps,
    }


def analytic_distinct_uniform(B: int, k: int, E: int) -> float:
    """Closed-form expected distinct experts under uniform routing.

    Each token picks ``k`` distinct experts uniformly; across B independent tokens
    (with replacement at the token level), P(a given expert NOT picked by any of
    the B tokens) = ((E-k)/E)**B (treating tokens as independent draws of a
    k-subset, which is exact for the *expectation* of the indicator)."""
    if B <= 0:
        return 0.0
    return E * (1 - ((E - k) / E) ** B)


# --------------------------------------------------------- working-set & traffic model

def working_set_bytes(distinct_per_layer: np.ndarray, arch: Arch, precision: str) -> float:
    """Total expert bytes that must be *touched* in one decode step, summed over
    all layers, given ``distinct_per_layer[L]`` distinct experts at each layer."""
    per_expert_bytes = bytes_for(arch.per_expert_params, precision)
    return float(distinct_per_layer.sum() * per_expert_bytes)


def weight_traffic_per_step_bytes(distinct_per_layer: np.ndarray, arch: Arch, precision: str) -> float:
    """Total weight bytes the GPU must read from HBM in one decode step.

    Includes: distinct expert weights (per layer), all attention weights (all
    layers, dense), all routers (small), and the lm_head (read once -- serves the
    whole batch). Embedding lookup ignored (per-token gather, negligible)."""
    exp_b = working_set_bytes(distinct_per_layer, arch, precision)
    att_b = bytes_for(arch.total_attn_params(), precision)
    rtr_b = bytes_for(arch.total_router_params(), precision)
    # lm_head: in AWQ configs we keep it fp16
    lm_b = bytes_for(arch.embed_params, "bf16" if precision == "int4_awq_g32" else precision)
    return exp_b + att_b + rtr_b + lm_b


# -------------------------------------------------------------- perf roofline (calibrated)

@dataclass
class Anchors:
    """Real measured numbers from this repo (RTX 5090, AWQ int4, MAX_NUM_SEQS=32 server)."""
    # Burst (24 short interactive requests at once, <=160 out tokens each)
    # From results/burst_*.json summaries / REPORT.md sec 4.3.
    B1_agg_tok_s: float = 238.0    # no-batching baseline = single-seq decode speed
    B8_agg_tok_s: float = 1083.0   # continuous B=8 aggregate output throughput
    # TTFT during burst
    B1_ttft_p50_ms: float = 4359.0
    B8_ttft_p50_ms: float = 641.0

    # Hardware
    hbm_peak_GBps: float = 1792.0  # RTX 5090 GDDR7 32GB peak bandwidth


def calibrate_roofline(arch: Arch, anchors: Anchors,
                       distinct_at_B8: float,
                       precision: str = "int4_awq_g32") -> dict:
    """Calibrate effective achievable HBM bandwidth as a function of batch size,
    using BOTH real measurements (B=1 and B=8 aggregate throughput) as anchors.

    At each B, decode time-per-step = bytes_per_step / eff_BW(B). MBU rises with
    batch (larger GEMMs hit higher tensor-core efficiency, more amortized overhead
    per kernel launch), so eff_BW is not constant. We fit a simple log-linear:
      eff_BW(B) = clip( a + b * log2(B), 0, peak ).
    Two anchors => two equations => closed-form (a, b)."""
    # Anchor 1: B=1, distinct = top_k uniformly
    d1 = np.full(arch.num_layers, arch.top_k)
    bps_1 = weight_traffic_per_step_bytes(d1, arch, precision)
    bw_1 = bps_1 * anchors.B1_agg_tok_s / 1e9  # GB/s

    # Anchor 2: B=8 with the (simulated) distinct per layer
    d8 = np.full(arch.num_layers, distinct_at_B8)
    bps_8 = weight_traffic_per_step_bytes(d8, arch, precision)
    # B8_agg = B / time_step  =>  bw = bps_8 * (B8_agg / 8)
    bw_8 = bps_8 * anchors.B8_agg_tok_s / 8.0 / 1e9

    # Fit: bw(B) = a + b * log2(B). bw(1) = a; bw(8) = a + 3b.
    a = bw_1
    b = (bw_8 - bw_1) / 3.0
    return {
        "eff_BW_GBps_at_B1":       bw_1,
        "eff_BW_GBps_at_B8":       bw_8,
        "MBU_at_B1":               bw_1 / anchors.hbm_peak_GBps,
        "MBU_at_B8":               bw_8 / anchors.hbm_peak_GBps,
        "fit_a_GBps":              a,
        "fit_b_GBps_per_log2B":    b,
        "hbm_peak_GBps":           anchors.hbm_peak_GBps,
    }


def eff_bandwidth_GBps(B: int, calib: dict) -> float:
    """eff_BW(B) clipped to [0, peak]."""
    raw = calib["fit_a_GBps"] + calib["fit_b_GBps_per_log2B"] * math.log2(max(B, 1))
    return float(max(0.0, min(raw, calib["hbm_peak_GBps"])))


def model_throughput_ttft(B: int, distinct_per_layer: np.ndarray, arch: Arch,
                          calib: dict, anchors: Anchors,
                          precision: str = "int4_awq_g32") -> dict:
    """Roofline throughput model using the log-linear-BW calibration.

    TTFT is *not* extrapolated -- it depends heavily on the offered-load regime
    (queueing-bound vs prefill-bound); see REPORT.md sec 4 for the real numbers.
    We report the queueing-free per-batch prefill time as a lower-bound proxy."""
    eff_BWps = eff_bandwidth_GBps(B, calib) * 1e9
    bytes_per_step = weight_traffic_per_step_bytes(distinct_per_layer, arch, precision)
    time_per_step_s = bytes_per_step / eff_BWps
    throughput_tok_s = B / time_per_step_s
    return {
        "B": B,
        "weight_bytes_per_step": bytes_per_step,
        "weight_GiB_per_step": bytes_per_step / GiB,
        "eff_BW_GBps": eff_BWps / 1e9,
        "MBU": (eff_BWps / 1e9) / anchors.hbm_peak_GBps,
        "step_time_ms": time_per_step_s * 1e3,
        "throughput_tok_s": throughput_tok_s,
        "throughput_per_seq_tok_s": throughput_tok_s / max(B, 1),
    }


# --------------------------------------------------------------------- CSV writers

def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        path.write_text("")
        return
    fn = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_heatmap_csv(path: Path, arr: np.ndarray, arch: Arch):
    """arr is [L, E]; write rows of (layer, expert_id, count, fraction_of_layer_tokens)."""
    totals_per_layer = arr.sum(axis=1, keepdims=True).clip(min=1)
    frac = arr / totals_per_layer
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "expert", "activation_count", "share_of_layer_topk_picks"])
        for L in range(arch.num_layers):
            for E in range(arch.num_experts):
                w.writerow([L, E, int(arr[L, E]), f"{frac[L, E]:.6f}"])


# ----------------------------------------------------------------------------- main

def main():
    arch = Arch()
    anchors = Anchors()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    detail_Bs = [1, 8, 32, 128]
    n_steps = 256
    alpha = 5.0  # moderate routing skew; tune to taste

    t0 = time.time()
    print(f"# Experiment: batch-size sweep -- TTFT / throughput / experts")
    print(f"# Arch: {arch.name}  L={arch.num_layers}  H={arch.hidden}  "
          f"E={arch.num_experts}  top_k={arch.top_k}  moe_inter={arch.moe_intermediate}")
    print(f"# total params = {arch.total_params()/1e9:.2f} B   "
          f"active/token = {arch.active_params_per_token()/1e9:.2f} B  (-> 30B-A3B)")

    # ---- 1. DRAM accounting
    dram = dram_accounting(arch)
    print("\n[1] DRAM accounting (whole model weights, GiB):")
    print(f"  {'precision':<15s}  {'experts':>9s}  {'attn':>7s}  {'embed+lm':>9s}  "
          f"{'total':>7s}  {'exp%bytes':>9s}")
    for r in dram["rows"]:
        print(f"  {r['precision']:<15s}  {r['experts_GiB']:>9.2f}  {r['attention_GiB']:>7.2f}"
              f"  {r['embed_lmhead_GiB']:>9.2f}  {r['total_weights_GiB']:>7.2f}"
              f"  {r['experts_pct_of_bytes']:>8.1f}%")
    write_csv(RES / "dram_accounting.csv", dram["rows"])
    write_csv(RES / "dram_per_layer.csv", per_layer_dram(arch))

    # ---- 2. Expert activation simulation
    print(f"\n[2] Simulating MoE routing for {len(batch_sizes)} batch sizes, "
          f"{n_steps} steps each (alpha={alpha})...")
    sim = simulate_activations(arch, batch_sizes, alpha=alpha, n_steps=n_steps,
                               seed=7, detail_for=detail_Bs)
    print("    done in", round(time.time() - t0, 1), "s")

    # report popularity skew
    pop = sim["popularity"]  # [L, E]
    max_per_layer = pop.max(axis=1) * arch.num_experts  # max_load / mean_load
    print(f"    routing load imbalance (alpha={alpha}): "
          f"max/mean per layer  median {np.median(max_per_layer):.2f}, "
          f"p95 {np.percentile(max_per_layer, 95):.2f}, max {max_per_layer.max():.2f}")

    # 2a. expert-saturation summary -- distinct experts/layer/step vs B
    sat_rows = []
    print("\n    distinct experts activated per layer per step (mean over layers):")
    print(f"    {'B':>4}  {'analytic_uniform':>18s}  {'sim_skewed_median':>18s}  "
          f"{'sim_skewed_p5':>14s}  {'sim_skewed_p95':>14s}")
    for B in batch_sizes:
        d = sim["saturation"][B]  # [L] mean per-step distinct
        au = analytic_distinct_uniform(B, arch.top_k, arch.num_experts)
        row = {
            "B": B,
            "analytic_uniform_distinct_per_layer": round(au, 3),
            "sim_skewed_mean_distinct": round(float(d.mean()), 3),
            "sim_skewed_median_distinct": round(float(np.median(d)), 3),
            "sim_skewed_p5_distinct":  round(float(np.percentile(d, 5)),  3),
            "sim_skewed_p95_distinct": round(float(np.percentile(d, 95)), 3),
            "frac_of_all_experts_uniform": round(au / arch.num_experts, 4),
            "frac_of_all_experts_skewed":  round(float(d.mean()) / arch.num_experts, 4),
        }
        sat_rows.append(row)
        print(f"    {B:>4}  {au:>18.2f}  {float(np.median(d)):>18.2f}  "
              f"{float(np.percentile(d, 5)):>14.2f}  {float(np.percentile(d, 95)):>14.2f}")
    write_csv(RES / "expert_saturation.csv", sat_rows)

    # 2b. detailed per-(layer, expert) cumulative activation count, for selected B
    for B in detail_Bs:
        write_heatmap_csv(RES / f"expert_heatmap_B{B}.csv", sim["cumulative"][B], arch)
    print(f"    wrote per-(layer, expert) heatmap CSVs for B in {detail_Bs}")

    # ---- 3. Working-set DRAM + roofline perf
    distinct_B8 = float(sim["saturation"][8].mean())
    calib = calibrate_roofline(arch, anchors, distinct_B8)
    print(f"\n[3] Calibrating roofline to real RTX 5090 anchors:")
    print(f"    anchor B=1  agg throughput {anchors.B1_agg_tok_s} tok/s "
          f"-> eff BW {calib['eff_BW_GBps_at_B1']:.0f} GB/s  "
          f"(MBU {calib['MBU_at_B1']*100:.1f}%)")
    print(f"    anchor B=8  agg throughput {anchors.B8_agg_tok_s} tok/s "
          f"-> eff BW {calib['eff_BW_GBps_at_B8']:.0f} GB/s  "
          f"(MBU {calib['MBU_at_B8']*100:.1f}%)")
    print(f"    fit: eff_BW(B) [GB/s] = {calib['fit_a_GBps']:.0f} + "
          f"{calib['fit_b_GBps_per_log2B']:.1f} * log2(B), clipped to peak {anchors.hbm_peak_GBps:.0f}")

    sweep_rows = []
    print("\n    batch sweep (skewed routing, int4-AWQ):")
    print(f"    {'B':>4}  {'distinct/layer':>14s}  {'work_set_GiB':>12s}  "
          f"{'bytes/step_GiB':>14s}  {'eff_BW_GBps':>11s}  {'thru_tok_s':>10s}  {'per_seq':>8s}")
    for B in batch_sizes:
        d = sim["saturation"][B]
        ws_bytes = working_set_bytes(d, arch, "int4_awq_g32")
        perf = model_throughput_ttft(B, d, arch, calib, anchors)
        row = {
            "B": B,
            "distinct_per_layer_mean": round(float(d.mean()), 3),
            "working_set_expert_bytes": int(ws_bytes),
            "working_set_expert_GiB":   round(ws_bytes / GiB, 3),
            "resident_expert_GiB_int4": round(bytes_for(arch.total_expert_params(),
                                                        "int4_awq_g32") / GiB, 3),
            "weight_bytes_per_step":    int(perf["weight_bytes_per_step"]),
            "weight_GiB_per_step":      round(perf["weight_GiB_per_step"], 3),
            "eff_BW_GBps":              round(perf["eff_BW_GBps"], 1),
            "MBU":                      round(perf["MBU"], 4),
            "step_time_ms":             round(perf["step_time_ms"], 3),
            "modeled_throughput_tok_s": round(perf["throughput_tok_s"], 1),
            "modeled_per_seq_tok_s":    round(perf["throughput_per_seq_tok_s"], 2),
        }
        sweep_rows.append(row)
        print(f"    {B:>4}  {row['distinct_per_layer_mean']:>14.2f}  "
              f"{row['working_set_expert_GiB']:>12.3f}  "
              f"{row['weight_GiB_per_step']:>14.3f}  "
              f"{row['eff_BW_GBps']:>11.0f}  "
              f"{row['modeled_throughput_tok_s']:>10.1f}  "
              f"{row['modeled_per_seq_tok_s']:>8.1f}")
    write_csv(RES / "batch_sweep.csv", sweep_rows)

    # Real measured TTFT/throughput points from REPORT.md (so plots can show them)
    real_points = [
        # (B, regime, ttft_p50_ms, agg_tok_s, source)
        {"B": 1, "regime": "burst no-batching",       "ttft_p50_ms": 4359, "agg_tok_s": 238,  "src": "results/burst_none.json"},
        {"B": 8, "regime": "burst static (NPU-style)","ttft_p50_ms":  904, "agg_tok_s": 863,  "src": "results/burst_static.json"},
        {"B": 8, "regime": "burst continuous",        "ttft_p50_ms":  641, "agg_tok_s": 1083, "src": "results/burst_continuous.json"},
    ]
    write_csv(RES / "real_anchors.csv", real_points)

    # ---- 4. Summary JSON
    headline = {
        "arch": asdict(arch),
        "anchors": asdict(anchors),
        "calibration": calib,
        "params": dram["params"],
        "dram_headline": {
            "experts_GiB_bf16": dram["rows"][0]["experts_GiB"],
            "experts_GiB_fp8":  dram["rows"][1]["experts_GiB"],
            "experts_GiB_int4_awq_g32": dram["rows"][2]["experts_GiB"],
            "total_weights_GiB_int4_awq_g32": dram["rows"][2]["total_weights_GiB"],
            "experts_pct_of_bytes_int4": dram["rows"][2]["experts_pct_of_bytes"],
        },
        "saturation": sat_rows,
        "batch_sweep": sweep_rows,
        "real_anchors": real_points,
        "simulation": {
            "alpha": alpha,
            "n_steps": n_steps,
            "median_max_to_mean_load": float(np.median(max_per_layer)),
            "p95_max_to_mean_load":    float(np.percentile(max_per_layer, 95)),
        },
        "notes": [
            "DRAM accounting is exact given the architecture (Qwen3-MoE Thinker).",
            "Expert activation is a *model* (Dirichlet popularity + Gumbel top-k). "
            "Replace with measured routing via instrument_vllm_experts.py on a real GPU.",
            "Throughput/TTFT are a roofline calibrated to repo's real 5090 anchors "
            "(B=1: 238 tok/s; B=8: 1083 tok/s aggregate). Treat as illustrative.",
        ],
    }
    with open(RES / "summary.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)
    print(f"\n[done] wrote {RES}/  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

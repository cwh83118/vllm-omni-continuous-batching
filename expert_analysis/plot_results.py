#!/usr/bin/env python3
"""Generate plots from results/ artifacts produced by run_experiment.py."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def load_csv(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def load_summary() -> dict:
    with open(RES / "summary.json") as f:
        return json.load(f)


# ------------------------------------------------------------------ DRAM breakdown

def plot_dram_breakdown(summary: dict):
    rows = summary["dram_headline"]
    precisions = ["bf16", "fp8", "int4_awq_g32"]
    exp = [rows["experts_GiB_bf16"], rows["experts_GiB_fp8"], rows["experts_GiB_int4_awq_g32"]]
    # other = total - experts (only int4 row has total in headline; compute others)
    dram_rows = {r["precision"]: r for r in summary["batch_sweep"]} if False else None
    # use dram_accounting.csv for the actual per-precision totals
    accounting = load_csv(RES / "dram_accounting.csv")
    by_prec = {r["precision"]: r for r in accounting}
    other = [float(by_prec[p]["total_weights_GiB"]) - float(by_prec[p]["experts_GiB"]) for p in precisions]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(precisions))
    bars_exp = ax.bar(x, exp, label="experts", color="#cf3a4a")
    bars_oth = ax.bar(x, other, bottom=exp, label="attn + embed/lm + router", color="#7e91a8")

    for i, p in enumerate(precisions):
        total = exp[i] + other[i]
        ax.text(i, total + 1.0, f"{total:.1f} GiB", ha="center", fontsize=10, fontweight="bold")
        pct = exp[i] / total * 100
        ax.text(i, exp[i] / 2, f"{exp[i]:.1f}\n({pct:.0f}% experts)", ha="center", va="center",
                color="white", fontsize=10)

    ax.set_xticks(x); ax.set_xticklabels(precisions)
    ax.set_ylabel("Weight DRAM (GiB)")
    ax.axhline(32, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.text(2.45, 32.4, "RTX 5090 (32 GiB)", fontsize=8, ha="right", alpha=0.6)
    ax.set_title(f"Qwen3-Omni-30B-A3B Thinker — expert weights dominate DRAM\n"
                 f"L={summary['arch']['num_layers']}, "
                 f"E={summary['arch']['num_experts']}, "
                 f"top_k={summary['arch']['top_k']}, "
                 f"moe_inter={summary['arch']['moe_intermediate']}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RES / "dram_breakdown.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------- expert saturation vs batch size

def plot_expert_saturation(summary: dict):
    rows = summary["saturation"]
    B = [r["B"] for r in rows]
    uni = [r["analytic_uniform_distinct_per_layer"] for r in rows]
    skewed = [r["sim_skewed_mean_distinct"] for r in rows]
    sk_p5 = [r["sim_skewed_p5_distinct"] for r in rows]
    sk_p95 = [r["sim_skewed_p95_distinct"] for r in rows]

    E = summary["arch"]["num_experts"]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(B, uni, marker="o", color="#1f77b4", label="uniform routing (analytic)")
    ax.plot(B, skewed, marker="s", color="#cf3a4a",
            label=f"skewed routing (sim, Dirichlet α={summary['simulation']['alpha']})")
    ax.fill_between(B, sk_p5, sk_p95, color="#cf3a4a", alpha=0.15, label="skewed p5-p95 across layers")
    ax.axhline(E, color="k", ls=":", lw=0.8)
    ax.text(B[-1], E - 5, f"all {E} experts", ha="right", fontsize=8, alpha=0.6)

    ax.set_xscale("log", base=2)
    ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B])
    ax.set_xlabel("Batch size B (concurrent decode sequences)")
    ax.set_ylabel("Distinct experts activated per layer per step\n(mean over layers, 256 steps)")
    ax.set_title("Expert saturation: how many of 128 experts each layer must read per decode step\n"
                 "Going from B=1 → B=64 takes the layer from 8 experts to ~all 128")
    ax.set_ylim(0, E + 8)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "expert_saturation_vs_batch.png", dpi=140)
    plt.close(fig)


# -------------------------------------------------------- per-(layer, expert) heatmaps

def plot_expert_heatmaps(summary: dict, Bs: list[int]):
    fig, axes = plt.subplots(1, len(Bs), figsize=(4.2 * len(Bs), 6.4), sharey=True)
    if len(Bs) == 1:
        axes = [axes]
    E = summary["arch"]["num_experts"]
    L = summary["arch"]["num_layers"]
    for ax, B in zip(axes, Bs):
        rows = load_csv(RES / f"expert_heatmap_B{B}.csv")
        arr = np.zeros((L, E))
        for r in rows:
            arr[int(r["layer"]), int(r["expert"])] = float(r["share_of_layer_topk_picks"])
        im = ax.imshow(arr, aspect="auto", cmap="magma", vmin=0,
                       vmax=max(0.05, arr.max() * 0.8))
        ax.set_title(f"B = {B}\n(distinct/layer ≈ "
                     f"{[r['sim_skewed_mean_distinct'] for r in summary['saturation'] if r['B']==B][0]:.0f}"
                     f" / {E})", fontsize=11)
        ax.set_xlabel("expert id (0–127)")
        if ax is axes[0]:
            ax.set_ylabel("layer (0 = early, 47 = late)")
        # zero cells -> mark distinctly. Overlay grey mask:
        zeros = (arr == 0).astype(float)
        ax.imshow(np.ma.masked_where(zeros == 0, zeros), aspect="auto",
                  cmap="Greys", vmin=0, vmax=1, alpha=0.6)
    fig.colorbar(im, ax=axes, shrink=0.7, label="share of layer's top-k picks (cumulative over 256 steps)")
    fig.suptitle("Per-layer per-expert activation share — colored = touched, grey = never touched",
                 fontsize=12)
    fig.savefig(RES / f"expert_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------- working-set DRAM vs batch size

def plot_working_set(summary: dict):
    rows = summary["batch_sweep"]
    B = [r["B"] for r in rows]
    ws = [r["working_set_expert_GiB"] for r in rows]
    res = rows[0]["resident_expert_GiB_int4"]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(res, color="#444", ls="-", lw=1.6,
               label=f"resident expert DRAM (all 128 experts) = {res:.2f} GiB")
    ax.plot(B, ws, marker="o", color="#cf3a4a",
            label="working-set expert DRAM (touched per step)")
    for b, w in zip(B, ws):
        pct = w / res * 100
        ax.annotate(f"{w:.2f} GiB\n({pct:.0f}%)", (b, w),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B])
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Expert weights, int4-AWQ (GiB)")
    ax.set_title("Static vs dynamic expert DRAM — the offloading window closes as batch grows\n"
                 "At B=1 only ~6% of experts are touched per step; at B≥64 it's essentially all of them")
    ax.set_ylim(0, res * 1.18)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "working_set_dram_vs_batch.png", dpi=140)
    plt.close(fig)


# -------------------------------------------------------- throughput / TTFT vs batch

def plot_throughput_ttft(summary: dict):
    rows = summary["batch_sweep"]
    B = [r["B"] for r in rows]
    agg = [r["modeled_throughput_tok_s"] for r in rows]
    per_seq = [r["modeled_per_seq_tok_s"] for r in rows]
    mbu = [r["MBU"] * 100 for r in rows]

    # real anchors
    real = summary["real_anchors"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    # left: aggregate + per-seq throughput
    axL.plot(B, agg, marker="o", color="#1f77b4", label="aggregate throughput (model)")
    axL.plot(B, per_seq, marker="s", color="#cf3a4a", label="per-seq throughput (model)")
    # mark anchors
    for r in real:
        c = {"burst no-batching": "#1f77b4", "burst continuous": "#1f77b4", "burst static (NPU-style)": "#888"}[r["regime"]]
        axL.scatter([r["B"]], [r["agg_tok_s"]], s=110, marker="*", color=c, edgecolor="k",
                    zorder=5, label=f"real: {r['regime']} (B={r['B']})")
    axL.set_xscale("log", base=2)
    axL.set_yscale("log")
    axL.set_xticks(B); axL.set_xticklabels([str(b) for b in B])
    axL.set_xlabel("Batch size B")
    axL.set_ylabel("Decode throughput (tok/s)")
    axL.set_title("Decode throughput vs batch — calibrated to real B=1 & B=8 anchors\n"
                  "Per-seq throughput drops fast: MoE expert reads grow with B")
    axL.grid(alpha=0.3, which="both")
    axL.legend(loc="best", fontsize=8)

    # right: TTFT real points
    rb = [r["B"] for r in real]
    rttft = [r["ttft_p50_ms"] for r in real]
    rlabels = [r["regime"] for r in real]
    axR.bar(range(len(rb)), rttft, color=["#7e91a8", "#888", "#1f77b4"], width=0.6)
    for i, (b, t, lab) in enumerate(zip(rb, rttft, rlabels)):
        axR.text(i, t + 100, f"{t} ms", ha="center", fontsize=9, fontweight="bold")
    axR.set_xticks(range(len(rb)))
    axR.set_xticklabels([f"{lab}\nB={b}" for lab, b in zip(rlabels, rb)], fontsize=8)
    axR.set_ylabel("TTFT p50 (ms, burst of 24 short reqs)")
    axR.set_ylim(0, max(rttft) * 1.15)
    axR.set_title("Real measured TTFT p50 — burst of 24 short requests (REPORT.md §4.3)\n"
                  "no-batching catastrophic; B=8 with continuous beats static by ~30%")
    # MBU overlay on left axis context
    axR.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "throughput_ttft.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------- master plot

def plot_master(summary: dict):
    """One-shot 4-panel summary for the report."""
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))

    # (a) DRAM stacks
    ax = axes[0, 0]
    accounting = load_csv(RES / "dram_accounting.csv")
    precs = [r["precision"] for r in accounting]
    exp = [float(r["experts_GiB"]) for r in accounting]
    other = [float(r["total_weights_GiB"]) - e for r, e in zip(accounting, exp)]
    x = np.arange(len(precs))
    ax.bar(x, exp, color="#cf3a4a", label="experts")
    ax.bar(x, other, bottom=exp, color="#7e91a8", label="attn + embed/lm + router")
    for i in range(len(precs)):
        total = exp[i] + other[i]
        ax.text(i, total + 1, f"{total:.1f} GiB", ha="center", fontweight="bold", fontsize=10)
        ax.text(i, exp[i] / 2, f"{exp[i]:.1f}\n({100*exp[i]/total:.0f}% exp)",
                ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(precs)
    ax.set_ylabel("Weight DRAM (GiB)")
    ax.axhline(32, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.text(2.45, 32.4, "RTX 5090 (32 GiB)", fontsize=8, ha="right", alpha=0.6)
    ax.set_title("(a) DRAM breakdown — experts dominate (90–95%)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # (b) expert saturation
    ax = axes[0, 1]
    rows = summary["saturation"]
    B = [r["B"] for r in rows]
    ax.plot(B, [r["analytic_uniform_distinct_per_layer"] for r in rows],
            marker="o", color="#1f77b4", label="uniform (analytic)")
    ax.plot(B, [r["sim_skewed_mean_distinct"] for r in rows],
            marker="s", color="#cf3a4a", label=f"skewed sim (α={summary['simulation']['alpha']})")
    ax.fill_between(B, [r["sim_skewed_p5_distinct"] for r in rows],
                    [r["sim_skewed_p95_distinct"] for r in rows], color="#cf3a4a", alpha=0.15)
    ax.axhline(128, color="k", ls=":", lw=0.8); ax.text(B[-1], 122, "all 128", ha="right", fontsize=8, alpha=0.6)
    ax.set_xscale("log", base=2); ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B])
    ax.set_xlabel("Batch size B"); ax.set_ylabel("Distinct experts / layer / step")
    ax.set_title("(b) Expert saturation: B=1 → 8 experts; B=64 → ~all 128")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")

    # (c) working-set
    ax = axes[1, 0]
    rows = summary["batch_sweep"]
    B = [r["B"] for r in rows]
    ws = [r["working_set_expert_GiB"] for r in rows]
    res = rows[0]["resident_expert_GiB_int4"]
    ax.axhline(res, color="#444", lw=1.6, label=f"resident (all experts) = {res:.2f} GiB")
    ax.plot(B, ws, marker="o", color="#cf3a4a", label="working-set per step")
    for b, w in zip(B, ws):
        ax.annotate(f"{w:.1f}\n({100*w/res:.0f}%)", (b, w), textcoords="offset points", xytext=(0,5), ha="center", fontsize=8)
    ax.set_xscale("log", base=2); ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B])
    ax.set_xlabel("Batch size B"); ax.set_ylabel("Expert DRAM int4 (GiB)")
    ax.set_title("(c) Dynamic working-set: offloading savings vanish past B≈64")
    ax.set_ylim(0, res * 1.18); ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # (d) throughput
    ax = axes[1, 1]
    agg = [r["modeled_throughput_tok_s"] for r in rows]
    per_seq = [r["modeled_per_seq_tok_s"] for r in rows]
    ax.plot(B, agg, marker="o", color="#1f77b4", label="agg throughput (model)")
    ax.plot(B, per_seq, marker="s", color="#cf3a4a", label="per-seq throughput (model)")
    for r in summary["real_anchors"]:
        if r["regime"] in ("burst no-batching", "burst continuous"):
            ax.scatter([r["B"]], [r["agg_tok_s"]], s=140, marker="*", color="gold",
                       edgecolor="k", zorder=5)
            ax.annotate(f"real {r['regime']}\nB={r['B']}: {r['agg_tok_s']} tok/s",
                        (r["B"], r["agg_tok_s"]), textcoords="offset points",
                        xytext=(8, -4), fontsize=8)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B])
    ax.set_xlabel("Batch size B"); ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("(d) Throughput — calibrated to repo's real 5090 anchors")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Qwen3-Omni-30B-A3B Thinker — batch size vs experts vs DRAM vs throughput",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RES / "master.png", dpi=140)
    plt.close(fig)


def main():
    summary = load_summary()
    plot_dram_breakdown(summary)
    plot_expert_saturation(summary)
    plot_expert_heatmaps(summary, [1, 8, 32, 128])
    plot_working_set(summary)
    plot_throughput_ttft(summary)
    plot_master(summary)
    print(f"wrote PNGs to {RES}/")


if __name__ == "__main__":
    main()

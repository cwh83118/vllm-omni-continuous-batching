#!/usr/bin/env python3
"""Throughput hero charts for realistic cabin sweep.

Continuous batching's two benefits are TTFT (covered in plot_realistic_bars)
and throughput (here):
  (a) request throughput  — reqs completed per wall-clock second
  (b) output throughput   — tokens generated per wall-clock second
  (c) busy span           — total time to clear all queued work
  (d) mean per-stream decode tps — illustrates the per-stream / aggregate tradeoff

Figures produced:
  results/realistic_throughput_bars.png       — req/s + tok/s side-by-side, 4 scenarios
  results/realistic_throughput_tradeoff.png   — TTFT p50 vs req/s scatter (showing
                                                 continuous wins on BOTH axes)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = [
    ("cabin_solo",       "Solo · conservative\n(2.0 req/s sensor)"),
    ("cabin_solo_prod",  "Solo · production\n(3.8 req/s sensor)"),
    ("cabin_family",     "Family · conservative\n(2.0 req/s sensor)"),
    ("cabin_family_prod","Family · production\n(3.8 req/s sensor)"),
]
MODES = [
    ("none",           "none",       "#b71c1c"),
    ("static",         "static",     "#ef6c00"),
    ("static_vip",     "static+VIP", "#7b1fa2"),
    ("continuous",     "continuous", "#2e7d32"),
    ("continuous_pri", "cont+pri",   "#00695c"),
]


def load(scenario, mode):
    p = f"results/{scenario}_{mode}.json"
    with open(p) as f:
        return json.load(f)["summary"]


def metric_req_per_s(s):
    n = s["n_requests_total"]; bs = s.get("busy_span_s", 0)
    return n / bs if bs > 0 else 0


def metric_tok_per_s(s):
    return s.get("busy_output_tok_per_s", 0)


def metric_busy(s):
    return s.get("busy_span_s", 0)


def fig_throughput_grid():
    """2 rows × 4 cols: top row req/s, bottom row tok/s. Linear y."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey="row")
    for col, (sc_key, sc_title) in enumerate(SCENARIOS):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        labels = [m[1] for m in MODES]
        colors = [m[2] for m in MODES]
        rps = [metric_req_per_s(load(sc_key, m[0])) for m in MODES]
        tps = [metric_tok_per_s(load(sc_key, m[0])) for m in MODES]

        bars1 = ax_top.bar(range(len(labels)), rps, color=colors, alpha=0.92,
                           edgecolor="#222", linewidth=0.6)
        ax_top.set_title(sc_title, fontsize=11)
        ax_top.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars1, rps):
            ax_top.text(bar.get_x() + bar.get_width()/2, v + max(rps)*0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_top.set_ylim(0, max(rps) * 1.15)
        ax_top.set_xticks([])

        bars2 = ax_bot.bar(range(len(labels)), tps, color=colors, alpha=0.92,
                           edgecolor="#222", linewidth=0.6)
        for bar, v in zip(bars2, tps):
            ax_bot.text(bar.get_x() + bar.get_width()/2, v + max(tps)*0.02,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_bot.set_ylim(0, max(tps) * 1.15)
        ax_bot.set_xticks(range(len(labels)))
        ax_bot.set_xticklabels(labels, fontsize=10)
        ax_bot.grid(axis="y", alpha=0.3)

    axes[0, 0].set_ylabel("Request throughput\n(reqs/s)", fontsize=11)
    axes[1, 0].set_ylabel("Output throughput\n(tokens/s)", fontsize=11)
    fig.suptitle(
        "Throughput across 4 realistic cabin scenarios × 5 batching modes\n"
        "Top: requests completed per wall-clock second  ·  Bottom: total tokens generated per wall-clock second",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = "results/realistic_throughput_bars.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def fig_tradeoff():
    """Latency vs throughput scatter — show continuous Pareto-dominates."""
    fig, ax = plt.subplots(figsize=(11, 8))
    for sc_key, sc_title in SCENARIOS:
        for mkey, mname, c in MODES:
            s = load(sc_key, mkey)
            rps = metric_req_per_s(s)
            ttft = (s.get("interactive_ttft_p50_s") or 0) * 1000
            marker = {"cabin_solo": "o", "cabin_solo_prod": "s",
                      "cabin_family": "^", "cabin_family_prod": "D"}[sc_key]
            ax.scatter([rps], [ttft], s=180, marker=marker, color=c,
                       edgecolor="#222", linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Request throughput (reqs/s) — higher is better →", fontsize=12)
    ax.set_ylabel("Interactive TTFT p50 (ms, log) — lower is better ↓", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Latency vs throughput Pareto — continuous dominates BOTH axes",
                 fontsize=13)
    # Legend: mode colors
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    mode_handles = [Patch(color=c, label=mname) for _, mname, c in MODES]
    shape_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=12, label="Solo · conservative"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888", markersize=12, label="Solo · production"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#888", markersize=12, label="Family · conservative"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#888", markersize=12, label="Family · production"),
    ]
    leg1 = ax.legend(handles=mode_handles, loc="upper left", title="Mode", fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, loc="lower right", title="Scenario (marker shape)", fontsize=10)
    ax.annotate("← target zone\n(low latency + high throughput)",
                xy=(1.6, 300), xytext=(1.0, 100),
                fontsize=10, color="#2e7d32", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.5))
    fig.tight_layout()
    out = "results/realistic_throughput_tradeoff.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_throughput_grid()
    fig_tradeoff()

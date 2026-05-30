#!/usr/bin/env python3
"""Hero bar charts for realistic cabin sweep (cabin_solo / cabin_family × prod/cons × 5 modes).

Two figures:
  results/realistic_ttft_p50_4x5.png  — 4 scenarios × 5 modes, log-y to handle none catastrophe
  results/realistic_ttft_breakdown.png — single scenario (cabin_solo_prod) p50/p95/max breakdown
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


def fig1():
    """4 scenarios × 5 modes — interactive TTFT p50 (log y)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    for ax, (sc_key, sc_title) in zip(axes, SCENARIOS):
        labels = [m[1] for m in MODES]
        colors = [m[2] for m in MODES]
        vals_ms = []
        for mkey, _, _ in MODES:
            s = load(sc_key, mkey)
            v = s.get("interactive_ttft_p50_s")
            vals_ms.append(0 if v is None or v != v else v * 1000)
        bars = ax.bar(range(len(labels)), vals_ms, color=colors, alpha=0.92,
                      edgecolor="#222", linewidth=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10, rotation=0)
        ax.set_title(sc_title, fontsize=11)
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.set_yscale("log")
        # value labels
        for bar, v in zip(bars, vals_ms):
            if v <= 0:
                continue
            txt = f"{v:.0f}ms" if v < 10000 else f"{v/1000:.1f}s"
            ax.text(bar.get_x() + bar.get_width()/2, v * 1.05,
                    txt, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.margins(x=0.05)
        ax.set_ylim(10, 1_000_000)
    axes[0].set_ylabel("Interactive TTFT p50 (ms, log scale)", fontsize=11)
    fig.suptitle(
        "Interactive TTFT p50 across 4 realistic cabin scenarios × 5 batching modes\n"
        "(CX1-throttled 5090 @ BW≈68 GB/s; 0 errors across all 20 runs; same seed = byte-identical arrival sequence)",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = "results/realistic_ttft_p50_4x5.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def fig2():
    """cabin_solo_prod 5 modes — p50/p95/max breakdown."""
    sc_key = "cabin_solo_prod"
    KEYS = [
        ("interactive_ttft_p50_s", "Interactive TTFT p50"),
        ("interactive_ttft_p95_s", "Interactive TTFT p95"),
        ("interactive_ttft_max_s", "Interactive TTFT max"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    labels = [m[1] for m in MODES]
    colors = [m[2] for m in MODES]
    for ax, (key, title) in zip(axes, KEYS):
        vals_ms = []
        for mkey, _, _ in MODES:
            s = load(sc_key, mkey)
            v = s.get(key)
            vals_ms.append(0 if v is None or v != v else v * 1000)
        bars = ax.bar(range(len(labels)), vals_ms, color=colors, alpha=0.92,
                      edgecolor="#222", linewidth=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3, which="both")
        for bar, v in zip(bars, vals_ms):
            if v <= 0:
                continue
            txt = f"{v:.0f}ms" if v < 10000 else f"{v/1000:.1f}s"
            ax.text(bar.get_x() + bar.get_width()/2, v * 1.06,
                    txt, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.margins(x=0.05)
        ax.set_ylim(10, 1_000_000)
    axes[0].set_ylabel("ms (log scale)", fontsize=11)
    fig.suptitle(
        "cabin_solo_prod (3.8 req/s sensors + 8-turn dialogues) — Interactive TTFT × 5 batching modes\n"
        "p50 = typical | p95 = 1-in-20 worst | max = worst-case wait",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = "results/realistic_ttft_breakdown_solo_prod.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def fig3():
    """Highlight chart: arrival vs service, showing why none/static can't keep up."""
    fig, ax = plt.subplots(figsize=(14, 6))
    # cabin_solo_prod busy span (lower = better)
    sc_key = "cabin_solo_prod"
    labels = []
    busy_s = []
    inter_p50 = []
    colors = []
    for mkey, mname, c in MODES:
        s = load(sc_key, mkey)
        labels.append(mname)
        busy_s.append(s.get("busy_span_s", 0))
        v = s.get("interactive_ttft_p50_s") or 0
        inter_p50.append(v * 1000)
        colors.append(c)

    x = np.arange(len(labels))
    width = 0.4
    ax1 = ax
    bars1 = ax1.bar(x - width/2, inter_p50, width, color=colors, alpha=0.85,
                    edgecolor="#222", linewidth=0.6, label="Interactive TTFT p50 (ms)")
    ax1.set_yscale("log")
    ax1.set_ylim(10, 1_000_000)
    ax1.set_ylabel("Interactive TTFT p50 (ms, log)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    for bar, v in zip(bars1, inter_p50):
        if v <= 0: continue
        txt = f"{v:.0f}ms" if v < 10000 else f"{v/1000:.1f}s"
        ax1.text(bar.get_x() + bar.get_width()/2, v * 1.1,
                 txt, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, busy_s, width, color="#444", alpha=0.40,
                    edgecolor="#222", linewidth=0.6, label="Busy span (s)")
    ax2.set_ylabel("Busy span — total time to clear all 500 reqs (s)", fontsize=11)
    for bar, v in zip(bars2, busy_s):
        ax2.text(bar.get_x() + bar.get_width()/2, v + max(busy_s)*0.01,
                 f"{v:.0f}s", ha="center", va="bottom", fontsize=9, color="#444")

    fig.suptitle("cabin_solo_prod — Interactive TTFT p50 (left bars) and busy span (right bars)\n"
                 "Continuous batching: 6.3× faster user response AND ~2× shorter total processing",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = "results/realistic_user_vs_throughput.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1(); fig2(); fig3()

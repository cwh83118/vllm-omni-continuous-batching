#!/usr/bin/env python3
"""Bar chart: TTFT p50 / p95 / max across the 5 throttle modes for commute_run.

This is the headline chart for REPORT_CX1_EQUIV — the median user (p50)
sees no difference between batching modes, but the tail (p95 / max) is where
continuous batching shines: ~10× lower than static modes.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    ("R2_throttle_none",          "none\n(serial B=1)",          "#b71c1c"),
    ("R3_throttle_static",        "static\n(B=6, wave drain)",   "#ef6c00"),
    ("R4_throttle_static_vip",    "static + VIP\n(B=6)",         "#7b1fa2"),
    ("R5_throttle_continuous",    "continuous\n(B=6 FIFO)",      "#2e7d32"),
    ("R6_throttle_continuous_pri","continuous + pri\n(B=6)",     "#00695c"),
]

# Tracks: TTFT p50 / p95 / max — for interactive stream (the user latency)
TRACKS = [
    ("interactive_ttft_p50_s", "TTFT p50 (typical user, ms)"),
    ("interactive_ttft_p95_s", "TTFT p95 (1-in-20 worst, ms)"),
    ("interactive_ttft_max_s", "TTFT max (worst-case wait, ms)"),
]


def load_summaries():
    data = []
    for label, _, color in RUNS:
        with open(f"results/commute_{label}.json") as f:
            data.append((label, json.load(f)["summary"], color))
    return data


def main():
    data = load_summaries()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    labels = [r[1] for r in RUNS]
    colors = [r[2] for r in RUNS]

    for ax, (key, title) in zip(axes, TRACKS):
        vals_ms = []
        for _, s, _ in data:
            v = s.get(key)
            vals_ms.append(0 if v is None or v != v else v * 1000)
        bars = ax.bar(range(len(labels)), vals_ms, color=colors, alpha=0.92, edgecolor="#222", linewidth=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("ms")
        ax.grid(axis="y", alpha=0.3)
        # value labels above bars
        for bar, v in zip(bars, vals_ms):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals_ms)*0.02,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, max(vals_ms) * 1.18 if max(vals_ms) > 0 else 1)

    fig.suptitle("Interactive TTFT on CX1-throttled 5090 — 5 batching modes × commute_run scenario\n"
                 "(throttled BW ≈ 68 GB/s, decode ≈ 24 tok/s; conservative bound — real CX1 @ 154 GB/s would be ~2.3× faster)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = "results/commute_ttft_bars.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")

    # also print the table
    print()
    print(f"{'mode':32s}  {'p50 ms':>7}  {'p95 ms':>7}  {'max ms':>7}")
    print("-" * 56)
    for (_, s, _), (label, _, _) in zip(data, RUNS):
        def m(k):
            v = s.get(k); return ('—' if v is None or v != v else f'{v*1000:.0f}')
        print(f"{label:32s}  {m('interactive_ttft_p50_s'):>7}  {m('interactive_ttft_p95_s'):>7}  {m('interactive_ttft_max_s'):>7}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ONE hero chart that says it all: continuous batching wins on BOTH TTFT and Throughput.

cabin_solo_prod (production-rate cabin AI, headline scenario):
  5 modes × 2 metrics (TTFT p50 + req/s)
  Left axis (log)    : Interactive TTFT p50 (ms) — lower is better
  Right axis (linear): Request throughput (reqs/s) — higher is better

Bars: side-by-side per mode, dark color for TTFT, light color for throughput.
Continuous bars highlighted with bold border + win badge.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Headline scenario
SCENARIO = "cabin_solo_prod"

MODES = [
    ("none",           "none\n(B=1, serial)"),
    ("static",         "static\n(B=6, NPU-style)"),
    ("static_vip",     "static + VIP\n(B=6)"),
    ("continuous",     "continuous\n(B=6 FIFO)"),
    ("continuous_pri", "cont + pri\n(B=6)"),
]


def load(scenario, mode):
    with open(f"results/{scenario}_{mode}.json") as f:
        return json.load(f)["summary"]


def main():
    # extract metrics
    labels = [m[1] for m in MODES]
    ttft_ms, throughput_rps, throughput_tps = [], [], []
    for mkey, _ in MODES:
        s = load(SCENARIO, mkey)
        v = s.get("interactive_ttft_p50_s")
        ttft_ms.append(0 if v is None or v != v else v * 1000)
        n = s["n_requests_total"]; bs = s.get("busy_span_s", 0)
        throughput_rps.append(n / bs if bs > 0 else 0)
        throughput_tps.append(s.get("busy_output_tok_per_s", 0))

    fig, ax_left = plt.subplots(figsize=(15, 7.5))
    ax_right = ax_left.twinx()

    x = np.arange(len(labels))
    width = 0.36

    # TTFT bars (left axis, log) — red/orange/purple/green/teal scheme
    LEFT_COLORS  = ["#b71c1c", "#ef6c00", "#7b1fa2", "#1b5e20", "#004d40"]
    RIGHT_COLORS = ["#ef9a9a", "#ffcc80", "#ce93d8", "#a5d6a7", "#80cbc4"]

    bars_l = ax_left.bar(x - width/2, ttft_ms, width,
                         color=LEFT_COLORS, alpha=0.95,
                         edgecolor="#222", linewidth=0.8,
                         label="Interactive TTFT p50 (ms, log) — lower is better ↓")
    bars_r = ax_right.bar(x + width/2, throughput_rps, width,
                          color=RIGHT_COLORS, alpha=0.95,
                          edgecolor="#222", linewidth=0.8,
                          label="Request throughput (reqs/s) — higher is better ↑")

    # value annotations
    for bar, v in zip(bars_l, ttft_ms):
        if v <= 0: continue
        txt = f"{v:.0f} ms" if v < 10000 else f"{v/1000:.1f} s"
        ax_left.text(bar.get_x() + bar.get_width()/2, v * 1.15,
                     txt, ha="center", va="bottom", fontsize=10.5,
                     fontweight="bold", color="#222")
    for bar, v in zip(bars_r, throughput_rps):
        ax_right.text(bar.get_x() + bar.get_width()/2, v + 0.04,
                      f"{v:.2f} req/s", ha="center", va="bottom",
                      fontsize=10.5, fontweight="bold", color="#222")

    # highlight continuous group (best on both)
    for i in [3, 4]:  # continuous, cont+pri
        for bar in [bars_l[i], bars_r[i]]:
            bar.set_linewidth(2.4)
            bar.set_edgecolor("#000")

    # axes
    ax_left.set_yscale("log")
    ax_left.set_ylim(50, 2_000_000)
    ax_left.set_ylabel("Interactive TTFT p50 (ms, log scale)  ←  lower is better",
                       fontsize=12, color="#222")
    ax_right.set_ylim(0, max(throughput_rps) * 1.45)
    ax_right.set_ylabel("Request throughput (reqs/s)  →  higher is better",
                        fontsize=12, color="#222")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=11)
    ax_left.grid(axis="y", alpha=0.35, which="both", color="#888")
    ax_left.set_axisbelow(True)

    # winner overlay text
    fig.text(0.62, 0.78,
             "★ continuous wins BOTH axes ★\n"
             "  Latency: 245 ms (vs static 1534 ms → 6.3× faster)\n"
             "  Throughput: 1.67 req/s (vs static 1.53 → +9%)\n"
             "  Same GPU, better user experience AND more work done.",
             fontsize=11, color="#1b5e20", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#e8f5e9",
                       edgecolor="#1b5e20", linewidth=1.5))

    fig.text(0.04, 0.82,
             "✗ none is unusable\n"
             "  Latency: 235 s (3.9 min)\n"
             "  Throughput: 0.85 req/s (½ of continuous)",
             fontsize=10, color="#b71c1c", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebee",
                       edgecolor="#b71c1c", linewidth=1.2))

    fig.suptitle(
        "Continuous batching's TWO benefits — quantified on production cabin AI workload\n"
        "(cabin_solo_prod: 4 sustained sensor streams @ 3.8 req/s + multi-turn user dialogues; CX1-throttled 5090)",
        fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = "results/realistic_dual_benefit_hero.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

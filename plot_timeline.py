#!/usr/bin/env python3
"""Render side-by-side request timelines from cabin_demo.py JSON dumps.

Each request is drawn as a horizontal bar on a wall-clock x-axis:
  * the faint segment  (t_submit -> t_first_token)  = time spent waiting in the
    engine queue + prefill before the user sees anything (TTFT);
  * the solid segment  (t_first_token -> t_finish)  = generating / streaming tokens.
Rows are stacked in submission order; proactive vs interactive are coloured differently.

Usage:
  python plot_timeline.py                       # uses results/run_off.json + results/run_on.json
  python plot_timeline.py --off A.json --on B.json --out results/timeline_cabin.png --title "..."
"""
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

COL = {
    "proactive":   {"wait": "#f3c9b6", "gen": "#e8743b"},   # orange
    "interactive": {"wait": "#bcd4e6", "gen": "#2a6f97"},   # blue
}


def load(path):
    with open(path) as f:
        return json.load(f)


def draw_panel(ax, data, title):
    reqs = sorted(data["requests"], key=lambda r: (r["t_submit"], r["brain"], r["idx"]))
    xmax = max((r["t_finish"] for r in reqs if r["t_finish"]), default=1.0)
    for row, r in enumerate(reqs):
        y = len(reqs) - 1 - row  # newest at bottom
        c = COL[r["brain"]]
        t_s, t_ft, t_f = r["t_submit"], r["t_first_token"], r["t_finish"]
        if not t_f:
            continue
        if t_ft and t_ft > t_s:
            ax.barh(y, t_ft - t_s, left=t_s, height=0.8, color=c["wait"],
                    edgecolor="none", zorder=2)
        gen_start = t_ft if t_ft else t_s
        ax.barh(y, max(t_f - gen_start, 0.01), left=gen_start, height=0.8,
                color=c["gen"], edgecolor="none", zorder=3)
        # mark first-token instant
        if t_ft:
            ax.plot([t_ft], [y], marker="|", ms=9, mew=1.6, color="#222", zorder=4)
        # label interactive rows with their TTFT
        if r["brain"] == "interactive" and r.get("ttft_s") is not None:
            ax.text(t_f + xmax * 0.006, y, f"{r['ttft_s']*1000:.0f} ms",
                    va="center", ha="left", fontsize=6.5, color="#444")
    ax.set_xlim(0, xmax * 1.10)
    ax.set_ylim(-0.7, len(reqs) - 0.3)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_yticks([])
    s = data["summary"]
    sub = (f"interactive TTFT  p50={s['interactive_ttft_p50_s']*1000:.0f} ms  "
           f"p95={s['interactive_ttft_p95_s']*1000:.0f} ms  max={s['interactive_ttft_max_s']*1000:.0f} ms\n"
           f"throughput over busy span = {s['busy_output_tok_per_s']:.0f} tok/s   "
           f"({s['n_interactive']}+{s['n_proactive']} reqs, {s['total_output_tokens']} tok in {s['busy_span_s']:.1f} s)")
    ax.set_title(f"{title}\n{sub}", fontsize=9.5, loc="left")
    ax.grid(axis="x", alpha=0.25, zorder=0)


def legend(fig):
    handles = [
        Patch(facecolor=COL["interactive"]["gen"], label="interactive — generating"),
        Patch(facecolor=COL["interactive"]["wait"], label="interactive — queued / prefill (before 1st token = TTFT)"),
        Patch(facecolor=COL["proactive"]["gen"], label="proactive — generating"),
        Patch(facecolor=COL["proactive"]["wait"], label="proactive — queued / prefill"),
        Line2D([0], [0], marker="|", ms=10, mew=1.8, color="#222", ls="none", label="first token delivered"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default="results/run_off.json")
    ap.add_argument("--on", default="results/run_on.json")
    ap.add_argument("--out", default="results/timeline_cabin.png")
    ap.add_argument("--title", default="In-car cabin assistant — one vLLM-Omni (Qwen3-Omni-30B-A3B Thinker) engine, two brains")
    ap.add_argument("--off-label", default=None)
    ap.add_argument("--on-label", default=None)
    args = ap.parse_args()

    try:
        d_off, d_on = load(args.off), load(args.on)
    except FileNotFoundError as e:
        sys.exit(f"missing input: {e}")

    off_lbl = args.off_label or f"WITHOUT continuous batching  (max_num_seqs={d_off['max_num_seqs']})"
    on_lbl = args.on_label or f"WITH continuous batching  (max_num_seqs={d_on['max_num_seqs']})"

    n = max(len(d_off["requests"]), len(d_on["requests"]))
    fig, axes = plt.subplots(1, 2, figsize=(15, max(4.5, 0.22 * n + 1.8)), sharey=False)
    fig.suptitle(args.title, fontsize=12, y=0.995)
    draw_panel(axes[0], d_off, off_lbl)
    draw_panel(axes[1], d_on, on_lbl)
    legend(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")

    # text comparison
    a, b = d_off["summary"], d_on["summary"]
    print("\n  metric                                  | WITHOUT (seqs={:>2}) | WITH (seqs={:>2}) | factor".format(
        d_off["max_num_seqs"], d_on["max_num_seqs"]))
    print("  " + "-" * 86)
    def row(name, va, vb, unit="", inv=False, fmt="{:.3f}"):
        try:
            f = (va / vb) if not inv else (vb / va)
            fs = f"{f:6.1f}x"
        except ZeroDivisionError:
            fs = "   -  "
        print(f"  {name:<40s}| {fmt.format(va)+unit:>17s} | {fmt.format(vb)+unit:>14s} | {fs}")
    row("interactive TTFT p50",  a["interactive_ttft_p50_s"]*1000, b["interactive_ttft_p50_s"]*1000, " ms", fmt="{:.0f}")
    row("interactive TTFT p95",  a["interactive_ttft_p95_s"]*1000, b["interactive_ttft_p95_s"]*1000, " ms", fmt="{:.0f}")
    row("interactive TTFT max",  a["interactive_ttft_max_s"]*1000, b["interactive_ttft_max_s"]*1000, " ms", fmt="{:.0f}")
    row("interactive e2e p50",   a["interactive_e2e_p50_s"]*1000,  b["interactive_e2e_p50_s"]*1000,  " ms", fmt="{:.0f}")
    if a.get("proactive_e2e_max_s") == a.get("proactive_e2e_max_s") and a.get("proactive_e2e_max_s") is not None:
        row("proactive e2e max", a["proactive_e2e_max_s"]*1000,    b["proactive_e2e_max_s"]*1000,    " ms", fmt="{:.0f}")
    row("output throughput (busy span)", a["busy_output_tok_per_s"], b["busy_output_tok_per_s"], " tok/s", inv=True, fmt="{:.0f}")
    print()


if __name__ == "__main__":
    main()

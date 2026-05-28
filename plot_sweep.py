#!/usr/bin/env python3
"""Render batch-size sweep curves from N cabin_demo.py JSON dumps.

Takes all JSONs for ONE scenario (e.g. mixed_3agent) across modes {none, static,
continuous} × B {1, 2, 4, 8, 16}, and produces a 2×3 grid:

  top row    = interactive/agent TTFT  p50  /  p95  /  max   vs B
  bottom row = proactive TTFT          p50  /  p95  /  max   vs B

Three lines per axis (one per mode). `none` is flat (B has no effect there).

Optional --metric switches the metric family.

Usage:
  python plot_sweep.py --scenario mixed_3agent \
      --in results/run_mixed_3agent_*.json \
      --out results/sweep_mixed_3agent_ttft.png
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE_COLOR = {
    "none":       "#b71c1c",   # red
    "static":     "#ef6c00",   # orange
    "continuous": "#2e7d32",   # green
}
MODE_MARKER = {"none": "x", "static": "s", "continuous": "o"}

# Map --metric -> list of (subplot_title, summary_key, multiplier, unit)
# Each axis row is (interactive_or_agent, proactive); columns are p50/p95/max.
METRIC_SUITES = {
    "ttft": [
        ("Interactive/Agent TTFT p50", ("interactive_ttft_p50_s", "agent_ttft_p50_s"), 1000, "ms"),
        ("Interactive/Agent TTFT p95", ("interactive_ttft_p95_s", "agent_ttft_p95_s"), 1000, "ms"),
        ("Interactive/Agent TTFT max", ("interactive_ttft_max_s",), 1000, "ms"),
        ("Proactive TTFT p50",         ("proactive_ttft_p50_s",), 1000, "ms"),
        ("Proactive e2e p50",          ("proactive_e2e_p50_s",), 1000, "ms"),
        ("Proactive e2e max",          ("proactive_e2e_max_s",), 1000, "ms"),
    ],
    "e2e": [
        ("Interactive/Agent e2e p50", ("interactive_e2e_p50_s", "agent_e2e_p50_s"), 1000, "ms"),
        ("Interactive/Agent e2e p95", ("interactive_e2e_p95_s",), 1000, "ms"),
        ("Proactive e2e p50",         ("proactive_e2e_p50_s",), 1000, "ms"),
        ("Proactive e2e max",         ("proactive_e2e_max_s",), 1000, "ms"),
        ("Busy span",                 ("busy_span_s",), 1, "s"),
        ("Total output tokens",       ("total_output_tokens",), 1, "tok"),
    ],
    "queue_wait": [
        ("Interactive queue-wait p50", ("interactive_queue_wait_p50_s",), 1000, "ms"),
        ("Busy span",                  ("busy_span_s",), 1, "s"),
        ("# requests total",           ("n_requests_total",), 1, ""),
        ("# errors",                   ("n_errors",), 1, ""),
        ("# waves (static)",           ("n_waves",), 1, ""),
        ("Output throughput (busy)",   ("busy_output_tok_per_s",), 1, "tok/s"),
    ],
    "throughput": [
        ("Output throughput (busy span)", ("busy_output_tok_per_s",), 1, "tok/s"),
        ("Mean per-request decode tps",   ("mean_decode_tok_per_s",), 1, "tok/s"),
        ("Total output tokens",           ("total_output_tokens",), 1, "tok"),
        ("Busy span",                     ("busy_span_s",), 1, "s"),
        ("Mean audio seconds per req",    ("mean_audio_seconds",), 1, "s"),
        ("# agent tasks finished",        ("n_agent_tasks",), 1, ""),
    ],
}


def load(paths):
    """Load JSONs, group by mode and order by batch_size."""
    by_mode = collections.defaultdict(list)
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
            continue
        s = d.get("summary", {})
        mode = s.get("mode") or d.get("mode")
        if not mode:
            continue
        B = s.get("batch_size") or d.get("batch_size") or 1
        by_mode[mode].append((B, s, p))
    for k in by_mode:
        by_mode[k].sort(key=lambda x: x[0])
    return by_mode


def value_for(summary, keys, mul):
    """Try several summary keys in order; return first non-NaN scalar * multiplier."""
    for k in keys:
        v = summary.get(k)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv:  # NaN
            continue
        return fv * mul
    return None


def plot_one(ax, by_mode, title, keys, mul, unit):
    for mode, color in MODE_COLOR.items():
        rows = by_mode.get(mode, [])
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [value_for(r[1], keys, mul) for r in rows]
        # filter None
        xy = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not xy:
            continue
        xs2, ys2 = zip(*xy)
        ax.plot(xs2, ys2, color=color, marker=MODE_MARKER[mode], lw=1.6, ms=6, label=mode)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("client batch cap  B")
    ax.set_ylabel(unit or "")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="scenario name (only used for figure title)")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="JSON files for this scenario across modes×B (glob-expanded)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", choices=list(METRIC_SUITES.keys()), default="ttft")
    args = ap.parse_args()

    # expand globs (shell may have done it already, but tolerate raw patterns)
    paths = []
    for p in args.inputs:
        if any(c in p for c in "*?["):
            paths.extend(sorted(glob.glob(p)))
        else:
            paths.append(p)
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        sys.exit("no input JSONs found")

    by_mode = load(paths)
    if not by_mode:
        sys.exit("no parseable summaries")

    suite = METRIC_SUITES[args.metric]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for ax, (title, keys, mul, unit) in zip(axes.flat, suite):
        plot_one(ax, by_mode, title, keys, mul, unit)
    # one legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.985))

    fig.suptitle(f"Batch-size sweep — scenario={args.scenario} — metric={args.metric}",
                 fontsize=12, y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out} (scanned {len(paths)} JSONs)")


if __name__ == "__main__":
    main()

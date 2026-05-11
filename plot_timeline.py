#!/usr/bin/env python3
"""Render side-by-side request timelines from cabin_demo.py JSON dumps (2 or 3 panels).

Each request is one horizontal bar on a wall-clock x-axis:
  * faint segment  (t_submit -> t_first_token)  = waited in the queue + prefill, the user
    has seen nothing yet  (this length = TTFT);
  * solid segment  (t_first_token -> t_finish)  = generating / streaming tokens.
Rows are stacked in arrival order; proactive vs interactive are coloured differently.
For a `static` (fixed-batch / NPU-style) panel, faint vertical lines mark each wave start.

Usage:
  python plot_timeline.py --panels results/run_none.json results/run_static.json results/run_continuous.json \
      --out results/timeline_3way.png --title "In-car cabin assistant — none vs static vs continuous"
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
MODE_TITLE = {
    "none":       "(1) NO batching  — max 1 in flight (≡ max_num_seqs=1)",
    "static":     "(2) STATIC / fixed-batch  — NPU-style: a wave of ≤B runs to drain before the next wave can start",
    "continuous": "(3) CONTINUOUS batching  — ≤B in flight, refill the instant a slot frees (vLLM-Omni)",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def panel_label(d):
    m = d.get("mode")
    B = d.get("batch_size", "?")
    base = MODE_TITLE.get(m, m or d.get("config", "?"))
    if m in ("static", "continuous"):
        base = base.replace("≤B", f"≤{B}").replace("of B", f"of {B}")
    return base


def draw_panel(ax, data, title, shared_xmax=None):
    reqs = sorted(data["requests"], key=lambda r: (r["t_submit"], r["brain"], r["idx"]))
    xmax = shared_xmax or max((r["t_finish"] for r in reqs if r["t_finish"]), default=1.0)

    # wave-start vertical lines (static mode)
    if data.get("mode") == "static":
        waves = {}
        for r in reqs:
            w = r.get("wave_id", -1)
            if w is None or w < 0:
                continue
            t = r.get("t_admitted") or r.get("t_first_token") or r["t_submit"]
            waves[w] = min(waves.get(w, 1e9), t)
        for w, t in sorted(waves.items()):
            ax.axvline(t, color="#888", lw=0.8, ls=":", zorder=1)
            ax.text(t, len(reqs) - 0.2, f"w{w}", fontsize=6, color="#888", ha="left", va="bottom")

    for row, r in enumerate(reqs):
        y = len(reqs) - 1 - row  # newest at bottom
        c = COL.get(r["brain"], COL["interactive"])
        t_s, t_ft, t_f = r["t_submit"], r.get("t_first_token") or 0.0, r["t_finish"]
        if not t_f:
            continue
        if t_ft and t_ft > t_s:
            ax.barh(y, t_ft - t_s, left=t_s, height=0.8, color=c["wait"], edgecolor="none", zorder=2)
        gen_start = t_ft if t_ft else t_s
        ax.barh(y, max(t_f - gen_start, 0.01), left=gen_start, height=0.8, color=c["gen"], edgecolor="none", zorder=3)
        if t_ft:
            ax.plot([t_ft], [y], marker="|", ms=8, mew=1.4, color="#222", zorder=4)
        if r["brain"] == "interactive" and r.get("ttft_s") is not None:
            ms = r["ttft_s"] * 1000
            txt = f"{ms:.0f} ms" if ms < 1000 else f"{ms/1000:.1f} s"
            ax.text(t_f + xmax * 0.006, y, txt, va="center", ha="left", fontsize=6, color="#444")

    ax.set_xlim(0, xmax * 1.12)
    ax.set_ylim(-0.7, len(reqs) - 0.3 + (1.0 if data.get("mode") == "static" else 0.0))
    ax.set_xlabel("wall-clock time (s)")
    ax.set_yticks([])
    s = data["summary"]
    def fmt_ms(v):
        return f"{v*1000:.0f} ms" if v == v and v < 1.0 else (f"{v:.2f} s" if v == v else "—")
    sub = (f"interactive TTFT  p50={fmt_ms(s['interactive_ttft_p50_s'])}  "
           f"p95={fmt_ms(s['interactive_ttft_p95_s'])}  max={fmt_ms(s['interactive_ttft_max_s'])}\n"
           f"all reqs done in {s['busy_span_s']:.1f} s  ·  {s['busy_output_tok_per_s']:.0f} tok/s over that span"
           + (f"  ·  {s['n_waves']} waves" if s.get('n_waves') else ""))
    ax.set_title(f"{title}\n{sub}", fontsize=9, loc="left")
    ax.grid(axis="x", alpha=0.2, zorder=0)


def legend(fig):
    handles = [
        Patch(facecolor=COL["interactive"]["gen"], label="interactive — generating tokens"),
        Patch(facecolor=COL["interactive"]["wait"], label="interactive — queued / prefill (before 1st token = TTFT)"),
        Patch(facecolor=COL["proactive"]["gen"], label="proactive — generating tokens"),
        Patch(facecolor=COL["proactive"]["wait"], label="proactive — queued / prefill"),
        Line2D([0], [0], marker="|", ms=10, mew=1.8, color="#222", ls="none", label="first token delivered to the user"),
        Line2D([0], [0], color="#888", ls=":", lw=1.2, label="static-batch wave boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))


def print_table(datas):
    keys = [("interactive TTFT p50", "interactive_ttft_p50_s", 1000, "ms", "{:.0f}"),
            ("interactive TTFT p95", "interactive_ttft_p95_s", 1000, "ms", "{:.0f}"),
            ("interactive TTFT max", "interactive_ttft_max_s", 1000, "ms", "{:.0f}"),
            ("interactive e2e  p50", "interactive_e2e_p50_s", 1000, "ms", "{:.0f}"),
            ("interactive queue-wait p50", "interactive_queue_wait_p50_s", 1000, "ms", "{:.0f}"),
            ("proactive e2e max", "proactive_e2e_max_s", 1000, "ms", "{:.0f}"),
            ("all reqs done in (busy span)", "busy_span_s", 1, "s", "{:.2f}"),
            ("output throughput (busy span)", "busy_output_tok_per_s", 1, "tok/s", "{:.0f}")]
    cols = [d["summary"] for d in datas]
    names = [f"{d.get('mode','?')}(B={d.get('batch_size','?')})" for d in datas]
    w = 16
    print("\n  metric                          | " + " | ".join(f"{n:>{w}s}" for n in names))
    print("  " + "-" * (33 + len(names) * (w + 3)))
    for label, k, mul, unit, f in keys:
        vals = []
        for c in cols:
            v = c.get(k)
            vals.append(f"{f.format(v*mul)} {unit}" if v == v and v is not None else "—")
        print(f"  {label:<32s}| " + " | ".join(f"{v:>{w}s}" for v in vals))
    # factors relative to the LAST panel (assumed = continuous)
    base = cols[-1]
    print("\n  ratio vs " + names[-1] + ":")
    for label, k, mul, unit, f in keys:
        bv = base.get(k)
        if bv in (None, 0) or bv != bv:
            continue
        rs = []
        for c, n in zip(cols, names):
            v = c.get(k)
            if v in (None,) or v != v:
                rs.append("—"); continue
            # latency-ish keys: lower is better -> ratio = v / base ; throughput: higher better
            if "throughput" in k:
                rs.append(f"{(v/bv):.1f}x" if bv else "—")
            else:
                rs.append(f"{(v/bv):.1f}x" if bv else "—")
        print(f"    {label:<30s}: " + "  ".join(f"{n}={r}" for n, r in zip(names, rs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", required=True, help="JSON files, left to right (continuous last is recommended)")
    ap.add_argument("--out", default="results/timeline_3way.png")
    ap.add_argument("--title", default="In-car cabin assistant — three request-scheduling regimes on ONE vLLM-Omni engine")
    ap.add_argument("--share-x", action="store_true", help="use the same x-axis range for all panels")
    args = ap.parse_args()

    try:
        datas = [load(p) for p in args.panels]
    except FileNotFoundError as e:
        sys.exit(f"missing input: {e}")

    nrows = max(len(d["requests"]) for d in datas)
    npan = len(datas)
    fig, axes = plt.subplots(1, npan, figsize=(6.4 * npan, max(4.5, 0.22 * nrows + 2.0)))
    if npan == 1:
        axes = [axes]
    fig.suptitle(args.title, fontsize=12, y=0.995)
    xmax = max((r["t_finish"] for d in datas for r in d["requests"] if r["t_finish"]), default=1.0) if args.share_x else None
    for ax, d in zip(axes, datas):
        draw_panel(ax, d, panel_label(d), shared_xmax=xmax)
    legend(fig)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    print_table(datas)


if __name__ == "__main__":
    main()

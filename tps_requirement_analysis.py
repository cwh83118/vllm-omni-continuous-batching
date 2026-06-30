#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
座艙 AI 多步驟 Agentic 任務 — TPS 需求趨勢分析
=================================================

任務
----
- 一個任務會呼叫 N 次工具 (本題 N=3), 每次工具 decode 30 tokens, 最後再 decode 一段
  自然語言回覆 (TTS 第一個 chunk)。
- 端到端 deadline = 3.5 秒 (收到輸入 -> 全部算完 -> 產出第一個語音回覆的 token)。
- Agentic: 每一步 output 會 concat 到 context, 成為下一步 input 的尾段。

★ 序列 vs 併發 (指任務裡那 N 次「工具的 decode」) ★
----------------------------------------------------
- 序列 (Sequential): N 個工具一個接一個 decode -> 牆鐘要吐 N x 30 tokens。
- 併發 (Batch)    : N 個工具丟進同一個 batch 一起 decode -> 一輪同時吐 N 條的 token,
                    所以牆鐘只花「30 tokens」的時間 (與 N 無關!)。
  (前提: 這 N 個工具彼此獨立、可平行呼叫; 若嚴格 agentic 互相依賴就只能走序列。)

最後那段自然語言回覆是「一條」單獨生成 (無法平行), 兩種模式都要付 1 次。

SoC 性能 (輸入條件)
-------------------
- Prefill TPS: 單次 prefill token <= 1000 -> 4000 tok/s ; > 1000 -> 6000 tok/s
- ViT: 單張 480x320 影像 = 20 ms (不可 batch)
- Decode TPS: 未給 -> 這正是我們要「反推」的需求值
- Prefill / ViT 不可 batch; Decode 可 batch。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1) 參數區 (全部可調)
# ============================================================

# --- 任務結構 ---
N_TOOL_STEPS          = 3      # 本題工具呼叫次數 (= 圖上標出的實際任務點)
DECODE_TOK_PER_STEP   = 30     # 每次工具 decode 出的 token 數
FINAL_REPLY_TOK       = 30     # 最後自然語言回覆「第一個語音 chunk」的 token 數 (一句 ~30 tok)

# --- 輸入 ---
N_TEXT_TOKENS         = 400    # System Prompt 文字 token
N_IMAGES              = 6      # 圖片張數
IMG_W, IMG_H          = 480, 320

# --- SoC: Prefill ---
PREFILL_TPS_SMALL     = 4000
PREFILL_TPS_LARGE     = 6000
PREFILL_SIZE_THRESH   = 1000

# --- SoC: ViT ---
VIT_MS_PER_UNIT_IMG   = 20.0   # 每張「480x320 倍率=1」影像的 ViT 時間 (ms)
UNIT_IMG_PIXELS       = 480 * 320

# --- 任務 deadline ---
DEADLINE_S            = 3.5

# --- KV-cache / prefix caching: 步驟間只 prefill 新增尾段 ---
USE_KV_CACHE          = True

# --- (供圖上對照) 假想 SoC 單路 decode 速度, 換成你 SoC 實測 ---
SOC_DECODE_TPS_SINGLE = 50.0   # tok/s  <-- 你的 SoC 單流 decode


# ============================================================
# 2) Qwen ViT — Vision token 計算 (smart_resize)
# ============================================================
def qwen_vision_tokens(w, h, patch=14, merge=2,
                       min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28):
    """
    Qwen2-VL / Qwen2.5-VL: factor = patch*merge = 28, 長寬各 round 到 28 倍數,
    patch 數 = (H/14)*(W/14), 2x2 merge -> token = patch/4。
    """
    factor = patch * merge  # 28

    def round_by_factor(x, f):
        return max(f, round(x / f) * f)

    h_bar = round_by_factor(h, factor)
    w_bar = round_by_factor(w, factor)
    if h_bar * w_bar > max_pixels:
        beta = (h * w / max_pixels) ** 0.5
        h_bar = int(np.floor(h / beta / factor) * factor)
        w_bar = int(np.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (h * w)) ** 0.5
        h_bar = int(np.ceil(h * beta / factor) * factor)
        w_bar = int(np.ceil(w * beta / factor) * factor)

    grid_h, grid_w = h_bar // patch, w_bar // patch
    n_patches = grid_h * grid_w
    return n_patches // (merge * merge), (w_bar, h_bar), (grid_w, grid_h), n_patches


# ============================================================
# 3) 單次 prefill 時間 (依門檻選 TPS)
# ============================================================
def prefill_time_s(n_tokens):
    if n_tokens <= 0:
        return 0.0
    tps = PREFILL_TPS_SMALL if n_tokens <= PREFILL_SIZE_THRESH else PREFILL_TPS_LARGE
    return n_tokens / tps


# ============================================================
# 4) 固定成本 (ViT + Prefill) — 序列 / 併發都一樣 (輸入相同)
# ============================================================
def fixed_costs(N, verbose=False):
    """回傳 ViT 時間、prefill 時間、固定成本 F (秒), 及 vision tokens 等資訊。"""
    vtok_per_img, dims, grid, npatch = qwen_vision_tokens(IMG_W, IMG_H)
    vision_tokens = vtok_per_img * N_IMAGES

    ratio = (IMG_W * IMG_H) / UNIT_IMG_PIXELS
    vit_s = N_IMAGES * ratio * VIT_MS_PER_UNIT_IMG / 1000.0

    initial_input = N_TEXT_TOKENS + vision_tokens

    # prefill: 初始輸入 prefill 一次 (N 個工具共用同一段 context);
    # 加上 N 次工具 output 累積進 context 的尾段 (N x 30 tokens)。
    if USE_KV_CACHE:
        tail_tokens = N * DECODE_TOK_PER_STEP
        prefill_s = prefill_time_s(initial_input) + prefill_time_s(tail_tokens)
    else:
        # 無 KV-cache: 每步重 prefill 全文
        prefill_s, ctx = 0.0, initial_input
        prefill_s += prefill_time_s(ctx)
        for _ in range(N):
            ctx += DECODE_TOK_PER_STEP
            prefill_s += prefill_time_s(ctx)

    F = vit_s + prefill_s
    info = dict(vision_tokens=vision_tokens, initial_input=initial_input,
                vit_s=vit_s, prefill_s=prefill_s, F=F,
                vtok_per_img=vtok_per_img, dims=dims, grid=grid, npatch=npatch)
    if verbose:
        print("=" * 64)
        print(f"固定成本拆解 (N={N} 工具)")
        print("=" * 64)
        print(f"Qwen ViT: 480x320 -> resize {dims[0]}x{dims[1]}, grid {grid[0]}x{grid[1]} "
              f"= {npatch} patches -> {vtok_per_img} tok/張")
        print(f"Vision tokens : {vtok_per_img} x {N_IMAGES} = {vision_tokens}")
        print(f"初始 Input    : {N_TEXT_TOKENS} text + {vision_tokens} vision = {initial_input} tok")
        print(f"ViT 時間      : {N_IMAGES} x {VIT_MS_PER_UNIT_IMG:.0f}ms = {vit_s*1000:.1f} ms")
        print(f"Prefill 時間  : {prefill_s*1000:.1f} ms  (初始 {prefill_time_s(initial_input)*1000:.1f} "
              f"+ 尾段 {N}x30)")
        print(f"固定成本 F    : ViT+Prefill = {F*1000:.1f} ms")
        print(f"decode 預算   : {DEADLINE_S}s - F = {(DEADLINE_S-F)*1000:.1f} ms")
        print()
    return info


# ============================================================
# 5) Decode 需求 — 用「依賴波次 W」統一序列 / 部分平行 / 全平行
# ============================================================
#
# 把 N 個工具按依賴關係分成 W 個「波 (wave)」: 同一波彼此獨立 -> 可 batch 一輪
# (牆鐘只花 30 tok); 波與波之間有依賴 -> 必須序列。
#   W = N  -> 全序列   (每個工具自己一波)            tool decode = N*30
#   W = 1  -> 全平行   (所有工具同一波, 一輪吐完)      tool decode =   30
#   1<W<N  -> 部分平行 (例: 3 工具 = 2 平行 + 1 依賴 -> W=2)  tool decode = W*30
#
# 牆鐘 decode token = W*30 + 30(最後回覆, 單獨一條無法平行)
# 需求 (單流) decode TPS = (W*30 + 30) / (T - F)
#
def waves_for(mode, N):
    """回傳該模式的波次數 W。partial: 成對平行 -> ceil(N/2) 波。"""
    if mode == "seq":      return N                 # 全序列
    if mode == "batch":    return 1                 # 全平行
    if mode == "partial":  return int(np.ceil(N / 2))  # 部分平行 (成對)
    raise ValueError(mode)

def decode_tokens_for_waves(W):
    return W * DECODE_TOK_PER_STEP + FINAL_REPLY_TOK

def required_tps(decode_tokens, F, T=DEADLINE_S):
    denom = T - F
    return np.inf if denom <= 0 else decode_tokens / denom


# ============================================================
# 6) 主程式
# ============================================================
def main():
    T = DEADLINE_S
    fixed_costs(N_TOOL_STEPS, verbose=True)

    MODES = [("seq", "Sequential (W=N)", "#d62728"),
             ("partial", "Partial (W=ceil(N/2))", "#ff7f0e"),
             ("batch", "Batch (W=1)", "#2ca02c")]

    print("=" * 84)
    print("序列 / 部分平行 / 全平行 的 decode TPS 需求 (deadline 3.5s)")
    print("=" * 84)
    print(f"{'N工具':>5} | {'budget':>8} | "
          f"{'序列(W=N)':>12} | {'部分(W=⌈N/2⌉)':>14} | {'全平行(W=1)':>12}")
    print("-" * 84)
    for N in range(1, 9):
        F = fixed_costs(N)["F"]
        budget = T - F
        cells = []
        for mode, _, _ in MODES:
            W = waves_for(mode, N)
            dt = decode_tokens_for_waves(W)
            tps = required_tps(dt, F)
            cells.append(f"{W}波 {dt}t {tps:>4.0f}t/s")
        mark = "  <- 本題" if N == N_TOOL_STEPS else ""
        print(f"{N:>5} | {budget*1000:>5.0f} ms | {cells[0]:>12} | {cells[1]:>14} | {cells[2]:>12}{mark}")
    print()

    info3 = fixed_costs(N_TOOL_STEPS)
    F3 = info3["F"]
    print(f"[本題 N={N_TOOL_STEPS}]  budget = {(T-F3)*1000:.0f} ms,  SoC 單流 {SOC_DECODE_TPS_SINGLE:.0f} tok/s")
    for mode, label, _ in MODES:
        W = waves_for(mode, N_TOOL_STEPS)
        dt = decode_tokens_for_waves(W)
        tps = required_tps(dt, F3)
        print(f"  {label:<22}: {W} 波, tool {W*DECODE_TOK_PER_STEP}+final {FINAL_REPLY_TOK}"
              f"={dt} tok -> 需求 {tps:.1f} tok/s  "
              f"[{'OK' if SOC_DECODE_TPS_SINGLE>=tps else 'FAIL'}]")
    print()

    # ================= 畫圖 =================
    Ns = np.arange(1, 9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- 左圖: 需求 decode TPS vs 工具數 N, 三條線 ----
    for mode, label, color in MODES:
        ys = np.array([required_tps(decode_tokens_for_waves(waves_for(mode, n)),
                                    fixed_costs(n)["F"]) for n in Ns])
        ax1.plot(Ns, ys, "o-", color=color, lw=2.4, label=label)
        # 標出 N=3 的數值
        yv = required_tps(decode_tokens_for_waves(waves_for(mode, N_TOOL_STEPS)), F3)
        ax1.annotate(f"{yv:.0f}", xy=(N_TOOL_STEPS, yv), xytext=(N_TOOL_STEPS+0.12, yv),
                     color=color, fontsize=10, fontweight="bold", va="center")
    ax1.axhline(SOC_DECODE_TPS_SINGLE, color="black", lw=1.6, ls="--",
                label=f"SoC single-stream decode = {SOC_DECODE_TPS_SINGLE:.0f}")
    ax1.axvline(N_TOOL_STEPS, color="gray", lw=1.3, ls=":")
    ax1.text(N_TOOL_STEPS + 0.08, 12, f"this task\nN={N_TOOL_STEPS}", color="gray", fontsize=9)
    ax1.set_xlabel("N  (number of tool calls in the task)", fontsize=12)
    ax1.set_ylabel("Required decode TPS (tokens/s)", fontsize=12)
    ax1.set_title("Required decode-TPS vs #tools\nPartial-parallel sits between Sequential and Batch",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="upper left")

    # ---- 右圖: N=3 的時間軸拆解 (三種模式, 用 SoC 單流速度換算) ----
    def timeline(tool_wall_tokens):
        return (info3["vit_s"] * 1000, info3["prefill_s"] * 1000,
                tool_wall_tokens / SOC_DECODE_TPS_SINGLE * 1000,
                FINAL_REPLY_TOK / SOC_DECODE_TPS_SINGLE * 1000)

    segs = [("ViT", "#8c564b"), ("Prefill", "#1f77b4"),
            ("Tool decode", "#d62728"), ("Final reply decode", "#9467bd")]
    bars = []  # (y, name, tool_wall)
    for i, (mode, label, _) in enumerate(MODES):
        W = waves_for(mode, N_TOOL_STEPS)
        bars.append((len(MODES) - 1 - i, label.split(" (")[0], W * DECODE_TOK_PER_STEP))
    for y, name, tool_wall in bars:
        left = 0
        for (lab, color), v in zip(segs, timeline(tool_wall)):
            ax2.barh(y, v, left=left, color=color, edgecolor="white",
                     label=lab if y == bars[0][0] else None)
            if v > 130:
                ax2.text(left + v/2, y, f"{v:.0f}", ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold")
            left += v
        ax2.text(left + 30, y, f"{name}: {left:.0f} ms", va="center", fontsize=9.5, fontweight="bold")
    ax2.axvline(DEADLINE_S * 1000, color="black", lw=2, ls="--")
    ax2.text(DEADLINE_S*1000 - 30, len(MODES)-0.4, f"deadline {DEADLINE_S*1000:.0f} ms",
             ha="right", color="black", fontsize=10, fontweight="bold")
    ax2.set_yticks([b[0] for b in bars]); ax2.set_yticklabels([b[1] for b in bars], fontsize=10)
    ax2.set_xlabel(f"wall-clock time (ms)  @ SoC decode {SOC_DECODE_TPS_SINGLE:.0f} tok/s", fontsize=12)
    ax2.set_title(f"Timeline breakdown for this task (N={N_TOOL_STEPS} tools)\n"
                  "tool-decode = waves x 30 tok  (90 / 60 / 30)",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.6, len(MODES) - 0.1)
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.grid(True, axis="x", alpha=0.3)

    txt = (f"Inputs: {N_IMAGES} imgs (Qwen ViT {info3['vision_tokens']} vis-tok) + {N_TEXT_TOKENS} txt "
           f"= {info3['initial_input']} tok  |  ViT {info3['vit_s']*1000:.0f}ms + "
           f"Prefill {info3['prefill_s']*1000:.0f}ms = F {F3*1000:.0f}ms  |  "
           f"tool 30 tok/call, final {FINAL_REPLY_TOK} tok  |  T={T}s  |  KV-cache "
           f"{'ON' if USE_KV_CACHE else 'OFF'}")
    fig.suptitle("Cabin AI — 3-step Agentic task: Sequential / Partial / Batch tool-decode TPS requirement",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02, txt, ha="center", fontsize=9, family="monospace")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.15, wspace=0.22)

    out = "tps_requirement_trend.png"
    fig.savefig(out, dpi=130)
    print(f"已輸出趨勢圖: {out}")


if __name__ == "__main__":
    main()

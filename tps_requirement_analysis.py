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
# 5) Decode 需求 — 序列 vs 併發 batch
# ============================================================
#
# decode 預算時間: budget = T - F
#
# 牆鐘要 decode 的 token 數 (這是 序列/併發 唯一的差別):
#   序列 Sequential : N 個工具逐一 decode  = N x 30  ; 再加最後回覆 30  -> N*30 + 30
#   併發 Batch      : N 個工具一輪同時 decode = 30    ; 再加最後回覆 30  ->   30 + 30
#
# 需求 (單流) decode TPS = 牆鐘 decode token / budget
#   序列: (N*30 + 30) / budget   -> 隨 N 線性上升
#   併發: (  30 + 30) / budget   -> 與 N 無關 (近水平)
#   工具階段差異倍率 = N   (N*30 vs 30)
#
def seq_decode_tokens(N):
    return N * DECODE_TOK_PER_STEP + FINAL_REPLY_TOK

def batch_decode_tokens(N):
    return DECODE_TOK_PER_STEP + FINAL_REPLY_TOK

def required_tps(decode_tokens, F, T=DEADLINE_S):
    denom = T - F
    return np.inf if denom <= 0 else decode_tokens / denom


# ============================================================
# 6) 主程式
# ============================================================
def main():
    T = DEADLINE_S
    fixed_costs(N_TOOL_STEPS, verbose=True)

    print("=" * 78)
    print("序列 vs 併發 batch 的 decode TPS 需求 (deadline 3.5s)")
    print("=" * 78)
    print(f"{'N工具':>5} | {'F(ViT+prefill)':>13} | {'budget':>8} | "
          f"{'序列decode':>9} {'需求TPS':>8} | {'併發decode':>9} {'需求TPS':>8} | {'倍率':>4}")
    print("-" * 88)
    rows = []
    for N in range(1, 9):
        info = fixed_costs(N)
        F = info["F"]
        budget = T - F
        sd, bd = seq_decode_tokens(N), batch_decode_tokens(N)
        stps = required_tps(sd, F)
        btps = required_tps(bd, F)
        rows.append((N, F, budget, sd, bd, stps, btps))
        mark = "  <- 本題" if N == N_TOOL_STEPS else ""
        if budget <= 0:
            print(f"{N:>5} | {F*1000:>10.0f} ms | {'<=0':>8} | {sd:>9} {'INF':>8} | "
                  f"{bd:>9} {'INF':>8} | {'-':>4}{mark}")
        else:
            print(f"{N:>5} | {F*1000:>10.0f} ms | {budget*1000:>5.0f} ms | "
                  f"{sd:>6} tok {stps:>6.0f} | {bd:>6} tok {btps:>6.0f} | "
                  f"{sd/bd:>3.1f}x{mark}")
    print()

    # 針對本題 N=3 的結論
    info3 = fixed_costs(N_TOOL_STEPS)
    F3 = info3["F"]
    b3 = T - F3
    s3 = required_tps(seq_decode_tokens(N_TOOL_STEPS), F3)
    bt3 = required_tps(batch_decode_tokens(N_TOOL_STEPS), F3)
    print(f"[本題 N={N_TOOL_STEPS}]  budget = {b3*1000:.0f} ms")
    print(f"  序列: decode {seq_decode_tokens(N_TOOL_STEPS)} tok -> 需求 {s3:.1f} tok/s")
    print(f"  併發: decode {batch_decode_tokens(N_TOOL_STEPS)} tok -> 需求 {bt3:.1f} tok/s")
    print(f"  SoC 單流 {SOC_DECODE_TPS_SINGLE:.0f} tok/s -> "
          f"序列 {'OK' if SOC_DECODE_TPS_SINGLE>=s3 else 'FAIL'} / "
          f"併發 {'OK' if SOC_DECODE_TPS_SINGLE>=bt3 else 'FAIL'}")
    print()

    # ================= 畫圖 =================
    Ns = np.arange(1, 9)
    seq_tps = np.array([required_tps(seq_decode_tokens(n), fixed_costs(n)["F"]) for n in Ns])
    bat_tps = np.array([required_tps(batch_decode_tokens(n), fixed_costs(n)["F"]) for n in Ns])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- 左圖: 需求 decode TPS vs 工具數 N ----
    ax1.plot(Ns, seq_tps, "o-", color="#d62728", lw=2.4,
             label="SEQUENTIAL: (N*30 + 30) / budget")
    ax1.plot(Ns, bat_tps, "o-", color="#2ca02c", lw=2.4,
             label="BATCH (parallel tools): (30 + 30) / budget")
    ax1.axhline(SOC_DECODE_TPS_SINGLE, color="#ff7f0e", lw=2, ls="--",
                label=f"SoC single-stream decode = {SOC_DECODE_TPS_SINGLE:.0f}")
    ax1.axvline(N_TOOL_STEPS, color="gray", lw=1.3, ls=":")
    ax1.text(N_TOOL_STEPS + 0.07, ax1.get_ylim()[0] if False else 12,
             f"this task\nN={N_TOOL_STEPS}", color="gray", fontsize=9)
    # 標 N=3 的兩點數值
    ax1.annotate(f"{seq_tps[N_TOOL_STEPS-1]:.0f}", xy=(N_TOOL_STEPS, seq_tps[N_TOOL_STEPS-1]),
                 xytext=(N_TOOL_STEPS+0.12, seq_tps[N_TOOL_STEPS-1]*1.06),
                 color="#d62728", fontsize=10, fontweight="bold")
    ax1.annotate(f"{bat_tps[N_TOOL_STEPS-1]:.0f}", xy=(N_TOOL_STEPS, bat_tps[N_TOOL_STEPS-1]),
                 xytext=(N_TOOL_STEPS+0.12, bat_tps[N_TOOL_STEPS-1]*0.80),
                 color="#2ca02c", fontsize=10, fontweight="bold")
    ax1.set_xlabel("N  (number of tool calls in the task)", fontsize=12)
    ax1.set_ylabel("Required decode TPS (tokens/s)", fontsize=12)
    ax1.set_title("Required decode-TPS vs #tools\nSequential grows ~linearly, Batch stays flat",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="upper left")

    # ---- 右圖: N=3 的時間軸拆解 (用 SoC 單流速度換算 decode 時間) ----
    def timeline(decode_tool_tokens):
        vit = info3["vit_s"] * 1000
        pf  = info3["prefill_s"] * 1000
        td  = decode_tool_tokens / SOC_DECODE_TPS_SINGLE * 1000
        fd  = FINAL_REPLY_TOK / SOC_DECODE_TPS_SINGLE * 1000
        return vit, pf, td, fd

    seq_tool_wall = N_TOOL_STEPS * DECODE_TOK_PER_STEP   # 90
    bat_tool_wall = DECODE_TOK_PER_STEP                  # 30
    segs = [("ViT", "#8c564b"), ("Prefill", "#1f77b4"),
            ("Tool decode", "#d62728"), ("Final reply decode", "#9467bd")]
    data = {
        "Sequential": timeline(seq_tool_wall),
        "Batch":      timeline(bat_tool_wall),
    }
    ypos = {"Sequential": 1, "Batch": 0}
    for name, vals in data.items():
        left = 0
        for (label, color), v in zip(segs, vals):
            ax2.barh(ypos[name], v, left=left, color=color, edgecolor="white",
                     label=label if name == "Sequential" else None)
            if v > 120:
                ax2.text(left + v/2, ypos[name], f"{v:.0f}", ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold")
            left += v
        ax2.text(left + 30, ypos[name], f"total {left:.0f} ms",
                 va="center", fontsize=10, fontweight="bold")
    ax2.axvline(DEADLINE_S * 1000, color="black", lw=2, ls="--")
    ax2.text(DEADLINE_S*1000 - 30, 1.45, f"deadline {DEADLINE_S*1000:.0f} ms",
             ha="right", color="black", fontsize=10, fontweight="bold")
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Batch", "Sequential"], fontsize=11)
    ax2.set_xlabel(f"wall-clock time (ms)  @ SoC decode {SOC_DECODE_TPS_SINGLE:.0f} tok/s", fontsize=12)
    ax2.set_title(f"Timeline breakdown for this task (N={N_TOOL_STEPS} tools)\n"
                  "Batch shrinks the tool-decode block 3x -> 30 tok",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.6, 1.7)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, axis="x", alpha=0.3)

    txt = (f"Inputs: {N_IMAGES} imgs (Qwen ViT {info3['vision_tokens']} vis-tok) + {N_TEXT_TOKENS} txt "
           f"= {info3['initial_input']} tok  |  ViT {info3['vit_s']*1000:.0f}ms + "
           f"Prefill {info3['prefill_s']*1000:.0f}ms = F {F3*1000:.0f}ms  |  "
           f"tool 30 tok/call, final {FINAL_REPLY_TOK} tok  |  T={T}s  |  KV-cache "
           f"{'ON' if USE_KV_CACHE else 'OFF'}")
    fig.suptitle("Cabin AI — 3-step Agentic task: Sequential vs Batch tool-decode TPS requirement",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02, txt, ha="center", fontsize=9, family="monospace")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.15, wspace=0.22)

    out = "tps_requirement_trend.png"
    fig.savefig(out, dpi=130)
    print(f"已輸出趨勢圖: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
座艙 AI 多步驟 Agentic 任務 — TPS 需求趨勢分析
=================================================

情境
----
- 3 步驟任務 = 連續呼叫 3 次工具, 最後合成第一個自然語音回覆 (TTS 第一個 chunk)
- 端到端 deadline = 3.5 秒 (從收到輸入 -> 算完所有事 -> 產出第一個語音回覆所需的 token)
- Agentic: 每一步的 output 會 concat 到 context, 成為下一步 input 的尾段
- 比較兩種服務方式:
    * 序列 (Sequential)        : N 個任務一個一個做
    * 併發 (Concurrent, N 路)  : N 個任務同時做 (Decode 可 batch, Prefill 不行)

SoC 性能 (輸入條件)
-------------------
- Prefill TPS: 一次 prefill 的 token 數 <= 1000 -> 4000 tok/s ; > 1000 -> 6000 tok/s
- ViT: 單張 480x320 影像 = 20 ms (不可 batch)
- Decode TPS: 未給 -> 這正是我們要 "反推" 的需求值

關鍵物理事實 (用於併發推導)
---------------------------
- Prefill 目前無法 batch  -> N 路的 prefill 與 ViT 會被「序列化」, 時間 = N x 單路
- Decode 可以 batch        -> 一個 batch step 同時替 N 條序列各吐 1 個 token,
                              權重只從記憶體讀一次 -> 聚合吞吐 (aggregate TPS) ~ 隨 N 近線性成長
                              直到 compute-bound 撞到 roofline 才飽和
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1) 參數區 (全部可調)
# ============================================================

# --- 任務結構 ---
N_TOOL_STEPS          = 3      # 呼叫工具次數 (步驟數)
DECODE_TOK_PER_STEP   = 30     # 每一步 (每次工具呼叫) 模型 decode 出的 token 數
FINAL_REPLY_TOK       = 30     # 最後自然語言回覆 "第一個語音 chunk" 需要的 token 數 (一句話 ~ 30 tok)

# --- 輸入 ---
N_TEXT_TOKENS         = 400    # System Prompt 文字 token
N_IMAGES              = 6      # 圖片張數
IMG_W, IMG_H          = 480, 320

# --- SoC: Prefill ---
PREFILL_TPS_SMALL     = 4000   # 單次 prefill token <= 門檻時的 TPS
PREFILL_TPS_LARGE     = 6000   # 單次 prefill token >  門檻時的 TPS
PREFILL_SIZE_THRESH   = 1000   # 門檻 (tokens)

# --- SoC: ViT ---
VIT_MS_PER_UNIT_IMG   = 20.0   # 每張「480x320 倍率=1」影像的 ViT 時間 (ms)
UNIT_IMG_PIXELS       = 480 * 320

# --- 任務 deadline ---
DEADLINE_S            = 3.5

# --- 是否假設 KV-cache / prefix caching (步驟間只 prefill 新增的尾段) ---
USE_KV_CACHE          = True

# --- (僅供圖上對照) 假想 SoC 的 decode 能力, 你可填真實量測值 ---
#  single-stream decode TPS, 以及 batch 後聚合吞吐的 roofline 上限
SOC_DECODE_TPS_SINGLE = 50.0    # 單路 decode 速度 (tok/s)  <-- 換成你 SoC 實測
SOC_DECODE_TPS_ROOF   = 1200.0  # batch 後聚合 decode 吞吐上限 (tok/s) <-- 換成你 SoC 實測


# ============================================================
# 2) Qwen ViT — Vision token 計算 (smart_resize)
# ============================================================
def qwen_vision_tokens(w, h, patch=14, merge=2,
                       min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28):
    """
    Qwen2-VL / Qwen2.5-VL 的 vision token 計算方式。
    - factor = patch * merge = 28, 影像長寬各 round 到 28 的倍數 (smart_resize)
    - patch 數 = (H/14) * (W/14)
    - 2x2 spatial merge -> token 數 = patch 數 / 4
    """
    factor = patch * merge  # 28

    def round_by_factor(x, f):
        return max(f, round(x / f) * f)

    h_bar = round_by_factor(h, factor)
    w_bar = round_by_factor(w, factor)

    # (本題影像很小, min/max pixels 不會觸發, 保留完整邏輯)
    if h_bar * w_bar > max_pixels:
        beta = (h * w / max_pixels) ** 0.5
        h_bar = int(np.floor(h / beta / factor) * factor)
        w_bar = int(np.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (h * w)) ** 0.5
        h_bar = int(np.ceil(h * beta / factor) * factor)
        w_bar = int(np.ceil(w * beta / factor) * factor)

    grid_h = h_bar // patch
    grid_w = w_bar // patch
    n_patches = grid_h * grid_w
    n_tokens = n_patches // (merge * merge)
    return n_tokens, (w_bar, h_bar), (grid_w, grid_h), n_patches


# ============================================================
# 3) 單次 prefill 時間 (依門檻選 TPS)
# ============================================================
def prefill_time_s(n_tokens):
    if n_tokens <= 0:
        return 0.0
    tps = PREFILL_TPS_SMALL if n_tokens <= PREFILL_SIZE_THRESH else PREFILL_TPS_LARGE
    return n_tokens / tps


# ============================================================
# 4) 單路 (single path) 成本拆解
# ============================================================
def single_path_costs(verbose=True):
    # --- vision tokens ---
    vtok_per_img, dims, grid, npatch = qwen_vision_tokens(IMG_W, IMG_H)
    vision_tokens = vtok_per_img * N_IMAGES

    # --- ViT 時間: 每張影像相對 480x320 的倍率 x 20ms ---
    ratio = (IMG_W * IMG_H) / UNIT_IMG_PIXELS          # 本題=1.0
    vit_s = N_IMAGES * ratio * VIT_MS_PER_UNIT_IMG / 1000.0

    # --- 初始輸入 token = 文字 + vision ---
    initial_input = N_TEXT_TOKENS + vision_tokens

    # --- prefill 時間 (含 agentic 各步) ---
    prefill_phases = []
    if USE_KV_CACHE:
        # 步驟間只需 prefill 新增的尾段 (前一步 output / 工具結果)
        prefill_phases.append(("initial", initial_input, prefill_time_s(initial_input)))
        for i in range(N_TOOL_STEPS):
            tail = DECODE_TOK_PER_STEP   # 上一步 output concat 進來的尾段
            prefill_phases.append((f"tail_step{i+1}", tail, prefill_time_s(tail)))
    else:
        # 無 KV-cache: 每一步把累積 context 整段重 prefill
        ctx = initial_input
        prefill_phases.append(("step1_full", ctx, prefill_time_s(ctx)))
        for i in range(N_TOOL_STEPS):
            ctx += DECODE_TOK_PER_STEP
            prefill_phases.append((f"step{i+2}_full", ctx, prefill_time_s(ctx)))

    prefill_s = sum(p[2] for p in prefill_phases)

    # --- decode token 總數: 3 次工具呼叫 + 最後回覆第一個 chunk ---
    decode_tokens = N_TOOL_STEPS * DECODE_TOK_PER_STEP + FINAL_REPLY_TOK

    # 單路「非 decode」固定成本 (序列化的部分): prefill + ViT
    fixed_s = prefill_s + vit_s

    if verbose:
        print("=" * 64)
        print("單路成本拆解 (single path)")
        print("=" * 64)
        print(f"Qwen ViT: 每張 480x320 -> resize {dims[0]}x{dims[1]}, "
              f"grid {grid[0]}x{grid[1]} = {npatch} patches -> {vtok_per_img} tokens")
        print(f"Vision tokens : {vtok_per_img} x {N_IMAGES} 張 = {vision_tokens}")
        print(f"Text tokens   : {N_TEXT_TOKENS}")
        print(f"初始 Input    : {initial_input} tokens")
        print(f"ViT 時間      : {N_IMAGES} 張 x {VIT_MS_PER_UNIT_IMG:.0f}ms = {vit_s*1000:.1f} ms")
        print(f"KV-cache      : {'ON (步驟間只 prefill 尾段)' if USE_KV_CACHE else 'OFF (每步重 prefill 全文)'}")
        print("-- prefill 各階段 --")
        for name, ntok, t in prefill_phases:
            tps = PREFILL_TPS_SMALL if ntok <= PREFILL_SIZE_THRESH else PREFILL_TPS_LARGE
            print(f"   {name:<14} {ntok:>5} tok @ {tps} tps = {t*1000:7.2f} ms")
        print(f"Prefill 總時間: {prefill_s*1000:.1f} ms")
        print(f"固定成本 F    : prefill+ViT = {fixed_s*1000:.1f} ms")
        print(f"Decode tokens : {N_TOOL_STEPS}x{DECODE_TOK_PER_STEP} + {FINAL_REPLY_TOK} "
              f"= {decode_tokens} tokens")
        print()

    return dict(vision_tokens=vision_tokens, initial_input=initial_input,
                vit_s=vit_s, prefill_s=prefill_s, fixed_s=fixed_s,
                decode_tokens=decode_tokens)


# ============================================================
# 5) TPS 需求公式
# ============================================================
#
# 令 F = 單路 (prefill + ViT) 序列化固定成本, D = 單路 decode token 數, T = deadline(3.5s)
#
# 【序列 Sequential】 任務一個一個做, 每個任務各自要在 T 內完成:
#       需求 decode TPS = D / (T - F)            (與 N 無關, 但服務 N 個人總延遲 = N x T)
#
# 【併發 Concurrent (N 路)】 Prefill/ViT 不能 batch -> 序列化 = N x F
#   剩餘時間給「可 batch 的 decode」: T - N x F
#   N 路 decode 總工作量 = N x D tokens, 需在剩餘時間內由 batch 引擎吐完:
#       需求「聚合」decode TPS  A(N) = (N x D) / (T - N x F)
#       需求「單路」decode TPS      = A(N)/N = D / (T - N x F)
#   可行條件: N x F < T  (否則光 prefill+ViT 就吃光 3.5s) -> 硬上限 N_max = floor(T / F)
#
def seq_required_tps(D, F, T=DEADLINE_S):
    return D / (T - F)

def conc_required_aggregate_tps(N, D, F, T=DEADLINE_S):
    denom = T - N * F
    return np.where(denom > 0, (N * D) / denom, np.inf)

def conc_required_perstream_tps(N, D, F, T=DEADLINE_S):
    denom = T - N * F
    return np.where(denom > 0, D / denom, np.inf)

def soc_aggregate_capability(N):
    """SoC batch 後可達到的聚合 decode 吞吐: 近線性成長到 roofline 飽和。"""
    return np.minimum(N * SOC_DECODE_TPS_SINGLE, SOC_DECODE_TPS_ROOF)


# ============================================================
# 6) 主程式: 計算 + 印表 + 畫圖
# ============================================================
def main():
    c = single_path_costs(verbose=True)
    D, F = c["decode_tokens"], c["fixed_s"]
    T = DEADLINE_S

    N_max = int(np.floor(T / F))   # prefill 序列化的硬上限
    seq_tps = seq_required_tps(D, F, T)

    print("=" * 64)
    print("結果")
    print("=" * 64)
    print(f"[序列] 每個任務需求 decode TPS = {seq_tps:.1f} tok/s (與 N 無關)")
    print(f"       服務 N 人總延遲 = N x {T}s (第 N 人要等 {T*1:.1f}*N 秒)")
    print(f"[併發] Prefill 不可 batch 的硬上限 N_max = floor({T}/{F:.3f}) = {N_max} 路")
    print(f"       (N>{N_max} 時, 光 prefill+ViT 就 > {T}s, 不可行)")
    print()
    print(f"{'N':>3} | {'剩餘decode時間':>13} | {'聚合需求TPS':>11} | {'單路需求TPS':>11} | {'SoC聚合能力':>10} | 可行?")
    print("-" * 78)
    for N in range(1, N_max + 3):
        denom = T - N * F
        if denom <= 0:
            print(f"{N:>3} | {'<=0 (prefill 吃光)':>13} | {'INF':>11} | {'INF':>11} | "
                  f"{soc_aggregate_capability(N):>10.0f} | NO")
            continue
        agg = (N * D) / denom
        per = D / denom
        cap = soc_aggregate_capability(N)
        ok = "YES" if cap >= agg else "NO"
        print(f"{N:>3} | {denom*1000:>10.0f} ms | {agg:>11.0f} | {per:>11.0f} | {cap:>10.0f} | {ok}")
    print()

    # -------- 畫圖 --------
    Ns = np.arange(1, max(N_max + 2, 10) + 1)
    agg_req = conc_required_aggregate_tps(Ns, D, F, T)
    per_req = conc_required_perstream_tps(Ns, D, F, T)
    cap = soc_aggregate_capability(Ns)

    # 把 inf 換成 nan 以免畫面爆掉
    agg_plot = np.where(np.isfinite(agg_req), agg_req, np.nan)
    per_plot = np.where(np.isfinite(per_req), per_req, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左圖: 聚合 decode TPS 需求 vs SoC 能力
    ax1.plot(Ns, agg_plot, "o-", color="#d62728", lw=2.2, label="Concurrent: required AGGREGATE decode TPS")
    ax1.plot(Ns, cap, "s--", color="#2ca02c", lw=2, label=f"SoC aggregate capability (roof={SOC_DECODE_TPS_ROOF:.0f})")
    ax1.axhline(seq_tps, color="#1f77b4", lw=2, ls=":",
                label=f"Sequential required TPS = {seq_tps:.0f} (const)")
    ax1.axvline(N_max + 0.5, color="gray", lw=1.5, ls="--")
    ax1.text(N_max + 0.55, ax1.get_ylim()[1]*0.5 if False else 1,
             f"  Prefill hard limit\n  N_max = {N_max}", color="gray", va="bottom", fontsize=10)
    ax1.set_xlabel("N  (concurrent paths)", fontsize=12)
    ax1.set_ylabel("Decode TPS (tokens/s)", fontsize=12)
    ax1.set_title("Aggregate decode-TPS requirement vs SoC capability", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9, loc="upper left")

    # 右圖: 單路 decode TPS 需求 + 時間預算拆解
    ax2.plot(Ns, per_plot, "o-", color="#9467bd", lw=2.2,
             label="Concurrent: required PER-STREAM decode TPS")
    ax2.axhline(seq_tps, color="#1f77b4", lw=2, ls=":",
                label=f"Sequential required = {seq_tps:.0f}")
    ax2.axhline(SOC_DECODE_TPS_SINGLE, color="#ff7f0e", lw=2, ls="--",
                label=f"SoC single-stream decode = {SOC_DECODE_TPS_SINGLE:.0f}")
    ax2.axvline(N_max + 0.5, color="gray", lw=1.5, ls="--")
    ax2.set_xlabel("N  (concurrent paths)", fontsize=12)
    ax2.set_ylabel("Per-stream decode TPS (tokens/s)", fontsize=12)
    ax2.set_title("Per-stream decode-TPS requirement\n(rises because serialized prefill eats the 3.5s budget)",
                  fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="upper left")

    txt = (f"Inputs:  {N_IMAGES} imgs (Qwen ViT {c['vision_tokens']} vis-tok) + {N_TEXT_TOKENS} txt "
           f"= {c['initial_input']} input tok\n"
           f"Per path:  ViT {c['vit_s']*1000:.0f}ms + Prefill {c['prefill_s']*1000:.0f}ms "
           f"= F {F*1000:.0f}ms   |   Decode D = {D} tok   |   Deadline T = {T}s   "
           f"|   KV-cache {'ON' if USE_KV_CACHE else 'OFF'}")
    fig.suptitle("Cabin AI — 3-step Agentic task: TPS requirement trend (Sequential vs Concurrent)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.03, txt, ha="center", fontsize=9.5, family="monospace")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.16, wspace=0.20)

    out = "tps_requirement_trend.png"
    fig.savefig(out, dpi=130)
    print(f"已輸出趨勢圖: {out}")


if __name__ == "__main__":
    main()

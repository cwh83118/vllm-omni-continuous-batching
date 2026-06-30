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
# 5) TPS 需求公式  (★同一個 3.5s total, N 路全部要做完★)
# ============================================================
#
# 令 F = 單路 (prefill + ViT) 序列化固定成本, D = 單路 decode token 數, T = deadline(3.5s)
#
# 兩種模式「都」要在同一個 T 內把 N 路全部做完。
# Prefill / ViT 都不能 batch -> 兩種模式的序列化固定成本都是 N x F。
# 留給 decode 的時間都是 (T - N x F); 要 decode 的總量都是 N x D。
# 可行條件 (兩者共用): N x F < T  -> 硬上限 N_max = floor(T / F)
#
# 【序列 Sequential】 一條 stream 把 N x D 個 token 逐一吐完:
#       需求「單流」decode TPS = (N x D) / (T - N x F)        <-- 必須由"一條"stream 達成
#
# 【併發 Batch (N 路)】 N 條 stream 一起 decode (每個 step 同時吐 N 個 token):
#       decode 牆鐘 = D / 每流TPS  (與 N 無關!)
#       需求「每流」decode TPS  = D / (T - N x F)             <-- 每條只要這麼快
#       需求「聚合」decode TPS  = (N x D) / (T - N x F)       <-- 與序列的單流需求同值
#
# ★ TPS 差異: 聚合需求相同, 但序列要"一條"stream 扛, batch 拆給 N 條 ->
#   每流門檻低 N 倍。這就是 batch 的優勢 (= 倍率 N)。
#
def seq_required_tps(N, D, F, T=DEADLINE_S):
    """序列: 單一條 stream 要吐完 N x D 個 token 的需求 TPS。"""
    denom = T - N * F
    return np.where(denom > 0, (N * D) / denom, np.inf)

def batch_required_perstream_tps(N, D, F, T=DEADLINE_S):
    """併發 batch: 每一條 stream 的需求 TPS。"""
    denom = T - N * F
    return np.where(denom > 0, D / denom, np.inf)

def batch_required_aggregate_tps(N, D, F, T=DEADLINE_S):
    """併發 batch: 聚合需求 TPS (= 序列的單流需求同值)。"""
    denom = T - N * F
    return np.where(denom > 0, (N * D) / denom, np.inf)

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

    N_max = int(np.floor(T / F))   # prefill 序列化的硬上限 (兩模式共用)

    print("=" * 72)
    print("結果  (同一個 3.5s total, N 路全部要做完)")
    print("=" * 72)
    print(f"硬上限 N_max = floor({T}/{F:.3f}) = {N_max} 路  "
          f"(N>{N_max} 時光 prefill+ViT 就 > {T}s, 兩種模式都不可行)")
    print()
    print(f"{'N':>3} | {'decode預算':>9} | {'序列單流需求':>11} | {'batch每流需求':>12} | "
          f"{'聚合(=序列)':>10} | {'差異倍率':>7}")
    print("-" * 72)
    for N in range(1, N_max + 3):
        denom = T - N * F
        if denom <= 0:
            print(f"{N:>3} | {'<=0':>9} | {'INF':>11} | {'INF':>12} | {'INF':>10} | {N:>6}x")
            continue
        seq = (N * D) / denom          # 序列: 單流需求
        per = D / denom                # batch: 每流需求
        agg = (N * D) / denom          # batch: 聚合需求 (= 序列單流需求)
        print(f"{N:>3} | {denom*1000:>6.0f} ms | {seq:>9.0f} t/s | {per:>10.0f} t/s | "
              f"{agg:>8.0f} t/s | {N:>6}x")
    print()
    print("解讀: 聚合需求兩者相同; 序列要『一條 stream』扛下整個聚合 (極難),")
    print("      batch 拆成 N 條 -> 每流門檻低 N 倍, 這就是 batch 的 TPS 優勢。")
    print()

    # -------- 畫圖 --------
    Ns = np.arange(1, max(N_max + 2, 10) + 1)
    seq_req   = seq_required_tps(Ns, D, F, T)                 # 序列: 單流需求 (= 聚合)
    batch_per = batch_required_perstream_tps(Ns, D, F, T)     # batch: 每流需求
    cap_agg   = soc_aggregate_capability(Ns)

    seq_plot   = np.where(np.isfinite(seq_req), seq_req, np.nan)
    batch_plot = np.where(np.isfinite(batch_per), batch_per, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左圖: ★TPS 差異★ — 序列(單流) vs batch(每流) 的需求 decode TPS
    ax1.plot(Ns, seq_plot, "o-", color="#d62728", lw=2.4,
             label="SEQUENTIAL: required single-stream TPS = N*D/(T-N*F)")
    ax1.plot(Ns, batch_plot, "o-", color="#9467bd", lw=2.4,
             label="BATCH: required per-stream TPS = D/(T-N*F)")
    ax1.axhline(SOC_DECODE_TPS_SINGLE, color="#ff7f0e", lw=2, ls="--",
                label=f"SoC single-stream decode = {SOC_DECODE_TPS_SINGLE:.0f}")
    ax1.axvline(N_max + 0.5, color="gray", lw=1.5, ls="--")
    ax1.text(N_max + 0.55, SOC_DECODE_TPS_SINGLE,
             f"  Prefill hard limit\n  N_max = {N_max}", color="gray", va="bottom", fontsize=10)
    # 標出幾個點的 N 倍差距
    for n in [2, 4, 6, 8]:
        if T - n * F > 0:
            ax1.annotate(f"{n}x", xy=(n, (n*D)/(T-n*F)), xytext=(n-0.15, (n*D)/(T-n*F)*1.15),
                         color="#d62728", fontsize=9, fontweight="bold")
    ax1.set_xlabel("N  (paths to finish within the SAME 3.5s)", fontsize=12)
    ax1.set_ylabel("Required decode TPS per stream (tokens/s)", fontsize=12)
    ax1.set_title("TPS DIFFERENCE: Sequential vs Batch\n(per-stream requirement; gap = N x)",
                  fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9, loc="upper left")

    # 右圖: 聚合需求(兩者相同) vs SoC 聚合能力 -> 看硬體 roofline 可不可行
    ax2.plot(Ns, seq_plot, "o-", color="#d62728", lw=2.4,
             label="Required AGGREGATE decode TPS (same for both)")
    ax2.plot(Ns, cap_agg, "s--", color="#2ca02c", lw=2,
             label=f"SoC aggregate capability (single {SOC_DECODE_TPS_SINGLE:.0f}, roof {SOC_DECODE_TPS_ROOF:.0f})")
    ax2.axvline(N_max + 0.5, color="gray", lw=1.5, ls="--")
    ax2.set_xlabel("N  (paths to finish within the SAME 3.5s)", fontsize=12)
    ax2.set_ylabel("Aggregate decode TPS (tokens/s)", fontsize=12)
    ax2.set_title("Aggregate requirement vs SoC roofline\n(feasibility: green must stay above red)",
                  fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9, loc="upper left")

    txt = (f"Inputs:  {N_IMAGES} imgs (Qwen ViT {c['vision_tokens']} vis-tok) + {N_TEXT_TOKENS} txt "
           f"= {c['initial_input']} input tok\n"
           f"Per path:  ViT {c['vit_s']*1000:.0f}ms + Prefill {c['prefill_s']*1000:.0f}ms "
           f"= F {F*1000:.0f}ms   |   Decode D = {D} tok   |   Deadline T = {T}s (TOTAL for all N)   "
           f"|   KV-cache {'ON' if USE_KV_CACHE else 'OFF'}")
    fig.suptitle("Cabin AI — 3-step Agentic task: Sequential vs Batch decode-TPS requirement",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.03, txt, ha="center", fontsize=9.5, family="monospace")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.16, wspace=0.20)

    out = "tps_requirement_trend.png"
    fig.savefig(out, dpi=130)
    print(f"已輸出趨勢圖: {out}")


if __name__ == "__main__":
    main()

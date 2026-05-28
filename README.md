# vLLM-Omni × Continuous Batching — 三種 request 排程方式的體驗對比（RTX 5090 實機）

在一台 **RTX 5090（Blackwell sm_120, 32 GB）** 上用 **vLLM-Omni 0.20.0** serve **Qwen3-Omni-30B-A3B 的 Thinker**，
量出一個推論服務面對「陸續進來的 request 流」時，三種排程方式的體驗差異 —— 對應到車用座艙助手「主動偵測」與
「交互對話」兩個應用共用同一顆模型的情境。

> 📄 **完整圖文報告（背景 → 三種排程方式 → 實驗方法 → 結果與三 panel 時間軸圖 → 體驗解釋 → 為什麼是這些數字 → cross-check）：[`REPORT.md`](REPORT.md)**
>
> 🚗 **進階版（多模態 + 多步代理 + batch size 1/2/4/8/16 sweep + in-flight 並發帶視覺化 + CX1 啟發）：[`REPORT_DUAL.md`](REPORT_DUAL.md)**
> 在原版（純文字、固定 B=8）的基礎上把輸入換成 **真實 audio + image + 車輛 JSON**、interactive 升級成 **多步 tool-loop agent**、
> 跑了 **55 個 sweep run**（5 情境 × 3 mode × B {1,2,4,8,16}），並在每張 timeline 下方加了 **in-flight count over time** 條，
> 讓「哪些 request 同時在 GPU 上被一起 decode」一眼看得出來。

---

## 三種排程方式

| | 同時在跑的上限 | 某條算完、空出 slot 後… | 對應誰 |
|---|--:|---|---|
| **(1) no batching** | 1 | 等這 1 條算完才送下一條 | 最樸素的 `model.generate(單條)`；`max_num_seqs=1` 的 server |
| **(2) static / 固定 batch** | B | 這一波 ≤B 條**全部排空**後才湊下一波；波進行中新 request 一律等 | 典型 NPU runtime / 靜態 graph（TensorRT static engine 等）—— batch 維度啟動時定死、跑中不能加 |
| **(3) continuous batching** | B | 任一條算完的**瞬間**就從 queue 補一條進來；新 request 下一個 step 就併入正在跑的 batch、誰先算完誰先釋出 | **vLLM / vLLM-Omni 的做法** |

vLLM 的 V1 engine 永遠是 (3)、沒有「靜態 batch」開關 —— 所以三種模式都用**同一個 server**（`max_num_seqs=32`，
不是瓶頸）+ 三種 **client 端的進場/補位規則** 來跑，唯一變數就是排程策略。本次 B=8（mode 1 等同 B=1）。

## 結論（數據）

同一顆模型、同一個 server、同一份負載（同一個 `--seed` → 三次到達時間、每個 request 的取樣 seed 完全一樣）：

**車艙情境**（主動偵測每 2.5s 看一次場景 + 交互 Poisson ~1.6/s 隨機進來，24s，共 30 交互 + 10 主動偵測）

| 指標 | (1) no batching `B=1` | (2) static `B=8` | (3) continuous `B=8` |
|---|--:|--:|--:|
| 交互 TTFT p50 | **1 020 ms** | **199 ms** | **18 ms** |
| 交互 TTFT p95 / 最差 | 2 096 / 2 131 ms | 561 / 602 ms | 25 / 27 ms |
| 交互 端到端 p50 | 1 252 ms | 640 ms | 438 ms |
| 主動偵測 端到端 最差 | 1 985 ms | 1 132 ms | 888 ms |
| 相對 (3) 的 TTFT p50 倍數 | **57×** | **11×** | 1× |

**飽和爆量**（24 個 request 同一瞬間進來，各 ≤160 tokens）

| 指標 | (1) no batching | (2) static `B=8` | (3) continuous `B=8` |
|---|--:|--:|--:|
| 交互 TTFT p50 / 最差 | 4 359 / 8 857 ms | 904 / 1 626 ms | 641 / 1 412 ms |
| 24 個全部算完 | **9.16 s** | **2.46 s** | **1.90 s** |
| 輸出吞吐（忙碌區間） | **238 tok/s** | **863 tok/s** | **1 083 tok/s** |

→ `continuous` 比 `static` 再快約 11×（TTFT），比 `no batching` 快約 57×；`static` 介於兩者中間（拿到了「一波多條一起算」
的好處、比沒 batching 好很多，但新 request 仍卡在「上一批還沒算完」的批界停頓）。decode 是記憶體頻寬瓶頸，一次權重讀
同時推進整個 batch → 飽和時吞吐 238 → 863 → 1 083 tok/s。

時間軸圖：[`results/timeline_3way.png`](results/timeline_3way.png)（車艙情境）、[`results/timeline_3way_burst.png`](results/timeline_3way_burst.png)（飽和爆量）。
左／中／右 ＝ (1) no batching ／ (2) static ／ (3) continuous；淺色 ＝ 排隊+prefill 還沒看到字（這段長度 = TTFT）、
深色 ＝ 正在吐 token、黑色 `|` ＝ 第一個 token 送達、(2) 的淡色虛線 ＝ 一個「波」的開始。

---

## 為什麼是「Thinker 經由 `vllm serve`」而不是完整 omni pipeline

Qwen3-Omni-30B-A3B 的完整三段 pipeline（Thinker+Talker+Code2Wav）官方 deploy config 是在 **2× H100-80G** 上驗證的；
三段全擠到單卡 32 GB 會 OOM（Thinker 一個 stage 就吃 ~20 GB 權重 + KV cache）。而 **continuous batching 的行為完全活在
AR scheduler 裡**，跟 audio 那兩段無關。剛好 plain `vllm serve <Qwen3-Omni 模型>` 會把 arch
`Qwen3OmniMoeForConditionalGeneration` 映射成只出文字的 **Thinker**（`Qwen3OmniMoeThinkerForConditionalGeneration`），
只載 ~20 GB 權重、KV cache 拿滿 ~11 GB（≈ 120k tokens）。這個 Thinker AR engine 就是 vLLM-Omni stage-0 內部用的同一套
vLLM engine，所以這裡量到的行為 ＝ 你在 vLLM-Omni 的 Thinker stage 會看到的行為。完整 omni（含語音輸出）建議 ≥2 張 80 GB 卡。

---

## 環境

```bash
conda create -n vllm_omni python=3.12 -y
pip install "vllm==0.20.0"        # cu130 prebuilt wheel（含 sm_120）；不要加 uv 的 --torch-backend=auto
                                  # （它只認到 cu128 會給 torch==2.11.0+cu128 → libcudart.so.13: cannot open shared object file）
pip install "vllm-omni==0.20.0" matplotlib
```
模型：`cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`（HF；compressed-tensors int4, group_size 32 —— ~20 GB Thinker
權重，單卡 32 GB 放得下；bf16 版用 `Qwen/Qwen3-Omni-30B-A3B-Instruct`，需更多 VRAM）。可用 `MODEL=<path or HF id>` 覆寫。

## 怎麼跑

```bash
cd vllm-omni-continuous-batching   # clone 後的目錄；先 `conda activate vllm_omni`

# 1) 起一個夠大的 server（三種模式都在 client 端 emulate，server 不是瓶頸）
MAX_NUM_SEQS=32 bash run_server.sh           # 等 logs/server_seqs32.log 出現 "Application startup complete"（約 30–40s）
                                             # curl http://localhost:8901/v1/models 確認

# 2) 車艙情境（同一個 --seed → 三次到達時間/取樣完全一樣）
python cabin_demo.py --mode none       --batch-size 1 --max-num-seqs 32 --out results/run_none.json
python cabin_demo.py --mode static     --batch-size 8 --max-num-seqs 32 --out results/run_static.json
python cabin_demo.py --mode continuous --batch-size 8 --max-num-seqs 32 --out results/run_continuous.json

# 3) 飽和爆量（24 條一瞬間進來）
for m in none static continuous; do B=8; [ "$m" = none ] && B=1; \
  python cabin_demo.py --mode $m --batch-size $B --max-num-seqs 32 --burst 24 --interactive-max-tokens 160 --out results/burst_$m.json; done

# 4) 畫圖（三 panel）+ 列三欄表
python plot_timeline.py --panels results/run_none.json results/run_static.json results/run_continuous.json --out results/timeline_3way.png
python plot_timeline.py --panels results/burst_none.json results/burst_static.json results/burst_continuous.json --out results/timeline_3way_burst.png

# 5)（選用）用 vLLM 內建 benchmark 跑 concurrency sweep：bash bench_sweep.sh
```

`cabin_demo.py` 主要參數：`--mode {none,static,continuous}`、`--batch-size B`、`--duration`（到達時間窗）、
`--proactive-interval`、`--proactive-max-tokens`、`--interactive-rate`（Poisson req/s）、`--interactive-max-tokens`、
`--n-interactive`（上限）、`--burst N`（飽和模式：N 條一次到達、不跑主動偵測）、`--seed`。

## 檔案

| 檔案 | 說明 |
|---|---|
| `REPORT.md` | 完整圖文報告（三種排程方式的對比） |
| `run_server.sh` | 起 vLLM serve（Qwen3-Omni-30B-A3B Thinker，text-only，`--skip-mm-profiling`）；`MAX_NUM_SEQS` 控 server 的 batch 上限 |
| `cabin_demo.py` | 兩條 request 流（主動偵測 + 交互）打同一個 engine，用 client 端 admission controller emulate `--mode {none,static,continuous}`，記每個 request 的 t_submit/t_admitted/t_first_token/t_finish/wave_id，印時間軸 log 與 summary，輸出 JSON；`--burst N` 飽和模式 |
| `plot_timeline.py` | `--panels a.json b.json c.json` → 並排時間軸圖（N panel）+ 三欄對比表；static panel 畫波界虛線 |
| `bench_sweep.sh` | (選用) 用 `vllm bench serve` 跑 concurrency sweep，輸出 CSV |
| `results/run_{none,static,continuous}.json` | 車艙情境的逐 request 資料 + summary |
| `results/burst_{none,static,continuous}.json` | 飽和爆量的資料 |
| `results/timeline_3way.png` / `timeline_3way_burst.png` | 三 panel 時間軸對比圖 |
| `results/run_off.json` / `run_on.json` / `timeline_cabin.png` / `timeline_burst.png` | 附錄：第一階段用**真實** `max_num_seqs=1` vs `16` server 跑的兩類資料，當 emulation 的 cross-check |
| `logs/server_seqs*.log` | 各次 server 的完整 log（含 KV cache 大小、並發上限等） |

## 注意

- server 跑起來佔 ~30 GB / 32 GB GPU；用完 `pkill -f "vllm serve"`。這台機器上其他程序（如 ollama）也可能搶 GPU —
  server 起不來 OOM 時先 `nvidia-smi --query-compute-apps` / `ollama ps` 看誰在佔。
- `run_server.sh` 假設你已 `conda activate vllm_omni`（它從 PATH 找 `vllm`）。
- mode (2)「static」是 NPU 為主 / 靜態圖的**合理模型**；本 emulation 對它偏寬鬆（vLLM 在底下會把跑中的 batch 隨 seq
  結束縮小、且我們沒有「等湊滿 batch」的批集延遲）→ 量到的差距是下界，真實 NPU 會更慘。mode (1)/(3) 與真實
  `max_num_seqs=1`/`B` server 的數字 cross-check 對得上（見 `REPORT.md` 附錄）。

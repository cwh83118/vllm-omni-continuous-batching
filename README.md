# vLLM-Omni × Continuous Batching — 5090 實機 Demo

評估 **vLLM-Omni** 是否支援 continuous batching（iteration-level / in-flight batching），並在這台
**RTX 5090（Blackwell sm_120, 32 GB）** 上用 **Qwen3-Omni-30B-A3B** 實際跑出「有 vs 沒有」的差異。

> 📄 **完整圖文報告（背景 → 實驗方法 → 結果與時間軸圖 → 體驗解釋）：[`REPORT.md`](REPORT.md)**

---

## 結論（先講）

1. **vLLM-Omni 支援 continuous batching — 是的。** vLLM-Omni 把 omni 模型拆成多個 stage（Thinker / Talker /
   Code2Wav），每個 AR stage 各跑一個標準 vLLM engine。continuous batching 就住在那個 AR scheduler 裡：
   - waiting / running queue，每個 decode step 重排 running batch，某條 seq 打到 EoS 就立刻從 running queue
     移除、把結果釋出，新進來的 request 下一個 step 就能併入正在跑的 batch；
   - 由 `max_num_seqs` / `max_num_batched_tokens` 控批量（vLLM-Omni 的 stage deploy schema 裡 AR stage 的
     `max_num_seqs` 預設 64）。
   官方專案：<https://github.com/vllm-project/vllm-omni> ｜ 文件：<https://docs.vllm.ai/projects/vllm-omni/>
2. **這台 5090 上跑得起來。** conda env `vllm_omni`（Python 3.12）裝了 `vllm==0.20.0` + `vllm-omni==0.20.0`
   （prebuilt cu130 wheel，含 sm_120 — 不需自己編 kernel）。Qwen3-Omni-30B-A3B（AWQ-4bit / compressed-tensors）
   的 **Thinker** 在單卡 32 GB 上 serve 起來只用 ~20 GB 權重 + ~11 GB KV cache（~120k tokens，約 14.6×
   並發）。
3. **數據（同一顆模型、同一個 engine、唯一差別是 `max_num_seqs`）：**

   | 情境 | 指標 | 沒有 continuous batching `max_num_seqs=1` | 有 continuous batching `max_num_seqs=16` | 倍數 |
   |---|---|---:|---:|---:|
   | **車艙情境**（主動偵測每 2.5s + 交互 Poisson ~1.6/s，共 30+10 req） | 交互 TTFT p50 | **946 ms** | **19 ms** | **~50×** |
   | | 交互 TTFT p95 | 2 239 ms | 27 ms | **~83×** |
   | | 交互 TTFT max | 2 357 ms | 67 ms | ~35× |
   | | 交互 端到端 p50 | 1 114 ms | 430 ms | ~2.6× |
   | | 主動偵測 端到端 max | 2 192 ms | 806 ms | ~2.7× |
   | **飽和爆量**（同一瞬間丟 24 個 request） | 輸出吞吐（busy span） | **239 tok/s** | **1 456 tok/s** | **~6.1×** |
   | | 全部 24 個算完耗時 | 9.2 s | 1.5 s | ~6.2× |
   | | 交互 TTFT p50 | 4 499 ms | 52 ms | ~87× |
   | | 交互 TTFT max | 8 615 ms | 583 ms | ~15× |

   圖：`results/timeline_cabin.png`、`results/timeline_burst.png`（左＝沒有，右＝有；淺色＝排隊+prefill 還沒看到字，
   深色＝正在吐 token，黑色 `|` ＝第一個 token 送達）。

### 兩層意義（對應你要說明的）

- **第一層：體驗。** 一個 vLLM-Omni engine 同時服務「主動偵測」（每 2.5s 看一次艙內/艙外場景 → 輸出 function call
  與理由）和「交互」（使用者隨時語音問問題）。**沒有** continuous batching 時這兩條互等：交互的 request 卡在
  queue 裡等主動偵測那一輪算完，使用者按下說話到聽到回應要等 ~0.9 s（p50），最壞 ~2.4 s；爆量時甚至要等 ~4.5–8.6 s。
  **有** continuous batching 時，交互的 request 在主動偵測 decode 到一半時直接併入同一個 batch，~19 ms 就吐第一個字，
  主動偵測完全不受影響、繼續跑；某個 request 打到 EoS 就馬上把結果交還，下一個輸入隨時可以再進來。
- **第二層：頻寬 / 數據。** decode 階段是記憶體頻寬瓶頸 —— 每個 step 都要把整顆模型權重從 HBM 讀一遍。沒有
  batching 時這一次權重讀只服務 1 條 seq；有 batching 時同一次權重讀同時服務 16 條 → 同樣的權重 bytes 做了 ~6×
  的有效工作（吞吐 239 → 1456 tok/s）。倍數沒到 16× 是因為 batch 16 時模型已部分變成 compute-bound，而且這顆
  30B-A3B（每 token 只 active ~3B）在 batch 1 時用 AWQ + CUDA graph 已經很快（~240 tok/s），ratio 看起來「只有」
  ~6×；換更大或更密的模型會更接近 batch size。每個 request 進 batch 運算的時間 / 何時算完 / 何時釋出給 user，
  都記在 `results/*.json` 的 `requests[]`（`t_submit` / `t_first_token` / `t_finish`）。

---

## 為什麼是「Thinker 經由 `vllm serve`」而不是完整 omni pipeline

Qwen3-Omni-30B-A3B 的完整三段 pipeline（Thinker+Talker+Code2Wav）官方 deploy config 標明是在 **2× H100-80G**
上驗證的；三段全擠到單卡 32 GB 會 OOM（Thinker 一個 stage 就吃 ~20 GB 權重 + KV cache，剩不到 ~10 GB 給
Talker+Code2Wav）。而 **continuous batching 的行為完全活在 AR scheduler 裡**，跟 audio 那兩段無關。

剛好 plain `vllm serve <Qwen3-Omni 模型>` 會把 arch `Qwen3OmniMoeForConditionalGeneration` 映射到
`Qwen3OmniMoeThinkerForConditionalGeneration`（只出文字、不生音訊），所以只載 ~20 GB Thinker 權重、KV cache 拿滿
~11 GB。這個 Thinker AR engine 就是 vLLM-Omni stage-0 內部用的同一套 vLLM engine，所以這裡量到的 continuous
batching 行為＝你在 vLLM-Omni 的 Thinker stage 會看到的行為。

> 註：`vllm-omni serve <model> --omni`（完整 omni pipeline）也已驗證裝得起來、會載入（在 32 GB 上 stage 0 載到
> 19.5 GB、再起 stage 1/2 才 OOM）。另外 torch 2.11 / sm_120 上 omni 的 multimodal *profiling* 路徑有個 Triton
> codegen 小毛病，`--skip-mm-profiling` 可繞過。要跑完整 omni（含語音輸出）建議 ≥2 張 80GB 等級的卡。

---

## 環境

- conda env：`vllm_omni`（Python 3.12）。建立方式：
  ```bash
  conda create -n vllm_omni python=3.12 -y
  uv pip install --python ~/miniconda3/envs/vllm_omni/bin/python "vllm==0.20.0"   # 帶 cu130 prebuilt wheel（含 sm_120）
  uv pip install --python ~/miniconda3/envs/vllm_omni/bin/python "vllm-omni==0.20.0" matplotlib
  ```
  （注意：vLLM 0.20.0 的 PyPI wheel 是 CUDA 13 build，需要 `torch==2.11.0+cu130`；別用 `uv ... --torch-backend=auto`，
  它只認到 cu128 會給錯版本的 torch、造成 `libcudart.so.13: cannot open shared object file`。直接 `pip install vllm`
  讓它解預設 PyPI 的 `torch==2.11.0`（已是 cu13 build）即可。）
- 模型：`cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`（HF；compressed-tensors, pack-quantized int4,
  group_size 32 —— ~20 GB Thinker 權重，單卡 32 GB 放得下；bf16 版用 `Qwen/Qwen3-Omni-30B-A3B-Instruct`）。

## 怎麼跑

```bash
cd vllm-omni-continuous-batching   # clone 後的目錄；先 `conda activate vllm_omni`

# --- 1. 起 server（唯一變數：MAX_NUM_SEQS）---
MAX_NUM_SEQS=1  bash run_server.sh    # "沒有 continuous batching"：一次只跑一條 seq、FCFS
# 或
MAX_NUM_SEQS=16 bash run_server.sh    # "有 continuous batching"
# 等到 logs/server_seqs<N>.log 出現 "Application startup complete"（約 30–40s），
# 可 curl http://localhost:8901/v1/models 確認。

# --- 2. 跑 demo（要對著對應的 server 跑兩次）---
# 車艙情境（體驗 + 時間軸）：
python cabin_demo.py --config off --max-num-seqs 1  --out results/run_off.json   # 對 MAX_NUM_SEQS=1 的 server
python cabin_demo.py --config on  --max-num-seqs 16 --out results/run_on.json    # 對 MAX_NUM_SEQS=16 的 server
# 飽和爆量（頻寬 / 吞吐）：
python cabin_demo.py --config off_burst --max-num-seqs 1  --burst 24 --interactive-max-tokens 160 --out results/burst_off.json
python cabin_demo.py --config on_burst  --max-num-seqs 16 --burst 24 --interactive-max-tokens 160 --out results/burst_on.json

# --- 3. 畫圖 + 列對比表 ---
python plot_timeline.py --off results/run_off.json   --on results/run_on.json   --out results/timeline_cabin.png
python plot_timeline.py --off results/burst_off.json --on results/burst_on.json --out results/timeline_burst.png \
    --title "Saturated burst — 24 requests submitted at once to one vLLM-Omni engine"

# --- (選用) 用 vLLM 內建 benchmark 跑 concurrency sweep ---
# 起好 server 後：bash bench_sweep.sh   （再用 MAX_NUM_SEQS=1 起一次、再跑一次當 baseline）
```

`cabin_demo.py` 主要參數：`--duration`（送 request 的時間窗）、`--proactive-interval`、`--proactive-max-tokens`、
`--interactive-rate`（Poisson req/s）、`--interactive-max-tokens`、`--n-interactive`（上限）、`--burst N`
（飽和模式：一瞬間丟 N 個 request、不跑主動偵測）、`--seed`。

## 檔案

| 檔案 | 說明 |
|---|---|
| `run_server.sh` | 起 vLLM serve（Qwen3-Omni-30B-A3B Thinker，text-only，`--skip-mm-profiling`）；`MAX_NUM_SEQS` 控 batching 開關 |
| `cabin_demo.py` | 兩條 request 流（主動偵測 + 交互）打同一個 engine，記每個 request 的 t_submit / t_first_token / t_finish，印時間軸 log 與對比表，輸出 JSON。也支援 `--burst N` 飽和模式 |
| `plot_timeline.py` | 讀兩份 JSON → 並排時間軸圖（左＝沒有 / 右＝有），並印對比表 |
| `bench_sweep.sh` | (選用) 用 `vllm bench serve` 跑 concurrency sweep，輸出 CSV |
| `results/run_off.json`, `run_on.json` | 車艙情境的逐 request 資料 + summary（max_num_seqs 1 vs 16） |
| `results/burst_off.json`, `burst_on.json` | 飽和爆量的資料 |
| `results/timeline_cabin.png`, `timeline_burst.png` | 時間軸對比圖 |
| `logs/server_seqs<N>.log` | 各次 server 的完整 log（含 KV cache 大小、並發上限等） |

## 注意

- server 跑起來會佔 ~30 GB / 32 GB GPU；用完記得 `pkill -f "vllm serve"`。這台機器上其他程序（如 ollama 的
  `qwen3-vl:8b`、別的 Claude session）也可能搶 GPU — 如果 server 起不來 OOM，先 `nvidia-smi` 看誰在佔。
- 兩個 config 要分別重啟 server（`max_num_seqs` 是啟動時參數）；`cabin_demo.py` 的 `--seed` 固定 → 兩次的
  request 到達時間一樣，比較才公平。
- `run_off` 用 `max_num_seqs=1` 來代表「沒有 continuous batching」：vLLM 仍然是 iteration-level 排程，只是
  running batch 最多 1 條 → request 排隊、一個一個算到完，等同於最樸素的 `model.generate()` 迴圈。這是業界
  示範 continuous batching 價值的標準做法。

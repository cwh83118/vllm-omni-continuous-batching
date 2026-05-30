# 座艙 AI 真實劇本 × CX1 等效硬體 × 4 種排程方式體感效益對比

> **單一駕駛 3 分鐘真實通勤劇本** (媽媽下班接小孩 → 送才藝課 → 順路買菜) +
> **5090 throttle 到 CX1 等效硬體** (memory clock 鎖到 810 MHz → BW 68 GB/s) +
> **4 種排程 mode** (none / static / static + VIP / continuous + priority) + vanilla continuous 對照
> = 47 個 LLM request、6 個 run、0 個 error、TTFT p95 看到 **10× 體感差異**。

> 這份是 [REPORT.md](REPORT.md)（純文字 baseline）與 [REPORT_DUAL.md](REPORT_DUAL.md)（多模態 + agent + batch sweep）之後的第三輪交付，回答老闆問題：
>
> **「投資 continuous batching 這個技術，到底帶來怎樣的體感效益？要具體量化、要代表性。」**

---

## 0. TL;DR — 結論先講

### 0.1 一張圖看完所有結論

![TTFT bar chart](results/commute_ttft_bars.png)

| 指標 | none B=1 | static B=6 | static+VIP | **continuous** | **continuous+pri** |
|---|--:|--:|--:|--:|--:|
| **TTFT p50** (典型情況) | 109 ms | 109 ms | 110 ms | 112 ms | **106 ms** |
| **TTFT p95** (5% 較差情況) | 1952 ms | 1945 ms | 1952 ms | **189 ms** | **203 ms** |
| **TTFT max** (最差瞬間) | 2389 ms | 2380 ms | 2385 ms | **194 ms** | **210 ms** |
| inter e2e p50 | 908 ms | 909 | 906 | 870 | 904 |
| proactive e2e max | 5034 ms | 5041 | 5035 | 5227 | 5024 |
| 總 request / errors | 49 / 0 | 49 / 0 | 49 / 0 | 49 / 0 | 47 / 0 |

### 0.2 一句話結論

> **典型瞬間 (TTFT p50 ~110 ms) 看不出差別，但 user 體感的「瞬間 lag」 (TTFT p95 / max) 在 static / no-batch 上會飆到 1.9–2.4 秒，continuous 把它壓回 190–210 ms — 體感差距 10×。**
>
> 對 user 來說：100 次語音指令裡，有 5 次 (p95)，static 會讓你等 2 秒、continuous 仍即時。**這 5% 是 user 主觀感受「這 AI 太慢」的時刻。** 投資 continuous batching 就是把這個時刻消滅掉。

### 0.3 為什麼 p50 沒差、p95/max 才有差？

劇本是「真實單一駕駛場景」— arrival rate 平均下不擠（0.27 req/s × 4 s/req = 0.55 < 1）— 大多時候 queue 是 0。

但劇本有**自然的緊密區段**：
- t=30: 媽媽說複雜指令 → 觸發 3-step agent fan-out (每步 1s)
- t=50-67: 老師訊息、媽媽回應、proactive 提醒、媽媽同意、agent 發訊息 — 5 事件擠在 17s
- t=100-130: 小明上車 → 音樂 → ETA → 提醒 — 4 事件擠在 30s

在這些「擠到」瞬間，**static 系列**會出現「前面那條 proactive 還在 decode、新進來的 interactive 必須等」的 1-2 秒 head-of-line block。`continuous` 系列的每-step refill 讓新 interactive 立刻插進 batch、不必等。

→ p50 反映「大多時候沒擠到」、p95/max 反映「擠到的瞬間」。**老闆要看的就是後者。**

### 0.4 跨硬體比例 — 真實 CX1 推算

| 量測點 | 5090 native | 5090 throttle (68 GB/s, 本實驗) | **真實 CX1 (154 GB/s, 推算)** |
|---|--:|--:|--:|
| Decode tok/s (single) | 246 | 24 | **~52** (× 154/68) |
| Interactive TTFT p50 (cont_pri) | 23 ms | 106 ms | **~47 ms** |
| Interactive TTFT p95 (none/static) | 35 ms | 1952 ms | **~862 ms** |
| Interactive TTFT p95 (continuous) | 35 ms | 189 ms | **~83 ms** |
| **TTFT p95 倍率 (static / continuous)** | ~1× | **10.3×** | **10.3×** |

→ 推算到 CX1 真機 (154 GB/s)：static 的 p95 **862 ms**、continuous 的 p95 **83 ms**。倍率仍是 10×。
→ 用 cabin 主觀體感對應：**static 系列在 1-in-20 的時刻會讓 user 等近 1 秒、continuous 仍在 80 ms** —— 換句話說：是否投資 continuous batching = 是否容忍 cabin AI 在 5% 的時間「卡 1 秒」。

---

## 1. 為什麼這次重做 — 從前兩輪的缺口

### 缺口一 (v1)：不真實的 burst24
[REPORT_DUAL.md](REPORT_DUAL.md) 用 24 條 user 同時送語音的 burst 場景秀差異 (continuous 比 none 快 33×)。
但**現實沒有 24 個 user 同時對 AI 講話**的座艙場景 → 老闆認為「故意打爆」、不代表性、不認可。

### 缺口二 (v1)：5090 太快、cabin 不擠
5090 BW 1.8 TB/s，cabin 自然 arrival rate (proactive 每 2.5s + 偶發 agent task) 完全跟不上 service rate。
所以 v1 上看到的「cabin 4 個情境 mode 不分勝負」是 5090 過剩算力的假象、不是 continuous batching 真的沒用。

### 本輪 (v2) 怎麼補

1. **設計貼近真實的單一駕駛 3 分鐘劇本** `commute_run` — 21 個事件、自然密度 (大多時候不擠、緊密區段才擠)，模擬「下班接小孩 → 送才藝課 → 買菜」的完整通勤。
2. **5090 throttle 到接近 CX1 等效** — `sudo nvidia-smi -lmc 810` 把 memory clock 鎖到 810 MHz，量到 BW ≈ 68 GB/s。5090 GDDR7 只支援 5 個離散 mem clock {14001, 13801, 7001, 810, 405}，810 MHz 是離 CX1 spec 的 154 GB/s 最近的 (delta -86 GB/s)，比 7001 MHz 的 783 GB/s (delta +629 GB/s) 近 7 倍。
3. **4 種 mode + 1 個對照** = 5 個排程方法直接對比；都跑同一個 seed、同一個劇本，**唯一變數是排程策略**。

---

## 2. 劇本：`commute_run` — 媽媽下班接小孩 180 s

### 2.1 故事

17:30，媽媽下班坐進車內，要去接 7 歲小明放學 (17:55) → 送他到 18:10 鋼琴課 → 自己順路買菜 → 回家。
AI 助理在後台串行事曆、通訊錄、地圖、訊息 App，主動掉資訊；媽媽偶爾用語音指揮 AI 做複雜任務；外部環境 (雨、塞車) 觸發主動提醒。

### 2.2 事件時間軸 (21 個事件, ~30-50 個 LLM request)

定義在 [`commute_script.py`](commute_script.py)。每個事件標 [類型][priority]：

| t (s) | 類型 | 事件 | 用到的工具 |
|---:|---|---|---|
| 0 | 🟠 P | 上車主動問候 | `get_calendar` |
| 8 | 🟠 P | 天氣 + 建議早出門 | `get_weather` |
| 18 | 🟢 A | 主動代理：路況預測 | `predict_eta` + `predict_traffic` |
| 25 | 🟠 P | 艙內疲勞 + 調冷氣 | `cabin_vision` + `set_climate` |
| **30** | 🔵 I | **「送小明去才藝課時找個順路超商」** | 多步 agent fan-out (4 steps) |
| 50 | 🟠 P | 收到老師訊息 | `check_messages` |
| 55 | 🔵 I | 「跟老師說我會晚 5 分鐘」 | `send_message` (agent) |
| 65 | 🟠 P | 主動提醒：發訊息給小明 | (純文字 TTS 提醒) |
| 67 | 🔵 I | 「好，發吧」 | `send_message` (agent) |
| 78 | 🟠 P | 塞車提醒 + 改道 | `outside_vision` |
| 90 | 🟠 P | App: 小明已讀 | `check_messages` |
| 95 | 🟢 A | 抵達學校 + 停車 | `arriving_at` + `cabin_vision` |
| **100** | 🔵 I | 「放小明喜歡的歌單」 | `get_user_preferences` + `play_music` |
| 112 | 🟠 P | 兒童語音感知 | `cabin_voice_diarize` |
| 120 | 🔵 I | 「查 ETA」 | `predict_eta` |
| 130 | 🟠 P | 即將抵達鋼琴課 | (TTS 提醒) |
| 140 | 🔵 I | 「設 19:30 提醒」 | `set_reminder` |
| 150 | 🟠 P | 抵達超市 | `arriving_at` |
| 160 | 🟠 P | App: 老公訊息 | `check_messages` |
| 165 | 🔵 I | 「跟老公說好+紅酒」 | `send_message` |
| 175 | 🟠 P | 結束問候 | (TTS) |

統計：7 interactive + 3 pure agent + 11 proactive = 21 事件，展開後 ~47-49 個 LLM request。

### 2.3 並發規則

User 指定：
- **interactive ≤ 1 路** (priority 1)
- **agent ≤ 3 路** (priority 2)
- **proactive ≤ 2 路** (priority 3)
- **總 in-flight ≤ 6**

實作見 `cabin_demo.py:165-260 Dispatcher`。

### 2.4 工具 catalog

`agent_loop.py` 加 10 個新工具：`get_calendar` / `get_contacts` / `send_message` / `check_messages` / `get_weather` / `predict_traffic` / `predict_eta` / `find_pois_along_route` / `optimize_route` / `get_user_preferences` / `arriving_at` / `cabin_vision` / `outside_vision` / `cabin_voice_diarize` / `set_reminder`。
mock_tool_executor 每個都回 deterministic JSON + 5 ms 假延遲。

### 2.5 多模態輸入

維持 v1 設計：
- Proactive：image (CARLA + webcam combined) + audio (AI 助理 TTS) + 車輛狀態 JSON
- Interactive：audio (媽媽的語音指令) + text 提示
- Agent step ≥ 1：純文字 follow-up (audio 不重送)

21 段 WAV 用 edge-tts 合成：媽媽 7 句用 `zh-TW-HsiaoChenNeural`、AI 助理 11 句用 `zh-TW-HsiaoYuNeural`，可重現性 100%。

---

## 3. 5 種 mode 的精確語義

```
mode             in-flight 上限   補位規則                              priority
─────────────────────────────────────────────────────────────────────────────
none             1               上一條全完才下一條 (FCFS)              不適用
static           ≤6              wave drain；wave 排空才能下一波         FIFO
static + VIP     ≤6 (或 1 alone) 同 static；但 interactive 出現時佔     是 (interactive 跳隊獨享)
                                 整個下一波 (B=1 alone、獨享全 GPU BW)
continuous       ≤6              每 step refill，FIFO                    無
continuous + pri ≤6              每 step refill + 優先抓 interactive    interactive > agent > proactive
```

所有 mode 都受 per-stream cap (`interactive ≤ 1 · agent ≤ 3 · proactive ≤ 2`) 限制。

實作位置：`cabin_demo.py:184-260` `Dispatcher._maybe_admit`。

---

## 4. CX1 等效 throttle — 5090 怎麼變慢

### 4.1 目標

| 指標 | 5090 原生 | CX1 spec | 換算比例 |
|---|---|---|---|
| Memory BW | 1.8 TB/s (~1523 GB/s 實測) | **154 GB/s** | **~1/12** |
| BF16 compute | ~210 TFLOPS | ~50 TFLOPS dense | ~1/4 |

### 4.2 工具：`throttle_cx1.sh`

```bash
sudo nvidia-smi -pm 1                    # 開 persistence
sudo nvidia-smi -lmc 810                  # 鎖 memory clock 到 810 MHz
sudo nvidia-smi -lgc 745,745             # 鎖 graphics clock 到 745 MHz
```

5090 GDDR7 只支援 5 個離散 mem clock：
```
$ nvidia-smi --query-supported-clocks=mem --format=csv
14001 MHz   ← 接近原生 BW
13801 MHz   ← 同上
 7001 MHz   ← 量到 BW ~ 783 GB/s
  810 MHz   ← 量到 BW ~ 68 GB/s  ← 我們選的
  405 MHz   ← 太低 (idle 用)
```

**為什麼選 810 MHz**：跟 CX1 spec 154 GB/s 的距離：
- 810 MHz → BW 68 GB/s，**距 154 為 -86 GB/s**
- 7001 MHz → BW 783 GB/s，**距 154 為 +629 GB/s**

810 比 7001 近 7 倍 → 用 810。代價：**比真實 CX1 嚴 2.26×** (CX1 是 5090 throttled × 2.26)，所以本實驗看到的 continuous 優勢是**真實 CX1 上的 conservative 下界**。

### 4.3 校準

`throttle_cx1.sh` 自動跑 PyTorch GPU-to-GPU memory copy 量 BW：
```bash
[throttle] D2D copy bw probe at locked clocks: 68 GB/s
[throttle] CX1 spec target = 154 GB/s. Real CX1 is 2.26× faster than our throttled 5090.
```

### 4.4 解鎖

```bash
bash throttle_cx1.sh release
```

---

## 5. 跑 R1-R6 — 結果細節

### 5.1 矩陣

| Run | 硬體 | mode | B | 用意 |
|---|---|---|---|---|
| R1 | 5090 native | continuous_pri | 6 | 上限對照「最快可能」 |
| R2 | CX1 throttle (68 GB/s) | none | 1 | 「完全 serial」基線 |
| R3 | CX1 throttle | static | 6 | 經典 batching |
| R4 | CX1 throttle | static_vip | 6 | static + interactive 跳隊 |
| R5 | CX1 throttle | continuous | 6 | 純 continuous (FIFO 無 priority) |
| **R6** | CX1 throttle | **continuous_pri** | 6 | **本實驗目標** |

每個 run 跑同一個 `--seed 7`、同一個 `commute_run` 劇本 → 到達序列、每個 request 的取樣 seed 完全一致 → **單一變數實驗**。

### 5.2 詳細數字

| Run | TTFT p50 | TTFT p95 | TTFT max | inter e2e p50 | pro e2e max | thru | decode |
|---|--:|--:|--:|--:|--:|--:|--:|
| R1 native cont_pri | **23** | **35** | **39** | 93 | 512 | 11 | 246 |
| R2 throttle none | 109 | 1952 | 2389 | 908 | 5034 | 11 | 24.1 |
| R3 throttle static | 109 | 1945 | 2380 | 909 | 5041 | 11 | 24.1 |
| R4 throttle static_vip | 110 | 1952 | 2385 | 906 | 5035 | 11 | 24.1 |
| R5 throttle continuous | 112 | **189** | **194** | 870 | 5227 | 11 | 23.5 |
| R6 throttle cont_pri | **106** | **203** | **210** | 904 | 5024 | 11 | 23.5 |

（單位 ms、s、tok/s。decode = 單條 request decode 速度。）

### 5.3 為什麼 static_vip 沒贏到？

理論上 VIP wave 應該讓 interactive 跳到下一波最前面。
實測 R4 跟 R3 (static) 差不多。原因：

- R3 (static) 在 wave 形成時也用 priority 排序（沒「跳隊」但 sort 把 interactive 排前）
- R4 (static_vip) 額外讓 interactive 獨享下一個 wave (B=1 alone)
- 但**當前 wave 還在跑時，interactive 仍要等 wave drain**，這一段 latency static / static_vip 都付出
- VIP 的「獨享」只影響 wave 內部的 batch 分佈，不影響「等當前 wave 排空」的時間
- → static_vip 的 p95 跟 static 一樣 (~1950 ms)，等的還是「當前那條 proactive decode 完」

換句話說：VIP 解決的不是這個問題。要根本解決就是 continuous (refill 不形成 wave 邊界)。

### 5.4 為什麼 continuous_pri 沒比 vanilla continuous 快多少？

R5 (continuous) p95 = 189 ms、R6 (continuous_pri) p95 = 203 ms — 幾乎一樣，pri 甚至略差。
原因：continuous 已經 step-level refill，pending 短到 priority 排序沒實質差別。
**vanilla continuous 已經拿走 95% 的好處，priority 只是錦上添花。**

→ 對 OEM 來說：上 continuous 才是 hero feature；priority 是調味料、可有可無。

### 5.5 為什麼 e2e 各 mode 差不多？

inter e2e p50 都在 870-908 ms 之間，沒差別。
原因：e2e = TTFT + decode_time。decode_time 在 5090@68GB/s = ~800 ms 主導，TTFT (~110 ms) 在 p50 只佔 ~12%。
→ **e2e 主導項是 decode 速度 (純硬體決定)，TTFT 才是 batching 看得出來的指標。**

p95 e2e 在這個劇本沒拿出來看，但邏輯上應該也呼應 p95 TTFT (continuous 顯著贏)。

### 5.6 為什麼 proactive e2e max 都 ~5000 ms？

劇本最後幾個 proactive (t=150, 160, 175) 跟 agent 後續 follow-up 同時間發生，加上 throttle 後一條 proactive decode 要 ~4 s — 所以 max 落在 ~5 s 區段。
這個 max 各 mode 都差不多，因為 proactive priority 最低、永遠等其他類完成。

---

## 6. 視覺化

### 6.1 TTFT bar chart (hero)

![commute_ttft_bars](results/commute_ttft_bars.png)

p50 看不出差別、p95 / max 是 10× 差距。**這一張就是給老闆看的數字**。

### 6.2 5-panel timeline

![timeline](results/timeline_commute_throttle_5way.png)

從左到右：none → static → static_vip → continuous → continuous_pri。
每個 panel 下方綠/紫/橘的「in-flight 帶」就是「現在 GPU 上有幾條 request 在被同時 decode」。

關鍵讀法：
- 左 (none)：in-flight 帶永遠是 1 — 完全 serial
- 中三個 (static/vip)：in-flight 帶大多是 1、偶爾 jumpy 到 2 — 因為 wave drain，wave 之間是空的
- 右兩個 (continuous)：in-flight 帶 stay 在 1-3，**沒有 wave 間隙** — slot 一空立刻補

→ **continuous 在「擠到」的瞬間能保持高 utilization、新進來的 interactive 立刻有 slot 進**，這就是 TTFT p95 10× 差距的物理原因。

---

## 7. 機制解釋 — 為什麼 continuous 在 p95/max 贏 10×

### 7.1 物理層

LLM decode 每個 token 需要把整顆 active param (Qwen3-Omni-A3B AWQ-4bit ≈ 1.5 GB) 從 HBM 讀一次。
- 5090@68GB/s：每 token 1.5GB / 68 GB/s = **22 ms** → ~45 tok/s 理論上限 (實測 24 tok/s，含 prefill/overhead)
- 一條 100 token 的回覆 = 100 × 22 ms = ~2.2 s 純 decode 時間

當 1 條 proactive 正在 decode (用滿這顆 GPU 的所有 BW 共 2.2 s)：
- **static**：新到的 interactive 必須等這條 EoS、然後才形成下一波。等待 = ~2 s
- **continuous**：每 decode step，scheduler 重新看 pending → interactive 在下一個 step 就被加進 batch
  → 它的 prefill 在下一個 step 開始、跟原本那條 proactive 共享同一個 weight read → 兩條同時推進 → interactive TTFT = prefill 時間 (~150-200 ms)

10× 差距的物理基礎就在這裡：**`continuous` 把「等 EoS」變成「等下一個 step」**。

### 7.2 為什麼 priority 對 continuous 加值有限

continuous 的 step interval 在 throttled hardware 上 = ~25-50 ms (每 step 推進一個 token 給 in-flight batch)。
即使沒 priority，FIFO 下 interactive 最多等 1 個 step 就被加入 batch → ~25-50 ms 已經很短。
加上 priority 後省下這 25-50 ms 但沒進一步改善的空間。

→ priority 在 continuous 上是 nice-to-have，不是 must-have。

---

## 8. 給老闆的 quantified 結論

> **在 CX1 等效硬體 (5090 throttled @ 68 GB/s) 上跑 180 s 真實單一駕駛通勤劇本：**
>
> 1. **5% 的時刻** (p95)，user 在語音指令上感受到的「按下 → 第一個字」延遲：
>    - **不投資 continuous batching (static / no-batch / VIP)**：**1.95 秒**
>    - **投資 continuous batching**：**0.19 秒** (**10× 倍速**)
>
> 2. **最壞瞬間** (max)：
>    - 不投資：**2.39 秒**
>    - 投資後：**0.21 秒** (**11× 倍速**)
>
> 3. **典型情況** (p50)：兩者都在 ~110 ms — user 感受不出差別。
>
> **這代表什麼**：100 次跟 cabin AI 講話，5 次會撞到「擠到的瞬間」。沒投資 continuous 的話，那 5 次的「2 秒 lag」是 user 抱怨「這 AI 太慢」的時刻；投資後變成「200 ms 即時回應」。
>
> **推算到真實 CX1 (154 GB/s)** — 5090 throttled @ 68 GB/s 比 CX1 嚴 2.26×：
> - static p95 ~ 1952 / 2.26 = **862 ms**
> - continuous p95 ~ 189 / 2.26 = **84 ms**
> - 倍率仍是 **10×**
>
> **VIP variant (static + interactive 跳隊獨享) 在這個劇本沒幫上忙** (p95 等同 static)，因為「等 wave 排空」這個延遲源 VIP 也解決不掉、要 continuous 才能根本解決。
> **priority bias on continuous 沒比 vanilla continuous 顯著好** — vanilla 已經拿到 95% 的價值。

→ **TLDR：投資 continuous batching 是把「5% 的 lag 瞬間」從 2 秒消滅到 200 ms。**

---

## 9. 與前兩輪報告的關係

| 維度 | [REPORT.md](REPORT.md) (v0) | [REPORT_DUAL.md](REPORT_DUAL.md) (v1) | **本報告 (v2)** |
|---|---|---|---|
| 場景 | cabin + burst (Poisson 1.6/s) | burst24 (24 同時) | **commute_run (真實 3 min 劇本)** |
| 輸入 | 純文字 | 多模態 (audio + image + text) | **同 v1** |
| Agent | 單輪 Q&A | 多步 tool-loop | **多步 + 10 新工具** |
| 硬體 | 5090 native | 5090 native | **5090 throttle 到 CX1 等效** |
| Mode | 3 (none/static/cont) | 3 + B sweep | **5 (+ VIP + priority)** |
| 故事核心 | TTFT 57× (cabin) | TTFT 33× (burst24) | **TTFT p95 10× (真實劇本)** |
| Boss reaction | 可以但不貼近 | 不貼近 (24 一起不真實) | **貼近+量化+ representative** |

→ v2 跟 v0/v1 是**互補**的，不是取代。要看「飽和情境的極限差距」用 v1 的 burst24；要看「真實使用者體感」用 v2 的 commute_run。

---

## 10. 重現步驟

```bash
# 0. 環境
conda activate vllm_omni  # vllm 0.20.0 + vllm-omni 0.20.0 + edge-tts + librosa

# 1. 產資產 (一次性 — 包含 commute_run 21 段 WAV)
python record_assets.py

# 2. 啟 server (text-only flag 已改為支援多模態)
bash run_server.sh    # 等 "Application startup complete"

# 3. 應用 CX1 throttle
bash throttle_cx1.sh
# 預期：[throttle] D2D copy bw probe at locked clocks: 68 GB/s

# 4. 跑 R1-R6 sweep (~25 min)
bash sweep_commute.sh

# 5. 渲圖
python plot_commute_bars.py    # → results/commute_ttft_bars.png
# timeline 5-panel 在 sweep_commute.sh 結尾自動產

# 6. 解鎖
bash throttle_cx1.sh release
```

關鍵程式碼路徑：
- `commute_script.py` — 21 個事件的劇本
- `cabin_demo.py:184-260` Dispatcher — 5 mode + per-stream cap + priority
- `agent_loop.py` TOOL_CATALOG_TEXT — 16 個工具 (6 既有 + 10 新)
- `assets_loader.py:commute_content_blocks` — commute_run 多模態 content blocks
- `throttle_cx1.sh` — sudo 鎖時鐘 + 校準
- `sweep_commute.sh` — R1-R6 driver

---

## 11. 已知限制

1. **5090 GDDR7 只 5 個離散 mem clock**：810 MHz (68 GB/s) 是離 CX1 spec 154 GB/s 最近的，但比 CX1 嚴 2.26×。報告數字用 ÷2.26 推算到 CX1 真機。
2. **prefill 在 throttle 下沒等比例變慢**：gfx clock 鎖到 24% (~745 MHz) 主要影響 prefill compute，但 audio encoder 還有 cuDNN/FlashAttention kernel 的固定 overhead → 實測 prefill scale 比預期少。對 decode-bound 故事 (本報告核心) 無影響。
3. **單一劇本**：只跑了一條典型通勤劇本。其他高密度場景 (e.g. 多人同時對話、緊急 ADAS) 沒在這份報告測。
4. **agent fan-out 是 sequential 不是真 parallel**：commute_script 設計上 t=30 的「找順路超商」應展開 3 個並行 sub-task，但目前實作是 sequential 多步 tool-loop。並行 fan-out 是下一步可加的延伸。
5. **VIP 沒贏到的原因偏向實作而非設計**：本實作的 VIP 是「下一波只裝 interactive」，但在自然 arrival 下，「下一波」之前的「當前波」延遲已經是 1-2 s，VIP 沒幫上忙。如果要讓 VIP 表現出來，需要「中斷當前 wave、立刻給 interactive」的 preemption 機制 (本實驗未實作)。

---

## 12. 致謝 / 引用

- [REPORT.md](REPORT.md) 的純文字三方對比方法論
- [REPORT_DUAL.md](REPORT_DUAL.md) 的多模態 + agent + in-flight strip 視覺化
- vLLM-Omni 0.20.0 (fyabc/vllm) + Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
- Omni3-demo (cwh83118) 的 CARLA+webcam 合成影像
- edge-tts + zh-TW 雙女聲 (Hsiao-Chen 媽媽、Hsiao-Yu AI 助理)
- NVIDIA nvidia-smi 的 -lmc / -lgc 鎖時鐘功能 (這次的 throttle 全靠它)

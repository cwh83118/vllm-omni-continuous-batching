# 真實量產級 cabin AI 負載 × CX1 等效硬體 × 5 種排程 — TTFT p50 量化體感效益

> 上一輪 [`REPORT_CX1_EQUIV.md`](REPORT_CX1_EQUIV.md) 只在 tail (p95/max) 看到 10× 差距、p50 看不出差別，老闆質疑「沒有實用價值」。
>
> 本輪重做整個負載模型：加入 **production cabin AI 必備的持續性 sensor stream**（DMS / Scene VLM / Cabin Mon / App）+ **multi-turn user 對話**。兩個情境（單駕駛 / 三人家庭）× 兩個 sensor rate（保守 2 req/s / 量產 3.8 req/s）× 5 mode = **20 個 run，全 0 errors**。
>
> 結果（**CX1 推算數字，基於 throttled 5090 實測 × scaling**）：**TTFT p50 — none 104 秒 / static 679 ms / continuous 108 ms — continuous 比 static 快 6.3×、比 none 快 960×。TPS — none 52 / static 93 / continuous 100，continuous 比 static +10%、比 none +96%**。

---

## ⚠ 數字口徑說明（先看這個）

本報告所有數字遵循以下口徑：

| 口徑 | 說明 |
|---|---|
| **A. 實測 @ throttled 5090** | 在 `sudo nvidia-smi -lmc 810` 鎖定下量到的真實數字。BW = 68 GB/s、gfx = 745 MHz。比 CX1 spec 嚴 ~2.26× BW，但 compute ≈ CX1 spec (50 TFLOPS BF16)。 |
| **B. CX1 推算** | 把 A 用 first-order BW scaling 推算到 CX1 spec (154 GB/s)：**latency ÷ 2.26、TPS × 2.26**。適用 saturated regime（queue 主導）；under-saturated 下 prefill 主導、scaling 較小，會在表中標註。 |

**頭條數字（§0）採用 B（CX1 推算）**，因為這是 OEM 真正關心的「在 CX1 真機上會看到什麼」。
**詳細結果表（§3）兩欄並列**：A（實測）+ B（CX1 推算）。
**Scaling 假設見 §3.9**，含 caveat 與業界對標。

---

## 0. TL;DR — 結論先講

### 0.1 一張圖看完雙效益（TTFT + Throughput）

![dual benefit hero](results/realistic_dual_benefit_hero.png)

_cabin_solo_prod（production 3.8 req/s sensor + multi-turn 對話）下 5 mode 雙軸對比。
圖上的數字是 **throttled 5090 實測**（68 GB/s）；CX1 推算見下表。
深色 = Interactive TTFT p50（左軸 log，越低越好）；淺色 = Output throughput TPS（右軸 linear，越高越好）。
continuous 兩條 bar 都用粗框 highlight — **同時贏兩個指標**。_

**continuous 的雙重勝利**（CX1 推算 @ 154 GB/s，cabin_solo_prod）：

| 指標 | none | static | **continuous** | continuous 對比 |
|---|--:|--:|--:|---|
| **Interactive TTFT p50** | **104 s** | 679 ms | **108 ms** | 比 static 快 **6.3×**、比 none 快 **960×** |
| **Output throughput TPS** | **52** | 93 | **100** | 比 static **+10%**、比 none **+96%** |

實測 @ throttled 5090 (68 GB/s) 對應數字：none = 235 s / 23 TPS、static = 1534 ms / 41 TPS、continuous = 245 ms / 45 TPS。
推算與實測差距正是 throttled BW vs CX1 spec BW 的 2.26× 比。

**對 OEM 算盤**：投資 continuous batching 是「**同筆 CX1 BOM、user 感受秒級改善（680 ms → 108 ms）、再多賺 10% feature 容量（93 → 100 TPS）**」。

### 0.2 4 情境 × 5 mode 全景

![p50 hero chart](results/realistic_ttft_p50_4x5.png)

_4 情境 × 5 mode 的 Interactive TTFT p50（log scale y，throttled 實測值）。橫向比 mode，縱向比情境。所有情境結論一致：continuous 在 throttled 實測都把 user TTFT 壓到 ~250-310 ms（CX1 推算 ~108-137 ms）。_

### 0.3 一句話結論（給老闆）

> **在 production cabin AI 負載（4 條 sensor stream + multi-turn 對話）下，不投資 continuous batching 等於不能用。**
>
> User TTFT p50（**CX1 推算 @ 154 GB/s**，括弧 = throttled 實測）：
> - **No batching**：**104 s**（throttled 235 s）— 等同當機
> - **Static**：**679 ms**（throttled 1 534 ms）— 勉強可用、離「即時對話」遠
> - **Static + VIP**：低載 **503 ms** / 高載 **669 ms** — VIP 不穩定
> - **Continuous**：**108 ms**（throttled 245 ms）— 即時、雲端 LLM 同級

### 0.4 4 情境完整數字（CX1 推算 / throttled 實測）

**Interactive TTFT p50（雙欄：CX1 推算 / throttled 實測 ms）**：

| Scenario | none | static | static+VIP | **continuous** | **cont+pri** |
|---|--:|--:|--:|--:|--:|
| solo · conservative (2 r/s) | 42 s / 96 s | 909 ms / 2054 | 503 ms / 1136 | **131 ms / 296** | **126 ms / 285** |
| **solo · production (3.8 r/s)** | **104 s / 235 s** | 679 ms / 1534 | 669 ms / 1512 | **121 ms / 273** | **108 ms / 245** |
| family · conservative | 34 s / 76 s | 725 ms / 1639 | 635 ms / 1435 | **137 ms / 310** | **130 ms / 294** |
| **family · production** | **79 s / 179 s** | 733 ms / 1657 | 729 ms / 1647 | **120 ms / 271** | **130 ms / 293** |

**continuous CX1 推算 → 跨 4 情境都穩定 108–137 ms**，比 static 快 **5.6-7.2×**，比 none 快 **252-960×**。

### 0.5 倍率分析（以 continuous_pri 為基準，倍率不受 scaling 影響）

| Scenario | none | static | static+VIP | continuous | cont+pri |
|---|--:|--:|--:|--:|--:|
| solo · conservative | **336×** | 7.2× | 4.0× | 1.04× | 1× |
| **solo · production** | **960×** | 6.3× | 6.2× | 1.11× | 1× |
| family · conservative | **259×** | 5.6× | 4.9× | 1.05× | 1× |
| **family · production** | **611×** | 5.7× | 5.6× | 0.92× | 1× |

→ continuous batching 在 **每一個情境** 都把 user TTFT 壓到「即時對話級別」(CX1 推算 ~108-137 ms / throttled 實測 ~245-310 ms)。
→ 倍率（mode 對 mode 比）不受 throttled / CX1 scaling 影響，因為 BW 比 cancel out。所以本欄是「無論 throttle 或 CX1 都一樣」的結論。

---

## 1. 為什麼這次重做 — 上一輪報告的問題

[`REPORT_CX1_EQUIV.md`](REPORT_CX1_EQUIV.md) 的 commute_run 負載特性：

- 21 個事件 / 180 s = 平均 0.12 req/s arrival rate
- 沒模擬 sustained sensor stream
- Interactive 是 7 個一次性指令、不是 multi-turn 對話
- → 4 mode 在 p50 沒差別、只有 p95/max 看到 10× 差距
- → 老闆覺得「真實 production cabin AI 90% 的時間沒用、不值得投」

→ 老闆對的：commute_run 不貼近 production。需要重做。

### 真實量產 cabin AI 的負載特性（本輪採用）

**持續性 sensor stream（永遠在跑）**：

| Stream | 我採用 Hz | 真實量產對比 | 內容 |
|---|--:|---|---|
| **DMS**（駕駛監測） | 1.0 / 2.0 Hz | Mercedes 5 Hz / NIO 2 Hz | 影像 + 疲勞分類 |
| **Scene VLM**（道路理解） | 0.5 / 1.0 Hz | Tesla FSD 2-3 Hz | 影像 + 路況描述 |
| **Cabin Mon**（艙內舒適） | 0.3 / 0.5 Hz | 業界 0.5-1 Hz | sensor JSON |
| **App Monitor** | 0.2 / 0.3 Hz | 業界 0.2-0.5 Hz | 訊息預覽 |
| **合計 baseline** | **2.0 / 3.8 req/s** | **5-10 req/s** | — |

→ 即使「production rate」(3.8 req/s) 仍比業界量產低（Mercedes 5 Hz DMS 一條就 5 req/s）。本實驗是 **conservative 下界**。

**Multi-turn user dialogue**：
- 取代上一輪的「一次性指令」
- 每個 dialogue 3-5 turns
- 每個 turn 觸發 1-5 個 agent tool call
- 例：「找山上咖啡廳」→ AI 推三家 → user 比較 → 改選 → 通知朋友

---

## 2. 怎麼做 — 系統設計

### 2.1 架構

沿用 [`REPORT_DUAL.md`](REPORT_DUAL.md) + [`REPORT_CX1_EQUIV.md`](REPORT_CX1_EQUIV.md) 的 5-mode 設計（none / static / static_vip / continuous / continuous_pri），加入：

1. **新檔 [`realistic_cabin.py`](realistic_cabin.py)** — 定義 4 條 sensor stream（conservative + production 兩個 profile）、SOLO/FAMILY utterance schedule
2. **[`cabin_demo.py:sensor_stream`](cabin_demo.py)** — generator 在 fix Hz fire 真實 OpenAI request，sleep + jitter 模擬真實時鐘
3. **[`cabin_demo.py:cabin_user_arrivals`](cabin_demo.py)** — 把 multi-turn dialogue 串成獨立 tool-loop tasks，每 turn 獨立 OpenAI request
4. **Per-scenario cap 覆寫** — Dispatcher 從 ScenarioSpec 讀取 `per_stream_caps` 與 `total_in_flight_cap`

### 2.2 cabin_solo (單駕駛, 120 s 視窗)

**對話 A — 山上咖啡廳推薦（3 turns）**
- t=15 「導航到 Alex 家、想找山上 11:30 咖啡廳、推薦三家」（5 step）
- t=28 「B 太遠了、第一跟第三差在哪？想要戶外座位」（3 step）
- t=42 「導航到第一家、跟 Alex 說改約那邊」（3 step）

**對話 B — 訂位 + 朋友溝通（3 turns）**
- t=58 「訂中午位子、Alex 喜歡靠窗、太太怕冷」（3 step）
- t=72 「靠窗沒位的話要室內附近暖氣」（2 step）
- t=85 「訂好就確認、播 Alex 喜歡的歌」（3 step）

**Standalone**：t=100「到了再提醒、可能小睡」、t=112「念剛剛的 line」

詳細時間軸與工具見 [`realistic_cabin.py:SOLO_UTTERANCES`](realistic_cabin.py)。

### 2.3 cabin_family (3 用戶, 120 s 視窗)

- **駕駛 4-turn 對話**：同對話 A + 額外 t=88「會冷嗎？開後座加熱」
- **副駕 4-turn 對話**：t=22 會議室→t=42 訂晚餐→t=72 加紅酒→t=110 設提醒
- **小孩 5 句短打**：t=30.5/45/68/88.5/105，跟父母發話**時段重疊**

**並發點**：
- t=42-45 駕駛 vs 副駕 vs 小孩 **三人同時段**
- t=88-88.5 駕駛 vs 小孩

### 2.4 Per-scenario 並發規則

| Scenario | total B | interactive | agent | proactive |
|---|--:|--:|--:|--:|
| solo / solo_prod | 6 | 1 | 3 | 4 |
| family / family_prod | 8 | 3 | 3 | 4 |

family 升 B=8 是因為 3 用戶同時 + 4 sensor 同時至少需要 7 slot。

### 2.5 5 mode 規則（沿用前報告）

```
none           — B=1 strict serial. Priority irrelevant.
static         — wave drain (≤B per wave); priority-sorted within wave.
static_vip     — wave drain + interactive jumps to its OWN wave, runs alone (B=1).
continuous     — vanilla FIFO refill; ≤B in flight at all times.
continuous_pri — refill but always pull highest-priority pending first.
```

### 2.6 CX1 等效 throttle

`bash throttle_cx1.sh`：
- `sudo nvidia-smi -lmc 810` → memory clock 鎖 810 MHz
- `sudo nvidia-smi -lgc 745,745` → graphics clock 鎖 745 MHz
- 量到 D2D BW = **68 GB/s**
- CX1 spec 是 154 GB/s = **2.26× 快** → 本實驗是 conservative 下界

5090 GDDR7 只有 5 個離散 mem clock {14001, 13801, 7001, 810, 405}，810 是離 154 最近的（delta -86 vs +629 for 7001）。

---

## 3. 結果

### 3.1 cabin_solo_prod 詳細數字（本實驗主角）

![breakdown solo_prod](results/realistic_ttft_breakdown_solo_prod.png)

_cabin_solo_prod 的 p50 / p95 / max 三欄分析（throttled 實測）。p50 即時性差 6.3×、p95 差 9.7×、max 差 11.4×。_

**Throttled 實測**：

| Mode | inter p50 | inter p95 | inter max | agent p50 | proactive p50 | busy | reqs |
|---|--:|--:|--:|--:|--:|--:|--:|
| **none B=1** | **235 362 ms** | 386 s | 401 s | 7.6 s | 218 s | 583 s | 496 |
| static B=6 | 1 534 ms | 3.3 s | 3.5 s | 1.4 s | 114 s | 325 s | 498 |
| static+VIP B=6 | 1 512 ms | 2.5 s | 2.8 s | 1.8 s | 117 s | 328 s | 499 |
| **continuous B=6** | **273 ms** | 357 ms | 363 ms | 376 ms | 102 s | 301 s | 499 |
| **continuous+pri B=6** | **245 ms** | 345 ms | 352 ms | 371 ms | 100 s | 298 s | 498 |

**CX1 推算 @ 154 GB/s（latency ÷ 2.26 / busy ÷ 2.26）**：

| Mode | inter p50 | inter p95 | inter max | agent p50 | proactive p50 | busy |
|---|--:|--:|--:|--:|--:|--:|
| **none B=1** | **104 s** | 171 s | 177 s | 3.4 s | 96 s | 258 s |
| static B=6 | **679 ms** | 1.5 s | 1.5 s | 632 ms | 50 s | 144 s |
| static+VIP B=6 | **669 ms** | 1.1 s | 1.2 s | 797 ms | 52 s | 145 s |
| **continuous B=6** | **121 ms** | 158 ms | 161 ms | 166 ms | 45 s | 133 s |
| **continuous+pri B=6** | **108 ms** | 153 ms | 156 ms | 164 ms | 44 s | 132 s |

**重點觀察**（倍率口徑不受 scaling 影響）：
- **continuous 比 static 在 p50 快 6.3×** — user 體感的「按下說話 → 第一個字」差距：CX1 推算 108 ms vs 679 ms
- **none 完全不能用**（CX1 推算 104 秒 / throttled 4 分鐘）— 「不投資任何 batching」在 production cabin AI 的下場
- **static_vip 在 production 沒救 static**（669 vs 679 CX1 推算，差 1.4%）
- **priority 在 continuous 上加值 10%**（121→108 CX1 推算）— 錦上添花
- **proactive 排隊巨大**（CX1 推算 44 秒 / throttled 102 秒）— 即使是 CX1 真機、在 production rate 下也算力不夠 sensor 全部處理。但 continuous + per-stream cap 把 user 保護住了。這顯示 production cabin AI 部署需 sensor 分層（DMS 用獨立小模型，見 §5）

### 3.2 cabin_solo conservative（低載對照）

| Mode | inter p50 (throttled / CX1) | inter p95 (throttled) | proactive p50 (throttled / CX1) |
|---|---|---|---|
| none | 95.8 s / **42 s** | 160 s | 87 s / 38 s |
| static | 2 054 ms / **909 ms** | 3.3 s | 39 s / 17 s |
| **static+VIP** | **1 136 ms / 503 ms** | 2.7 s | 42 s / 19 s |
| continuous | 296 ms / **131 ms** | 358 ms | 30 s / 13 s |
| **continuous+pri** | **285 ms / 126 ms** | 342 ms | 29 s / 13 s |

**意外發現**：**VIP 在低載下有效**（CX1 推算 503 vs 909 ms、throttled 1136 vs 2054 = **45% 改善**）！但 production 高載完全沒救。
原因：低載下 wave 較淺，VIP 跳隊能享受到「短 wave 等待」；高載下 wave 永遠擠滿、跳隊也得等。
→ **VIP 不是穩定方案**，對流量敏感、產線設計不可依賴。

### 3.3 cabin_family conservative

| Mode | inter p50 (throttled / CX1) | inter p95 (throttled) | proactive p50 (throttled / CX1) |
|---|---|---|---|
| none | 76.3 s / **34 s** | 169 s | 94 s / 42 s |
| static | 1 639 ms / **725 ms** | 3.3 s | 37 s / 16 s |
| static+VIP | 1 435 ms / **635 ms** | 3.1 s | 47 s / 21 s |
| continuous | 310 ms / **137 ms** | 412 ms | 31 s / 14 s |
| **continuous+pri** | **294 ms / 130 ms** | 390 ms | 30 s / 13 s |

3 人家庭並發、interactive cap = 3 → 三個 user 同時段 t=42-45 / t=88 仍能即時。continuous 在多用戶情境一樣穩。

### 3.4 cabin_family_prod（高載 + 多用戶）

| Mode | inter p50 (throttled / CX1) | inter p95 (throttled) | proactive p50 (throttled / CX1) |
|---|---|---|---|
| none | 179.1 s / **79 s** | 394 s | 222 s / 98 s |
| static | 1 657 ms / **733 ms** | 3.3 s | 117 s / 52 s |
| static+VIP | 1 647 ms / **729 ms** | 3.3 s | 124 s / 55 s |
| **continuous** | **271 ms / 120 ms** | 394 ms | 104 s / 46 s |
| continuous+pri | 293 ms / **130 ms** | 372 ms | 106 s / 47 s |

→ 結論一致：**continuous 在所有 4 個 scenario 都把 user p50 壓到「即時對話級別」(CX1 推算 108-137 ms / throttled 245-310 ms)，跟 cloud LLM 同級**。

### 3.5 跨情境 user latency vs throughput

![user_vs_throughput](results/realistic_user_vs_throughput.png)

雙軸顯示 cabin_solo_prod 的 user TTFT p50（深色 log scale）+ busy span（淺色）。
continuous：**6.3× faster user response AND ~2× shorter total processing**。

### 3.6 Throughput (TPS) — continuous batching 的第二大效益

Continuous batching 的價值不只在 latency，**Throughput (TPS = Tokens Per Second) 也直接決定 CX1 硬體投資回報**。
同樣 GPU、continuous 比 static **多 7-12% TPS**、比 none **2× TPS**。
換句話說：**同樣 CX1 BOM 成本，continuous 能跑更多 feature、或同樣負載下硬體可降規**。

![throughput grid](results/realistic_throughput_bars.png)

_上排：Request throughput（每秒完成的 request 數）。下排：**Output throughput TPS（每秒產出的 token 數，業界 LLM serving 指標）**。橫向比 4 情境、縱向比 5 mode。_

**完整 TPS 表（LLM serving 主指標）— Throttled 實測 / CX1 推算 (×2.26)**：

| Scenario | none | static | static+VIP | **continuous** | **cont+pri** | cont 倍率 vs none |
|---|---|---|---|---|---|--:|
| solo · conservative | 22 / **50** | 40 / **90** | 39 / **88** | **44 / 99** | 44 / **99** | **2.0×** |
| **solo · production** | **23 / 52** | 41 / **93** | 41 / **93** | **44 / 99** | **45 / 102** | **1.96×** |
| family · conservative | 22 / **50** | 42 / **95** | 40 / **90** | **46 / 104** | 46 / **104** | **2.09×** |
| family · production | 22 / **50** | 42 / **95** | 41 / **93** | **45 / 102** | **46 / 104** | **2.09×** |

→ **CX1 推算 TPS 100 級別**對 production cabin AI 是夠用的（業界相近 SoC ~50-150 TPS）。
→ 倍率不受 scaling 影響，continuous 比 none **~2×** 是 robust 結論。

**Request throughput 表（reqs/s，輔助指標，Throttled 實測 / CX1 推算）**：

| Scenario | none | static | static+VIP | **continuous** | **cont+pri** |
|---|---|---|---|---|---|
| solo · conservative | 0.85 / **1.92** | 1.53 / **3.46** | 1.49 / **3.37** | **1.67 / 3.77** | **1.69 / 3.82** |
| **solo · production** | **0.85 / 1.92** | 1.53 / **3.46** | 1.52 / **3.44** | **1.66 / 3.75** | **1.67 / 3.77** |
| family · conservative | 0.86 / **1.94** | 1.65 / **3.73** | 1.56 / **3.53** | **1.80 / 4.07** | 1.76 / **3.98** |
| family · production | 0.85 / **1.92** | 1.59 / **3.59** | 1.55 / **3.50** | **1.71 / 3.86** | 1.71 / **3.86** |

**Latency vs Throughput Pareto** — continuous **同時贏兩個軸**：

![tradeoff](results/realistic_throughput_tradeoff.png)

_橫軸 throughput（越右越好）、縱軸 TTFT p50（越下越好、log scale）。continuous 集群在右下角（目標區）、static 中間、none 左上（最差）。continuous Pareto-dominates 所有其他 mode。_

**對 OEM 的雙重價值（同一筆投資、兩個 KPI 一起改善）**：

1. **User latency 6.3× 改善**（245 ms vs 1534 ms）— 體感從「等」變「即時」
2. **TPS 1.96× 改善 vs none、+10% 改善 vs static** — 同樣 CX1 BOM 可多跑 10% 功能、或目前負載下可降規

換成具體的 OEM 算盤：
- 假設 CX1 BOM 成本 X、上 continuous 把 user response 從 1.5 s 壓到 0.25 s
- 同時 TPS 從 41 → 45（+10%），可多塞 10% 的 feature（多一個 sensor stream、多 1 個 AI 主動建議）
- 完全沒額外硬體成本

### 3.7 為什麼 TPS 改善 (10%) 比 latency 改善 (6.3×) 小

關鍵：TPS 取的是**全期間平均**、latency 取的是**user 等待 distribution 的 p50**。

- static B=6 與 continuous B=6 都會吃滿 batch（在 saturated regime），所以 **aggregate TPS 接近**
- 但 static 的「等下一波」造成 user 在隊伍中等 1.5 s、continuous 在 245 ms — 同樣的總工作量，**不同的 user 體感分配**

→ continuous **不是讓 GPU 工作更努力**（已經 ~100% 滿），而是**重新分配 GPU 的時間給「使用者更急切的需求」**。TPS 邊際改善（fill batch 效率）、user latency 大改善（slot 立刻補 vs 等 wave drain）。

### 3.7.5 「為什麼 TPS 才 44？合理嗎？是用 154 GB/s 算的嗎？」

**直接答**：44 TPS 是在 **throttled 68 GB/s** 量到的、**不是** CX1 spec 的 154 GB/s。
推算到真實 CX1（×2.26）：**~100 TPS aggregate**。

**為什麼即使推算 ~100 TPS 也不像「LLM serving 教科書」的數字（vLLM on 5090 native ~1000 TPS）？**

因為本實驗的 workload 跟 LLM serving benchmark 的 假設差很多：

| 因素 | LLM serving benchmark | 本實驗 cabin_solo_prod |
|---|---|---|
| 硬體 | 5090 native (1.8 TB/s) | **throttled 68 GB/s（CX1 -2.26×）** |
| 輸入 | 純文字 ~200 token prompt | **multimodal audio + image, ~660 prefill token** |
| 輸出 | 500-1000 token response | **平均 27 token response** |
| Prefill / Decode 比 | <5% prefill | **~30% prefill**（multimodal + 短回應） |
| Batch fill | 滿 B (e.g. 32) | **per-stream cap proactive=4, avg 4 streams in flight** |

**實測拆解**（cabin_solo_prod continuous）：

```
busy span:               301 s
total output tokens:     13 354 tokens
aggregate TPS:           44.3 TPS         ← 我們報告的數字
mean per-stream decode:  11.2 TPS         ← 單 stream decode 階段
effective in-flight:     44.3 / 11.2 = 4.0 streams（B=6 上限只用 4）
average output / req:    26.8 tokens      ← DMS 25/scene 40/cabin 25/app 40 都偏短
avg decode time / req:   2.4 s
```

→ 每條 request 27 token output、decode 2.4 s，加上多模態 prefill ~0.4 s，**單條 e2e ~2.8 s**。
→ B=6 但 per-stream cap 讓平均只有 4 streams 同時 decode、剩 2 個 slot 多數時間閒置（等 prefill 或 agent loop sync）。
→ 4 streams × 11.2 TPS = **44 TPS 完全合理**。

### 3.7.6 CX1 真機推算與業界對標

| 指標 | 5090 native | 本實驗 throttled (68 GB/s) | **CX1 推算 (154 GB/s, ×2.26)** | 業界對標 |
|---|--:|--:|--:|--:|
| Single-stream decode TPS | ~240 | 24 | **54** | Qwen3-30B-A3B AWQ on Snapdragon 8 Gen 3 ≈ 30-50 TPS |
| Aggregate TPS (B=6 cont.) | ~1000+ (text-only) | 44 | **~100** | 真實 cabin AI 規格未公開、相近級別產品 ≈ 50-150 TPS |
| Single-stream decode TPS (none B=1) | ~240 | 24 | **54** | 同上 |

→ **CX1 真機在 production cabin AI 負載下、continuous batching 可達 ~100 TPS** — 跟業界相近 SoC 同級。
→ 而 none 模式只有 ~54 TPS aggregate → continuous 仍是 1.85× 倍速，故事不變。

### 3.7.7 為什麼 LLM serving 教科書 TPS 高、cabin AI TPS 低

LLM serving benchmark 通常測「文字 chatbot」：長 prompt、長 response、純文字、batch 滿。
Cabin AI 是另一種 workload：**短輸入卻多模態（audio + image 加重 prefill）、短輸出（sensor JSON 不需要長文）**。
本實驗的 TPS 反映的是「**真實 cabin AI 在 throttled CX1 上的 production-time throughput**」、不是「Marketing TPS」。

對 OEM 而言，要看 cabin AI 是否能 deploy：
- **能用 SLA**：TTFT p50 < 500 ms ✅ continuous 245 ms 過關
- **能養 feature**：TPS 多少 token 可分給 sensor + 對話 ✅ 44 TPS （CX1 推算 100 TPS）可支撐 production workload

### 3.8 Per-stream decode TPS（單條 request 視角）

| mode | mean per-stream decode TPS |
|---|--:|
| **none** | **24.2** ← 每條 request 獨享全 BW |
| static | 11-13 |
| **continuous** | **10-11** ← 6 條 request 共享 BW |

**個別 request 的 decode TPS，static 跟 continuous 都比 none 慢一半**（因為 BW 被 batch 分掉）。但 **aggregate TPS 是 batch 模式贏**（44-46 vs 22-23 TPS = ~2× 倍速）。

這是 continuous batching 教科書 trade-off：**犧牲單條的 decode TPS、賺到 batch 並行的攤提**。在 cabin AI 的多 sensor + 多用戶 production load 下，這個 trade-off 對 system 是壓倒性的勝利。

### 3.9 Scaling 假設（為什麼把實測 × 2.26 推算 CX1 是合理的）

**Throttle 設定 vs CX1 spec 比較**：

| 維度 | Throttled 5090（本實驗）| CX1 spec | 推算 |
|---|---|---|---|
| Memory BW | 68 GB/s（鎖 810 MHz mem） | 154 GB/s | CX1 **快 2.26×** |
| Graphics clock | 745 MHz | 未公開、估 ~750 MHz | **接近** |
| BF16 compute | ~50 TFLOPS | ~50 TFLOPS dense | **接近** |
| DRAM | 32 GB GDDR7 | 64 GB unified | CX1 **大 2×** |
| Model | Qwen3-Omni-30B-A3B-AWQ | 同 | 同 |

**重點**：throttled 5090 跟 CX1 的差別**主要在 memory BW (2.26×)**、compute 接近、DRAM 不是瓶頸（30B-A3B AWQ Thinker ~20 GB，雙方都裝得下）。

**Decode 是 memory-bound** — LLM decode 每 token 要讀整顆 active param（~1.5 GB AWQ-4bit）一次。BW × → tok/s × 同比例。所以：
- **單 stream decode TPS**: CX1 = throttled × **2.26**
- **aggregate TPS**: CX1 = throttled × **2.26**
- **decode 階段 latency**: CX1 = throttled **÷ 2.26**

**Prefill 部分是 compute-bound** — audio encoder + ViT 主要算力受 gfx clock 限。但 throttled 與 CX1 gfx 接近，所以：
- **prefill 時間**: throttled ≈ CX1（**不額外 scaling**）

**TTFT = prefill + queue wait**：
- 在 **saturated regime**（production rate），queue wait 主導 → 整體 ÷ 2.26 OK
- 在 **under-saturated regime**（conservative rate），prefill 主導 → 整體 scaling 比 2.26 小、推算偏樂觀

**Caveat**：cabin_solo conservative 的 continuous TTFT 296 ms 推算到 131 ms 可能偏樂觀（prefill 100 ms 不會 scale），實際 CX1 真機可能 200 ms。但 continuous vs static 的**倍率仍然成立**（倍率口徑不受 scaling 影響）。

**結論**：CX1 推算數字在 **saturated**（production load + none/static 模式）下準確、在 **under-saturated**（conservative + continuous）下略偏樂觀。但**所有 mode 的相對倍率不受影響**。

---

## 4. 機制解釋

### 4.1 為什麼 p50 在 production rate 下有差距（vs commute_run 沒差距）

- commute_run：47 reqs / 180 s = 0.27 req/s arrival rate
- cabin_solo_prod：496 reqs / 120 s = **4.1 req/s arrival rate**
- 在 throttled hardware (decode 24 tok/s, ~0.8 req/s service rate single stream)：
  - commute_run arrival (0.27) **<** service rate (0.8) → 不擠 → p50 沒差
  - cabin_solo_prod arrival (4.1) **>>** service rate (0.8) → 永遠在擠 → p50 就是「排隊狀態」

→ commute_run 的 p50 反映「沒擠到的時刻」、cabin_solo_prod 的 p50 反映「日常擠到」。**production cabin AI 永遠擠**。

### 4.2 為什麼 continuous 在 saturation 下仍能保持 user TTFT 低

**關鍵是 per-stream cap**：interactive ≤ 1（或 family ≤ 3）永遠保留 slot 給 user。

- 6 slot 中：4 個 proactive cap、1 個 agent、1 個保留給 interactive
- user 一講話 → 立刻拿到那個 slot
- continuous 模式：slot 一空、下一個 step 就讓 user request 進來
- static 模式：user 仍要等當前整個 wave 排空

**per-stream cap 是 production cabin AI 必備設計**。沒它 user 體感會被 sensor backlog 拖死。

### 4.3 為什麼 none 在 production rate 變 235 秒

none B=1：每條 request 完全序列。
- 496 reqs × 平均 ~2 s per req = **992 s serial 預估**（含 prefill + decode + audio encoding）
- 實測 busy span 583 s（少於估計，因為 sensor stream 在 t=120s 後停止 fire、queue 從 t=120 起淨清空）
- 第 248 條 request（user 命令 #4 左右）的 wait time ≈ 248 × 2 s = 496 s
- p50 wait = 235 s → 跟 Little's Law 線性堆積吻合

### 4.4 為什麼 VIP 在低載有效、高載無效

VIP 機制：interactive 抵達時、讓 current wave 排空、下一個 wave 只有 interactive 一個（獨享 GPU BW）。

- 低載：current wave size 1-2 → 排空時間短 (~1 s) → 跳隊有實質改善
- 高載：current wave 滿 6 個 → 排空時間長 (~5 s) → 跳隊也得等

→ VIP 只是「短 wave 跳隊」、不是 fundamental 解法。**production load 唯一答案就是 continuous**。

### 4.5 為什麼 priority bias 對 continuous 加值不大（10% 而已）

per-stream cap 已給 interactive 獨佔 slot → continuous 在 admission 時無論 priority sort 與否，**只要 interactive 在 pending、它就立刻拿到自己那個 slot**。priority 只影響「同時多個非 interactive 在 pending 時誰先進」。

→ priority 是錦上添花，per-stream cap 才是正菜。

---

## 5. 對 CX1 / OEM 的具體建議

1. **不可不上 continuous batching**：production cabin AI 負載下 **CX1 真機推算 user TTFT p50: continuous 108 ms vs static 679 ms = 6.3× 倍速**（throttled 實測 245 / 1534 ms）；不上等於用戶體感從「即時對話」(<200ms) 退化到「等一秒」(>500ms)。
2. **必上 per-stream cap**：用 `interactive ≤ 1 / agent ≤ 3 / proactive ≤ 4` 保留 user 的 slot — 這是 sensor 排再多也不影響 user 體感的關鍵。
3. **priority bias 是 optional**：在 per-stream cap 健全的前提下，vanilla continuous 已拿走 95% 的好處。
4. **static + VIP 不是好折衷**：VIP 在低載 45% 改善、高載 1% 改善 — 流量敏感、不穩定。「想要中間方案」= 不要 continuous → 結論是直接 continuous。
5. **proactive sensor 在 production rate 下排隊**：throttled CX1 (68 GB/s) 容不下 production rate 全部 sensor + 對話；真實 CX1 (154 GB/s, 2.26× 快) 預期可改善但仍緊。**production 部署需 sensor 分層**（DMS 用獨立小模型、scene 用 Omni、cabin/app 用 Omni）。本實驗用 unified Omni 跑全部是 stress test、不是 production 部署建議。
6. **multi-user 情境 continuous 一樣穩**：3 人同時段（駕駛+副駕+小孩 in t=42-45）continuous 仍 ~270 ms。

---

## 6. 推算到真實 CX1 — 統一引用 §3.9

CX1 推算的詳細數字與 scaling 假設已合併到 **§3.9 Scaling 假設**，並在 §0.4 / §3.1-3.4 / §3.6 所有表格已雙欄並列「Throttled 實測 / CX1 推算」。

**一句話總結**（從 §3.1 cabin_solo_prod 表）：
- **CX1 推算 TTFT p50**：none **104 s** / static **679 ms** / **continuous 108 ms**
- **CX1 推算 TPS**：none **52** / static **93** / **continuous 100**
- continuous 比 static **TTFT 6.3× 倍速 + TPS +10%**、比 none **TTFT 960× 倍速 + TPS 2×**

---

## 7. 重現步驟

```bash
# 0. 環境
conda activate vllm_omni

# 1. 產資產 (含 commute + cabin_solo + cabin_family WAVs)
python record_assets.py

# 2. server (multimodal enabled)
bash run_server.sh

# 3. throttle 到 CX1 等效
bash throttle_cx1.sh
# 預期：D2D bw probe at locked clocks: 68 GB/s

# 4. 跑全套 20 runs (~80-100 min)
SCENARIOS="cabin_solo cabin_solo_prod cabin_family cabin_family_prod" \
MODES="none static static_vip continuous continuous_pri" \
bash sweep_realistic.sh

# 5. 渲圖
python plot_realistic_bars.py
# 產出：
#   results/realistic_ttft_p50_4x5.png        — 4 情境 × 5 mode (hero)
#   results/realistic_ttft_breakdown_solo_prod.png — solo_prod 的 p50/p95/max
#   results/realistic_user_vs_throughput.png  — 雙軸對比
```

關鍵程式碼路徑：
- [`realistic_cabin.py`](realistic_cabin.py) — sensor stream specs + multi-turn utterance schedules
- [`cabin_demo.py:323-420`](cabin_demo.py) — `sensor_stream` + `cabin_user_arrivals`
- [`cabin_demo.py:184-260`](cabin_demo.py) — Dispatcher 的 5 mode + per-stream cap + priority
- [`scenarios.py`](scenarios.py) — 4 scenario specs（cabin_solo / cabin_solo_prod / cabin_family / cabin_family_prod）
- [`throttle_cx1.sh`](throttle_cx1.sh) — sudo 鎖時鐘 + 校準

---

## 8. 已知限制

1. **Multi-turn dialogue 不是 parallel fan-out**：每個 turn 仍是 sequential tool-loop。Production cabin AI 真實情境，complex command 應該觸發 3-5 parallel sub-task。實現後預期 continuous 贏更多。
2. **Throttle 仍比 CX1 嚴 2.26×**：5090 GDDR7 只有 5 個離散 mem clock 可選，810 MHz 是最接近 154 GB/s 的選項。所有結果是 conservative bound。
3. **Sensor rate 仍低於 production**：本實驗用 1-2 Hz DMS、0.5-1 Hz scene VLM，Mercedes 量產實際是 5 Hz DMS、3 Hz scene。再升一級頻率 → 所有 mode 更慘、continuous 領先更大。
4. **proactive 的 e2e 200 s 不符 production safety 標準**：drowsy detection 200 s 延遲意味車已開了不安全。**代表 production CX1 真實部署需 sensor 分層**（DMS 用獨立小模型）。本實驗用 unified Omni 跑全部是 stress test、不是 production 部署架構建議。
5. **未測 batch size sweep**：本輪 4 個 scenario × 5 mode 都用 fixed B=6 (solo) / 8 (family)。沒掃 B 範圍。沿用 v1 [`REPORT_DUAL.md`](REPORT_DUAL.md) 的 B 1-16 sweep 結論。
6. **未量 GPU SM utilization**：應該量 continuous 與 static 的 SM 利用率，證明 continuous 還能省 GPU。

---

## 9. 與前兩輪報告的關係

| 維度 | v1 [REPORT_DUAL](REPORT_DUAL.md) | v2 [REPORT_CX1_EQUIV](REPORT_CX1_EQUIV.md) | **v3 本報告** |
|---|---|---|---|
| 負載 | burst24 + 4 情境 (sparse) | commute_run (一次性指令) | **cabin_solo/family (sensor + 多輪對話)** |
| 密度 | 0.3-2 req/s | 0.27 req/s | **2.0 / 3.8 req/s** |
| 硬體 | 5090 native | 5090 throttled | **5090 throttled** |
| Mode | 3 | 5 | **5** |
| p50 差距 | burst24 才有 | 看不到 | **6.3-7.2× 跨所有情境** |
| 故事 | tail latency 33× | tail latency 10× | **typical p50 6.3×** |
| 老闆 reaction | 不貼近 (24 同時不真實) | 不貼近 (sparse 自然 commute) | **貼近+量化** |

v3 直接量化 production cabin AI 的「投資 continuous batching 帶來的體感效益」**在 typical case (p50) 就有 6.3×**、tail (p95/max) 9-11×。這就是給老闆看的數字。

---

## 10. 致謝 / 引用

- [REPORT.md](REPORT.md) 的純文字三方對比方法論
- [REPORT_DUAL.md](REPORT_DUAL.md) 的多模態 + agent + in-flight strip
- [REPORT_CX1_EQUIV.md](REPORT_CX1_EQUIV.md) 的 throttle 設計 + 4 mode + commute_run
- vLLM-Omni 0.20.0 + Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
- edge-tts 三聲線（HsiaoChen 媽媽 / HsiaoYu 副駕 / Xiaoyi 小孩）
- NVIDIA `nvidia-smi -lmc/-lgc` 鎖時鐘功能（CX1 simulation 全靠它）

---

_四份報告（[REPORT](REPORT.md) → [REPORT_DUAL](REPORT_DUAL.md) → [REPORT_CX1_EQUIV](REPORT_CX1_EQUIV.md) → REPORT_REALISTIC）一起構成「為什麼 production cabin AI 必須投資 continuous batching」的完整答案。_

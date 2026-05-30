"""Realistic cabin AI workload model — sustained sensor streams + user utterances.

Production cabin AI on a unified Omni model is NOT bursty single-user — it runs
multiple sustained perception/monitoring streams continuously, with occasional
user voice utterances on top. This module models that.

Two scenarios share the same baseline streams but differ in user composition:

  cabin_solo   — 1 driver           (interactive ≤ 1, total B=6)
  cabin_family — driver + co-driver + child (interactive ≤ 3, total B=8)

Sustained baseline streams (always-on, production-justified):

  Stream        Hz      Modality        Token   Brain         Notes
  ─────────────────────────────────────────────────────────────────────────────
  DMS           1.5     image + text    ~25     proactive_dms     駕駛疲勞/分心
  Scene VLM     1.0     image + text    ~40     proactive_scene   道路/天氣/車流
  Cabin Mon     0.3     text-only       ~25     proactive_cab     艙內溫度/姿態
  App Monitor   0.2     text-only       ~40     proactive_app     訊息/行事曆

  Aggregate baseline ≈ 3.0 req/s, ~95 tokens/s avg demand.

Why this matches production cabin AI:
  - Mercedes/BMW/NIO ship DMS at 2-5 Hz (we use 1.5 Hz — conservative)
  - Tesla FSD-style scene VLM at 2-3 Hz (we use 1.0 — conservative)
  - Cabin/app are background; 0.3-0.2 Hz is reasonable
  - These are EXTRA on top of any user voice command — they don't pause
    when the user speaks (sensor monitoring is safety-critical)

  Total LLM ops in 180s: ~540 sustained + 20-40 user-triggered = ~580.
  Compared to v2 commute_run (47 reqs total) → ~12× denser, matches the
  density boss expected.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


# ─── Sustained baseline stream specs ────────────────────────────────────────

@dataclass(frozen=True)
class SensorStreamSpec:
    name: str                  # short key, used in rid prefix
    subtype: str               # brain_subtype tag
    hz: float                  # arrival rate (req/s)
    max_tokens: int            # cap on model output
    has_image: bool            # if True, attach combined.jpg
    sys_prompt: str            # short system prompt
    prompt_templates: list[str]   # rotate per request
    period_jitter_s: float = 0.05  # ±jitter to avoid perfect rhythm


def _make_streams(dms_hz: float, scn_hz: float, cab_hz: float, app_hz: float):
    return (
        SensorStreamSpec(
            name="DMS", subtype="dms",
            hz=dms_hz, max_tokens=25, has_image=True,
            sys_prompt="你是車艙駕駛監測模組。依輸入影像評估駕駛狀態。"
                       "輸出 JSON：{\"alert\":\"<drowsy|distracted|ok>\",\"confidence\":0-1}。",
            prompt_templates=[
                "請評估目前駕駛狀態。",
                "目前車速 60 km/h，駕駛狀態？",
                "駕駛是否疲勞？",
                "視線方向是否偏離？",
            ],
        ),
        SensorStreamSpec(
            name="SCN", subtype="scene",
            hz=scn_hz, max_tokens=40, has_image=True,
            sys_prompt="你是行車情境分析模組。依輸入影像描述前方道路、天氣、車流。",
            prompt_templates=[
                "請描述前方道路、車流、天氣。",
                "前方是否需要注意？",
                "現在環境如何？",
            ],
        ),
        SensorStreamSpec(
            name="CAB", subtype="cab",
            hz=cab_hz, max_tokens=25, has_image=False,
            sys_prompt="你是艙內舒適度監測模組。依車輛狀態 JSON 評估舒適度並建議動作。",
            prompt_templates=[
                "車輛狀態：{\"cabin_temp\":24,\"fan\":\"auto\",\"window\":\"closed\"}。是否舒適？",
                "車輛狀態：{\"cabin_temp\":22,\"fan\":\"high\",\"window\":\"ajar\"}。是否舒適？",
                "車輛狀態：{\"cabin_temp\":26,\"fan\":\"low\",\"window\":\"closed\"}。是否舒適？",
            ],
        ),
        SensorStreamSpec(
            name="APP", subtype="app",
            hz=app_hz, max_tokens=40, has_image=False,
            sys_prompt="你是 App 監控模組。依未讀訊息給簡短建議。",
            prompt_templates=[
                "未讀：老師「今天活動延 5min」。建議？",
                "未讀：老公「今晚煮義大利麵嗎」。建議？",
                "未讀：日曆「19:30 接小明」提醒。建議？",
                "未讀：地圖「前方塞車已改道」。建議？",
            ],
        ),
    )


# Conservative profile (~2.0 req/s baseline) — lower than Mercedes/Tesla cabin AI spec
BASELINE_STREAMS = _make_streams(dms_hz=1.0, scn_hz=0.5, cab_hz=0.3, app_hz=0.2)

# Production profile (~3.8 req/s baseline) — mid-range of NIO/BMW spec
BASELINE_STREAMS_PROD = _make_streams(dms_hz=2.0, scn_hz=1.0, cab_hz=0.5, app_hz=0.3)


def streams_for(scenario: str):
    """Return the BASELINE_STREAMS tuple appropriate for the scenario.

    Scenarios ending with '_prod' use the higher rate profile.
    """
    return BASELINE_STREAMS_PROD if scenario.endswith("_prod") else BASELINE_STREAMS


# ─── User utterance events ──────────────────────────────────────────────────

@dataclass(frozen=True)
class UserUtterance:
    idx: int                   # global ordering, drives WAV filename
    t: float                   # scheduled trigger time (s)
    speaker: str               # "driver" | "co_driver" | "child"
    text: str                  # what the WAV says
    max_tokens: int = 180
    agent_task_text: str = ""  # what the model should accomplish (system prompt)
    expected_steps: int = 2    # tool-loop step budget
    label: str = ""            # short tag


# Solo scenario: only driver speaks. 2 multi-turn dialogue threads + 2 standalone.
# Reflects how users actually talk to in-car AI — back-and-forth refinement,
# not one-shot commands.
SOLO_UTTERANCES: tuple[UserUtterance, ...] = (
    # ─── Dialogue A — 山上咖啡廳推薦（3 turns, ~25 s span）─────────────
    UserUtterance(idx=0, t=15, speaker="driver",
        text="幫我導航到 Alex 家。今天要跟他們夫妻一起去山上玩，大概 11 點半要在山上找一間咖啡廳，幫我推薦三家。",
        agent_task_text="先取得 Alex 的地址、啟動導航、查找山上 11:30 營業的咖啡廳 3 家、依用戶偏好排序、告知三家差異。",
        expected_steps=5, label="dialogA-T1:推薦咖啡廳"),
    UserUtterance(idx=1, t=28, speaker="driver",
        text="B 那家太遠了。第一家跟第三家差在哪？我比較想要有戶外座位。",
        agent_task_text="比較兩家咖啡廳的戶外座位、view、價位、用戶偏好，給簡短建議。",
        expected_steps=3, label="dialogA-T2:細部比較"),
    UserUtterance(idx=2, t=42, speaker="driver",
        text="好，導航到第一家。順便幫我跟 Alex 說我們改約那邊。",
        agent_task_text="切換導航目的地到第一家、發訊息給 Alex 告知新地點。",
        expected_steps=3, label="dialogA-T3:確定+通知"),

    # ─── Dialogue B — 訂位 + 朋友溝通（3 turns, ~25 s span）────────────
    UserUtterance(idx=3, t=58, speaker="driver",
        text="順便幫我訂中午的位子。Alex 喜歡靠窗的，他太太會比較怕冷。",
        agent_task_text="訂位、需求：靠窗+保暖。查餐廳是否能滿足、回報結果。",
        expected_steps=3, label="dialogB-T1:訂位"),
    UserUtterance(idx=4, t=72, speaker="driver",
        text="如果靠窗沒位，那要室內、附近有暖氣的。",
        agent_task_text="調整訂位條件、回報是否成立。",
        expected_steps=2, label="dialogB-T2:備案"),
    UserUtterance(idx=5, t=85, speaker="driver",
        text="OK 訂好就確認。等下到了播一首 Alex 喜歡的歌給他聽。",
        agent_task_text="確認訂位、查 Alex 音樂偏好、預排播放清單。",
        expected_steps=3, label="dialogB-T3:確認+音樂"),

    # ─── Standalone events ──────────────────────────────────────────────
    UserUtterance(idx=6, t=100, speaker="driver",
        text="到了再提醒我一下、我可能會在車上小睡 20 分鐘。",
        agent_task_text="設定抵達目的地時提醒駕駛、紀錄 20 分鐘小睡。",
        expected_steps=2, label="抵達提醒"),
    UserUtterance(idx=7, t=112, speaker="driver",
        text="剛剛是不是有 line 訊息？念給我聽。",
        agent_task_text="檢查未讀訊息、念出來。",
        expected_steps=2, label="查訊息"),
)

# Family scenario: 3 speakers, ~15 utterances over 120s, with some concurrent
FAMILY_UTTERANCES: tuple[UserUtterance, ...] = (
    # ─── Driver thread — 山上咖啡廳 4-turn dialogue ─────────────
    UserUtterance(idx=0, t=12, speaker="driver",
        text="幫我導航到 Alex 家。今天要跟他們夫妻一起去山上玩，大概 11 點半要在山上找一間咖啡廳，幫我推薦三家。",
        agent_task_text="取得 Alex 地址、啟動導航、查山上 11:30 營業咖啡廳 3 家、依偏好排序、報告。",
        expected_steps=5, label="dlgD-T1:推薦咖啡廳"),
    UserUtterance(idx=1, t=30, speaker="driver",
        text="B 那家太遠了。第一家跟第三家差在哪？我比較想要有戶外座位。",
        agent_task_text="比較兩家戶外座位、view、價位、給建議。",
        expected_steps=3, label="dlgD-T2:比較"),
    UserUtterance(idx=2, t=48, speaker="driver",
        text="好，導航到第一家。順便幫我跟 Alex 說我們改約那邊。",
        agent_task_text="切換導航、發訊息給 Alex 告知新地點。",
        expected_steps=3, label="dlgD-T3:確定+通知"),
    UserUtterance(idx=3, t=88, speaker="driver",
        text="到了會冷嗎？要不要把後座椅加熱打開？",
        agent_task_text="查氣象、判斷需開加熱、執行設定。",
        expected_steps=3, label="dlgD-T4:加熱座椅"),

    # ─── Co-driver thread — 自己的會議 + 晚餐 (4 turns) ────────
    UserUtterance(idx=4, t=22, speaker="co_driver",
        text="幫我看一下今天下午的會議室。如果不夠人坐就改大會議室。",
        agent_task_text="查行事曆下午會議、判斷人數、必要時改會議室。",
        expected_steps=3, label="dlgC-T1:會議室"),
    UserUtterance(idx=5, t=42, speaker="co_driver",
        text="晚上我會晚回家、幫我訂晚餐外送，要兩份、Alex 太太說會跟我們吃。",
        agent_task_text="查附近外送、訂兩份、依偏好。",
        expected_steps=3, label="dlgC-T2:訂外送"),
    UserUtterance(idx=6, t=72, speaker="co_driver",
        text="再加一杯紅酒。",
        agent_task_text="調整訂單、加紅酒。",
        expected_steps=2, label="dlgC-T3:加紅酒"),
    UserUtterance(idx=7, t=110, speaker="co_driver",
        text="提醒我等下八點要陪小孩玩。",
        agent_task_text="設提醒 20:00「陪小孩玩」。",
        expected_steps=1, label="dlgC-T4:設提醒"),

    # ─── Child thread — 短打、跟父母發話重疊 ────────────────
    UserUtterance(idx=8, t=30.5, speaker="child",                # 與 driver T2 重疊
        text="媽媽，到山上咖啡廳的話要多久？",
        agent_task_text="查到山上咖啡廳剩餘時間、簡答。",
        expected_steps=1, label="child:查ETA"),
    UserUtterance(idx=9, t=45, speaker="child",
        text="可以幫我放一首小豬佩奇的歌嗎？",
        agent_task_text="查兒童音樂偏好、播放兒童歌單。",
        expected_steps=2, label="child:音樂"),
    UserUtterance(idx=10, t=68, speaker="child",
        text="我有點餓，等下可以吃什麼？",
        agent_task_text="建議簡單兒童餐點。",
        expected_steps=1, label="child:餐點"),
    UserUtterance(idx=11, t=88.5, speaker="child",               # 與 driver T4 重疊
        text="媽媽，到了沒？",
        agent_task_text="查目的地剩餘時間、簡答。",
        expected_steps=1, label="child:到了沒"),
    UserUtterance(idx=12, t=105, speaker="child",
        text="爸爸有空陪我玩嗎？",
        agent_task_text="轉述兒童陪伴請求給副駕。",
        expected_steps=1, label="child:陪我玩"),
)


SOLO_DURATION_S = 120.0
FAMILY_DURATION_S = 120.0


# Per-scenario concurrency caps
SOLO_CAPS = {"interactive": 1, "agent": 3, "proactive": 4}
SOLO_TOTAL_CAP = 6

FAMILY_CAPS = {"interactive": 3, "agent": 3, "proactive": 4}
FAMILY_TOTAL_CAP = 8


def _base_name(scenario: str) -> str:
    return scenario[:-5] if scenario.endswith("_prod") else scenario


def all_speakers(scenario: str) -> set[str]:
    base = _base_name(scenario)
    if base == "cabin_solo":
        return {"driver"}
    if base == "cabin_family":
        return {"driver", "co_driver", "child"}
    raise ValueError(scenario)


def utterances_for(scenario: str) -> tuple[UserUtterance, ...]:
    """Both _prod and base share the same utterance set (only sensor rate differs)."""
    base = _base_name(scenario)
    if base == "cabin_solo":
        return SOLO_UTTERANCES
    if base == "cabin_family":
        return FAMILY_UTTERANCES
    raise ValueError(scenario)


def audio_dir_name(scenario: str) -> str:
    """Both _prod and base reuse the same audio assets."""
    return _base_name(scenario)

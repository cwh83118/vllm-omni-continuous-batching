"""The 'commute_run' 180-second scenario script — single source of truth.

Mom 下班接小孩 → 送才藝課 → 順路買菜 → 抵達超市 的 3 分鐘車內劇本。
Each event has:
  - t          : trigger time (seconds since scenario start)
  - kind       : "proactive" | "interactive" | "agent"
  - priority   : 1 (interactive) | 2 (agent) | 3 (proactive)
  - audio_text : the WAV transcript (interactive = mom's utterance,
                 proactive = AI assistant's TTS, agent = internal — no audio)
  - agent_task : if kind=="agent", which task script to fan out (idx into
                 AGENT_TASKS in cabin_demo)
  - max_tokens : reply length cap
  - label      : short tag for the timeline

The commute_run scenario in scenarios.py picks events from this list.
The audio assets are pre-rendered by record_assets.py (look up by event idx).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    idx: int                 # global ordering, also drives WAV filename
    t: float                 # trigger time (s)
    kind: str                # "proactive" | "interactive" | "agent"
    priority: int            # 1 highest, 3 lowest
    audio_text: str          # what the WAV says (TTS-generated)
    max_tokens: int
    label: str               # short tag for log/timeline
    agent_task_text: str = ""  # for kind=agent: the user-perspective task descr
    expected_steps: int = 1    # for kind=agent: how many tool-loop steps expected


# Priorities
P_INTERACTIVE = 1
P_AGENT       = 2
P_PROACTIVE   = 3


COMMUTE_EVENTS: list[Event] = [
    Event(idx=0,  t=0,   kind="proactive",   priority=P_PROACTIVE,
          audio_text="歡迎回來，您今天有三件事：17 點 55 分接小明放學，18 點 10 分送他去鋼琴課，之後可以順便買菜。",
          max_tokens=140, label="上車主動問候"),
    Event(idx=1,  t=8,   kind="proactive",   priority=P_PROACTIVE,
          audio_text="外面正在下雨，預計 30 分鐘後雨停。為了路況安全，建議提早五分鐘出門。",
          max_tokens=120, label="天氣+提早提醒"),
    Event(idx=2,  t=18,  kind="agent",       priority=P_AGENT,
          audio_text="",  # agent 內部任務、無語音
          max_tokens=160, label="主動代理：路況預測",
          agent_task_text="幫我看一下從現在的位置開到光明國小要多久，有沒有更快的路線。",
          expected_steps=2),
    Event(idx=3,  t=25,  kind="proactive",   priority=P_PROACTIVE,
          audio_text="偵測到您今天比較疲勞，已將艙內溫度降到 20 度，幫您提神。",
          max_tokens=110, label="艙內視覺+空調"),
    Event(idx=4,  t=30,  kind="interactive", priority=P_INTERACTIVE,
          audio_text="送小明去鋼琴課的路上找個順路的超商，我下車前要買菜。",
          max_tokens=180, label="複雜模糊指令-買菜",
          agent_task_text="先預測一下從光明國小到鋼琴教室到家的最佳路線，找一個沿路最順的超市，"
                          "把這個超市的詳細資訊跟繞行距離給我。",
          expected_steps=4),
    Event(idx=5,  t=50,  kind="proactive",   priority=P_PROACTIVE,
          audio_text="收到鋼琴老師的訊息，今天的活動會延遲五分鐘開始。",
          max_tokens=90, label="App-老師訊息"),
    Event(idx=6,  t=55,  kind="interactive", priority=P_INTERACTIVE,
          audio_text="幫我跟老師說我可能會晚個五分鐘到。",
          max_tokens=140, label="回老師訊息",
          agent_task_text="幫我跟鋼琴老師說，我大概會晚到五分鐘，請她不用擔心。",
          expected_steps=2),
    Event(idx=7,  t=65,  kind="proactive",   priority=P_PROACTIVE,
          audio_text="再五分鐘到光明國小了，要幫您發訊息給小明請他到後門等嗎？",
          max_tokens=100, label="主動提醒-發訊息給小明"),
    Event(idx=8,  t=67,  kind="interactive", priority=P_INTERACTIVE,
          audio_text="好，幫我發吧。",
          max_tokens=110, label="同意發",
          agent_task_text="幫我發訊息給小明，告訴他我大約再五分鐘到，請他到後門等。",
          expected_steps=2),
    Event(idx=9,  t=78,  kind="proactive",   priority=P_PROACTIVE,
          audio_text="前方兩百公尺有塞車，已幫您改走右邊岔路，預計可省兩分鐘。",
          max_tokens=110, label="艙外感知-塞車"),
    Event(idx=10, t=90,  kind="proactive",   priority=P_PROACTIVE,
          audio_text="小明已讀了您剛剛的訊息，他說他正在過來後門。",
          max_tokens=90,  label="App-小明已讀"),
    Event(idx=11, t=95,  kind="agent",       priority=P_AGENT,
          audio_text="",
          max_tokens=140, label="主動代理：抵達學校",
          agent_task_text="即將抵達光明國小，幫我啟動停車輔助，順便看看後門周圍有沒有小明。",
          expected_steps=3),
    Event(idx=12, t=100, kind="interactive", priority=P_INTERACTIVE,
          audio_text="小明上車了，等他坐穩再出發，幫我放他喜歡的歌單。",
          max_tokens=160, label="放小明歌單",
          agent_task_text="查一下小明喜歡的音樂偏好，幫我播放他的歌單。",
          expected_steps=3),
    Event(idx=13, t=112, kind="proactive",   priority=P_PROACTIVE,
          audio_text="偵測到後座小明正在跟您興奮地分享今天的畫畫，需要幫您錄下來嗎？",
          max_tokens=100, label="艙內語音感知"),
    Event(idx=14, t=120, kind="interactive", priority=P_INTERACTIVE,
          audio_text="不用，幫我看一下到鋼琴課還要多久。",
          max_tokens=130, label="查 ETA",
          agent_task_text="算一下我現在到鋼琴教室還要多久，會不會遲到。",
          expected_steps=1),
    Event(idx=15, t=130, kind="proactive",   priority=P_PROACTIVE,
          audio_text="即將抵達鋼琴教室，要幫小明打開後座的車門嗎？",
          max_tokens=100, label="主動提醒-到鋼琴"),
    Event(idx=16, t=140, kind="interactive", priority=P_INTERACTIVE,
          audio_text="順便幫我設個提醒，19 點 30 分提醒我回來接他。",
          max_tokens=130, label="設提醒",
          agent_task_text="幫我設一個 19 點 30 分的提醒，內容是「接小明下鋼琴」。",
          expected_steps=1),
    Event(idx=17, t=150, kind="proactive",   priority=P_PROACTIVE,
          audio_text="已抵達全聯成功店，預計停留十分鐘，買完菜會提醒您回車上。",
          max_tokens=100, label="抵達超市"),
    Event(idx=18, t=160, kind="proactive",   priority=P_PROACTIVE,
          audio_text="老公發訊息問您今晚要不要煮義大利麵。",
          max_tokens=80,  label="App-老公訊息"),
    Event(idx=19, t=165, kind="interactive", priority=P_INTERACTIVE,
          audio_text="跟他說好，順便問他要不要加紅酒。",
          max_tokens=130, label="回老公",
          agent_task_text="幫我跟老公說，今晚煮義大利麵 OK，順便問他要不要加紅酒。",
          expected_steps=2),
    Event(idx=20, t=175, kind="proactive",   priority=P_PROACTIVE,
          audio_text="請慢慢買菜，我會在 19 點 25 分提醒您回來接小明。",
          max_tokens=80,  label="結束問候"),
]

# Sanity
assert len(COMMUTE_EVENTS) == 21
INTER_EVENTS = [e for e in COMMUTE_EVENTS if e.kind == "interactive"]
AGENT_EVENTS = [e for e in COMMUTE_EVENTS if e.kind == "agent"]
PROA_EVENTS  = [e for e in COMMUTE_EVENTS if e.kind == "proactive"]
# 7 interactive, 3 pure agent, 11 proactive. Interactive utterances also fire
# agent tasks (when agent_task_text != "") so total LLM requests ≈ 30.

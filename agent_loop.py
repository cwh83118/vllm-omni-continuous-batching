"""Multi-step agent tool-loop for the interactive cockpit stream.

A single user utterance ("導航到星光餐廳、把冷氣調到 22 度、播放音樂") becomes
3-6 sequential LLM requests:

  step 0 (with audio):
      messages = [system(catalog), user(audio + nudge)]
      -> model output should contain <tool_call>{"name":..,"args":..}</tool_call>
         or <done>...</done>

  step k>0 (text only, audio NOT re-sent):
      messages = [system, user(audio+nudge), assistant(model's prev output),
                  tool(mocked result JSON), <maybe more rounds>]
      -> next tool_call or <done>

Each step is its own OpenAI chat completion -> goes through the Dispatcher's
admission queue -> shows up as a separate row on the timeline. That temporal
interleaving with proactive ticks is what makes continuous batching's value
visible at small B.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

# ----- tool catalog (compressed from Omni3-demo backend/agentic_brain.py:60-133) ----

TOOL_CATALOG_TEXT = """\
可用工具（每一輪只能呼叫其中一個）：
- navigate_to_poi(poi_id): poi_id ∈ {"restaurant","shopping","beach","gas_station","cafe","home"}
- control_windows(window, action): window ∈ {"all","front_left","front_right","rear_left","rear_right"}, action ∈ {"open","close"}
- set_climate(temperature, fan_speed): temperature in °C (16-30), fan_speed ∈ {"low","mid","high","auto"}
- play_music(query): query 是字串，例如 "輕快歌單" 或 "抒情音樂"
- find_nearby(type, radius_km): type ∈ {"restaurant","gas_station","cafe","shopping","beach","parking"}, radius_km in 1..20
- order_food(restaurant, items): restaurant 是字串，items 是字串陣列

輸出規範：每一輪只能輸出以下兩種之一（單一行 JSON，不要解釋）：
  (A) 工具呼叫：<tool_call>{"name":"<工具名>","args":{...}}</tool_call>
  (B) 任務完成：<done>用一兩句中文回答使用者</done>
請依照使用者的多步驟任務，依序呼叫工具，每完成一步等待工具回傳結果再決定下一步。
"""

INTERACTIVE_SYSTEM_AGENT = (
    "你是車艙智慧助手的『交互代理』模組。使用者會用語音給出多步驟任務，"
    "你必須一步一步地透過呼叫工具完成它。\n"
    + TOOL_CATALOG_TEXT
)


# ----- parse helpers -----------------------------------------------------------

_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_DONE_RE = re.compile(r"<done>(.*?)</done>", re.DOTALL)


@dataclass
class StepResult:
    kind: str          # "tool_call" | "done" | "unparsed"
    tool_name: str = ""
    tool_args: dict | None = None
    done_text: str = ""
    raw_output: str = ""


def parse_step_output(text: str) -> StepResult:
    """Extract the first tool_call or done tag from the model's output."""
    m = _TOOL_RE.search(text)
    if m:
        try:
            payload = json.loads(m.group(1))
        except Exception:  # noqa: BLE001
            return StepResult(kind="unparsed", raw_output=text)
        name = payload.get("name", "")
        args = payload.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}
        return StepResult(kind="tool_call", tool_name=name, tool_args=args, raw_output=text)
    m = _DONE_RE.search(text)
    if m:
        return StepResult(kind="done", done_text=m.group(1).strip(), raw_output=text)
    return StepResult(kind="unparsed", raw_output=text)


# ----- deterministic mock tool executor ---------------------------------------

_POI_TABLE = {
    "restaurant":  {"name": "星光米其林餐廳", "eta_min": 7,  "distance_km": 2.3},
    "shopping":    {"name": "濱海購物中心",   "eta_min": 12, "distance_km": 4.2},
    "beach":       {"name": "藍灣海灘",       "eta_min": 20, "distance_km": 8.0},
    "gas_station": {"name": "中油忠孝站",     "eta_min": 4,  "distance_km": 1.4},
    "cafe":        {"name": "Cafe Reverie",   "eta_min": 5,  "distance_km": 1.7},
    "home":        {"name": "回家",            "eta_min": 18, "distance_km": 6.5},
}


async def mock_tool_executor(name: str, args: dict) -> dict:
    """Return a deterministic JSON result + a tiny artificial latency."""
    await asyncio.sleep(0.005)  # 5 ms — visible on the timeline but doesn't dominate
    if name == "navigate_to_poi":
        poi = _POI_TABLE.get(args.get("poi_id", "restaurant"), _POI_TABLE["restaurant"])
        return {"status": "navigating", "destination": poi["name"],
                "eta_min": poi["eta_min"], "distance_km": poi["distance_km"]}
    if name == "control_windows":
        return {"status": "ok", "window": args.get("window", "all"),
                "action": args.get("action", "close")}
    if name == "set_climate":
        return {"status": "ok", "temperature": args.get("temperature", 22),
                "fan_speed": args.get("fan_speed", "auto")}
    if name == "play_music":
        return {"status": "playing", "query": args.get("query", "輕音樂"), "track_count": 25}
    if name == "find_nearby":
        t = args.get("type", "restaurant")
        poi = _POI_TABLE.get(t, _POI_TABLE["restaurant"])
        return {"results": [{"id": t, "name": poi["name"], "distance_km": poi["distance_km"]}]}
    if name == "order_food":
        return {"status": "reserved", "restaurant": args.get("restaurant", "星光米其林餐廳"),
                "time": "19:00", "items": args.get("items", []) or ["主廚推薦"]}
    return {"status": "unknown_tool", "name": name}


# ----- run a complete agent task as a sequence of dispatched requests --------

MAX_AGENT_STEPS = 6
INITIAL_MAX_TOKENS = 180   # step 0 (model needs room to think + first tool_call)
FOLLOWUP_MAX_TOKENS = 96   # step >=1 (tool_call or short done)
INITIAL_TEMPERATURE = 0.6
FOLLOWUP_TEMPERATURE = 0.2


async def run_agent_task(
    *,
    submit_step,           # Callable[[req_factory], asyncio.Event-like await] -> awaits Req completion
    task_idx: int,         # which INTERACTIVE task this corresponds to (0..9)
    task_global_id: str,   # stable parent ID, e.g. "A3"
    audio_blocks_step0,    # callable returning list[dict] for step 0 content
    next_gidx,             # Callable[[], int]
    log,
):
    """Drive one user task to completion.

    `submit_step(make_req)` is provided by cabin_demo.py — it constructs a Req from
    `make_req(step_idx, content_blocks, parent_rid, max_tokens, temperature)`, hands it
    to the dispatcher, and returns an awaitable that resolves with the completed Req.
    """
    # Conversation history accumulated across steps. Step 0 carries audio; later
    # steps reuse the same audio messages (so the model can still re-read the
    # task) but append assistant+tool turns. The vLLM prefix cache should reuse
    # the audio prefill across steps when --enable-prefix-caching is on.
    audio_user_content = audio_blocks_step0()

    history: list[dict] = [
        {"role": "system",  "content": INTERACTIVE_SYSTEM_AGENT},
        {"role": "user",    "content": audio_user_content},
    ]

    for step in range(MAX_AGENT_STEPS):
        is_step0 = (step == 0)
        max_tokens = INITIAL_MAX_TOKENS if is_step0 else FOLLOWUP_MAX_TOKENS
        temp = INITIAL_TEMPERATURE if is_step0 else FOLLOWUP_TEMPERATURE

        # The Dispatcher's run_request takes a Req with either user_prompt (legacy)
        # or content_blocks (multimodal/agent). For tool-loop we pass the full
        # `messages` list verbatim via a new Req.full_messages field.
        req = await submit_step(
            step_idx=step,
            parent_rid=task_global_id,
            messages=list(history),         # snapshot
            max_tokens=max_tokens,
            temperature=temp,
            gidx=next_gidx(),
            audio_seconds=(0.0 if not is_step0 else None),   # let cabin_demo fill it
        )

        # `req` is the completed Req. Parse and decide next step.
        result = parse_step_output(req.content or "")
        if result.kind == "done":
            log(f"  [task {task_global_id}] step {step}: DONE -> {result.done_text[:60]!r}")
            return
        if result.kind == "unparsed":
            log(f"  [task {task_global_id}] step {step}: unparsed model output, "
                f"treating as terminal. raw[:80]={(req.content or '')[:80]!r}")
            return

        # tool_call -> mock execution, append assistant+tool messages, loop
        tool_result = await mock_tool_executor(result.tool_name, result.tool_args or {})
        log(f"  [task {task_global_id}] step {step}: tool_call {result.tool_name}"
            f"({result.tool_args}) -> {tool_result}")
        history.append({"role": "assistant", "content": req.content or ""})
        # Use a user-role wrapped tag instead of OpenAI's "tool" role: not all chat
        # templates (Qwen, etc.) ship a stable tool-role rendering, but every
        # template will quote plain user text into the prompt verbatim.
        tool_blob = json.dumps(tool_result, ensure_ascii=False)
        history.append({"role": "user",
                        "content": f"<tool_result name=\"{result.tool_name}\">{tool_blob}</tool_result>"})

    log(f"  [task {task_global_id}] hit MAX_AGENT_STEPS={MAX_AGENT_STEPS}, terminating.")

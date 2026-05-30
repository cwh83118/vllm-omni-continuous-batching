#!/usr/bin/env python3
"""In-car cabin-assistant demo — compare three request-scheduling regimes on ONE engine.

Two request streams hit the *same* vLLM-Omni (Qwen3-Omni-30B-A3B Thinker) engine:

  * "proactive" brain  — every PROACTIVE_INTERVAL s it looks at the cabin/exterior
    scene and emits a short decision / function call.  Periodic, predictable.
  * "interactive" brain — the user speaks at arbitrary moments and waits for a reply.
    Latency-sensitive.

We run the engine once with a generous ``max_num_seqs`` and emulate three scheduling
regimes purely on the client side (an *admission controller*), so the only variable is
the admission/refill policy — same model, same server, same workload (same ``--seed`` =>
identical arrival times, identical per-request sampling ``seed`` => identical outputs):

  --mode none        : no batching.  At most 1 request in flight; the next is admitted
                       only when the current one finishes.  (== a max_num_seqs=1 server.)
  --mode static  -B  : "fixed-batch" / NPU-style.  Admit a wave of up to B queued
                       requests; while the wave runs, hold ALL new arrivals; a finished
                       request streams out immediately, but its freed slot stays empty
                       until the whole wave drains; then admit the next wave of up to B.
  --mode continuous -B: continuous batching.  At most B in flight; the instant any request
                       finishes, immediately admit the next queued one (refill the slot).
                       This is what vLLM / vLLM-Omni does internally (cap = B here).

Per request we record: t_submit (arrived / entered the client queue), t_admitted (left
the queue, request actually started), t_first_token (=> TTFT = first_token - submit),
t_finish (EoS / result released), n_out_tokens, and wave_id (which static wave).

Usage:
  python cabin_demo.py --mode none       --batch-size 1 --out results/run_none.json
  python cabin_demo.py --mode static     --batch-size 8 --out results/run_static.json
  python cabin_demo.py --mode continuous --batch-size 8 --out results/run_continuous.json
  python cabin_demo.py --mode static --batch-size 8 --burst 24 --out results/burst_static.json
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field, asdict

from openai import AsyncOpenAI

# Multimodal / agent extensions (in-place — text-only baseline still works).
import assets_loader
import scenarios as _scenarios
from agent_loop import run_agent_task, INTERACTIVE_SYSTEM_AGENT

# ----------------------------------------------------------------------------- scenes

PROACTIVE_SYSTEM = (
    "你是車艙智慧助手的『主動偵測』模組，每隔幾秒收到一段艙內/艙外場景描述。請依序輸出四部分："
    "(1)【觀察】兩三句描述你看到的重點；"
    "(2)【動作】一行 JSON：{\"action\":\"<名稱>\",\"args\":{...}}（adjust_temperature / close_window / "
    "open_vent / play_music / suggest_rest / none 等）；"
    "(3)【理由】兩三句說明為什麼這樣判斷；"
    "(4)【後續】一句話提醒駕駛接下來要留意什麼。請完整寫完四部分。"
)
PROACTIVE_SCENES = [
    "艙內：駕駛連續打了三個噴嚏，溫度顯示 18°C，後排左側窗戶開了一條縫；艙外：天氣晴、車速 60km/h。",
    "艙內：後座兒童把腳放到前座椅背上，安全帶顯示未繫；艙外：前方 200m 有施工錐桶。",
    "艙內：駕駛打哈欠、眨眼變慢，已連續駕駛 2 小時；艙外：高速公路、夜間、車流順暢。",
    "艙內：副駕在講電話、音樂音量偏大；艙外：開始下小雨，雨刷未啟動。",
    "艙內：駕駛皺眉看著儀表、空調出風口對著臉吹冷風；艙外：氣溫 9°C、市區走走停停。",
    "艙內：後排乘客睡著、頭一直撞到車窗；艙外：彎道多的山路、車速適中。",
    "艙內：駕駛伸手去拿副駕座位上的水瓶，視線離開路面；艙外：直線道路、前車距離 30m。",
    "艙內：車內有淡淡焦味、空調循環為內循環；艙外：剛經過一段塞車路段、隧道口。",
]
PROACTIVE_PROMPT_PREFIX = "場景："

INTERACTIVE_SYSTEM = (
    "你是車艙智慧助手的『交互對話』模組。使用者會用語音問你問題，請用親切、口語的中文回答，"
    "先直接回答重點，再補充一點實用建議或下一步，大約四到六句。"
)
INTERACTIVE_QUERIES = [
    "幫我規劃一條順路、又能停下來吃飯的路線，現在大概還要開多久到台中？",
    "我有點累了，前面有沒有適合休息的服務區？順便提醒我做點什麼讓自己清醒一點。",
    "等一下到家之後幫我把家裡的冷氣先開好，然後播放放鬆一點的音樂可以嗎？",
    "幫我看一下今天的行程，我下午三點那個會議會不會遲到？要不要現在出發？",
    "車子好像有點怪聲音，從引擎那邊傳來的，這樣還能繼續開嗎？我該注意什麼？",
    "幫我用四川話跟我女兒講一句『放學記得在校門口等我』，我等下放給她聽。",
    "外面開始下雨了，幫我把該關的窗戶關一關，雨刷也開一下，謝謝。",
    "我想找一首適合長途開車聽的歌單，輕快一點不要太吵的，順便幫我放出來。",
    "副駕的人有點冷，可以幫他那邊調暖一點嗎？不要影響我這邊的溫度。",
    "前面那段一直在塞，有沒有別條路可以繞過去？大概可以省多少時間？",
]

# ----------------------------------------------------------------------------- request

@dataclass
class Req:
    rid: str
    brain: str            # "proactive" | "interactive" | "agent"
    idx: int
    gidx: int             # global arrival index (for deterministic per-request seed)
    sys_prompt: str
    user_prompt: object   # str (legacy) OR list[dict] (multimodal content blocks)
    max_tokens: int
    temperature: float
    # ---- multimodal / agent extensions (defaults preserve backward compat) ----
    brain_subtype: str = ""        # e.g. "proactive_audio", "agent_step0", "agent_step1"
    parent_rid: str = ""           # for agent follow-up steps, the parent task id
    step_idx: int = 0              # 0 = first request of the task; >0 = follow-up
    n_audio_sec: float = 0.0       # duration of audio attached to this request
    n_image: int = 0               # how many images attached
    full_messages: list = None     # if set, overrides sys_prompt+user_prompt above
    priority: int = 3              # 1 = interactive (top), 2 = agent, 3 = proactive
    event_idx: int = -1            # commute_run only: which COMMUTE_EVENTS entry triggered this
    # ---- timestamps & accounting ----
    t_submit: float = 0.0       # entered the client queue (= "arrived")
    t_admitted: float = 0.0     # left the queue, request actually started
    t_first_token: float = 0.0
    t_finish: float = 0.0
    n_out_tokens: int = 0
    wave_id: int = -1           # which static wave (static mode only)
    content: str = ""
    error: str = ""

    @property
    def ttft(self) -> float:
        return (self.t_first_token - self.t_submit) if self.t_first_token else float("nan")

    @property
    def queue_wait(self) -> float:
        return (self.t_admitted - self.t_submit) if self.t_admitted else float("nan")

    @property
    def e2e(self) -> float:
        return (self.t_finish - self.t_submit) if self.t_finish else float("nan")

    @property
    def decode_tps(self) -> float:
        dt = self.t_finish - self.t_first_token
        return (self.n_out_tokens - 1) / dt if (self.t_first_token and dt > 0 and self.n_out_tokens > 1) else float("nan")


async def run_request(client, model, req: Req, t0: float, seed: int, log):
    req.t_admitted = time.monotonic()
    wq = req.queue_wait * 1000
    label = f"{req.brain:<11s} #{req.idx:<2d}"
    if req.parent_rid:
        label += f" [parent={req.parent_rid} step={req.step_idx}]"
    log(f"[t={req.t_admitted - t0:6.2f}s] {label} admitted   "
        f"(waited {wq:6.0f} ms in client queue" + (f", wave #{req.wave_id}" if req.wave_id >= 0 else "") + ")")
    try:
        messages = req.full_messages or [
            {"role": "system", "content": req.sys_prompt},
            {"role": "user",   "content": req.user_prompt},
        ]
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            seed=seed,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    if not req.t_first_token:
                        req.t_first_token = time.monotonic()
                        log(f"[t={req.t_first_token - t0:6.2f}s] {label} first token "
                            f"(TTFT {req.ttft*1000:6.0f} ms = {req.queue_wait*1000:.0f} queue + "
                            f"{(req.t_first_token-req.t_admitted)*1000:.0f} prefill)")
                    req.content += delta.content
            if getattr(chunk, "usage", None):
                req.n_out_tokens = chunk.usage.completion_tokens
        req.t_finish = time.monotonic()
        if not req.n_out_tokens:
            req.n_out_tokens = max(1, len(req.content) // 2)
        log(f"[t={req.t_finish - t0:6.2f}s] {label} DONE  "
            f"({req.n_out_tokens:3d} tok, e2e {req.e2e:5.2f}s, decode {req.decode_tps:5.1f} tok/s)  "
            f"-> {req.content[:42].replace(chr(10),' ')!r}")
    except Exception as e:  # noqa: BLE001
        req.t_finish = time.monotonic()
        req.error = repr(e)
        log(f"[t={req.t_finish - t0:6.2f}s] {label} ERROR {e!r}")


# --------------------------------------------------------------------- admission controller

class Dispatcher:
    """Client-side admission controller emulating five scheduling regimes:

      none           — B=1 strict serial. Priority irrelevant.
      static         — wave drain (≤B per wave); within a wave priority-sorted.
      static_vip     — wave drain + interactive jumps to its OWN wave, runs alone (B=1).
      continuous     — vanilla FIFO refill; ≤B in flight at all times.
      continuous_pri — refill but always pull highest-priority pending first.

    All modes additionally enforce per-stream concurrency caps:
        interactive ≤ 1, agent ≤ 3, proactive ≤ 2  (total still ≤ B).
    """

    PRIORITY = {"interactive": 1, "agent": 2, "proactive": 3}
    PER_STREAM_CAP = {"interactive": 1, "agent": 3, "proactive": 2}

    def __init__(self, client, args, t0, log):
        self.client = client
        self.args = args
        self.t0 = t0
        self.log = log
        self.mode = args.mode                       # none | static | static_vip | continuous | continuous_pri
        self.B = 1 if self.mode == "none" else max(1, args.batch_size)
        self.wave_mode = self.mode in ("none", "static", "static_vip")
        self.priority_aware = self.mode in ("static_vip", "continuous_pri")
        self.pending: list[Req] = []
        self.in_flight: set[asyncio.Task] = set()
        self.in_flight_by_brain: dict[str, int] = {"interactive": 0, "agent": 0, "proactive": 0}
        self.vip_active: bool = False               # static_vip: current wave is a VIP wave
        self.reqs: list[Req] = []
        self.kick = asyncio.Event()
        self.arrivals_done = False
        self.wave_counter = 0
        self.done_events: dict[str, asyncio.Event] = {}

    def submit(self, req: Req):
        """Called by the arrival generators when a request 'arrives'."""
        req.t_submit = time.monotonic()
        self.reqs.append(req)
        self.pending.append(req)
        if req.rid not in self.done_events:
            self.done_events[req.rid] = asyncio.Event()
        label = f"{req.brain:<11s} #{req.idx:<2d}"
        if req.parent_rid:
            label += f" [parent={req.parent_rid} step={req.step_idx}]"
        self.log(f"[t={req.t_submit - self.t0:6.2f}s] {label} arrived    "
                 f"(pending {len(self.pending)}, in-flight {len(self.in_flight)})")
        self.kick.set()

    async def submit_and_await(self, req: Req) -> Req:
        """Submit a request and wait for its completion (used by agent_loop)."""
        self.submit(req)
        await self.done_events[req.rid].wait()
        return req

    def mark_arrivals_done(self):
        self.arrivals_done = True
        self.kick.set()

    # -- internal --
    def _start(self, req: Req, wave_id: int):
        req.wave_id = wave_id
        seed = self.args.seed * 100003 + req.gidx
        task = asyncio.create_task(run_request(self.client, self.args.model, req, self.t0, seed, self.log))
        task.add_done_callback(lambda t, r=req: self._on_done(t, r))
        self.in_flight.add(task)
        self.in_flight_by_brain[req.brain] = self.in_flight_by_brain.get(req.brain, 0) + 1

    def _on_done(self, task, req: Req):
        self.in_flight.discard(task)
        self.in_flight_by_brain[req.brain] = max(0, self.in_flight_by_brain.get(req.brain, 0) - 1)
        if self.vip_active and not self.in_flight:
            self.vip_active = False
        ev = self.done_events.get(req.rid)
        if ev is not None:
            ev.set()
        self.kick.set()

    def _sort_pending_by_priority(self) -> list[Req]:
        """Return a stable priority-sorted copy of pending (no mutation)."""
        return sorted(self.pending,
                      key=lambda r: (self.PRIORITY.get(r.brain, 9), r.t_submit))

    def _stream_cap_ok(self, req: Req) -> bool:
        cap = self.PER_STREAM_CAP.get(req.brain)
        return cap is None or self.in_flight_by_brain.get(req.brain, 0) < cap

    def _maybe_admit(self):
        # --- wave-style modes ---
        if self.wave_mode:
            if self.in_flight or not self.pending:
                return

            # static_vip: if any interactive in pending, run it ALONE in a VIP wave.
            if self.mode == "static_vip":
                vip = next((r for r in self.pending if r.brain == "interactive"), None)
                if vip is not None:
                    self.pending.remove(vip)
                    wid = self.wave_counter; self.wave_counter += 1
                    self.vip_active = True
                    self.log(f"[t={time.monotonic() - self.t0:6.2f}s] "
                             f"--- VIP wave #{wid}: interactive {vip.rid} alone (full GPU) ---")
                    self._start(vip, wave_id=wid)
                    return

            # regular wave: priority-sorted pick (interactive first if any), respect per-stream caps
            order = self._sort_pending_by_priority() if self.priority_aware else list(self.pending)
            picked: list[Req] = []
            picked_brains = {"interactive": 0, "agent": 0, "proactive": 0}
            for r in order:
                if len(picked) >= self.B:
                    break
                cap = self.PER_STREAM_CAP.get(r.brain, self.B)
                if picked_brains[r.brain] >= cap:
                    continue
                picked.append(r)
                picked_brains[r.brain] += 1
            if not picked:
                return
            wid = self.wave_counter; self.wave_counter += 1
            had = len(self.pending)
            if self.mode in ("static", "static_vip"):
                self.log(f"[t={time.monotonic() - self.t0:6.2f}s] --- wave #{wid}: admitting {len(picked)} req "
                         f"(pending was {had}, by brain {picked_brains}) ---")
            for r in picked:
                self.pending.remove(r)
                self._start(r, wave_id=wid)
            return

        # --- continuous modes ---
        # iterate pending in priority order (or FIFO for vanilla continuous)
        order = self._sort_pending_by_priority() if self.priority_aware else list(self.pending)
        for r in order:
            if len(self.in_flight) >= self.B:
                break
            if not self._stream_cap_ok(r):
                continue
            self.pending.remove(r)
            self._start(r, wave_id=-1)

    async def run(self):
        while True:
            self._maybe_admit()
            if self.arrivals_done and not self.pending and not self.in_flight:
                return
            self.kick.clear()
            try:
                await asyncio.wait_for(self.kick.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass


# ------------------------------------------------------------------------- arrival generators

PROACTIVE_SYSTEM_AUDIO = (
    "你是車艙智慧助手的『主動偵測』模組。每次收到：一張艙內外合成影像、一段艙內現場語音描述、"
    "一段車輛狀態 JSON。請依序輸出："
    "(1)【觀察】兩三句描述你看到/聽到的重點；"
    "(2)【動作】一行 JSON：{\"action\":\"<名稱>\",\"args\":{...}}（adjust_temperature / close_window / "
    "open_vent / play_music / suggest_rest / none 等）；"
    "(3)【理由】兩三句說明為什麼這樣判斷；"
    "(4)【後續】一句話提醒駕駛接下來要留意什麼。請完整寫完四部分。"
)


async def proactive_arrivals(disp: Dispatcher, args, t0):
    """Periodic proactive ticks. Multimodal when --use-audio (default for scenarios)."""
    i = 0
    await asyncio.sleep(0.5)
    while time.monotonic() - t0 < args.duration:
        if args.use_audio:
            scene_idx = i % 8                                      # 8 WAV/JPG/vehicle rows
            blocks = assets_loader.proactive_content_blocks(scene_idx)
            audio_sec = assets_loader.audio_duration_for_proactive(scene_idx)
            disp.submit(Req(rid=f"P{i}", brain="proactive", idx=i, gidx=_next_gidx(),
                            sys_prompt=PROACTIVE_SYSTEM_AUDIO, user_prompt=blocks,
                            brain_subtype="proactive_audio",
                            n_audio_sec=audio_sec, n_image=1,
                            max_tokens=args.proactive_max_tokens, temperature=0.3))
        else:
            scene = PROACTIVE_SCENES[i % len(PROACTIVE_SCENES)]
            disp.submit(Req(rid=f"P{i}", brain="proactive", idx=i, gidx=_next_gidx(),
                            sys_prompt=PROACTIVE_SYSTEM, user_prompt=PROACTIVE_PROMPT_PREFIX + scene,
                            max_tokens=args.proactive_max_tokens, temperature=0.3))
        i += 1
        await asyncio.sleep(args.proactive_interval)


async def interactive_arrivals(disp: Dispatcher, args, t0):
    rng = random.Random(args.seed)
    times = []
    t = 1.5
    while t < args.duration - 1.0 and len(times) < args.n_interactive:
        times.append(t)
        t += rng.expovariate(args.interactive_rate)
    i = 0
    for at in times:
        now = time.monotonic() - t0
        if at > now:
            await asyncio.sleep(at - now)
        q = INTERACTIVE_QUERIES[i % len(INTERACTIVE_QUERIES)]
        disp.submit(Req(rid=f"I{i}", brain="interactive", idx=i, gidx=_next_gidx(),
                        sys_prompt=INTERACTIVE_SYSTEM, user_prompt=q,
                        max_tokens=args.interactive_max_tokens, temperature=0.6))
        i += 1


async def agent_arrivals(disp: Dispatcher, args, t0, agent_launches):
    """Launch each agent task at its scheduled start_time_s.

    Each launched task is a tool-loop driven by agent_loop.run_agent_task, which
    submits N follow-up Reqs through disp.submit_and_await. Spawning is concurrent
    (an asyncio.Task per agent), so multiple tasks can be in different steps at
    the same wall-clock — exactly what continuous batching needs to demonstrate
    its advantage.
    """
    coros = []

    async def launch(start_s: float, task_idx: int):
        now = time.monotonic() - t0
        if start_s > now:
            await asyncio.sleep(start_s - now)
        task_global_id = f"A{task_idx}"
        i_ref = [_next_agent_step_counter()]   # mutable, captured by submit_step

        async def submit_step(*, step_idx, parent_rid, messages, max_tokens,
                              temperature, gidx, audio_seconds):
            rid = f"{task_global_id}_s{step_idx}"
            # audio_seconds=None means "step 0, look it up"; else "no audio re-sent"
            audio_sec = (assets_loader.audio_duration_for_interactive(task_idx)
                         if audio_seconds is None else float(audio_seconds))
            n_img = 0
            req = Req(rid=rid, brain="agent", idx=i_ref[0],
                      gidx=gidx,
                      sys_prompt="", user_prompt="",   # ignored because full_messages set
                      brain_subtype=f"agent_step{step_idx}",
                      parent_rid=parent_rid, step_idx=step_idx,
                      n_audio_sec=audio_sec, n_image=n_img,
                      full_messages=messages,
                      max_tokens=max_tokens, temperature=temperature)
            i_ref[0] = _next_agent_step_counter()
            return await disp.submit_and_await(req)

        def step0_blocks():
            return assets_loader.interactive_content_blocks_step0(task_idx)

        await run_agent_task(
            submit_step=submit_step,
            task_idx=task_idx,
            task_global_id=task_global_id,
            audio_blocks_step0=step0_blocks,
            next_gidx=_next_gidx,
            log=disp.log,
        )

    for launch_spec in agent_launches:
        coros.append(launch(launch_spec.start_time_s, launch_spec.task_idx))
    if coros:
        await asyncio.gather(*coros)


# Counter for the per-task idx field; survives across all agent tasks so each
# Req gets a unique idx within the brain="agent" stream.
_AGENT_STEP_COUNTER = [0]
def _next_agent_step_counter() -> int:
    v = _AGENT_STEP_COUNTER[0]
    _AGENT_STEP_COUNTER[0] += 1
    return v


# === commute_run: the 180s mom-pickup scenario =============================

async def commute_arrivals(disp: Dispatcher, args, t0):
    """Drive arrivals from commute_script.COMMUTE_EVENTS.

    Each event is scheduled at its event.t. Behavior by kind:
      - proactive:    submit one multimodal Req at time t.
      - interactive:  if event has agent_task_text, treat utterance as the kick-off
                      of an agent task → submit step-0 (with audio) then continue
                      tool-loop. Otherwise, single-turn audio Q&A.
      - agent (pure): no audio, kick off an agent tool-loop driven by agent_task_text.
    """
    import commute_script
    from agent_loop import run_agent_task

    proa_counter = [0]; inter_counter = [0]; agent_counter = [0]

    async def fire_event(ev):
        nonlocal proa_counter, inter_counter, agent_counter
        # sleep to event time
        target = ev.t
        now = time.monotonic() - t0
        if target > now:
            await asyncio.sleep(target - now)

        if ev.kind == "proactive":
            i = proa_counter[0]; proa_counter[0] += 1
            blocks = assets_loader.commute_content_blocks("proactive", ev.idx, with_image=True)
            audio_sec = assets_loader.load_commute_audio("proactive", ev.idx).duration_s
            disp.submit(Req(rid=f"P{i}", brain="proactive", idx=i, gidx=_next_gidx(),
                            sys_prompt=PROACTIVE_SYSTEM_AUDIO, user_prompt=blocks,
                            brain_subtype="proactive_commute",
                            n_audio_sec=audio_sec, n_image=1,
                            event_idx=ev.idx, priority=ev.priority,
                            max_tokens=ev.max_tokens, temperature=0.3))
            return

        if ev.kind in ("interactive", "agent"):
            # Both kick off an agent tool-loop. interactive has audio; agent has none.
            task_global_id = f"{'I' if ev.kind == 'interactive' else 'A'}{(inter_counter[0] if ev.kind=='interactive' else agent_counter[0])}"
            if ev.kind == "interactive":
                inter_counter[0] += 1
            else:
                agent_counter[0] += 1
            audio_blocks_step0 = (
                (lambda e=ev: assets_loader.commute_content_blocks("interactive", e.idx, with_image=False))
                if ev.kind == "interactive"
                else (lambda e=ev: [{"type": "text",
                                     "text": "（系統內部主動任務）" + e.agent_task_text}])
            )

            async def submit_step(*, step_idx, parent_rid, messages, max_tokens,
                                  temperature, gidx, audio_seconds):
                rid = f"{task_global_id}_s{step_idx}"
                if ev.kind == "interactive" and step_idx == 0:
                    audio_sec = assets_loader.load_commute_audio("interactive", ev.idx).duration_s
                else:
                    audio_sec = 0.0
                # Brain tagging: keep step 0 of "interactive" tagged interactive
                # so per-stream cap (interactive ≤ 1) protects user latency.
                # Tag follow-up steps as "agent" so they share the agent pool.
                if ev.kind == "interactive" and step_idx == 0:
                    brain_tag = "interactive"
                    pri = 1
                else:
                    brain_tag = "agent"
                    pri = 2
                req = Req(rid=rid, brain=brain_tag, idx=step_idx,
                          gidx=gidx,
                          sys_prompt="", user_prompt="",
                          brain_subtype=f"{ev.kind}_step{step_idx}",
                          parent_rid=task_global_id, step_idx=step_idx,
                          n_audio_sec=audio_sec, n_image=0,
                          full_messages=messages,
                          event_idx=ev.idx, priority=pri,
                          max_tokens=max_tokens, temperature=temperature)
                return await disp.submit_and_await(req)

            await run_agent_task(
                submit_step=submit_step,
                task_idx=ev.idx,
                task_global_id=task_global_id,
                audio_blocks_step0=audio_blocks_step0,
                next_gidx=_next_gidx,
                log=disp.log,
            )

    coros = [fire_event(ev) for ev in commute_script.COMMUTE_EVENTS]
    await asyncio.gather(*coros)


async def burst_arrivals(disp: Dispatcher, args, t0):
    await asyncio.sleep(1.0)
    disp.log(f"[t={time.monotonic()-t0:6.2f}s] --- BURST: {args.burst} requests arrive at once ---")
    for i in range(args.burst):
        q = INTERACTIVE_QUERIES[i % len(INTERACTIVE_QUERIES)]
        disp.submit(Req(rid=f"B{i}", brain="interactive", idx=i, gidx=_next_gidx(),
                        sys_prompt=INTERACTIVE_SYSTEM, user_prompt=q,
                        max_tokens=args.interactive_max_tokens, temperature=0.6))


# tiny global gidx counter so the per-request sampling seed is stable across modes
_GIDX = [0]
def _next_gidx() -> int:
    v = _GIDX[0]
    _GIDX[0] += 1
    return v


# ------------------------------------------------------------------------------ summary

def pct(xs, p):
    xs = sorted(v for v in xs if v == v)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(reqs, wall, args):
    inter = [r for r in reqs if r.brain == "interactive" and not r.error]
    pro = [r for r in reqs if r.brain == "proactive" and not r.error]
    agent_all = [r for r in reqs if r.brain == "agent" and not r.error]
    agent_step0 = [r for r in agent_all if r.step_idx == 0]
    agent_followup = [r for r in agent_all if r.step_idx > 0]
    ok = [r for r in reqs if not r.error and r.t_finish]
    total_out = sum(r.n_out_tokens for r in reqs if not r.error)
    busy = (max(r.t_finish for r in ok) - min(r.t_submit for r in ok)) if ok else float("nan")
    n_waves = (max((r.wave_id for r in reqs if r.wave_id >= 0), default=-1) + 1)
    n_agent_tasks = len({r.parent_rid for r in agent_all if r.parent_rid})
    audio_secs = [r.n_audio_sec for r in reqs if r.n_audio_sec > 0]
    return {
        "mode": args.mode, "batch_size": (1 if args.mode == "none" else args.batch_size),
        "config": args.config, "server_max_num_seqs": args.max_num_seqs,
        "scenario": getattr(args, "scenario", None),
        "wall_clock_s": round(wall, 3),
        "busy_span_s": round(busy, 3),
        "busy_output_tok_per_s": round(total_out / busy, 1) if busy and busy > 0 else 0.0,
        "n_requests_total": len(reqs), "n_errors": sum(1 for r in reqs if r.error),
        "n_interactive": len(inter), "n_proactive": len(pro),
        "n_agent_requests": len(agent_all), "n_agent_tasks": n_agent_tasks,
        "n_agent_step0": len(agent_step0), "n_agent_followup": len(agent_followup),
        "n_waves": n_waves,
        "interactive_ttft_p50_s": round(pct([r.ttft for r in inter], 50), 3),
        "interactive_ttft_p95_s": round(pct([r.ttft for r in inter], 95), 3),
        "interactive_ttft_max_s": round(max([r.ttft for r in inter], default=float("nan")), 3),
        "interactive_e2e_p50_s": round(pct([r.e2e for r in inter], 50), 3),
        "interactive_e2e_p95_s": round(pct([r.e2e for r in inter], 95), 3),
        "interactive_queue_wait_p50_s": round(pct([r.queue_wait for r in inter], 50), 3),
        "proactive_ttft_p50_s": round(pct([r.ttft for r in pro], 50), 3),
        "proactive_e2e_p50_s": round(pct([r.e2e for r in pro], 50), 3),
        "proactive_e2e_max_s": round(max([r.e2e for r in pro], default=float("nan")), 3),
        "agent_ttft_p50_s": round(pct([r.ttft for r in agent_all], 50), 3),
        "agent_ttft_p95_s": round(pct([r.ttft for r in agent_all], 95), 3),
        "agent_step0_ttft_p50_s": round(pct([r.ttft for r in agent_step0], 50), 3),
        "agent_followup_ttft_p50_s": round(pct([r.ttft for r in agent_followup], 50), 3),
        "agent_e2e_p50_s": round(pct([r.e2e for r in agent_all], 50), 3),
        "mean_audio_seconds": round(statistics.fmean(audio_secs), 2) if audio_secs else 0.0,
        "total_output_tokens": total_out,
        "mean_decode_tok_per_s": round(statistics.fmean(
            [r.decode_tps for r in reqs if not r.error and r.decode_tps == r.decode_tps]), 1)
            if any(r.decode_tps == r.decode_tps for r in reqs if not r.error) else float("nan"),
    }


def print_table(s):
    print("\n" + "=" * 80)
    print(f" SUMMARY  mode={s['mode']!r}  batch_size={s['batch_size']}  "
          f"server_max_num_seqs={s['server_max_num_seqs']}  "
          f"scenario={s.get('scenario')!r}")
    print("=" * 80)
    rows = [
        ("counts (interactive / proactive / agent reqs · tasks)",
         f"{s['n_interactive']} / {s['n_proactive']} / "
         f"{s.get('n_agent_requests', 0)} · {s.get('n_agent_tasks', 0)}  "
         f"(errors {s['n_errors']}, waves {s['n_waves']})"),
        ("interactive TTFT  p50 / p95 / max  (s)", f"{s['interactive_ttft_p50_s']} / {s['interactive_ttft_p95_s']} / {s['interactive_ttft_max_s']}"),
        ("interactive  e2e  p50 / p95        (s)", f"{s['interactive_e2e_p50_s']} / {s['interactive_e2e_p95_s']}"),
        ("interactive queue-wait p50          (s)", f"{s['interactive_queue_wait_p50_s']}"),
        ("proactive  TTFT p50 / e2e p50 / e2e max (s)", f"{s['proactive_ttft_p50_s']} / {s['proactive_e2e_p50_s']} / {s['proactive_e2e_max_s']}"),
        ("agent     TTFT p50 / p95 / e2e p50 (s)", f"{s.get('agent_ttft_p50_s')} / {s.get('agent_ttft_p95_s')} / {s.get('agent_e2e_p50_s')}"),
        ("agent step0 TTFT p50 / followup TTFT p50 (s)", f"{s.get('agent_step0_ttft_p50_s')} / {s.get('agent_followup_ttft_p50_s')}"),
        ("mean audio seconds per request", s.get("mean_audio_seconds", 0.0)),
        ("total output tokens", s["total_output_tokens"]),
        ("busy span (first submit -> last finish) (s)", s["busy_span_s"]),
        ("output throughput over busy span (tok/s)", s["busy_output_tok_per_s"]),
        ("mean per-request decode speed (tok/s)", s["mean_decode_tok_per_s"]),
    ]
    for k, v in rows:
        print(f"  {k:<46s}: {v}")
    print("=" * 80 + "\n")


# --------------------------------------------------------------------------------- main

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["none", "static", "static_vip", "continuous", "continuous_pri"],
                    default="continuous",
                    help="scheduling regime to emulate")
    ap.add_argument("--batch-size", type=int, default=8, help="batch cap B for static/continuous (none forces 1)")
    ap.add_argument("--config", default=None, help="label for this run (default = mode)")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="the server's max_num_seqs (for the record); must be >= --batch-size")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--duration", type=float, default=24.0, help="seconds to keep accepting new arrivals")
    ap.add_argument("--proactive-interval", type=float, default=2.5)
    ap.add_argument("--proactive-max-tokens", type=int, default=220)
    ap.add_argument("--interactive-rate", type=float, default=1.6, help="interactive Poisson arrival rate (req/s)")
    ap.add_argument("--interactive-max-tokens", type=int, default=180)
    ap.add_argument("--n-interactive", type=int, default=30, help="cap on number of interactive requests")
    ap.add_argument("--burst", type=int, default=0,
                    help="if >0: saturated mode -- this many interactive requests arrive at once; no proactive")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    # --- multimodal / agent extensions ---
    ap.add_argument("--scenario", default=None,
                    choices=list(_scenarios.SCENARIOS.keys()),
                    help="named scenario (overrides --burst / --duration / agent schedule). "
                         "Pick one of: pure_proactive / pure_agent / mixed_1agent / mixed_3agent / burst24.")
    ap.add_argument("--use-audio", dest="use_audio", action="store_true", default=None,
                    help="Send audio+image+vehicle JSON for proactive ticks (default: on when --scenario "
                         "is set, off otherwise).")
    ap.add_argument("--no-use-audio", dest="use_audio", action="store_false",
                    help="Force text-only proactive prompts (backward compat with the original benchmark).")
    args = ap.parse_args()
    if args.config is None:
        args.config = args.mode
    if args.out is None:
        args.out = f"results/run_{args.config}.json"
    if args.scenario:
        spec = _scenarios.get(args.scenario)
        args.duration = spec.duration_s
        args.proactive_interval = spec.proactive_interval_s
        args.proactive_max_tokens = spec.proactive_max_tokens
        if spec.burst_n > 0:
            args.burst = spec.burst_n
            args.interactive_max_tokens = spec.burst_max_tokens
        # default-on audio when running a scenario
        if args.use_audio is None:
            args.use_audio = True
    else:
        spec = None
        if args.use_audio is None:
            args.use_audio = False

    client = AsyncOpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="EMPTY", timeout=180.0)

    B = 1 if args.mode == "none" else args.batch_size
    print(f"# cabin_demo  mode={args.mode}  batch_size={B}  server_max_num_seqs={args.max_num_seqs}  "
          f"server=http://{args.host}:{args.port}  model={args.model}")
    if args.burst > 0:
        print(f"# burst: {args.burst} interactive requests arrive at once (<= {args.interactive_max_tokens} tok)")
    else:
        print(f"# proactive every {args.proactive_interval}s (<= {args.proactive_max_tokens} tok), "
              f"interactive Poisson {args.interactive_rate}/s cap {args.n_interactive} (<= {args.interactive_max_tokens} tok), "
              f"arrival window {args.duration}s")
    print("-" * 80)

    log_lines = []
    def log(line):
        print(line, flush=True)
        log_lines.append(line)

    _GIDX[0] = 0
    _AGENT_STEP_COUNTER[0] = 0
    t0 = time.monotonic()
    disp = Dispatcher(client, args, t0, log)

    async def arrivals():
        if spec is not None:
            # Scenario-driven arrivals (multimodal + agent tool-loop)
            sub = []
            if spec.use_commute_script:
                sub.append(commute_arrivals(disp, args, t0))
            elif spec.burst_n > 0:
                sub.append(burst_arrivals(disp, args, t0))
            else:
                if spec.proactive_enabled:
                    sub.append(proactive_arrivals(disp, args, t0))
                if spec.agent_launches:
                    sub.append(agent_arrivals(disp, args, t0, spec.agent_launches))
            if sub:
                await asyncio.gather(*sub)
        elif args.burst > 0:
            await burst_arrivals(disp, args, t0)
        else:
            await asyncio.gather(proactive_arrivals(disp, args, t0),
                                 interactive_arrivals(disp, args, t0))
        disp.mark_arrivals_done()

    await asyncio.gather(disp.run(), arrivals())
    wall = time.monotonic() - t0

    out_reqs = []
    for r in disp.reqs:
        d = asdict(r)
        for k in ("sys_prompt", "user_prompt", "full_messages"):
            d.pop(k, None)
        for k in ("t_submit", "t_admitted", "t_first_token", "t_finish"):
            d[k] = round(d[k] - t0, 4) if d[k] else 0.0
        d["ttft_s"] = round(r.ttft, 4) if r.ttft == r.ttft else None
        d["queue_wait_s"] = round(r.queue_wait, 4) if r.queue_wait == r.queue_wait else None
        d["e2e_s"] = round(r.e2e, 4) if r.e2e == r.e2e else None
        d["decode_tps"] = round(r.decode_tps, 2) if r.decode_tps == r.decode_tps else None
        out_reqs.append(d)
    s = summarize(disp.reqs, wall, args)
    print_table(s)

    payload = {"mode": args.mode, "batch_size": B, "config": args.config,
               "max_num_seqs": args.max_num_seqs, "host": args.host, "port": args.port,
               "model": args.model, "args": vars(args), "wall_clock_s": round(wall, 3),
               "summary": s, "requests": out_reqs, "log": log_lines}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

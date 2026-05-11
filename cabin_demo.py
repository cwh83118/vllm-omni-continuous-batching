#!/usr/bin/env python3
"""In-car cabin-assistant continuous-batching demo.

Two request streams hit the *same* vLLM-Omni (Qwen3-Omni-30B-A3B Thinker) engine:

  * "proactive" brain  — every PROACTIVE_INTERVAL s it looks at the cabin/exterior
    scene and emits a short decision / function call.  Periodic, predictable.
  * "interactive" brain — the user speaks at arbitrary moments and waits for a reply.
    Latency-sensitive.

Run the engine once with a given thinker `max_num_seqs`, then run this script:

  * max_num_seqs = 1   -> "no continuous batching": the engine runs one sequence at a
    time, FCFS.  An interactive request that arrives while a proactive inference is in
    flight has to wait for it to finish.  Everyone waits for everyone.
  * max_num_seqs >= 2  -> continuous batching: a request arriving mid-generation joins
    the running batch on the very next decode step; a request that hits EoS is released
    immediately, regardless of what else is still generating.

For every request we record: t_submit, t_first_token (TTFT), t_finish (EoS / result
released to the user), number of output tokens.  Results are dumped to JSON and a
comparison table is printed.

Usage:
  python cabin_demo.py --config off  --port 8901   # against a max_num_seqs=1 server
  python cabin_demo.py --config on   --port 8901   # against a max_num_seqs=8 server
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field, asdict

from openai import AsyncOpenAI

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

# ----------------------------------------------------------------------------- model

@dataclass
class Req:
    rid: str
    brain: str            # "proactive" | "interactive"
    idx: int
    t_submit: float = 0.0
    t_first_token: float = 0.0
    t_finish: float = 0.0
    n_out_tokens: int = 0
    content: str = ""
    error: str = ""

    @property
    def ttft(self) -> float:
        return (self.t_first_token - self.t_submit) if self.t_first_token else float("nan")

    @property
    def e2e(self) -> float:
        return (self.t_finish - self.t_submit) if self.t_finish else float("nan")

    @property
    def decode_tps(self) -> float:
        dt = self.t_finish - self.t_first_token
        return (self.n_out_tokens - 1) / dt if (self.t_first_token and dt > 0 and self.n_out_tokens > 1) else float("nan")


async def run_request(client, model, sys_prompt, user_prompt, max_tokens, temperature,
                      req: Req, t0: float, log):
    req.t_submit = time.monotonic()
    log(f"[t={req.t_submit - t0:6.2f}s] {req.brain:<11s} #{req.idx:<2d} submitted")
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    if not req.t_first_token:
                        req.t_first_token = time.monotonic()
                        log(f"[t={req.t_first_token - t0:6.2f}s] {req.brain:<11s} #{req.idx:<2d} first token "
                            f"(waited {req.ttft*1000:6.0f} ms after submit)")
                    req.content += delta.content
            if getattr(chunk, "usage", None):
                req.n_out_tokens = chunk.usage.completion_tokens
        req.t_finish = time.monotonic()
        if not req.n_out_tokens:
            req.n_out_tokens = max(1, len(req.content) // 2)
        log(f"[t={req.t_finish - t0:6.2f}s] {req.brain:<11s} #{req.idx:<2d} DONE  "
            f"({req.n_out_tokens:3d} tok, e2e {req.e2e:5.2f}s, decode {req.decode_tps:5.1f} tok/s)  "
            f"-> {req.content[:46].replace(chr(10),' ')!r}")
    except Exception as e:  # noqa: BLE001
        req.t_finish = time.monotonic()
        req.error = repr(e)
        log(f"[t={req.t_finish - t0:6.2f}s] {req.brain:<11s} #{req.idx:<2d} ERROR {e!r}")


async def burst_loop(client, args, reqs, t0, log):
    """Saturated scenario: fire args.burst interactive requests (almost) simultaneously
    at t~=1.0s. Shows the throughput / weight-bandwidth amortisation story: with
    continuous batching all of them decode together (one weight read per step shared
    by the whole batch); without it they run one after another."""
    await asyncio.sleep(1.0)
    log(f"[t={time.monotonic()-t0:6.2f}s] --- BURST: submitting {args.burst} requests at once ---")
    tasks = []
    for i in range(args.burst):
        q = INTERACTIVE_QUERIES[i % len(INTERACTIVE_QUERIES)]
        r = Req(rid=f"B{i}", brain="interactive", idx=i)
        reqs.append(r)
        tasks.append(asyncio.create_task(run_request(
            client, args.model, INTERACTIVE_SYSTEM, q,
            args.interactive_max_tokens, 0.6, r, t0, log)))
    await asyncio.gather(*tasks)


async def proactive_loop(client, args, reqs, t0, log, stop_at):
    i = 0
    # first proactive fires shortly after t0 so an interactive can land mid-inference
    await asyncio.sleep(0.5)
    tasks = []
    while time.monotonic() - t0 < stop_at:
        scene = PROACTIVE_SCENES[i % len(PROACTIVE_SCENES)]
        r = Req(rid=f"P{i}", brain="proactive", idx=i)
        reqs.append(r)
        tasks.append(asyncio.create_task(run_request(
            client, args.model, PROACTIVE_SYSTEM, PROACTIVE_PROMPT_PREFIX + scene,
            args.proactive_max_tokens, 0.3, r, t0, log)))
        i += 1
        await asyncio.sleep(args.proactive_interval)
    if tasks:
        await asyncio.gather(*tasks)


async def interactive_loop(client, args, reqs, t0, log, stop_at):
    rng = random.Random(args.seed)
    # Poisson arrivals at rate args.interactive_rate (req/s), first arrival ~1.5s in,
    # capped at args.n_interactive requests.
    times = []
    t = 1.5
    while t < stop_at - 1.0 and len(times) < args.n_interactive:
        times.append(t)
        t += rng.expovariate(args.interactive_rate)
    tasks = []
    i = 0
    for at in times:
        now = time.monotonic() - t0
        if at > now:
            await asyncio.sleep(at - now)
        q = INTERACTIVE_QUERIES[i % len(INTERACTIVE_QUERIES)]
        r = Req(rid=f"I{i}", brain="interactive", idx=i)
        reqs.append(r)
        tasks.append(asyncio.create_task(run_request(
            client, args.model, INTERACTIVE_SYSTEM, q,
            args.interactive_max_tokens, 0.6, r, t0, log)))
        i += 1
    if tasks:
        await asyncio.gather(*tasks)


def pct(xs, p):
    xs = sorted(v for v in xs if v == v)  # drop NaN
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(reqs, wall, config, max_num_seqs):
    inter = [r for r in reqs if r.brain == "interactive" and not r.error]
    pro = [r for r in reqs if r.brain == "proactive" and not r.error]
    ok = [r for r in reqs if not r.error and r.t_finish]
    total_out = sum(r.n_out_tokens for r in reqs if not r.error)
    busy = (max(r.t_finish for r in ok) - min(r.t_submit for r in ok)) if ok else float("nan")
    s = {
        "config": config, "max_num_seqs": max_num_seqs,
        "wall_clock_s": round(wall, 3),
        "busy_span_s": round(busy, 3),
        "busy_output_tok_per_s": round(total_out / busy, 1) if busy and busy > 0 else 0.0,
        "n_requests_total": len(reqs), "n_errors": sum(1 for r in reqs if r.error),
        "n_interactive": len(inter), "n_proactive": len(pro),
        "interactive_ttft_p50_s": round(pct([r.ttft for r in inter], 50), 3),
        "interactive_ttft_p95_s": round(pct([r.ttft for r in inter], 95), 3),
        "interactive_ttft_max_s": round(max([r.ttft for r in inter], default=float("nan")), 3),
        "interactive_e2e_p50_s": round(pct([r.e2e for r in inter], 50), 3),
        "interactive_e2e_p95_s": round(pct([r.e2e for r in inter], 95), 3),
        "proactive_ttft_p50_s": round(pct([r.ttft for r in pro], 50), 3),
        "proactive_e2e_p50_s": round(pct([r.e2e for r in pro], 50), 3),
        "proactive_e2e_max_s": round(max([r.e2e for r in pro], default=float("nan")), 3),
        "total_output_tokens": total_out,
        "aggregate_output_tok_per_s": round(total_out / wall, 1) if wall > 0 else 0.0,
        "mean_decode_tok_per_s": round(statistics.fmean(
            [r.decode_tps for r in reqs if not r.error and r.decode_tps == r.decode_tps]), 1)
            if any(r.decode_tps == r.decode_tps for r in reqs if not r.error) else float("nan"),
    }
    return s


def print_table(s):
    print("\n" + "=" * 78)
    print(f" SUMMARY  config={s['config']!r}  thinker max_num_seqs={s['max_num_seqs']}")
    print("=" * 78)
    rows = [
        ("wall clock (s)", s["wall_clock_s"]),
        ("requests done (interactive / proactive)", f"{s['n_interactive']} / {s['n_proactive']}  (errors {s['n_errors']})"),
        ("interactive TTFT  p50 / p95 / max  (s)", f"{s['interactive_ttft_p50_s']} / {s['interactive_ttft_p95_s']} / {s['interactive_ttft_max_s']}"),
        ("interactive e2e   p50 / p95        (s)", f"{s['interactive_e2e_p50_s']} / {s['interactive_e2e_p95_s']}"),
        ("proactive   TTFT  p50  / e2e p50 / e2e max (s)", f"{s['proactive_ttft_p50_s']} / {s['proactive_e2e_p50_s']} / {s['proactive_e2e_max_s']}"),
        ("total output tokens", s["total_output_tokens"]),
        ("busy span (first submit -> last finish) (s)", s["busy_span_s"]),
        ("aggregate output throughput over busy span (tok/s)", s["busy_output_tok_per_s"]),
        ("mean per-request decode speed (tok/s)", s["mean_decode_tok_per_s"]),
    ]
    for k, v in rows:
        print(f"  {k:<46s}: {v}")
    print("=" * 78 + "\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="on", help="label for this run (e.g. on / off)")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="thinker max_num_seqs the server was launched with (for the record / label)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to keep submitting new requests")
    ap.add_argument("--proactive-interval", type=float, default=2.5)
    ap.add_argument("--proactive-max-tokens", type=int, default=220)
    ap.add_argument("--interactive-rate", type=float, default=1.2, help="interactive Poisson arrival rate (req/s)")
    ap.add_argument("--interactive-max-tokens", type=int, default=180)
    ap.add_argument("--n-interactive", type=int, default=30, help="cap on number of interactive requests")
    ap.add_argument("--burst", type=int, default=0,
                    help="if >0: saturated mode -- fire this many interactive requests at once, no proactive loop")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/run_{args.config}.json"

    client = AsyncOpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="EMPTY", timeout=120.0)

    print(f"# cabin_demo  config={args.config}  max_num_seqs={args.max_num_seqs}  "
          f"server=http://{args.host}:{args.port}  model={args.model}")
    print(f"# proactive every {args.proactive_interval}s (<= {args.proactive_max_tokens} tok), "
          f"interactive Poisson {args.interactive_rate}/s cap {args.n_interactive} (<= {args.interactive_max_tokens} tok), "
          f"submit window {args.duration}s")
    print("-" * 78)

    log_lines = []
    def log(line):
        print(line, flush=True)
        log_lines.append(line)

    reqs: list[Req] = []
    t0 = time.monotonic()
    if args.burst > 0:
        await burst_loop(client, args, reqs, t0, log)
    else:
        await asyncio.gather(
            proactive_loop(client, args, reqs, t0, log, args.duration),
            interactive_loop(client, args, reqs, t0, log, args.duration),
        )
    wall = time.monotonic() - t0

    # serialize, relative to t0
    out_reqs = []
    for r in reqs:
        d = asdict(r)
        for k in ("t_submit", "t_first_token", "t_finish"):
            d[k] = round(d[k] - t0, 4) if d[k] else 0.0
        d["ttft_s"] = round(r.ttft, 4) if r.ttft == r.ttft else None
        d["e2e_s"] = round(r.e2e, 4) if r.e2e == r.e2e else None
        d["decode_tps"] = round(r.decode_tps, 2) if r.decode_tps == r.decode_tps else None
        out_reqs.append(d)
    s = summarize(reqs, wall, args.config, args.max_num_seqs)
    print_table(s)

    payload = {"config": args.config, "max_num_seqs": args.max_num_seqs,
               "host": args.host, "port": args.port, "model": args.model,
               "args": vars(args), "wall_clock_s": round(wall, 3),
               "summary": s, "requests": out_reqs, "log": log_lines}
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

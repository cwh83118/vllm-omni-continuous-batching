"""Cache base64-encoded WAV/JPG assets so each request's content blocks reuse them.

We encode each asset once at startup (record_assets.py already produced them)
and hand the base64 string back to cabin_demo.py's request builders. Encoding
inside the asyncio request hot path would add ~1-3 ms per request and would
fight the timeline measurements.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


@lru_cache(maxsize=None)
def _read_wav_b64(path: str) -> tuple[str, float]:
    """Return (base64_data, duration_sec) for a 16 kHz mono PCM_16 WAV file."""
    with open(path, "rb") as f:
        data = f.read()
    info = sf.info(path)
    return base64.b64encode(data).decode("ascii"), float(info.duration)


@lru_cache(maxsize=None)
def _read_jpg_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@dataclass(frozen=True)
class AudioClip:
    path: str
    b64: str
    duration_s: float


def load_proactive_audio(idx: int) -> AudioClip:
    p = str(ASSETS / "audio" / "proactive" / f"scene_{idx:02d}.wav")
    b64, dur = _read_wav_b64(p)
    return AudioClip(path=p, b64=b64, duration_s=dur)


def load_interactive_audio(idx: int) -> AudioClip:
    p = str(ASSETS / "audio" / "interactive" / f"task_{idx:02d}.wav")
    b64, dur = _read_wav_b64(p)
    return AudioClip(path=p, b64=b64, duration_s=dur)


def load_commute_audio(kind: str, event_idx: int) -> AudioClip:
    """Load a WAV for the commute_run scenario.

    kind ∈ {"interactive", "proactive"}; event_idx is the COMMUTE_EVENTS index.
    """
    prefix = {"interactive": "inter", "proactive": "proa"}[kind]
    p = str(ASSETS / "audio" / "commute" / f"{prefix}_{event_idx:02d}.wav")
    b64, dur = _read_wav_b64(p)
    return AudioClip(path=p, b64=b64, duration_s=dur)


def commute_content_blocks(kind: str, event_idx: int, with_image: bool = True) -> list[dict]:
    """Content blocks for a commute_run event.

    Proactive events: audio + optional image + brief context text.
    Interactive events: audio + brief nudge text (no image — user's voice only).
    """
    audio = load_commute_audio(kind, event_idx)
    blocks = []
    if kind == "proactive" and with_image:
        # rotate through the 8 combined images for variety
        img = load_combined_image(event_idx % 8)
        blocks.append({"type": "image_url",
                       "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    blocks.append({"type": "input_audio",
                   "input_audio": {"data": audio.b64, "format": "wav"}})
    if kind == "interactive":
        blocks.append({"type": "text",
                       "text": "請依語音指示一步步使用工具完成任務。"})
    else:
        blocks.append({"type": "text",
                       "text": "這是即將要播報給駕駛的主動提醒，請簡短重述要點並輸出對應 JSON action。"})
    return blocks


def load_combined_image(idx: int) -> str:
    p = str(ASSETS / "images" / f"combined_{idx:04d}.jpg")
    return _read_jpg_b64(p)


@lru_cache(maxsize=1)
def load_vehicle_statuses() -> list[dict]:
    p = ASSETS / "vehicle_status.json"
    with p.open() as f:
        return json.load(f)


def vehicle_status_for(idx: int) -> dict:
    rows = load_vehicle_statuses()
    return rows[idx % len(rows)]


# ----- content-block builders --------------------------------------------------

def proactive_content_blocks(scene_idx: int) -> list[dict]:
    """OpenAI chat-completions content blocks for one proactive tick.

    Layout matches the design A.3: image + audio + text (vehicle status JSON).
    """
    img = load_combined_image(scene_idx)
    audio = load_proactive_audio(scene_idx)
    veh = vehicle_status_for(scene_idx)
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        {"type": "input_audio", "input_audio": {"data": audio.b64, "format": "wav"}},
        {"type": "text",
         "text": "車輛狀態 JSON：" + json.dumps(veh, ensure_ascii=False)
                 + "\n請依系統指示輸出觀察/動作 JSON/理由/後續。"},
    ]


def interactive_content_blocks_step0(task_idx: int) -> list[dict]:
    """First turn of an agent task: voice (audio) + a short text nudge."""
    audio = load_interactive_audio(task_idx)
    return [
        {"type": "input_audio", "input_audio": {"data": audio.b64, "format": "wav"}},
        {"type": "text", "text": "請完成上述語音任務，依規範使用工具。"},
    ]


def audio_duration_for_proactive(scene_idx: int) -> float:
    return load_proactive_audio(scene_idx).duration_s


def audio_duration_for_interactive(task_idx: int) -> float:
    return load_interactive_audio(task_idx).duration_s

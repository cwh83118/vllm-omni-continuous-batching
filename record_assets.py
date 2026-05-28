#!/usr/bin/env python3
"""Generate cockpit benchmark assets in one shot.

Outputs (under ./assets/):
  audio/proactive/scene_0{0..7}.wav   ~5-7 s each, ~車內描述
  audio/interactive/task_0{0..9}.wav  ~10-14 s each, ~多步驟代理任務
  images/combined_000{0..7}.jpg       copied from Omni3-demo proactive_images/
  vehicle_status.json                 8 entries, 1:1 with proactive scenes

edge-tts produces MP3 by default; we convert to 16 kHz mono PCM16 WAV via
librosa+soundfile (matches Qwen-Omni's Whisper-style feature extractor expectations).

Run:
  python record_assets.py
"""
import asyncio
import json
import shutil
from pathlib import Path

import edge_tts
import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
PROACTIVE_DIR = ASSETS / "audio" / "proactive"
INTERACTIVE_DIR = ASSETS / "audio" / "interactive"
IMAGES_DIR = ASSETS / "images"

VOICE = "zh-TW-HsiaoChenNeural"  # Mandarin TW female
RATE = "+10%"  # slightly faster so clips stay short

PROACTIVE_AUDIO_SCRIPTS = [
    "駕駛剛剛連續打了三個噴嚏，車內溫度顯示十八度，後排左側窗戶開了一條縫；外面天氣晴朗、車速六十公里。",
    "後座小朋友把腳放到前座椅背上，安全帶沒繫好；前方兩百公尺有施工錐桶，要注意。",
    "駕駛打了幾個哈欠，眨眼變慢，已經連續開了兩個小時；現在是夜間高速公路、車流順暢。",
    "副駕駛正在講電話，音樂音量偏大；外面開始下小雨，雨刷還沒啟動。",
    "駕駛皺著眉看儀表，空調出風口直接吹到臉上，外面溫度只有九度，正在市區走走停停。",
    "後排乘客睡著了，頭一直撞到車窗；目前在彎道很多的山路上，車速適中。",
    "駕駛伸手去拿副駕座位上的水瓶，視線離開了路面；前方是直線道路，前車距離三十公尺。",
    "車內有一點淡淡的焦味，空調是內循環模式；剛剛經過一段塞車路段，現在到了隧道口。",
]

# 10 multi-step agent task scripts (matches design A.4)
INTERACTIVE_AUDIO_SCRIPTS = [
    "先導航到附近最近的星光米其林餐廳，到了之後幫我訂兩個人的位子，順便把冷氣調到二十二度。",
    "我要去海邊，先放放鬆的音樂，然後等到了海邊把所有車窗都打開。",
    "現在帶我去最近的購物中心，到了再開一首逛街用的歌單，記得把車內溫度降到二十度。",
    "我有點累，先靠邊找一家咖啡店休息半小時，然後再上路去藍灣海灘，路上把窗戶關起來。",
    "幫我規劃今天下午的行程，先去星光餐廳吃中餐、再去海邊散步、最後到購物中心，每段都幫我設定好導航。",
    "天氣變熱了，幫我把冷氣調到十九度同時把駕駛側窗戶關起來，然後播放輕快點的音樂。",
    "導航到最近的加油站加油，加完之後直接帶我回購物中心繼續逛。",
    "先把後排兩個窗戶都打開讓孩子吹吹風，然後找一家評價最好的餐廳，訂今晚七點的位子。",
    "等下先導航到星光餐廳吃飯，吃完直接帶我去海邊看夕陽，整路放抒情音樂。",
    "現在 OK 麻煩你，把空調風量調到中等、把全部窗戶關上、播放電子音樂、然後導航到購物中心。",
]

VEHICLE_STATUSES = [
    {"scene_idx": 0, "speed_kmh": 60, "cabin_temp_c": 18, "ac_fan": "high", "ac_target_c": 24, "window_rear_left": "ajar",  "fuel_pct": 65, "destination": "台中"},
    {"scene_idx": 1, "speed_kmh": 80, "cabin_temp_c": 23, "ac_fan": "auto", "ac_target_c": 23, "window_rear_left": "closed","fuel_pct": 72, "destination": "新竹"},
    {"scene_idx": 2, "speed_kmh":110, "cabin_temp_c": 22, "ac_fan": "low",  "ac_target_c": 22, "window_rear_left": "closed","fuel_pct": 48, "destination": "高雄"},
    {"scene_idx": 3, "speed_kmh": 50, "cabin_temp_c": 24, "ac_fan": "auto", "ac_target_c": 24, "window_rear_left": "closed","fuel_pct": 60, "destination": "市區"},
    {"scene_idx": 4, "speed_kmh": 30, "cabin_temp_c": 19, "ac_fan": "high", "ac_target_c": 22, "window_rear_left": "closed","fuel_pct": 78, "destination": "公司"},
    {"scene_idx": 5, "speed_kmh": 55, "cabin_temp_c": 23, "ac_fan": "auto", "ac_target_c": 23, "window_rear_left": "closed","fuel_pct": 55, "destination": "陽明山"},
    {"scene_idx": 6, "speed_kmh": 90, "cabin_temp_c": 22, "ac_fan": "low",  "ac_target_c": 22, "window_rear_left": "closed","fuel_pct": 68, "destination": "桃園"},
    {"scene_idx": 7, "speed_kmh": 40, "cabin_temp_c": 27, "ac_fan": "auto", "ac_target_c": 23, "window_rear_left": "closed","fuel_pct": 32, "destination": "回家"},
]


def mp3_to_wav_16k_mono(mp3: Path, wav: Path) -> None:
    """Re-encode MP3 to 16 kHz mono PCM16 WAV (Qwen-Omni audio encoder format)."""
    y, _sr = librosa.load(str(mp3), sr=16000, mono=True)
    sf.write(str(wav), y, 16000, subtype="PCM_16")


async def synth_one(text: str, out_wav: Path) -> None:
    """Synthesize one clip with edge-tts and convert to 16k mono WAV."""
    mp3 = out_wav.with_suffix(".mp3")
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    await communicate.save(str(mp3))
    mp3_to_wav_16k_mono(mp3, out_wav)
    mp3.unlink(missing_ok=True)


async def synth_batch(scripts: list[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(scripts):
        wav = out_dir / f"{prefix}_{i:02d}.wav"
        if wav.exists():
            print(f"  skip {wav.name} (exists)")
            continue
        print(f"  synth {wav.name} <- {text[:30]}...")
        await synth_one(text, wav)


def copy_images() -> None:
    src_dir = Path("/home/davidchang/Documents/Omni3-demo/proactive_images")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        src = src_dir / f"combined_{i:04d}.jpg"
        dst = IMAGES_DIR / f"combined_{i:04d}.jpg"
        if dst.exists():
            print(f"  skip {dst.name} (exists)")
            continue
        if not src.exists():
            print(f"  MISSING source: {src}")
            continue
        shutil.copy(src, dst)
        print(f"  copied {dst.name}")


def write_vehicle_status() -> None:
    out = ASSETS / "vehicle_status.json"
    with out.open("w") as f:
        json.dump(VEHICLE_STATUSES, f, ensure_ascii=False, indent=2)
    print(f"  wrote {out}")


async def main() -> None:
    assert len(PROACTIVE_AUDIO_SCRIPTS) == 8
    assert len(INTERACTIVE_AUDIO_SCRIPTS) == 10
    assert len(VEHICLE_STATUSES) == 8

    print("[1/4] proactive WAVs ...")
    await synth_batch(PROACTIVE_AUDIO_SCRIPTS, PROACTIVE_DIR, "scene")

    print("[2/4] interactive WAVs ...")
    await synth_batch(INTERACTIVE_AUDIO_SCRIPTS, INTERACTIVE_DIR, "task")

    print("[3/4] images ...")
    copy_images()

    print("[4/4] vehicle status ...")
    write_vehicle_status()

    print("done.")


if __name__ == "__main__":
    asyncio.run(main())

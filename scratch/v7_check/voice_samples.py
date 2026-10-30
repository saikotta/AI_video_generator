r"""
Generate the same narration line in several Microsoft Edge voices so you can listen and choose.
Needs internet (it calls the same edge-tts service the pipeline uses).

Run from C:\AiVideoGenerator:
    .\.venv\Scripts\python.exe voice_samples.py
Then open media\voice_samples, listen, and pick one:
    $env:TTS_VOICE = "en-US-AndrewNeural"      (for that PowerShell window)
or set TTS_VOICE=... in your .env to keep it.
"""
import asyncio
import os

import edge_tts

LINE = ("Green beans enter a rotating drum heated to two hundred degrees. "
        "They crackle, turn brown, release their oils, and cool. Then the roasted beans pass through burrs.")
SHORTLIST = [
    "en-US-AndrewNeural", "en-US-BrianNeural", "en-US-GuyNeural", "en-US-ChristopherNeural",
    "en-US-AriaNeural", "en-US-JennyNeural", "en-US-AvaNeural", "en-US-EmmaNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-AU-WilliamNeural",
]


async def main() -> None:
    current = os.getenv("TTS_VOICE", "(default from your .env / config)")
    voices = await edge_tts.list_voices()
    available = {v["ShortName"] for v in voices}
    chosen = [v for v in SHORTLIST if v in available]
    for v in voices:  # top up with other English voices so there is always something to compare
        if len(chosen) >= 12:
            break
        if v["Locale"] in ("en-US", "en-GB", "en-AU", "en-CA") and v["ShortName"] not in chosen:
            chosen.append(v["ShortName"])
    out_dir = os.path.join("media", "voice_samples")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Current TTS_VOICE: {current}\n")
    for name in chosen:
        path = os.path.join(out_dir, f"{name}.mp3")
        try:
            await edge_tts.Communicate(LINE, name).save(path)
            print(f"  saved  {name}")
        except Exception as e:
            print(f"  FAILED {name}: {e}")
    print(f"\nOpen the folder: {os.path.abspath(out_dir)}")
    print('Pick one, then run:  $env:TTS_VOICE = "NAME"   (or add TTS_VOICE=NAME to .env)')


if __name__ == "__main__":
    asyncio.run(main())

"""Tests for the audio-pause trimming and the gentle push-in.   Run: python test_phase4_polish.py"""
import os, sys, types, subprocess, tempfile, wave
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
import numpy as np
from PIL import Image
import imageio_ffmpeg

from app.services import audio_trim
from app.services.audio_trim import trim_edge_silence, trim_narration_edges, measure_duration_seconds, pause_seconds_per_scene
from app.services.video_assembly import apply_push_in, shot_zoom_limit, get_single_visual_frame

FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 24000
fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

def speechlike(seconds, seed=0):
    """Amplitude-modulated noise bursts that behave like spoken words."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (rng.standard_normal(len(t)) * 0.12 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).clip(-0.9, 0.9)

def make_clip(path, lead, speech_parts, tail):
    """lead/tail seconds of near-silence around speech parts separated by short inner pauses; saved as mp3."""
    quiet = lambda s: np.random.default_rng(9).standard_normal(int(s * SR)) * 0.0003   # about -70 dB
    pieces = [quiet(lead)]
    for i, (dur, pause) in enumerate(speech_parts):
        pieces.append(speechlike(dur, seed=i)); pieces.append(quiet(pause))
    pieces.append(quiet(tail))
    sig = np.concatenate(pieces)
    wav = path + ".wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((sig * 32767).astype(np.int16).tobytes())
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", wav, "-c:a", "libmp3lame", "-q:a", "2", path], check=True)
    os.remove(wav)
    return sig

def pcm(path):
    r = subprocess.run([FF, "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float64) / 32768

rms_db = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)

with tempfile.TemporaryDirectory() as d:
    # ---- trimming removes edge silence, keeps speech and inner pauses ----
    p = os.path.join(d, "clip.mp3")
    parts = [(0.9, 0.30), (1.1, 0.25), (0.8, 0.0)]            # speech, inner pauses (kept)
    make_clip(p, lead=0.6, speech_parts=parts, tail=0.7)
    before_d, before = measure_duration_seconds(p), pcm(p)
    changed = trim_edge_silence(p)
    after_d, after = measure_duration_seconds(p), pcm(p)
    check("trim reports the file was changed", changed)
    check("edge silence removed: ~1.05s shorter (0.6-0.1 lead, 0.7-0.15 tail)", 0.85 <= before_d - after_d <= 1.25)
    check("a short natural margin is kept at both ends (>= 0.05s each)", after_d >= (0.9 + 0.3 + 1.1 + 0.25 + 0.8) + 0.2)
    sp_b = before[int(0.6 * SR):int((0.6 + 3.35) * SR)]
    sp_a = after[int(0.10 * SR):int((0.10 + 3.35) * SR)]
    check("speech loudness unchanged (within 0.5 dB)", abs(rms_db(sp_b) - rms_db(sp_a)) < 0.5)
    check("speech peaks unchanged (>= 95%)", np.abs(after).max() >= 0.95 * np.abs(before).max())
    check("inner pauses are not removed (total length = speech + pauses + margins)",
          abs(after_d - (3.35 + 0.10 + 0.15)) < 0.15)

    # ---- the pipeline hook: returns the trimmed duration, falls back safely ----
    p2 = os.path.join(d, "clip2.mp3"); make_clip(p2, 0.6, [(1.5, 0.0)], 0.7)
    orig = measure_duration_seconds(p2)
    new = trim_narration_edges(p2, orig)
    check("trim_narration_edges returns the new, shorter measured duration", new < orig - 0.8 and abs(new - measure_duration_seconds(p2)) < 0.02)
    os.environ["TTS_TRIM_EDGE_SILENCE"] = "0"
    p3 = os.path.join(d, "clip3.mp3"); make_clip(p3, 0.6, [(1.5, 0.0)], 0.7)
    o3 = measure_duration_seconds(p3)
    check("TTS_TRIM_EDGE_SILENCE=0 leaves audio and duration untouched",
          trim_narration_edges(p3, o3) == o3 and abs(measure_duration_seconds(p3) - o3) < 0.01)
    check("pause per scene is 1.1s when trimming is off", pause_seconds_per_scene() == 1.1)
    os.environ.pop("TTS_TRIM_EDGE_SILENCE", None)
    check("pause per scene is 0.3s when trimming is on", pause_seconds_per_scene() == 0.3)
    check("missing file returns the original duration", trim_narration_edges(os.path.join(d, "nope.mp3"), 3.21) == 3.21)
    # all-silent file must never be 'trimmed away'
    p4 = os.path.join(d, "silent.mp3"); make_clip(p4, 2.0, [], 0.0)
    s_before = measure_duration_seconds(p4)
    trim_narration_edges(p4, s_before)
    check("an all-silent clip is not destroyed", measure_duration_seconds(p4) >= 0.6 * s_before)

# ---- push-in ----
rng = np.random.default_rng(3)
img = Image.fromarray(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8))
a0 = apply_push_in(img, 0.0, 1.08)
check("push-in at progress 0 is identical to the source", np.abs(np.asarray(a0, int) - np.asarray(img, int)).max() <= 1)
a1 = apply_push_in(img, 1.0, 1.08)
check("push-in keeps the output size", a1.size == img.size)
check("push-in at progress 1 actually changes the frame", np.abs(np.asarray(a1, int) - np.asarray(img, int)).mean() > 5)
check("max_zoom <= 1 returns the image untouched", apply_push_in(img, 1.0, 1.0) is img)

class FakeClip:
    def __init__(self, moving, duration=4.0):
        self.duration, self.moving = duration, moving
    def get_frame(self, t):
        r = np.random.default_rng(int(t * 1000) if self.moving else 1)
        return r.integers(0, 255, (72, 128, 3), dtype=np.uint8) if self.moving else np.full((72, 128, 3), 120, dtype=np.uint8)
check("near-static clip gets the stronger zoom (1.08)", shot_zoom_limit({"video_clip": FakeClip(False)}) == 1.08)
check("clip that already moves gets the gentler zoom (1.035)", shot_zoom_limit({"video_clip": FakeClip(True)}) == 1.035)

def frame_of(t, moving=False):
    vd = {"is_multi_shot": True, "fps": 24.0, "total_frames": 96,
          "shots": [{"start_frame": 0, "end_frame": 96, "is_video": True, "video_clip": FakeClip(moving, 6.0)}]}
    return get_single_visual_frame(vd, t, 4.0, (640, 360), 24.0)
f_start, f_end = frame_of(0.0), frame_of(3.9)
check("multi-shot frames keep the target size with push-in on", f_start.size == (640, 360) and f_end.size == (640, 360))
# a static grey clip stays identical frame to frame without zoom (uniform colour), so use a gradient clip
class GradClip(FakeClip):
    def get_frame(self, t):
        g = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (72, 1)); return np.stack([g, g, g], axis=2)
vd = {"is_multi_shot": True, "fps": 24.0, "total_frames": 96,
      "shots": [{"start_frame": 0, "end_frame": 96, "is_video": True, "video_clip": GradClip(False, 6.0)}]}
g_start = np.asarray(get_single_visual_frame(vd, 0.0, 4.0, (640, 360), 24.0), int)
g_end = np.asarray(get_single_visual_frame(vd, 3.9, 4.0, (640, 360), 24.0), int)
check("a static shot now changes over its length (it is no longer a slide)", np.abs(g_end - g_start).mean() > 1.0)
os.environ["VIDEO_PUSH_IN"] = "0"
import importlib, app.services.video_assembly as va
importlib.reload(va)
g2_start = np.asarray(va.get_single_visual_frame(vd, 0.0, 4.0, (640, 360), 24.0), int)
g2_end = np.asarray(va.get_single_visual_frame(vd, 3.9, 4.0, (640, 360), 24.0), int)
check("VIDEO_PUSH_IN=0 restores the old behaviour (static shot stays identical)", np.abs(g2_end - g2_start).mean() < 0.5)

print("\n%d failed" % len(fails)); sys.exit(1 if fails else 0)

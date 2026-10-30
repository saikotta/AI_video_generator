"""Tests for the phase-3 fixes: safe validators, abstract-scene title cards, placeholder setting,
near-black rejection and short-video word budget.   Run: python test_phase3_fixes.py"""
import os, sys, types, subprocess, tempfile
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
try:
    import av  # noqa: F401
except Exception:
    sys.modules["av"] = types.ModuleType("av")
import numpy as np
from PIL import Image
from pydantic import ValidationError

from app.schemas.visual_spec import (ShotSpecificationSchema, VisualSpecificationSchema,
                                     find_scientific_contradictions, PLACEHOLDER_SETTING)
from app.services.visual_verification import HeuristicVisualVerifier as H, VerificationStatus as S
from app.services.pipeline_orchestrator import extract_scene_subject

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

# ---- validators: correct text must pass, wrong text must still be caught ----
ok_cases = [
    ("seedling", "Roots anchor in soil while the shoot grows upward"),
    ("plant", "Root hairs absorb water as the stem pushes upward"),
    ("beetroot", "Beetroot leaves growing upward in a garden bed"),
    ("vegetable stall", "Camera tilts upward over a stall of root vegetables"),
    ("root system", "The root system grows downward into dark soil"),
    ("plant", "Water travels upward from the roots to the leaves"),
    ("root tip", "Root tip pushes downward into dark soil"),
    ("dry seed", "A dry seed swells as water enters its coat, before it becomes a seedling"),
]
for subj, act in ok_cases:
    check(f"validator accepts: {act}", find_scientific_contradictions(subj, act) == [])

bad_cases = [
    ("root tip", "Root tip elongates upward from soil"),
    ("emerging root tip", "A root tip emerges and elongates upward toward the sky"),
    ("root radicle", "Macro close-up of root radicle growing upward towards the surface"),
    ("stem", "The stem pushes downward into the soil"),
    ("dry seed", "A dry seed that is already a green leafy seedling"),
    ("enzymes", "Enzymes shown by pouring into a glass of water"),
]
for subj, act in bad_cases:
    check(f"validator still catches: {act}", find_scientific_contradictions(subj, act) != [])

def raises(fn):
    try: fn()
    except ValidationError: return True
    return False
check("ShotSpecificationSchema rejects root growing upward",
      raises(lambda: ShotSpecificationSchema(shot_index=1, primary_subject="emerging root tip",
             required_visible_action="A root tip emerges and elongates upward toward the sky",
             required_setting="dark soil", search_keywords=["root growing", "germination macro"],
             provisional_start_seconds=0.0, provisional_end_seconds=3.0)))
check("ShotSpecificationSchema accepts a correct shoot+root sentence",
      not raises(lambda: ShotSpecificationSchema(shot_index=1, primary_subject="seedling",
             required_visible_action="Roots anchor in soil while the shoot grows upward",
             required_setting="garden", search_keywords=["seedling growing", "plant growth macro"],
             provisional_start_seconds=0.0, provisional_end_seconds=3.0)))
check("VisualSpecificationSchema accepts a correct shoot+root scene prompt",
      not raises(lambda: VisualSpecificationSchema(primary_subject="seedling",
             visual_prompt="Macro shot of roots anchoring in soil while the shoot grows upward",
             search_keywords=["seedling roots soil", "plant growth macro"])))

# ---- subject extraction on the real prompts from the seed-to-plant run ----
topic = "How a seed grows into a plant"
real = [
 ("Extreme macro close-up of a dry seed as a single water droplet contacts its surface, the droplet spreads.", ["macro seed water"], "dry seed"),
 ("Extreme macro close-up of a plant root tip emerging from dark soil, fibers spreading.", ["macro root tip"], "root tip"),
 ("Macro close-up of a green plant shoot extending upward, then a fresh leaf unfurling from the stem.", ["macro plant shoot"], "plant shoot"),
]
for prompt, kws, want in real:
    got = extract_scene_subject({"visual_prompt": prompt, "search_keywords": kws}, fallback_topic=topic)
    check(f"subject from real prompt -> '{want}' (got '{got}')", want in got and got != topic)

# ---- placeholder setting + near-black ----
rng = np.random.default_rng(0)
def frames(lo, hi):
    out = []
    for t in (0.5, 1.0, 1.5):
        a = rng.integers(lo, hi, (64, 64, 3), dtype=np.uint8); out.append((t, Image.fromarray(a), a))
    return out
req = {"primary_subject": "root tip", "required_visible_action": "root tip emerging", "required_setting": PLACEHOLDER_SETTING,
       "required_objects": [], "explicit_exclusions": []}
res = H.evaluate(frames(60, 200), req, {"clip_text": "root tip appropriate setting"})
check("placeholder setting is never counted as evidence", "Setting terms matched" not in " ".join(res.confirmed_matches))
check("near-black footage is REJECTED even when text matches",
      H.evaluate(frames(0, 12), req, {"clip_text": "macro root tip emerging from soil"}).status == S.REJECTED)

# ---- title card renders a real, decodable clip of the right length ----
from app.services.info_card import render_info_card
with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, "card.mp4")
    render_info_card("Enzymes activate, breaking stored food into sugars.", out, 5.0)
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate,duration",
                         "-of", "default=noprint_wrappers=1", out], capture_output=True, text=True)
    info = pr.stdout
    check("title card is a 1280x720 h264 24fps clip", all(k in info for k in ("h264", "width=1280", "height=720", "24/1")))
    dur = float([l for l in info.splitlines() if l.startswith("duration=")][0].split("=")[1])
    check("title card duration is ~5s", 4.8 <= dur <= 5.3)

# ---- short-video word budget ----
from app.services.providers.groq_script_provider import GroqScriptProvider as G
g = G.__new__(G); g.words_per_minute = 130
check("15s video -> ~27 words (was 33)", g.calculate_target_words(15) == 27)
check("5-minute video still uses the configured 130 wpm", g.calculate_target_words(300) == 650)

print("\n%d failed" % len(fails)); sys.exit(1 if fails else 0)

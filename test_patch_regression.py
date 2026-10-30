"""Regression tests for the verification / classifier / prompt / path fixes.
Run:  python test_patch_regression.py     (no pytest needed)
"""
import os, sys, types
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
try:
    import av  # noqa: F401
except Exception:                      # verifier tests don't decode video
    sys.modules["av"] = types.ModuleType("av")

import numpy as np
from PIL import Image
from app.services.visual_verification import HeuristicVisualVerifier as H, VerificationStatus as S

def moving_frames():
    rng = np.random.default_rng(0)
    out = []
    for t in (0.5, 1.0, 1.5):
        arr = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
        out.append((t, Image.fromarray(arr), arr))
    return out

REQ = {"primary_subject": "wheat grains", "required_visible_action": "pouring grain",
       "required_setting": "barn", "required_objects": ["hands"], "explicit_exclusions": []}
F = moving_frames()
fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

# --- verifier: the circular-evidence bug ---
check("search query alone can no longer ACCEPT a clip",
      H.evaluate(F, REQ, {"keyword_used": "wheat grains pouring", "search_query": "wheat grains pouring"}).status != S.ACCEPTED)
check("high semantic_score with no clip text is not ACCEPTED",
      H.evaluate(F, REQ, {"semantic_score": 99.0}).status != S.ACCEPTED)
check("photographer name is not evidence",
      H.evaluate(F, REQ, {"photographer": "wheat grains pouring"}).status != S.ACCEPTED)
check("subject-only clip text is UNCERTAIN (needs corroboration)",
      H.evaluate(F, REQ, {"clip_text": "wheat grains on a table"}).status == S.UNCERTAIN)
check("subject + action in clip text remains UNCERTAIN without visual verification",
      H.evaluate(F, REQ, {"clip_text": "farmer pouring wheat grains from hands"}).status == S.UNCERTAIN)
check("subject + setting in clip text remains UNCERTAIN without visual verification",
      H.evaluate(F, REQ, {"clip_text": "wheat grains stored in old barn"}).status == S.UNCERTAIN)
check("unrelated clip text is not ACCEPTED",
      H.evaluate(F, REQ, {"clip_text": "city traffic at night"}).status != S.ACCEPTED)
check("explicit exclusion in clip text is REJECTED",
      H.evaluate(F, {**REQ, "explicit_exclusions": ["bathtub"]}, {"clip_text": "wheat grains pouring bathtub"}).status == S.REJECTED)

# --- classifier ---
from app.services.providers.groq_script_provider import GroqScriptProvider as G
c = G._classify_topic
check("'ancient seed germination' -> science (was history)", c("ancient seed germination") == "science")
check("'history of coffee' still -> history", c("the history of coffee") == "history")
check("'ancient Rome' (no other category) -> history", c("ancient Rome") == "history")
check("'how coffee is roasted' -> cooking", c("how coffee is roasted") == "cooking")

# --- prompts ---
import inspect
src = inspect.getsource(G.generate_script)
check("script prompt no longer asks for 'rich, immersive' prose", "rich, immersive" not in src and "rich in detail" not in src)

# --- portability ---
import re
asm = open("app/services/video_assembly.py", encoding="utf-8").read()
check("no hard-coded c:/AiVideoGenerator paths in video_assembly", not re.search(r"[cC]:[/\\\\]AiVideoGenerator", asm))

orch = open("app/services/pipeline_orchestrator.py", encoding="utf-8").read()
check("best_effort clips are flagged unverified, not frame-verified",
      "v_res.is_visual_frame_verified = False" in orch and '"unverified_best_effort"' in orch)
check("orchestrator comment no longer claims uncertain footage never reaches assembly",
      "is NEVER sent to final assembly" not in orch)

print("\n%d failed" % len(fails)); sys.exit(1 if fails else 0)

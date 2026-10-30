"""
Targeted regression tests for Phase 2.1 and Phase 2.2:
- Scene-specific subject extraction (replacing topic-level matching)
- Dry seed versus grown seedling rejection
- Root direction validation (downward accepted, upward rejected)
- Enzyme & biochemical activity classification (ConceptClassifier)
- Prevention of substituting unrelated physical actions for abstract processes
- Truthful observable alternative vs. requiring animation / manual review
"""
import os
import sys
import types
from contextlib import contextmanager

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")

@contextmanager
def assert_raises(exc_type):
    try:
        yield
    except exc_type as e:
        return
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"Expected {exc_type.__name__} was not raised")

try:
    import av  # noqa: F401
except Exception:
    sys.modules["av"] = types.ModuleType("av")

import numpy as np
from PIL import Image
from pydantic import ValidationError

from app.schemas.visual_spec import ShotSpecificationSchema, VisualSpecificationSchema
from app.services.visual_verification import (
    ConceptClassifier,
    ConceptFeasibility,
    HeuristicVisualVerifier,
    VerificationStatus,
)
from app.services.pipeline_orchestrator import extract_scene_subject


def create_dummy_moving_frames():
    rng = np.random.default_rng(42)
    frames = []
    for t in (0.5, 1.0, 1.5):
        arr = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
        frames.append((t, Image.fromarray(arr), arr))
    return frames


# ======================================================================
# 1. SCENE-SPECIFIC SUBJECT EXTRACTION
# ======================================================================

def test_scene_specific_subject_extraction_replaces_topic():
    topic = "How a seed grows into a plant"

    # Scene 1: Seed imbibition
    sc1 = {
        "scene_text": "Water enters the dry seed, initiating the germination process.",
        "visual_prompt": "Cinematic macro shot of a dry seed resting in dark soil as water droplets fall onto it.",
        "search_keywords": ["dry seed soil water", "seed absorbing water macro"],
    }
    extracted1 = extract_scene_subject(sc1, fallback_topic=topic)
    assert extracted1 == "dry seed", f"Expected 'dry seed', got '{extracted1}'"
    assert extracted1 != topic, "Must not default to full topic string"

    # Scene 2: Root emergence
    sc2 = {
        "scene_text": "A root tip emerges, pushing downward into the soil to absorb nutrients.",
        "visual_prompt": "Extreme close-up of a root tip emerging from a cracked seed coat and pushing downward into the dark soil.",
        "search_keywords": ["root tip emerging soil", "seed germination root macro"],
    }
    extracted2 = extract_scene_subject(sc2, fallback_topic=topic)
    assert "root tip" in extracted2, f"Expected 'root tip' in extracted subject, got '{extracted2}'"
    assert extracted2 != topic, "Must not default to full topic string"

    # Scene 3: Explicit primary_subject preservation
    sc3 = {
        "primary_subject": "pale radicle in soil",
        "scene_text": "The primary root anchors into the ground.",
        "visual_prompt": "Extreme close-up of white root pushing into moist dirt",
        "search_keywords": ["white root soil close up"],
    }
    extracted3 = extract_scene_subject(sc3, fallback_topic=topic)
    assert extracted3 == "pale radicle in soil"


# ======================================================================
# 2. DRY SEED VERSUS GROWN SEEDLING REJECTION
# ======================================================================

def test_dry_seed_rejects_grown_seedling():
    frames = create_dummy_moving_frames()
    req = {
        "primary_subject": "dry seed in soil",
        "required_visible_action": "water enters dry seed absorbing moisture",
        "required_setting": "dark soil",
        "required_objects": ["seed", "water droplets"],
        "explicit_exclusions": [],
    }

    # Benchmark candidate failure case (Pexels 30252372 match: young green pea seedling)
    seedling_candidate = {
        "clip_text": "Young Green Pea Seedling Growing in Pot with healthy green leaves and sprouted foliage",
        "title": "green pea seedling sprout",
    }
    res_rejected = HeuristicVisualVerifier.evaluate(frames, req, seedling_candidate)
    assert res_rejected.status == VerificationStatus.REJECTED, (
        f"Grown seedling must be REJECTED for dry seed requirement, got {res_rejected.status}"
    )
    assert any("Developmental stage contradiction" in c for c in res_rejected.contradictions), (
        f"Expected developmental stage contradiction, got: {res_rejected.contradictions}"
    )

    # Truthful dry seed clip
    accurate_seed_candidate = {
        "clip_text": "macro shot of dry seed resting in dark soil as water enters and moistens the seed coat",
        "title": "dry seed in soil with water",
    }
    res_accepted = HeuristicVisualVerifier.evaluate(frames, req, accurate_seed_candidate)
    assert not any(
        "Developmental stage contradiction" in c
        for c in res_accepted.contradictions
    )
    assert res_accepted.status == VerificationStatus.UNCERTAIN
    assert res_accepted.is_visual_frame_verified is False


# ======================================================================
# 3. ROOT DIRECTION VALIDATION (GRAVITROPISM & PHOTOTROPISM)
# ======================================================================

def test_root_direction_validation_rejects_upward_growth():
    # 1. ShotSpecificationSchema rejects roots described as growing upward
    with assert_raises(ValidationError):
        ShotSpecificationSchema(
            shot_index=1,
            primary_subject="emerging root tip",
            required_visible_action="A root tip emerges and elongates upward toward the sky",
            required_setting="dark soil",
            search_keywords=["root growing macro", "seed germination timelapse"],
            provisional_start_seconds=0.0,
            provisional_end_seconds=3.0,
        )

    # 2. ShotSpecificationSchema accepts roots described as growing downward into soil
    valid_shot = ShotSpecificationSchema(
        shot_index=1,
        primary_subject="emerging root tip",
        required_visible_action="A root tip emerges and pushes downward into dark soil",
        required_setting="dark soil",
        search_keywords=["root growing downward soil", "root radicle emergence macro"],
        provisional_start_seconds=0.0,
        provisional_end_seconds=3.0,
    )
    assert valid_shot.primary_subject == "emerging root tip"
    assert valid_shot.expected_duration == 3.0

    # 3. VisualSpecificationSchema rejects scene prompt with root growing upward
    with assert_raises(ValidationError):
        VisualSpecificationSchema(
            primary_subject="root radicle",
            visual_prompt="Macro close-up of root radicle growing upward towards the surface",
            search_keywords=["root radicle growing", "plant growth macro"],
        )

    # 4. Verifier rejects above-ground canopy footage for subterranean root emergence
    frames = create_dummy_moving_frames()
    root_req = {
        "primary_subject": "root tip",
        "required_visible_action": "root tip emerges pushing downward into soil",
        "required_setting": "dark soil underground",
        "required_objects": ["root tip", "soil"],
        "explicit_exclusions": [],
    }
    canopy_candidate = {
        "clip_text": "sunny green tree canopy against blue sky with leaves fluttering in wind",
        "title": "treetop leaves canopy",
    }
    res_canopy = HeuristicVisualVerifier.evaluate(frames, root_req, canopy_candidate)
    assert res_canopy.status == VerificationStatus.REJECTED
    assert any("Botanical contradiction" in c for c in res_canopy.contradictions)


# ======================================================================
# 4. ENZYME & BIOCHEMICAL ACTIVITY IDENTIFICATION (CONCEPT CLASSIFIER)
# ======================================================================

def test_enzyme_activity_classified_as_conceptual_abstract():
    # Biochemical processes must be recognized as conceptual abstract
    test_cases = [
        ("Inside, enzymes awaken, breaking down stored food into simple sugars.", "enzymes in seed"),
        ("Active enzymes catalyze starch breakdown into glucose molecules.", "enzymes starch conversion"),
        ("Cellular respiration and atp synthesis in plant mitochondria", "cellular respiration"),
        ("Stored food into simple sugars for metabolic energy", "starch conversion"),
    ]

    for action, subj in test_cases:
        feas, reason = ConceptClassifier.classify_feasibility(action, subj)
        assert feas == ConceptFeasibility.CONCEPTUAL_ABSTRACT, (
            f"Expected CONCEPTUAL_ABSTRACT for '{action}', got {feas}"
        )
        assert reason is not None and "internal" in reason

    # Filmable macroscopic actions remain STOCK_FILMABLE
    filmable_action = "Pouring golden cooking oil into hot skillet"
    filmable_subj = "cooking oil in pan"
    feas_filmable, _ = ConceptClassifier.classify_feasibility(filmable_action, filmable_subj)
    assert feas_filmable == ConceptFeasibility.STOCK_FILMABLE


# ======================================================================
# 5. PREVENTION OF UNRELATED PHYSICAL ACTION SUBSTITUTION
# ======================================================================

def test_prevents_substituting_unrelated_actions_for_enzymes():
    frames = create_dummy_moving_frames()

    enzyme_req = {
        "primary_subject": "enzymes breaking down food",
        "required_visible_action": "enzymes activate and break down stored starch into sugars",
        "required_setting": "interior seed cotyledon",
        "required_objects": [],
        "explicit_exclusions": [],
    }

    # Benchmark candidate failure case (Pexels 6381725: pouring water from bottle into glass in kitchen)
    kitchen_glass_candidate = {
        "clip_text": "pouring water from a bottle into a drinking glass in modern kitchen",
        "title": "water poured into glass kitchen",
    }

    res = HeuristicVisualVerifier.evaluate(frames, enzyme_req, kitchen_glass_candidate)
    assert res.status == VerificationStatus.REJECTED, (
        f"Kitchen water pouring must be REJECTED for enzyme activation, got {res.status}"
    )
    assert any("Physical substitution contradiction" in c for c in res.contradictions), (
        f"Expected physical substitution contradiction, got: {res.contradictions}"
    )
    assert res.recommended_strategy == "diagram_or_motion_graphic"


# ======================================================================
# 6. TRUTHFUL OBSERVABLE ALTERNATIVE VS ANIMATION REQUIREMENT
# ======================================================================

def test_truthful_observable_alternative_vs_animation():
    # A truthful observable macroscopic manifestation is STOCK_FILMABLE
    observable_action = "Observable macro timelapse of seed coat swelling and softening in moist soil"
    observable_subj = "seed coat swelling"
    feas, _ = ConceptClassifier.classify_feasibility(observable_action, observable_subj)
    assert feas == ConceptFeasibility.STOCK_FILMABLE, (
        "Observable macroscopic seed coat swelling is filmable"
    )

    # Internal molecular enzyme catalysis is CONCEPTUAL_ABSTRACT requiring diagram
    internal_action = "enzymes converting starch into glucose at molecular level"
    internal_subj = "amylase enzymes"
    feas_internal, reason_internal = ConceptClassifier.classify_feasibility(internal_action, internal_subj)
    assert feas_internal == ConceptFeasibility.CONCEPTUAL_ABSTRACT
    assert "internal" in reason_internal


if __name__ == "__main__":
    tests = [
        ("test_scene_specific_subject_extraction_replaces_topic", test_scene_specific_subject_extraction_replaces_topic),
        ("test_dry_seed_rejects_grown_seedling", test_dry_seed_rejects_grown_seedling),
        ("test_root_direction_validation_rejects_upward_growth", test_root_direction_validation_rejects_upward_growth),
        ("test_enzyme_activity_classified_as_conceptual_abstract", test_enzyme_activity_classified_as_conceptual_abstract),
        ("test_prevents_substituting_unrelated_actions_for_enzymes", test_prevents_substituting_unrelated_actions_for_enzymes),
        ("test_truthful_observable_alternative_vs_animation", test_truthful_observable_alternative_vs_animation),
    ]

    failed = []
    print("=" * 70)
    print("RUNNING TARGETED PHASE 2.1 & 2.2 ACCURACY REGRESSION SUITE")
    print("=" * 70)
    for name, test_func in tests:
        try:
            test_func()
            print(f"PASS {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failed.append((name, e))

    print("\n%d failed out of %d tests" % (len(failed), len(tests)))
    sys.exit(1 if failed else 0)

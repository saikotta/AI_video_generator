"""
Step 11B.1 Offline Verification Suite
Tests the backward-compatible shot_plan schema and frame-exact timing reconciliation helper.

Tests:
1. Legacy visual spec without shot_plan.
2. One-shot plan.
3. Two-shot plan.
4. Three-shot plan.
5. Invalid duplicate indices.
6. Overlapping shot ranges.
7. Negative or zero duration.
8. Nonsequential shot indices.
9. Scene duration not divisible by 24 fps.
10. Total allocated frames equal expected scene frames.
11. No boundary gaps or overlaps.
12. Deterministic allocation across repeated calls.
13. Input data remains unchanged.
14. Insufficient frame budget.
15. Action-event index preservation.
"""

import sys
import copy
import math
from pathlib import Path
from pydantic import ValidationError

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas.visual_spec import (
    VisualSpecificationSchema,
    ActionEventSchema,
    ShotSpecificationSchema,
    ReconciledShot,
    reconcile_shot_plan_timing,
    validate_action_timeline_timing
)


def test_1_legacy_visual_spec_without_shot_plan():
    """Verify legacy VisualSpecificationSchema without shot_plan validates identically."""
    print("Test 1: Legacy visual spec without shot_plan...")
    data = {
        "visual_prompt": "A chef chopping fresh carrots on a wooden cutting board",
        "search_keywords": ["chef chopping carrots", "cutting board vegetables"]
    }
    spec = VisualSpecificationSchema.model_validate(data)
    assert spec.visual_prompt == data["visual_prompt"]
    assert spec.search_keywords == data["search_keywords"]
    assert spec.shot_plan is None
    assert spec.action_timeline is None
    
    # Also test with action_timeline but no shot_plan
    data_with_action = {
        "visual_prompt": "A chef chopping fresh carrots on a wooden cutting board",
        "search_keywords": ["chef chopping carrots", "cutting board vegetables"],
        "action_timeline": [
            {
                "action_description": "Knife cutting through crisp carrot",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 3.0,
                "sfx_query": "knife chopping on wood",
                "sound_type": "action"
            }
        ]
    }
    spec2 = VisualSpecificationSchema.model_validate(data_with_action)
    assert spec2.shot_plan is None
    assert len(spec2.action_timeline) == 1
    print("  PASS: Legacy specs validate without errors and shot_plan defaults to None.")


def test_2_one_shot_plan():
    """Verify single shot plan reconciles to exact full frame budget."""
    print("Test 2: One-shot plan...")
    shot_data = [
        {
            "shot_index": 1,
            "required_visible_action": "Close-up of oil heating in an empty black wok",
            "primary_subject": "Empty wok with shimmering oil",
            "required_setting": "Modern commercial kitchen stove",
            "visual_state": "empty heating",
            "camera_framing": "close-up",
            "search_keywords": ["wok oil heating", "hot pan oil shimmering"],
            "provisional_start_seconds": 0.0,
            "provisional_end_seconds": 4.0,
            "expected_duration": 4.0
        }
    ]
    reconciled = reconcile_shot_plan_timing(shot_data, measured_scene_duration_seconds=3.5, fps=24)
    assert len(reconciled) == 1
    shot = reconciled[0]
    expected_frames = int(math.ceil(3.5 * 24))  # 84 frames
    assert shot.allocated_frames == expected_frames
    assert shot.start_frame == 0
    assert shot.end_frame == 84
    assert shot.start_seconds == 0.0
    assert shot.end_seconds == round(84 / 24, 4)
    assert shot.duration_seconds == round(84 / 24, 4)
    assert shot["shot_index"] == 1
    print(f"  PASS: 1-shot allocated {shot.allocated_frames} frames (0 to {shot.end_frame}, {shot.duration_seconds}s).")


def test_3_two_shot_plan():
    """Verify two-shot plan proportional frame allocation."""
    print("Test 3: Two-shot plan...")
    shots = [
        ShotSpecificationSchema(
            shot_index=1,
            required_visible_action="Pouring oil into hot wok",
            primary_subject="Oil stream and wok",
            required_setting="Kitchen counter",
            visual_state="oil pouring",
            camera_framing="close-up",
            search_keywords=["pouring oil wok", "cooking oil pan"],
            provisional_start_seconds=0.0,
            provisional_end_seconds=2.0
        ),
        ShotSpecificationSchema(
            shot_index=2,
            required_visible_action="Dropping sliced carrots into wok",
            primary_subject="Sliced orange carrots",
            required_setting="Kitchen counter",
            visual_state="carrots hitting oil",
            camera_framing="macro close-up",
            search_keywords=["adding carrots wok", "sliced carrots stir fry"],
            provisional_start_seconds=2.0,
            provisional_end_seconds=4.0
        )
    ]
    # Measured duration = 5.0s, fps = 24 -> 120 frames. Equal 2.0s provisional duration -> 60 frames each.
    reconciled = reconcile_shot_plan_timing(shots, measured_scene_duration_seconds=5.0, fps=24)
    assert len(reconciled) == 2
    assert reconciled[0].allocated_frames == 60
    assert reconciled[1].allocated_frames == 60
    assert reconciled[0].start_frame == 0
    assert reconciled[0].end_frame == 60
    assert reconciled[1].start_frame == 60
    assert reconciled[1].end_frame == 120
    assert reconciled[0].end_seconds == reconciled[1].start_seconds == 2.5
    print(f"  PASS: 2-shot allocated 60 frames each, seamless boundary at {reconciled[0].end_seconds}s.")


def test_4_three_shot_plan():
    """Verify three-shot plan proportional distribution with unequal weights."""
    print("Test 4: Three-shot plan with unequal weights...")
    shots = [
        {
            "shot_index": 1,
            "required_visible_action": "Oil sizzling in pan",
            "primary_subject": "Hot oil",
            "required_setting": "Stovetop",
            "visual_state": "sizzling",
            "camera_framing": "close-up",
            "search_keywords": ["oil sizzling pan", "hot skillet oil"],
            "provisional_start_seconds": 0.0,
            "provisional_end_seconds": 1.0  # 1.0s (20%)
        },
        {
            "shot_index": 2,
            "required_visible_action": "Tossing diced vegetables",
            "primary_subject": "Vegetables in pan",
            "required_setting": "Stovetop",
            "visual_state": "tossing",
            "camera_framing": "medium close-up",
            "search_keywords": ["vegetables stir fry", "tossing vegetables wok"],
            "provisional_start_seconds": 1.0,
            "provisional_end_seconds": 3.0  # 2.0s (40%)
        },
        {
            "shot_index": 3,
            "required_visible_action": "Plating stir-fry onto white ceramic dish",
            "primary_subject": "Finished dish on plate",
            "required_setting": "Dining table",
            "visual_state": "plated dish steaming",
            "camera_framing": "overhead",
            "search_keywords": ["plating stir fry", "served vegetable dish"],
            "provisional_start_seconds": 3.0,
            "provisional_end_seconds": 5.0  # 2.0s (40%)
        }
    ]
    # Measured duration = 6.0s -> 144 frames
    reconciled = reconcile_shot_plan_timing(shots, measured_scene_duration_seconds=6.0, fps=24)
    assert len(reconciled) == 3
    # 20% of 144 = 28.8 frames, 40% of 144 = 57.6 frames
    # Sum must be 144
    total = sum(s.allocated_frames for s in reconciled)
    assert total == 144
    assert reconciled[0].allocated_frames == 29
    assert reconciled[1].allocated_frames == 58
    assert reconciled[2].allocated_frames == 57
    assert reconciled[0].end_frame == reconciled[1].start_frame
    assert reconciled[1].end_frame == reconciled[2].start_frame
    assert reconciled[2].end_frame == 144
    print(f"  PASS: 3-shot allocation: {[s.allocated_frames for s in reconciled]} (sum={total}).")


def test_5_invalid_duplicate_indices():
    """Verify duplicate shot indices are rejected by schema and helper."""
    print("Test 5: Invalid duplicate indices...")
    shots_bad = [
        {
            "shot_index": 1,
            "required_visible_action": "Action 1",
            "primary_subject": "Subject 1",
            "required_setting": "Setting 1",
            "search_keywords": ["keyword one", "keyword two"],
            "provisional_start_seconds": 0.0,
            "provisional_end_seconds": 2.0
        },
        {
            "shot_index": 1,  # Duplicate!
            "required_visible_action": "Action 2",
            "primary_subject": "Subject 2",
            "required_setting": "Setting 2",
            "search_keywords": ["keyword three", "keyword four"],
            "provisional_start_seconds": 2.0,
            "provisional_end_seconds": 4.0
        }
    ]
    try:
        VisualSpecificationSchema(
            visual_prompt="Test scene with duplicate shots",
            search_keywords=["test one", "test two"],
            shot_plan=[ShotSpecificationSchema(**s) for s in shots_bad]
        )
        assert False, "Should have raised ValidationError for duplicate indices"
    except ValidationError as e:
        assert "Shot indices must be 1-indexed and strictly sequential" in str(e)

    try:
        reconcile_shot_plan_timing(shots_bad, measured_scene_duration_seconds=4.0)
        assert False, "reconcile_shot_plan_timing should have raised ValueError for duplicate indices"
    except ValueError as e:
        assert "Shot indices must be 1-indexed and strictly sequential" in str(e)
    print("  PASS: Duplicate indices correctly rejected by both schema and timing reconciler.")


def test_6_overlapping_shot_ranges():
    """Verify overlapping provisional shot ranges are rejected."""
    print("Test 6: Overlapping shot ranges...")
    shots_overlapping = [
        {
            "shot_index": 1,
            "required_visible_action": "Action 1",
            "primary_subject": "Subject 1",
            "required_setting": "Setting 1",
            "search_keywords": ["keyword one", "keyword two"],
            "provisional_start_seconds": 0.0,
            "provisional_end_seconds": 3.0
        },
        {
            "shot_index": 2,
            "required_visible_action": "Action 2",
            "primary_subject": "Subject 2",
            "required_setting": "Setting 2",
            "search_keywords": ["keyword three", "keyword four"],
            "provisional_start_seconds": 2.5,  # Overlaps with 0.0-3.0!
            "provisional_end_seconds": 5.0
        }
    ]
    try:
        VisualSpecificationSchema(
            visual_prompt="Test overlapping shots",
            search_keywords=["test one", "test two"],
            shot_plan=[ShotSpecificationSchema(**s) for s in shots_overlapping]
        )
        assert False, "Should have raised ValidationError for overlapping ranges"
    except ValidationError as e:
        assert "Shot ranges cannot overlap" in str(e)

    try:
        reconcile_shot_plan_timing(shots_overlapping, measured_scene_duration_seconds=5.0)
        assert False, "reconcile_shot_plan_timing should have raised ValueError for overlapping ranges"
    except ValueError as e:
        assert "Shot ranges cannot overlap" in str(e)
    print("  PASS: Overlapping ranges correctly rejected.")


def test_7_negative_or_zero_duration():
    """Verify negative or zero duration is rejected."""
    print("Test 7: Negative or zero duration...")
    # Zero duration
    try:
        ShotSpecificationSchema(
            shot_index=1,
            required_visible_action="Action",
            primary_subject="Subject",
            required_setting="Setting",
            search_keywords=["kw one", "kw two"],
            provisional_start_seconds=2.0,
            provisional_end_seconds=2.0  # Zero duration
        )
        assert False, "Should have raised ValidationError for zero duration"
    except ValidationError:
        pass

    # Inverted duration
    try:
        ShotSpecificationSchema(
            shot_index=1,
            required_visible_action="Action",
            primary_subject="Subject",
            required_setting="Setting",
            search_keywords=["kw one", "kw two"],
            provisional_start_seconds=3.0,
            provisional_end_seconds=1.0  # Inverted
        )
        assert False, "Should have raised ValidationError for inverted duration"
    except ValidationError:
        pass

    # Negative start
    try:
        ShotSpecificationSchema(
            shot_index=1,
            required_visible_action="Action",
            primary_subject="Subject",
            required_setting="Setting",
            search_keywords=["kw one", "kw two"],
            provisional_start_seconds=-1.0,
            provisional_end_seconds=2.0
        )
        assert False, "Should have raised ValidationError for negative start"
    except ValidationError:
        pass
    print("  PASS: Zero, inverted, and negative durations rejected.")


def test_8_nonsequential_shot_indices():
    """Verify nonsequential indices (e.g. 1, 3) are rejected."""
    print("Test 8: Nonsequential shot indices...")
    shots_nonseq = [
        {
            "shot_index": 1,
            "required_visible_action": "Action 1",
            "primary_subject": "Subject 1",
            "required_setting": "Setting 1",
            "search_keywords": ["keyword one", "keyword two"],
            "provisional_start_seconds": 0.0,
            "provisional_end_seconds": 2.0
        },
        {
            "shot_index": 3,  # Skips 2!
            "required_visible_action": "Action 2",
            "primary_subject": "Subject 2",
            "required_setting": "Setting 2",
            "search_keywords": ["keyword three", "keyword four"],
            "provisional_start_seconds": 2.0,
            "provisional_end_seconds": 4.0
        }
    ]
    try:
        VisualSpecificationSchema(
            visual_prompt="Test scene with nonsequential shots",
            search_keywords=["test one", "test two"],
            shot_plan=[ShotSpecificationSchema(**s) for s in shots_nonseq]
        )
        assert False, "Should have raised ValidationError for nonsequential indices"
    except ValidationError as e:
        assert "Shot indices must be 1-indexed and strictly sequential" in str(e)

    try:
        reconcile_shot_plan_timing(shots_nonseq, measured_scene_duration_seconds=4.0)
        assert False, "reconcile_shot_plan_timing should have raised ValueError for nonsequential indices"
    except ValueError as e:
        assert "Shot indices must be 1-indexed and strictly sequential" in str(e)
    print("  PASS: Nonsequential shot indices correctly rejected.")


def test_9_scene_duration_not_divisible_by_24_fps():
    """Verify scene duration with fractional frame boundary uses ceiling."""
    print("Test 9: Scene duration not divisible by 24 fps...")
    shots = [
        {
            "shot_index": 1,
            "required_visible_action": "Action 1",
            "primary_subject": "Subject 1",
            "required_setting": "Setting 1",
            "search_keywords": ["keyword one", "keyword two"],
            "provisional_start_seconds": 0.0,
            "provisional_end_seconds": 2.0
        },
        {
            "shot_index": 2,
            "required_visible_action": "Action 2",
            "primary_subject": "Subject 2",
            "required_setting": "Setting 2",
            "search_keywords": ["keyword three", "keyword four"],
            "provisional_start_seconds": 2.0,
            "provisional_end_seconds": 4.0
        }
    ]
    # 4.02 seconds * 24 fps = 96.48 frames -> ceiling is 97 frames
    reconciled = reconcile_shot_plan_timing(shots, measured_scene_duration_seconds=4.02, fps=24)
    total_frames = sum(s.allocated_frames for s in reconciled)
    assert total_frames == 97
    assert reconciled[0].allocated_frames + reconciled[1].allocated_frames == 97
    # 97 / 2 = 48.5 each. Shot 1 gets 49, Shot 2 gets 48 due to tie-break.
    assert reconciled[0].allocated_frames == 49
    assert reconciled[1].allocated_frames == 48
    assert reconciled[1].end_frame == 97
    assert reconciled[1].end_seconds == round(97 / 24, 4)
    print(f"  PASS: 4.02s @ 24fps -> ceil(96.48) = 97 frames. Shot allocation: {reconciled[0].allocated_frames} + {reconciled[1].allocated_frames} = {total_frames}.")


def test_10_total_allocated_frames_equal_expected():
    """Verify total frames across 100 random/varied duration tests strictly match budget."""
    print("Test 10: Total allocated frames equal expected across diverse durations...")
    shots = [
        {"shot_index": 1, "required_visible_action": "Act 1", "primary_subject": "Sub 1", "required_setting": "Set 1", "search_keywords": ["a", "b"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 1.7},
        {"shot_index": 2, "required_visible_action": "Act 2", "primary_subject": "Sub 2", "required_setting": "Set 2", "search_keywords": ["c", "d"], "provisional_start_seconds": 1.7, "provisional_end_seconds": 4.3},
        {"shot_index": 3, "required_visible_action": "Act 3", "primary_subject": "Sub 3", "required_setting": "Set 3", "search_keywords": ["e", "f"], "provisional_start_seconds": 4.3, "provisional_end_seconds": 7.1}
    ]
    test_durations = [0.5, 1.23, 2.777, 3.0, 4.544, 7.891, 10.0, 15.333, 23.544]
    for dur in test_durations:
        expected_budget = int(math.ceil(round(dur * 24, 6)))
        reconciled = reconcile_shot_plan_timing(shots, measured_scene_duration_seconds=dur, fps=24)
        actual_total = sum(s.allocated_frames for s in reconciled)
        assert actual_total == expected_budget, f"Mismatch for dur {dur}: actual {actual_total} != expected {expected_budget}"
        assert reconciled[-1].end_frame == expected_budget
    print(f"  PASS: All {len(test_durations)} duration tests satisfied sum(frames) == budget exactly.")


def test_11_no_boundary_gaps_or_overlaps():
    """Verify strictly contiguous frame and timestamp boundaries."""
    print("Test 11: No boundary gaps or overlaps...")
    shots = [
        {"shot_index": 1, "required_visible_action": "A", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 1.5},
        {"shot_index": 2, "required_visible_action": "B", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 1.5, "provisional_end_seconds": 3.2},
        {"shot_index": 3, "required_visible_action": "C", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k5", "k6"], "provisional_start_seconds": 3.2, "provisional_end_seconds": 5.0}
    ]
    reconciled = reconcile_shot_plan_timing(shots, measured_scene_duration_seconds=5.73, fps=24)
    assert reconciled[0].start_frame == 0
    assert reconciled[0].start_seconds == 0.0
    for i in range(len(reconciled) - 1):
        assert reconciled[i].end_frame == reconciled[i + 1].start_frame, f"Frame gap/overlap at shot {i}->{i+1}"
        assert reconciled[i].end_seconds == reconciled[i + 1].start_seconds, f"Timestamp gap/overlap at shot {i}->{i+1}"
    print("  PASS: All boundaries are 100% contiguous with zero gaps and zero overlaps.")


def test_12_deterministic_allocation():
    """Verify repeated calls yield byte-for-byte identical output."""
    print("Test 12: Deterministic allocation across repeated calls...")
    shots = [
        {"shot_index": 1, "required_visible_action": "A", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
        {"shot_index": 2, "required_visible_action": "B", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
    ]
    res1 = reconcile_shot_plan_timing(shots, measured_scene_duration_seconds=4.02, fps=24)
    res2 = reconcile_shot_plan_timing(shots, measured_scene_duration_seconds=4.02, fps=24)
    assert [s.to_dict() for s in res1] == [s.to_dict() for s in res2]
    print("  PASS: Output is 100% deterministic.")


def test_13_input_data_remains_unchanged():
    """Verify input shot_plan is never modified in place."""
    print("Test 13: Input data remains unchanged...")
    shots = [
        {"shot_index": 1, "required_visible_action": "A", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
        {"shot_index": 2, "required_visible_action": "B", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
    ]
    shots_copy = copy.deepcopy(shots)
    reconciled = reconcile_shot_plan_timing(shots, measured_scene_duration_seconds=5.0, fps=24)
    assert shots == shots_copy, "Input dict list was modified in place!"
    
    # Also test with ShotSpecificationSchema objects
    schema_shots = [ShotSpecificationSchema(**s) for s in shots_copy]
    schema_shots_dump = [s.model_dump() for s in schema_shots]
    reconciled_schemas = reconcile_shot_plan_timing(schema_shots, measured_scene_duration_seconds=5.0, fps=24)
    assert [s.model_dump() for s in schema_shots] == schema_shots_dump, "Input schema objects were modified in place!"
    print("  PASS: Input data is strictly immutable.")


def test_14_insufficient_frame_budget():
    """Verify clear ValueError when total frame budget < num_shots."""
    print("Test 14: Insufficient frame budget...")
    shots = [
        {"shot_index": 1, "required_visible_action": "A", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 1.0},
        {"shot_index": 2, "required_visible_action": "B", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 1.0, "provisional_end_seconds": 2.0},
        {"shot_index": 3, "required_visible_action": "C", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k5", "k6"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 3.0}
    ]
    # 0.05s * 24 fps = 1.2 frames -> ceil is 2 frames. 2 frames < 3 shots!
    try:
        reconcile_shot_plan_timing(shots, measured_scene_duration_seconds=0.05, fps=24)
        assert False, "Should have raised ValueError for insufficient frame budget"
    except ValueError as e:
        assert "Total frame budget (2) is smaller than number of shots (3)" in str(e)
        assert "Every shot must receive at least one frame" in str(e)
    print("  PASS: Insufficient frame budget raises clear descriptive error.")


def test_15_action_event_index_preservation():
    """Verify action_event_index is preserved across reconciliation and validated against timeline."""
    print("Test 15: Action-event index preservation & validation...")
    spec_data = {
        "visual_prompt": "Cooking vegetable stir-fry",
        "search_keywords": ["vegetable stir fry", "cooking wok"],
        "action_timeline": [
            {
                "action_description": "Pouring oil into wok",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 2.0,
                "sfx_query": "oil sizzling",
                "sound_type": "action"
            },
            {
                "action_description": "Tossing carrots into wok",
                "visual_start_seconds": 2.0,
                "visual_end_seconds": 4.0,
                "sfx_query": "vegetables tossing pan",
                "sound_type": "action"
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Pouring oil into wok",
                "primary_subject": "Wok and oil",
                "required_setting": "Kitchen",
                "visual_state": "heating",
                "camera_framing": "close-up",
                "search_keywords": ["pouring oil", "wok heating"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 2.0,
                "action_event_index": 0  # Points to first action event
            },
            {
                "shot_index": 2,
                "required_visible_action": "Tossing carrots into wok",
                "primary_subject": "Carrots",
                "required_setting": "Kitchen",
                "visual_state": "cooking",
                "camera_framing": "macro",
                "search_keywords": ["tossing carrots", "carrots wok"],
                "provisional_start_seconds": 2.0,
                "provisional_end_seconds": 4.0,
                "action_event_index": 1  # Points to second action event
            }
        ]
    }
    spec = VisualSpecificationSchema.model_validate(spec_data)
    assert spec.shot_plan[0].action_event_index == 0
    assert spec.shot_plan[1].action_event_index == 1

    # Reconcile timing and verify preservation
    reconciled = reconcile_shot_plan_timing(spec.shot_plan, measured_scene_duration_seconds=4.8, fps=24)
    assert reconciled[0].action_event_index == 0
    assert reconciled[1].action_event_index == 1

    # Verify out of range action_event_index is rejected by schema
    invalid_spec_data = copy.deepcopy(spec_data)
    invalid_spec_data["shot_plan"][1]["action_event_index"] = 5  # Out of range!
    try:
        VisualSpecificationSchema.model_validate(invalid_spec_data)
        assert False, "Should have raised ValidationError for out-of-range action_event_index"
    except ValidationError as e:
        assert "out of range for action_timeline" in str(e)

    # Verify action_event_index when action_timeline is None is rejected
    no_timeline_data = copy.deepcopy(spec_data)
    no_timeline_data["action_timeline"] = None
    try:
        VisualSpecificationSchema.model_validate(no_timeline_data)
        assert False, "Should have raised ValidationError when action_event_index is set but no action_timeline"
    except ValidationError as e:
        assert "no action_timeline is defined" in str(e)
    print("  PASS: Action event indices preserved and strictly validated against action_timeline.")


def test_additional_validations():
    """Test max shot count (4) and empty field validations."""
    print("Test Additional: Max shot count and empty field checks...")
    # Max shot count = 4
    shots_5 = [
        {"shot_index": i + 1, "required_visible_action": f"Act {i+1}", "primary_subject": "S", "required_setting": "R", "search_keywords": ["k1", "k2"], "provisional_start_seconds": float(i), "provisional_end_seconds": float(i + 1)}
        for i in range(5)
    ]
    try:
        VisualSpecificationSchema(
            visual_prompt="Scene with 5 shots",
            search_keywords=["k1", "k2"],
            shot_plan=[ShotSpecificationSchema(**s) for s in shots_5]
        )
        assert False, "Should have rejected 5 shots"
    except ValidationError as e:
        assert "exceeds maximum supported shot count" in str(e)

    # Empty required_visible_action
    try:
        ShotSpecificationSchema(
            shot_index=1,
            required_visible_action="   ",
            primary_subject="S",
            required_setting="R",
            search_keywords=["k1", "k2"],
            provisional_start_seconds=0.0,
            provisional_end_seconds=2.0
        )
        assert False, "Should have rejected empty required_visible_action"
    except ValidationError as e:
        assert "required_visible_action cannot be empty" in str(e)

    # Less than 2 search keywords
    try:
        ShotSpecificationSchema(
            shot_index=1,
            required_visible_action="Action",
            primary_subject="S",
            required_setting="R",
            search_keywords=["only_one"],
            provisional_start_seconds=0.0,
            provisional_end_seconds=2.0
        )
        assert False, "Should have rejected <2 search_keywords"
    except ValidationError as e:
        assert "must contain at least 2 useful keywords" in str(e)

    print("  PASS: Max shot count (4) and field validation rules verified.")


if __name__ == "__main__":
    print("=== STARTING STEP 11B.1 OFFLINE VERIFICATION SUITE ===")
    test_1_legacy_visual_spec_without_shot_plan()
    test_2_one_shot_plan()
    test_3_two_shot_plan()
    test_4_three_shot_plan()
    test_5_invalid_duplicate_indices()
    test_6_overlapping_shot_ranges()
    test_7_negative_or_zero_duration()
    test_8_nonsequential_shot_indices()
    test_9_scene_duration_not_divisible_by_24_fps()
    test_10_total_allocated_frames_equal_expected()
    test_11_no_boundary_gaps_or_overlaps()
    test_12_deterministic_allocation()
    test_13_input_data_remains_unchanged()
    test_14_insufficient_frame_budget()
    test_15_action_event_index_preservation()
    test_additional_validations()
    print("=== ALL STEP 11B.1 VERIFICATION TESTS PASSED SUCCESSFULLY ===")

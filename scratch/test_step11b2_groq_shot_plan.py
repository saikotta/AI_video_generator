"""
Step 11B.2 Offline Verification Suite
Tests Groq sub-scene shot plan prompt formatting, JSON parsing, schema validation,
action linkage, fallback behavior, and regression safety.

Tests:
1. Single-action cooking scene.
2. Two-action cooking scene.
3. Three-action manufacturing scene.
4. Seed germination and root growth.
5. Technology assembly process.
6. Historical documentary with a legitimate setting transition.
7. Naturally silent physical action.
8. Multiple actions with valid action_event_index linkage.
9. Multiple actions without a corresponding sound event.
10. Invalid shot indices.
11. Overlapping provisional shot ranges.
12. Too many shots.
13. Invalid action_event_index.
14. Missing optional shot_plan.
15. Malformed Groq JSON.
16. Preservation of existing visual_prompt and search_keywords.
17. Backward compatibility with legacy visual specifications.
18. No fabricated actions unrelated to narration in representative mocked cases.
"""

import sys
import json
import asyncio
from pathlib import Path
from pydantic import ValidationError

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas.visual_spec import (
    VisualSpecificationSchema,
    ActionEventSchema,
    ShotSpecificationSchema,
    reconcile_shot_plan_timing
)
from app.services.providers.groq_script_provider import GroqScriptProvider
from app.services.providers.exceptions import InvalidScriptResponseError


def create_mock_provider(mock_response_str: str) -> GroqScriptProvider:
    """Creates a GroqScriptProvider instance with a mocked _post_groq method."""
    provider = GroqScriptProvider(api_key="gsk_mock_test_key_for_offline_tests")
    
    async def mock_post_groq(prompt: str, is_json: bool = False, temperature: float = 0.7) -> str:
        # Record prompt on provider for verification
        provider.last_prompt = prompt
        return mock_response_str

    provider._post_groq = mock_post_groq
    return provider


async def test_1_single_action_cooking():
    print("Test 1: Single-action cooking scene...")
    raw_json = json.dumps({
        "visual_prompt": "Cinematic close-up of a seasoned black wok heating over a bright blue gas flame until the oil shimmers.",
        "search_keywords": ["wok heating flame", "hot skillet oil shimmering", "cooking oil heating"],
        "action_timeline": [
            {
                "action_description": "Oil shimmers as wok heats over flame",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 4.0,
                "sound_type": "action",
                "sfx_query": "gas stove flame burner hiss",
                "sfx_start_seconds": 0.2,
                "sfx_duration_seconds": 3.0,
                "sfx_volume": 0.3
            }
        ],
        "shot_plan": None
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Heat the wok over medium flame until it shimmers.")
    assert res["shot_plan"] is None
    assert len(res["action_timeline"]) == 1
    assert "wok heating flame" in res["search_keywords"]
    assert not provider.last_shot_plan_fallback_occurred
    print("  PASS: Single-action cooking scene produces valid spec with shot_plan=None.")


async def test_2_two_action_cooking():
    print("Test 2: Two-action cooking scene...")
    raw_json = json.dumps({
        "visual_prompt": "Cinematic close-up of golden oil poured into hot wok, followed by crisp sliced carrots added to sizzle.",
        "search_keywords": ["pouring oil pan", "cooking carrots wok", "sliced carrots stir fry"],
        "action_timeline": [
            {
                "action_description": "Pouring cooking oil into hot wok",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 2.0,
                "sound_type": "action",
                "sfx_query": "oil sizzling hot pan",
                "sfx_start_seconds": 0.5,
                "sfx_duration_seconds": 1.5,
                "sfx_volume": 0.35
            },
            {
                "action_description": "Adding sliced carrots into sizzling pan",
                "visual_start_seconds": 2.0,
                "visual_end_seconds": 4.0,
                "sound_type": "action",
                "sfx_query": "vegetables dropping hot oil",
                "sfx_start_seconds": 2.2,
                "sfx_duration_seconds": 1.5,
                "sfx_volume": 0.4
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Pouring golden cooking oil into hot pan",
                "primary_subject": "Cooking oil streaming into pan",
                "required_setting": "Stovetop skillet",
                "visual_state": "oil heating shimmering",
                "camera_framing": "close-up",
                "search_keywords": ["pouring oil pan", "hot skillet oil shimmering"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 2.0,
                "expected_duration": 2.0,
                "action_event_index": 0
            },
            {
                "shot_index": 2,
                "required_visible_action": "Dropping fresh sliced carrots into hot oil",
                "primary_subject": "Sliced orange carrots",
                "required_setting": "Stovetop skillet",
                "visual_state": "carrots hitting hot oil sizzling",
                "camera_framing": "macro close-up",
                "search_keywords": ["adding carrots wok", "sliced carrots stir fry"],
                "provisional_start_seconds": 2.0,
                "provisional_end_seconds": 4.0,
                "expected_duration": 2.0,
                "action_event_index": 1
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Pour oil into the hot pan, then add sliced carrots.")
    assert len(res["shot_plan"]) == 2
    assert res["shot_plan"][0]["shot_index"] == 1
    assert res["shot_plan"][1]["shot_index"] == 2
    assert res["shot_plan"][0]["action_event_index"] == 0
    assert res["shot_plan"][1]["action_event_index"] == 1
    print("  PASS: Two-action cooking scene produces 2 valid sequential shots with exact action linkage.")


async def test_3_three_action_manufacturing():
    print("Test 3: Three-action manufacturing scene...")
    raw_json = json.dumps({
        "visual_prompt": "Industrial automated vehicle assembly showing welding, clamping, and conveyor movement.",
        "search_keywords": ["robotic welding automotive", "hydraulic clamp factory", "overhead conveyor assembly"],
        "action_timeline": [
            {
                "action_description": "Robotic arm welding chassis seam",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 2.0,
                "sound_type": "action",
                "sfx_query": "welding torch arc buzzing sparks",
                "sfx_start_seconds": 0.2,
                "sfx_duration_seconds": 1.8,
                "sfx_volume": 0.5
            },
            {
                "action_description": "Hydraulic clamp securing joint",
                "visual_start_seconds": 2.0,
                "visual_end_seconds": 4.0,
                "sound_type": "action",
                "sfx_query": "hydraulic clamp pressure pneumatic",
                "sfx_start_seconds": 2.1,
                "sfx_duration_seconds": 1.5,
                "sfx_volume": 0.4
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Robotic arm emitting bright sparks while welding chassis",
                "primary_subject": "Robotic welding torch and steel frame",
                "required_setting": "Automated vehicle assembly factory",
                "visual_state": "sparks flying glowing metal",
                "camera_framing": "medium close-up",
                "search_keywords": ["robot arm welding car", "factory welding sparks"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 2.0,
                "action_event_index": 0
            },
            {
                "shot_index": 2,
                "required_visible_action": "Heavy hydraulic clamps pressing steel joint firmly into position",
                "primary_subject": "Hydraulic clamps",
                "required_setting": "Automated vehicle assembly factory",
                "visual_state": "clamping pressure locked",
                "camera_framing": "close-up",
                "search_keywords": ["hydraulic clamp factory", "pneumatic industrial clamp"],
                "provisional_start_seconds": 2.0,
                "provisional_end_seconds": 4.0,
                "action_event_index": 1
            },
            {
                "shot_index": 3,
                "required_visible_action": "Overhead motorized conveyor rolling finished component down the line",
                "primary_subject": "Overhead transport conveyor",
                "required_setting": "Automated vehicle assembly factory",
                "visual_state": "conveyor moving smoothly",
                "camera_framing": "wide angle",
                "search_keywords": ["factory conveyor belt", "automotive assembly line motion"],
                "provisional_start_seconds": 4.0,
                "provisional_end_seconds": 6.0,
                "action_event_index": None  # No SFX event for this shot
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Robotic arms weld chassis, hydraulic clamps lock joints, and overhead conveyors move components.")
    assert len(res["shot_plan"]) == 3
    assert res["shot_plan"][2]["action_event_index"] is None
    print("  PASS: Three-action manufacturing scene validates with 3 shots and mixed action event linkage.")


async def test_4_seed_germination_root_growth():
    print("Test 4: Seed germination and root growth...")
    raw_json = json.dumps({
        "visual_prompt": "Underground botanical timelapse showing a seed coat swelling followed by a white taproot growing into soil.",
        "search_keywords": ["seed germination timelapse", "taproot growing underground soil"],
        "action_timeline": [
            {
                "action_description": "Seed absorbs moisture in dark soil",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 2.5,
                "sound_type": "none",
                "sfx_query": None
            },
            {
                "action_description": "White taproot emerges and penetrates downward",
                "visual_start_seconds": 2.5,
                "visual_end_seconds": 5.0,
                "sound_type": "none",
                "sfx_query": None
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Moisture surrounds dark seed coat as it visibly swells",
                "primary_subject": "Swelling plant seed",
                "required_setting": "Underground dark organic soil cross section",
                "visual_state": "hydrating swelling seed",
                "camera_framing": "macro close-up",
                "search_keywords": ["seed germination timelapse", "seed swelling water dirt"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 2.5,
                "action_event_index": 0
            },
            {
                "shot_index": 2,
                "required_visible_action": "Delicate white taproot pushes steadily down through dark particles",
                "primary_subject": "Growing white taproot tip",
                "required_setting": "Underground dark organic soil cross section",
                "visual_state": "root growth elongating",
                "camera_framing": "macro underground",
                "search_keywords": ["root growing underground timelapse", "roots dirt cross section"],
                "provisional_start_seconds": 2.5,
                "provisional_end_seconds": 5.0,
                "action_event_index": 1
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Beneath dark soil, the seed swells, and a white taproot pushes downward.")
    assert len(res["shot_plan"]) == 2
    assert res["action_timeline"][0]["sound_type"] == "none"
    assert res["action_timeline"][1]["sound_type"] == "none"
    print("  PASS: Biological growth scene validates with silent actions and concrete sequential shots.")


async def test_5_technology_assembly():
    print("Test 5: Technology assembly process...")
    raw_json = json.dumps({
        "visual_prompt": "Electronics lab technician soldering microchip onto circuit board, then inspecting joints under microscope.",
        "search_keywords": ["soldering circuit board", "microscope electronic inspection"],
        "action_timeline": [
            {
                "action_description": "Precision soldering iron melting wire onto circuit pin",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 2.5,
                "sound_type": "action",
                "sfx_query": "solder iron sizzle flux smoke",
                "sfx_start_seconds": 0.3,
                "sfx_duration_seconds": 1.2,
                "sfx_volume": 0.3
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Soldering iron applying molten silver wire to microchip lead",
                "primary_subject": "Soldering tip and chip pin",
                "required_setting": "Cleanroom electronics workstation",
                "visual_state": "flux vaporizing shiny solder melting",
                "camera_framing": "extreme close-up",
                "search_keywords": ["soldering microchip macro", "solder iron circuit board"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 2.5,
                "action_event_index": 0
            },
            {
                "shot_index": 2,
                "required_visible_action": "Stereo optical microscope illuminated view inspecting circuit traces",
                "primary_subject": "Gold circuit traces under lens",
                "required_setting": "Cleanroom electronics workstation",
                "visual_state": "inspected polished solder joints",
                "camera_framing": "microscopic view",
                "search_keywords": ["circuit board microscope inspection", "microchip inspection macro"],
                "provisional_start_seconds": 2.5,
                "provisional_end_seconds": 5.0,
                "action_event_index": None
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("An engineer solders a high-density microchip, then inspects circuit traces under magnification.")
    assert len(res["shot_plan"]) == 2
    assert res["shot_plan"][0]["action_event_index"] == 0
    assert res["shot_plan"][1]["action_event_index"] is None
    print("  PASS: Technology assembly validates with mixed SFX linkage and valid framing.")


async def test_6_historical_documentary_transition():
    print("Test 6: Historical documentary with legitimate setting transition...")
    raw_json = json.dumps({
        "visual_prompt": "Historical transition from fertile ancient river valley farming to arid desert caravan trade routes.",
        "search_keywords": ["ancient river valley farming", "desert caravan dunes trail"],
        "action_timeline": [
            {
                "action_description": "River stream flows beside ancient terraced fields",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 3.0,
                "sound_type": "ambience",
                "sfx_query": "river stream flowing gentle breeze",
                "sfx_start_seconds": 0.0,
                "sfx_duration_seconds": 3.0,
                "sfx_volume": 0.25
            },
            {
                "action_description": "Desert trade caravan moving across sand dunes",
                "visual_start_seconds": 3.0,
                "visual_end_seconds": 6.0,
                "sound_type": "ambience",
                "sfx_query": "desert wind blowing sand dunes",
                "sfx_start_seconds": 3.0,
                "sfx_duration_seconds": 3.0,
                "sfx_volume": 0.25
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Farmers threshing wild grains alongside flowing riverbanks",
                "primary_subject": "Grain harvest near river",
                "required_setting": "Lush ancient river valley",
                "visual_state": "active harvest golden fields",
                "camera_framing": "medium shot",
                "search_keywords": ["ancient river valley agriculture", "harvesting grain ancient fields"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 3.0,
                "action_event_index": 0
            },
            {
                "shot_index": 2,
                "required_visible_action": "Camels and merchant caravan trekking slowly across vast sand dunes",
                "primary_subject": "Desert merchant caravan",
                "required_setting": "Arid sand dunes trade trail",
                "visual_state": "caravan moving dust rising",
                "camera_framing": "wide aerial",
                "search_keywords": ["desert trade caravan dunes", "sand dunes camels walking"],
                "provisional_start_seconds": 3.0,
                "provisional_end_seconds": 6.0,
                "action_event_index": 1
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("In river valleys, early farmers harvested grains, before trade caravans traveled across desert dunes.")
    assert len(res["shot_plan"]) == 2
    assert res["shot_plan"][0]["required_setting"] != res["shot_plan"][1]["required_setting"]
    print("  PASS: Legitimate narrative setting transition between shots validates cleanly.")


async def test_7_naturally_silent_physical_action():
    print("Test 7: Naturally silent physical action...")
    raw_json = json.dumps({
        "visual_prompt": "Microscopic view of root hairs absorbing moisture in dark soil without sound.",
        "search_keywords": ["root hairs soil macro", "plant roots microscopic dirt"],
        "action_timeline": [
            {
                "action_description": "Microscopic root hairs absorb moisture droplets in soil",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 3.5,
                "sound_type": "none",
                "sfx_query": None
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Microscopic root hairs absorbing water film in dark soil",
                "primary_subject": "Fine root hair structures",
                "required_setting": "Soil cross section",
                "visual_state": "moisture absorption",
                "camera_framing": "microscopic view",
                "search_keywords": ["root hairs absorbing water", "microscopic plant roots soil"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 3.5,
                "action_event_index": 0
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Beneath the soil, microscopic root hairs absorb moisture in total silence.")
    assert res["action_timeline"][0]["sound_type"] == "none"
    assert res["action_timeline"][0]["sfx_query"] is None
    print("  PASS: Silent biological action validated without fabricated sound effects.")


async def test_8_multiple_actions_with_event_linkage():
    print("Test 8: Multiple actions with valid action_event_index linkage...")
    raw_json = json.dumps({
        "visual_prompt": "Pouring oil and adding carrots in hot wok.",
        "search_keywords": ["pouring oil wok", "carrots stir fry"],
        "action_timeline": [
            {
                "action_description": "Action 0: Pour oil",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 2.0,
                "sound_type": "action",
                "sfx_query": "oil sizzling"
            },
            {
                "action_description": "Action 1: Drop carrots",
                "visual_start_seconds": 2.0,
                "visual_end_seconds": 4.0,
                "sound_type": "action",
                "sfx_query": "vegetables dropping"
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Pour oil",
                "primary_subject": "Oil",
                "required_setting": "Kitchen",
                "search_keywords": ["pouring oil", "hot wok oil"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 2.0,
                "action_event_index": 0
            },
            {
                "shot_index": 2,
                "required_visible_action": "Drop carrots",
                "primary_subject": "Carrots",
                "required_setting": "Kitchen",
                "search_keywords": ["adding carrots", "sliced carrots wok"],
                "provisional_start_seconds": 2.0,
                "provisional_end_seconds": 4.0,
                "action_event_index": 1
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Pour oil and drop carrots.")
    assert res["shot_plan"][0]["action_event_index"] == 0
    assert res["shot_plan"][1]["action_event_index"] == 1
    print("  PASS: Sequential 0-based action_event_index links validated.")


async def test_9_multiple_actions_without_corresponding_sound():
    print("Test 9: Multiple actions without corresponding sound event...")
    raw_json = json.dumps({
        "visual_prompt": "Chef garnishing finished dish with parsley leaves.",
        "search_keywords": ["chef garnishing dish", "parsley on food plate"],
        "action_timeline": [],  # Silent scene, no action events
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Chef carefully places green herb sprig onto food",
                "primary_subject": "Garnish sprig and hands",
                "required_setting": "Kitchen plating station",
                "search_keywords": ["chef placing garnish", "garnishing plate food"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 3.0,
                "action_event_index": None
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Garnish the dish with a sprig of fresh parsley.")
    assert res["shot_plan"][0]["action_event_index"] is None
    print("  PASS: Shot plan without sound event correctly validates with action_event_index=None.")


async def test_10_invalid_shot_indices():
    print("Test 10: Invalid duplicate shot indices trigger fallback...")
    raw_json = json.dumps({
        "visual_prompt": "Cooking vegetables in pan.",
        "search_keywords": ["cooking vegetables", "pan stir fry"],
        "action_timeline": [
            {
                "action_description": "Vegetables sizzle in pan",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 4.0,
                "sound_type": "action",
                "sfx_query": "vegetables sizzling pan"
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Action 1",
                "primary_subject": "Sub 1",
                "required_setting": "Set 1",
                "search_keywords": ["kw one", "kw two"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 2.0
            },
            {
                "shot_index": 1,  # Duplicate!
                "required_visible_action": "Action 2",
                "primary_subject": "Sub 2",
                "required_setting": "Set 2",
                "search_keywords": ["kw three", "kw four"],
                "provisional_start_seconds": 2.0,
                "provisional_end_seconds": 4.0
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    # Default fallback: safe fallback to single shot (shot_plan=None)
    res = await provider.generate_visual_specification("Cook vegetables.")
    assert res["shot_plan"] is None
    assert provider.last_shot_plan_fallback_occurred
    assert "Shot indices must be 1-indexed and strictly sequential" in provider.last_shot_plan_error
    assert len(res["action_timeline"]) == 1

    # Strict mode (no fallback): raises InvalidScriptResponseError
    provider.fallback_to_single_shot = False
    try:
        await provider.generate_visual_specification("Cook vegetables.")
        assert False, "Should have raised InvalidScriptResponseError"
    except InvalidScriptResponseError as e:
        assert "Shot indices must be 1-indexed and strictly sequential" in str(e)

    print("  PASS: Invalid duplicate shot indices handled with safe fallback and strict error reporting.")


async def test_11_overlapping_provisional_shot_ranges():
    print("Test 11: Overlapping provisional shot ranges...")
    raw_json = json.dumps({
        "visual_prompt": "Cooking vegetables in pan.",
        "search_keywords": ["cooking vegetables", "pan stir fry"],
        "action_timeline": [],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Action 1",
                "primary_subject": "Sub 1",
                "required_setting": "Set 1",
                "search_keywords": ["kw one", "kw two"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 3.0
            },
            {
                "shot_index": 2,
                "required_visible_action": "Action 2",
                "primary_subject": "Sub 2",
                "required_setting": "Set 2",
                "search_keywords": ["kw three", "kw four"],
                "provisional_start_seconds": 2.5,  # Overlaps with 0.0-3.0!
                "provisional_end_seconds": 5.0
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Cook vegetables.")
    assert res["shot_plan"] is None
    assert provider.last_shot_plan_fallback_occurred
    assert "Shot ranges cannot overlap" in provider.last_shot_plan_error
    print("  PASS: Overlapping provisional ranges rejected and safely fallen back to single shot.")


async def test_12_too_many_shots():
    print("Test 12: Too many shots (exceeding MAX_SHOTS = 4)...")
    raw_json = json.dumps({
        "visual_prompt": "Long multi-step process with 5 shots.",
        "search_keywords": ["factory production", "assembly sequence"],
        "shot_plan": [
            {
                "shot_index": i + 1,
                "required_visible_action": f"Step {i+1}",
                "primary_subject": "Part",
                "required_setting": "Factory",
                "search_keywords": ["factory work", "industrial step"],
                "provisional_start_seconds": float(i),
                "provisional_end_seconds": float(i + 1)
            }
            for i in range(5)  # 5 shots! Max is 4.
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Multi-step factory production.")
    assert res["shot_plan"] is None
    assert provider.last_shot_plan_fallback_occurred
    assert "exceeds maximum supported shot count" in provider.last_shot_plan_error
    print("  PASS: Plans with > 4 shots safely fall back to single shot.")


async def test_13_invalid_action_event_index():
    print("Test 13: Invalid action_event_index out of bounds...")
    raw_json = json.dumps({
        "visual_prompt": "Pouring oil.",
        "search_keywords": ["pouring oil", "cooking wok"],
        "action_timeline": [
            {
                "action_description": "Pour oil",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 3.0,
                "sound_type": "action",
                "sfx_query": "oil sizzling"
            }
        ],
        "shot_plan": [
            {
                "shot_index": 1,
                "required_visible_action": "Pour oil",
                "primary_subject": "Oil",
                "required_setting": "Kitchen",
                "search_keywords": ["pouring oil", "hot oil"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 3.0,
                "action_event_index": 8  # Only 1 event in action_timeline (index 0)!
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Pour oil into hot wok.")
    assert res["shot_plan"] is None
    assert provider.last_shot_plan_fallback_occurred
    assert "out of range for action_timeline" in provider.last_shot_plan_error
    print("  PASS: Out-of-bounds action_event_index caught and safely fallen back.")


async def test_14_missing_optional_shot_plan():
    print("Test 14: Missing optional shot_plan in JSON...")
    raw_json = json.dumps({
        "visual_prompt": "A chef chopping fresh carrots on a cutting board.",
        "search_keywords": ["chef chopping carrots", "cutting board vegetables"],
        "action_timeline": [
            {
                "action_description": "Knife cutting carrot",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 3.0,
                "sound_type": "action",
                "sfx_query": "knife chop wood"
            }
        ]
        # shot_plan completely omitted from JSON
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Chop fresh carrots on a board.")
    assert res["shot_plan"] is None
    assert not provider.last_shot_plan_fallback_occurred
    assert res["visual_prompt"] == "A chef chopping fresh carrots on a cutting board."
    print("  PASS: JSON missing shot_plan validates cleanly with shot_plan=None.")


async def test_15_malformed_groq_json():
    print("Test 15: Malformed Groq JSON syntax...")
    raw_bad_json = "```json\n{ visual_prompt: unquoted, search_keywords: [ broken json \n```"
    provider = create_mock_provider(raw_bad_json)
    try:
        await provider.generate_visual_specification("Some scene.")
        assert False, "Should have raised InvalidScriptResponseError for malformed JSON"
    except InvalidScriptResponseError as e:
        assert "malformed json" in str(e).lower()
    print("  PASS: Malformed JSON syntax raises InvalidScriptResponseError.")


async def test_16_preservation_of_visual_prompt_and_keywords():
    print("Test 16: Preservation of visual_prompt and search_keywords during fallback...")
    raw_json = json.dumps({
        "visual_prompt": "Cinematic high-angle shot of sizzling stir-fry vegetables with steam billowing upward.",
        "search_keywords": ["vegetable stir fry pan", "steaming wok vegetables", "healthy cooking skillet"],
        "action_timeline": [
            {
                "action_description": "Vegetables sizzle actively",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 4.0,
                "sound_type": "action",
                "sfx_query": "sizzling pan"
            }
        ],
        "shot_plan": [
            {
                "shot_index": 5,  # Invalid shot_index!
                "required_visible_action": "Bad shot",
                "primary_subject": "S",
                "required_setting": "R",
                "search_keywords": ["k1", "k2"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 2.0
            }
        ]
    })
    provider = create_mock_provider(raw_json)
    res = await provider.generate_visual_specification("Sauté the vegetables.")
    assert res["visual_prompt"] == "Cinematic high-angle shot of sizzling stir-fry vegetables with steam billowing upward."
    assert res["search_keywords"] == ["vegetable stir fry pan", "steaming wok vegetables", "healthy cooking skillet"]
    assert len(res["action_timeline"]) == 1
    assert res["shot_plan"] is None
    print("  PASS: Visual prompt, keywords, and action timeline preserved 100% when fallback occurs.")


async def test_17_backward_compatibility_with_legacy_specs():
    print("Test 17: Backward compatibility with legacy specifications...")
    legacy_dict = {
        "visual_prompt": "Old legacy prompt without shot_plan or timeline",
        "search_keywords": ["legacy keyword 1", "legacy keyword 2"]
    }
    validated = VisualSpecificationSchema.model_validate(legacy_dict)
    assert validated.shot_plan is None
    assert validated.action_timeline is None
    dumped = validated.model_dump()
    assert dumped["shot_plan"] is None
    print("  PASS: Legacy specifications without shot_plan validate seamlessly.")


async def test_18_no_fabricated_actions():
    print("Test 18: No fabricated actions unrelated to narration in representative cases...")
    # Verify prompt text enforces no unrelated actions and strict relevance
    provider = create_mock_provider(json.dumps({
        "visual_prompt": "A chef cooking stir-fry.",
        "search_keywords": ["cooking stir fry", "wok pan vegetables"],
        "shot_plan": None
    }))
    await provider.generate_visual_specification("Drizzle a spoonful of golden sesame oil, then toss sliced carrots.")
    prompt = provider.last_prompt
    assert "STRICTLY FORBIDDEN: Do NOT include broad thematic concepts, historical/cultural associations" in prompt
    assert "NEVER create extra shots solely to increase shot count" in prompt
    assert "NEVER invent actions absent from the spoken narration or supported scene context" in prompt
    assert "Preserve the original scene narration meaning and chronological sequence exactly" in prompt
    print("  PASS: Prompt strictly forbids fabricated or unrelated actions.")


async def main():
    print("=== STARTING STEP 11B.2 OFFLINE VERIFICATION SUITE ===")
    await test_1_single_action_cooking()
    await test_2_two_action_cooking()
    await test_3_three_action_manufacturing()
    await test_4_seed_germination_root_growth()
    await test_5_technology_assembly()
    await test_6_historical_documentary_transition()
    await test_7_naturally_silent_physical_action()
    await test_8_multiple_actions_with_event_linkage()
    await test_9_multiple_actions_without_corresponding_sound()
    await test_10_invalid_shot_indices()
    await test_11_overlapping_provisional_shot_ranges()
    await test_12_too_many_shots()
    await test_13_invalid_action_event_index()
    await test_14_missing_optional_shot_plan()
    await test_15_malformed_groq_json()
    await test_16_preservation_of_visual_prompt_and_keywords()
    await test_17_backward_compatibility_with_legacy_specs()
    await test_18_no_fabricated_actions()
    print("=== ALL 18 STEP 11B.2 OFFLINE TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    asyncio.run(main())

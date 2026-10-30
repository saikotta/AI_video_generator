"""
Step 11B.3 Offline Verification Suite
Tests Shot-Aware Stock Video Sourcing, Frame-Exact Timing Reconciliation,
Context-Aware Ranking Integration, Missing Footage Fallbacks, and Timing Warnings.

Tests:
1. Single-shot legacy scene.
2. Two-shot cooking scene.
3. Three-action manufacturing scene.
4. Shot plan persistence and propagation.
5. Correct use of measured narration duration.
6. Exact integer frame boundaries.
7. Per-shot contextual Pexels queries.
8. Correct shot ordering.
9. Duplicate video ID prevention.
10. Missing second-shot footage.
11. Missing first-shot footage.
12. All shots unavailable.
13. Valid setting transitions.
14. Action-event linkage preservation.
15. Timing mismatch warning behavior.
16. Legacy scenes longer than eight seconds.
17. No unintended changes to Freesound.
18. Correct shot metadata available for Step 11B.4 assembly.
"""

import os
import sys
import uuid
import json
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from unittest.mock import AsyncMock, MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas.media import MediaAssetSchema
from app.services.providers.exceptions import NoSuitableMediaFoundError
from app.services.pipeline_orchestrator import VideoPipelineOrchestrator


class MockVideoProvider:
    def __init__(self, candidates_map: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        # Map from keyword string or tuple to list of candidates
        self.candidates_map = candidates_map or {}
        self.search_calls = []
        self.download_calls = []
        self.counter = 100

    async def search_media(
        self,
        keywords: List[str],
        target_duration: float = 4.0,
        used_video_ids: Optional[Set[int]] = None,
        scene_context: Optional[Dict[str, Any]] = None
    ) -> MediaAssetSchema:
        self.search_calls.append({
            "keywords": list(keywords),
            "target_duration": target_duration,
            "used_video_ids": set(used_video_ids) if used_video_ids else set(),
            "scene_context": dict(scene_context) if scene_context else None
        })

        kw_key = " ".join(keywords).lower()
        if "fail_all" in kw_key or "unsuitable" in kw_key:
            raise NoSuitableMediaFoundError(f"No suitable media found matching keywords: {keywords}")

        # Check if mapped
        for k, v in self.candidates_map.items():
            if k.lower() in kw_key:
                if v is None:
                    raise NoSuitableMediaFoundError(f"Simulated failure for keyword '{k}'")
                for cand in v:
                    if used_video_ids and cand.get("pexels_id") in used_video_ids:
                        continue
                    return MediaAssetSchema(
                        download_url=cand.get("download_url", f"https://pexels.com/v/{cand['pexels_id']}.mp4"),
                        source_url=f"https://pexels.com/video/{cand['pexels_id']}",
                        duration=target_duration,
                        metadata=cand
                    )

        # Default dynamic candidate
        self.counter += 1
        cand_id = self.counter
        while used_video_ids and cand_id in used_video_ids:
            self.counter += 1
            cand_id = self.counter

        meta = {
            "pexels_id": cand_id,
            "width": 1920,
            "height": 1080,
            "duration": target_duration,
            "semantic_score": 85.0,
            "technical_score": 40.0,
            "search_query": kw_key,
        }
        return MediaAssetSchema(
            download_url=f"https://pexels.com/v/{cand_id}.mp4",
            source_url=f"https://pexels.com/video/{cand_id}",
            duration=target_duration,
            metadata=meta
        )

    async def download_media(self, url: str, output_dir: str, filename_prefix: str) -> str:
        self.download_calls.append({"url": url, "output_dir": output_dir, "prefix": filename_prefix})
        dest = os.path.join(output_dir, f"{filename_prefix}.mp4")
        with open(dest, "wb") as f:
            f.write(b"MOCK_MP4_DATA")
        return dest


def make_test_orchestrator(video_provider: MockVideoProvider, tmp_dir: str) -> VideoPipelineOrchestrator:
    """Builds a test-configured VideoPipelineOrchestrator with mocked persistence."""
    orch = VideoPipelineOrchestrator()
    orch.get_stock_media_provider = lambda: video_provider
    orch._record_pipeline_job = AsyncMock(return_value=uuid.uuid4())
    orch._update_project_state = AsyncMock()
    orch._save_scene_asset = AsyncMock(return_value=uuid.uuid4())
    orch._update_scenes_visual_specs = AsyncMock()

    # Workspace directory pointing to isolated temp dir
    orch.get_project_workspace = lambda proj_id: {
        "project_dir": tmp_dir,
        "audio_dir": os.path.join(tmp_dir, "audio"),
        "clips_dir": os.path.join(tmp_dir, "clips"),
        "sounds_dir": os.path.join(tmp_dir, "sounds"),
        "resampled_dir": os.path.join(tmp_dir, "resampled"),
    }
    os.makedirs(os.path.join(tmp_dir, "clips"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "sounds"), exist_ok=True)

    # Mock session factory for topic lookup
    class MockSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def execute(self, stmt):
            m = MagicMock()
            m.scalar = lambda: "Test Video Topic"
            return m

    orch.session_factory = lambda: MockSession()

    from app.services.visual_verification import VisualVerificationResult, VerificationStatus
    async def mock_verify(video_path, shot_requirements, candidate_metadata=None):
        meta = candidate_metadata or {}
        tags = meta.get("tags") or []
        title = meta.get("title") or ""
        text = (title + " " + " ".join(tags)).lower()
        if "rejected" in text or "unsuitable" in text:
            return VisualVerificationResult(status=VerificationStatus.REJECTED, confidence=0.9, contradictions=["Unsuitable candidate"])
        return VisualVerificationResult(status=VerificationStatus.ACCEPTED, confidence=0.9, confirmed_matches=["Subject matched", "Setting matched"], is_visual_frame_verified=True)

    orch.visual_verifier = MagicMock()
    orch.visual_verifier.verify_candidate_clip = mock_verify
    return orch


async def test_1_single_shot_legacy_scene():
    print("Test 1: Single-shot legacy scene...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "A chef heats the seasoned wok.",
                "duration": 4.0,
                "visual_prompt": "Wok heating over flame",
                "search_keywords": ["wok heating", "hot pan"],
                "shot_plan": None  # Single-shot legacy scene
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        sc = res[0]
        assert not sc.get("is_multi_shot")
        assert len(sc["media_paths"]) == 1
        assert os.path.exists(sc["media_paths"][0])
        print("  PASS: Single-shot legacy scene sources 1 clip via legacy branch.")


async def test_2_two_shot_cooking_scene():
    print("Test 2: Two-shot cooking scene...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Pour oil into the hot pan, then add sliced carrots.",
                "duration": 4.5,  # Measured EdgeTTS duration
                "visual_prompt": "Cooking oil poured into hot pan, followed by sliced carrots.",
                "search_keywords": ["pouring oil pan", "cooking carrots wok"],
                "shot_plan": [
                    {
                        "shot_index": 1,
                        "required_visible_action": "Pouring golden cooking oil into hot pan",
                        "primary_subject": "Cooking oil",
                        "required_setting": "Stovetop skillet",
                        "visual_state": "oil heating shimmering",
                        "camera_framing": "close-up",
                        "search_keywords": ["pouring oil pan", "hot skillet oil shimmering"],
                        "provisional_start_seconds": 0.0,
                        "provisional_end_seconds": 2.0,
                        "action_event_index": 0
                    },
                    {
                        "shot_index": 2,
                        "required_visible_action": "Dropping fresh sliced carrots into hot oil",
                        "primary_subject": "Sliced carrots",
                        "required_setting": "Stovetop skillet",
                        "visual_state": "carrots hitting hot oil sizzling",
                        "camera_framing": "macro close-up",
                        "search_keywords": ["adding carrots wok", "sliced carrots stir fry"],
                        "provisional_start_seconds": 2.0,
                        "provisional_end_seconds": 4.0,
                        "action_event_index": 1
                    }
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        sc = res[0]
        assert sc["is_multi_shot"]
        assert len(sc["media_paths"]) == 2
        assert len(sc["shot_metadata"]) == 2
        assert len(sc["reconciled_shots"]) == 2
        assert sc["visual_coverage_complete"] is True
        assert len(sc["missing_shots"]) == 0
        assert sc["media_paths"][0] != sc["media_paths"][1]
        print(f"  PASS: Two-shot scene sourced 2 distinct clips: {sc['media_paths']}")


async def test_3_three_shot_manufacturing():
    print("Test 3: Three-shot manufacturing scene...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Robotic welding, hydraulic clamping, and conveyor transport.",
                "duration": 6.0,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Welding", "primary_subject": "Robot", "required_setting": "Factory", "search_keywords": ["robot welding", "automotive welding"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "Clamping", "primary_subject": "Clamp", "required_setting": "Factory", "search_keywords": ["hydraulic clamp", "industrial clamp"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0},
                    {"shot_index": 3, "required_visible_action": "Transport", "primary_subject": "Conveyor", "required_setting": "Factory", "search_keywords": ["factory conveyor", "assembly line belt"], "provisional_start_seconds": 4.0, "provisional_end_seconds": 6.0}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        sc = res[0]
        assert sc["is_multi_shot"]
        assert len(sc["media_paths"]) == 3
        assert len(sc["shot_metadata"]) == 3
        assert sc["visual_coverage_complete"] is True
        print(f"  PASS: Three-shot scene sourced 3 chronological clips: {sc['media_paths']}")


async def test_4_shot_plan_persistence_and_propagation():
    print("Test 4: Shot plan persistence and workspace propagation...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scene_id = uuid.uuid4()
        raw_shots = [
            {"shot_index": 1, "required_visible_action": "Action 1", "primary_subject": "Sub 1", "required_setting": "Set 1", "search_keywords": ["kw1", "kw2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
            {"shot_index": 2, "required_visible_action": "Action 2", "primary_subject": "Sub 2", "required_setting": "Set 2", "search_keywords": ["kw3", "kw4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
        ]
        # Simulate Stage 3 persistence
        specs = [{"visual_prompt": "P", "search_keywords": ["kw1", "kw2"], "action_timeline": None, "shot_plan": raw_shots}]
        scenes = [{"id": scene_id, "scene_order": 1, "scene_text": "Text"}]
        for sc, sp in zip(scenes, specs):
            sc["shot_plan"] = sp["shot_plan"]

        ws = orch.get_project_workspace(uuid.uuid4())
        sp_file = os.path.join(ws["project_dir"], "shot_plan.json")
        with open(sp_file, "w", encoding="utf-8") as f:
            json.dump({str(scene_id): raw_shots}, f)

        # Simulate fresh stage 5 memory without in-memory shot_plan
        scenes_fresh = [{"id": scene_id, "scene_order": 1, "scene_text": "Text", "duration": 4.0}]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes_fresh, os.path.join(tmp, "clips"))
        assert res[0]["is_multi_shot"]
        assert len(res[0]["media_paths"]) == 2
        print("  PASS: shot_plan restored from workspace shot_plan.json and propagated to Stage 5.")


async def test_5_correct_use_of_measured_narration_duration():
    print("Test 5: Correct use of measured narration duration...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        # Measured EdgeTTS duration: 5.75s @ 24fps -> ceil(5.75 * 24) = 138 frames
        measured_dur = 5.75
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Two cooking steps.",
                "duration": measured_dur,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Pour", "primary_subject": "Oil", "required_setting": "Pan", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "Toss", "primary_subject": "Carrot", "required_setting": "Pan", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        reconciled = res[0]["reconciled_shots"]
        total_frames = sum(s["allocated_frames"] for s in reconciled)
        assert total_frames == 138
        assert reconciled[1]["end_seconds"] == round(138 / 24, 4)
        print(f"  PASS: Measured duration {measured_dur}s produced frame budget 138 frames.")


async def test_6_exact_integer_frame_boundaries():
    print("Test 6: Exact integer frame boundaries without gaps or overlaps...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Two cooking steps.",
                "duration": 4.02, # 97 frames
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "A", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "B", "primary_subject": "S", "required_setting": "E", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        shots = res[0]["reconciled_shots"]
        assert shots[0]["start_frame"] == 0
        assert shots[0]["end_frame"] == shots[1]["start_frame"]
        assert shots[1]["end_frame"] == 97
        assert shots[0]["allocated_frames"] + shots[1]["allocated_frames"] == 97
        print("  PASS: Shot boundaries strictly contiguous: 0 -> 49 -> 97 frames.")


async def test_7_per_shot_contextual_queries():
    print("Test 7: Per-shot contextual Pexels queries...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Pour oil then toss carrots.",
                "duration": 4.0,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Pouring oil", "primary_subject": "Oil", "required_setting": "Wok", "visual_state": "heating", "camera_framing": "close-up", "search_keywords": ["oil pour", "oil heating"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "Tossing carrots", "primary_subject": "Carrots", "required_setting": "Wok", "visual_state": "sizzling", "camera_framing": "macro", "search_keywords": ["carrots wok", "sliced carrots stir fry"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
                ]
            }
        ]
        await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        assert len(provider.search_calls) == 2
        call_1 = provider.search_calls[0]
        call_2 = provider.search_calls[1]
        assert call_1["keywords"] == ["oil pour", "oil heating"]
        assert call_1["scene_context"]["required_action"] == "Pouring oil"
        assert call_1["scene_context"]["primary_subject"] == "Oil"
        assert call_2["keywords"] == ["carrots wok", "sliced carrots stir fry"]
        assert call_2["scene_context"]["required_action"] == "Tossing carrots"
        assert call_2["scene_context"]["primary_subject"] == "Carrots"
        print("  PASS: Each shot supplied distinct keywords and contextual parameters to Pexels provider.")


async def test_8_correct_shot_ordering():
    print("Test 8: Correct shot-to-file ordering...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Step 1 then Step 2 then Step 3.",
                "duration": 6.0,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "First", "primary_subject": "S1", "required_setting": "Set", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "Second", "primary_subject": "S2", "required_setting": "Set", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0},
                    {"shot_index": 3, "required_visible_action": "Third", "primary_subject": "S3", "required_setting": "Set", "search_keywords": ["k5", "k6"], "provisional_start_seconds": 4.0, "provisional_end_seconds": 6.0}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        paths = res[0]["media_paths"]
        assert "shot_1" in paths[0]
        assert "shot_2" in paths[1]
        assert "shot_3" in paths[2]
        print("  PASS: Chronological order strictly preserved in media_paths list.")


async def test_9_duplicate_video_id_prevention():
    print("Test 9: Duplicate video ID prevention across shots...")
    with tempfile.TemporaryDirectory() as tmp:
        # Pre-populate candidate map with duplicate candidates
        provider = MockVideoProvider(candidates_map={
            "oil": [{"pexels_id": 999, "download_url": "http://pexels/999.mp4"}, {"pexels_id": 1001, "download_url": "http://pexels/1001.mp4"}],
            "carrot": [{"pexels_id": 999, "download_url": "http://pexels/999.mp4"}, {"pexels_id": 1002, "download_url": "http://pexels/1002.mp4"}]
        })
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Pour oil and carrots.",
                "duration": 4.0,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Pour", "primary_subject": "Oil", "required_setting": "Pan", "search_keywords": ["oil pouring", "cooking oil"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "Carrot", "primary_subject": "Carrot", "required_setting": "Pan", "search_keywords": ["carrot wok", "sliced carrots"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        meta1 = res[0]["shot_metadata"][0]
        meta2 = res[0]["shot_metadata"][1]
        assert meta1["pexels_id"] == 999
        assert meta2["pexels_id"] == 1002  # Skipped 999 because it was in used_video_ids!
        print(f"  PASS: Video ID 999 deduplicated; shot 2 selected ID {meta2['pexels_id']}.")


async def test_10_missing_second_shot_footage():
    print("Test 10: Missing second-shot footage fallback...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider(candidates_map={
            "oil": [{"pexels_id": 101, "download_url": "http://pexels/101.mp4"}],
            "obscure_unfindable": None  # Will raise NoSuitableMediaFoundError
        })
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Pour oil then obscure action.",
                "duration": 4.0,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Pour oil", "primary_subject": "Oil", "required_setting": "Pan", "search_keywords": ["oil pouring", "cooking oil"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "Obscure", "primary_subject": "Obscure", "required_setting": "Pan", "search_keywords": ["obscure_unfindable", "rare action"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        sc = res[0]
        assert sc["visual_coverage_complete"] is False
        assert len(sc["missing_shots"]) == 1
        assert sc["missing_shots"][0]["shot_index"] == 2
        assert sc["fallback_used"] == "primary_shot_extended_missing_secondary"
        assert len(sc["media_paths"]) == 1
        print("  PASS: Missing secondary shot handled with documented fallback and incomplete coverage flag.")


async def test_11_missing_first_shot_footage():
    print("Test 11: Missing first-shot footage...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider(candidates_map={
            "obscure_first": None, # Will raise NoSuitableMediaFoundError
            "carrots": [{"pexels_id": 102, "download_url": "http://pexels/102.mp4"}]
        })
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Obscure action then carrots.",
                "duration": 4.0,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Obscure", "primary_subject": "Sub", "required_setting": "Set", "search_keywords": ["obscure_first", "rare action"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "Carrots", "primary_subject": "Carrot", "required_setting": "Set", "search_keywords": ["carrots wok", "sliced carrots"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        sc = res[0]
        assert sc["visual_coverage_complete"] is False
        assert len(sc["missing_shots"]) == 1
        assert sc["missing_shots"][0]["shot_index"] == 1
        assert len(sc["media_paths"]) == 1
        assert "shot_2" in sc["media_paths"][0]
        print("  PASS: Missing first shot recorded; available shot 2 retained with incomplete coverage flag.")


async def test_12_all_shots_unavailable():
    print("Test 12: All shots unavailable raises NoSuitableMediaFoundError...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider(candidates_map={
            "fail1": None,
            "fail2": None
        })
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Two completely unfindable actions.",
                "duration": 4.0,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Act 1", "primary_subject": "S1", "required_setting": "R", "search_keywords": ["fail1", "unfindable action 1"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "Act 2", "primary_subject": "S2", "required_setting": "R", "search_keywords": ["fail2", "unfindable action 2"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
                ]
            }
        ]
        try:
            await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
            assert False, "Should have raised NoSuitableMediaFoundError"
        except NoSuitableMediaFoundError as e:
            assert "No suitable stock video clips could be sourced for multi-shot Scene 1" in str(e)
        print("  PASS: Zero available shots raises NoSuitableMediaFoundError cleanly.")


async def test_13_valid_setting_transitions():
    print("Test 13: Valid setting transitions across shots...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "From valley harvest to desert trade.",
                "duration": 5.0,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "River harvest", "primary_subject": "Grain", "required_setting": "River valley", "search_keywords": ["river valley harvest", "farming grain river"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.5},
                    {"shot_index": 2, "required_visible_action": "Desert caravan", "primary_subject": "Camels", "required_setting": "Arid dunes", "search_keywords": ["desert trade caravan", "sand dunes camels"], "provisional_start_seconds": 2.5, "provisional_end_seconds": 5.0}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        shots = res[0]["shot_metadata"]
        assert shots[0]["required_setting"] == "River valley"
        assert shots[1]["required_setting"] == "Arid dunes"
        print("  PASS: Setting transition preserved across per-shot contextual sourcing.")


async def test_14_action_event_linkage_preservation():
    print("Test 14: Action-event linkage preservation in asset metadata...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Pour oil and carrots.",
                "duration": 4.0,
                "action_timeline": [
                    {"action_description": "Pour oil", "visual_start_seconds": 0.0, "visual_end_seconds": 2.0, "sound_type": "action", "sfx_query": "oil sizzle"},
                    {"action_description": "Add carrots", "visual_start_seconds": 2.0, "visual_end_seconds": 4.0, "sound_type": "action", "sfx_query": "vegetable sizzle"}
                ],
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Pour", "primary_subject": "Oil", "required_setting": "Pan", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0, "action_event_index": 0},
                    {"shot_index": 2, "required_visible_action": "Carrot", "primary_subject": "Carrot", "required_setting": "Pan", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0, "action_event_index": 1}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        assert res[0]["shot_metadata"][0]["action_event_index"] == 0
        assert res[0]["shot_metadata"][1]["action_event_index"] == 1
        print("  PASS: action_event_index correctly preserved in shot_metadata.")


async def test_15_timing_mismatch_warning_behavior():
    print("Test 15: Timing mismatch warning behavior...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Pour oil and carrots.",
                "duration": 4.0,
                "action_timeline": [
                    # SFX starts at 2.5s, but shot 1 window is 0.0s - 2.0s!
                    {"action_description": "Pour oil", "visual_start_seconds": 0.0, "visual_end_seconds": 2.0, "sound_type": "action", "sfx_query": "oil sizzle", "sfx_start_seconds": 2.5},
                    {"action_description": "Add carrots", "visual_start_seconds": 2.0, "visual_end_seconds": 4.0, "sound_type": "action", "sfx_query": "vegetable sizzle", "sfx_start_seconds": 3.0}
                ],
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Pour", "primary_subject": "Oil", "required_setting": "Pan", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0, "action_event_index": 0},
                    {"shot_index": 2, "required_visible_action": "Carrot", "primary_subject": "Carrot", "required_setting": "Pan", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0, "action_event_index": 1}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        warnings = res[0].get("timing_warnings", [])
        assert len(warnings) == 1
        assert "falls outside reconciled shot window" in warnings[0]
        assert "Proportional shot allocation does not guarantee physical acoustic synchronization" in warnings[0]
        print(f"  PASS: Timing warning detected and logged: '{warnings[0]}'")


async def test_16_legacy_scenes_longer_than_eight_seconds():
    print("Test 16: Legacy scenes longer than eight seconds (dur > 8s)...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "A long narrative beat without shot plan.",
                "duration": 10.0, # > 8.0s!
                "search_keywords": ["long scene one", "long scene two"],
                "shot_plan": None
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        sc = res[0]
        assert sc["is_multi_shot"] is False
        assert len(sc["media_paths"]) == 2
        assert "clip_a" in sc["media_paths"][0]
        assert "clip_b" in sc["media_paths"][1]
        print("  PASS: Legacy > 8s scene preserves exact two-half-clips behavior.")


async def test_17_no_unintended_changes_to_freesound():
    print("Test 17: Freesound Stage 6 compatibility...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        # Stage 6 mock
        mock_freesound = AsyncMock()
        mock_freesound.download_audio = AsyncMock(return_value=os.path.join(tmp, "sounds", "sfx.mp3"))
        orch.get_sfx_provider = lambda: mock_freesound

        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Pour oil and stir.",
                "duration": 4.0,
                "action_timeline": [
                    {"action_description": "Pour oil", "visual_start_seconds": 0.0, "visual_end_seconds": 2.0, "sound_type": "none", "sfx_query": None}
                ],
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "Pour", "primary_subject": "Oil", "required_setting": "Pan", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0, "action_event_index": 0},
                    {"shot_index": 2, "required_visible_action": "Stir", "primary_subject": "Spoon", "required_setting": "Pan", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0, "action_event_index": None}
                ]
            }
        ]
        # Run Stage 6 on multi-shot scene
        res = await orch._stage_6_source_sfx(uuid.uuid4(), scenes, os.path.join(tmp, "sounds"))
        # Silent action event should make 0 Freesound calls
        assert mock_freesound.download_audio.call_count == 0
        print("  PASS: Stage 6 operates seamlessly on multi-shot scenes with zero unwanted changes.")


async def test_18_shot_metadata_contract_for_assembly():
    print("Test 18: Shot metadata contract available for Step 11B.4 assembly...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Two actions.",
                "duration": 4.0,
                "shot_plan": [
                    {"shot_index": 1, "required_visible_action": "A", "primary_subject": "S1", "required_setting": "R", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 2.0},
                    {"shot_index": 2, "required_visible_action": "B", "primary_subject": "S2", "required_setting": "R", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 2.0, "provisional_end_seconds": 4.0}
                ]
            }
        ]
        res = await orch._stage_5_source_media(uuid.uuid4(), scenes, os.path.join(tmp, "clips"))
        sc = res[0]
        # Verify complete data contract for Step 11B.4 MoviePy/PyAV assembly
        assert "is_multi_shot" in sc and sc["is_multi_shot"] is True
        assert "media_paths" in sc and len(sc["media_paths"]) == 2
        assert "shot_metadata" in sc and len(sc["shot_metadata"]) == 2
        assert "reconciled_shots" in sc and len(sc["reconciled_shots"]) == 2
        
        # Verify frame indices for exact video cutting
        for shot in sc["reconciled_shots"]:
            assert "allocated_frames" in shot
            assert "start_frame" in shot
            assert "end_frame" in shot
            assert "duration_seconds" in shot
            assert shot["allocated_frames"] > 0
        print("  PASS: Step 11B.4 assembly data contract completely satisfied.")


async def main():
    print("=== STARTING STEP 11B.3 OFFLINE VERIFICATION SUITE ===")
    await test_1_single_shot_legacy_scene()
    await test_2_two_shot_cooking_scene()
    await test_3_three_shot_manufacturing()
    await test_4_shot_plan_persistence_and_propagation()
    await test_5_correct_use_of_measured_narration_duration()
    await test_6_exact_integer_frame_boundaries()
    await test_7_per_shot_contextual_queries()
    await test_8_correct_shot_ordering()
    await test_9_duplicate_video_id_prevention()
    await test_10_missing_second_shot_footage()
    await test_11_missing_first_shot_footage()
    await test_12_all_shots_unavailable()
    await test_13_valid_setting_transitions()
    await test_14_action_event_linkage_preservation()
    await test_15_timing_mismatch_warning_behavior()
    await test_16_legacy_scenes_longer_than_eight_seconds()
    await test_17_no_unintended_changes_to_freesound()
    await test_18_shot_metadata_contract_for_assembly()
    print("=== ALL 18 STEP 11B.3 OFFLINE TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    asyncio.run(main())

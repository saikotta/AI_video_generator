"""
Targeted Verification Suite: Manual Approval Gate & Accuracy Enforcement
Covers Task 1, Task 2, and Task 3 client demo sprint requirements:
1. Uncertain candidate blocked from automatic assembly.
2. Manually approved candidate accepted.
3. Rejected candidate excluded.
4. Approval tied to exact candidate and shot.
5. No unrelated fallback footage.
6. Approved assets reused without unnecessary downloads.
7. Missing vision provider handled safely.
8. No infinite retry loops.
"""

import os
import sys
import uuid
import json
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av
import numpy as np
from PIL import Image

from app.schemas.media import MediaAssetSchema
from app.services.providers.exceptions import (
    StockMediaError,
    NoSuitableMediaFoundError,
    UncertainFootageRequiresApprovalError
)
from app.services.pipeline_orchestrator import VideoPipelineOrchestrator
from app.services.visual_verification import (
    VisualVerifier,
    VerificationStatus,
    VisualVerificationResult,
    FrameExtractor
)


def create_synthetic_mp4(file_path: str, duration_sec: float = 2.0, fps: int = 24, width: int = 320, height: int = 240, color: tuple = (120, 150, 180)):
    """Creates a minimal valid synthetic MP4 file using PyAV for testing."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    container = av.open(file_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        # Slightly alter color per frame to simulate motion
        c = (
            min(255, max(0, color[0] + (i % 20))),
            min(255, max(0, color[1] + (i % 15))),
            min(255, max(0, color[2] + (i % 10)))
        )
        img = Image.new("RGB", (width, height), color=c)
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return file_path


class MockTargetedVideoProvider:
    def __init__(self, candidates_per_shot: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.candidates_per_shot = candidates_per_shot or {}
        self.search_calls = []
        self.download_calls = []

    async def search_media_candidates(
        self,
        keywords: List[str],
        target_duration: float = 4.0,
        used_video_ids: Optional[Set[int]] = None,
        scene_context: Optional[Dict[str, Any]] = None,
        max_candidates: int = 3
    ) -> List[MediaAssetSchema]:
        self.search_calls.append({
            "keywords": list(keywords),
            "target_duration": target_duration,
            "used_video_ids": set(used_video_ids) if used_video_ids else set(),
            "scene_context": dict(scene_context) if scene_context else None
        })

        shot_idx = (scene_context or {}).get("shot_index", 1)
        scene_order = (scene_context or {}).get("scene_order", 1)
        key = f"scene_{scene_order}_shot_{shot_idx}"

        raw_cands = self.candidates_per_shot.get(key, [])
        res = []
        for c in raw_cands[:max_candidates]:
            p_id = c.get("pexels_id")
            if used_video_ids and p_id in used_video_ids:
                continue
            res.append(MediaAssetSchema(
                download_url=c.get("download_url", f"https://mock.pexels.com/video_{p_id}.mp4"),
                source_url=f"https://pexels.com/video/{p_id}",
                duration=target_duration,
                metadata=c
            ))
        return res

    async def download_media(self, download_url: str, output_dir: str, filename_prefix: str) -> str:
        self.download_calls.append({"url": download_url, "prefix": filename_prefix})
        dest = os.path.join(output_dir, f"{filename_prefix}.mp4")
        if not os.path.exists(dest):
            create_synthetic_mp4(dest, duration_sec=2.0)
        return dest


def make_test_orchestrator(provider, tmp_dir):
    orch = VideoPipelineOrchestrator(
        session_factory=None,
        stock_media_provider=provider,
        base_media_dir=tmp_dir
    )
    def get_ws(pid):
        pdir = os.path.join(tmp_dir, str(pid))
        os.makedirs(pdir, exist_ok=True)
        os.makedirs(os.path.join(pdir, "clips"), exist_ok=True)
        os.makedirs(os.path.join(pdir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(pdir, "sounds"), exist_ok=True)
        os.makedirs(os.path.join(pdir, "contact_sheets"), exist_ok=True)
        return {
            "project_dir": pdir,
            "audio_dir": os.path.join(pdir, "audio"),
            "clips_dir": os.path.join(pdir, "clips"),
            "sounds_dir": os.path.join(pdir, "sounds"),
            "resampled_dir": os.path.join(pdir, "resampled"),
            "output_path": os.path.join(pdir, f"video_{pid}.mp4")
        }
    orch.get_project_workspace = get_ws

    class MockSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def execute(self, stmt):
            m = MagicMock()
            m.scalar = lambda: "Test Topic"
            return m
        async def commit(self):
            pass

    orch.session_factory = lambda: MockSession()
    orch._save_scene_asset = AsyncMock(return_value=uuid.uuid4())
    orch._record_pipeline_job = AsyncMock(return_value=uuid.uuid4())
    orch._update_project_state = AsyncMock()
    return orch


async def test_1_uncertain_candidate_blocked_from_automatic_assembly():
    print("Test 1: Uncertain candidate blocked from automatic assembly...")
    with tempfile.TemporaryDirectory() as tmp:
        # Candidate has generic keywords and no explicit contradictions, resulting in UNCERTAIN/MANUAL_REVIEW
        cands = {
            "scene_1_shot_1": [
                {
                    "pexels_id": 1001,
                    "title": "Kitchen atmosphere cooking",
                    "tags": ["kitchen", "food", "chef"],
                    "width": 1920,
                    "height": 1080,
                    "duration": 4.0
                }
            ]
        }
        provider = MockTargetedVideoProvider(cands)
        orch = make_test_orchestrator(provider, tmp)
        proj_id = uuid.uuid4()

        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Toss the freshly chopped bell peppers in the smoking wok.",
                "duration": 3.0,
                "shot_plan": [
                    {
                        "shot_index": 1,
                        "provisional_start_seconds": 0.0,
                        "provisional_end_seconds": 3.0,
                        "required_visible_action": "Tossing sliced bell peppers in a hot wok with flame",
                        "primary_subject": "bell peppers in wok",
                        "required_setting": "kitchen wok station",
                        "search_keywords": ["tossing bell peppers wok", "vegetable stir fry"]
                    }
                ]
            }
        ]

        # Sourcing must raise UncertainFootageRequiresApprovalError
        raised = False
        try:
            await orch._stage_5_source_media(proj_id, scenes, os.path.join(tmp, "clips"), topic="Vegetable stir-fry")
        except UncertainFootageRequiresApprovalError as e:
            raised = True
            assert len(e.pending_approvals) >= 1
            entry = e.pending_approvals[0]
            assert entry["candidate_video_id"] == 1001
            assert "scene_1_shot_1" in entry["shot_key"]

        assert raised, "Expected UncertainFootageRequiresApprovalError was NOT raised!"

        # Verify contact sheet was generated
        ws = orch.get_project_workspace(proj_id)
        cs_path = os.path.join(ws["project_dir"], "contact_sheets", "scene_1_shot_1_cand_1001.jpg")
        assert os.path.exists(cs_path), f"Contact sheet not found at {cs_path}"
        assert os.path.getsize(cs_path) > 500, "Contact sheet is empty"

        # Verify recorded in pending reviews registry
        pending = orch.get_pending_approvals(proj_id)
        assert len(pending) == 1
        assert pending[0]["candidate_video_id"] == 1001

        print("  PASS: Uncertain candidate successfully blocked and contact sheet generated.")


async def test_2_manually_approved_candidate_accepted():
    print("Test 2: Manually approved candidate accepted...")
    with tempfile.TemporaryDirectory() as tmp:
        cands = {
            "scene_1_shot_1": [
                {
                    "pexels_id": 1002,
                    "title": "Chef tossing vegetables",
                    "tags": ["chef", "cooking"],
                    "width": 1920,
                    "height": 1080,
                    "duration": 4.0
                }
            ]
        }
        provider = MockTargetedVideoProvider(cands)
        orch = make_test_orchestrator(provider, tmp)
        proj_id = uuid.uuid4()

        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Toss the vegetables in the pan.",
                "duration": 3.0,
                "shot_plan": [
                    {
                        "shot_index": 1,
                        "provisional_start_seconds": 0.0,
                        "provisional_end_seconds": 3.0,
                        "required_visible_action": "Tossing sliced vegetables",
                        "primary_subject": "vegetables in pan",
                        "required_setting": "kitchen",
                        "search_keywords": ["tossing vegetables", "wok cooking"]
                    }
                ]
            }
        ]

        # 1. Run initially: blocked as uncertain
        ws = orch.get_project_workspace(proj_id)
        try:
            await orch._stage_5_source_media(proj_id, scenes, ws["clips_dir"], topic="Vegetable stir-fry")
        except UncertainFootageRequiresApprovalError:
            pass

        # 2. Explicitly approve candidate 1002
        orch.approve_candidate(
            project_id=proj_id,
            scene_order=1,
            shot_index=1,
            candidate_video_id=1002,
            reviewer="client_qa",
            notes="Verified acceptable wok motion"
        )

        # 3. Re-run: must proceed successfully with is_manually_approved=True
        res = await orch._stage_5_source_media(proj_id, scenes, ws["clips_dir"], topic="Vegetable stir-fry")
        assert len(res) == 1
        meta = res[0]["shot_metadata"][0]
        assert meta["pexels_id"] == 1002
        assert meta.get("is_manually_approved") is True
        assert meta["verification_status"] == VerificationStatus.ACCEPTED.value

        print("  PASS: Manually approved candidate successfully accepted into assembly.")


async def test_3_rejected_candidate_excluded():
    print("Test 3: Rejected candidate excluded...")
    with tempfile.TemporaryDirectory() as tmp:
        # Candidate 1003 is rejected, candidate 1004 is uncertain
        cands = {
            "scene_1_shot_1": [
                {
                    "pexels_id": 1003,
                    "title": "Bad candidate",
                    "tags": ["unrelated"],
                    "width": 1920,
                    "height": 1080,
                    "duration": 4.0
                },
                {
                    "pexels_id": 1004,
                    "title": "Alternative candidate",
                    "tags": ["vegetables"],
                    "width": 1920,
                    "height": 1080,
                    "duration": 4.0
                }
            ]
        }
        provider = MockTargetedVideoProvider(cands)
        orch = make_test_orchestrator(provider, tmp)
        proj_id = uuid.uuid4()

        # Reject candidate 1003 ahead of time
        orch.reject_candidate(
            project_id=proj_id,
            scene_order=1,
            shot_index=1,
            candidate_video_id=1003,
            reviewer="qa",
            reason="Shows garden instead of wok"
        )

        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Toss the vegetables in the pan.",
                "duration": 3.0,
                "shot_plan": [
                    {
                        "shot_index": 1,
                        "provisional_start_seconds": 0.0,
                        "provisional_end_seconds": 3.0,
                        "required_visible_action": "Tossing vegetables",
                        "primary_subject": "vegetables",
                        "required_setting": "kitchen",
                        "search_keywords": ["tossing vegetables", "wok cooking"]
                    }
                ]
            }
        ]

        # Sourcing will evaluate candidate 1004 (1003 skipped)
        ws = orch.get_project_workspace(proj_id)
        try:
            await orch._stage_5_source_media(proj_id, scenes, ws["clips_dir"], topic="Cooking")
        except UncertainFootageRequiresApprovalError as e:
            # Candidate 1004 should be the one pending, NOT 1003
            assert e.pending_approvals[0]["candidate_video_id"] == 1004

        # Confirm 1003 was never accepted or in pending
        pending = orch.get_pending_approvals(proj_id)
        assert not any(p["candidate_video_id"] == 1003 for p in pending)

        print("  PASS: Rejected candidate strictly excluded from consideration.")


async def test_4_approval_tied_to_exact_candidate_and_shot():
    print("Test 4: Approval tied to exact candidate and shot...")
    with tempfile.TemporaryDirectory() as tmp:
        provider = MockTargetedVideoProvider()
        orch = make_test_orchestrator(provider, tmp)
        proj_id = uuid.uuid4()

        # Approve video 2001 for Scene 1 Shot 1
        orch.approve_candidate(proj_id, scene_order=1, shot_index=1, candidate_video_id=2001)

        # 1. Matching candidate & shot: True
        assert orch.is_candidate_approved(proj_id, scene_order=1, shot_index=1, candidate_video_id=2001)

        # 2. Different candidate same shot: False
        assert not orch.is_candidate_approved(proj_id, scene_order=1, shot_index=1, candidate_video_id=2002)

        # 3. Same candidate different shot (Shot 2): False
        assert not orch.is_candidate_approved(proj_id, scene_order=1, shot_index=2, candidate_video_id=2001)

        # 4. Same candidate different scene (Scene 2 Shot 1): False
        assert not orch.is_candidate_approved(proj_id, scene_order=2, shot_index=1, candidate_video_id=2001)

        # 5. Different project: False
        assert not orch.is_candidate_approved(uuid.uuid4(), scene_order=1, shot_index=1, candidate_video_id=2001)

        print("  PASS: Approval is strictly scoped to exact project, scene, shot, and candidate ID.")


async def test_5_no_unrelated_fallback_footage():
    print("Test 5: No unrelated fallback footage...")
    with tempfile.TemporaryDirectory() as tmp:
        # Multi-shot scene where Shot 1 has no suitable clips and is uncertain
        cands = {
            "scene_1_shot_1": [
                {
                    "pexels_id": 3001,
                    "title": "Unrelated footage",
                    "tags": ["beach", "waves"],
                    "width": 1920,
                    "height": 1080,
                    "duration": 4.0
                }
            ]
        }
        provider = MockTargetedVideoProvider(cands)
        orch = make_test_orchestrator(provider, tmp)
        proj_id = uuid.uuid4()

        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Microscopic view of seed cellular mitosis.",
                "duration": 4.0,
                "shot_plan": [
                    {
                        "shot_index": 1,
                        "provisional_start_seconds": 0.0,
                        "provisional_end_seconds": 4.0,
                        "required_visible_action": "Microscopic cell division inside germinating seed embryo",
                        "primary_subject": "plant cell chromosomes dividing",
                        "required_setting": "laboratory microscope",
                        "search_keywords": ["plant mitosis", "cellular biology"]
                    }
                ]
            }
        ]

        # Beach footage must NEVER be silently used for microscopic cell division
        try:
            ws = orch.get_project_workspace(proj_id)
            await orch._stage_5_source_media(proj_id, scenes, ws["clips_dir"], topic="Seed germination")
            assert False, "Pipeline should have blocked unrelated footage!"
        except (UncertainFootageRequiresApprovalError, NoSuitableMediaFoundError) as e:
            # Expected block
            pass

        print("  PASS: Unrelated footage blocked; no misleading stock footage accepted.")


async def test_6_approved_assets_reused_without_unnecessary_downloads():
    print("Test 6: Approved assets reused without unnecessary downloads...")
    with tempfile.TemporaryDirectory() as tmp:
        cands = {
            "scene_1_shot_1": [
                {
                    "pexels_id": 4001,
                    "title": "Vegetable stir fry",
                    "tags": ["wok"],
                    "width": 1920,
                    "height": 1080,
                    "duration": 4.0
                }
            ]
        }
        provider = MockTargetedVideoProvider(cands)
        orch = make_test_orchestrator(provider, tmp)
        proj_id = uuid.uuid4()
        ws = orch.get_project_workspace(proj_id)
        clips_dir = ws["clips_dir"]

        # Pre-place synthetic clip on disk
        clip_file = os.path.join(clips_dir, "scene_1_shot_1_4001.mp4")
        create_synthetic_mp4(clip_file, duration_sec=2.5)
        initial_downloads = len(provider.download_calls)

        # Pre-approve candidate 4001 with local path recorded
        orch.approve_candidate(
            project_id=proj_id,
            scene_order=1,
            shot_index=1,
            candidate_video_id=4001,
            reviewer="qa",
            notes="Existing verified asset"
        )
        # Update registry with local path
        reg = orch.load_approval_registry(proj_id)
        reg["approved_candidates"]["scene_1_shot_1"]["local_clip_path"] = clip_file
        orch.save_approval_registry(proj_id, reg)

        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Toss vegetables in wok.",
                "duration": 2.5,
                "shot_plan": [
                    {
                        "shot_index": 1,
                        "provisional_start_seconds": 0.0,
                        "provisional_end_seconds": 2.5,
                        "required_visible_action": "Tossing vegetables in wok",
                        "primary_subject": "vegetables in wok",
                        "required_setting": "kitchen",
                        "search_keywords": ["vegetable stir fry", "wok cooking"]
                    }
                ]
            }
        ]

        # Run sourcing: must reuse local clip directly with ZERO download calls
        res = await orch._stage_5_source_media(proj_id, scenes, clips_dir, topic="Stir fry")
        assert len(provider.download_calls) == initial_downloads, "Unnecessary download occurred!"
        assert res[0]["media_paths"][0] == clip_file

        print("  PASS: Approved footage reused directly from disk with zero downloads.")


async def test_7_missing_vision_provider_handled_safely():
    print("Test 7: Missing vision provider handled safely...")
    with tempfile.TemporaryDirectory() as tmp:
        clip_path = os.path.join(tmp, "test_clip.mp4")
        create_synthetic_mp4(clip_path, duration_sec=2.0)

        # VisualVerifier initializes without vision model credentials
        verifier = VisualVerifier(num_sample_frames=3)

        # Candidate with ambiguous/generic metadata where vision model would be needed to verify
        res = await verifier.verify_candidate_clip(
            video_path=clip_path,
            shot_requirements={
                "required_action": "Condensation forming on cold glass surface",
                "primary_subject": "condensation droplets",
                "required_setting": "glass surface",
                "search_keywords": ["condensation forming", "cold glass"]
            },
            candidate_metadata={
                "pexels_id": 5001,
                "title": "Window pane surface in room",
                "tags": ["window", "glass", "room", "indoor"]
            }
        )
        print("  Candidate evaluation outcome:", res.status, "| Evidence:", res.evidence[-1])
        assert res.status in (VerificationStatus.UNCERTAIN, VerificationStatus.MANUAL_REVIEW_REQUIRED)
        assert res.is_manually_approved is False
        assert any("not confirmed" in ev.lower() or "uncertain" in ev.lower() or "review" in ev.lower() for ev in res.evidence)

        # Contact sheet generates without error for manual gate review
        cs_path = os.path.join(tmp, "contact_sheet.jpg")
        verifier.generate_contact_sheet(
            video_path=clip_path,
            output_path=cs_path,
            shot_requirements={"required_action": "Condensation on glass", "scene_order": 1, "shot_index": 1},
            candidate_id=5001,
            evidence=res.evidence
        )
        assert os.path.exists(cs_path)
        assert os.path.getsize(cs_path) > 1000

        print("  PASS: Missing vision provider falls back safely to manual review and contact sheet.")


async def test_8_no_infinite_retry_loops():
    print("Test 8: No infinite retry loops...")
    with tempfile.TemporaryDirectory() as tmp:
        cands = {
            "scene_1_shot_1": [
                {
                    "pexels_id": 6001,
                    "title": "Indeterminate clip",
                    "tags": ["general"],
                    "width": 1920,
                    "height": 1080,
                    "duration": 3.0
                }
            ]
        }
        provider = MockTargetedVideoProvider(cands)
        orch = make_test_orchestrator(provider, tmp)
        proj_id = uuid.uuid4()

        scenes = [
            {
                "id": uuid.uuid4(),
                "scene_order": 1,
                "scene_text": "Plant root hair osmosis absorption.",
                "duration": 3.0,
                "shot_plan": [
                    {
                        "shot_index": 1,
                        "provisional_start_seconds": 0.0,
                        "provisional_end_seconds": 3.0,
                        "required_visible_action": "Osmosis fluid transfer across plant root membrane",
                        "primary_subject": "root hair cells",
                        "required_setting": "cellular soil",
                        "search_keywords": ["osmosis plant root cells", "plant cellular biology"]
                    }
                ]
            }
        ]

        # Repeated calls without approval must deterministically halt with UncertainFootageRequiresApprovalError
        ws = orch.get_project_workspace(proj_id)
        for i in range(2):
            try:
                await orch._stage_5_source_media(proj_id, scenes, ws["clips_dir"], topic="Botany")
                assert False, f"Iteration {i} did not halt as expected!"
            except UncertainFootageRequiresApprovalError:
                pass

        # Total search calls should be bounded (not looped)
        assert len(provider.search_calls) == 2

        print("  PASS: Clean deterministic termination; no infinite retry loops.")


async def main():
    print("=" * 70)
    print("RUNNING TARGETED MANUAL APPROVAL GATE VERIFICATION SUITE")
    print("=" * 70)

    await test_1_uncertain_candidate_blocked_from_automatic_assembly()
    await test_2_manually_approved_candidate_accepted()
    await test_3_rejected_candidate_excluded()
    await test_4_approval_tied_to_exact_candidate_and_shot()
    await test_5_no_unrelated_fallback_footage()
    await test_6_approved_assets_reused_without_unnecessary_downloads()
    await test_7_missing_vision_provider_handled_safely()
    await test_8_no_infinite_retry_loops()

    print("=" * 70)
    print("ALL 8 TARGETED MANUAL APPROVAL GATE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

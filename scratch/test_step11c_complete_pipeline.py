"""
Step 11C Offline Verification Suite
Complete Offline Multi-Shot Pipeline Integration and Render Verification

Tests:
1. Complete seven-stage pipeline execution on synthetic multi-action project.
2. Encoded MP4 container & stream parameters (1280x720, 24 FPS, H.264, AAC).
3. Exact frame-boundary inspection around all shot cuts (straight cuts, zero blend).
4. Subtitle continuity and narration alignment across cuts.
5. Missing first shot recovery (never shifts shot 2).
6. Missing middle shot recovery (extends preceding shot).
7. Missing final shot recovery (extends preceding shot).
8. All shots unavailable failure policy (NoSuitableMediaFoundError).
9. Ambiguous shot-to-file mapping error.
10. Multi-scene integration: multi-shot scene with inter-scene crossfade to legacy scene.
11. Windows decoder resource cleanup and lock release.
"""

import os
import sys
import gc
import uuid
import json
import shutil
import tempfile
import asyncio
import unittest
import numpy as np
import av
import wave
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from unittest.mock import AsyncMock, MagicMock
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models.enums import ProjectStatus, AssetType, JobStatus
from app.schemas.media import MediaAssetSchema
from app.services.providers.exceptions import NoSuitableMediaFoundError
from app.services.pipeline_orchestrator import VideoPipelineOrchestrator
from app.services.video_assembly import (
    assemble_video,
    compute_frame_exact_duration,
    close_visual_data,
    load_scene_visual_data,
    get_single_visual_frame,
    render_scene_frame,
    map_shots_to_media_files,
)

# Colors for the 4 synthetic shots
COLOR_SHOT_1 = (255, 200, 0)   # Oil pouring: Golden Yellow
COLOR_SHOT_2 = (255, 100, 0)   # Carrots added: Bright Orange
COLOR_SHOT_3 = (0, 180, 0)     # Vegetables stirring: Green
COLOR_SHOT_4 = (180, 0, 80)    # Finished dish served: Crimson

def create_synthetic_mp4_clip(
    path: str,
    color: tuple,
    duration: float = 2.0,
    fps: int = 24,
    size: tuple = (1280, 720)
) -> str:
    """Creates a valid H.264 MP4 video file with PyAV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    container = av.open(path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = size[0]
    stream.height = size[1]
    stream.pix_fmt = 'yuv420p'
    num_frames = int(round(duration * fps))

    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:, :] = color

    for _ in range(num_frames):
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path

def create_synthetic_wav_audio(
    path: str,
    duration: float = 2.0,
    frequency: float = 440.0,
    sample_rate: int = 44100
) -> str:
    """Creates a valid 16-bit stereo PCM WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    num_samples = int(round(duration * sample_rate))
    t = np.linspace(0, duration, num_samples, endpoint=False)
    tone = (np.sin(2 * np.pi * frequency * t) * 8000).astype(np.int16)
    stereo = np.column_stack([tone, tone])
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())
    return path


class MockScriptProvider:
    """Mocked Groq Script Provider delivering valid multi-shot plans."""
    def __init__(self, shot_plan: Optional[List[Dict[str, Any]]] = None, action_timeline: Optional[List[Dict[str, Any]]] = None):
        self.shot_plan = shot_plan
        self.action_timeline = action_timeline

    async def generate_script(self, topic: str, target_duration: float) -> str:
        return "Pour oil into a hot pan, then add sliced carrots. Stir the vegetables gently before serving."

    async def split_into_scenes(self, script: str, target_duration: float) -> List[Dict[str, Any]]:
        return [{
            "scene_order": 1,
            "scene_text": script,
            "estimated_duration": target_duration
        }]

    async def split_script_into_scenes(self, script: str, target_duration: float) -> List[Dict[str, Any]]:
        return await self.split_into_scenes(script, target_duration)

    async def generate_visual_specification(self, scene_text: str) -> Dict[str, Any]:
        return {
            "visual_prompt": "Cooking stir-fry in a hot wok pan",
            "search_keywords": ["stir fry", "cooking pan", "fresh vegetables"],
            "action_timeline": self.action_timeline,
            "shot_plan": self.shot_plan
        }


class MockTTSProvider:
    """Mocked EdgeTTS Provider generating synthetic voiceover audio."""
    def __init__(self, measured_duration: float = 6.0):
        self.measured_duration = measured_duration

    async def generate_audio(self, text: str, output_path: str) -> float:
        create_synthetic_wav_audio(output_path, duration=self.measured_duration, frequency=300.0)
        return self.measured_duration


class MockVideoProvider:
    """Mocked Pexels Provider generating colored synthetic clips."""
    def __init__(self, fail_shots: Optional[Set[int]] = None):
        self.fail_shots = fail_shots or set()
        self.shot_colors = {
            1: COLOR_SHOT_1,
            2: COLOR_SHOT_2,
            3: COLOR_SHOT_3,
            4: COLOR_SHOT_4,
        }

    async def search_media(
        self,
        keywords: List[str],
        target_duration: float = 4.0,
        used_video_ids: Optional[Set[int]] = None,
        scene_context: Optional[Dict[str, Any]] = None
    ) -> MediaAssetSchema:
        shot_idx = scene_context.get("shot_index", 1) if scene_context else 1
        if shot_idx in self.fail_shots:
            raise NoSuitableMediaFoundError(f"Simulated media sourcing failure for shot {shot_idx}")

        cand_id = 1000 + shot_idx
        return MediaAssetSchema(
            download_url=f"https://pexels.com/v/{cand_id}.mp4",
            source_url=f"https://pexels.com/video/{cand_id}",
            duration=target_duration,
            metadata={
                "pexels_id": cand_id,
                "width": 1280,
                "height": 720,
                "duration": target_duration,
                "shot_index": shot_idx,
            }
        )

    async def download_media(self, url: str, output_dir: str, filename_prefix: str) -> str:
        # Determine shot index from filename prefix
        color = (128, 128, 128)
        for s_idx, c in self.shot_colors.items():
            if f"shot_{s_idx}" in filename_prefix:
                color = c
                break

        out_path = os.path.join(output_dir, f"{filename_prefix}.mp4")
        create_synthetic_mp4_clip(out_path, color=color, duration=2.5, fps=24, size=(1280, 720))
        return out_path


class MockFreesoundProvider:
    """Mocked Freesound Provider generating synthetic action SFX."""
    async def search_segment_action_sfx(self, keyword: str, segment_duration: float = 1.0, used_sound_ids: Optional[Set[int]] = None) -> Optional[Dict[str, Any]]:
        return {
            "freesound_id": 9991,
            "name": "sizzling_pan",
            "download_url": "https://freesound.org/s/9991.wav",
            "license": "CC0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "action_name": "sizzle",
        }

    async def download_audio(self, url: str, output_dir: str, filename_prefix: str) -> str:
        out_path = os.path.join(output_dir, f"{filename_prefix}.wav")
        create_synthetic_wav_audio(out_path, duration=1.0, frequency=800.0)
        return out_path


def build_test_orchestrator(
    tmp_dir: str,
    script_provider: MockScriptProvider,
    tts_provider: MockTTSProvider,
    video_provider: MockVideoProvider,
    freesound_provider: MockFreesoundProvider,
    topic: str = "How to prepare a simple vegetable stir-fry",
    duration: float = 6.0
) -> VideoPipelineOrchestrator:
    """Instantiates a VideoPipelineOrchestrator with all external providers mocked."""
    orch = VideoPipelineOrchestrator()
    orch.base_media_dir = tmp_dir
    orch.get_script_provider = lambda: script_provider
    orch.get_tts_provider = lambda: tts_provider
    orch.get_stock_media_provider = lambda: video_provider
    orch.get_sound_effects_provider = lambda: freesound_provider

    from app.services.visual_verification import VisualVerificationResult, VerificationStatus
    async def mock_verify(video_path, shot_requirements, candidate_metadata=None):
        return VisualVerificationResult(
            status=VerificationStatus.ACCEPTED,
            confidence=0.9,
            confirmed_matches=["Subject verified", "Action verified"],
            is_visual_frame_verified=True
        )
    orch.visual_verifier = MagicMock()
    orch.visual_verifier.verify_candidate_clip = mock_verify

    # In-memory mock database & persistence operations
    orch._record_pipeline_job = AsyncMock(return_value=uuid.uuid4())
    orch._update_project_state = AsyncMock()
    orch._save_scene_asset = AsyncMock(return_value=uuid.uuid4())
    orch._update_scenes_visual_specs = AsyncMock()

    async def mock_save_scenes(proj_id, raw_scenes):
        created = []
        for s in raw_scenes:
            sc_dict = dict(s)
            sc_dict["id"] = uuid.uuid4()
            sc_dict["project_id"] = proj_id
            created.append(sc_dict)
        return created

    orch._save_initial_scenes = mock_save_scenes

    # Workspace directory pointing to isolated temp dir
    orch.get_project_workspace = lambda proj_id: {
        "project_dir": tmp_dir,
        "audio_dir": os.path.join(tmp_dir, "audio"),
        "clips_dir": os.path.join(tmp_dir, "clips"),
        "sounds_dir": os.path.join(tmp_dir, "sounds"),
        "resampled_dir": os.path.join(tmp_dir, "resampled"),
        "output_path": os.path.join(tmp_dir, f"video_{proj_id}.mp4"),
    }

    class MockProjectRow:
        def __init__(self):
            self.id = uuid.uuid4()
            self.topic = topic
            self.desired_duration_seconds = duration
            self.status = ProjectStatus.QUEUED

    class MockSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def execute(self, stmt):
            m = MagicMock()
            m.scalar = lambda: topic
            m.scalars = lambda: m
            m.first = lambda: MockProjectRow()
            m.scalar_one_or_none = lambda: MockProjectRow()
            return m
        async def commit(self):
            pass

    orch.session_factory = lambda: MockSession()
    return orch


class TestStep11CCompletePipeline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="step11c_integration_")
        self.fps = 24

        # 4-shot plan definition
        self.shot_plan = [
            {
                "shot_index": 1,
                "required_visible_action": "Pouring cooking oil into hot pan",
                "primary_subject": "cooking oil",
                "required_setting": "kitchen",
                "visual_state": "liquid oil spreading",
                "camera_framing": "close-up",
                "search_keywords": ["pour oil", "hot pan"],
                "provisional_start_seconds": 0.0,
                "provisional_end_seconds": 1.5,
                "action_event_index": 0,
            },
            {
                "shot_index": 2,
                "required_visible_action": "Adding sliced orange carrots into sizzling pan",
                "primary_subject": "sliced carrots",
                "required_setting": "kitchen",
                "visual_state": "carrots sizzling",
                "camera_framing": "medium-shot",
                "search_keywords": ["add carrots", "vegetables pan"],
                "provisional_start_seconds": 1.5,
                "provisional_end_seconds": 3.0,
                "action_event_index": 1,
            },
            {
                "shot_index": 3,
                "required_visible_action": "Stirring mixed vegetables in pan with spatula",
                "primary_subject": "mixed vegetables",
                "required_setting": "kitchen",
                "visual_state": "steam rising",
                "camera_framing": "over-the-shoulder",
                "search_keywords": ["stir vegetables", "cooking spatula"],
                "provisional_start_seconds": 3.0,
                "provisional_end_seconds": 4.5,
                "action_event_index": 2,
            },
            {
                "shot_index": 4,
                "required_visible_action": "Serving freshly cooked stir-fry dish onto white plate",
                "primary_subject": "cooked stir-fry dish",
                "required_setting": "kitchen dining table",
                "visual_state": "finished dish served",
                "camera_framing": "wide-shot",
                "search_keywords": ["serving food", "plate stir fry"],
                "provisional_start_seconds": 4.5,
                "provisional_end_seconds": 6.0,
                "action_event_index": None,
            },
        ]

        self.action_timeline = [
            {
                "action_description": "drizzle cooking oil",
                "sound_type": "foley",
                "sfx_query": "oil pour sizzle",
                "visual_start_seconds": 0.0,
                "visual_end_seconds": 1.5,
                "sfx_start_seconds": 0.2,
                "sfx_duration_seconds": 1.0,
                "sfx_volume": 0.4,
            },
            {
                "action_description": "toss carrots into pan",
                "sound_type": "foley",
                "sfx_query": "vegetable sizzle",
                "visual_start_seconds": 1.5,
                "visual_end_seconds": 3.0,
                "sfx_start_seconds": 1.6,
                "sfx_duration_seconds": 1.2,
                "sfx_volume": 0.5,
            },
            {
                "action_description": "stir vegetables with spatula",
                "sound_type": "foley",
                "sfx_query": "spatula pan scrape",
                "visual_start_seconds": 3.0,
                "visual_end_seconds": 4.5,
                "sfx_start_seconds": 3.1,
                "sfx_duration_seconds": 1.0,
                "sfx_volume": 0.35,
            },
        ]

    def tearDown(self):
        gc.collect()
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    # =========================================================================
    # Test 1: Complete Seven-Stage Pipeline Execution, Stream & Boundary Verification
    # =========================================================================
    async def test_01_complete_seven_stage_pipeline_and_container(self):
        script_p = MockScriptProvider(self.shot_plan, self.action_timeline)
        tts_p = MockTTSProvider(measured_duration=6.0)
        video_p = MockVideoProvider()
        sfx_p = MockFreesoundProvider()

        orch = build_test_orchestrator(self.test_dir, script_p, tts_p, video_p, sfx_p, duration=6.0)
        proj_id = uuid.uuid4()

        final_mp4 = await orch.run_pipeline(proj_id)
        self.assertTrue(os.path.exists(final_mp4), f"Output MP4 missing: {final_mp4}")
        self.assertGreater(os.path.getsize(final_mp4), 10000, "MP4 file is empty or suspiciously small")

        # Container & Stream Inspection via PyAV
        container = av.open(final_mp4)
        try:
            self.assertEqual(len(container.streams.video), 1, "Expected exactly 1 video stream")
            self.assertEqual(len(container.streams.audio), 1, "Expected exactly 1 audio stream")

            v_stream = container.streams.video[0]
            a_stream = container.streams.audio[0]

            # 1. Resolution
            self.assertEqual(v_stream.width, 1280)
            self.assertEqual(v_stream.height, 720)

            # 2. Frame Rate
            self.assertEqual(float(v_stream.average_rate), 24.0)

            # 3. Video Codec
            self.assertEqual(v_stream.codec_context.name, "h264")

            # 4. Audio Codec & Sample Rate
            self.assertEqual(a_stream.codec_context.name, "aac")
            self.assertEqual(a_stream.rate, 44100)

            # 5. Frame Count: 6.0s * 24fps = 144 frames
            frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 144, f"Expected exactly 144 frames, got {len(frames)}")

            # Helper to sample upper-left patch (50:150, 50:150) away from subtitle box
            def get_patch_rgb(f_idx: int) -> np.ndarray:
                arr = frames[f_idx].to_ndarray(format="rgb24")
                return arr[50:150, 50:150, :].mean(axis=(0, 1))

            # 6. Shot 1 interior and last frame
            rgb_0 = get_patch_rgb(0)
            rgb_35 = get_patch_rgb(35)
            self.assertGreater(rgb_0[0], 200, "Shot 1 frame 0 should be Yellow (high red)")
            self.assertGreater(rgb_0[1], 150, "Shot 1 frame 0 should be Yellow (high green)")
            self.assertGreater(rgb_35[0], 200, "Shot 1 frame 35 should be Yellow")

            # 7. Boundary 1: Cut at frame 36 (Shot 1 -> Shot 2)
            rgb_36 = get_patch_rgb(36)
            self.assertGreater(rgb_36[0], 200, "Shot 2 frame 36 should be Orange (high red)")
            self.assertLess(rgb_36[1], 130, "Shot 2 frame 36 green should drop from Yellow to Orange")
            diff_35_36 = np.abs(rgb_35[1] - rgb_36[1])
            self.assertGreater(diff_35_36, 50, "Green channel should step down sharply at cut 1")

            # 8. Boundary 2: Cut at frame 72 (Shot 2 -> Shot 3)
            rgb_71 = get_patch_rgb(71)
            rgb_72 = get_patch_rgb(72)
            self.assertGreater(rgb_71[0], 200, "Shot 2 frame 71 should be Orange")
            self.assertGreater(rgb_72[1], 140, "Shot 3 frame 72 should be Green")
            self.assertLess(rgb_72[0], 50, "Shot 3 frame 72 red should be near 0")

            # 9. Boundary 3: Cut at frame 108 (Shot 3 -> Shot 4)
            rgb_107 = get_patch_rgb(107)
            rgb_108 = get_patch_rgb(108)
            self.assertGreater(rgb_107[1], 140, "Shot 3 frame 107 should be Green")
            self.assertGreater(rgb_108[0], 140, "Shot 4 frame 108 should be Crimson")
            self.assertLess(rgb_108[1], 50, "Shot 4 frame 108 green should be near 0")

            # 10. Check no black or corrupted frames across entire video
            for idx, fr in enumerate(frames):
                arr = fr.to_ndarray(format="rgb24")
                mean_lum = arr.mean()
                self.assertGreater(mean_lum, 10.0, f"Frame {idx} appears completely black (mean lum: {mean_lum})")

            # 11. Subtitle continuity: Subtitle box is drawn at bottom center (y: 630 to 680)
            sub_patch_35 = frames[35].to_ndarray(format="rgb24")[630:680, 500:780, :]
            sub_patch_36 = frames[36].to_ndarray(format="rgb24")[630:680, 500:780, :]
            self.assertLess(sub_patch_35.mean(), 100, "Subtitle box should be present on frame 35")
            self.assertLess(sub_patch_36.mean(), 100, "Subtitle box should be present on frame 36")

        finally:
            container.close()

        # 12. Audio stream verification in dedicated container
        c_audio = av.open(final_mp4)
        try:
            audio_frames = list(c_audio.decode(audio=0))
            self.assertGreater(len(audio_frames), 0)
            total_audio_samples = sum(af.samples for af in audio_frames)
            expected_samples = int(round(6.0 * 44100))
            self.assertAlmostEqual(total_audio_samples, expected_samples, delta=2048)
        finally:
            c_audio.close()

    # =========================================================================
    # Test 2: Missing First Shot Recovery (Never shifts shot 2)
    # =========================================================================
    async def test_02_missing_first_shot_recovery(self):
        # 4 shots of 0.5s each = 2.0s (48 frames)
        shot_plan_2s = [
            {**s, "provisional_start_seconds": (i * 0.5), "provisional_end_seconds": ((i + 1) * 0.5)}
            for i, s in enumerate(self.shot_plan)
        ]
        script_p = MockScriptProvider(shot_plan_2s, self.action_timeline)
        tts_p = MockTTSProvider(measured_duration=2.0)
        video_p = MockVideoProvider(fail_shots={1})
        sfx_p = MockFreesoundProvider()

        orch = build_test_orchestrator(self.test_dir, script_p, tts_p, video_p, sfx_p, duration=2.0)
        proj_id = uuid.uuid4()
        final_mp4 = await orch.run_pipeline(proj_id)

        container = av.open(final_mp4)
        try:
            frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 48)

            # Shot 1 used Shot 2 (Orange) as fallback
            # Shot 2 plays its own Orange footage starting at frame 12
            # Shot 3 (Green) starts at frame 24
            patch_24 = frames[24].to_ndarray(format="rgb24")[50:150, 50:150, :].mean(axis=(0, 1))
            self.assertGreater(patch_24[1], 140, "Shot 3 must still start at frame 24 (Green)")
            self.assertLess(patch_24[0], 50)
        finally:
            container.close()

    # =========================================================================
    # Test 3: Missing Middle Shot Recovery (Extends preceding shot)
    # =========================================================================
    async def test_03_missing_middle_shot_recovery(self):
        shot_plan_2s = [
            {**s, "provisional_start_seconds": (i * 0.5), "provisional_end_seconds": ((i + 1) * 0.5)}
            for i, s in enumerate(self.shot_plan)
        ]
        script_p = MockScriptProvider(shot_plan_2s, self.action_timeline)
        tts_p = MockTTSProvider(measured_duration=2.0)
        video_p = MockVideoProvider(fail_shots={2})
        sfx_p = MockFreesoundProvider()

        orch = build_test_orchestrator(self.test_dir, script_p, tts_p, video_p, sfx_p, duration=2.0)
        proj_id = uuid.uuid4()
        final_mp4 = await orch.run_pipeline(proj_id)

        container = av.open(final_mp4)
        try:
            frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 48)

            # In interval 2 (frame 12 to 23), Shot 1 (Yellow) is extended as fallback
            patch_18 = frames[18].to_ndarray(format="rgb24")[50:150, 50:150, :].mean(axis=(0, 1))
            self.assertGreater(patch_18[0], 200, "Interval 2 should use Shot 1 (Yellow) fallback")
            self.assertGreater(patch_18[1], 150)

            # At frame 24, Shot 3 (Green) begins cleanly
            patch_24 = frames[24].to_ndarray(format="rgb24")[50:150, 50:150, :].mean(axis=(0, 1))
            self.assertGreater(patch_24[1], 140, "Shot 3 begins cleanly at frame 24")
        finally:
            container.close()

    # =========================================================================
    # Test 4: Missing Final Shot Recovery (Extends preceding shot)
    # =========================================================================
    async def test_04_missing_final_shot_recovery(self):
        shot_plan_2s = [
            {**s, "provisional_start_seconds": (i * 0.5), "provisional_end_seconds": ((i + 1) * 0.5)}
            for i, s in enumerate(self.shot_plan)
        ]
        script_p = MockScriptProvider(shot_plan_2s, self.action_timeline)
        tts_p = MockTTSProvider(measured_duration=2.0)
        video_p = MockVideoProvider(fail_shots={4})
        sfx_p = MockFreesoundProvider()

        orch = build_test_orchestrator(self.test_dir, script_p, tts_p, video_p, sfx_p, duration=2.0)
        proj_id = uuid.uuid4()
        final_mp4 = await orch.run_pipeline(proj_id)

        container = av.open(final_mp4)
        try:
            frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 48)

            # In interval 4 (frame 36 to 47), Shot 3 (Green) is extended
            patch_40 = frames[40].to_ndarray(format="rgb24")[50:150, 50:150, :].mean(axis=(0, 1))
            self.assertGreater(patch_40[1], 140, "Interval 4 should use Shot 3 (Green) fallback")
        finally:
            container.close()

    # =========================================================================
    # Test 5: All Shots Unavailable Failure Policy
    # =========================================================================
    async def test_05_all_shots_unavailable_raises_error(self):
        script_p = MockScriptProvider(self.shot_plan, self.action_timeline)
        tts_p = MockTTSProvider(measured_duration=6.0)
        video_p = MockVideoProvider(fail_shots={1, 2, 3, 4})
        sfx_p = MockFreesoundProvider()

        orch = build_test_orchestrator(self.test_dir, script_p, tts_p, video_p, sfx_p, duration=6.0)
        proj_id = uuid.uuid4()

        with self.assertRaises(NoSuitableMediaFoundError):
            await orch.run_pipeline(proj_id)

    # =========================================================================
    # Test 9: Ambiguous Shot-to-File Mapping Error
    # =========================================================================
    def test_08_ambiguous_shot_mapping_raises(self):
        reconciled = [
            {"shot_index": 1, "start_frame": 0, "end_frame": 36, "allocated_frames": 36},
            {"shot_index": 2, "start_frame": 36, "end_frame": 72, "allocated_frames": 36},
        ]
        with self.assertRaises(RuntimeError) as ctx:
            map_shots_to_media_files(
                reconciled_shots=reconciled,
                media_paths=["clip_unknown.mp4"],
                shot_metadata=[{"shot_index": 1}, {"shot_index": 2}],
            )
        self.assertIn("unambiguously", str(ctx.exception).lower())

    # =========================================================================
    # Test 10: Multi-Scene Integration with Inter-Scene Crossfade
    # =========================================================================
    def test_09_multi_scene_multi_shot_with_crossfade(self):
        # Scene 1: Multi-shot (2 shots, 2.0s)
        # Scene 2: Legacy single-shot (2.0s)
        c1 = os.path.join(self.test_dir, "s1_shot_1.mp4")
        c2 = os.path.join(self.test_dir, "s1_shot_2.mp4")
        c3 = os.path.join(self.test_dir, "s2_legacy.mp4")
        create_synthetic_mp4_clip(c1, color=(255, 0, 0), duration=1.0, fps=24, size=(640, 360))
        create_synthetic_mp4_clip(c2, color=(0, 255, 0), duration=1.0, fps=24, size=(640, 360))
        create_synthetic_mp4_clip(c3, color=(0, 0, 255), duration=2.0, fps=24, size=(640, 360))

        narr1 = os.path.join(self.test_dir, "narr1.wav")
        narr2 = os.path.join(self.test_dir, "narr2.wav")
        create_synthetic_wav_audio(narr1, duration=2.0, frequency=300.0)
        create_synthetic_wav_audio(narr2, duration=2.0, frequency=500.0)

        scenes = [
            {
                "scene_order": 1,
                "scene_text": "Scene 1 multi-shot narration.",
                "duration": 2.0,
                "audio_path": narr1,
                "is_multi_shot": True,
                "reconciled_shots": [
                    {"shot_index": 1, "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0},
                    {"shot_index": 2, "start_frame": 24, "end_frame": 48, "allocated_frames": 24, "duration_seconds": 1.0},
                ],
                "media_paths": [c1, c2],
                "shot_metadata": [{"shot_index": 1, "pexels_id": 11}, {"shot_index": 2, "pexels_id": 12}],
            },
            {
                "scene_order": 2,
                "scene_text": "Scene 2 single-shot narration.",
                "duration": 2.0,
                "audio_path": narr2,
                "is_multi_shot": False,
                "media_path": c3,
                "media_type": "video",
            }
        ]

        out_mp4 = os.path.join(self.test_dir, "multi_scene_output.mp4")
        resampled_dir = os.path.join(self.test_dir, "resampled_ms")

        final_path = assemble_video(
            scenes=scenes,
            output_path=out_mp4,
            fps=24,
            crossfade_duration=0.3,
            resampled_dir=resampled_dir
        )
        self.assertTrue(os.path.exists(final_path))

        c = av.open(final_path)
        try:
            frames = list(c.decode(video=0))
            # Total 4.0s @ 24fps = 96 frames
            self.assertEqual(len(frames), 96)

            # At frame 24: intra-scene straight cut from Shot 1 (Red) to Shot 2 (Green)
            rgb_23 = frames[23].to_ndarray(format="rgb24")[20:80, 20:80, :].mean(axis=(0, 1))
            rgb_24 = frames[24].to_ndarray(format="rgb24")[20:80, 20:80, :].mean(axis=(0, 1))
            self.assertGreater(rgb_23[0], 200, "Frame 23 should be Red")
            self.assertGreater(rgb_24[1], 200, "Frame 24 should be Green")

            # At frame 47 (t = 47/24 = 1.96s): during inter-scene crossfade to Scene 2 (Blue)
            # Should contain a blend of Scene 1 tail (Green) and Scene 2 head (Blue)
            rgb_47 = frames[47].to_ndarray(format="rgb24")[20:80, 20:80, :].mean(axis=(0, 1))
            self.assertGreater(rgb_47[1], 10, "Should have residual green")
            self.assertGreater(rgb_47[2], 50, "Should have blended blue")

            # At frame 55 (t = 55/24 = 2.29s): well inside Scene 2 (pure Blue)
            rgb_55 = frames[55].to_ndarray(format="rgb24")[20:80, 20:80, :].mean(axis=(0, 1))
            self.assertGreater(rgb_55[2], 200, "Scene 2 should be Blue")
        finally:
            c.close()

    # =========================================================================
    # Test 11: Windows Decoder Cleanup and Resource Lock Release
    # =========================================================================
    def test_10_decoder_cleanup_and_lock_release(self):
        c1 = os.path.join(self.test_dir, "lock_test.mp4")
        create_synthetic_mp4_clip(c1, color=(120, 120, 120), duration=1.0, fps=24, size=(320, 240))

        scene = {
            "scene_order": 1,
            "duration": 1.0,
            "is_multi_shot": True,
            "reconciled_shots": [{"shot_index": 1, "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0}],
            "media_paths": [c1],
            "shot_metadata": [{"shot_index": 1, "pexels_id": 99}],
        }
        vis = load_scene_visual_data(scene, duration=1.0, fps=24)
        close_visual_data(vis)
        gc.collect()

        # Delete immediately to verify zero Windows sharing violation
        try:
            os.remove(c1)
            deleted = True
        except PermissionError:
            deleted = False

        self.assertTrue(deleted, "File remained locked after close_visual_data")


if __name__ == "__main__":
    unittest.main()

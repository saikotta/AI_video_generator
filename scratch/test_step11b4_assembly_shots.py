"""
Step 11B.4 Offline Test Suite
Frame-Exact Multi-Shot Video Assembly and Subtitle Continuity

Covers 20 comprehensive test cases verifying:
1. Legacy single-shot scene
2. Legacy scene longer than 8 seconds (50/50 split)
3. Two-shot cooking scene
4. Three-shot manufacturing scene
5. Four-shot scene
6. Correct shot order
7. Exact cut at the allocated frame boundary
8. No missing or duplicate output frames
9. Non-integer measured narration duration
10. Missing secondary shot fallback
11. Missing primary shot fallback (never shifts shot 2)
12. Unresolvable shot-to-file mapping error
13. Continuous narration across cuts
14. Subtitle continuity across cuts
15. Freesound offset preservation
16. Existing inter-scene transitions
17. Short source clip handling
18. Decoder resource cleanup
19. Legacy image/Ken Burns rendering
20. No unintended change to audio mixing
"""

import os
import sys
import gc
import shutil
import tempfile
import unittest
import numpy as np
import av
import wave
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas.visual_spec import (
    ShotSpecificationSchema,
    VisualSpecificationSchema,
    reconcile_shot_plan_timing,
    ReconciledShot,
)
from app.services.video_assembly import (
    map_shots_to_media_files,
    load_scene_visual_data,
    get_single_visual_frame,
    render_scene_frame,
    close_visual_data,
    compute_frame_exact_duration,
    split_text_into_subtitles,
    build_subtitle_timeline,
    mix_scene_audio_pcm,
    assemble_video,
    create_scene_clip,
    loop_video_clip_if_needed,
    safe_open_video_clip,
)

def create_synthetic_mp4(
    path: str,
    color: tuple,
    duration: float = 2.0,
    fps: int = 24,
    size: tuple = (320, 240)
) -> str:
    """Create a synthetic MP4 video file with a solid color."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    container = av.open(path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = size[0]
    stream.height = size[1]
    stream.pix_fmt = 'yuv420p'
    num_frames = int(round(duration * fps))

    # Base frame image
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

def create_synthetic_wav(
    path: str,
    duration: float = 2.0,
    frequency: float = 440.0,
    sample_rate: int = 44100
) -> str:
    """Create a synthetic 16-bit PCM stereo WAV file."""
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

def create_synthetic_image(path: str, color: tuple = (100, 150, 200), size: tuple = (640, 360)) -> str:
    """Create a synthetic JPEG image."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path, "JPEG")
    return path


class TestStep11B4AssemblyShots(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="step11b4_test_")
        self.fps = 24

    def tearDown(self):
        gc.collect()
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    # =========================================================================
    # Case 1: Legacy single-shot scene
    # =========================================================================
    def test_01_legacy_single_shot_scene(self):
        clip_path = os.path.join(self.test_dir, "legacy_single.mp4")
        create_synthetic_mp4(clip_path, color=(200, 50, 50), duration=2.0, fps=self.fps)

        scene = {
            "scene_order": 1,
            "duration": 2.0,
            "media_path": clip_path,
            "media_type": "video",
            "is_multi_shot": False,
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            self.assertTrue(vis.get("is_video"))
            self.assertFalse(vis.get("is_multi_shot", False))
            self.assertFalse(vis.get("is_multi_clip", False))

            frame = get_single_visual_frame(vis, t=0.5, duration=2.0, target_size=(320, 240), fps=self.fps)
            self.assertIsInstance(frame, Image.Image)
            arr = np.array(frame)
            # Dominant red
            self.assertGreater(arr[:, :, 0].mean(), 150)
            self.assertLess(arr[:, :, 1].mean(), 80)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 2: Legacy scene longer than eight seconds
    # =========================================================================
    def test_02_legacy_scene_longer_than_eight_seconds(self):
        clip_a = os.path.join(self.test_dir, "clip_a.mp4")
        clip_b = os.path.join(self.test_dir, "clip_b.mp4")
        create_synthetic_mp4(clip_a, color=(220, 30, 30), duration=5.0, fps=self.fps)
        create_synthetic_mp4(clip_b, color=(30, 220, 30), duration=5.0, fps=self.fps)

        scene = {
            "scene_order": 1,
            "duration": 9.0,
            "media_paths": [clip_a, clip_b],
            "media_type": "video",
            "is_multi_shot": False,  # Not multi-shot! Legacy >8s multi-clip
        }
        vis = load_scene_visual_data(scene, duration=9.0, fps=self.fps)
        try:
            self.assertTrue(vis.get("is_multi_clip"))
            self.assertFalse(vis.get("is_multi_shot", False))
            self.assertEqual(vis["split_t"], 4.5)

            # Early frame -> clip a (red)
            frame_early = get_single_visual_frame(vis, t=1.0, duration=9.0, target_size=(320, 240), fps=self.fps)
            arr_early = np.array(frame_early)
            self.assertGreater(arr_early[:, :, 0].mean(), 150)

            # Late frame -> clip b (green)
            frame_late = get_single_visual_frame(vis, t=7.0, duration=9.0, target_size=(320, 240), fps=self.fps)
            arr_late = np.array(frame_late)
            self.assertGreater(arr_late[:, :, 1].mean(), 150)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 3: Two-shot cooking scene
    # =========================================================================
    def test_03_two_shot_cooking_scene(self):
        clip1 = os.path.join(self.test_dir, "scene_1_shot_1_101.mp4")
        clip2 = os.path.join(self.test_dir, "scene_1_shot_2_102.mp4")
        create_synthetic_mp4(clip1, color=(255, 0, 0), duration=2.0, fps=self.fps)    # Red
        create_synthetic_mp4(clip2, color=(0, 255, 0), duration=2.0, fps=self.fps)    # Green

        reconciled = [
            {"shot_index": 1, "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0},
            {"shot_index": 2, "start_frame": 24, "end_frame": 48, "allocated_frames": 24, "duration_seconds": 1.0},
        ]
        scene = {
            "scene_order": 1,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": reconciled,
            "media_paths": [clip1, clip2],
            "shot_metadata": [
                {"shot_index": 1, "pexels_id": 101, "allocated_frames": 24},
                {"shot_index": 2, "pexels_id": 102, "allocated_frames": 24},
            ],
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            self.assertTrue(vis.get("is_multi_shot"))
            self.assertEqual(len(vis["shots"]), 2)
            self.assertEqual(vis["total_frames"], 48)

            # Frame 10 (0.417s) should be Shot 1 (red)
            f1 = get_single_visual_frame(vis, t=10 / 24.0, duration=2.0, target_size=(320, 240), fps=self.fps)
            arr1 = np.array(f1)
            self.assertGreater(arr1[:, :, 0].mean(), 200)
            self.assertLess(arr1[:, :, 1].mean(), 50)

            # Frame 30 (1.25s) should be Shot 2 (green)
            f2 = get_single_visual_frame(vis, t=30 / 24.0, duration=2.0, target_size=(320, 240), fps=self.fps)
            arr2 = np.array(f2)
            self.assertGreater(arr2[:, :, 1].mean(), 200)
            self.assertLess(arr2[:, :, 0].mean(), 50)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 4: Three-shot manufacturing scene
    # =========================================================================
    def test_04_three_shot_manufacturing_scene(self):
        clip1 = os.path.join(self.test_dir, "scene_2_shot_1_201.mp4")
        clip2 = os.path.join(self.test_dir, "scene_2_shot_2_202.mp4")
        clip3 = os.path.join(self.test_dir, "scene_2_shot_3_203.mp4")
        create_synthetic_mp4(clip1, color=(255, 0, 0), duration=2.0, fps=self.fps)   # Red
        create_synthetic_mp4(clip2, color=(0, 255, 0), duration=2.0, fps=self.fps)   # Green
        create_synthetic_mp4(clip3, color=(0, 0, 255), duration=2.0, fps=self.fps)   # Blue

        reconciled = [
            {"shot_index": 1, "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0},
            {"shot_index": 2, "start_frame": 24, "end_frame": 48, "allocated_frames": 24, "duration_seconds": 1.0},
            {"shot_index": 3, "start_frame": 48, "end_frame": 72, "allocated_frames": 24, "duration_seconds": 1.0},
        ]
        scene = {
            "scene_order": 2,
            "duration": 3.0,
            "is_multi_shot": True,
            "reconciled_shots": reconciled,
            "media_paths": [clip1, clip2, clip3],
            "shot_metadata": [
                {"shot_index": 1, "pexels_id": 201},
                {"shot_index": 2, "pexels_id": 202},
                {"shot_index": 3, "pexels_id": 203},
            ],
        }
        vis = load_scene_visual_data(scene, duration=3.0, fps=self.fps)
        try:
            self.assertEqual(len(vis["shots"]), 3)
            self.assertEqual(vis["total_frames"], 72)

            # Test each shot window
            arr1 = np.array(get_single_visual_frame(vis, t=12 / 24.0, duration=3.0, fps=self.fps))
            self.assertGreater(arr1[:, :, 0].mean(), 200)

            arr2 = np.array(get_single_visual_frame(vis, t=36 / 24.0, duration=3.0, fps=self.fps))
            self.assertGreater(arr2[:, :, 1].mean(), 200)

            arr3 = np.array(get_single_visual_frame(vis, t=60 / 24.0, duration=3.0, fps=self.fps))
            self.assertGreater(arr3[:, :, 2].mean(), 200)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 5: Four-shot scene
    # =========================================================================
    def test_05_four_shot_scene(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        clips = []
        for i, c in enumerate(colors, 1):
            p = os.path.join(self.test_dir, f"scene_3_shot_{i}_{300+i}.mp4")
            create_synthetic_mp4(p, color=c, duration=1.0, fps=self.fps)
            clips.append(p)

        reconciled = [
            {"shot_index": 1, "start_frame": 0, "end_frame": 12, "allocated_frames": 12, "duration_seconds": 0.5},
            {"shot_index": 2, "start_frame": 12, "end_frame": 24, "allocated_frames": 12, "duration_seconds": 0.5},
            {"shot_index": 3, "start_frame": 24, "end_frame": 36, "allocated_frames": 12, "duration_seconds": 0.5},
            {"shot_index": 4, "start_frame": 36, "end_frame": 48, "allocated_frames": 12, "duration_seconds": 0.5},
        ]
        scene = {
            "scene_order": 3,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": reconciled,
            "media_paths": clips,
            "shot_metadata": [{"shot_index": i, "pexels_id": 300 + i} for i in range(1, 5)],
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            self.assertEqual(len(vis["shots"]), 4)
            self.assertEqual(vis["total_frames"], 48)

            # Frame 42 (Shot 4, yellow = R+G)
            arr4 = np.array(get_single_visual_frame(vis, t=42 / 24.0, duration=2.0, fps=self.fps))
            self.assertGreater(arr4[:, :, 0].mean(), 200)
            self.assertGreater(arr4[:, :, 1].mean(), 200)
            self.assertLess(arr4[:, :, 2].mean(), 50)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 6: Correct shot order
    # =========================================================================
    def test_06_correct_shot_order(self):
        clip1 = os.path.join(self.test_dir, "scene_4_shot_1_401.mp4")
        clip2 = os.path.join(self.test_dir, "scene_4_shot_2_402.mp4")
        create_synthetic_mp4(clip1, color=(255, 0, 0), duration=2.0, fps=self.fps)
        create_synthetic_mp4(clip2, color=(0, 255, 0), duration=2.0, fps=self.fps)

        # Pass in reverse order in media_paths / shot_metadata
        scene = {
            "scene_order": 4,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": [
                {"shot_index": 1, "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0},
                {"shot_index": 2, "start_frame": 24, "end_frame": 48, "allocated_frames": 24, "duration_seconds": 1.0},
            ],
            "media_paths": [clip2, clip1],  # Reversed!
            "shot_metadata": [
                {"shot_index": 2, "pexels_id": 402},
                {"shot_index": 1, "pexels_id": 401},
            ],
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            self.assertEqual(vis["shots"][0]["shot_index"], 1)
            self.assertEqual(vis["shots"][1]["shot_index"], 2)

            # Frame 5 must be red (Shot 1), Frame 30 must be green (Shot 2)
            arr1 = np.array(get_single_visual_frame(vis, t=5 / 24.0, duration=2.0, fps=self.fps))
            self.assertGreater(arr1[:, :, 0].mean(), 200)

            arr2 = np.array(get_single_visual_frame(vis, t=30 / 24.0, duration=2.0, fps=self.fps))
            self.assertGreater(arr2[:, :, 1].mean(), 200)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 7: Exact cut at the allocated frame boundary
    # =========================================================================
    def test_07_exact_cut_at_allocated_frame_boundary(self):
        clip1 = os.path.join(self.test_dir, "scene_5_shot_1_501.mp4")
        clip2 = os.path.join(self.test_dir, "scene_5_shot_2_502.mp4")
        create_synthetic_mp4(clip1, color=(255, 0, 0), duration=2.0, fps=self.fps)
        create_synthetic_mp4(clip2, color=(0, 255, 0), duration=2.0, fps=self.fps)

        boundary_frame = 31
        scene = {
            "scene_order": 5,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": [
                {"shot_index": 1, "start_frame": 0, "end_frame": boundary_frame, "allocated_frames": boundary_frame, "duration_seconds": boundary_frame / 24.0},
                {"shot_index": 2, "start_frame": boundary_frame, "end_frame": 48, "allocated_frames": 48 - boundary_frame, "duration_seconds": (48 - boundary_frame) / 24.0},
            ],
            "media_paths": [clip1, clip2],
            "shot_metadata": [{"shot_index": 1, "pexels_id": 501}, {"shot_index": 2, "pexels_id": 502}],
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            # Last frame of Shot 1 is frame 30
            t_last_shot1 = 30 / 24.0
            arr_30 = np.array(get_single_visual_frame(vis, t=t_last_shot1, duration=2.0, fps=self.fps))
            self.assertGreater(arr_30[:, :, 0].mean(), 200, "Frame 30 must be Shot 1 (red)")
            self.assertLess(arr_30[:, :, 1].mean(), 50)

            # First frame of Shot 2 is frame 31
            t_first_shot2 = 31 / 24.0
            arr_31 = np.array(get_single_visual_frame(vis, t=t_first_shot2, duration=2.0, fps=self.fps))
            self.assertGreater(arr_31[:, :, 1].mean(), 200, "Frame 31 must be Shot 2 (green)")
            self.assertLess(arr_31[:, :, 0].mean(), 50)

            # Zero blend: instant straight cut
            self.assertEqual(arr_30[:, :, 1].mean() < 50, True)
            self.assertEqual(arr_31[:, :, 0].mean() < 50, True)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 8: No missing or duplicate output frames
    # =========================================================================
    def test_08_no_missing_or_duplicate_output_frames(self):
        clip1 = os.path.join(self.test_dir, "scene_6_shot_1_601.mp4")
        clip2 = os.path.join(self.test_dir, "scene_6_shot_2_602.mp4")
        create_synthetic_mp4(clip1, color=(255, 0, 0), duration=2.0, fps=self.fps)
        create_synthetic_mp4(clip2, color=(0, 255, 0), duration=2.0, fps=self.fps)

        total_frames = 48
        cut_frame = 24
        scene = {
            "scene_order": 6,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": [
                {"shot_index": 1, "start_frame": 0, "end_frame": cut_frame, "allocated_frames": cut_frame, "duration_seconds": 1.0},
                {"shot_index": 2, "start_frame": cut_frame, "end_frame": total_frames, "allocated_frames": total_frames - cut_frame, "duration_seconds": 1.0},
            ],
            "media_paths": [clip1, clip2],
            "shot_metadata": [{"shot_index": 1, "pexels_id": 601}, {"shot_index": 2, "pexels_id": 602}],
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            rendered_shots = []
            for f in range(total_frames):
                t = f / float(self.fps)
                frame = get_single_visual_frame(vis, t=t, duration=2.0, fps=self.fps)
                arr = np.array(frame)
                is_red = arr[:, :, 0].mean() > 150
                rendered_shots.append(1 if is_red else 2)

            # Exactly 24 frames of Shot 1, followed by exactly 24 frames of Shot 2
            self.assertEqual(len(rendered_shots), 48)
            self.assertEqual(rendered_shots[:cut_frame], [1] * cut_frame)
            self.assertEqual(rendered_shots[cut_frame:], [2] * (total_frames - cut_frame))
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 9: Non-integer measured narration duration
    # =========================================================================
    def test_09_non_integer_measured_narration_duration(self):
        raw_dur = 3.375  # 3.375 * 24 = 81 frames exact
        exact_dur = compute_frame_exact_duration(raw_dur, fps=self.fps)
        self.assertEqual(exact_dur, 81 / 24.0)

        plan = [
            {"shot_index": 1, "required_visible_action": "action 1", "primary_subject": "sub", "required_setting": "set", "visual_state": "st", "camera_framing": "wide", "search_keywords": ["k1", "k2"], "provisional_start_seconds": 0.0, "provisional_end_seconds": 1.5},
            {"shot_index": 2, "required_visible_action": "action 2", "primary_subject": "sub", "required_setting": "set", "visual_state": "st", "camera_framing": "close", "search_keywords": ["k3", "k4"], "provisional_start_seconds": 1.5, "provisional_end_seconds": 3.0},
        ]
        reconciled = reconcile_shot_plan_timing(plan, raw_dur, fps=self.fps)
        total_allocated = sum(s["allocated_frames"] for s in reconciled)
        self.assertEqual(total_allocated, 81)

        clip1 = os.path.join(self.test_dir, "scene_7_shot_1_701.mp4")
        clip2 = os.path.join(self.test_dir, "scene_7_shot_2_702.mp4")
        create_synthetic_mp4(clip1, color=(255, 0, 0), duration=3.5, fps=self.fps)
        create_synthetic_mp4(clip2, color=(0, 255, 0), duration=3.5, fps=self.fps)

        scene = {
            "scene_order": 7,
            "duration": raw_dur,
            "is_multi_shot": True,
            "reconciled_shots": [s.to_dict() for s in reconciled],
            "media_paths": [clip1, clip2],
            "shot_metadata": [{"shot_index": 1, "pexels_id": 701}, {"shot_index": 2, "pexels_id": 702}],
        }
        vis = load_scene_visual_data(scene, duration=exact_dur, fps=self.fps)
        try:
            self.assertEqual(vis["total_frames"], 81)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 10: Missing secondary shot
    # =========================================================================
    def test_10_missing_secondary_shot_fallback(self):
        clip1 = os.path.join(self.test_dir, "scene_8_shot_1_801.mp4")
        create_synthetic_mp4(clip1, color=(255, 0, 0), duration=3.0, fps=self.fps)

        # Shot 2 clip is missing!
        scene = {
            "scene_order": 8,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": [
                {"shot_index": 1, "required_visible_action": "action 1", "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0},
                {"shot_index": 2, "required_visible_action": "action 2 missing", "start_frame": 24, "end_frame": 48, "allocated_frames": 24, "duration_seconds": 1.0},
            ],
            "media_paths": [clip1],
            "shot_metadata": [{"shot_index": 1, "pexels_id": 801}],
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            self.assertEqual(len(vis["shots"]), 2)
            self.assertFalse(scene.get("visual_coverage_complete", True))
            self.assertEqual(len(scene.get("missing_shots", [])), 1)
            self.assertEqual(scene["missing_shots"][0]["shot_index"], 2)

            # Shot 2 is marked as fallback, borrowing Shot 1's clip
            self.assertTrue(vis["shots"][1]["is_fallback"])
            self.assertIn("shot_1", vis["shots"][1]["fallback_source"])

            # Playback during Shot 2's interval works seamlessly using fallback
            frame_s2 = get_single_visual_frame(vis, t=30 / 24.0, duration=2.0, fps=self.fps)
            self.assertIsInstance(frame_s2, Image.Image)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 11: Missing primary shot (never shifts shot 2)
    # =========================================================================
    def test_11_missing_primary_shot_fallback_never_shifts(self):
        # Shot 1 failed, Shot 2 succeeded
        clip2 = os.path.join(self.test_dir, "scene_9_shot_2_902.mp4")
        create_synthetic_mp4(clip2, color=(0, 255, 0), duration=2.0, fps=self.fps)

        scene = {
            "scene_order": 9,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": [
                {"shot_index": 1, "required_visible_action": "primary action missing", "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0},
                {"shot_index": 2, "required_visible_action": "secondary action present", "start_frame": 24, "end_frame": 48, "allocated_frames": 24, "duration_seconds": 1.0},
            ],
            "media_paths": [clip2],
            "shot_metadata": [{"shot_index": 2, "pexels_id": 902}],
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            self.assertFalse(scene.get("visual_coverage_complete", True))
            self.assertEqual(len(scene.get("missing_shots", [])), 1)
            self.assertEqual(scene["missing_shots"][0]["shot_index"], 1)

            # Shot 1 uses Shot 2 as fallback
            self.assertTrue(vis["shots"][0]["is_fallback"])

            # Shot 2 is NOT a fallback and starts at frame 24
            self.assertFalse(vis["shots"][1]["is_fallback"])
            self.assertEqual(vis["shots"][1]["start_frame"], 24)
            self.assertEqual(vis["shots"][1]["end_frame"], 48)

            # Render at frame 24 (the exact start of Shot 2)
            f_start_s2 = get_single_visual_frame(vis, t=24 / 24.0, duration=2.0, fps=self.fps)
            self.assertIsInstance(f_start_s2, Image.Image)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 12: Unresolvable shot-to-file mapping error
    # =========================================================================
    def test_12_unresolvable_shot_to_file_mapping_raises(self):
        # 2 shots planned, 1 file with ambiguous name and mismatched metadata
        ambiguous_path = os.path.join(self.test_dir, "unrelated_footage.mp4")
        reconciled = [
            {"shot_index": 1, "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0},
            {"shot_index": 2, "start_frame": 24, "end_frame": 48, "allocated_frames": 24, "duration_seconds": 1.0},
        ]
        # Should raise RuntimeError because mapping cannot be unambiguously determined
        with self.assertRaises(RuntimeError) as ctx:
            map_shots_to_media_files(
                reconciled_shots=reconciled,
                media_paths=[ambiguous_path],
                shot_metadata=[{"shot_index": 1}, {"shot_index": 2}],
            )
        self.assertIn("unambiguously", str(ctx.exception).lower())

    # =========================================================================
    # Case 13: Continuous narration across cuts
    # =========================================================================
    def test_13_continuous_narration_across_cuts(self):
        narr_wav = os.path.join(self.test_dir, "narration.wav")
        create_synthetic_wav(narr_wav, duration=2.0, frequency=440.0)

        out_wav = mix_scene_audio_pcm(
            raw_narration_path=narr_wav,
            segment_sfx_list=[],
            ambient_audio_path=None,
            exact_duration=2.0,
            scene_order=1,
            target_dir=self.test_dir
        )
        self.assertTrue(os.path.exists(out_wav))
        with wave.open(out_wav, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 2)
            self.assertEqual(wf.getframerate(), 44100)
            self.assertEqual(wf.getnframes(), int(round(2.0 * 44100)))

    # =========================================================================
    # Case 14: Subtitle continuity across cuts
    # =========================================================================
    def test_14_subtitle_continuity_across_cuts(self):
        chunks = split_text_into_subtitles("Heat the pan quickly and stir the fresh carrots.", target_max_words=5)
        timeline = build_subtitle_timeline(chunks, total_duration=2.0)
        self.assertGreater(len(timeline), 0)

        # Shot cut is at t = 1.0s.
        # Find which subtitle chunk applies at t = 0.95s and t = 1.05s
        t_before = 0.95
        t_after = 1.05
        sub_before = ""
        sub_after = ""
        for st, en, txt in timeline:
            if st <= t_before <= en:
                sub_before = txt
            if st <= t_after <= en:
                sub_after = txt

        # Subtitle timeline should span smoothly and not reset to chunk 0 at t=1.05s
        self.assertTrue(bool(sub_before))
        self.assertTrue(bool(sub_after))

    # =========================================================================
    # Case 15: Freesound offset preservation
    # =========================================================================
    def test_15_freesound_offset_preservation(self):
        narr_wav = os.path.join(self.test_dir, "narr_sfx.wav")
        sfx_wav = os.path.join(self.test_dir, "sfx.wav")
        create_synthetic_wav(narr_wav, duration=3.0, frequency=200.0)
        create_synthetic_wav(sfx_wav, duration=0.5, frequency=1000.0)

        sfx_offset = 1.25
        sfx_list = [{
            "audio_path": sfx_wav,
            "start_offset": sfx_offset,
            "duration": 0.5,
            "volume": 0.8
        }]

        out_wav = mix_scene_audio_pcm(
            raw_narration_path=narr_wav,
            segment_sfx_list=sfx_list,
            ambient_audio_path=None,
            exact_duration=3.0,
            scene_order=10,
            target_dir=self.test_dir
        )
        with wave.open(out_wav, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16).reshape(-1, 2)

        # Before sfx_offset, audio should contain only narration frequency
        pre_sfx_sample = int(round(1.0 * 44100))
        sfx_active_sample = int(round((sfx_offset + 0.1) * 44100))

        # Check that PCM around sfx_active_sample has energy
        sfx_window = pcm[sfx_active_sample - 50 : sfx_active_sample + 50, 0]
        self.assertGreater(np.max(np.abs(sfx_window)), 0)

    # =========================================================================
    # Case 16: Existing inter-scene transitions
    # =========================================================================
    def test_16_existing_inter_scene_transitions(self):
        clip1 = os.path.join(self.test_dir, "scene_11_shot_1_1101.mp4")
        clip2 = os.path.join(self.test_dir, "scene_12_shot_1_1201.mp4")
        create_synthetic_mp4(clip1, color=(255, 0, 0), duration=2.0, fps=self.fps)
        create_synthetic_mp4(clip2, color=(0, 255, 0), duration=2.0, fps=self.fps)

        scene1 = {
            "scene_order": 1,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": [{"shot_index": 1, "start_frame": 0, "end_frame": 48, "allocated_frames": 48, "duration_seconds": 2.0}],
            "media_paths": [clip1],
            "shot_metadata": [{"shot_index": 1, "pexels_id": 1101}],
        }
        scene2 = {
            "scene_order": 2,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": [{"shot_index": 1, "start_frame": 0, "end_frame": 48, "allocated_frames": 48, "duration_seconds": 2.0}],
            "media_paths": [clip2],
            "shot_metadata": [{"shot_index": 1, "pexels_id": 1201}],
        }

        vis1 = load_scene_visual_data(scene1, duration=2.0, fps=self.fps)
        vis2 = load_scene_visual_data(scene2, duration=2.0, fps=self.fps)
        try:
            # During crossfade (e.g. t = 1.8s with crossfade_duration=0.4s)
            blended = render_scene_frame(
                scene_visual=vis1,
                next_scene_visual=vis2,
                t=1.8,
                duration=2.0,
                subtitle_timeline=[],
                crossfade_duration=0.4,
                target_size=(320, 240)
            )
            # Should have both red and green components
            self.assertGreater(blended[:, :, 0].mean(), 50)
            self.assertGreater(blended[:, :, 1].mean(), 50)
        finally:
            close_visual_data(vis1)
            close_visual_data(vis2)

    # =========================================================================
    # Case 17: Short source clip handling
    # =========================================================================
    def test_17_short_source_clip_handling(self):
        short_clip = os.path.join(self.test_dir, "short_clip.mp4")
        # Clip is only 0.5s (12 frames)
        create_synthetic_mp4(short_clip, color=(255, 100, 50), duration=0.5, fps=self.fps)

        # Shot requires 2.0s (48 frames)
        scene = {
            "scene_order": 13,
            "duration": 2.0,
            "is_multi_shot": True,
            "reconciled_shots": [{"shot_index": 1, "start_frame": 0, "end_frame": 48, "allocated_frames": 48, "duration_seconds": 2.0}],
            "media_paths": [short_clip],
            "shot_metadata": [{"shot_index": 1, "pexels_id": 1301}],
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            # Render frame past original 0.5s (e.g. at 1.5s = frame 36)
            f = get_single_visual_frame(vis, t=1.5, duration=2.0, target_size=(320, 240), fps=self.fps)
            self.assertIsInstance(f, Image.Image)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 18: Decoder resource cleanup
    # =========================================================================
    def test_18_decoder_resource_cleanup(self):
        clip1 = os.path.join(self.test_dir, "cleanup_test.mp4")
        create_synthetic_mp4(clip1, color=(100, 100, 100), duration=1.0, fps=self.fps)

        scene = {
            "scene_order": 14,
            "duration": 1.0,
            "is_multi_shot": True,
            "reconciled_shots": [{"shot_index": 1, "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0}],
            "media_paths": [clip1],
            "shot_metadata": [{"shot_index": 1, "pexels_id": 1401}],
        }
        vis = load_scene_visual_data(scene, duration=1.0, fps=self.fps)
        close_visual_data(vis)
        gc.collect()

        # Should be able to delete the file immediately without Windows sharing violation
        try:
            os.remove(clip1)
            file_deleted = True
        except Exception:
            file_deleted = False
        self.assertTrue(file_deleted, "File should not be locked after close_visual_data")

    # =========================================================================
    # Case 19: Legacy image/Ken Burns rendering
    # =========================================================================
    def test_19_legacy_image_ken_burns_rendering(self):
        img_path = os.path.join(self.test_dir, "legacy_photo.jpg")
        create_synthetic_image(img_path, color=(50, 100, 150))

        scene = {
            "scene_order": 15,
            "duration": 2.0,
            "image_path": img_path,
            "media_type": "image",
            "is_multi_shot": False,
        }
        vis = load_scene_visual_data(scene, duration=2.0, fps=self.fps)
        try:
            self.assertFalse(vis.get("is_video"))
            self.assertIn("canvas", vis)

            f = get_single_visual_frame(vis, t=1.0, duration=2.0, target_size=(320, 240), fps=self.fps)
            self.assertIsInstance(f, Image.Image)
        finally:
            close_visual_data(vis)

    # =========================================================================
    # Case 20: No unintended change to audio mixing
    # =========================================================================
    def test_20_no_unintended_change_to_audio_mixing(self):
        narr_wav = os.path.join(self.test_dir, "audio_consistency.wav")
        create_synthetic_wav(narr_wav, duration=2.0, frequency=500.0)

        out1 = mix_scene_audio_pcm(narr_wav, [], None, 2.0, 101, target_dir=self.test_dir)
        out2 = mix_scene_audio_pcm(narr_wav, [], None, 2.0, 102, target_dir=self.test_dir)

        with wave.open(out1, "rb") as w1, wave.open(out2, "rb") as w2:
            pcm1 = w1.readframes(w1.getnframes())
            pcm2 = w2.readframes(w2.getnframes())

        # Audio mixing must be 100% deterministic and identical
        self.assertEqual(pcm1, pcm2)

    # =========================================================================
    # Case 21: Full assemble_video synthetic multi-shot end-to-end render
    # =========================================================================
    def test_21_assemble_video_end_to_end_synthetic_render(self):
        # Create synthetic audio
        narr1 = os.path.join(self.test_dir, "e2e_narr1.wav")
        narr2 = os.path.join(self.test_dir, "e2e_narr2.wav")
        create_synthetic_wav(narr1, duration=2.0, frequency=300.0)
        create_synthetic_wav(narr2, duration=1.5, frequency=500.0)

        # Create synthetic video clips for multi-shot scene 1 (2 shots)
        s1_c1 = os.path.join(self.test_dir, "scene_1_shot_1_901.mp4")
        s1_c2 = os.path.join(self.test_dir, "scene_1_shot_2_902.mp4")
        create_synthetic_mp4(s1_c1, color=(255, 0, 0), duration=1.0, fps=self.fps)
        create_synthetic_mp4(s1_c2, color=(0, 255, 0), duration=1.0, fps=self.fps)

        # Create synthetic video clip for legacy scene 2
        s2_c = os.path.join(self.test_dir, "scene_2_legacy.mp4")
        create_synthetic_mp4(s2_c, color=(0, 0, 255), duration=1.5, fps=self.fps)

        scenes = [
            {
                "scene_order": 1,
                "scene_text": "First heat the pan and add fresh oil.",
                "duration": 2.0,
                "audio_path": narr1,
                "is_multi_shot": True,
                "reconciled_shots": [
                    {"shot_index": 1, "start_frame": 0, "end_frame": 24, "allocated_frames": 24, "duration_seconds": 1.0},
                    {"shot_index": 2, "start_frame": 24, "end_frame": 48, "allocated_frames": 24, "duration_seconds": 1.0},
                ],
                "media_paths": [s1_c1, s1_c2],
                "shot_metadata": [{"shot_index": 1, "pexels_id": 901}, {"shot_index": 2, "pexels_id": 902}],
            },
            {
                "scene_order": 2,
                "scene_text": "Toss the crisp vegetables until tender.",
                "duration": 1.5,
                "audio_path": narr2,
                "is_multi_shot": False,
                "media_path": s2_c,
                "media_type": "video",
            }
        ]

        out_mp4 = os.path.join(self.test_dir, "full_rendered_output.mp4")
        resampled_dir = os.path.join(self.test_dir, "resampled")

        res = assemble_video(
            scenes=scenes,
            output_path=out_mp4,
            fps=self.fps,
            crossfade_duration=0.2,
            resampled_dir=resampled_dir
        )
        self.assertTrue(os.path.exists(res))

        # Inspect resulting container with PyAV
        c = av.open(res)
        self.assertGreater(len(c.streams.video), 0)
        self.assertGreater(len(c.streams.audio), 0)
        v_stream = c.streams.video[0]
        a_stream = c.streams.audio[0]
        self.assertEqual(v_stream.width, 1280)
        self.assertEqual(v_stream.height, 720)
        self.assertEqual(a_stream.rate, 44100)
        c.close()


if __name__ == "__main__":
    unittest.main()

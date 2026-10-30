import os
import sys
import gc
import math
import wave
import time
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import av
from PIL import Image, ImageDraw, ImageFont, ImageOps
import moviepy as mp
from moviepy import (
    VideoClip,
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    concatenate_videoclips,
    concatenate_audioclips,
    afx
)

# Portable default (no machine-specific drive paths). The pipeline always passes a
# per-project directory; this is only a fallback for standalone use.
DEFAULT_RESAMPLED_DIR = os.path.join(os.getenv("MEDIA_STORAGE_PATH", "./media"), "audio_resampled")

def ensure_resampled_audio(audio_path: str, target_dir: str = DEFAULT_RESAMPLED_DIR) -> str:
    """
    Resample input audio file to 44100 Hz stereo 16-bit PCM WAV using PyAV.
    Preserves real-world pitch and spoken duration exactly.
    """
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(audio_path))[0] + "_44k.wav"
    output_wav = os.path.join(target_dir, filename)

    # Reuse the cached WAV only if it is not older than its source (avoids stale audio
    # when a scene's narration is regenerated under the same filename).
    if (
        os.path.exists(output_wav)
        and os.path.getsize(output_wav) > 1000
        and os.path.getmtime(output_wav) >= os.path.getmtime(audio_path)
    ):
        return output_wav

    in_container = av.open(audio_path)
    in_stream = in_container.streams.audio[0]

    resampler = av.AudioResampler(format='s16', layout='stereo', rate=44100)

    pcm_bytes = bytearray()
    for frame in in_container.decode(audio=0):
        resampled_frames = resampler.resample(frame)
        if resampled_frames:
            for rf in resampled_frames:
                pcm_bytes.extend(rf.to_ndarray().tobytes())

    flushed = resampler.resample(None)
    if flushed:
        for rf in flushed:
            pcm_bytes.extend(rf.to_ndarray().tobytes())

    in_container.close()

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(44100)
        wf.writeframes(pcm_bytes)

    return output_wav

def compute_frame_exact_duration(orig_duration: float, fps: float = 24.0) -> float:
    """
    Calculate ceiling frame-exact scene duration for a given fps (24.0) so that
    video duration >= audio duration, preserving the full untrimmed narration audio.
    """
    num_frames = int(np.ceil(orig_duration * fps))
    return num_frames / float(fps)

def split_text_into_subtitles(text: str, target_max_words: int = 7) -> List[str]:
    """
    Split scene narration text into short readable subtitle chunks (~4-8 words),
    respecting natural clause and phrase boundaries (punctuation, conjunctions, prepositions).
    """
    text = text.strip()
    if not text:
        return []

    raw_clauses = re.split(r'(?<=[.,;:!?])\s+', text)
    chunks = []

    conjunctions = {
        "and", "but", "where", "which", "that", "with", "from", "as", "to",
        "when", "while", "after", "because", "or", "so", "into", "through"
    }

    for clause in raw_clauses:
        clause = clause.strip()
        if not clause:
            continue

        words = clause.split()
        if len(words) <= target_max_words:
            chunks.append(clause)
        else:
            sub_curr = []
            for idx, w in enumerate(words):
                sub_curr.append(w)
                w_clean = w.lower().strip(".,;:!?")
                is_boundary_word = (w_clean in conjunctions)

                if len(sub_curr) >= 4 and (is_boundary_word or len(sub_curr) >= target_max_words):
                    chunks.append(" ".join(sub_curr))
                    sub_curr = []
            if sub_curr:
                chunks.append(" ".join(sub_curr))

    return chunks

def build_subtitle_timeline(chunks: List[str], total_duration: float) -> List[Tuple[float, float, str]]:
    """
    Allocate subtitle timing proportionally across exact total_duration based on word count per chunk.
    Returns list of (start_time, end_time, subtitle_text).
    """
    if not chunks or total_duration <= 0:
        return []

    total_words = sum(len(c.split()) for c in chunks)
    if total_words <= 0:
        return [(0.0, total_duration, chunks[0])] if chunks else []

    timeline = []
    curr_time = 0.0

    for idx, c in enumerate(chunks):
        chunk_words = len(c.split())
        chunk_dur = total_duration * (chunk_words / float(total_words))
        start_t = curr_time
        end_t = min(total_duration, curr_time + chunk_dur)
        if idx == len(chunks) - 1:
            end_t = total_duration
        timeline.append((start_t, end_t, c))
        curr_time = end_t

    return timeline

def prep_16_9_canvas(img: Image.Image, canvas_size: Tuple[int, int] = (1920, 1080)) -> Image.Image:
    """Pre-fit base image to high-res 16:9 canvas using ImageOps.fit."""
    return ImageOps.fit(img, canvas_size, Image.Resampling.LANCZOS)

def get_ken_burns_crop_box_and_frame(
    canvas: Image.Image,
    t: float,
    duration: float,
    start_scale: float = 1.00,
    end_scale: float = 1.22,
    target_size: Tuple[int, int] = (1280, 720)
) -> Tuple[Tuple[float, float, float, float], float, Image.Image]:
    """
    Calculate crop box, scale factor, and resized frame image for static photo fallbacks.
    """
    cw, ch = canvas.size
    progress = max(0.0, min(1.0, t / max(0.001, duration)))
    scale = start_scale + (end_scale - start_scale) * progress

    crop_w = float(cw) / scale
    crop_h = float(ch) / scale

    left = (cw - crop_w) * 0.5
    top = (ch - crop_h) * 0.5
    right = left + crop_w
    bottom = top + crop_h

    crop_box = (left, top, right, bottom)
    cropped = canvas.crop(crop_box)
    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)

    return crop_box, scale, resized

def loop_video_clip_if_needed(video_clip: VideoFileClip, target_duration: float) -> VideoClip:
    """Trim video clip if longer than target_duration, or loop seamlessly if shorter."""
    if video_clip.duration >= target_duration:
        return video_clip.subclipped(0, target_duration)

    num_loops = int(np.ceil(target_duration / max(0.1, video_clip.duration)))
    clips = [video_clip] * num_loops
    concatenated = concatenate_videoclips(clips)
    return concatenated.subclipped(0, target_duration)

# Slow "push-in" on video shots: static stock clips (a hand holding beans, a sight glass) look like slides,
# so every shot gets a gentle zoom, stronger on near-static clips. Turn off with VIDEO_PUSH_IN=0.
PUSH_IN_ENABLED = os.getenv("VIDEO_PUSH_IN", "1").strip().lower() not in ("0", "false", "no", "off")
PUSH_IN_STATIC_ZOOM = 1.08    # end zoom for near-static clips
PUSH_IN_MOVING_ZOOM = 1.035   # end zoom for clips that already move
PUSH_IN_MOTION_THRESHOLD = 6.0


def apply_push_in(img: Image.Image, progress: float, max_zoom: float) -> Image.Image:
    """Zoom into the centre of img by 1.0 -> max_zoom as progress goes 0 -> 1 (sub-pixel, same output size)."""
    if max_zoom <= 1.0:
        return img
    z = 1.0 + (max_zoom - 1.0) * max(0.0, min(1.0, progress))
    w, h = img.size
    a = 1.0 / z
    return img.transform((w, h), Image.AFFINE, (a, 0.0, (w - w * a) / 2.0, 0.0, a, (h - h * a) / 2.0),
                         resample=Image.BICUBIC)


def shot_zoom_limit(shot: Dict[str, Any]) -> float:
    """Pick the end zoom for a shot from how much its clip moves (measured once, cached on the shot)."""
    cached = shot.get("_zoom_limit")
    if cached is not None:
        return cached
    limit = (PUSH_IN_STATIC_ZOOM + PUSH_IN_MOVING_ZOOM) / 2.0
    try:
        clip = shot["video_clip"]
        d = max(0.1, float(clip.duration))
        diffs = []
        for a_t, b_t in ((0.15, 0.30), (0.45, 0.60), (0.70, 0.85)):
            fa = np.asarray(Image.fromarray(clip.get_frame(d * a_t)).convert("L").resize((64, 36)), dtype=np.int16)
            fb = np.asarray(Image.fromarray(clip.get_frame(d * b_t)).convert("L").resize((64, 36)), dtype=np.int16)
            diffs.append(float(np.abs(fa - fb).mean()))
        limit = PUSH_IN_STATIC_ZOOM if (sum(diffs) / len(diffs)) < PUSH_IN_MOTION_THRESHOLD else PUSH_IN_MOVING_ZOOM
    except Exception:
        pass
    shot["_zoom_limit"] = limit
    return limit


def get_single_visual_frame(
    visual_data: Dict[str, Any],
    t: float,
    duration: float,
    target_size: Tuple[int, int] = (1280, 720),
    fps: float = 24.0
) -> Image.Image:
    """Render raw visual frame at time t for a scene (supporting multi-shot sequential clips, multi-clip cuts, and static fallbacks)."""
    target_w, target_h = target_size

    if visual_data.get("is_multi_shot"):
        fps_val = visual_data.get("fps", fps)
        total_frames = visual_data.get("total_frames", int(np.ceil(duration * fps_val)))
        shots = visual_data.get("shots", [])

        if not shots:
            return Image.new("RGB", (target_w, target_h), (0, 0, 0))

        # Deterministic integer frame index calculation (MoviePy samples at t = k / fps)
        frame_idx = min(total_frames - 1, max(0, int(round(t * fps_val))))

        # Chronological straight cuts based on authoritative integer boundaries
        target_shot = None
        for s in shots:
            if s["start_frame"] <= frame_idx < s["end_frame"]:
                target_shot = s
                break
        if target_shot is None:
            target_shot = shots[-1]

        intra_frame = frame_idx - target_shot["start_frame"]
        intra_t = intra_frame / float(fps_val)

        if target_shot.get("is_video", True):
            v_clip = target_shot["video_clip"]
            t_bounded = max(0.0, min(v_clip.duration - 0.04, intra_t % max(0.04, v_clip.duration)))
            raw_frame = v_clip.get_frame(t_bounded)
            shot_img = ImageOps.fit(Image.fromarray(raw_frame), (target_w, target_h), Image.Resampling.LANCZOS)
            if PUSH_IN_ENABLED:
                shot_len = max(0.1, (target_shot["end_frame"] - target_shot["start_frame"]) / float(fps_val))
                shot_img = apply_push_in(shot_img, intra_t / shot_len, shot_zoom_limit(target_shot))
            return shot_img
        else:
            canvas = target_shot["canvas"]
            shot_dur = target_shot.get("duration_seconds", max(0.1, (target_shot["end_frame"] - target_shot["start_frame"]) / float(fps_val)))
            _, _, frame_img = get_ken_burns_crop_box_and_frame(canvas, intra_t, shot_dur, 1.00, 1.22, target_size)
            return frame_img

    elif visual_data.get("is_multi_clip"):
        clips_list = visual_data["clips_list"]
        split_t = visual_data["split_t"]
        crossfade_in = 0.4

        if t < split_t - crossfade_in:
            v_clip_a = clips_list[0]
            t_bounded = max(0.0, min(v_clip_a.duration - 0.04, t))
            raw_a = v_clip_a.get_frame(t_bounded)
            return ImageOps.fit(Image.fromarray(raw_a), (target_w, target_h), Image.Resampling.LANCZOS)
        elif t < split_t:
            alpha = (t - (split_t - crossfade_in)) / crossfade_in
            v_clip_a = clips_list[0]
            v_clip_b = clips_list[1]

            t_bounded_a = max(0.0, min(v_clip_a.duration - 0.04, t))
            t_bounded_b = max(0.0, min(v_clip_b.duration - 0.04, t - split_t + crossfade_in))

            raw_a = v_clip_a.get_frame(t_bounded_a)
            raw_b = v_clip_b.get_frame(t_bounded_b)

            img_a = ImageOps.fit(Image.fromarray(raw_a), (target_w, target_h), Image.Resampling.LANCZOS)
            img_b = ImageOps.fit(Image.fromarray(raw_b), (target_w, target_h), Image.Resampling.LANCZOS)

            return Image.blend(img_a.convert("RGB"), img_b.convert("RGB"), alpha)
        else:
            v_clip_b = clips_list[1]
            t_local = t - split_t
            t_bounded = max(0.0, min(v_clip_b.duration - 0.04, t_local))
            raw_b = v_clip_b.get_frame(t_bounded)
            return ImageOps.fit(Image.fromarray(raw_b), (target_w, target_h), Image.Resampling.LANCZOS)

    elif visual_data.get("is_video"):
        v_clip = visual_data["video_clip"]
        t_bounded = max(0.0, min(v_clip.duration - 0.04, t))
        raw_frame = v_clip.get_frame(t_bounded)
        return ImageOps.fit(Image.fromarray(raw_frame), (target_w, target_h), Image.Resampling.LANCZOS)
    else:
        canvas = visual_data["canvas"]
        _, _, frame_img = get_ken_burns_crop_box_and_frame(canvas, t, duration, 1.00, 1.22, target_size)
        return frame_img

def render_scene_frame(
    scene_visual: Dict[str, Any],
    next_scene_visual: Optional[Dict[str, Any]],
    t: float,
    duration: float,
    subtitle_timeline: List[Tuple[float, float, str]],
    crossfade_duration: float = 0.4,
    target_size: Tuple[int, int] = (1280, 720)
) -> np.ndarray:
    """Render single video frame with inter-scene crossfade transition and subtitle overlay."""
    target_w, target_h = target_size

    frame1_img = get_single_visual_frame(scene_visual, t, duration, target_size)

    fade_start = duration - crossfade_duration
    if next_scene_visual is not None and t >= fade_start and crossfade_duration > 0:
        alpha = max(0.0, min(1.0, (t - fade_start) / crossfade_duration))
        frame2_img = get_single_visual_frame(next_scene_visual, 0.0, duration, target_size)

        f1 = frame1_img.convert("RGB")
        f2 = frame2_img.convert("RGB")
        blended_img = Image.blend(f1, f2, alpha)
    else:
        blended_img = frame1_img

    frame_rgba = blended_img.convert("RGBA")

    active_subtitle = ""
    for start_t, end_t, text in subtitle_timeline:
        if start_t <= t <= end_t:
            active_subtitle = text
            break
    if not active_subtitle and subtitle_timeline:
        active_subtitle = subtitle_timeline[-1][2]

    if active_subtitle:
        overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), active_subtitle, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        padding_x = 24
        padding_y = 14
        box_w = text_w + (padding_x * 2)
        box_h = text_h + (padding_y * 2)

        box_left = (target_w - box_w) / 2.0
        box_top = target_h - box_h - 45
        box_right = box_left + box_w
        box_bottom = box_top + box_h

        draw.rounded_rectangle(
            [box_left, box_top, box_right, box_bottom],
            radius=10,
            fill=(0, 0, 0, 190),
            outline=(255, 255, 255, 60),
            width=1
        )

        text_x = box_left + padding_x
        text_y = box_top + padding_y - 2
        draw.text((text_x, text_y), active_subtitle, font=font, fill=(255, 255, 255, 255))

        frame_rgba = Image.alpha_composite(frame_rgba, overlay)

    return np.array(frame_rgba.convert("RGB"))

def safe_open_video_clip(media_path: str, max_retries: int = 3, retry_delay: float = 0.3) -> VideoFileClip:
    """Safely open VideoFileClip with sequential retry-with-delay logic to handle Windows pipe timing glitches."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return VideoFileClip(media_path)
        except Exception as e:
            last_err = e
            time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"Failed to open video clip {media_path} after {max_retries} attempts: {last_err}")

def map_shots_to_media_files(
    reconciled_shots: List[Any],
    media_paths: List[str],
    shot_metadata: Optional[List[Dict[str, Any]]] = None
) -> Dict[int, str]:
    """
    Deterministically map each planned shot_index to its corresponding media file path.
    Inspects shot_metadata and available filename patterns.
    Never assumes len(media_paths) == len(reconciled_shots).
    Raises RuntimeError if existing metadata cannot unambiguously map shots to files.
    """
    mapped: Dict[int, str] = {}
    valid_shot_indices = set()
    for s in reconciled_shots:
        idx = s.get("shot_index") if isinstance(s, dict) else getattr(s, "shot_index")
        valid_shot_indices.add(idx)

    # 1. If shot_metadata is provided, use explicit linkages
    if shot_metadata:
        # Check if each shot_metadata has an explicit local_path
        has_paths_in_meta = all(
            (m.get("local_path") or m.get("file_path")) for m in shot_metadata
        )
        if has_paths_in_meta:
            for m in shot_metadata:
                idx = m.get("shot_index")
                p = m.get("local_path") or m.get("file_path")
                if idx in valid_shot_indices and p and os.path.exists(p):
                    mapped[idx] = p
            return mapped

        # If len(shot_metadata) == len(media_paths), they are 1-to-1 parallel lists from Stage 5
        if len(shot_metadata) == len(media_paths):
            for meta, path in zip(shot_metadata, media_paths):
                meta_idx = meta.get("shot_index")
                base = os.path.basename(path)
                fn_match = re.search(r'(?:^|_)shot_(\d+)(?:_|\.|$)', base, re.IGNORECASE)
                if fn_match:
                    fn_idx = int(fn_match.group(1))
                    if fn_idx != meta_idx:
                        raise RuntimeError(
                            f"Conflicting shot metadata contract: metadata specifies shot_index {meta_idx} "
                            f"but file '{base}' indicates shot_index {fn_idx}."
                        )
                if meta_idx in valid_shot_indices:
                    mapped[meta_idx] = path
            return mapped

        # If lengths differ, try matching filenames via _shot_{idx}_ or pexels_id
        for path in media_paths:
            base = os.path.basename(path)
            fn_match = re.search(r'(?:^|_)shot_(\d+)(?:_|\.|$)', base, re.IGNORECASE)
            if fn_match:
                fn_idx = int(fn_match.group(1))
                if fn_idx in valid_shot_indices:
                    mapped[fn_idx] = path
                    continue

            # Try matching via pexels_id
            matched = False
            for meta in shot_metadata:
                pid = str(meta.get("pexels_id") or meta.get("id", ""))
                if pid and pid in base:
                    meta_idx = meta.get("shot_index")
                    if meta_idx in valid_shot_indices:
                        mapped[meta_idx] = path
                        matched = True
                        break
            if not matched:
                raise RuntimeError(
                    f"Unresolvable shot-to-file mapping: media file '{base}' cannot be unambiguously mapped "
                    f"to shot_metadata (shot_metadata count: {len(shot_metadata)}, media_paths count: {len(media_paths)})."
                )
        return mapped

    # 2. If shot_metadata is not provided
    if len(media_paths) == len(reconciled_shots):
        # 1-to-1 sequential mapping
        for s, path in zip(reconciled_shots, media_paths):
            idx = s.get("shot_index") if isinstance(s, dict) else getattr(s, "shot_index")
            mapped[idx] = path
        return mapped

    # If counts differ without shot_metadata, inspect filenames
    for path in media_paths:
        base = os.path.basename(path)
        fn_match = re.search(r'(?:^|_)shot_(\d+)(?:_|\.|$)', base, re.IGNORECASE)
        if fn_match:
            fn_idx = int(fn_match.group(1))
            if fn_idx in valid_shot_indices:
                mapped[fn_idx] = path
                continue
        raise RuntimeError(
            f"Unresolvable shot-to-file mapping: media_paths count ({len(media_paths)}) does not match "
            f"reconciled_shots count ({len(reconciled_shots)}) and file '{base}' lacks shot index identifier."
        )

    return mapped

def load_scene_visual_data(
    scene_dict: Dict[str, Any],
    duration: float,
    fps: float = 24.0
) -> Dict[str, Any]:
    """Load visual data for a scene (supporting multi-shot sequential clips, multi-clip lists, or single clip/image)."""
    # 1. MULTI-SHOT SEQUENTIAL RENDERING BRANCH (Step 11B.4)
    is_multi_shot = bool(scene_dict.get("is_multi_shot") or scene_dict.get("reconciled_shots"))
    if is_multi_shot and scene_dict.get("reconciled_shots"):
        reconciled_shots = scene_dict["reconciled_shots"]
        media_paths = scene_dict.get("media_paths") or []
        shot_metadata = scene_dict.get("shot_metadata") or []

        # Map shot_index -> media file path
        shot_to_path = map_shots_to_media_files(reconciled_shots, media_paths, shot_metadata)

        # Detect any missing shots
        missing_shots = []
        for r_shot in reconciled_shots:
            s_idx = r_shot.get("shot_index") if isinstance(r_shot, dict) else r_shot.shot_index
            p = shot_to_path.get(s_idx)
            if not p or not os.path.exists(p):
                missing_shots.append({
                    "shot_index": s_idx,
                    "required_visible_action": r_shot.get("required_visible_action") if isinstance(r_shot, dict) else getattr(r_shot, "required_visible_action", None),
                    "primary_subject": r_shot.get("primary_subject") if isinstance(r_shot, dict) else getattr(r_shot, "primary_subject", None),
                    "reason": "Missing or non-existent media file"
                })

        if missing_shots:
            scene_dict["visual_coverage_complete"] = False
            if "missing_shots" not in scene_dict or not scene_dict["missing_shots"]:
                scene_dict["missing_shots"] = missing_shots
            if not scene_dict.get("fallback_used"):
                scene_dict["fallback_used"] = "fallback_shot_applied_for_missing_footage"

        # Determine available shot indices
        available_shot_indices = [
            (r_shot.get("shot_index") if isinstance(r_shot, dict) else r_shot.shot_index)
            for r_shot in reconciled_shots
            if shot_to_path.get(r_shot.get("shot_index") if isinstance(r_shot, dict) else r_shot.shot_index)
            and os.path.exists(shot_to_path[r_shot.get("shot_index") if isinstance(r_shot, dict) else r_shot.shot_index])
        ]

        scene_fallback_path = scene_dict.get("media_path") or scene_dict.get("image_path") or scene_dict.get("video_path")

        if not available_shot_indices and (not scene_fallback_path or not os.path.exists(scene_fallback_path)):
            raise FileNotFoundError(
                f"No visual media available for multi-shot scene {scene_dict.get('scene_order', 1)}: "
                f"all {len(reconciled_shots)} shots missing and no scene-level media path found."
            )

        opened_clips: Dict[str, VideoFileClip] = {}
        unique_opened_clips: List[VideoFileClip] = []

        def get_or_open_clip(p: str) -> VideoFileClip:
            if p not in opened_clips:
                clip = safe_open_video_clip(p)
                opened_clips[p] = clip
                unique_opened_clips.append(clip)
            return opened_clips[p]

        loaded_shots = []
        for r_shot in reconciled_shots:
            s_idx = r_shot.get("shot_index") if isinstance(r_shot, dict) else r_shot.shot_index
            s_start_frame = r_shot.get("start_frame") if isinstance(r_shot, dict) else r_shot.start_frame
            s_end_frame = r_shot.get("end_frame") if isinstance(r_shot, dict) else r_shot.end_frame
            s_allocated = r_shot.get("allocated_frames") if isinstance(r_shot, dict) else r_shot.allocated_frames
            s_dur = s_allocated / float(fps)

            target_path = shot_to_path.get(s_idx)
            is_fallback = False
            fallback_source = None

            if not target_path or not os.path.exists(target_path):
                is_fallback = True
                preceding = [idx for idx in available_shot_indices if idx < s_idx]
                subsequent = [idx for idx in available_shot_indices if idx > s_idx]

                if preceding:
                    chosen_idx = max(preceding)
                    target_path = shot_to_path[chosen_idx]
                    fallback_source = f"shot_{chosen_idx}_extended"
                elif subsequent:
                    chosen_idx = min(subsequent)
                    target_path = shot_to_path[chosen_idx]
                    fallback_source = f"shot_{chosen_idx}_borrowed"
                elif scene_fallback_path and os.path.exists(scene_fallback_path):
                    target_path = scene_fallback_path
                    fallback_source = "scene_fallback_media"
                else:
                    raise FileNotFoundError(f"Cannot resolve fallback for missing shot {s_idx} in scene {scene_dict.get('scene_order', 1)}")

            is_vid = target_path.lower().endswith(".mp4") or scene_dict.get("media_type") == "video"
            if is_vid:
                raw_clip = get_or_open_clip(target_path)
                aligned_clip = loop_video_clip_if_needed(raw_clip, s_dur)
                loaded_shots.append({
                    "shot_index": s_idx,
                    "start_frame": s_start_frame,
                    "end_frame": s_end_frame,
                    "allocated_frames": s_allocated,
                    "duration_seconds": s_dur,
                    "is_video": True,
                    "video_clip": aligned_clip,
                    "is_fallback": is_fallback,
                    "fallback_source": fallback_source,
                    "media_path": target_path
                })
            else:
                img = Image.open(target_path).convert("RGB")
                canvas = prep_16_9_canvas(img, (1920, 1080))
                loaded_shots.append({
                    "shot_index": s_idx,
                    "start_frame": s_start_frame,
                    "end_frame": s_end_frame,
                    "allocated_frames": s_allocated,
                    "duration_seconds": s_dur,
                    "is_video": False,
                    "canvas": canvas,
                    "is_fallback": is_fallback,
                    "fallback_source": fallback_source,
                    "media_path": target_path
                })

        total_frames = reconciled_shots[-1].get("end_frame") if isinstance(reconciled_shots[-1], dict) else reconciled_shots[-1].end_frame

        return {
            "is_multi_shot": True,
            "shots": loaded_shots,
            "total_frames": total_frames,
            "fps": fps,
            "exact_duration": duration,
            "opened_clips": unique_opened_clips
        }

    # 2. LEGACY BRANCHES (Scenes >8s multi-clip, single video, or Ken Burns photo)
    media_paths = scene_dict.get("media_paths")

    if media_paths and isinstance(media_paths, list) and len(media_paths) >= 2:
        split_t = duration / 2.0
        dur_a = split_t
        dur_b = duration - split_t

        clip_a = safe_open_video_clip(media_paths[0])
        time.sleep(0.1)
        clip_b = safe_open_video_clip(media_paths[1])

        aligned_a = loop_video_clip_if_needed(clip_a, dur_a)
        aligned_b = loop_video_clip_if_needed(clip_b, dur_b)

        return {
            "is_multi_clip": True,
            "clips_list": [aligned_a, aligned_b],
            "split_t": split_t,
            "media_paths": media_paths
        }

    media_path = (
        scene_dict.get("media_path")
        or scene_dict.get("image_path")
        or scene_dict.get("video_path")
        or (media_paths[0] if (media_paths and isinstance(media_paths, list) and len(media_paths) > 0) else None)
    )
    media_type = scene_dict.get("media_type")

    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError(f"Media asset missing for scene {scene_dict.get('scene_order')}: {media_path}")

    is_video = (media_type == "video") or media_path.lower().endswith(".mp4")

    if is_video:
        raw_v_clip = safe_open_video_clip(media_path)
        aligned_v_clip = loop_video_clip_if_needed(raw_v_clip, duration)
        return {
            "is_video": True,
            "video_clip": aligned_v_clip,
            "media_path": media_path
        }
    else:
        img = Image.open(media_path).convert("RGB")
        canvas = prep_16_9_canvas(img, (1920, 1080))
        return {
            "is_video": False,
            "canvas": canvas,
            "media_path": media_path
        }

def apply_edge_fades(
    pcm_chunk: np.ndarray,
    sample_rate: int = 44100,
    fade_ms: float = 50.0
) -> np.ndarray:
    """
    Apply smooth linear fade-in and fade-out to audio chunk to prevent boundary clicks.
    Handles short clips (< 100ms) safely by scaling fade length to at most half the chunk length.
    Preserves audio pitch and spoken duration.
    """
    if pcm_chunk is None or len(pcm_chunk) == 0:
        return pcm_chunk

    chunk = pcm_chunk.copy()
    n_samples = len(chunk)

    target_fade_samples = int(round((fade_ms / 1000.0) * sample_rate))
    # Cap fade to at most half of the available chunk length
    fade_len = min(target_fade_samples, n_samples // 2)

    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

        if chunk.ndim == 2:
            fade_in = fade_in[:, np.newaxis]
            fade_out = fade_out[:, np.newaxis]

        chunk[:fade_len] *= fade_in
        chunk[-fade_len:] *= fade_out

    return chunk

def parse_sfx_volume(raw_volume: Any, default: float = 1.0) -> float:
    """
    Parse action-specific SFX volume safely.
    Valid range: 0.0 to 1.0.
    Missing (None) retains existing default behavior (1.0 scale factor).
    0.0 means completely silent.
    Invalid values (negative, > 1.0, non-numeric, NaN, Inf) safely fallback to default.
    """
    if raw_volume is None:
        return default
    try:
        val = float(raw_volume)
        if math.isnan(val) or math.isinf(val):
            return default
        if 0.0 <= val <= 1.0:
            return val
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return default

def mix_scene_audio_pcm(
    raw_narration_path: str,
    segment_sfx_list: List[Dict[str, Any]],
    ambient_audio_path: Optional[str],
    exact_duration: float,
    scene_order: int,
    target_dir: str = DEFAULT_RESAMPLED_DIR
) -> str:
    """
    Direct PCM numpy audio stream mixing for three independent layers:
      1. Narration — 100% authoritative primary voiceover baseline.
      2. Ambience — Quiet environmental background audio (~10% relative amplitude).
      3. Action SFX — Event-triggered sound effects at precise offsets with ~50ms edge fades
         and per-event volume control.
    Pre-mixes audio into a single clean 44.1kHz 16-bit stereo PCM WAV file.
    Eliminates MoviePy audio tree recursion and float64 memory allocation errors.
    """
    os.makedirs(target_dir, exist_ok=True)
    output_wav = os.path.join(target_dir, f"scene_{scene_order}_mixed_audio.wav")

    resample_narr = ensure_resampled_audio(raw_narration_path, target_dir=target_dir)

    with wave.open(resample_narr, "rb") as wf:
        narr_bytes = wf.readframes(wf.getnframes())
        narr_pcm = np.frombuffer(narr_bytes, dtype=np.int16).astype(np.float32).reshape(-1, 2)

    target_samples = int(round(exact_duration * 44100))
    mixed_pcm = np.zeros((target_samples, 2), dtype=np.float32)

    # 1. LAYER 1: Narration (100% authoritative primary layer)
    narr_len = min(target_samples, len(narr_pcm))
    mixed_pcm[:narr_len] = narr_pcm[:narr_len]

    # Reference loudness anchor from pure narration
    rms_narr = float(np.sqrt(np.mean(narr_pcm[:narr_len] ** 2))) if narr_len > 0 else 1000.0
    if rms_narr < 1.0 or math.isnan(rms_narr):
        rms_narr = 1000.0

    max_gain_cap = 1.50

    # 2. LAYER 2: Ambience (Quiet environmental background, independent of SFX)
    amb_base_ratio = 0.10  # ~10% of narration RMS (quiet environmental bed)
    if ambient_audio_path and os.path.exists(ambient_audio_path):
        amb_resample_wav = ensure_resampled_audio(ambient_audio_path, target_dir=target_dir)
        with wave.open(amb_resample_wav, "rb") as wf:
            amb_bytes = wf.readframes(wf.getnframes())
            amb_pcm = np.frombuffer(amb_bytes, dtype=np.int16).astype(np.float32).reshape(-1, 2)

        if len(amb_pcm) > 0:
            if len(amb_pcm) < target_samples:
                num_loops = int(np.ceil(target_samples / float(len(amb_pcm))))
                amb_pcm = np.tile(amb_pcm, (num_loops, 1))

            amb_chunk = amb_pcm[:target_samples].copy()
            rms_amb = float(np.sqrt(np.mean(amb_chunk ** 2))) if amb_chunk.size > 0 else 1000.0
            amb_vol_mult = (rms_narr / max(10.0, rms_amb)) * amb_base_ratio
            amb_vol_mult = min(max_gain_cap, amb_vol_mult)

            # Silence if ambient clip is imperceptible (< 5% narration even at max cap)
            if (rms_amb * max_gain_cap) / rms_narr < 0.05:
                amb_vol_mult = 0.0

            if amb_vol_mult > 0.0:
                peak_val = float(np.max(np.abs(amb_chunk))) if amb_chunk.size > 0 else 1.0
                if peak_val * amb_vol_mult > 24000.0:
                    amb_vol_mult = 24000.0 / peak_val
                amb_vol_mult = max(0.01, min(max_gain_cap, amb_vol_mult))

                # Smooth edge fade on ambient loop boundary
                faded_amb = apply_edge_fades(amb_chunk, sample_rate=44100, fade_ms=50.0)
                mixed_pcm[:target_samples] += faded_amb * amb_vol_mult

    # 3. LAYER 3: Action SFX (Event-triggered sounds at precise offsets)
    sfx_base_ratio = 0.20  # ~20% of narration RMS
    if segment_sfx_list:
        for seg_info in segment_sfx_list:
            seg_path = seg_info.get("audio_path")
            start_off = float(seg_info.get("start_offset", 0.0))
            seg_dur = float(seg_info.get("duration", exact_duration))

            # Read action-specific volume (0.0 to 1.0)
            raw_vol = seg_info.get("volume") if "volume" in seg_info else seg_info.get("sfx_volume")
            event_vol = parse_sfx_volume(raw_vol, default=1.0)

            # 0.0 volume means silent
            if event_vol <= 0.0:
                continue

            if seg_path and os.path.exists(seg_path):
                seg_resample_wav = ensure_resampled_audio(seg_path, target_dir=target_dir)
                with wave.open(seg_resample_wav, "rb") as wf:
                    seg_bytes = wf.readframes(wf.getnframes())
                    seg_pcm = np.frombuffer(seg_bytes, dtype=np.int16).astype(np.float32).reshape(-1, 2)

                if len(seg_pcm) == 0:
                    continue

                seg_dur_samples = int(round(seg_dur * 44100))
                if len(seg_pcm) < seg_dur_samples:
                    num_loops = int(np.ceil(seg_dur_samples / float(len(seg_pcm))))
                    seg_pcm = np.tile(seg_pcm, (num_loops, 1))

                seg_chunk = seg_pcm[:seg_dur_samples].copy()
                rms_seg = float(np.sqrt(np.mean(seg_chunk ** 2))) if seg_chunk.size > 0 else 1000.0

                # Target amplitude scaled by per-event volume (applied once here)
                target_amp_ratio = sfx_base_ratio * event_vol
                vol_mult = (rms_narr / max(10.0, rms_seg)) * target_amp_ratio
                vol_mult = min(max_gain_cap, vol_mult)

                # Silence if clip is too quiet even at 1.5x cap (< 5% narration amplitude)
                if (rms_seg * max_gain_cap) / rms_narr < 0.05:
                    vol_mult = 0.0

                if vol_mult > 0.0:
                    # Peak limiting for SFX headroom (max 26000 on int16 scale)
                    peak_val = float(np.max(np.abs(seg_chunk))) if seg_chunk.size > 0 else 1.0
                    if peak_val * vol_mult > 26000.0:
                        vol_mult = 26000.0 / peak_val
                    vol_mult = max(0.001, min(max_gain_cap, vol_mult))

                    # Apply ~50ms smooth linear fade-in and fade-out to prevent clicks
                    faded_sfx = apply_edge_fades(seg_chunk, sample_rate=44100, fade_ms=50.0)

                    start_sample = max(0, int(round(start_off * 44100)))
                    end_sample = min(target_samples, start_sample + seg_dur_samples)

                    available_len = end_sample - start_sample
                    if available_len > 0 and start_sample < target_samples:
                        chunk_to_add = faded_sfx[:available_len].copy()
                        # If truncated by scene boundary, fade-out the truncated end safely
                        if available_len < seg_dur_samples:
                            trunc_fade = min(2205, available_len // 2)
                            if trunc_fade > 0:
                                fade_out_ramp = np.linspace(1.0, 0.0, trunc_fade, dtype=np.float32)[:, np.newaxis]
                                chunk_to_add[-trunc_fade:] *= fade_out_ramp

                        mixed_pcm[start_sample:end_sample] += chunk_to_add * vol_mult

    # Master mix protection: prevent clipping distortion when layers overlap
    max_peak = float(np.max(np.abs(mixed_pcm))) if mixed_pcm.size > 0 else 0.0
    if max_peak > 32000.0:
        mixed_pcm = mixed_pcm * (32000.0 / max_peak)

    # Sanitize any nonfinite values (NaN / Inf)
    mixed_pcm = np.nan_to_num(mixed_pcm, copy=False, nan=0.0, posinf=32000.0, neginf=-32000.0)

    mixed_int16 = np.clip(mixed_pcm, -32768, 32767).astype(np.int16)

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(mixed_int16.tobytes())

    return output_wav

def close_visual_data(visual_data: Optional[Dict[str, Any]]) -> None:
    """Explicitly close all open video clip handles in a visual data dictionary."""
    if not visual_data:
        return
    if visual_data.get("is_multi_shot"):
        for c in visual_data.get("opened_clips", []):
            try:
                c.close()
            except Exception:
                pass
        for s in visual_data.get("shots", []):
            try:
                if s.get("is_video") and s.get("video_clip"):
                    s["video_clip"].close()
            except Exception:
                pass
    elif visual_data.get("is_multi_clip"):
        for c in visual_data.get("clips_list", []):
            try:
                c.close()
            except Exception:
                pass
    elif visual_data.get("is_video"):
        try:
            visual_data["video_clip"].close()
        except Exception:
            pass

def create_scene_clip(
    scene_dict: Dict[str, Any],
    next_scene_dict: Optional[Dict[str, Any]] = None,
    crossfade_duration: float = 0.4,
    target_size: Tuple[int, int] = (1280, 720),
    fps: int = 24
) -> VideoClip:
    """
    Create a standalone single scene VideoClip with subtitles and lookahead crossfade.
    (Maintained for single-scene testing / backward compatibility).
    """
    scene_text = scene_dict["scene_text"]
    orig_duration = float(scene_dict["duration"])
    exact_duration = compute_frame_exact_duration(orig_duration, fps=fps)

    scene_visual = load_scene_visual_data(scene_dict, exact_duration, fps=float(fps))
    next_scene_visual = load_scene_visual_data(next_scene_dict, exact_duration, fps=float(fps)) if next_scene_dict else None

    sub_chunks = split_text_into_subtitles(scene_text)
    sub_timeline = build_subtitle_timeline(sub_chunks, exact_duration)

    def make_frame(t: float) -> np.ndarray:
        return render_scene_frame(
            scene_visual=scene_visual,
            next_scene_visual=next_scene_visual,
            t=t,
            duration=exact_duration,
            subtitle_timeline=sub_timeline,
            crossfade_duration=crossfade_duration,
            target_size=target_size
        )

    clip = VideoClip(make_frame, duration=exact_duration)
    orig_close = clip.close

    def _cleanup_and_close():
        orig_close()
        close_visual_data(scene_visual)
        if next_scene_visual:
            close_visual_data(next_scene_visual)

    clip.close = _cleanup_and_close
    return clip

def assemble_video(
    scenes: List[Dict[str, Any]],
    output_path: str,
    fps: int = 24,
    crossfade_duration: float = 0.4,
    resampled_dir: Optional[str] = None
) -> str:
    """
    Assemble list of scene dicts into a single continuous MP4 video using a sliding-window clip reader:
    - Aligns every scene to ceiling 24fps duration (preserving untrimmed narration audio).
    - Supports multiple short video clips per scene (Part B for scenes > 8s).
    - Composites balanced action SFX at ~20% relative amplitude (capped at max 1.50x gain).
    - Sliding-Window Clip Management: Only holds current scene + next scene visual clips open (<= 4 clips max).
    - Concatenates into MP4 (H.264 video, AAC audio, 1280x720 res).
    - Project-Specific Audio Isolation: Optional resampled_dir isolates all intermediate audio WAV files.
    """
    if not scenes:
        raise ValueError("Cannot assemble video with empty scenes list.")

    target_resampled_dir = resampled_dir or DEFAULT_RESAMPLED_DIR

    sorted_scenes = sorted(scenes, key=lambda x: x["scene_order"])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    scene_timeline = []
    full_audio_pcm_list = []
    cum_t = 0.0

    num_scenes = len(sorted_scenes)
    for idx in range(num_scenes):
        sc = sorted_scenes[idx]
        raw_narration_path = sc["audio_path"]
        orig_duration = float(sc["duration"])
        exact_duration = compute_frame_exact_duration(orig_duration, fps=fps)
        scene_order = sc.get("scene_order", idx + 1)
        segment_sfx_list = sc.get("segment_sfx_list") or []
        ambient_audio_path = sc.get("ambient_audio_path") or sc.get("sound_path")

        st_t = cum_t
        end_t = cum_t + exact_duration
        cum_t = end_t

        # Mix scene PCM audio into project-specific resampled directory
        mixed_wav = mix_scene_audio_pcm(
            raw_narration_path=raw_narration_path,
            segment_sfx_list=segment_sfx_list,
            ambient_audio_path=ambient_audio_path,
            exact_duration=exact_duration,
            scene_order=scene_order,
            target_dir=target_resampled_dir
        )

        with wave.open(mixed_wav, "rb") as wf:
            raw_b = wf.readframes(wf.getnframes())
            full_audio_pcm_list.append(np.frombuffer(raw_b, dtype=np.int16))

        sub_chunks = split_text_into_subtitles(sc["scene_text"])
        sub_timeline = build_subtitle_timeline(sub_chunks, exact_duration)

        scene_timeline.append({
            "index": idx,
            "scene_dict": sc,
            "exact_duration": exact_duration,
            "start_t": st_t,
            "end_t": end_t,
            "subtitle_timeline": sub_timeline
        })

    total_duration = cum_t

    # Write unified full-length audio WAV file
    full_audio_wav = os.path.join(os.path.dirname(os.path.abspath(output_path)), "full_render_audio.wav")
    full_pcm = np.concatenate(full_audio_pcm_list)
    del full_audio_pcm_list
    with wave.open(full_audio_wav, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(full_pcm.tobytes())

    gc.collect()
    full_audio_clip = AudioFileClip(full_audio_wav)

    # Sliding-window visual cache: holds at most 2 scenes (<= 4 clip handles total)
    active_visual_cache: Dict[int, Dict[str, Any]] = {}

    def get_scene_visual_cached(scene_idx: int) -> Dict[str, Any]:
        if scene_idx not in active_visual_cache:
            info = scene_timeline[scene_idx]
            active_visual_cache[scene_idx] = load_scene_visual_data(info["scene_dict"], info["exact_duration"], fps=float(fps))
        return active_visual_cache[scene_idx]

    def make_master_frame(t: float) -> np.ndarray:
        cur_idx = len(scene_timeline) - 1
        for st_info in scene_timeline:
            if st_info["start_t"] <= t < st_info["end_t"]:
                cur_idx = st_info["index"]
                break

        # Evict older scenes to maintain constant <= 4 open clip handles
        for old_idx in list(active_visual_cache.keys()):
            if old_idx < cur_idx:
                close_visual_data(active_visual_cache.pop(old_idx, None))
                gc.collect()

        cur_info = scene_timeline[cur_idx]
        local_t = max(0.0, min(cur_info["exact_duration"], t - cur_info["start_t"]))

        scene_vis = get_scene_visual_cached(cur_idx)
        next_vis = get_scene_visual_cached(cur_idx + 1) if (cur_idx + 1 < len(scene_timeline)) else None

        return render_scene_frame(
            scene_visual=scene_vis,
            next_scene_visual=next_vis,
            t=local_t,
            duration=cur_info["exact_duration"],
            subtitle_timeline=cur_info["subtitle_timeline"],
            crossfade_duration=crossfade_duration,
            target_size=(1280, 720)
        )

    tmp_output_path = output_path.replace(".mp4", ".tmp.mp4") if output_path.endswith(".mp4") else (output_path + ".tmp.mp4")
    if os.path.exists(tmp_output_path):
        try:
            os.remove(tmp_output_path)
        except Exception:
            pass

    master_video = None
    try:
        master_video = VideoClip(make_master_frame, duration=total_duration)
        master_video = master_video.with_audio(full_audio_clip)

        master_video.write_videofile(
            tmp_output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=2,
            ffmpeg_params=["-f", "mp4"],
            logger=None
        )

        master_video.close()
        full_audio_clip.close()

        # Verify container integrity and moov atom presence before renaming
        try:
            test_c = av.open(tmp_output_path)
            if not test_c.streams.video or not test_c.streams.audio:
                raise ValueError("Generated MP4 is missing video or audio stream.")
            test_c.close()
        except Exception as err:
            raise RuntimeError(f"Rendered video failed container/moov integrity check: {err}")

        # Atomically rename to final output filename
        if os.path.exists(output_path):
            os.remove(output_path)
        os.replace(tmp_output_path, output_path)

        return os.path.abspath(output_path)

    finally:
        if master_video:
            try:
                master_video.close()
            except Exception:
                pass
        try:
            full_audio_clip.close()
        except Exception:
            pass
        for vis in active_visual_cache.values():
            close_visual_data(vis)
        active_visual_cache.clear()
        gc.collect()

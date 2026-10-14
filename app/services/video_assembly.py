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

def ensure_resampled_audio(audio_path: str, target_dir: str = "c:/AiVideoGenerator/media/audio_resampled") -> str:
    """
    Resample input audio file to 44100 Hz stereo 16-bit PCM WAV using PyAV.
    Preserves real-world pitch and spoken duration exactly.
    """
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(audio_path))[0] + "_44k.wav"
    output_wav = os.path.join(target_dir, filename)

    if os.path.exists(output_wav) and os.path.getsize(output_wav) > 1000:
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

def get_single_visual_frame(
    visual_data: Dict[str, Any],
    t: float,
    duration: float,
    target_size: Tuple[int, int] = (1280, 720)
) -> Image.Image:
    """Render raw visual frame at time t for a scene (supporting multi-clip cuts and static fallbacks)."""
    target_w, target_h = target_size

    if visual_data.get("is_multi_clip"):
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

def load_scene_visual_data(scene_dict: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Load visual data for a scene (supporting multi-clip lists or single clip/image)."""
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

def mix_scene_audio_pcm(
    raw_narration_path: str,
    segment_sfx_list: List[Dict[str, Any]],
    ambient_audio_path: Optional[str],
    exact_duration: float,
    scene_order: int,
    target_dir: str = "c:/AiVideoGenerator/media/audio_resampled"
) -> str:
    """
    Direct PCM numpy audio stream mixing for scene narration + segment action SFX.
    Pre-mixes audio into a single clean 44.1kHz 16-bit PCM WAV file.
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

    # Place 100% authoritative narration audio
    narr_len = min(target_samples, len(narr_pcm))
    mixed_pcm[:narr_len] = narr_pcm[:narr_len]

    rms_narr = float(np.sqrt(np.mean(mixed_pcm ** 2))) if mixed_pcm.size > 0 else 1000.0
    if rms_narr < 1.0:
        rms_narr = 1000.0

    # Composite segment-level action SFX at ~20% relative amplitude (15-25% target range) with 1.50x max gain cap
    target_amp_ratio = 0.20
    max_gain_cap = 1.50

    if segment_sfx_list:
        for seg_info in segment_sfx_list:
            seg_path = seg_info.get("audio_path")
            start_off = float(seg_info.get("start_offset", 0.0))
            seg_dur = float(seg_info.get("duration", exact_duration))

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

                seg_chunk = seg_pcm[:seg_dur_samples]
                rms_seg = float(np.sqrt(np.mean(seg_chunk ** 2))) if seg_chunk.size > 0 else 1000.0
                vol_mult = (rms_narr / max(10.0, rms_seg)) * target_amp_ratio
                vol_mult = min(max_gain_cap, vol_mult)

                # If source clip is too quiet even at 1.5x cap (< 5% narration amplitude), fallback to silence
                if (rms_seg * max_gain_cap) / rms_narr < 0.05:
                    vol_mult = 0.0

                if vol_mult > 0.0:
                    # Peak limiting to protect against clipping (max headroom 28000 on int16 scale)
                    peak_val = float(np.max(np.abs(seg_chunk))) if seg_chunk.size > 0 else 1.0
                    if peak_val * vol_mult > 28000.0:
                        vol_mult = 28000.0 / peak_val
                    vol_mult = max(0.02, min(max_gain_cap, vol_mult))

                    start_sample = int(round(start_off * 44100))
                    end_sample = min(target_samples, start_sample + seg_dur_samples)

                    available_len = end_sample - start_sample
                    if available_len > 0 and start_sample < target_samples:
                        mixed_pcm[start_sample:end_sample] += seg_chunk[:available_len] * vol_mult

    elif ambient_audio_path and os.path.exists(ambient_audio_path):
        amb_resample_wav = ensure_resampled_audio(ambient_audio_path, target_dir=target_dir)
        with wave.open(amb_resample_wav, "rb") as wf:
            amb_bytes = wf.readframes(wf.getnframes())
            amb_pcm = np.frombuffer(amb_bytes, dtype=np.int16).astype(np.float32).reshape(-1, 2)

        if len(amb_pcm) > 0:
            if len(amb_pcm) < target_samples:
                num_loops = int(np.ceil(target_samples / float(len(amb_pcm))))
                amb_pcm = np.tile(amb_pcm, (num_loops, 1))

            amb_chunk = amb_pcm[:target_samples]
            rms_amb = float(np.sqrt(np.mean(amb_chunk ** 2))) if amb_chunk.size > 0 else 1000.0
            vol_mult = (rms_narr / max(10.0, rms_amb)) * target_amp_ratio
            vol_mult = min(max_gain_cap, vol_mult)

            if (rms_amb * max_gain_cap) / rms_narr < 0.05:
                vol_mult = 0.0

            if vol_mult > 0.0:
                peak_val = float(np.max(np.abs(amb_chunk))) if amb_chunk.size > 0 else 1.0
                if peak_val * vol_mult > 28000.0:
                    vol_mult = 28000.0 / peak_val
                vol_mult = max(0.02, min(max_gain_cap, vol_mult))

                mixed_pcm[:target_samples] += amb_chunk * vol_mult

    mixed_int16 = np.clip(mixed_pcm, -32768, 32767).astype(np.int16)

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(mixed_int16.tobytes())

    return output_wav

def create_scene_clip(
    scene_dict: Dict[str, Any],
    next_scene_dict: Optional[Dict[str, Any]] = None,
    crossfade_duration: float = 0.4,
    target_size: Tuple[int, int] = (1280, 720),
    fps: float = 24.0
) -> VideoClip:
    """
    Create MoviePy VideoClip for a single scene beat:
    - Calculates ceiling frame-exact duration for 24fps (preserving full narration audio).
    - Composites 100% authoritative narration audio with secondary 15% volume ambient sound track (if present).
    - Supports multiple short clips per scene (Part B for scenes > 8s).
    """
    raw_narration_path = scene_dict["audio_path"]
    scene_text = scene_dict["scene_text"]
    orig_duration = float(scene_dict["duration"])

    if not os.path.exists(raw_narration_path):
        raise FileNotFoundError(f"Scene narration audio file missing: {raw_narration_path}")

def close_visual_data(visual_data: Optional[Dict[str, Any]]) -> None:
    """Explicitly close all open video clip handles in a visual data dictionary."""
    if not visual_data:
        return
    if visual_data.get("is_multi_clip"):
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

    scene_visual = load_scene_visual_data(scene_dict, exact_duration)
    next_scene_visual = load_scene_visual_data(next_scene_dict, exact_duration) if next_scene_dict else None

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

    return VideoClip(make_frame, duration=exact_duration)

def assemble_video(scenes: List[Dict[str, Any]], output_path: str, fps: int = 24, crossfade_duration: float = 0.4) -> str:
    """
    Assemble list of scene dicts into a single continuous MP4 video using a sliding-window clip reader:
    - Aligns every scene to ceiling 24fps duration (preserving untrimmed narration audio).
    - Supports multiple short video clips per scene (Part B for scenes > 8s).
    - Composites balanced action SFX at ~20% relative amplitude (capped at max 1.50x gain).
    - Sliding-Window Clip Management: Only holds current scene + next scene visual clips open (<= 4 clips max).
    - Concatenates into MP4 (H.264 video, AAC audio, 1280x720 res).
    """
    if not scenes:
        raise ValueError("Cannot assemble video with empty scenes list.")

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

        # Mix scene PCM audio
        mixed_wav = mix_scene_audio_pcm(
            raw_narration_path=raw_narration_path,
            segment_sfx_list=segment_sfx_list,
            ambient_audio_path=ambient_audio_path,
            exact_duration=exact_duration,
            scene_order=scene_order
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
            active_visual_cache[scene_idx] = load_scene_visual_data(info["scene_dict"], info["exact_duration"])
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

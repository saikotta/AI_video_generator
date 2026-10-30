import os
import sys
import time
import uuid
import json
import shutil
import hashlib
import traceback
import subprocess
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set

# Ensure stdout handles UTF-8 on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\AiVideoGenerator")

import av
import numpy as np
from PIL import Image
from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.enums import ProjectStatus, JobStatus, AssetType
from app.models.project import Project
from app.models.user import User
from app.models.scene import Scene
from app.models.scene_asset import SceneAsset
from app.models.pipeline_job import PipelineJob
from app.services.pipeline_orchestrator import (
    VideoPipelineOrchestrator,
    _PROCESS_PIPELINE_LOCK
)


def compute_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


async def run_preflight_checks() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("[PART 1] PREFLIGHT SAFETY & ENVIRONMENT CHECKS")
    print("=" * 80)

    # 1. Verify working directory
    cwd = os.getcwd()
    print(f"  [Preflight 1/7] Working Directory: {cwd}")
    if os.path.normpath(cwd).lower() != os.path.normpath(r"C:\AiVideoGenerator").lower():
        raise RuntimeError(f"Invalid working directory: {cwd}. Expected C:\\AiVideoGenerator.")

    # 2. Verify disk space
    total, used, free = shutil.disk_usage(r"C:\\")
    free_gb = free / (1024 ** 3)
    print(f"  [Preflight 2/7] Available Disk Space: {free_gb:.2f} GB free")
    if free_gb < 3.0:
        raise RuntimeError(f"Low disk space ({free_gb:.2f} GB). Minimum 3.0 GB required.")

    # 3. Verify Python environment
    print(f"  [Preflight 3/7] Python Executable: {sys.executable}")
    print(f"                 PyAV: {av.__version__} | NumPy: {np.__version__}")

    # 4. Verify PostgreSQL connectivity and test user
    test_user_id = None
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(User.id, User.email).limit(1))
        user = res.first()
        if not user:
            raise RuntimeError("No user records found in PostgreSQL database.")
        test_user_id = user.id
        print(f"  [Preflight 4/7] Database Connected: User {user.id} ({user.email})")

    # 5. Verify required credentials without printing secret values
    for name, val in [
        ("GROQ_API_KEY", settings.GROQ_API_KEY),
        ("PEXELS_API_KEY", settings.PEXELS_API_KEY),
        ("DATABASE_URL", settings.DATABASE_URL)
    ]:
        if not val or not val.strip():
            raise RuntimeError(f"Missing required credential: {name}")
        print(f"  [Preflight 5/7] Credential Present: {name} (length: {len(val.strip())})")

    if settings.FREESOUND_API_KEY and settings.FREESOUND_API_KEY.strip():
        print(f"                 Credential Present: FREESOUND_API_KEY (length: {len(settings.FREESOUND_API_KEY.strip())})")
    else:
        print("                 FREESOUND_API_KEY not configured (fallback enabled)")

    # 6. Verify provider configuration & pipeline entrypoint
    print(f"  [Preflight 6/7] Providers:")
    print(f"                 Script: {settings.SCRIPT_PROVIDER} (model: {settings.GROQ_MODEL})")
    print(f"                 TTS Voice: {settings.TTS_VOICE}")
    print(f"                 Media: Pexels API | SFX: Freesound API")
    if _PROCESS_PIPELINE_LOCK.locked():
        raise RuntimeError("Process pipeline lock is already held.")
    if not hasattr(VideoPipelineOrchestrator, "run_pipeline"):
        raise RuntimeError("VideoPipelineOrchestrator missing run_pipeline method.")
    print("                 Pipeline Lock: Free | run_pipeline entrypoint: Verified")

    # 7. Verify protected reference media files and record SHA256 hashes
    ref_files = [
        r"C:\AiVideoGenerator\media\output_5scenes.mp4",
        r"C:\AiVideoGenerator\media\output_27scenes.mp4",
        r"C:\AiVideoGenerator\media\full_render_audio.wav",
        r"C:\AiVideoGenerator\media\projects\cfa3f995-c9dd-4ecb-9c5b-11b8d8534e41\video_cfa3f995-c9dd-4ecb-9c5b-11b8d8534e41.mp4",
        r"C:\AiVideoGenerator\media\projects\cfa3f995-c9dd-4ecb-9c5b-11b8d8534e41\full_render_audio.wav",
        r"C:\AiVideoGenerator\media\projects\666cf9ab-f9a4-4e92-9840-bb93495db262\video_666cf9ab-f9a4-4e92-9840-bb93495db262.mp4",
        r"C:\AiVideoGenerator\media\projects\666cf9ab-f9a4-4e92-9840-bb93495db262\full_render_audio.wav"
    ]
    preflight_hashes = {}
    print("  [Preflight 7/7] Locking Protected Reference Media:")
    for f in ref_files:
        if not os.path.exists(f):
            raise RuntimeError(f"Protected reference file missing: {f}")
        h = compute_sha256(f)
        preflight_hashes[f] = h
        print(f"                 • {os.path.basename(f)} (SHA-256: {h[:16]}...)")

    print("\n  >>> ALL PREFLIGHT CHECKS PASSED SUCCESSFULLY. READY FOR LIVE RUN. <<<")
    return {
        "test_user_id": test_user_id,
        "ref_files": ref_files,
        "preflight_hashes": preflight_hashes
    }


async def main():
    print("=" * 80)
    print("PHASE B — STEP 11D: CONTROLLED LIVE MULTI-SHOT VIDEO GENERATION")
    print("Topic: 'How to prepare a simple vegetable stir-fry'")
    print("Requested Duration: 20 seconds")
    print("=" * 80)

    # =========================================================================
    # PART 1: PREFLIGHT
    # =========================================================================
    preflight_data = await run_preflight_checks()
    test_user_id = preflight_data["test_user_id"]
    ref_files = preflight_data["ref_files"]
    preflight_hashes = preflight_data["preflight_hashes"]

    # =========================================================================
    # PART 2: CREATE EXACTLY ONE NEW PROJECT RECORD
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PART 2] REGISTERING EXACTLY ONE FRESH PROJECT RECORD")
    print("=" * 80)
    new_project_id = uuid.uuid4()
    topic = "How to prepare a simple vegetable stir-fry"
    desired_duration_seconds = 20
    now = datetime.now(timezone.utc)

    project_workspace_dir = os.path.join(r"C:\AiVideoGenerator\media\projects", str(new_project_id))
    if os.path.exists(project_workspace_dir):
        raise RuntimeError(f"Workspace unexpectedly exists for fresh UUID: {project_workspace_dir}")

    new_project = Project(
        id=new_project_id,
        user_id=test_user_id,
        topic=topic,
        desired_duration_seconds=desired_duration_seconds,
        status=ProjectStatus.QUEUED,
        progress_percent=0,
        current_stage="Queued for controlled generation",
        created_at=now,
        updated_at=now
    )

    async with AsyncSessionLocal() as session:
        session.add(new_project)
        await session.commit()

    print(f"  ✓ Fresh Project Registered:")
    print(f"    - Project ID: {new_project_id}")
    print(f"    - Topic: '{topic}'")
    print(f"    - Requested Duration: {desired_duration_seconds}s")
    print(f"    - Workspace: media/projects/{new_project_id}/")

    # =========================================================================
    # PART 3: EXECUTE PIPELINE (EXACTLY ONE LIVE RUN)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PART 3] EXECUTING SEVEN-STAGE PIPELINE (ONE LIVE RUN)")
    print("=" * 80)
    orchestrator = VideoPipelineOrchestrator()
    t_start = time.perf_counter()
    execution_success = False
    final_video_path = None
    pipeline_error = None

    try:
        final_video_path = await orchestrator.run_pipeline(new_project_id)
        execution_success = True
        t_elapsed = time.perf_counter() - t_start
        print(f"\n  ✓ Pipeline completed successfully in {t_elapsed:.2f}s!")
        print(f"    Final Rendered Video: {final_video_path}")
    except Exception as e:
        t_elapsed = time.perf_counter() - t_start
        pipeline_error = str(e)
        tb = traceback.format_exc()
        print(f"\n  ✗ Pipeline execution failed in {t_elapsed:.2f}s: {pipeline_error}")
        print(f"\nTraceback:\n{tb}")

    # =========================================================================
    # PART 4: COLLECT DIAGNOSTICS FROM DATABASE & WORKSPACE
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PART 4] COLLECTING STAGE DIAGNOSTICS")
    print("=" * 80)

    async with AsyncSessionLocal() as session:
        p_res = await session.execute(select(Project).where(Project.id == new_project_id))
        proj_row = p_res.scalars().first()

        s_res = await session.execute(select(Scene).where(Scene.project_id == new_project_id).order_by(Scene.scene_order))
        scenes_rows = s_res.scalars().all()

        j_res = await session.execute(select(PipelineJob).where(PipelineJob.project_id == new_project_id).order_by(PipelineJob.created_at))
        jobs_rows = j_res.scalars().all()

        scene_ids = [s.id for s in scenes_rows]
        assets_rows = []
        if scene_ids:
            a_res = await session.execute(select(SceneAsset).where(SceneAsset.scene_id.in_(scene_ids)).order_by(SceneAsset.created_at))
            assets_rows = a_res.scalars().all()

    proj_status_val = getattr(proj_row.status, 'value', str(proj_row.status)) if proj_row else 'None'
    script_text = proj_row.script if (proj_row and proj_row.script) else ""
    script_words = len(script_text.split())

    print(f"\n  Database Project Summary:")
    print(f"    - Final Status: {proj_status_val}")
    print(f"    - Current Stage: {proj_row.current_stage if proj_row else 'None'}")
    print(f"    - Progress: {proj_row.progress_percent if proj_row else 0}%")
    print(f"    - Script Words: {script_words}")
    print(f"    - Generated Script:\n      \"{script_text}\"")

    print(f"\n  Pipeline Job Stages ({len(jobs_rows)} stages):")
    stages_status = {}
    stage_durations = {}
    for j in jobs_rows:
        dur_sec = (j.completed_at - j.started_at).total_seconds() if (j.started_at and j.completed_at) else None
        st_val = getattr(j.status, 'value', str(j.status))
        stages_status[j.stage] = st_val
        stage_durations[j.stage] = round(dur_sec, 2) if dur_sec is not None else None
        dur_str = f"{dur_sec:.1f}s" if dur_sec is not None else "N/A"
        print(f"    • [{j.stage}] {st_val} (Provider: {j.provider_name}, Time: {dur_str})")
        if j.error_message:
            print(f"      Error: {j.error_message}")

    # Inspect workspace shot plan & action timeline files
    ws = orchestrator.get_project_workspace(new_project_id)
    tl_file = os.path.join(ws["project_dir"], "action_timeline.json")
    persisted_timeline = {}
    if os.path.exists(tl_file):
        try:
            with open(tl_file, "r", encoding="utf-8") as f:
                persisted_timeline = json.load(f)
        except Exception as e:
            print(f"    Could not read timeline json: {e}")

    shot_plan_file = os.path.join(ws["project_dir"], "shot_plan.json")
    persisted_shot_plans = {}
    if os.path.exists(shot_plan_file):
        try:
            with open(shot_plan_file, "r", encoding="utf-8") as f:
                persisted_shot_plans = json.load(f)
        except Exception as e:
            print(f"    Could not read shot_plan json: {e}")

    # Build scene & shot details
    scenes_detail = []
    total_measured_audio_dur = 0.0
    total_planned_shots = 0
    total_sourced_shots = 0
    total_missing_shots = 0

    print(f"\n  Scene-by-Scene Breakdown ({len(scenes_rows)} scenes):")
    for s in scenes_rows:
        s_dur = float(s.duration_seconds or 0.0)
        total_measured_audio_dur += s_dur
        kw = s.search_keywords or []
        sc_tl = persisted_timeline.get(str(s.id), [])
        sc_shot_data = persisted_shot_plans.get(str(s.id), {})
        sc_shots = sc_shot_data.get("shots") or []

        num_planned = len(sc_shots) if sc_shots else 1
        total_planned_shots += num_planned

        print(f"\n    --- Scene {s.scene_order} ({s_dur:.2f}s) ---")
        print(f"      Narration: \"{s.scene_text}\"")
        print(f"      Visual Prompt: \"{s.visual_prompt}\"")
        print(f"      Pexels Keywords: {kw}")
        print(f"      Planned Shots ({len(sc_shots)} shots):")
        for sh in sc_shots:
            print(f"        • Shot {sh.get('shot_index')}: Action: \"{sh.get('required_visible_action')}\" | Subject: {sh.get('primary_subject')} | Keywords: {sh.get('search_keywords')} | Interval: [{sh.get('frame_start')}:{sh.get('frame_end')}] ({sh.get('duration_frames')}f)")

        scenes_detail.append({
            "scene_order": s.scene_order,
            "scene_id": str(s.id),
            "scene_text": s.scene_text,
            "duration_seconds": s_dur,
            "visual_prompt": s.visual_prompt,
            "search_keywords": kw,
            "action_timeline": sc_tl,
            "shot_plan_data": sc_shot_data
        })

    # Assets breakdown
    video_assets = []
    sfx_assets = []
    voiceover_assets = []

    print(f"\n  Sourced Media Assets ({len(assets_rows)} assets):")
    for a in assets_rows:
        meta = a.asset_metadata or {}
        if a.asset_type == AssetType.VIDEO:
            v_id = meta.get("pexels_id") or meta.get("video_id")
            shot_meta = meta.get("shot_metadata", [])
            coverage_complete = meta.get("visual_coverage_complete", True)
            fallbacks_used = meta.get("fallbacks_used", [])

            num_sourced = len(shot_meta) if shot_meta else 1
            total_sourced_shots += num_sourced
            if fallbacks_used:
                total_missing_shots += len(fallbacks_used)

            v_data = {
                "scene_id": str(a.scene_id),
                "provider": a.provider_name,
                "local_path": a.local_path,
                "source_url": a.source_url,
                "pexels_id": v_id,
                "score": meta.get("score"),
                "semantic_score": meta.get("semantic_score"),
                "technical_score": meta.get("technical_score"),
                "semantic_reasons": meta.get("semantic_reasons"),
                "photographer": meta.get("photographer"),
                "shot_metadata": shot_meta,
                "media_paths": meta.get("media_paths", [a.local_path]),
                "visual_coverage_complete": coverage_complete,
                "fallbacks_used": fallbacks_used
            }
            video_assets.append(v_data)
            print(f"    • [VIDEO] Scene ID: {a.scene_id} | Sourced Shots: {num_sourced} | Coverage Complete: {coverage_complete} | Fallbacks: {fallbacks_used}")
            for sm in shot_meta:
                print(f"      - Shot {sm.get('shot_index')}: Pexels ID {sm.get('pexels_id')} | Action: \"{sm.get('action')}\" | File: {os.path.basename(sm.get('file_path') or '')} | Frames: [{sm.get('frame_start')}:{sm.get('frame_end')}]")

        elif a.asset_type == AssetType.AUDIO and a.provider_name == "freesound":
            s_data = {
                "scene_id": str(a.scene_id),
                "provider": a.provider_name,
                "local_path": a.local_path,
                "freesound_id": meta.get("freesound_id"),
                "sound_name": meta.get("name"),
                "license": meta.get("license"),
                "license_url": meta.get("license_url"),
                "action_desc": meta.get("action_description"),
                "volume": meta.get("volume"),
                "start_offset": meta.get("start_offset"),
                "duration": a.duration_seconds
            }
            sfx_assets.append(s_data)
            print(f"    • [SFX] Scene ID: {a.scene_id} | ID: {meta.get('freesound_id')} | Name: '{meta.get('name')}' | License: '{meta.get('license')}' | Offset: {meta.get('start_offset')}s | Vol: {meta.get('volume')}")

        elif a.asset_type == AssetType.AUDIO:
            voiceover_assets.append({
                "scene_id": str(a.scene_id),
                "provider": a.provider_name,
                "local_path": a.local_path,
                "duration": a.duration_seconds
            })
            print(f"    • [VOICEOVER] Scene ID: {a.scene_id} | Duration: {a.duration_seconds:.2f}s | File: {os.path.basename(a.local_path or '')}")

    # =========================================================================
    # PART 5: PYAV DECODE, TECHNICAL VALIDATION & KEYFRAME EXTRACTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PART 5] TECHNICAL MP4 VALIDATION & KEYFRAME EXTRACTION")
    print("=" * 80)
    video_metadata = {}
    audio_analysis = {}
    black_frame_count = 0
    total_frames = 0
    extracted_keyframes = []

    keyframes_dir = os.path.join(ws["project_dir"], "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    if execution_success and final_video_path and os.path.exists(final_video_path):
        file_size_bytes = os.path.getsize(final_video_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        try:
            container = av.open(final_video_path)
            v_stream = container.streams.video[0]
            a_stream = container.streams.audio[0] if container.streams.audio else None

            width = v_stream.codec_context.width
            height = v_stream.codec_context.height
            fps = float(v_stream.average_rate or 24.0)
            dur_sec = float(container.duration / av.time_base) if container.duration else 0.0

            all_video_frames = []
            frame_idx = 0
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="rgb24")
                all_video_frames.append(arr)
                lum = float(np.mean(arr))
                if lum < 2.0:
                    black_frame_count += 1
                frame_idx += 1
            total_frames = frame_idx
            container.close()

            # Decode audio in separate container
            audio_container = av.open(final_video_path)
            audio_samples_list = []
            if audio_container.streams.audio:
                for frame in audio_container.decode(audio=0):
                    audio_samples_list.append(frame.to_ndarray())
            audio_container.close()

            audio_min = 0
            audio_max = 0
            audio_rms = 0.0
            is_clipped = False

            if audio_samples_list:
                full_audio = np.concatenate(audio_samples_list, axis=1)
                audio_min = int(np.min(full_audio))
                audio_max = int(np.max(full_audio))
                audio_rms = float(np.sqrt(np.mean(full_audio.astype(np.float32) ** 2)))
                is_clipped = (audio_max >= 32767) or (audio_min <= -32768)
                audio_analysis = {
                    "sample_rate": a_stream.codec_context.sample_rate if a_stream else 44100,
                    "channels": a_stream.codec_context.channels if a_stream else 2,
                    "codec": a_stream.codec_context.name if a_stream else "unknown",
                    "total_samples": full_audio.shape[1],
                    "peak_min": audio_min,
                    "peak_max": audio_max,
                    "rms": round(audio_rms, 1),
                    "hard_clipped": is_clipped
                }

            video_metadata = {
                "file_size_bytes": file_size_bytes,
                "file_size_mb": round(file_size_mb, 2),
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "fps": round(fps, 2),
                "duration_seconds": round(dur_sec, 2),
                "total_decoded_frames": total_frames,
                "black_frame_count": black_frame_count,
                "video_codec": v_stream.codec_context.name,
                "audio_stream_present": a_stream is not None,
                "audio_analysis": audio_analysis
            }

            print(f"    - Resolution: {width}x{height} (Codec: {v_stream.codec_context.name})")
            print(f"    - Frame Rate: {fps:.2f} fps")
            print(f"    - Video Duration: {dur_sec:.2f}s (Total Frames: {total_frames}, Black Frames: {black_frame_count})")
            if a_stream:
                print(f"    - Audio Stream: {a_stream.codec_context.name}, {a_stream.codec_context.sample_rate} Hz, {a_stream.codec_context.channels} channels")
                print(f"    - Waveform Peak: [{audio_min}, {audio_max}], RMS: {audio_rms:.1f}, Clipped: {is_clipped}")

            # Extract keyframes for each shot
            print("\n  Extracting and Saving Keyframe Images for Inspection...")
            global_frame_cursor = 0
            for sc in scenes_rows:
                sc_id_str = str(sc.id)
                v_asset = next((v for v in video_assets if v["scene_id"] == sc_id_str), None)
                shot_meta = v_asset.get("shot_metadata", []) if v_asset else []

                sc_fps = 24.0
                sc_dur = float(sc.duration_seconds or 0.0)
                sc_frames = int(round(sc_dur * sc_fps))

                if shot_meta:
                    for sm in shot_meta:
                        f_start = sm.get("frame_start", 0)
                        f_end = sm.get("frame_end", f_start + 1)
                        # Extract at center of shot
                        mid_local = (f_start + f_end) // 2
                        mid_global = global_frame_cursor + mid_local

                        if mid_global < len(all_video_frames):
                            frame_arr = all_video_frames[mid_global]
                            img_path = os.path.join(keyframes_dir, f"scene_{sc.scene_order}_shot_{sm.get('shot_index')}_frame_{mid_global}.jpg")
                            Image.fromarray(frame_arr).save(img_path, quality=85)

                            # Calculate RGB luminance and check subtitle area
                            sub_region = frame_arr[int(height * 0.85):, :, :]
                            sub_lum = float(np.mean(sub_region))

                            extracted_keyframes.append({
                                "scene_order": sc.scene_order,
                                "shot_index": sm.get("shot_index"),
                                "global_frame": mid_global,
                                "local_frame": mid_local,
                                "path": img_path,
                                "action": sm.get("action"),
                                "pexels_id": sm.get("pexels_id"),
                                "mean_luminance": round(float(np.mean(frame_arr)), 1),
                                "subtitle_region_luminance": round(sub_lum, 1)
                            })
                            print(f"    ✓ Scene {sc.scene_order} Shot {sm.get('shot_index')} (Frame {mid_global}): Lum={float(np.mean(frame_arr)):.1f} -> {os.path.basename(img_path)}")
                else:
                    mid_global = global_frame_cursor + (sc_frames // 2)
                    if mid_global < len(all_video_frames):
                        frame_arr = all_video_frames[mid_global]
                        img_path = os.path.join(keyframes_dir, f"scene_{sc.scene_order}_single_frame_{mid_global}.jpg")
                        Image.fromarray(frame_arr).save(img_path, quality=85)
                        extracted_keyframes.append({
                            "scene_order": sc.scene_order,
                            "shot_index": 1,
                            "global_frame": mid_global,
                            "local_frame": sc_frames // 2,
                            "path": img_path,
                            "action": sc.visual_prompt,
                            "pexels_id": v_asset.get("pexels_id") if v_asset else None,
                            "mean_luminance": round(float(np.mean(frame_arr)), 1)
                        })
                        print(f"    ✓ Scene {sc.scene_order} Single (Frame {mid_global}): Lum={float(np.mean(frame_arr)):.1f} -> {os.path.basename(img_path)}")

                global_frame_cursor += sc_frames

        except Exception as e:
            print(f"    ✗ PyAV Inspection Failed: {e}")
            video_metadata["inspection_error"] = str(e)
            traceback.print_exc()

    # =========================================================================
    # PART 6: REFERENCE MEDIA INTEGRITY VERIFICATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PART 6] VERIFYING PROTECTED REFERENCE MEDIA (ZERO ALTERATION GUARANTEE)")
    print("=" * 80)
    post_hashes = {}
    all_refs_match = True
    for f in ref_files:
        pre_h = preflight_hashes[f]
        post_h = compute_sha256(f)
        post_hashes[f] = post_h
        match = (pre_h == post_h)
        if not match:
            all_refs_match = False
            print(f"  ✗ [CRITICAL VIOLATION] Protected file {os.path.basename(f)} WAS ALTERED!")
        else:
            print(f"  ✓ {os.path.basename(f)}: 100% UNCHANGED (SHA-256 match)")

    assert all_refs_match, "CRITICAL VIOLATION: One or more protected reference files were altered!"

    # =========================================================================
    # PART 7: SUMMARY ARTIFACT GENERATION
    # =========================================================================
    summary_report = {
        "project_id": str(new_project_id),
        "topic": topic,
        "requested_duration_seconds": desired_duration_seconds,
        "overall_status": "SUCCESS" if execution_success else "FAILED",
        "pipeline_error": pipeline_error,
        "total_execution_time_seconds": round(t_elapsed, 2),
        "measured_narration_duration_total": round(total_measured_audio_dur, 2),
        "final_video_duration": video_metadata.get("duration_seconds"),
        "script_text": script_text,
        "script_words": script_words,
        "stages_count": len(jobs_rows),
        "stages_status": stages_status,
        "stage_durations": stage_durations,
        "num_scenes": len(scenes_rows),
        "total_planned_shots": total_planned_shots,
        "total_sourced_shots": total_sourced_shots,
        "total_missing_shots": total_missing_shots,
        "scenes_detail": scenes_detail,
        "video_assets": video_assets,
        "sfx_assets": sfx_assets,
        "voiceover_assets": voiceover_assets,
        "final_video_path": final_video_path,
        "video_metadata": video_metadata,
        "extracted_keyframes": extracted_keyframes,
        "reference_media_hashes_match": all_refs_match
    }

    summary_file = r"C:\AiVideoGenerator\scratch\step11d_summary_report.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("STEP 11D CONTROLLED LIVE TEST COMPLETE:")
    print(f"  Project UUID: {new_project_id}")
    print(f"  Status: {'SUCCESS' if execution_success else 'FAILED'}")
    print(f"  Total Runtime: {t_elapsed:.2f}s")
    print(f"  Script Words: {script_words}")
    print(f"  Narration Duration: {total_measured_audio_dur:.2f}s")
    print(f"  Video Duration: {video_metadata.get('duration_seconds')}s")
    print(f"  Scenes: {len(scenes_rows)}")
    print(f"  Planned Shots: {total_planned_shots} | Sourced: {total_sourced_shots} | Missing: {total_missing_shots}")
    print(f"  Video Assets: {len(video_assets)}")
    print(f"  SFX Assets: {len(sfx_assets)}")
    print(f"  Final MP4: {final_video_path}")
    print(f"  Summary Report: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

import os
import re
import json
import uuid
import inspect
import logging
import asyncio
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Callable

import av
from sqlalchemy import select, update, func
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.enums import ProjectStatus, AssetType, JobStatus
from app.models.project import Project
from app.models.scene import Scene
from app.models.scene_asset import SceneAsset
from app.models.pipeline_job import PipelineJob

from app.schemas.media import MediaAssetSchema
from app.schemas.visual_spec import validate_action_timeline_timing, reconcile_shot_plan_timing, PLACEHOLDER_SETTING
from app.services.info_card import render_info_card
from app.services.providers.base import ScriptProvider
from app.services.providers.exceptions import (
    StockMediaError,
    NoSuitableMediaFoundError,
    UncertainFootageRequiresApprovalError,
)
from app.services.providers.factory import get_script_provider
from app.services.providers.stock_media_base import StockMediaProvider
from app.services.providers.tts_base import TTSProvider
from app.services.providers.edge_tts_provider import EdgeTTSProvider
from app.services.providers.pexels_video_provider import PexelsVideoProvider
from app.services.providers.freesound_provider import FreesoundAudioProvider
from app.services.video_assembly import assemble_video, compute_frame_exact_duration
from app.services.orchestrator import process_visual_specifications_parallel
from app.services.visual_verification import (
    VisualVerifier,
    VerificationStatus,
    VisualVerificationResult,
    ConceptClassifier,
    ConceptFeasibility
)

logger = logging.getLogger(__name__)

# Single-execution lock per worker process to prevent CPU/FFmpeg thread saturation
_PROCESS_PIPELINE_LOCK = asyncio.Lock()

# Protected reference filenames that MUST NEVER be overwritten by generated projects
PROTECTED_REFERENCE_MEDIA = frozenset({
    "output_5scenes.mp4",
    "output_27scenes.mp4",
    "output_5scenes.mp4TEMP_MPY_wvf_snd.mp4",
    "output_5scenesTEMP_MPY_wvf_snd.mp4",
    "full_render_audio.wav"
})


def extract_scene_subject(
    scene_dict: Dict[str, Any],
    fallback_topic: Optional[str] = None
) -> str:
    """
    Extract a scene-specific primary subject rather than using the broad global topic name.
    Priority:
    1. 'primary_subject' explicitly provided in scene_dict (from visual specification).
    2. 'primary_subject' from the first shot in shot_plan, if present.
    3. Concrete subject extracted from visual_prompt or scene_text.
    4. First specific search keyword from search_keywords.
    5. Cleaned fallback noun phrase from topic.
    """
    # 1. Direct explicit scene-level primary_subject
    if scene_dict.get("primary_subject") and str(scene_dict["primary_subject"]).strip():
        return str(scene_dict["primary_subject"]).strip()

    # 2. Check shot_plan if present
    sp = scene_dict.get("shot_plan")
    if isinstance(sp, list) and len(sp) > 0:
        first_shot = sp[0]
        if isinstance(first_shot, dict) and first_shot.get("primary_subject"):
            return str(first_shot["primary_subject"]).strip()
        elif hasattr(first_shot, "primary_subject") and first_shot.primary_subject:
            return str(first_shot.primary_subject).strip()

    # 3. Extract from visual_prompt or scene_text
    prompt = str(scene_dict.get("visual_prompt") or scene_dict.get("scene_text") or "").strip()
    if prompt:
        m = re.search(
            r'(?:close-up|macro shot|shot|view|timelapse|photo|cinematic macro shot|extreme close-up|cinematic shot)\s+of\s+(?:a|an|the)?\s*([^,\.]+?)(?:\s+(?:resting|in|as|with|pushing|falling|sizzling|growing|emerging|on|into|breaking|swelling)|\.|,)',
            prompt,
            re.IGNORECASE
        )
        if m:
            extracted = m.group(1).strip()
            if len(extracted) >= 3 and len(extracted.split()) <= 4:
                return extracted

        m2 = re.search(r'^(?:a|an|the)?\s*([a-z\s]{3,30}?)\s+(?:enters|absorbs|grows|pushes|emerges|swells|awakens|sizzles|pours|simmers)', prompt, re.IGNORECASE)
        if m2:
            extracted = m2.group(1).strip()
            if len(extracted) >= 3:
                return extracted

    # 4. First search keyword if meaningful
    kws = scene_dict.get("search_keywords") or []
    if kws and isinstance(kws, list) and len(kws) > 0:
        first_kw = str(kws[0]).strip()
        kw_words = [w for w in re.split(r'\W+', first_kw) if len(w) > 2 and w.lower() not in ("macro", "close", "timelapse", "video", "shot")]
        if kw_words:
            candidate_subject = " ".join(kw_words[:2])
            if len(candidate_subject) >= 3:
                return candidate_subject

    # 5. Cleaned topic fallback
    if fallback_topic and str(fallback_topic).strip():
        cleaned_top = re.sub(r'^(how\s+(?:a|to|an)?|the\s+|why\s+)', '', str(fallback_topic).strip(), flags=re.IGNORECASE).strip()
        return cleaned_top if cleaned_top else "subject"

    return "subject"


class VideoPipelineOrchestrator:
    """
    Coordinates the seven-stage AI Text-to-Video generation pipeline:
      1. Narration Script Generation (Groq Cloud default / Ollama optional)
      2. Scene Splitting & Content Preservation (Groq / Ollama)
      3. Visual Specification Generation (Parallel Groq / Ollama, semaphore=5)
      4. Voiceover Narration Audio Generation (EdgeTTS + PyAV duration decode)
      5. Stock Video Media Sourcing (Pexels Video API)
      6. Action Sound Effects Sourcing (Freesound API with loudness threshold)
      7. Frame-Exact Video Assembly & Subtitle Integration (MoviePy 2.x + PyAV)

    Design Principles:
      - Never overwrites approved reference videos (media/output_5scenes.mp4, media/output_27scenes.mp4).
      - Isolates all project files under media/projects/{project_id}/.
      - Short-lived database transaction boundaries: never holds DB transactions open across API calls or rendering.
      - Single-job execution per process via asyncio.Lock to avoid CPU/RAM starvation.
      - Injectable provider and assembler dependencies for clean unit testing and mocking.
    """

    def __init__(
        self,
        session_factory=AsyncSessionLocal,
        script_provider: Optional[ScriptProvider] = None,
        tts_provider: Optional[TTSProvider] = None,
        stock_media_provider: Optional[StockMediaProvider] = None,
        sfx_provider: Optional[Any] = None,
        video_assembler_fn: Optional[Callable] = None,
        base_media_dir: Optional[str] = None,
        lock: Optional[asyncio.Lock] = None,
    ):
        self.session_factory = session_factory
        self._script_provider = script_provider
        self._tts_provider = tts_provider
        self._stock_media_provider = stock_media_provider
        self._sfx_provider = sfx_provider
        self._video_assembler_fn = video_assembler_fn or assemble_video
        self.base_media_dir = os.path.abspath(base_media_dir or settings.MEDIA_STORAGE_PATH or "./media")
        self.lock = lock or _PROCESS_PIPELINE_LOCK

    # -------------------------------------------------------------------------
    # Provider Lazy Resolvers (Injectable / Mockable)
    # -------------------------------------------------------------------------
    def get_script_provider(self) -> ScriptProvider:
        if self._script_provider is not None:
            return self._script_provider
        return get_script_provider()

    def get_tts_provider(self) -> TTSProvider:
        if self._tts_provider is not None:
            return self._tts_provider
        return EdgeTTSProvider(voice=settings.TTS_VOICE)

    def get_stock_media_provider(self) -> StockMediaProvider:
        if self._stock_media_provider is not None:
            return self._stock_media_provider
        return PexelsVideoProvider()

    def get_sfx_provider(self) -> Any:
        if self._sfx_provider is not None:
            return self._sfx_provider
        return FreesoundAudioProvider()

    # -------------------------------------------------------------------------
    # Workspace & Path Protection
    # -------------------------------------------------------------------------
    def get_project_workspace(self, project_id: uuid.UUID) -> Dict[str, str]:
        """
        Creates and returns isolated directory paths for a project under:
          media/projects/{project_id}/
        Enforces strict safety checks so protected reference media files can never be targeted.
        """
        str_id = str(project_id)
        project_dir = os.path.join(self.base_media_dir, "projects", str_id)
        audio_dir = os.path.join(project_dir, "audio")
        resampled_dir = os.path.join(project_dir, "audio_resampled")
        clips_dir = os.path.join(project_dir, "clips")
        sounds_dir = os.path.join(project_dir, "sounds")
        output_path = os.path.join(project_dir, f"video_{str_id}.mp4")

        # Verify output filename does not collide with protected reference files
        output_basename = os.path.basename(output_path)
        if output_basename in PROTECTED_REFERENCE_MEDIA:
            raise RuntimeError(
                f"Security violation: Target output filename '{output_basename}' matches protected reference media."
            )

        # Verify path does not escape into root media directory
        normalized_output = os.path.normpath(os.path.abspath(output_path))
        for protected_name in PROTECTED_REFERENCE_MEDIA:
            protected_full = os.path.normpath(os.path.join(self.base_media_dir, protected_name))
            if normalized_output == protected_full:
                raise RuntimeError(
                    f"Security violation: Target output path '{output_path}' would overwrite protected file '{protected_full}'."
                )

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(resampled_dir, exist_ok=True)
        os.makedirs(clips_dir, exist_ok=True)
        os.makedirs(sounds_dir, exist_ok=True)

        return {
            "project_dir": project_dir,
            "audio_dir": audio_dir,
            "resampled_dir": resampled_dir,
            "clips_dir": clips_dir,
            "sounds_dir": sounds_dir,
            "output_path": output_path,
        }

    # -------------------------------------------------------------------------
    # Manual Approval & Contact Sheet Registry Helpers
    # -------------------------------------------------------------------------
    def get_approval_registry_path(self, project_id: uuid.UUID) -> str:
        """Returns the file path for the project's manual approval registry JSON."""
        ws = self.get_project_workspace(project_id)
        return os.path.join(ws["project_dir"], "approval_registry.json")

    def load_approval_registry(self, project_id: uuid.UUID) -> Dict[str, Any]:
        """Loads the approval registry JSON from disk, or initializes an empty one."""
        reg_path = self.get_approval_registry_path(project_id)
        if os.path.exists(reg_path):
            try:
                with open(reg_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load approval registry from {reg_path}: {e}")
        return {
            "project_id": str(project_id),
            "pending_reviews": [],
            "approved_candidates": {},
            "rejected_candidates": {}
        }

    def save_approval_registry(self, project_id: uuid.UUID, registry: Dict[str, Any]) -> None:
        """Atomically saves the approval registry JSON to disk."""
        reg_path = self.get_approval_registry_path(project_id)
        os.makedirs(os.path.dirname(os.path.abspath(reg_path)), exist_ok=True)
        tmp_path = f"{reg_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp_path, reg_path)

    def is_candidate_approved(
        self,
        project_id: uuid.UUID,
        scene_order: int,
        shot_index: int,
        candidate_video_id: Any
    ) -> bool:
        """Checks if a candidate video ID has been explicitly approved for a specific shot."""
        registry = self.load_approval_registry(project_id)
        shot_key = f"scene_{scene_order}_shot_{shot_index}"
        approved = registry.get("approved_candidates", {}).get(shot_key)
        if approved:
            approved_id = str(approved.get("candidate_video_id"))
            return approved_id == str(candidate_video_id)
        return False

    def is_candidate_rejected(
        self,
        project_id: uuid.UUID,
        scene_order: int,
        shot_index: int,
        candidate_video_id: Any
    ) -> bool:
        """Checks if a candidate video ID has been explicitly rejected for a specific shot."""
        registry = self.load_approval_registry(project_id)
        shot_key = f"scene_{scene_order}_shot_{shot_index}"
        rejected_list = registry.get("rejected_candidates", {}).get(shot_key, [])
        return str(candidate_video_id) in [str(r) for r in rejected_list]

    def approve_candidate(
        self,
        project_id: uuid.UUID,
        scene_order: int,
        shot_index: int,
        candidate_video_id: Any,
        reviewer: str = "manual_reviewer",
        notes: str = ""
    ) -> Dict[str, Any]:
        """Explicitly approves an uncertain candidate video for a specific shot."""
        registry = self.load_approval_registry(project_id)
        shot_key = f"scene_{scene_order}_shot_{shot_index}"
        
        # Look for cached metadata in pending_reviews if available
        local_clip_path = None
        source_url = None
        for pr in registry.get("pending_reviews", []):
            if pr.get("shot_key") == shot_key and str(pr.get("candidate_video_id")) == str(candidate_video_id):
                local_clip_path = pr.get("local_clip_path")
                source_url = pr.get("source_url")
                break

        record = {
            "candidate_video_id": candidate_video_id,
            "local_clip_path": local_clip_path,
            "source_url": source_url,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "reviewer": reviewer,
            "notes": notes
        }
        if "approved_candidates" not in registry:
            registry["approved_candidates"] = {}
        registry["approved_candidates"][shot_key] = record

        # Remove from pending reviews if present
        registry["pending_reviews"] = [
            pr for pr in registry.get("pending_reviews", [])
            if not (pr.get("shot_key") == shot_key and str(pr.get("candidate_video_id")) == str(candidate_video_id))
        ]
        self.save_approval_registry(project_id, registry)
        logger.info(f"Candidate {candidate_video_id} explicitly APPROVED for {shot_key} by {reviewer}")
        return record

    def reject_candidate(
        self,
        project_id: uuid.UUID,
        scene_order: int,
        shot_index: int,
        candidate_video_id: Any,
        reviewer: str = "manual_reviewer",
        reason: str = ""
    ) -> Dict[str, Any]:
        """Explicitly rejects a candidate video for a specific shot."""
        registry = self.load_approval_registry(project_id)
        shot_key = f"scene_{scene_order}_shot_{shot_index}"
        if "rejected_candidates" not in registry:
            registry["rejected_candidates"] = {}
        if shot_key not in registry["rejected_candidates"]:
            registry["rejected_candidates"][shot_key] = []
        
        if str(candidate_video_id) not in [str(x) for x in registry["rejected_candidates"][shot_key]]:
            registry["rejected_candidates"][shot_key].append(candidate_video_id)

        # Remove from pending reviews
        registry["pending_reviews"] = [
            pr for pr in registry.get("pending_reviews", [])
            if not (pr.get("shot_key") == shot_key and str(pr.get("candidate_video_id")) == str(candidate_video_id))
        ]
        self.save_approval_registry(project_id, registry)
        logger.info(f"Candidate {candidate_video_id} explicitly REJECTED for {shot_key} by {reviewer}: {reason}")
        return {"candidate_video_id": candidate_video_id, "rejected": True, "reason": reason}

    def record_pending_review(
        self,
        project_id: uuid.UUID,
        scene_order: int,
        shot_index: int,
        candidate_video_id: Any,
        local_clip_path: str,
        source_url: str,
        contact_sheet_path: str,
        shot_reqs: Dict[str, Any],
        evidence: List[str]
    ) -> Dict[str, Any]:
        """Records an uncertain candidate shot awaiting manual review."""
        registry = self.load_approval_registry(project_id)
        shot_key = f"scene_{scene_order}_shot_{shot_index}"
        entry = {
            "shot_key": shot_key,
            "scene_order": scene_order,
            "shot_index": shot_index,
            "candidate_video_id": candidate_video_id,
            "local_clip_path": local_clip_path,
            "source_url": source_url,
            "contact_sheet_path": contact_sheet_path,
            "required_action": str(shot_reqs.get("required_action") or shot_reqs.get("required_visible_action") or ""),
            "primary_subject": str(shot_reqs.get("primary_subject") or ""),
            "required_setting": str(shot_reqs.get("required_setting") or ""),
            "evidence": evidence,
            "recorded_at": datetime.now(timezone.utc).isoformat()
        }
        existing_indices = [
            i for i, pr in enumerate(registry.get("pending_reviews", []))
            if pr.get("shot_key") == shot_key and str(pr.get("candidate_video_id")) == str(candidate_video_id)
        ]
        if existing_indices:
            registry["pending_reviews"][existing_indices[0]] = entry
        else:
            if "pending_reviews" not in registry:
                registry["pending_reviews"] = []
            registry["pending_reviews"].append(entry)

        self.save_approval_registry(project_id, registry)
        return entry

    def get_pending_approvals(self, project_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Returns all uncertain candidate shots currently awaiting manual review."""
        registry = self.load_approval_registry(project_id)
        return list(registry.get("pending_reviews", []))

    # -------------------------------------------------------------------------
    # Isolated Short-Lived Database Transaction Helpers
    # -------------------------------------------------------------------------
    async def _update_project_state(
        self,
        project_id: uuid.UUID,
        status: ProjectStatus,
        current_stage: str,
        progress_percent: int,
        script: Optional[str] = None,
        video_path: Optional[str] = None,
        error_message: Optional[str] = None,
        completed: bool = False,
    ) -> None:
        """Updates project status and progress in a standalone short-lived transaction."""
        now = datetime.now(timezone.utc)
        values: Dict[str, Any] = {
            "status": status,
            "current_stage": current_stage,
            "progress_percent": progress_percent,
            "updated_at": now,
        }
        if script is not None:
            values["script"] = script
        if video_path is not None:
            values["video_path"] = video_path
        if error_message is not None:
            values["error_message"] = error_message
        if completed:
            values["completed_at"] = now

        async with self.session_factory() as session:
            await session.execute(
                update(Project).where(Project.id == project_id).values(**values)
            )
            await session.commit()

    async def _record_pipeline_job(
        self,
        project_id: uuid.UUID,
        stage: str,
        status: JobStatus,
        provider_name: Optional[str] = None,
        error_message: Optional[str] = None,
        job_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """Records or updates a granular PipelineJob row in a standalone transaction."""
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            if job_id is None:
                new_job = PipelineJob(
                    id=uuid.uuid4(),
                    project_id=project_id,
                    stage=stage,
                    status=status,
                    attempt=1,
                    provider_name=provider_name,
                    error_message=error_message,
                    started_at=now,
                    completed_at=now if status in (JobStatus.COMPLETED, JobStatus.FAILED) else None,
                    created_at=now,
                )
                session.add(new_job)
                await session.commit()
                return new_job.id
            else:
                job_values: Dict[str, Any] = {"status": status}
                if error_message is not None:
                    job_values["error_message"] = error_message
                if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    job_values["completed_at"] = now
                await session.execute(
                    update(PipelineJob).where(PipelineJob.id == job_id).values(**job_values)
                )
                await session.commit()
                return job_id

    async def _save_initial_scenes(
        self,
        project_id: uuid.UUID,
        raw_scenes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Persists parsed scene beat rows into the scenes table in a standalone transaction."""
        now = datetime.now(timezone.utc)
        scene_records = []
        created_scenes = []

        async with self.session_factory() as session:
            for s in raw_scenes:
                scene_id = uuid.uuid4()
                order = int(s["scene_order"])
                text = str(s["scene_text"]).strip()

                scene_row = Scene(
                    id=scene_id,
                    project_id=project_id,
                    scene_order=order,
                    scene_text=text,
                    status="pending",
                    created_at=now,
                    updated_at=now,
                )
                scene_records.append(scene_row)
                created_scenes.append({
                    "id": scene_id,
                    "scene_order": order,
                    "scene_text": text,
                })

            session.add_all(scene_records)
            await session.commit()

        return created_scenes

    async def _update_scenes_visual_specs(
        self,
        project_id: uuid.UUID,
        scenes_with_specs: List[Dict[str, Any]],
    ) -> None:
        """Updates scenes with their generated visual prompts and keywords."""
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            for s in scenes_with_specs:
                await session.execute(
                    update(Scene)
                    .where(Scene.id == s["id"])
                    .values(
                        visual_prompt=s.get("visual_prompt"),
                        search_keywords=s.get("search_keywords"),
                        updated_at=now,
                    )
                )
            await session.commit()

    async def _save_scene_asset(
        self,
        scene_id: uuid.UUID,
        asset_type: AssetType,
        provider_name: str,
        local_path: str,
        duration_seconds: Optional[float] = None,
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        """Persists a media asset record (audio, video clip, or SFX) into scene_assets."""
        now = datetime.now(timezone.utc)
        asset_id = uuid.uuid4()
        async with self.session_factory() as session:
            asset = SceneAsset(
                id=asset_id,
                scene_id=scene_id,
                asset_type=asset_type,
                provider_name=provider_name,
                source_url=source_url,
                storage_url=local_path,
                local_path=local_path,
                duration_seconds=duration_seconds,
                asset_metadata=metadata,
                created_at=now,
            )
            session.add(asset)
            await session.commit()
        return asset_id

    @staticmethod
    def _is_verified_cc0_license(cand: Optional[Dict[str, Any]]) -> bool:
        """
        Fail-closed verification: returns True ONLY if the candidate is positively
        verified as Creative Commons 0 (Public Domain Dedication).
        Rejects CC-BY-NC, CC-BY, unknown licenses, or missing metadata.
        """
        if not cand or not isinstance(cand, dict):
            return False

        license_name = str(cand.get("license", "")).strip().lower()
        license_url = str(cand.get("license_url", "")).strip().lower()

        if not license_name or not license_url:
            return False

        # Positive CC0 markers
        is_cc0_name = ("cc0" in license_name) or ("public domain" in license_name)
        is_cc0_url = ("creativecommons.org/publicdomain/zero" in license_url) or ("/zero/1.0" in license_url)

        # Negative non-CC0 markers
        has_nc = ("-nc" in license_url) or ("non-commercial" in license_name) or ("by-nc" in license_name)
        has_by = ("by" in license_name and "cc0" not in license_name)

        if (is_cc0_name or is_cc0_url) and not has_nc and not has_by:
            return True

        return False

    # -------------------------------------------------------------------------
    # Stage-Level Execution Methods
    # -------------------------------------------------------------------------
    async def _stage_1_generate_script(
        self, project_id: uuid.UUID, topic: str, desired_duration: int
    ) -> str:
        """Stage 1: Generate full narration script matching duration bounds."""
        job_id = await self._record_pipeline_job(
            project_id=project_id,
            stage="script_generation",
            status=JobStatus.RUNNING,
            provider_name=settings.SCRIPT_PROVIDER,
        )
        await self._update_project_state(
            project_id=project_id,
            status=ProjectStatus.GENERATING_SCRIPT,
            current_stage="Writing narration script",
            progress_percent=15,
        )

        try:
            provider = self.get_script_provider()
            script_text = await provider.generate_script(topic, desired_duration)

            if not script_text or not script_text.strip():
                raise ValueError("Script provider returned an empty narration script.")

            await self._update_project_state(
                project_id=project_id,
                status=ProjectStatus.GENERATING_SCRIPT,
                current_stage="Script generated",
                progress_percent=25,
                script=script_text,
            )
            await self._record_pipeline_job(
                project_id=project_id,
                stage="script_generation",
                status=JobStatus.COMPLETED,
                job_id=job_id,
            )
            return script_text

        except Exception as e:
            await self._record_pipeline_job(
                project_id=project_id,
                stage="script_generation",
                status=JobStatus.FAILED,
                error_message=str(e),
                job_id=job_id,
            )
            raise

    async def _stage_2_split_scenes(
        self, project_id: uuid.UUID, script: str, desired_duration: int
    ) -> List[Dict[str, Any]]:
        """Stage 2: Split script into sequential duration-aware scene beats."""
        job_id = await self._record_pipeline_job(
            project_id=project_id,
            stage="scene_splitting",
            status=JobStatus.RUNNING,
            provider_name=settings.SCRIPT_PROVIDER,
        )
        await self._update_project_state(
            project_id=project_id,
            status=ProjectStatus.SPLITTING_SCENES,
            current_stage="Splitting script into scene beats",
            progress_percent=30,
        )

        try:
            provider = self.get_script_provider()
            raw_scenes = await provider.split_into_scenes(script, desired_duration)

            if not raw_scenes:
                raise ValueError("Scene splitting returned zero scene beats.")

            created_scenes = await self._save_initial_scenes(project_id, raw_scenes)

            await self._record_pipeline_job(
                project_id=project_id,
                stage="scene_splitting",
                status=JobStatus.COMPLETED,
                job_id=job_id,
            )
            return created_scenes

        except Exception as e:
            await self._record_pipeline_job(
                project_id=project_id,
                stage="scene_splitting",
                status=JobStatus.FAILED,
                error_message=str(e),
                job_id=job_id,
            )
            raise

    async def _stage_3_generate_visual_specs(
        self,
        project_id: uuid.UUID,
        scenes: List[Dict[str, Any]],
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Stage 3: Generate visual prompts and stock search keywords via parallel fan-out."""
        job_id = await self._record_pipeline_job(
            project_id=project_id,
            stage="visual_specification",
            status=JobStatus.RUNNING,
            provider_name=settings.SCRIPT_PROVIDER,
        )
        await self._update_project_state(
            project_id=project_id,
            status=ProjectStatus.GENERATING_VISUALS,
            current_stage="Generating visual cinematography specifications",
            progress_percent=45,
        )

        try:
            provider = self.get_script_provider()
            # Bounded parallel execution reusing existing orchestrator function
            specs = await process_visual_specifications_parallel(
                provider=provider,
                scenes=scenes,
                max_concurrency=settings.GROQ_MAX_CONCURRENCY,
                topic=topic,
            )

            for sc, sp in zip(scenes, specs):
                sc["visual_prompt"] = sp.get("visual_prompt", "")
                sc["search_keywords"] = sp.get("search_keywords", [])
                sc["action_timeline"] = sp.get("action_timeline")
                sc["shot_plan"] = sp.get("shot_plan")

            # Persist action timeline and shot plan to workspace to survive process/session boundaries
            try:
                ws = self.get_project_workspace(project_id)
                tl_file = os.path.join(ws["project_dir"], "action_timeline.json")
                with open(tl_file, "w", encoding="utf-8") as f:
                    json.dump({str(sc["id"]): sc.get("action_timeline") for sc in scenes}, f, indent=2)
                sp_file = os.path.join(ws["project_dir"], "shot_plan.json")
                with open(sp_file, "w", encoding="utf-8") as f:
                    json.dump({str(sc["id"]): sc.get("shot_plan") for sc in scenes}, f, indent=2)
            except Exception as persist_err:
                logger.debug(f"Could not persist action_timeline/shot_plan to workspace: {persist_err}")

            await self._update_scenes_visual_specs(project_id, scenes)

            await self._record_pipeline_job(
                project_id=project_id,
                stage="visual_specification",
                status=JobStatus.COMPLETED,
                job_id=job_id,
            )
            return scenes

        except Exception as e:
            await self._record_pipeline_job(
                project_id=project_id,
                stage="visual_specification",
                status=JobStatus.FAILED,
                error_message=str(e),
                job_id=job_id,
            )
            raise

    async def _stage_4_generate_voiceover(
        self, project_id: uuid.UUID, scenes: List[Dict[str, Any]], audio_dir: str
    ) -> List[Dict[str, Any]]:
        """Stage 4: Generate EdgeTTS narration audio and measure exact duration via PyAV."""
        job_id = await self._record_pipeline_job(
            project_id=project_id,
            stage="voiceover_generation",
            status=JobStatus.RUNNING,
            provider_name="edge_tts",
        )
        await self._update_project_state(
            project_id=project_id,
            status=ProjectStatus.GENERATING_VOICE,
            current_stage="Generating voiceover narration audio",
            progress_percent=60,
        )

        try:
            tts = self.get_tts_provider()
            for sc in scenes:
                order = sc["scene_order"]
                out_path = os.path.join(audio_dir, f"scene_{order}_narr.mp3")

                duration = await tts.generate_audio(sc["scene_text"], out_path)
                sc["duration"] = duration
                sc["audio_path"] = out_path

                # Update scene duration in database
                async with self.session_factory() as session:
                    await session.execute(
                        update(Scene)
                        .where(Scene.id == sc["id"])
                        .values(duration_seconds=duration, status="audio_ready")
                    )
                    await session.commit()

                # Record audio asset with persisted action timeline in JSONB metadata
                await self._save_scene_asset(
                    scene_id=sc["id"],
                    asset_type=AssetType.AUDIO,
                    provider_name="edge_tts",
                    local_path=out_path,
                    duration_seconds=duration,
                    metadata={
                        "voice": settings.TTS_VOICE,
                        "measured_method": "pyav_decode",
                        "action_timeline": sc.get("action_timeline"),
                        "shot_plan": sc.get("shot_plan"),
                    },
                )

            await self._record_pipeline_job(
                project_id=project_id,
                stage="voiceover_generation",
                status=JobStatus.COMPLETED,
                job_id=job_id,
            )
            return scenes

        except Exception as e:
            await self._record_pipeline_job(
                project_id=project_id,
                stage="voiceover_generation",
                status=JobStatus.FAILED,
                error_message=str(e),
                job_id=job_id,
            )
            raise

    async def _stage_5_source_media(
        self,
        project_id: uuid.UUID,
        scenes: List[Dict[str, Any]],
        clips_dir: str,
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Stage 5: Search and download HD stock video clips from Pexels with visual context preservation."""
        job_id = await self._record_pipeline_job(
            project_id=project_id,
            stage="media_sourcing",
            status=JobStatus.RUNNING,
            provider_name="pexels",
        )
        await self._update_project_state(
            project_id=project_id,
            status=ProjectStatus.GENERATING_VISUALS,
            current_stage="Sourcing HD video clips from Pexels",
            progress_percent=75,
        )

        try:
            if topic is None:
                async with self.session_factory() as session:
                    res = await session.execute(select(Project.topic).where(Project.id == project_id))
                    topic = res.scalar() or ""

            video_provider = self.get_stock_media_provider()
            used_video_ids: Set[int] = set()

            # Restore shot_plan from workspace if missing in-memory
            try:
                ws = self.get_project_workspace(project_id)
                sp_file = os.path.join(ws["project_dir"], "shot_plan.json")
                if os.path.exists(sp_file):
                    with open(sp_file, "r", encoding="utf-8") as f:
                        sp_map = json.load(f)
                    for sc in scenes:
                        if not sc.get("shot_plan") and str(sc["id"]) in sp_map:
                            sc["shot_plan"] = sp_map[str(sc["id"])]
            except Exception as res_err:
                logger.debug(f"Could not restore shot_plan from workspace: {res_err}")

            verifier = getattr(self, "visual_verifier", None) or VisualVerifier(num_sample_frames=3)

            for sc in scenes:
                order = sc["scene_order"]
                dur = compute_frame_exact_duration(sc["duration"], fps=24.0)
                kw_list = sc.get("search_keywords") or ["landscape", "cinematic"]

                # Build lightweight scene-specific visual context using existing data
                scene_ctx = {
                    "topic": topic,
                    "scene_order": order,
                    "scene_text": sc.get("scene_text", ""),
                    "visual_prompt": sc.get("visual_prompt", ""),
                    "action_timeline": sc.get("action_timeline"),
                    "search_keywords": kw_list,
                }

                had_explicit_shot_plan = bool(sc.get("shot_plan"))
                if not had_explicit_shot_plan and dur > 8.0:
                    half_dur = dur / 2.0
                    kw_a = kw_list[:2] if len(kw_list) >= 2 else kw_list
                    kw_b = kw_list[1:3] if len(kw_list) >= 3 else kw_list
                    cand_a = await video_provider.search_media(keywords=kw_a, target_duration=half_dur, used_video_ids=used_video_ids, scene_context=scene_ctx)
                    path_a = await video_provider.download_media(cand_a.download_url, output_dir=clips_dir, filename_prefix=f"scene_{order}_clip_a")
                    if cand_a.metadata and cand_a.metadata.get("pexels_id"):
                        used_video_ids.add(cand_a.metadata.get("pexels_id"))

                    cand_b = await video_provider.search_media(keywords=kw_b, target_duration=half_dur, used_video_ids=used_video_ids, scene_context=scene_ctx)
                    path_b = await video_provider.download_media(cand_b.download_url, output_dir=clips_dir, filename_prefix=f"scene_{order}_clip_b")
                    if cand_b.metadata and cand_b.metadata.get("pexels_id"):
                        used_video_ids.add(cand_b.metadata.get("pexels_id"))

                    sc["media_paths"] = [path_a, path_b]
                    sc["media_type"] = "video"
                    sc["is_multi_shot"] = False
                    sc["shot_metadata"] = [cand_a.metadata or {}, cand_b.metadata or {}]
                    continue

                shot_plan = sc.get("shot_plan")
                if not shot_plan:
                    scene_subject = extract_scene_subject(sc, fallback_topic=topic)
                    shot_plan = [{
                        "shot_index": 1,
                        "provisional_start_seconds": 0.0,
                        "provisional_end_seconds": dur,
                        "required_visible_action": sc.get("visual_prompt") or sc.get("scene_text") or "cinematic scene",
                        "primary_subject": scene_subject,
                        "search_keywords": kw_list,
                        "required_setting": sc.get("required_setting") or "appropriate setting",
                    }]

                if shot_plan:
                    # UNIFIED SHOT SOURCING BRANCH
                    reconciled_shots = reconcile_shot_plan_timing(
                        shot_plan=shot_plan,
                        measured_scene_duration_seconds=sc["duration"],
                        fps=24
                    )

                    # Timing alignment verification with action_timeline
                    timing_warnings = []
                    action_tl = sc.get("action_timeline") or []
                    for r_shot in reconciled_shots:
                        ev_idx = r_shot.get("action_event_index")
                        if ev_idx is not None and 0 <= ev_idx < len(action_tl):
                            ev = action_tl[ev_idx]
                            ev_dict = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
                            sfx_start = ev_dict.get("sfx_start_seconds")
                            if sfx_start is not None:
                                if sfx_start < r_shot["start_seconds"] or sfx_start > r_shot["end_seconds"]:
                                    warning_msg = (
                                        f"Scene {order} Shot {r_shot['shot_index']}: Action event {ev_idx} SFX offset "
                                        f"({sfx_start}s) falls outside reconciled shot window "
                                        f"({r_shot['start_seconds']}s - {r_shot['end_seconds']}s). "
                                        f"Proportional shot allocation does not guarantee physical acoustic synchronization."
                                    )
                                    timing_warnings.append(warning_msg)
                                    logger.warning(warning_msg)

                    sc["timing_warnings"] = timing_warnings

                    shot_clips = []
                    shot_metadata_list = []
                    missing_shots = []

                    for r_shot in reconciled_shots:
                        s_idx = r_shot["shot_index"]
                        s_keywords = r_shot.get("search_keywords") or kw_list
                        s_dur = r_shot["duration_seconds"]

                        shot_ctx = {
                            "topic": topic,
                            "scene_order": order,
                            "shot_index": s_idx,
                            "scene_text": sc.get("scene_text", ""),
                            "visual_prompt": r_shot.get("required_visible_action", ""),
                            "required_action": r_shot.get("required_visible_action", ""),
                            "primary_subject": r_shot.get("primary_subject", ""),
                            "required_setting": (
                                "" if r_shot.get("required_setting") == PLACEHOLDER_SETTING
                                else r_shot.get("required_setting", "")
                            ),
                            "visual_state": r_shot.get("visual_state", ""),
                            "camera_framing": r_shot.get("camera_framing", ""),
                            "required_objects": r_shot.get("required_objects") or [],
                            "explicit_exclusions": r_shot.get("explicit_exclusions") or [],
                            "action_timeline": action_tl,
                            "search_keywords": s_keywords,
                        }

                        shot_key = f"scene_{order}_shot_{s_idx}"
                        verified_cand = None
                        verified_file = None
                        v_res = None

                        # Check if already approved by manual reviewer and present on disk
                        reg = self.load_approval_registry(project_id)
                        approved_rec = reg.get("approved_candidates", {}).get(shot_key)
                        if approved_rec and str(approved_rec.get("candidate_video_id")):
                            app_vid = approved_rec.get("candidate_video_id")
                            local_f = approved_rec.get("local_clip_path") or os.path.join(clips_dir, f"scene_{order}_shot_{s_idx}_{app_vid}.mp4")
                            if os.path.exists(local_f) and os.path.getsize(local_f) > 1000:
                                logger.info(f"Reusing approved clip directly from disk for {shot_key} (video {app_vid}): {local_f}")
                                verified_cand = MediaAssetSchema(
                                    download_url="",
                                    source_url=approved_rec.get("source_url") or "",
                                    metadata={"pexels_id": app_vid, "is_manually_approved": True}
                                )
                                verified_file = local_f
                                v_res = VisualVerificationResult(
                                    status=VerificationStatus.ACCEPTED,
                                    confidence=1.0,
                                    confirmed_matches=["Explicitly approved by manual reviewer"],
                                    evidence=["Manual approval gate: verified and accepted by reviewer"],
                                    is_visual_frame_verified=True,
                                    is_manually_approved=True
                                )
                                used_video_ids.add(app_vid)

                        try:
                            if verified_cand is None:
                                # Determine rejected candidate IDs for this shot from registry and exclude upfront
                                rejected_for_shot = reg.get("rejected_candidates", {}).get(shot_key, [])
                                shot_excluded_ids = set(used_video_ids)
                                for r_id in rejected_for_shot:
                                    try:
                                        shot_excluded_ids.add(int(r_id))
                                    except (ValueError, TypeError):
                                        pass
                                    shot_excluded_ids.add(str(r_id))

                                # 1. Retrieve bounded pool of candidate clips (up to 3)
                                if hasattr(video_provider, "search_media_candidates"):
                                    candidates = await video_provider.search_media_candidates(
                                        s_keywords,
                                        target_duration=s_dur,
                                        used_video_ids=shot_excluded_ids,
                                        scene_context=shot_ctx,
                                        max_candidates=3
                                    )
                                else:
                                    single_m = await video_provider.search_media(
                                        s_keywords,
                                        target_duration=s_dur,
                                        used_video_ids=shot_excluded_ids,
                                        scene_context=shot_ctx
                                    )
                                    candidates = [single_m]

                                if not candidates:
                                    raise NoSuitableMediaFoundError(f"No candidate video clips returned for shot {s_idx}.")

                                # 2. Actual Video-Frame Verification Loop
                                best_uncertain = None
                                for cand in candidates:
                                    p_id = cand.metadata.get("pexels_id")

                                    # Check if previously rejected by manual reviewer
                                    if self.is_candidate_rejected(project_id, order, s_idx, p_id):
                                        logger.info(f"Skipping candidate {p_id} for {shot_key} (explicitly rejected by reviewer)")
                                        continue

                                    prefix = f"scene_{order}_shot_{s_idx}_{p_id}"
                                    expected_local = os.path.join(clips_dir, f"{prefix}.mp4")

                                    # Reuse downloaded clip if already on disk
                                    if os.path.exists(expected_local) and os.path.getsize(expected_local) > 1000:
                                        cand_f = expected_local
                                        logger.debug(f"Reusing existing downloaded clip on disk: {cand_f}")
                                    else:
                                        cand_f = await video_provider.download_media(
                                            cand.download_url,
                                            clips_dir,
                                            prefix
                                        )

                                    # Check if candidate was explicitly approved by manual reviewer
                                    if self.is_candidate_approved(project_id, order, s_idx, p_id):
                                        logger.info(f"Candidate {p_id} for {shot_key} is EXPLICITLY APPROVED by manual review.")
                                        verified_cand = cand
                                        verified_file = cand_f
                                        v_res = VisualVerificationResult(
                                            status=VerificationStatus.ACCEPTED,
                                            confidence=1.0,
                                            confirmed_matches=["Explicitly approved by manual reviewer"],
                                            evidence=["Manual approval gate: verified and accepted by reviewer"],
                                            is_visual_frame_verified=True,
                                            is_manually_approved=True
                                        )
                                        used_video_ids.add(p_id)
                                        break

                                    # Perform actual frame-level verification across the clip interval
                                    current_vres = await verifier.verify_candidate_clip(
                                        video_path=cand_f,
                                        shot_requirements=shot_ctx,
                                        candidate_metadata=cand.metadata
                                    )

                                    if current_vres.status == VerificationStatus.ACCEPTED:
                                        verified_cand = cand
                                        verified_file = cand_f
                                        v_res = current_vres
                                        used_video_ids.add(p_id)
                                        break
                                    elif current_vres.status in (VerificationStatus.UNCERTAIN, VerificationStatus.MANUAL_REVIEW_REQUIRED):
                                        # Generate visual contact sheet for reviewer inspection
                                        cs_dir = os.path.join(ws["project_dir"], "contact_sheets")
                                        cs_filename = f"scene_{order}_shot_{s_idx}_cand_{p_id}.jpg"
                                        cs_path = os.path.join(cs_dir, cs_filename)
                                        try:
                                            verifier.generate_contact_sheet(
                                                video_path=cand_f,
                                                output_path=cs_path,
                                                shot_requirements=shot_ctx,
                                                candidate_id=p_id,
                                                evidence=current_vres.evidence
                                            )
                                        except Exception as cs_err:
                                            logger.warning(f"Could not render contact sheet for candidate {p_id}: {cs_err}")

                                        # Record in pending manual review registry
                                        self.record_pending_review(
                                            project_id=project_id,
                                            scene_order=order,
                                            shot_index=s_idx,
                                            candidate_video_id=p_id,
                                            local_clip_path=cand_f,
                                            source_url=cand.source_url,
                                            contact_sheet_path=cs_path,
                                            shot_reqs=shot_ctx,
                                            evidence=current_vres.evidence
                                        )
                                        logger.warning(
                                            f"Candidate {p_id} for Scene {order} Shot {s_idx} is UNCERTAIN: "
                                            f"contact sheet saved to {cs_path}. Awaiting manual approval."
                                        )
                                        _score = float((cand.metadata or {}).get("semantic_score", 0.0) or 0.0)
                                        if best_uncertain is None or _score > best_uncertain[3]:
                                            best_uncertain = (cand, cand_f, current_vres, _score)
                                    else:
                                        logger.warning(
                                            f"Candidate {p_id} for Scene {order} Shot {s_idx} REJECTED: {'; '.join(current_vres.contradictions)}"
                                        )
                                        if os.path.exists(cand_f) and not self.is_candidate_approved(project_id, order, s_idx, p_id):
                                            try:
                                                os.remove(cand_f)
                                            except Exception:
                                                pass

                                # CRITICAL ACCURACY RULE:
                                # UNCERTAIN or MANUAL_REVIEW_REQUIRED footage is never treated as verified.
                                # With UNCERTAIN_FOOTAGE_POLICY="block" (default) it is never sent to final
                                # assembly. With "best_effort" the best unverified clip MAY be assembled,
                                # but it is explicitly flagged unverified (unverified_best_effort=True,
                                # is_visual_frame_verified=False) and stays in the pending-review registry.
                                if (
                                    verified_cand is None
                                    and best_uncertain is not None
                                    and str(settings.UNCERTAIN_FOOTAGE_POLICY).strip().lower() == "best_effort"
                                    and ConceptClassifier.classify_feasibility(shot_ctx.get("required_action", ""), shot_ctx.get("primary_subject", ""), shot_ctx.get("visual_prompt", ""))[0] != ConceptFeasibility.CONCEPTUAL_ABSTRACT
                                ):
                                    verified_cand, verified_file, v_res, _ = best_uncertain
                                    v_res.is_visual_frame_verified = False
                                    v_res.evidence.append(
                                        "BEST-EFFORT: no clip was verified; this UNVERIFIED clip was used "
                                        "because UNCERTAIN_FOOTAGE_POLICY=best_effort. Not approved."
                                    )
                                    used_video_ids.add(verified_cand.metadata.get("pexels_id"))
                                    logger.warning(
                                        f"Scene {order} Shot {s_idx}: no verified clip; using best UNVERIFIED candidate "
                                        f"{verified_cand.metadata.get('pexels_id')} (UNCERTAIN_FOOTAGE_POLICY=best_effort). "
                                        f"It is recorded as uncertain in the scene metadata and in the approval registry."
                                    )

                                pending_shot_reviews = [
                                    pr for pr in self.get_pending_approvals(project_id)
                                    if pr.get("shot_key") == f"scene_{order}_shot_{s_idx}"
                                ]
                                if verified_cand is None and settings.ABSTRACT_SHOT_CARD_ENABLED and not pending_shot_reviews:
                                    _feas, _feas_msg = ConceptClassifier.classify_feasibility(
                                        shot_ctx.get("required_action", ""),
                                        shot_ctx.get("primary_subject", ""),
                                        shot_ctx.get("visual_prompt", ""),
                                    )
                                    if _feas == ConceptFeasibility.CONCEPTUAL_ABSTRACT:
                                        # Stock footage cannot show this truthfully: render an honest title card
                                        card_path = os.path.join(clips_dir, f"scene_{order}_shot_{s_idx}_card.mp4")
                                        render_info_card(
                                            heading=sc.get("scene_text", "") or str(shot_ctx.get("primary_subject", "")),
                                            out_path=card_path,
                                            duration_seconds=float(r_shot["duration_seconds"]) + 1.0,
                                        )
                                        sc["requires_animation"] = True
                                        sc["recommended_strategy"] = "diagram_or_motion_graphic"
                                        sc["feasibility_reason"] = _feas_msg
                                        card_meta = {
                                            "shot_index": s_idx,
                                            "allocated_frames": r_shot["allocated_frames"],
                                            "start_frame": r_shot["start_frame"],
                                            "end_frame": r_shot["end_frame"],
                                            "start_seconds": r_shot["start_seconds"],
                                            "end_seconds": r_shot["end_seconds"],
                                            "duration_seconds": r_shot["duration_seconds"],
                                            "required_visible_action": r_shot.get("required_visible_action"),
                                            "primary_subject": r_shot.get("primary_subject"),
                                            "verification_status": "not_applicable_generated_card",
                                            "is_visual_frame_verified": False,
                                            "unverified_best_effort": False,
                                            "generated_info_card": True,
                                            "feasibility_reason": _feas_msg,
                                            "recommended_strategy": "diagram_or_motion_graphic",
                                        }
                                        await self._save_scene_asset(
                                            scene_id=sc["id"],
                                            asset_type=AssetType.VIDEO,
                                            provider_name="generated_card",
                                            local_path=card_path,
                                            source_url=None,
                                            duration_seconds=s_dur,
                                            metadata=card_meta,
                                        )
                                        shot_clips.append(card_path)
                                        shot_metadata_list.append(card_meta)
                                        logger.warning(
                                            f"Scene {order} Shot {s_idx}: abstract process ({_feas_msg}); "
                                            f"rendered a title card instead of unrelated stock footage."
                                        )
                                        continue

                                if verified_cand is None:
                                    pending_shot_reviews = [
                                        pr for pr in self.get_pending_approvals(project_id)
                                        if pr.get("shot_key") == f"scene_{order}_shot_{s_idx}"
                                    ]
                                    if pending_shot_reviews:
                                        cs_info = pending_shot_reviews[0].get("contact_sheet_path", "contact_sheets/")
                                        raise UncertainFootageRequiresApprovalError(
                                            f"Candidate footage for Scene {order} Shot {s_idx} ('{r_shot.get('required_visible_action')}') "
                                            f"is visually uncertain and requires manual approval. Contact sheet generated at: {cs_info}",
                                            pending_approvals=pending_shot_reviews
                                        )

                                    feas, feas_msg = ConceptClassifier.classify_feasibility(
                                        shot_ctx.get("required_action", ""),
                                        shot_ctx.get("primary_subject", ""),
                                        shot_ctx.get("visual_prompt", "")
                                    )
                                    reason_str = f"All candidate clips failed visual verification for shot {s_idx} ('{r_shot.get('required_visible_action')}')"
                                    if feas == ConceptFeasibility.CONCEPTUAL_ABSTRACT:
                                        sc["requires_animation"] = True
                                        sc["recommended_strategy"] = "diagram_or_motion_graphic"
                                        sc["feasibility_reason"] = feas_msg
                                        reason_str += f" ({feas_msg})"
                                    raise NoSuitableMediaFoundError(reason_str)

                            m = verified_cand
                            f = verified_file
                            shot_meta = {
                                **m.metadata,
                                "shot_index": s_idx,
                                "allocated_frames": r_shot["allocated_frames"],
                                "start_frame": r_shot["start_frame"],
                                "end_frame": r_shot["end_frame"],
                                "start_seconds": r_shot["start_seconds"],
                                "end_seconds": r_shot["end_seconds"],
                                "duration_seconds": r_shot["duration_seconds"],
                                "required_visible_action": r_shot.get("required_visible_action"),
                                "primary_subject": r_shot.get("primary_subject"),
                                "required_setting": r_shot.get("required_setting"),
                                "visual_state": r_shot.get("visual_state"),
                                "camera_framing": r_shot.get("camera_framing"),
                                "required_objects": r_shot.get("required_objects") or [],
                                "explicit_exclusions": r_shot.get("explicit_exclusions") or [],
                                "action_event_index": r_shot.get("action_event_index"),
                                "verification_status": v_res.status.value,
                                "verification_confidence": v_res.confidence,
                                "confirmed_matches": v_res.confirmed_matches,
                                "contradictions": v_res.contradictions,
                                "verification_evidence": v_res.evidence,
                                "evaluated_timestamps": v_res.evaluated_timestamps,
                                "is_visual_frame_verified": v_res.is_visual_frame_verified,
                                "unverified_best_effort": v_res.status != VerificationStatus.ACCEPTED,
                                "recommended_strategy": v_res.recommended_strategy
                            }
                            await self._save_scene_asset(
                                scene_id=sc["id"],
                                asset_type=AssetType.VIDEO,
                                provider_name="pexels",
                                local_path=f,
                                source_url=m.source_url,
                                duration_seconds=s_dur,
                                metadata=shot_meta,
                            )
                            shot_clips.append(f)
                            shot_metadata_list.append(shot_meta)

                        except NoSuitableMediaFoundError as e:
                            logger.warning(
                                f"Scene {order} Shot {s_idx} ('{r_shot.get('required_visible_action')}') "
                                f"could not be sourced: {e}"
                            )
                            feas, _ = ConceptClassifier.classify_feasibility(
                                r_shot.get("required_visible_action", ""),
                                r_shot.get("primary_subject", "")
                            )
                            missing_shots.append({
                                "shot_index": s_idx,
                                "required_visible_action": r_shot.get("required_visible_action"),
                                "primary_subject": r_shot.get("primary_subject"),
                                "required_setting": r_shot.get("required_setting"),
                                "status": "unresolved_conceptual_shot" if feas == ConceptFeasibility.CONCEPTUAL_ABSTRACT else "unresolved_footage_not_found",
                                "recommended_strategy": "diagram_or_motion_graphic" if feas == ConceptFeasibility.CONCEPTUAL_ABSTRACT else None,
                                "error": str(e)
                            })

                    if not shot_clips:
                        desc = "multi-shot Scene" if len(reconciled_shots) > 1 else "Scene"
                        raise NoSuitableMediaFoundError(
                            f"No suitable stock video clips could be sourced for {desc} {order} (attempted {len(reconciled_shots)} shots)."
                        )

                    if missing_shots:
                        # Secondary shot(s) missing -> safe documented fallback
                        sc["missing_shots"] = missing_shots
                        sc["visual_coverage_complete"] = False
                        sc["fallback_used"] = "primary_shot_extended_missing_secondary"
                        logger.warning(
                            f"Scene {order} has {len(missing_shots)} missing shot(s). "
                            f"Visual action coverage is incomplete. Fallback: primary shot extended."
                        )
                    else:
                        sc["missing_shots"] = []
                        sc["visual_coverage_complete"] = True
                        sc["fallback_used"] = None

                    sc["media_paths"] = shot_clips
                    sc["media_type"] = "video"
                    sc["shot_metadata"] = shot_metadata_list
                    sc["reconciled_shots"] = [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in reconciled_shots]
                    sc["is_multi_shot"] = bool(had_explicit_shot_plan and len(reconciled_shots) > 1)

            await self._record_pipeline_job(
                project_id=project_id,
                stage="media_sourcing",
                status=JobStatus.COMPLETED,
                job_id=job_id,
            )
            return scenes

        except Exception as e:
            await self._record_pipeline_job(
                project_id=project_id,
                stage="media_sourcing",
                status=JobStatus.FAILED,
                error_message=str(e),
                job_id=job_id,
            )
            raise

    async def _stage_6_source_sfx(
        self, project_id: uuid.UUID, scenes: List[Dict[str, Any]], sounds_dir: str, topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Stage 6: Search and place action SFX from Freesound (with silence fallback)."""
        job_id = await self._record_pipeline_job(
            project_id=project_id,
            stage="sfx_sourcing",
            status=JobStatus.RUNNING,
            provider_name="freesound",
        )
        await self._update_project_state(
            project_id=project_id,
            status=ProjectStatus.GENERATING_VISUALS,
            current_stage="Sourcing action sound effects from Freesound",
            progress_percent=85,
        )

        try:
            freesound = self.get_sfx_provider()
            used_sound_ids: Set[int] = set()

            # Attempt to retrieve workspace timeline if missing in-memory
            workspace = self.get_project_workspace(project_id)
            tl_file = os.path.join(workspace["project_dir"], "action_timeline.json")
            persisted_timelines: Dict[str, Any] = {}
            if os.path.exists(tl_file):
                try:
                    with open(tl_file, "r", encoding="utf-8") as f:
                        persisted_timelines = json.load(f)
                except Exception as tl_read_err:
                    logger.debug(f"Could not read {tl_file}: {tl_read_err}")

            for sc in scenes:
                order = sc["scene_order"]
                dur = compute_frame_exact_duration(sc["duration"], fps=24.0)
                media_paths = sc.get("media_paths", [])

                # 1. Retrieve action timeline (in-memory -> workspace file)
                timeline = sc.get("action_timeline")
                if not timeline and persisted_timelines:
                    timeline = persisted_timelines.get(str(sc["id"]))

                segment_sfx_list = []

                if timeline and isinstance(timeline, list) and len(timeline) > 0:
                    # Action Timeline Mode: dedicated action-specific sourcing
                    logger.info(f"Scene {order}: Processing {len(timeline)} structured action event(s) for SFX.")

                    for ev_idx, raw_ev in enumerate(timeline):
                        ev = raw_ev.model_dump() if hasattr(raw_ev, "model_dump") else dict(raw_ev)
                        sound_type = ev.get("sound_type", "none")
                        action_desc = ev.get("action_description", "")
                        sfx_query = ev.get("sfx_query")

                        # 1. sound_type="none" causes NO Freesound search (zero API calls)
                        if sound_type == "none" or not sfx_query or not str(sfx_query).strip():
                            logger.info(
                                f"Scene {order} Action {ev_idx + 1} ('{action_desc}') is silent "
                                f"(sound_type='{sound_type}'). Skipping Freesound search."
                            )
                            continue

                        # 2. Timing reconciliation against actual measured scene duration (dur)
                        v_start = float(ev.get("visual_start_seconds", 0.0))
                        v_end = float(ev.get("visual_end_seconds", dur))
                        s_start = float(ev.get("sfx_start_seconds")) if ev.get("sfx_start_seconds") is not None else v_start
                        s_dur = float(ev.get("sfx_duration_seconds")) if ev.get("sfx_duration_seconds") is not None else max(1.0, v_end - v_start)
                        s_vol = float(ev.get("sfx_volume")) if ev.get("sfx_volume") is not None else 0.35

                        # Reject negative or out-of-bounds offsets
                        if s_start < 0.0 or s_start >= dur:
                            logger.warning(
                                f"Scene {order} Action {ev_idx + 1}: SFX start offset {s_start:.2f}s is "
                                f"outside scene duration [0, {dur:.2f}s]. Skipping."
                            )
                            continue

                        # Clamp sound duration to scene boundary
                        avail_dur = max(0.5, dur - s_start)
                        seg_dur = min(s_dur, avail_dur)
                        start_off = s_start

                        # 3. Dedicated Sound Search
                        cand = await freesound.search_segment_action_sfx(
                            keyword=str(sfx_query).strip(),
                            segment_duration=seg_dur,
                            used_sound_ids=used_sound_ids,
                            topic=topic
                        )

                        if cand and cand.get("download_url"):
                            # Fail-closed Commercial CC0 Licensing check
                            if not self._is_verified_cc0_license(cand):
                                logger.warning(
                                    f"Rejecting candidate SFX '{cand.get('name')}' (ID {cand.get('freesound_id')}): "
                                    f"License '{cand.get('license')}' ({cand.get('license_url')}) is not verified CC0. "
                                    f"Commercial fail-closed policy requires CC0. Event will have NO SFX."
                                )
                                continue

                            s_id = cand["freesound_id"]
                            prefix = f"scene_{order}_action_{ev_idx + 1}_{s_id}"
                            sfx_file = await freesound.download_audio(cand["download_url"], sounds_dir, prefix)

                            segment_sfx_list.append({
                                "audio_path": sfx_file,
                                "start_offset": start_off,
                                "duration": seg_dur,
                                "freesound_id": s_id,
                                "action_name": cand.get("action_name"),
                                "volume": s_vol,
                                "sound_type": sound_type,
                                "action_description": action_desc,
                                "license": cand.get("license"),
                            })

                            await self._save_scene_asset(
                                scene_id=sc["id"],
                                asset_type=AssetType.AUDIO,
                                provider_name="freesound",
                                local_path=sfx_file,
                                duration_seconds=seg_dur,
                                metadata={
                                    **cand,
                                    "action_description": action_desc,
                                    "sound_type": sound_type,
                                    "volume": s_vol,
                                    "start_offset": start_off,
                                },
                            )

                    sc["segment_sfx_list"] = segment_sfx_list

                else:
                    # Legacy fallback for scenes without an action timeline
                    kw_list = sc.get("search_keywords") or []
                    if len(media_paths) >= 2:
                        half_dur = dur / 2.0
                        segments = [
                            ("clip_a", 0.0, half_dur, kw_list[0] if kw_list else "action"),
                            ("clip_b", half_dur, dur - half_dur, kw_list[1] if len(kw_list) > 1 else (kw_list[0] if kw_list else "action")),
                        ]
                    else:
                        segments = [
                            ("clip_single", 0.0, dur, kw_list[0] if kw_list else "action")
                        ]

                    for seg_label, start_off, seg_dur, kw in segments:
                        cand = await freesound.search_segment_action_sfx(
                            keyword=kw, segment_duration=seg_dur, used_sound_ids=used_sound_ids, topic=topic
                        )
                        if cand and cand.get("download_url"):
                            if not self._is_verified_cc0_license(cand):
                                logger.warning(
                                    f"Legacy SFX candidate '{cand.get('name')}' rejected: not verified CC0."
                                )
                                continue

                            s_id = cand["freesound_id"]
                            prefix = f"scene_{order}_{seg_label}_{s_id}"
                            sfx_file = await freesound.download_audio(cand["download_url"], sounds_dir, prefix)

                            segment_sfx_list.append({
                                "audio_path": sfx_file,
                                "start_offset": start_off,
                                "duration": seg_dur,
                                "freesound_id": s_id,
                                "action_name": cand.get("action_name"),
                            })
                            await self._save_scene_asset(
                                scene_id=sc["id"],
                                asset_type=AssetType.AUDIO,
                                provider_name="freesound",
                                local_path=sfx_file,
                                duration_seconds=seg_dur,
                                metadata=cand,
                            )

                    sc["segment_sfx_list"] = segment_sfx_list

            await self._record_pipeline_job(
                project_id=project_id,
                stage="sfx_sourcing",
                status=JobStatus.COMPLETED,
                job_id=job_id,
            )
            return scenes

        except Exception as e:
            await self._record_pipeline_job(
                project_id=project_id,
                stage="sfx_sourcing",
                status=JobStatus.FAILED,
                error_message=str(e),
                job_id=job_id,
            )
            raise

    def _validate_rendered_video(self, final_mp4: str, expected_output_path: str) -> None:
        """
        Validates that the rendered video exists, is a regular file, has non-zero size,
        strictly matches the expected project workspace path, does not collide with protected media,
        and contains a valid readable video stream via PyAV inspection.
        """
        if not final_mp4 or not isinstance(final_mp4, str) or not final_mp4.strip():
            raise ValueError("Output validation failed: Rendered video path is empty or invalid.")

        norm_final = os.path.normpath(os.path.abspath(final_mp4.strip()))
        norm_expected = os.path.normpath(os.path.abspath(expected_output_path.strip()))

        # 1. Verify path matches expected project workspace output path
        if norm_final != norm_expected:
            raise RuntimeError(
                f"Output validation failed: Rendered video path '{norm_final}' does not match "
                f"expected project path '{norm_expected}'."
            )

        # 2. Verify path does not collide with protected reference media
        output_basename = os.path.basename(norm_final)
        if output_basename in PROTECTED_REFERENCE_MEDIA:
            raise RuntimeError(
                f"Security violation: Rendered video '{output_basename}' matches protected reference media."
            )

        # 3. Verify file exists and is a regular file
        if not os.path.exists(norm_final):
            raise FileNotFoundError(f"Output validation failed: Video file does not exist at '{norm_final}'.")

        if not os.path.isfile(norm_final):
            raise ValueError(f"Output validation failed: Path '{norm_final}' is not a regular file.")

        # 4. Verify file size is greater than zero
        file_size = os.path.getsize(norm_final)
        if file_size <= 0:
            raise ValueError(f"Output validation failed: Video file '{norm_final}' is empty (0 bytes).")

        # 5. Verify container and video stream readability using PyAV
        try:
            container = av.open(norm_final)
            if not container.streams.video:
                container.close()
                raise ValueError(f"Output validation failed: No video stream found in '{norm_final}'.")

            video_stream = container.streams.video[0]
            width = video_stream.codec_context.width or 0
            height = video_stream.codec_context.height or 0
            if width <= 0 or height <= 0:
                container.close()
                raise ValueError(f"Output validation failed: Invalid video dimensions ({width}x{height}) in '{norm_final}'.")

            # Verify at least one video packet/frame is decodable
            frame_decoded = False
            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    frame_decoded = True
                    break
                if frame_decoded:
                    break

            container.close()
            if not frame_decoded:
                raise ValueError(f"Output validation failed: Unable to decode video frames from '{norm_final}'.")

        except Exception as av_err:
            if isinstance(av_err, (RuntimeError, FileNotFoundError)):
                raise
            if isinstance(av_err, ValueError) and not isinstance(av_err, av.FFmpegError):
                raise
            raise RuntimeError(f"Output validation failed: PyAV container inspection failed for '{norm_final}': {av_err}") from av_err

    async def _stage_7_assemble_video(
        self,
        project_id: uuid.UUID,
        scenes: List[Dict[str, Any]],
        output_path: str,
        resampled_dir: Optional[str] = None,
    ) -> str:
        """Stage 7: Assemble clips, mixed audio, and burned-in subtitles into final MP4."""
        job_id = await self._record_pipeline_job(
            project_id=project_id,
            stage="video_assembly",
            status=JobStatus.RUNNING,
            provider_name="moviepy",
        )
        await self._update_project_state(
            project_id=project_id,
            status=ProjectStatus.ASSEMBLING,
            current_stage="Rendering final video and subtitles",
            progress_percent=90,
        )

        try:
            # Execute assembly via off-thread runner to keep asyncio event loop responsive
            loop = asyncio.get_running_loop()
            sig = inspect.signature(self._video_assembler_fn)
            if "resampled_dir" in sig.parameters:
                final_mp4 = await loop.run_in_executor(
                    None,
                    self._video_assembler_fn,
                    scenes,
                    output_path,
                    24,     # fps
                    0.4,    # crossfade_duration
                    resampled_dir,
                )
            else:
                final_mp4 = await loop.run_in_executor(
                    None,
                    self._video_assembler_fn,
                    scenes,
                    output_path,
                    24,     # fps
                    0.4     # crossfade_duration
                )

            # Fix 2: Validate rendered video before updating status to DONE
            self._validate_rendered_video(final_mp4, output_path)

            await self._update_project_state(
                project_id=project_id,
                status=ProjectStatus.DONE,
                current_stage="Video generation complete",
                progress_percent=100,
                video_path=final_mp4,
                completed=True,
            )
            await self._record_pipeline_job(
                project_id=project_id,
                stage="video_assembly",
                status=JobStatus.COMPLETED,
                job_id=job_id,
            )
            return final_mp4

        except Exception as e:
            await self._record_pipeline_job(
                project_id=project_id,
                stage="video_assembly",
                status=JobStatus.FAILED,
                error_message=str(e),
                job_id=job_id,
            )
            raise

    # -------------------------------------------------------------------------
    # Main Pipeline Entry Points (Execution & Resumption)
    # -------------------------------------------------------------------------
    async def resume_pipeline(self, project_id: uuid.UUID) -> str:
        """
        Resumes a paused or failed project from Stage 5 (Media Sourcing) using approved footage
        and existing script, scene beats, and voiceover audio without regenerating them.
        """
        async with self.lock:
            # 1. Fetch project details
            async with self.session_factory() as session:
                stmt = select(Project).where(Project.id == project_id)
                res = await session.execute(stmt)
                project = res.scalars().first()
                if not project:
                    raise ValueError(f"Project with ID '{project_id}' not found.")

                topic = project.topic

                # Fetch existing scene beats
                stmt_scenes = select(Scene).where(Scene.project_id == project_id).order_by(Scene.scene_order)
                res_scenes = await session.execute(stmt_scenes)
                db_scenes = res_scenes.scalars().all()

                if not db_scenes:
                    raise RuntimeError(
                        f"Cannot resume project '{project_id}': no persisted scene records found in database. "
                        f"Run full pipeline instead."
                    )

                scenes = [
                    {
                        "id": sc.id,
                        "scene_order": sc.scene_order,
                        "scene_text": sc.scene_text,
                        "duration": sc.duration_seconds or 0.0,
                        "visual_prompt": sc.visual_prompt or "",
                        "search_keywords": sc.search_keywords or [],
                    }
                    for sc in db_scenes
                ]

            # 2. Prepare isolated workspace
            workspace = self.get_project_workspace(project_id)

            # 3. Restore persisted action timeline and shot plan from workspace files
            tl_file = os.path.join(workspace["project_dir"], "action_timeline.json")
            if os.path.exists(tl_file):
                try:
                    with open(tl_file, "r", encoding="utf-8") as f:
                        tl_map = json.load(f)
                    for sc in scenes:
                        sc["action_timeline"] = tl_map.get(str(sc["id"]))
                except Exception as tl_err:
                    logger.debug(f"Could not restore action_timeline: {tl_err}")

            sp_file = os.path.join(workspace["project_dir"], "shot_plan.json")
            if os.path.exists(sp_file):
                try:
                    with open(sp_file, "r", encoding="utf-8") as f:
                        sp_map = json.load(f)
                    for sc in scenes:
                        sc["shot_plan"] = sp_map.get(str(sc["id"]))
                except Exception as sp_err:
                    logger.debug(f"Could not restore shot_plan: {sp_err}")

            # 4. Verify voiceover audio files exist on disk
            missing_audio = []
            for sc in scenes:
                audio_path = os.path.join(workspace["audio_dir"], f"scene_{sc['scene_order']}_narr.mp3")
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 100:
                    sc["audio_path"] = audio_path
                else:
                    missing_audio.append(sc)

            if missing_audio:
                logger.info(f"Regenerating voiceover for {len(missing_audio)} scene(s)...")
                await self._stage_4_generate_voiceover(project_id, scenes, workspace["audio_dir"])

            try:
                # Stage 5: Stock Video Sourcing (reusing approved candidates from disk)
                scenes = await self._stage_5_source_media(project_id, scenes, workspace["clips_dir"], topic=topic)

                # Stage 6: Action SFX Sourcing (Freesound)
                scenes = await self._stage_6_source_sfx(project_id, scenes, workspace["sounds_dir"], topic=topic)

                # Stage 7: Video Assembly (MoviePy 2.x + PyAV)
                final_video_path = await self._stage_7_assemble_video(
                    project_id, scenes, workspace["output_path"], workspace["resampled_dir"]
                )

                logger.info(f"Pipeline successfully resumed and completed for project {project_id}: {final_video_path}")
                return final_video_path

            except Exception as e:
                err_clean = str(e).strip() or type(e).__name__
                tb = traceback.format_exc()
                logger.error(f"Resume pipeline failed for project {project_id}: {err_clean}\n{tb}")

                await self._update_project_state(
                    project_id=project_id,
                    status=ProjectStatus.FAILED,
                    current_stage="Failed",
                    progress_percent=0,
                    error_message=f"Generation failed: {err_clean}",
                )
                raise

    async def run_pipeline(self, project_id: uuid.UUID, resume: bool = False) -> str:
        """
        Executes the end-to-end video-generation pipeline for a Project.
        Acquires process-local execution lock to guarantee single-job rendering.
        If resume=True, resumes from Stage 5 using existing scenes and approved footage.
        """
        if resume:
            return await self.resume_pipeline(project_id)

        async with self.lock:
            # 1. Fetch project details within a short-lived transaction
            async with self.session_factory() as session:
                stmt = select(Project).where(Project.id == project_id)
                res = await session.execute(stmt)
                project = res.scalars().first()
                if not project:
                    raise ValueError(f"Project with ID '{project_id}' not found.")

                if project.status not in (ProjectStatus.QUEUED, ProjectStatus.FAILED):
                    raise RuntimeError(
                        f"Project '{project_id}' is currently in status '{project.status.value}' and cannot be started."
                    )

                # Safe Retry Handling
                if project.status == ProjectStatus.FAILED:
                    scene_count_res = await session.execute(
                        select(func.count(Scene.id)).where(Scene.project_id == project_id)
                    )
                    scene_count = scene_count_res.scalar() or 0
                    if scene_count > 0:
                        raise RuntimeError(
                            f"Project '{project_id}' contains {scene_count} persisted scene(s). "
                            f"To continue without regenerating script and narration, use resume_pipeline(project_id) "
                            f"or run_pipeline(project_id, resume=True)."
                        )

                topic = project.topic
                desired_duration = project.desired_duration_seconds

            # 2. Prepare safe isolated workspace
            workspace = self.get_project_workspace(project_id)

            try:
                # Stage 1: Script Generation
                script = await self._stage_1_generate_script(project_id, topic, desired_duration)

                # Stage 2: Scene Splitting
                scenes = await self._stage_2_split_scenes(project_id, script, desired_duration)

                # Stage 3: Visual Specifications (Parallel Groq Semaphore=5)
                scenes = await self._stage_3_generate_visual_specs(project_id, scenes, topic=topic)

                # Stage 4: Voiceover Audio (EdgeTTS + PyAV)
                scenes = await self._stage_4_generate_voiceover(project_id, scenes, workspace["audio_dir"])

                # Stage 5: Stock Video Sourcing (Pexels)
                scenes = await self._stage_5_source_media(project_id, scenes, workspace["clips_dir"], topic=topic)

                # Stage 6: Action SFX Sourcing (Freesound)
                scenes = await self._stage_6_source_sfx(project_id, scenes, workspace["sounds_dir"], topic=topic)

                # Stage 7: Video Assembly (MoviePy 2.x + PyAV)
                final_video_path = await self._stage_7_assemble_video(
                    project_id, scenes, workspace["output_path"], workspace["resampled_dir"]
                )

                logger.info(f"Pipeline successfully completed for project {project_id}: {final_video_path}")
                return final_video_path

            except Exception as e:
                err_clean = str(e).strip() or type(e).__name__
                tb = traceback.format_exc()
                logger.error(f"Pipeline failed for project {project_id}: {err_clean}\n{tb}")

                await self._update_project_state(
                    project_id=project_id,
                    status=ProjectStatus.FAILED,
                    current_stage="Failed",
                    progress_percent=0,
                    error_message=f"Generation failed: {err_clean}",
                )
                raise

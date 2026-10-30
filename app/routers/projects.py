import os
import uuid
import logging
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from app.database import get_db, AsyncSessionLocal
from app.dependencies import get_current_user
from app.models.enums import ProjectStatus, AssetType
from app.models.project import Project
from app.models.scene import Scene
from app.models.scene_asset import SceneAsset
from app.models.user import User
from app.schemas.project import (
    ProjectCreateRequest,
    ProjectResponse,
    ProjectListResponse,
    ApprovalActionRequest,
    RejectionActionRequest,
)
from app.services.pipeline_orchestrator import VideoPipelineOrchestrator
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["projects"])

ACTIVE_STATUSES = [
    ProjectStatus.QUEUED,
    ProjectStatus.GENERATING_SCRIPT,
    ProjectStatus.SPLITTING_SCENES,
    ProjectStatus.GENERATING_VISUALS,
    ProjectStatus.GENERATING_VOICE,
    ProjectStatus.ASSEMBLING,
]


async def _run_pipeline_background(project_id: uuid.UUID, resume: bool = False) -> None:
    """Background task executor for end-to-end pipeline."""
    orchestrator = VideoPipelineOrchestrator(session_factory=AsyncSessionLocal)
    try:
        await orchestrator.run_pipeline(project_id, resume=resume)
    except Exception as e:
        logger.error(f"Background pipeline failed for project {project_id}: {e}")


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_project(
    req: ProjectCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Creates an asynchronous video generation job for the authenticated user.
    Returns immediately with project ID and status='queued' without blocking the request.
    Prevents duplicate simultaneous submissions with identical active topics.
    """
    user_id = current_user.id

    # 1. Duplicate submission guard: check if user already has identical active job
    stmt_check = select(Project).where(
        Project.user_id == user_id,
        Project.topic == req.topic,
        Project.status.in_(ACTIVE_STATUSES)
    )
    existing_active = (await db.execute(stmt_check)).scalars().first()
    if existing_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An active video generation job for this topic is already in progress (Project ID: {existing_active.id})."
        )

    # 2. Persist new Project record in QUEUED state
    project = Project(
        user_id=user_id,
        topic=req.topic,
        desired_duration_seconds=req.desired_duration_seconds,
        status=ProjectStatus.QUEUED,
        progress_percent=0,
        current_stage="Queued for generation"
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    # 3. Schedule asynchronous rendering in background
    background_tasks.add_task(_run_pipeline_background, project.id, False)

    logger.info(f"User {user_id} queued video generation project {project.id} ('{req.topic[:50]}...')")
    return project


@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Lists all video projects belonging to the authenticated user."""
    stmt = (
        select(Project)
        .where(Project.user_id == current_user.id)
        .order_by(Project.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    projects = list(result.scalars().all())

    count_stmt = select(func.count(Project.id)).where(Project.user_id == current_user.id)
    total = (await db.execute(count_stmt)).scalar() or 0

    return ProjectListResponse(projects=projects, total=total)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Checks job progress and retrieves current status for a project.
    Strictly verifies ownership: users can only inspect their own projects.
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found."
        )

    if project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not have permission to view this project."
        )

    return project


@router.get("/{project_id}/report")
async def get_project_report(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves the complete per-shot inspection and verification report for a project.
    Includes scene breakdown, visual verifier evidence, approval registry, and settings.
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found."
        )

    if project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not have permission to view this project."
        )

    # Fetch scenes and assets
    scenes_stmt = select(Scene).where(Scene.project_id == project_id).order_by(Scene.scene_order)
    db_scenes = (await db.execute(scenes_stmt)).scalars().all()

    orchestrator = VideoPipelineOrchestrator(session_factory=AsyncSessionLocal)
    approval_registry = orchestrator.load_approval_registry(project_id)

    scenes_data = []
    for sc in db_scenes:
        assets_stmt = select(SceneAsset).where(SceneAsset.scene_id == sc.id)
        assets = (await db.execute(assets_stmt)).scalars().all()
        scenes_data.append({
            "scene_order": sc.scene_order,
            "scene_text": sc.scene_text,
            "duration_seconds": sc.duration_seconds,
            "visual_prompt": sc.visual_prompt,
            "search_keywords": sc.search_keywords,
            "assets": [
                {
                    "asset_type": a.asset_type.value if hasattr(a.asset_type, "value") else str(a.asset_type),
                    "provider_name": a.provider_name,
                    "duration_seconds": a.duration_seconds,
                    "local_path": a.local_path,
                    "metadata": getattr(a, "asset_metadata", None) or {}
                }
                for a in assets
            ]
        })

    return {
        "project_id": str(project.id),
        "topic": project.topic,
        "desired_duration_seconds": project.desired_duration_seconds,
        "status": project.status.value,
        "current_stage": project.current_stage,
        "progress_percent": project.progress_percent,
        "video_path": project.video_path,
        "error_message": project.error_message,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "completed_at": project.completed_at.isoformat() if project.completed_at else None,
        "uncertain_footage_policy": settings.UNCERTAIN_FOOTAGE_POLICY,
        "script_provider": settings.SCRIPT_PROVIDER,
        "scenes": scenes_data,
        "approval_registry": approval_registry
    }


@router.get("/{project_id}/video")
async def get_project_video(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Downloads the completed final rendered MP4 video.
    Strictly enforces user ownership and validates file existence.
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found."
        )

    if project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not have permission to download this video."
        )

    if project.status != ProjectStatus.DONE or not project.video_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video generation is not complete. Current project status: '{project.status.value}'."
        )

    if not os.path.exists(project.video_path) or os.path.getsize(project.video_path) < 1000:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file is not available on disk."
        )

    # Security check: ensure path does not escape media directory
    orchestrator = VideoPipelineOrchestrator(session_factory=AsyncSessionLocal)
    ws = orchestrator.get_project_workspace(project_id)
    normalized_video = os.path.normpath(os.path.abspath(project.video_path))
    normalized_ws = os.path.normpath(os.path.abspath(ws["project_dir"]))
    if not normalized_video.startswith(normalized_ws):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid video path: media path escape detected."
        )

    return FileResponse(
        path=project.video_path,
        media_type="video/mp4",
        filename=f"video_{project.id}.mp4"
    )


@router.get("/{project_id}/contact-sheets/{filename}")
async def get_contact_sheet(
    project_id: uuid.UUID,
    filename: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Serves contact sheet thumbnails for manual candidate review.
    Enforces user ownership and guards against path traversal.
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found."
        )

    if project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not have permission to view contact sheets for this project."
        )

    # Path traversal guard: only basename allowed
    safe_filename = os.path.basename(filename)
    orchestrator = VideoPipelineOrchestrator(session_factory=AsyncSessionLocal)
    ws = orchestrator.get_project_workspace(project_id)
    cs_path = os.path.join(ws["project_dir"], "contact_sheets", safe_filename)

    if not os.path.exists(cs_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contact sheet '{safe_filename}' not found."
        )

    return FileResponse(path=cs_path, media_type="image/jpeg")


@router.get("/{project_id}/approvals")
async def get_pending_approvals(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves all pending manual approvals and contact sheet paths for a project."""
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found."
        )

    if project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied."
        )

    orchestrator = VideoPipelineOrchestrator(session_factory=AsyncSessionLocal)
    pending = orchestrator.get_pending_approvals(project_id)
    return {"project_id": str(project_id), "pending_reviews": pending}


@router.post("/{project_id}/approve")
async def approve_candidate(
    project_id: uuid.UUID,
    action: ApprovalActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Explicitly approves an uncertain candidate video for a specific shot.
    Auditable and persisted in the project's approval registry.
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found."
        )

    if project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied."
        )

    orchestrator = VideoPipelineOrchestrator(session_factory=AsyncSessionLocal)
    record = orchestrator.approve_candidate(
        project_id=project_id,
        scene_order=action.scene_order,
        shot_index=action.shot_index,
        candidate_video_id=action.candidate_video_id,
        reviewer=action.reviewer or current_user.email,
        notes=action.notes or ""
    )
    return {"status": "approved", "project_id": str(project_id), "record": record}


@router.post("/{project_id}/reject")
async def reject_candidate(
    project_id: uuid.UUID,
    action: RejectionActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Explicitly rejects a candidate video for a specific shot.
    Auditable and persisted in the project's approval registry.
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found."
        )

    if project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied."
        )

    orchestrator = VideoPipelineOrchestrator(session_factory=AsyncSessionLocal)
    record = orchestrator.reject_candidate(
        project_id=project_id,
        scene_order=action.scene_order,
        shot_index=action.shot_index,
        candidate_video_id=action.candidate_video_id,
        reviewer=action.reviewer or current_user.email,
        reason=action.reason or ""
    )
    return {"status": "rejected", "project_id": str(project_id), "record": record}


@router.post("/{project_id}/resume", response_model=ProjectResponse, status_code=status.HTTP_202_ACCEPTED)
async def resume_project(
    project_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Resumes a paused or failed project from Stage 5 using approved footage
    and existing script, scene beats, and voiceover audio without regenerating them.
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found."
        )

    if project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied."
        )

    # Verify project has persisted scene records to resume from
    scene_count_res = await db.execute(select(func.count(Scene.id)).where(Scene.project_id == project_id))
    scene_count = scene_count_res.scalar() or 0
    if scene_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot resume project: no persisted scene beats found. Start a new generation job instead."
        )

    # Update project state to QUEUED for resumption
    project.status = ProjectStatus.QUEUED
    project.current_stage = "Resuming video generation from Stage 5"
    project.error_message = None
    await db.commit()
    await db.refresh(project)

    background_tasks.add_task(_run_pipeline_background, project.id, True)
    return project


@router.post("/{project_id}/retry", response_model=ProjectResponse, status_code=status.HTTP_202_ACCEPTED)
async def retry_project(
    project_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retries a failed video generation project safely.
    If scenes were already generated, resumes cleanly without duplicate script generation.
    """
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found."
        )

    if project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied."
        )

    if project.status not in (ProjectStatus.FAILED, ProjectStatus.QUEUED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot retry project currently in '{project.status.value}' status."
        )

    # Check whether scenes were already persisted
    scene_count_res = await db.execute(select(func.count(Scene.id)).where(Scene.project_id == project_id))
    has_scenes = (scene_count_res.scalar() or 0) > 0

    project.status = ProjectStatus.QUEUED
    project.current_stage = "Retrying video generation"
    project.error_message = None
    await db.commit()
    await db.refresh(project)

    background_tasks.add_task(_run_pipeline_background, project.id, has_scenes)
    return project

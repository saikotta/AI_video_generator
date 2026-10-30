"""
Generate ONE short test video through the real pipeline and print a per-shot report.

The API in this project only exposes /health and auth routes, so nothing there starts
the pipeline. This script does it directly: it creates a test user + project in your
database and runs VideoPipelineOrchestrator.run_pipeline().

Usage (from C:\\AiVideoGenerator, backend does NOT need to be running):
    .\\.venv\\Scripts\\python.exe run_test_video.py
    .\\.venv\\Scripts\\python.exe run_test_video.py --topic "How coffee is roasted" --seconds 15

Needs a valid .env (DATABASE_URL, GROQ_API_KEY, PEXELS_API_KEY, FREESOUND_API_KEY).
For a first diagnostic run set UNCERTAIN_FOOTAGE_POLICY=best_effort in .env, then
review every shot marked UNVERIFIED in the report before showing anyone.
"""
import argparse
import asyncio
import json
import os
import uuid

from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.enums import AssetType, ProjectStatus
from app.models.project import Project
from app.models.scene import Scene
from app.models.scene_asset import SceneAsset
from app.models.user import User
from app.security import get_password_hash
from app.services.pipeline_orchestrator import VideoPipelineOrchestrator
from app.services.providers.exceptions import UncertainFootageRequiresApprovalError

TEST_EMAIL = "test_runner@example.com"


async def get_or_create_user() -> uuid.UUID:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(User).where(User.email == TEST_EMAIL))
        user = res.scalars().first()
        if user:
            return user.id
        user = User(email=TEST_EMAIL, password_hash=get_password_hash("not-a-real-login"))
        session.add(user)
        await session.commit()
        return user.id


async def create_project(user_id: uuid.UUID, topic: str, seconds: int) -> uuid.UUID:
    async with AsyncSessionLocal() as session:
        project = Project(
            user_id=user_id,
            topic=topic,
            desired_duration_seconds=seconds,
            status=ProjectStatus.QUEUED,
        )
        session.add(project)
        await session.commit()
        return project.id


async def build_report(project_id: uuid.UUID) -> dict:
    async with AsyncSessionLocal() as session:
        proj = (await session.execute(select(Project).where(Project.id == project_id))).scalars().first()
        scenes = (
            await session.execute(
                select(Scene).where(Scene.project_id == project_id).order_by(Scene.scene_order)
            )
        ).scalars().all()
        report = {
            "project_id": str(project_id),
            "topic": proj.topic,
            "status": proj.status.value,
            "video_path": proj.video_path,
            "error": proj.error_message,
            "script_provider": settings.SCRIPT_PROVIDER,
            "uncertain_footage_policy": settings.UNCERTAIN_FOOTAGE_POLICY,
            "scenes": [],
        }
        for sc in scenes:
            assets = (
                await session.execute(
                    select(SceneAsset).where(
                        SceneAsset.scene_id == sc.id, SceneAsset.asset_type == AssetType.VIDEO
                    )
                )
            ).scalars().all()
            shots = []
            for a in assets:
                m = a.asset_metadata or {}
                shots.append({
                    "shot_index": m.get("shot_index"),
                    "wanted_action": m.get("required_visible_action"),
                    "wanted_subject": m.get("primary_subject"),
                    "pexels_url": a.source_url,
                    "verification_status": m.get("verification_status"),
                    "unverified_best_effort": m.get("unverified_best_effort"),
                    "confirmed_matches": m.get("confirmed_matches"),
                })
            report["scenes"].append({
                "scene_order": sc.scene_order,
                "narration": sc.scene_text,
                "visual_prompt": sc.visual_prompt,
                "search_keywords": sc.search_keywords,
                "shots": shots,
            })
        return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Topic: {report['topic']}   Status: {report['status']}")
    print(f"Script provider: {report['script_provider']}   "
          f"Uncertain-footage policy: {report['uncertain_footage_policy']}")
    if report["error"]:
        print(f"ERROR: {report['error']}")
    print("=" * 70)
    for sc in report["scenes"]:
        print(f"\nScene {sc['scene_order']} narration: {sc['narration']}")
        print(f"  search keywords: {sc['search_keywords']}")
        for sh in sc["shots"]:
            flag = "UNVERIFIED" if sh["unverified_best_effort"] else sh["verification_status"]
            print(f"  shot {sh['shot_index']}: [{flag}] wanted '{sh['wanted_action']}' "
                  f"({sh['wanted_subject']})")
            print(f"           clip: {sh['pexels_url']}")
    print("\nVideo:", report["video_path"])


async def main(topic: str, seconds: int) -> None:
    print(f"Provider={settings.SCRIPT_PROVIDER}  policy={settings.UNCERTAIN_FOOTAGE_POLICY}  "
          f"GROQ key set={bool(settings.GROQ_API_KEY)}  PEXELS key set={bool(settings.PEXELS_API_KEY)}  "
          f"FREESOUND key set={bool(settings.FREESOUND_API_KEY)}")
    user_id = await get_or_create_user()
    project_id = await create_project(user_id, topic, seconds)
    print(f"Created project {project_id}. Running pipeline (this can take a few minutes)...")

    orchestrator = VideoPipelineOrchestrator()
    try:
        await orchestrator.run_pipeline(project_id)
    except UncertainFootageRequiresApprovalError as e:
        print("\nSTOPPED: footage could not be verified and the policy is 'block'.")
        print(e)
        print("Contact sheets are in media\\projects\\<id>\\contact_sheets. For a diagnostic run, "
              "set UNCERTAIN_FOOTAGE_POLICY=best_effort in .env and run again.")
    except Exception as e:  # report whatever failed instead of hiding it
        print(f"\nPIPELINE FAILED: {type(e).__name__}: {e}")

    report = await build_report(project_id)
    print_report(report)

    out_dir = os.path.join(os.path.abspath(settings.MEDIA_STORAGE_PATH), "projects", str(project_id))
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="How a seed grows into a plant")
    ap.add_argument("--seconds", type=int, default=15)
    args = ap.parse_args()
    asyncio.run(main(args.topic, args.seconds))

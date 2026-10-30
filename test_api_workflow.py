"""Targeted API workflow test suite for AI Video Generator.
Validates:
1. Authentication & token validation (cookie & Bearer token).
2. Non-blocking job creation (HTTP 202, background scheduling).
3. Input validation errors (topic length, duration limits -> HTTP 422).
4. Duplicate submission prevention (HTTP 409).
5. Status polling & report retrieval.
6. Multi-user isolation & access control (HTTP 403 on foreign project access).
7. Path traversal & file download security.
8. Candidate approval & rejection persistence.
9. Resume & retry lifecycle.

Run: python test_api_workflow.py
"""

import os
import sys
import uuid
import asyncio
from unittest.mock import patch, AsyncMock

import httpx
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import NullPool

from app.main import app
from app.config import settings
from app.database import get_db
from app.models.user import User
from app.models.project import Project
from app.models.scene import Scene
from app.models.scene_asset import SceneAsset
from app.models.enums import ProjectStatus, AssetType
from app.security import create_access_token, get_password_hash
from app.services.pipeline_orchestrator import VideoPipelineOrchestrator

fails = []

def check(name: str, cond: bool):
    status_str = "PASS " if cond else "FAIL "
    print(status_str + name)
    if not cond:
        fails.append(name)


async def run_tests():
    test_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    conn = await test_engine.connect()
    trans = await conn.begin()

    test_session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)

    async def override_get_db():
        yield test_session

    app.dependency_overrides[get_db] = override_get_db

    # Create two test users in the transaction
    user_a_id = uuid.uuid4()
    user_a_email = f"user_a_{user_a_id.hex[:8]}@example.com"
    user_a = User(
        id=user_a_id,
        email=user_a_email,
        password_hash=get_password_hash("SecretPassword123!")
    )

    user_b_id = uuid.uuid4()
    user_b_email = f"user_b_{user_b_id.hex[:8]}@example.com"
    user_b = User(
        id=user_b_id,
        email=user_b_email,
        password_hash=get_password_hash("SecretPassword456!")
    )

    test_session.add(user_a)
    test_session.add(user_b)
    await test_session.commit()

    token_a = create_access_token({"sub": str(user_a_id), "email": user_a_email})
    token_b = create_access_token({"sub": str(user_b_id), "email": user_b_email})

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    try:
        # --- 1. Authentication & Security ---
        r1 = await client.post("/projects/", json={"topic": "Test video"})
        check("unauthenticated POST /projects/ returns 401", r1.status_code == 401)

        r2 = await client.get("/projects/")
        check("unauthenticated GET /projects/ returns 401", r2.status_code == 401)

        # Cookie authentication fallback test
        client_cookie = AsyncClient(transport=ASGITransport(app=app), base_url="http://test", cookies={"access_token": token_a})
        r_cookie = await client_cookie.get("/projects/")
        check("cookie authentication succeeds with 200", r_cookie.status_code == 200)
        await client_cookie.aclose()

        # --- 2. Input Validation ---
        r_empty = await client.post("/projects/", json={"topic": "", "desired_duration_seconds": 15}, headers=headers_a)
        check("empty topic rejected with 422", r_empty.status_code == 422)

        r_short = await client.post("/projects/", json={"topic": "Valid Topic", "desired_duration_seconds": 2}, headers=headers_a)
        check("duration < 5 rejected with 422", r_short.status_code == 422)

        r_long = await client.post("/projects/", json={"topic": "Valid Topic", "desired_duration_seconds": 600}, headers=headers_a)
        check("duration > 300 rejected with 422", r_long.status_code == 422)

        # --- 3. Job Creation (Non-blocking) & Duplicate Prevention ---
        with patch("app.routers.projects._run_pipeline_background", new_callable=AsyncMock) as mock_bg:
            payload = {
                "topic": "Cellular respiration and mitochondrial ATP synthesis",
                "desired_duration_seconds": 25
            }
            r_create = await client.post("/projects/", json=payload, headers=headers_a)
            check("job creation returns 202 Accepted", r_create.status_code == 202)
            cdata = r_create.json()
            proj_a_id = cdata.get("id")
            check("job response has queued status and initial progress",
                  cdata.get("status") == "queued" and cdata.get("progress_percent") == 0)
            check("background rendering task was scheduled", mock_bg.called)

            # Duplicate submission attempt for same active topic
            r_dup = await client.post("/projects/", json=payload, headers=headers_a)
            check("duplicate submission for active topic returns 409 Conflict", r_dup.status_code == 409)

        # --- 4. Status Polling & Multi-User Isolation ---
        r_poll_a = await client.get(f"/projects/{proj_a_id}", headers=headers_a)
        check("owner can poll project status (200 OK)", r_poll_a.status_code == 200 and r_poll_a.json()["id"] == proj_a_id)

        r_poll_b = await client.get(f"/projects/{proj_a_id}", headers=headers_b)
        check("foreign user polling project is blocked with 403 Forbidden", r_poll_b.status_code == 403)

        missing_uuid = uuid.uuid4()
        r_poll_missing = await client.get(f"/projects/{missing_uuid}", headers=headers_a)
        check("polling nonexistent project returns 404 Not Found", r_poll_missing.status_code == 404)

        # Report endpoint isolation
        r_rep_a = await client.get(f"/projects/{proj_a_id}/report", headers=headers_a)
        check("owner can access report endpoint (200 OK)", r_rep_a.status_code == 200)
        r_rep_b = await client.get(f"/projects/{proj_a_id}/report", headers=headers_b)
        check("foreign user accessing report is blocked with 403 Forbidden", r_rep_b.status_code == 403)

        # --- 5. Project Listing Isolation ---
        with patch("app.routers.projects._run_pipeline_background", new_callable=AsyncMock):
            await client.post("/projects/", json={"topic": "User B exclusive topic", "desired_duration_seconds": 15}, headers=headers_b)

        list_a = (await client.get("/projects/", headers=headers_a)).json()
        list_b = (await client.get("/projects/", headers=headers_b)).json()
        check("user A project listing contains only user A projects",
              all(p["user_id"] == str(user_a_id) for p in list_a["projects"]))
        check("user B project listing contains only user B projects",
              all(p["user_id"] == str(user_b_id) for p in list_b["projects"]))

        # --- 6. Video Download Security & Path Traversal Guards ---
        r_dl_early = await client.get(f"/projects/{proj_a_id}/video", headers=headers_a)
        check("video download rejected with 400 when not in DONE status", r_dl_early.status_code == 400)

        r_dl_foreign = await client.get(f"/projects/{proj_a_id}/video", headers=headers_b)
        check("video download by foreign user blocked with 403 Forbidden", r_dl_foreign.status_code == 403)

        # Path traversal guard on contact sheet
        r_cs_traversal = await client.get(f"/projects/{proj_a_id}/contact-sheets/..%2F..%2Fsecret.txt", headers=headers_a)
        check("contact sheet traversal attempt safely blocked (404/not allowed)", r_cs_traversal.status_code in (400, 404))

        # --- 7. Candidate Approval & Rejection Persistence ---
        r_app_foreign = await client.post(
            f"/projects/{proj_a_id}/approve",
            json={"scene_order": 1, "shot_index": 1, "candidate_video_id": "px_11111", "notes": "Approved by A"},
            headers=headers_b
        )
        check("foreign user cannot approve candidate (403 Forbidden)", r_app_foreign.status_code == 403)

        r_app_owner = await client.post(
            f"/projects/{proj_a_id}/approve",
            json={"scene_order": 1, "shot_index": 1, "candidate_video_id": "px_11111", "notes": "Approved by A"},
            headers=headers_a
        )
        check("owner can approve candidate (200 OK)", r_app_owner.status_code == 200)

        orchestrator = VideoPipelineOrchestrator()
        registry = orchestrator.load_approval_registry(uuid.UUID(proj_a_id))
        check("approval registry persists approved candidate ID",
              registry.get("approved_candidates", {}).get("scene_1_shot_1", {}).get("candidate_video_id") == "px_11111")

        r_rej_owner = await client.post(
            f"/projects/{proj_a_id}/reject",
            json={"scene_order": 1, "shot_index": 1, "candidate_video_id": "px_22222", "reason": "Incorrect biological motion"},
            headers=headers_a
        )
        check("owner can reject candidate (200 OK)", r_rej_owner.status_code == 200)

        registry = orchestrator.load_approval_registry(uuid.UUID(proj_a_id))
        check("approval registry persists rejected candidate ID",
              "px_22222" in [str(x) for x in registry.get("rejected_candidates", {}).get("scene_1_shot_1", [])])

        # --- 8. Resume & Retry Lifecycle ---
        with patch("app.routers.projects._run_pipeline_background", new_callable=AsyncMock) as mock_resume_bg:
            # Resume without scene beats should fail
            r_res_no_scenes = await client.post(f"/projects/{proj_a_id}/resume", headers=headers_a)
            check("resume without scene beats returns 400 Bad Request", r_res_no_scenes.status_code == 400)

            # Insert persisted scene beat for project
            scene = Scene(
                id=uuid.uuid4(),
                project_id=uuid.UUID(proj_a_id),
                scene_order=1,
                scene_text="The seed absorbs water and swells.",
                duration_seconds=4.0
            )
            test_session.add(scene)

            # Add SceneAsset records (one with metadata, one with None) to test report builder
            asset1 = SceneAsset(
                id=uuid.uuid4(),
                scene_id=scene.id,
                asset_type=AssetType.VIDEO,
                provider_name="pexels",
                local_path="/tmp/scene_1_shot_1.mp4",
                duration_seconds=4.0,
                asset_metadata={"shot_index": 1, "is_visual_frame_verified": True, "verification_status": "accepted"}
            )
            asset2 = SceneAsset(
                id=uuid.uuid4(),
                scene_id=scene.id,
                asset_type=AssetType.AUDIO,
                provider_name="edge_tts",
                local_path="/tmp/scene_1_narr.mp3",
                duration_seconds=4.0,
                asset_metadata=None
            )
            test_session.add_all([asset1, asset2])
            await test_session.commit()

            # Regression test: report endpoint on project with SceneAssets must return 200 (not 500 AttributeError)
            r_rep_with_assets = await client.get(f"/projects/{proj_a_id}/report", headers=headers_a)
            check("report endpoint on project with scene assets returns 200 (no 500 AttributeError)",
                  r_rep_with_assets.status_code == 200)
            rep_data = r_rep_with_assets.json()
            check("report scenes contain assets with serialized metadata",
                  len(rep_data["scenes"]) == 1 and len(rep_data["scenes"][0]["assets"]) == 2
                  and rep_data["scenes"][0]["assets"][0]["metadata"].get("is_visual_frame_verified") is True
                  and rep_data["scenes"][0]["assets"][1]["metadata"] == {})

            # Now resume should succeed and resume from Stage 5
            r_res_ok = await client.post(f"/projects/{proj_a_id}/resume", headers=headers_a)
            check("resume with scene beats succeeds with 202 Accepted", r_res_ok.status_code == 202)
            check("resume schedules background runner with resume=True",
                  mock_resume_bg.called and mock_resume_bg.call_args[0][1] is True)

            # Mark project failed and test retry
            proj_record = (await test_session.execute(select(Project).where(Project.id == uuid.UUID(proj_a_id)))).scalars().first()
            proj_record.status = ProjectStatus.FAILED
            await test_session.commit()

            r_retry_ok = await client.post(f"/projects/{proj_a_id}/retry", headers=headers_a)
            check("retry failed project succeeds with 202 Accepted", r_retry_ok.status_code == 202)

    finally:
        await client.aclose()
        app.dependency_overrides.clear()
        await test_session.close()
        await trans.rollback()
        await conn.close()
        await test_engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_tests())
    print("\n------------------------------------------------------------")
    if fails:
        print(f"FAILED ({len(fails)} failures): {fails}")
        sys.exit(1)
    else:
        print("ALL API WORKFLOW TESTS PASSED (0 failures)")
        sys.exit(0)

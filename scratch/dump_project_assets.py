import asyncio
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\AiVideoGenerator")

from app.database import AsyncSessionLocal
from app.models.scene_asset import SceneAsset
from app.models.scene import Scene
from sqlalchemy import select

async def dump_assets():
    project_id = "2143bbc8-8061-4639-b974-475f6071336d"
    async with AsyncSessionLocal() as session:
        # Get scenes
        s_res = await session.execute(select(Scene).where(Scene.project_id == project_id).order_by(Scene.scene_order))
        scenes = s_res.scalars().all()
        scene_map = {str(s.id): s for s in scenes}

        # Get assets
        a_res = await session.execute(select(SceneAsset).order_by(SceneAsset.created_at))
        assets = a_res.scalars().all()

        proj_assets = [a for a in assets if project_id in str(a.local_path)]
        print(f"Total project assets in DB: {len(proj_assets)}")

        for idx, a in enumerate(proj_assets):
            s = scene_map.get(str(a.scene_id))
            meta = a.asset_metadata or {}
            print(f"\n--- Asset {idx+1}: [{a.asset_type.value.upper()}] Scene {s.scene_order if s else '?'}")
            print(f"  File: {os.path.basename(a.local_path or '')}")
            print(f"  Provider: {a.provider_name} | Duration: {a.duration_seconds}s")
            if a.asset_type.value == "video":
                print(f"  Pexels ID: {meta.get('pexels_id')} | Score: {meta.get('score')} (Semantic: {meta.get('semantic_score')}, Tech: {meta.get('technical_score')})")
                print(f"  Photographer: {meta.get('photographer')}")
                print(f"  Semantic Match: {meta.get('semantic_reasons')}")
                print(f"  Visual Coverage Complete: {meta.get('visual_coverage_complete')}")
                print(f"  Fallbacks Used: {meta.get('fallbacks_used')}")
                print(f"  Shot Metadata: {meta.get('shot_metadata')}")
            elif a.asset_type.value == "audio" and a.provider_name == "freesound":
                print(f"  Freesound ID: {meta.get('freesound_id')} | Name: '{meta.get('name')}'")
                print(f"  License: {meta.get('license')} ({meta.get('license_url')})")
                print(f"  Action: '{meta.get('action_description')}' | Offset: {meta.get('start_offset')}s | Vol: {meta.get('volume')}")

asyncio.run(dump_assets())

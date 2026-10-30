import asyncio
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\AiVideoGenerator")

from app.database import AsyncSessionLocal
from app.models.scene_asset import SceneAsset
from sqlalchemy import select

async def get_urls():
    project_id = "2143bbc8-8061-4639-b974-475f6071336d"
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(SceneAsset).where(SceneAsset.asset_type == 'video').order_by(SceneAsset.created_at))
        for a in res.scalars().all():
            if project_id in str(a.local_path):
                meta = a.asset_metadata or {}
                print(os.path.basename(a.local_path))
                print('  URL:', a.source_url)
                print('  Photographer:', meta.get('photographer'))
                print('  Score:', meta.get('score'))
                print('  Semantic Reasons:', meta.get('semantic_reasons'))

asyncio.run(get_urls())

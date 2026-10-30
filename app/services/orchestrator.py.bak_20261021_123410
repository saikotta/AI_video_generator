import asyncio
from typing import List, Dict, Any
from app.services.providers.base import ScriptProvider

async def process_visual_specifications_parallel(
    provider: ScriptProvider,
    scenes: List[Dict[str, Any]],
    max_concurrency: int = 5
) -> List[Dict[str, Any]]:
    """
    Executes visual specification generation concurrently across scenes using bounded semaphore concurrency.
    Calls the exact unchanged provider.generate_visual_specification(scene_text) method.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_single_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
        scene_text = scene.get("scene_text", "")
        async with semaphore:
            return await provider.generate_visual_specification(scene_text)

    # Process all scenes in parallel with concurrency cap
    return await asyncio.gather(*[_process_single_scene(scene) for scene in scenes])

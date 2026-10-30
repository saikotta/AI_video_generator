import asyncio
import inspect
from typing import List, Dict, Any, Optional
from app.services.providers.base import ScriptProvider

async def process_visual_specifications_parallel(
    provider: ScriptProvider,
    scenes: List[Dict[str, Any]],
    max_concurrency: int = 5,
    topic: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Executes visual specification generation concurrently across scenes using bounded semaphore concurrency.
    Calls the exact unchanged provider.generate_visual_specification(scene_text) method.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    full_narration = " ".join(
        str(scene.get("scene_text", "")).strip()
        for scene in scenes
        if str(scene.get("scene_text", "")).strip()
    )

    try:
        supports_context = (
            "context" in inspect.signature(
                provider.generate_visual_specification
            ).parameters
        )
    except (TypeError, ValueError):
        supports_context = False

    async def _process_single_scene(
        index: int, scene: Dict[str, Any]
    ) -> Dict[str, Any]:
        scene_text = scene.get("scene_text", "")
        context = None
        if topic or full_narration:
            context = (
                f"Video topic: {topic or 'Not specified'}\n"
                f"Full narration: {full_narration}\n"
                f"This is scene {index + 1} of {len(scenes)}. "
                "Use this context to resolve references and pronouns, "
                "but depict only the actions relevant to this scene."
            )

        async with semaphore:
            if supports_context:
                return await provider.generate_visual_specification(
                    scene_text, context=context
                )
            return await provider.generate_visual_specification(scene_text)

    return await asyncio.gather(*[
        _process_single_scene(index, scene)
        for index, scene in enumerate(scenes)
    ])

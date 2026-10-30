import asyncio
import inspect
from typing import List, Dict, Any, Optional
from app.services.providers.base import ScriptProvider

async def process_visual_specifications_parallel(
    provider: ScriptProvider,
    scenes: List[Dict[str, Any]],
    max_concurrency: int = 5,
    topic: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Executes visual specification generation concurrently across scenes using bounded semaphore concurrency.
    Calls the exact unchanged provider.generate_visual_specification(scene_text) method.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    # A beat like "they crack, are cooled quickly." is meaningless alone (the planner once turned it into
    # a hot glass rod). Give every beat the video topic and the full narration so pronouns resolve.
    full_narration = " ".join(str(s.get("scene_text", "")).strip() for s in scenes)
    try:
        supports_context = "context" in inspect.signature(provider.generate_visual_specification).parameters
    except (TypeError, ValueError):
        supports_context = False

    def _context_for(index: int) -> Optional[str]:
        if not (topic and str(topic).strip()) and not full_narration.strip():
            return None
        lines = []
        if topic and str(topic).strip():
            lines.append(f"Video topic: {str(topic).strip()}")
        lines.append(f"Full narration: {full_narration}")
        lines.append(f"This beat is scene {index + 1} of {len(scenes)}.")
        return "\n".join(lines)

    async def _process_single_scene(index: int, scene: Dict[str, Any]) -> Dict[str, Any]:
        scene_text = scene.get("scene_text", "")
        async with semaphore:
            if supports_context:
                return await provider.generate_visual_specification(scene_text, context=_context_for(index))
            return await provider.generate_visual_specification(scene_text)

    # Process all scenes in parallel with concurrency cap
    return await asyncio.gather(*[_process_single_scene(i, scene) for i, scene in enumerate(scenes)])

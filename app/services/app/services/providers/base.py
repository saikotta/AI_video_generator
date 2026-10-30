from abc import ABC, abstractmethod
from typing import Optional

class ScriptProvider(ABC):
    @abstractmethod
    async def generate_script(self, topic: str, desired_duration_seconds: int) -> str:
        """
        Generates a narration script for a video given a topic and target duration in seconds.

        :param topic: The topic/subject of the video.
        :param desired_duration_seconds: The desired length of the video in seconds.
        :return: Pure narration text string matching constraints.
        """
        pass

    @abstractmethod
    async def split_into_scenes(
        self,
        script: str,
        desired_duration_seconds: int
    ) -> list[dict]:
        """
        Splits a narration script into duration-aware scene objects.

        :param script: Narration script text to split.
        :param desired_duration_seconds: Target video duration in seconds.
        :return: List of dicts matching [{'scene_order': int, 'scene_text': str}, ...]
        """
        pass

    @abstractmethod
    async def generate_visual_specification(
        self,
        scene_text: str,
        context: Optional[str] = None
    ) -> dict:
        """
        Generates visual prompt description and search keywords for a scene beat.

        :param scene_text: Spoken narration text for the scene.
        :return: Dict matching {'visual_prompt': str, 'search_keywords': list[str]}
        """
        pass

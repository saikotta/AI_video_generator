from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator

class SceneItemSchema(BaseModel):
    scene_order: int = Field(..., description="1-indexed sequential scene order integer")
    scene_text: str = Field(..., description="Non-empty spoken narration text for this scene beat")

    @field_validator('scene_text')
    @classmethod
    def validate_scene_text_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("scene_text cannot be empty")
        return cleaned

class SceneListSchema(BaseModel):
    scenes: List[SceneItemSchema] = Field(..., description="List of scene beat objects")

    @field_validator('scenes')
    @classmethod
    def validate_scenes_non_empty(cls, scenes: List[SceneItemSchema]) -> List[SceneItemSchema]:
        if not scenes:
            raise ValueError("scenes list cannot be empty")
        return scenes

    @model_validator(mode='after')
    def validate_scenes_structure(self) -> 'SceneListSchema':
        if not self.scenes:
            raise ValueError("scenes list cannot be empty")

        # 1. Verify strict sequential order starting at 1 (1, 2, 3, ..., N)
        for idx, scene in enumerate(self.scenes, start=1):
            if scene.scene_order != idx:
                raise ValueError(
                    f"Invalid scene_order: expected {idx}, got {scene.scene_order}. "
                    f"scene_order must start at 1 and increase sequentially without gaps or reordering."
                )

        # 2. Verify no duplicate scene_text across scenes
        seen_texts = set()
        for scene in self.scenes:
            norm_text = scene.scene_text.strip().lower()
            if norm_text in seen_texts:
                raise ValueError(f"Duplicate scene_text found: '{scene.scene_text}'")
            seen_texts.add(norm_text)

        return self

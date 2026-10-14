from typing import List
from pydantic import BaseModel, Field, field_validator

class VisualSpecificationSchema(BaseModel):
    visual_prompt: str = Field(..., description="Short concrete visual description of observable scene elements")
    search_keywords: List[str] = Field(..., description="List of concrete stock media search terms")

    @field_validator('visual_prompt')
    @classmethod
    def validate_visual_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("visual_prompt cannot be empty")
        return cleaned

    @field_validator('search_keywords')
    @classmethod
    def validate_search_keywords(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("search_keywords list cannot be empty")
        cleaned_keywords = [k.strip() for k in value if k and k.strip()]
        if len(cleaned_keywords) < 2:
            raise ValueError(f"search_keywords list must contain at least 2 useful keywords, got {len(cleaned_keywords)}")
        return cleaned_keywords

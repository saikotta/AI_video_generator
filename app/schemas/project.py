import uuid
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator


class ProjectCreateRequest(BaseModel):
    topic: str = Field(..., description="Topic or prompt for video generation", min_length=3, max_length=500)
    desired_duration_seconds: int = Field(default=15, ge=5, le=300, description="Desired video duration in seconds (5 to 300)")

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Topic cannot be empty or whitespace only")
        return cleaned


class ProjectResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    topic: str
    desired_duration_seconds: int
    status: str
    current_stage: Optional[str] = None
    progress_percent: int = 0
    video_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]
    total: int


class ApprovalActionRequest(BaseModel):
    scene_order: int = Field(..., ge=1, description="1-indexed sequence number of the scene")
    shot_index: int = Field(..., ge=1, description="1-indexed sequence number of the shot")
    candidate_video_id: Any = Field(..., description="ID of the candidate clip being approved")
    reviewer: Optional[str] = Field("manual_reviewer", description="Identifier of reviewer")
    notes: Optional[str] = Field("", description="Optional notes on reason for approval")


class RejectionActionRequest(BaseModel):
    scene_order: int = Field(..., ge=1, description="1-indexed sequence number of the scene")
    shot_index: int = Field(..., ge=1, description="1-indexed sequence number of the shot")
    candidate_video_id: Any = Field(..., description="ID of the candidate clip being rejected")
    reviewer: Optional[str] = Field("manual_reviewer", description="Identifier of reviewer")
    reason: Optional[str] = Field("", description="Optional reason for rejection")

import uuid
from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base
from app.models.enums import ProjectStatus

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.scene import Scene
    from app.models.pipeline_job import PipelineJob

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    desired_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    script: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[ProjectStatus] = mapped_column(
        SQLEnum(ProjectStatus, name="project_status", values_callable=lambda obj: [e.value for e in obj]),
        default=ProjectStatus.QUEUED,
        nullable=False
    )
    current_stage: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    video_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped["User"] = relationship("User", back_populates="projects")
    scenes: Mapped[List["Scene"]] = relationship(
        "Scene", back_populates="project", cascade="all, delete-orphan"
    )
    pipeline_jobs: Mapped[List["PipelineJob"]] = relationship(
        "PipelineJob", back_populates="project", cascade="all, delete-orphan"
    )

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Any, TYPE_CHECKING
from sqlalchemy import String, Text, Integer, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.models.base import Base

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.scene_asset import SceneAsset

class Scene(Base):
    __tablename__ = "scenes"
    __table_args__ = (
        UniqueConstraint("project_id", "scene_order", name="uq_scenes_project_scene_order"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    scene_order: Mapped[int] = mapped_column(Integer, nullable=False)
    scene_text: Mapped[str] = mapped_column(Text, nullable=False)
    visual_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    search_keywords: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False
    )

    project: Mapped["Project"] = relationship("Project", back_populates="scenes")
    scene_assets: Mapped[List["SceneAsset"]] = relationship(
        "SceneAsset", back_populates="scene", cascade="all, delete-orphan"
    )

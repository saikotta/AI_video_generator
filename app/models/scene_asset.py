import uuid
from datetime import datetime, timezone
from typing import Optional, Any, TYPE_CHECKING
from sqlalchemy import Text, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.models.base import Base
from app.models.enums import AssetType

if TYPE_CHECKING:
    from app.models.scene import Scene

class SceneAsset(Base):
    __tablename__ = "scene_assets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False
    )
    asset_type: Mapped[AssetType] = mapped_column(
        SQLEnum(AssetType, name="asset_type", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False
    )
    provider_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    storage_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # "metadata" column in database mapped without overriding DeclarativeBase metadata attribute
    asset_metadata: Mapped[Optional[Any]] = mapped_column("metadata", JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    scene: Mapped["Scene"] = relationship("Scene", back_populates="scene_assets")

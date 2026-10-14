import uuid
from datetime import datetime, timezone
from sqlalchemy import Text, Integer, Boolean, DateTime, UniqueConstraint, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base
from app.models.enums import ProviderStage

class ProviderConfig(Base):
    __tablename__ = "provider_configs"
    __table_args__ = (
        UniqueConstraint("stage", "provider_name", name="uq_provider_configs_stage_provider"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    stage: Mapped[ProviderStage] = mapped_column(
        SQLEnum(ProviderStage, name="provider_stage", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False
    )
    provider_name: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False
    )

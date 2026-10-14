import uuid
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base
from app.models.enums import JobStatus

if TYPE_CHECKING:
    from app.models.project import Project

class PipelineJob(Base):
    __tablename__ = "pipeline_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        SQLEnum(JobStatus, name="job_status", values_callable=lambda obj: [e.value for e in obj]),
        default=JobStatus.QUEUED,
        nullable=False
    )
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    provider_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    project: Mapped["Project"] = relationship("Project", back_populates="pipeline_jobs")

"""Database models for paper versioning and improvement tracking."""

from enum import Enum
from typing import Optional, List
from uuid import UUID

from sqlalchemy import String, Text, Integer, Float, ForeignKey, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import BaseModel


class VersionType(str, Enum):
    """Version increment type following semantic versioning."""

    MAJOR = "major"  # Complete rewrite, structural changes
    MINOR = "minor"  # Significant content improvements
    PATCH = "patch"  # Minor edits, typo fixes


class ImprovementStatus(str, Enum):
    """Status of improvement application."""

    SUGGESTED = "suggested"  # Generated but not applied
    APPLIED = "applied"  # Successfully applied to paper
    REVERTED = "reverted"  # Applied then reverted
    REJECTED = "rejected"  # User rejected the suggestion


class PaperVersion(BaseModel):
    """Paper version history with semantic versioning.

    Captures complete snapshot of paper state at each version.
    Enables rollback, comparison, and progress tracking.
    """

    __tablename__ = "paper_versions"

    # Foreign keys
    paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    parent_version_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("paper_versions.id", ondelete="SET NULL"), nullable=True
    )

    # Semantic version
    major: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    minor: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    patch: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Content snapshot
    content_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    sections_snapshot: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )  # {section_name: content}

    # Metadata
    version_type: Mapped[str] = mapped_column(
        String(20), default=VersionType.PATCH.value, nullable=False
    )
    change_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    paper: Mapped["Paper"] = relationship("Paper", back_populates="versions")
    parent: Mapped[Optional["PaperVersion"]] = relationship(
        "PaperVersion", remote_side="PaperVersion.id", back_populates="children"
    )
    children: Mapped[List["PaperVersion"]] = relationship(
        "PaperVersion", back_populates="parent"
    )
    improvements: Mapped[List["ImprovementHistory"]] = relationship(
        "ImprovementHistory",
        back_populates="version",
        cascade="all, delete-orphan",
    )

    @property
    def version_string(self) -> str:
        """Get semantic version string (e.g., '1.2.3')."""
        return f"{self.major}.{self.minor}.{self.patch}"


class ImprovementHistory(BaseModel):
    """History of applied improvements with learning data.

    Tracks each improvement attempt with before/after metrics,
    enabling learning from successful patterns via ChromaDB.
    """

    __tablename__ = "improvement_history"

    # Foreign keys
    paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    version_id: Mapped[UUID] = mapped_column(
        ForeignKey("paper_versions.id", ondelete="CASCADE"), nullable=False
    )

    # Improvement details
    section_name: Mapped[str] = mapped_column(String(200), nullable=False)
    original_content: Mapped[str] = mapped_column(Text, nullable=False)
    improved_content: Mapped[str] = mapped_column(Text, nullable=False)

    # Analysis
    improvement_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # clarity, coherence, methodology, etc.
    changes_made: Mapped[list] = mapped_column(
        JSON, nullable=False
    )  # ["Fixed passive voice", "Added citations"]

    # Metrics
    status: Mapped[str] = mapped_column(
        String(20), default=ImprovementStatus.SUGGESTED.value, nullable=False
    )
    quality_before: Mapped[float] = mapped_column(Float, nullable=False)
    quality_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    improvement_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Learning metadata
    user_feedback: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # User comments
    success_rating: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # 1-5 stars

    # ChromaDB integration
    pattern_embedding_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )

    # Relationships
    paper: Mapped["Paper"] = relationship("Paper", back_populates="improvement_history")
    version: Mapped["PaperVersion"] = relationship(
        "PaperVersion", back_populates="improvements"
    )


class IterationSession(BaseModel):
    """Iterative improvement session with target metrics.

    Manages multi-round improvement loops with quality targets.
    Tracks progress and applies improvements systematically.
    """

    __tablename__ = "iteration_sessions"

    # Foreign keys
    paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    start_version_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("paper_versions.id", ondelete="SET NULL"), nullable=True
    )
    current_version_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("paper_versions.id", ondelete="SET NULL"), nullable=True
    )

    # Session configuration
    target_score: Mapped[float] = mapped_column(
        Float, nullable=False
    )  # Target quality score
    max_iterations: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    focus_areas: Mapped[list] = mapped_column(
        JSON, nullable=False
    )  # ["clarity", "methodology"]

    # Progress tracking
    current_iteration: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_complete: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Results
    improvements_applied: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    score_improvement: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Relationships
    paper: Mapped["Paper"] = relationship("Paper", back_populates="iteration_sessions")

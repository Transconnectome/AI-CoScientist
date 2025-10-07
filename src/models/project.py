"""Project-related database models."""

from enum import Enum
from typing import List, Optional
from uuid import UUID

from sqlalchemy import String, Text, ForeignKey, Integer, Float, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import BaseModel


class ProjectStatus(str, Enum):
    """Project status enum."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    PAUSED = "paused"


class HypothesisStatus(str, Enum):
    """Hypothesis status enum."""
    GENERATED = "generated"
    VALIDATED = "validated"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ExperimentStatus(str, Enum):
    """Experiment status enum."""
    DESIGNED = "designed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PaperStatus(str, Enum):
    """Paper status enum."""
    DRAFT = "draft"
    REVIEW = "review"
    REVISION = "revision"
    FINAL = "final"
    PUBLISHED = "published"


class Project(BaseModel):
    """Research project model."""

    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    research_question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50),
        default=ProjectStatus.ACTIVE.value,
        nullable=False
    )

    # Relationships
    hypotheses: Mapped[List["Hypothesis"]] = relationship(
        "Hypothesis",
        back_populates="project",
        cascade="all, delete-orphan"
    )
    papers: Mapped[List["Paper"]] = relationship(
        "Paper",
        back_populates="project",
        cascade="all, delete-orphan"
    )


class Hypothesis(BaseModel):
    """Research hypothesis model."""

    __tablename__ = "hypotheses"

    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    novelty_score: Mapped[float] = mapped_column(Float, nullable=False)
    testability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50),
        default=HypothesisStatus.GENERATED.value,
        nullable=False
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="hypotheses")
    experiments: Mapped[List["Experiment"]] = relationship(
        "Experiment",
        back_populates="hypothesis",
        cascade="all, delete-orphan"
    )


class Experiment(BaseModel):
    """Experiment model."""

    __tablename__ = "experiments"

    hypothesis_id: Mapped[UUID] = mapped_column(
        ForeignKey("hypotheses.id", ondelete="CASCADE"),
        nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    protocol: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50),
        default=ExperimentStatus.DESIGNED.value,
        nullable=False
    )

    # Design parameters
    sample_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    effect_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    significance_level: Mapped[float] = mapped_column(Float, default=0.05, nullable=False)

    # Results and analysis
    results_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    statistical_results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    visualization_urls: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array of URLs
    interpretation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    hypothesis: Mapped["Hypothesis"] = relationship(
        "Hypothesis",
        back_populates="experiments"
    )


class Paper(BaseModel):
    """Scientific paper model."""

    __tablename__ = "papers"

    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        default=PaperStatus.DRAFT.value,
        nullable=False
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="papers")
    sections: Mapped[List["PaperSection"]] = relationship(
        "PaperSection",
        back_populates="paper",
        cascade="all, delete-orphan",
        order_by="PaperSection.order"
    )


class PaperSection(BaseModel):
    """Paper section model for structured content."""

    __tablename__ = "paper_sections"

    paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Relationships
    paper: Mapped["Paper"] = relationship("Paper", back_populates="sections")

    # Indexes
    __table_args__ = (
        Index('idx_paper_sections_paper_id', 'paper_id'),
        Index('idx_paper_sections_name', 'name'),
    )

"""Project-related Pydantic schemas."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class ProjectBase(BaseModel):
    """Base project schema."""
    name: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    domain: str = Field(..., min_length=1, max_length=255)
    research_question: Optional[str] = None


class ProjectCreate(ProjectBase):
    """Schema for creating a project."""
    pass


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    domain: Optional[str] = Field(None, min_length=1, max_length=255)
    research_question: Optional[str] = None
    status: Optional[str] = None


class ProjectStats(BaseModel):
    """Project statistics."""
    hypotheses_count: int = 0
    experiments_count: int = 0
    papers_count: int = 0


class Project(ProjectBase):
    """Project response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    created_at: datetime
    updated_at: datetime
    stats: Optional[ProjectStats] = None


class ProjectList(BaseModel):
    """List of projects with pagination."""
    projects: List[Project]
    total: int
    page: int
    limit: int
    pages: int


class HypothesisBase(BaseModel):
    """Base hypothesis schema."""
    content: str
    rationale: Optional[str] = None
    novelty_score: float = Field(..., ge=0.0, le=1.0)
    testability_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class HypothesisCreate(HypothesisBase):
    """Schema for creating a hypothesis."""
    project_id: UUID


class Hypothesis(HypothesisBase):
    """Hypothesis response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime


class HypothesisGenerateRequest(BaseModel):
    """Request for generating hypotheses."""
    research_question: str
    literature_context: Optional[List[str]] = None
    num_hypotheses: int = Field(5, ge=1, le=10)
    creativity_level: float = Field(0.7, ge=0.0, le=1.0)


class ExperimentBase(BaseModel):
    """Base experiment schema."""
    title: str = Field(..., min_length=1, max_length=500)
    protocol: Optional[str] = None
    results_summary: Optional[str] = None


class ExperimentCreate(ExperimentBase):
    """Schema for creating an experiment."""
    hypothesis_id: UUID


class Experiment(ExperimentBase):
    """Experiment response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    hypothesis_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime


class PaperBase(BaseModel):
    """Base paper schema."""
    title: str = Field(..., min_length=1, max_length=500)
    abstract: Optional[str] = None
    content: Optional[str] = None


class PaperCreate(PaperBase):
    """Schema for creating a paper."""
    project_id: UUID


class Paper(PaperBase):
    """Paper response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    version: int
    status: str
    created_at: datetime
    updated_at: datetime

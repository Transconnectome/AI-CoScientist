"""Pydantic schemas for paper-related operations."""

from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field


# Request schemas

class PaperAnalyzeRequest(BaseModel):
    """Request schema for paper analysis."""

    include_sections: bool = Field(
        default=True,
        description="Include section-by-section analysis"
    )
    include_coherence: bool = Field(
        default=True,
        description="Include coherence check between sections"
    )


class PaperImproveRequest(BaseModel):
    """Request schema for paper improvement."""

    section_name: Optional[str] = Field(
        default=None,
        description="Specific section to improve (None = all sections)"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Specific feedback to address in improvements"
    )


class SectionUpdateRequest(BaseModel):
    """Request schema for updating a paper section."""

    content: str = Field(..., description="Updated section content")


# Response schemas

class PaperSectionSchema(BaseModel):
    """Schema for paper section."""

    id: UUID
    paper_id: UUID
    name: str
    content: str
    order: int
    version: int

    model_config = {"from_attributes": True}


class PaperAnalysisResponse(BaseModel):
    """Response schema for paper analysis."""

    quality_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Overall quality score (0-10)"
    )
    strengths: List[str] = Field(
        ...,
        description="List of paper strengths"
    )
    weaknesses: List[str] = Field(
        ...,
        description="List of paper weaknesses"
    )
    suggestions: List[dict] = Field(
        ...,
        description="Section-specific improvement suggestions"
    )
    coherence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Logical coherence score (0-10)"
    )
    clarity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Writing clarity score (0-10)"
    )


class PaperImprovementResponse(BaseModel):
    """Response schema for paper improvement."""

    section_name: Optional[str] = Field(
        default=None,
        description="Section that was improved"
    )
    improved_content: Optional[str] = Field(
        default=None,
        description="Improved section content"
    )
    changes_summary: str = Field(
        ...,
        description="Summary of changes made"
    )
    improvement_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Estimated improvement quality (0-10)"
    )


class MultipleSectionImprovementsResponse(BaseModel):
    """Response schema for multiple section improvements."""

    improvements: List[dict] = Field(
        ...,
        description="List of improvements for all sections"
    )
    total_sections: int = Field(
        ...,
        description="Total number of sections processed"
    )


class CoherenceCheckResponse(BaseModel):
    """Response schema for coherence analysis."""

    coherence_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Overall coherence score (0-10)"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of coherence issues identified"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving coherence"
    )


class GapAnalysisResponse(BaseModel):
    """Response schema for gap analysis."""

    gaps: List[dict] = Field(
        ...,
        description="Identified gaps in paper content"
    )
    total_gaps: int = Field(
        ...,
        description="Total number of gaps identified"
    )

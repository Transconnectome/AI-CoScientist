"""Pydantic schemas for Phase 4 improvement operations."""

from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


# ========== Request Schemas ==========


class ApplyImprovementRequest(BaseModel):
    """Request schema for applying AI improvement to a section."""

    section_name: str = Field(..., description="Name of section to improve")
    improved_content: str = Field(..., description="Improved section content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Improvement metadata (type, changes, summary)",
    )


class IterativeImprovementRequest(BaseModel):
    """Request schema for starting iterative improvement session."""

    target_score: float = Field(
        ..., ge=0.0, le=10.0, description="Target quality score to achieve (0-10)"
    )
    max_iterations: int = Field(
        default=5, ge=1, le=10, description="Maximum improvement iterations"
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Focus areas for improvement (e.g., ['clarity', 'methodology'])",
    )


class VersionRollbackRequest(BaseModel):
    """Request schema for rolling back to a previous version."""

    target_version: str = Field(..., description="Target version to rollback to (e.g., '1.2.0')")
    create_backup: bool = Field(
        default=True,
        description="Create backup of current version before rollback",
    )


# ========== Response Schemas ==========


class ApplyImprovementResponse(BaseModel):
    """Response schema for improvement application."""

    success: bool = Field(..., description="Whether improvement was applied")
    new_version: str = Field(..., description="New semantic version string")
    quality_improvement: float = Field(
        ..., description="Quality score improvement (can be negative)"
    )
    section_updated: str = Field(..., description="Name of updated section")
    improvement_id: str = Field(..., description="Improvement record ID")


class IterativeImprovementResponse(BaseModel):
    """Response schema for iterative improvement session."""

    session_id: str = Field(..., description="Iteration session ID")
    iterations_completed: int = Field(..., description="Number of iterations run")
    improvements_applied: int = Field(..., description="Total improvements applied")
    initial_score: float = Field(..., description="Starting quality score")
    final_score: float = Field(..., description="Final quality score")
    score_improvement: float = Field(..., description="Total score improvement")
    target_reached: bool = Field(..., description="Whether target score was reached")


class SmartSuggestionItem(BaseModel):
    """Single smart suggestion with RAG context."""

    section_name: str = Field(..., description="Section to improve")
    improved_content: str = Field(..., description="Suggested improved content")
    metadata: Dict[str, Any] = Field(..., description="Suggestion metadata")
    expected_improvement: float = Field(
        ..., description="Expected quality improvement"
    )
    similar_patterns_used: int = Field(
        default=0, description="Number of similar patterns from ChromaDB"
    )
    exemplars_referenced: int = Field(
        default=0, description="Number of exemplar papers referenced"
    )


class SmartSuggestionResponse(BaseModel):
    """Response schema for RAG-powered smart suggestions."""

    suggestions: List[SmartSuggestionItem] = Field(
        ..., description="List of smart suggestions"
    )
    total_suggestions: int = Field(..., description="Total number of suggestions")
    rag_enhanced: bool = Field(
        default=True, description="Whether RAG was used for enhancement"
    )


class SectionDiff(BaseModel):
    """Diff information for a single section."""

    section_name: str = Field(..., description="Section name")
    diff: str = Field(..., description="Unified diff output")
    changes_count: int = Field(..., description="Number of changed lines")


class VersionComparisonResponse(BaseModel):
    """Response schema for version comparison."""

    version_a: str = Field(..., description="First version (e.g., '1.0.0')")
    version_b: str = Field(..., description="Second version (e.g., '1.2.0')")
    quality_score_a: Optional[float] = Field(None, description="Quality score of version A")
    quality_score_b: Optional[float] = Field(None, description="Quality score of version B")
    quality_change: float = Field(..., description="Quality score change (B - A)")
    section_diffs: List[SectionDiff] = Field(..., description="Diffs for each section")
    summary_a: Optional[str] = Field(None, description="Change summary for version A")
    summary_b: Optional[str] = Field(None, description="Change summary for version B")


class VersionInfo(BaseModel):
    """Information about a paper version."""

    version_string: str = Field(..., description="Semantic version (e.g., '1.2.3')")
    version_type: str = Field(..., description="Version type (major, minor, patch)")
    change_summary: Optional[str] = Field(None, description="Summary of changes")
    quality_score: Optional[float] = Field(None, description="Quality score at this version")
    created_at: str = Field(..., description="When this version was created")


class VersionHistoryResponse(BaseModel):
    """Response schema for version history."""

    paper_id: str = Field(..., description="Paper UUID")
    current_version: str = Field(..., description="Current version string")
    versions: List[VersionInfo] = Field(..., description="List of all versions")
    total_versions: int = Field(..., description="Total number of versions")


class ImprovementStatistics(BaseModel):
    """Statistics for improvements."""

    total_improvements: int = Field(..., description="Total improvements applied")
    successful_improvements: int = Field(..., description="Improvements with positive impact")
    average_improvement: float = Field(..., description="Average quality improvement")
    most_improved_section: Optional[str] = Field(
        None, description="Section with most improvements"
    )
    improvement_types: Dict[str, int] = Field(
        default_factory=dict, description="Count of each improvement type"
    )


class QualityProgression(BaseModel):
    """Quality progression metrics."""

    starting_score: float = Field(..., description="Initial quality score")
    current_score: float = Field(..., description="Current quality score")
    total_improvement: float = Field(..., description="Total improvement")
    improvement_percentage: float = Field(..., description="Improvement percentage")
    iterations_count: int = Field(..., description="Number of iterations")


class SectionImprovementStat(BaseModel):
    """Section improvement statistics."""

    section_name: str = Field(..., description="Section name")
    improvement_count: int = Field(..., description="Number of improvements")
    total_impact: float = Field(..., description="Total quality impact")
    average_impact: float = Field(..., description="Average quality impact")


class ImprovementTypeEffectiveness(BaseModel):
    """Improvement type effectiveness metrics."""

    improvement_type: str = Field(..., description="Type of improvement")
    count: int = Field(..., description="Number of times used")
    average_impact: float = Field(..., description="Average quality impact")
    success_rate: float = Field(..., description="Success rate percentage")


class IterationEfficiency(BaseModel):
    """Iteration efficiency metrics."""

    total_iterations: int = Field(..., description="Total iterations run")
    average_improvements_per_iteration: float = Field(
        ..., description="Average improvements per iteration"
    )
    average_score_gain: float = Field(..., description="Average score gain per iteration")
    diminishing_returns_detected: bool = Field(
        ..., description="Whether diminishing returns detected"
    )


class VersionTimelineEntry(BaseModel):
    """Version timeline entry."""

    version: str = Field(..., description="Version string")
    quality_score: float = Field(..., description="Quality score")
    timestamp: str = Field(..., description="Creation timestamp")
    change_type: str = Field(..., description="Type of change")


class LearningStats(BaseModel):
    """Learning system statistics."""

    improvement_patterns_count: int = Field(..., description="Stored improvement patterns")
    successful_papers_count: int = Field(..., description="Exemplar papers count")
    user_interactions_count: int = Field(..., description="User interactions logged")
    rag_usage_count: int = Field(..., description="RAG queries made")
    average_relevance_score: float = Field(..., description="Average RAG relevance")
    exemplars_referenced_count: int = Field(..., description="Exemplars referenced")


class AnalyticsDashboardResponse(BaseModel):
    """Response schema for analytics dashboard."""

    paper_id: str = Field(..., description="Paper UUID")
    current_version: str = Field(..., description="Current version")
    quality_progression: QualityProgression = Field(..., description="Quality progression")
    section_improvements: List[SectionImprovementStat] = Field(
        ..., description="Section improvement statistics"
    )
    improvement_type_effectiveness: List[ImprovementTypeEffectiveness] = Field(
        ..., description="Improvement type effectiveness"
    )
    iteration_efficiency: IterationEfficiency = Field(..., description="Iteration efficiency")
    version_timeline: List[VersionTimelineEntry] = Field(..., description="Version timeline")
    learning_stats: LearningStats = Field(..., description="Learning system stats")
    recommendations: List[str] = Field(..., description="Intelligent recommendations")

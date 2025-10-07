"""Experiment-related Pydantic schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ExperimentBase(BaseModel):
    """Base experiment schema."""
    title: str
    protocol: Optional[str] = None
    sample_size: Optional[int] = None
    power: Optional[float] = None
    effect_size: Optional[float] = None
    significance_level: float = 0.05


class ExperimentCreate(ExperimentBase):
    """Experiment creation schema."""
    hypothesis_id: UUID


class Experiment(ExperimentBase):
    """Experiment response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    hypothesis_id: UUID
    status: str
    results_summary: Optional[str] = None
    statistical_results: Optional[str] = None
    visualization_urls: Optional[str] = None
    interpretation: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ExperimentDesignRequest(BaseModel):
    """Request for experiment design."""
    research_question: str
    hypothesis_content: str
    desired_power: float = Field(default=0.8, ge=0.5, le=0.99)
    significance_level: float = Field(default=0.05, ge=0.01, le=0.1)
    expected_effect_size: Optional[float] = Field(default=None, ge=0.1, le=2.0)
    constraints: Optional[Dict[str, Any]] = None
    experimental_approach: Optional[str] = None


class ExperimentDesignResponse(BaseModel):
    """Response from experiment design."""
    experiment_id: UUID
    title: str
    protocol: str
    sample_size: int
    power: float
    effect_size: float
    significance_level: float
    estimated_duration: Optional[str] = None
    resource_requirements: Optional[Dict[str, Any]] = None
    suggested_methods: List[str]
    potential_confounds: List[str]
    mitigation_strategies: List[str]


class StatisticalTest(BaseModel):
    """Statistical test configuration."""
    test_name: str
    parameters: Dict[str, Any]
    assumptions: List[str]


class DataAnalysisRequest(BaseModel):
    """Request for data analysis."""
    experiment_id: UUID
    data: Dict[str, Any]  # Flexible data structure
    analysis_types: List[str] = Field(
        default=["descriptive", "inferential", "effect_size"]
    )
    visualization_types: List[str] = Field(
        default=["distribution", "comparison", "correlation"]
    )
    custom_tests: Optional[List[StatisticalTest]] = None


class StatisticalResult(BaseModel):
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    confidence_interval: Optional[List[float]] = None
    effect_size: Optional[float] = None
    interpretation: str


class VisualizationResult(BaseModel):
    """Visualization result."""
    visualization_type: str
    url: str
    description: str
    format: str = "png"


class DataAnalysisResponse(BaseModel):
    """Response from data analysis."""
    experiment_id: UUID
    descriptive_statistics: Dict[str, Any]
    statistical_tests: List[StatisticalResult]
    visualizations: List[VisualizationResult]
    overall_interpretation: str
    confidence_level: float
    recommendations: List[str]
    limitations: List[str]


class PowerAnalysisRequest(BaseModel):
    """Request for statistical power analysis."""
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    power: Optional[float] = None
    significance_level: float = 0.05
    test_type: str = "two_sample_t"


class PowerAnalysisResponse(BaseModel):
    """Response from power analysis."""
    effect_size: float
    sample_size: int
    power: float
    significance_level: float
    recommendation: str

"""Pydantic schemas for Hybrid RAG API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# Request Schemas

class EvaluatePaperRequest(BaseModel):
    """Request schema for paper evaluation."""
    paper_text: str = Field(..., description="Paper content to evaluate")
    section: str = Field(default="full", description="Section being evaluated")
    use_ensemble: bool = Field(default=True, description="Use ensemble scoring")


class SummarizePaperRequest(BaseModel):
    """Request schema for paper summarization."""
    paper_text: str = Field(..., description="Paper content to summarize")
    max_length: int = Field(default=200, description="Maximum summary length in words")
    style: str = Field(default="concise", description="Summary style (concise, detailed, bullet_points)")


class ExtractInformationRequest(BaseModel):
    """Request schema for information extraction."""
    paper_text: str = Field(..., description="Paper content")
    fields: List[str] = Field(..., description="Fields to extract (e.g., methodology, results)")


class RetrieveSimilarPapersRequest(BaseModel):
    """Request schema for similar paper retrieval."""
    query: str = Field(..., description="Search query")
    top_k_retrieve: int = Field(default=10, description="Number of candidates to retrieve")
    top_k_rerank: int = Field(default=5, description="Number of final results after reranking")


# Response Schemas

class EvaluationProviderResult(BaseModel):
    """Individual provider evaluation result."""
    overall_quality: float
    novelty: float
    methodology: float
    clarity: float
    significance: float
    feedback: str
    provider: str
    confidence: float
    latency_ms: float


class EvaluatePaperResponse(BaseModel):
    """Response schema for paper evaluation."""
    overall_quality: float
    novelty: float
    methodology: float
    clarity: float
    significance: float
    feedback: str
    provider_scores: Dict[str, EvaluationProviderResult]
    ensemble_confidence: float
    total_latency_ms: float


class SummarizePaperResponse(BaseModel):
    """Response schema for paper summarization."""
    summary: str
    provider: str = "nemotron"
    latency_ms: Optional[float] = None


class ExtractInformationResponse(BaseModel):
    """Response schema for information extraction."""
    extracted_fields: Dict[str, str]
    provider: str = "nemotron"
    latency_ms: Optional[float] = None


class RelevantPaper(BaseModel):
    """A single relevant paper result."""
    paper_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    relevance_score: float
    metadata: Optional[Dict] = None


class RetrieveSimilarPapersResponse(BaseModel):
    """Response schema for similar paper retrieval."""
    papers: List[RelevantPaper]
    query: str
    total_retrieved: int
    total_reranked: int


class HybridRAGStatusResponse(BaseModel):
    """Response schema for hybrid RAG status."""
    hybrid_mode: bool
    enabled_providers: List[str]
    ensemble_weights: Dict[str, float]
    nemotron_services: Dict[str, str]
    configuration: Dict[str, any]

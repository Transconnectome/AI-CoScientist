"""
Research API endpoints with GPT Researcher integration.

Provides systematic literature review and hypothesis validation endpoints.
"""

from typing import List, Optional
from uuid import UUID
import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from src.core.database import get_db
from src.services.external.gpt_researcher_service import GPTResearcherService
from src.services.hypothesis.generator import HypothesisGenerator
from src.services.llm.service import LLMService
from src.services.knowledge_base.search import KnowledgeBaseSearch


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


# Request/Response Models
class LiteratureReviewRequest(BaseModel):
    """Request for systematic literature review."""
    research_question: str = Field(..., description="Research question to investigate")
    domain: Optional[str] = Field(None, description="Research domain (e.g., neuroscience)")
    depth: str = Field("medium", description="Search depth: quick, medium, deep")


class LiteratureReviewResponse(BaseModel):
    """Response from systematic literature review."""
    success: bool
    research_question: str
    domain: Optional[str]
    report: str
    sources: List[str]
    num_sources: int
    timestamp: str


class QuestionDecompositionRequest(BaseModel):
    """Request for research question decomposition."""
    question: str
    num_subquestions: int = Field(5, ge=2, le=10)


class QuestionDecompositionResponse(BaseModel):
    """Response with decomposed sub-questions."""
    original_question: str
    subquestions: List[str]
    num_subquestions: int


class MultiHopSearchRequest(BaseModel):
    """Request for multi-hop literature search."""
    initial_query: str
    max_hops: int = Field(3, ge=1, le=5)
    entities_per_hop: int = Field(3, ge=1, le=5)


class MultiHopSearchResponse(BaseModel):
    """Response from multi-hop search."""
    initial_query: str
    total_hops: int
    total_sources: int
    hop_history: List[dict]
    all_sources: List[str]


class HypothesisValidationRequest(BaseModel):
    """Request for hypothesis validation."""
    hypothesis: str
    domain: str


class HypothesisValidationResponse(BaseModel):
    """Response from hypothesis validation."""
    hypothesis: str
    validation_report: str
    supporting_sources: List[str]
    novelty_score: float
    is_novel: bool
    num_related_papers: int


class ResearchTrendsRequest(BaseModel):
    """Request for research trends analysis."""
    domain: str
    timeframe: str = Field("recent", description="recent (1-2 years) or current (current year)")


class ResearchTrendsResponse(BaseModel):
    """Response from trends analysis."""
    domain: str
    timeframe: str
    trends_report: str
    sources: List[str]
    num_sources: int


# Dependencies
def get_gpt_researcher() -> GPTResearcherService:
    """Get GPT Researcher service."""
    try:
        return GPTResearcherService()
    except Exception as e:
        logger.error(f"Failed to initialize GPT Researcher: {e}")
        raise HTTPException(
            status_code=503,
            detail="GPT Researcher service unavailable. Check OPENAI_API_KEY configuration."
        )


# Endpoints
@router.post("/literature-review", response_model=LiteratureReviewResponse)
async def systematic_literature_review(
    request: LiteratureReviewRequest,
    gpt_researcher: GPTResearcherService = Depends(get_gpt_researcher)
):
    """
    Perform systematic literature review with query decomposition.

    This endpoint uses GPT Researcher to:
    - Decompose research questions systematically
    - Search multiple sources comprehensively
    - Generate structured literature review report

    **Example Request:**
    ```json
    {
      "research_question": "What are the latest methods for fMRI analysis using deep learning?",
      "domain": "neuroscience",
      "depth": "medium"
    }
    ```

    **Returns:**
    - Comprehensive literature review report
    - List of source URLs
    - Number of sources reviewed
    """
    logger.info(f"Literature review request: {request.research_question}")

    try:
        result = await gpt_researcher.systematic_literature_review(
            research_question=request.research_question,
            domain=request.domain,
            depth=request.depth
        )

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Literature review failed: {result.get('error', 'Unknown error')}"
            )

        return LiteratureReviewResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Literature review error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decompose-question", response_model=QuestionDecompositionResponse)
async def decompose_research_question(
    request: QuestionDecompositionRequest,
    gpt_researcher: GPTResearcherService = Depends(get_gpt_researcher)
):
    """
    Decompose complex research question into focused sub-questions.

    Useful for systematic literature review and hypothesis generation.

    **Example Request:**
    ```json
    {
      "question": "How can AI improve psychiatric diagnosis?",
      "num_subquestions": 5
    }
    ```

    **Returns:**
    - List of focused, answerable sub-questions
    - Coverage of current knowledge, methods, limitations, and developments
    """
    logger.info(f"Question decomposition: {request.question}")

    try:
        subquestions = await gpt_researcher.decompose_research_question(
            question=request.question,
            num_subquestions=request.num_subquestions
        )

        return QuestionDecompositionResponse(
            original_question=request.question,
            subquestions=subquestions,
            num_subquestions=len(subquestions)
        )

    except Exception as e:
        logger.error(f"Question decomposition error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-hop-search", response_model=MultiHopSearchResponse)
async def multi_hop_literature_search(
    request: MultiHopSearchRequest,
    background_tasks: BackgroundTasks,
    gpt_researcher: GPTResearcherService = Depends(get_gpt_researcher)
):
    """
    Perform multi-hop literature search following entities and concepts.

    This explores literature in depth by:
    1. Initial broad search
    2. Following promising entities/concepts
    3. Deep dive into relevant areas
    4. Synthesis of all findings

    **Example Request:**
    ```json
    {
      "initial_query": "Brain-computer interfaces for motor rehabilitation",
      "max_hops": 3,
      "entities_per_hop": 3
    }
    ```

    **Returns:**
    - Complete hop history with queries and sources
    - All unique sources discovered
    - Total coverage metrics
    """
    logger.info(f"Multi-hop search: {request.initial_query}")

    try:
        result = await gpt_researcher.multi_hop_literature_search(
            initial_query=request.initial_query,
            max_hops=request.max_hops,
            entities_per_hop=request.entities_per_hop
        )

        return MultiHopSearchResponse(**result)

    except Exception as e:
        logger.error(f"Multi-hop search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-hypothesis", response_model=HypothesisValidationResponse)
async def validate_hypothesis_against_literature(
    request: HypothesisValidationRequest,
    gpt_researcher: GPTResearcherService = Depends(get_gpt_researcher)
):
    """
    Validate hypothesis against existing literature.

    Analyzes:
    - Similar existing work
    - Supporting evidence
    - Contradicting evidence
    - Novelty assessment (paradigm shift vs incremental)
    - Testability and feasibility

    **Example Request:**
    ```json
    {
      "hypothesis": "Multi-frequency brain stimulation enhances memory consolidation through cross-frequency coupling",
      "domain": "neuroscience"
    }
    ```

    **Returns:**
    - Validation report with evidence analysis
    - Novelty score (0-1)
    - Related papers and similarity assessment
    """
    logger.info(f"Hypothesis validation: {request.hypothesis}")

    try:
        result = await gpt_researcher.validate_hypothesis_against_literature(
            hypothesis=request.hypothesis,
            domain=request.domain
        )

        return HypothesisValidationResponse(**result)

    except Exception as e:
        logger.error(f"Hypothesis validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research-trends", response_model=ResearchTrendsResponse)
async def get_research_trends(
    request: ResearchTrendsRequest,
    gpt_researcher: GPTResearcherService = Depends(get_gpt_researcher)
):
    """
    Identify research trends and hot topics in a domain.

    Analyzes:
    - Emerging methodologies
    - Novel applications
    - Recent breakthroughs
    - Open challenges
    - Future directions

    **Example Request:**
    ```json
    {
      "domain": "computational neuroscience",
      "timeframe": "recent"
    }
    ```

    **Returns:**
    - Comprehensive trends report
    - Supporting sources
    - Temporal context
    """
    logger.info(f"Research trends: {request.domain}")

    try:
        result = await gpt_researcher.get_research_trends(
            domain=request.domain,
            timeframe=request.timeframe
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return ResearchTrendsResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trends analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def research_health_check():
    """Check if GPT Researcher service is available."""
    try:
        service = GPTResearcherService()
        return {
            "status": "healthy",
            "service": "GPT Researcher",
            "available": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "GPT Researcher",
            "available": False,
            "error": str(e)
        }

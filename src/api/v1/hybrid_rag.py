"""Hybrid RAG API endpoints for intelligent paper evaluation."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from src.schemas.hybrid_rag import (
    EvaluatePaperRequest,
    EvaluatePaperResponse,
    SummarizePaperRequest,
    SummarizePaperResponse,
    ExtractInformationRequest,
    ExtractInformationResponse,
    RetrieveSimilarPapersRequest,
    RetrieveSimilarPapersResponse,
    HybridRAGStatusResponse,
    EvaluationProviderResult,
    RelevantPaper,
)
from src.services.hybrid_rag_service import (
    HybridRAGService,
    create_hybrid_rag_service,
    ModelProvider,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instance (initialized on startup)
_hybrid_rag_service: HybridRAGService = None


async def get_hybrid_rag_service() -> HybridRAGService:
    """Get or create hybrid RAG service instance."""
    global _hybrid_rag_service
    if _hybrid_rag_service is None:
        _hybrid_rag_service = await create_hybrid_rag_service()
    return _hybrid_rag_service


@router.post("/evaluate", response_model=EvaluatePaperResponse)
async def evaluate_paper(request: EvaluatePaperRequest):
    """
    Evaluate paper quality using hybrid ensemble of GPT-4, Claude, and Nemotron.

    **Task Routing**:
    - GPT-4: High-quality evaluation with proven track record
    - Claude: Alternative high-quality evaluation perspective
    - Nemotron: Optional third opinion for ensemble diversity

    **Ensemble Scoring**: Weighted combination of provider scores based on configuration

    Args:
        request: Paper evaluation request

    Returns:
        Ensemble evaluation with provider-specific scores and combined feedback

    Raises:
        HTTPException: 500 if all evaluation providers fail
    """
    service = await get_hybrid_rag_service()

    try:
        result = await service.evaluate_paper(
            paper_text=request.paper_text,
            section=request.section,
            use_ensemble=request.use_ensemble,
        )

        # Convert to response schema
        provider_scores_dict = {}
        for provider_key, eval_result in result.provider_scores.items():
            provider_scores_dict[provider_key] = EvaluationProviderResult(
                overall_quality=eval_result.overall_quality,
                novelty=eval_result.novelty,
                methodology=eval_result.methodology,
                clarity=eval_result.clarity,
                significance=eval_result.significance,
                feedback=eval_result.feedback,
                provider=eval_result.provider.value,
                confidence=eval_result.confidence,
                latency_ms=eval_result.latency_ms,
            )

        return EvaluatePaperResponse(
            overall_quality=result.overall_quality,
            novelty=result.novelty,
            methodology=result.methodology,
            clarity=result.clarity,
            significance=result.significance,
            feedback=result.feedback,
            provider_scores=provider_scores_dict,
            ensemble_confidence=result.ensemble_confidence,
            total_latency_ms=result.total_latency_ms,
        )

    except Exception as e:
        logger.error(f"Paper evaluation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.post("/summarize", response_model=SummarizePaperResponse)
async def summarize_paper(request: SummarizePaperRequest):
    """
    Summarize paper using Nemotron (fast, cost-effective).

    **Task Routing**: Nemotron for summarization (optimized for speed and cost)

    Args:
        request: Paper summarization request

    Returns:
        Summary text with metadata

    Raises:
        HTTPException: 500 if summarization fails
    """
    service = await get_hybrid_rag_service()

    try:
        import time
        start_time = time.time()

        summary = await service.summarize_paper(
            paper_text=request.paper_text,
            max_length=request.max_length,
            style=request.style,
        )

        latency_ms = (time.time() - start_time) * 1000

        return SummarizePaperResponse(
            summary=summary,
            provider="nemotron",
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Paper summarization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/extract", response_model=ExtractInformationResponse)
async def extract_information(request: ExtractInformationRequest):
    """
    Extract specific information fields from paper using Nemotron.

    **Task Routing**: Nemotron for extraction (optimized for structured extraction)

    Args:
        request: Information extraction request

    Returns:
        Extracted fields as key-value pairs

    Raises:
        HTTPException: 500 if extraction fails
    """
    service = await get_hybrid_rag_service()

    try:
        import time
        start_time = time.time()

        extracted_fields = await service.extract_information(
            paper_text=request.paper_text,
            fields=request.fields,
        )

        latency_ms = (time.time() - start_time) * 1000

        return ExtractInformationResponse(
            extracted_fields=extracted_fields,
            provider="nemotron",
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Information extraction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@router.post("/retrieve", response_model=RetrieveSimilarPapersResponse)
async def retrieve_similar_papers(request: RetrieveSimilarPapersRequest):
    """
    Retrieve and rerank similar papers using NeMo Retriever.

    **Pipeline**:
    1. Embed query using Llama 3.2 EmbedQA 1B
    2. Retrieve top-k candidates from vector store
    3. Rerank using Llama 3.2 RerankQA 1B

    Args:
        request: Similar paper retrieval request

    Returns:
        Reranked list of relevant papers with scores

    Raises:
        HTTPException: 500 if retrieval fails
    """
    service = await get_hybrid_rag_service()

    try:
        reranked_docs, scores = await service.retrieve_similar_papers(
            query=request.query,
            top_k_retrieve=request.top_k_retrieve,
            top_k_rerank=request.top_k_rerank,
        )

        # Convert to response schema
        papers = []
        for doc, score in zip(reranked_docs, scores):
            papers.append(
                RelevantPaper(
                    paper_id=doc.get("metadata", {}).get("paper_id"),
                    title=doc.get("metadata", {}).get("title"),
                    content=doc.get("page_content", ""),
                    relevance_score=score,
                    metadata=doc.get("metadata", {}),
                )
            )

        return RetrieveSimilarPapersResponse(
            papers=papers,
            query=request.query,
            total_retrieved=request.top_k_retrieve,
            total_reranked=len(papers),
        )

    except Exception as e:
        logger.error(f"Similar paper retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )


@router.get("/status", response_model=HybridRAGStatusResponse)
async def get_hybrid_rag_status():
    """
    Get hybrid RAG service status and configuration.

    Returns:
        Service status, enabled providers, and configuration details

    Raises:
        HTTPException: 500 if status check fails
    """
    service = await get_hybrid_rag_service()

    try:
        enabled_providers = []
        if service.use_gpt4_for_evaluation:
            enabled_providers.append("gpt4")
        if service.use_claude_for_evaluation:
            enabled_providers.append("claude")
        if service.use_nemotron_for_summarization or service.use_nemotron_for_extraction:
            enabled_providers.append("nemotron")

        ensemble_weights = {
            "gpt4": service.ensemble_weight_gpt4,
            "claude": service.ensemble_weight_claude,
            "nemotron": service.ensemble_weight_nemotron,
        }

        nemotron_services = {
            "llm": service.nemotron_llm.base_url,
            "embedder": service.nemo_embedder.base_url,
            "reranker": service.nemo_reranker.base_url,
        }

        configuration = {
            "use_gpt4_for_evaluation": service.use_gpt4_for_evaluation,
            "use_claude_for_evaluation": service.use_claude_for_evaluation,
            "use_nemotron_for_summarization": service.use_nemotron_for_summarization,
            "use_nemotron_for_extraction": service.use_nemotron_for_extraction,
            "nemotron_confidence_threshold": service.nemotron_confidence_threshold,
        }

        return HybridRAGStatusResponse(
            hybrid_mode=service.hybrid_mode,
            enabled_providers=enabled_providers,
            ensemble_weights=ensemble_weights,
            nemotron_services=nemotron_services,
            configuration=configuration,
        )

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )


@router.post("/health")
async def hybrid_rag_health_check():
    """
    Health check for all hybrid RAG services.

    Checks connectivity to:
    - Nemotron LLM
    - NeMo Embedder
    - NeMo Reranker
    - OpenAI API (if enabled)
    - Anthropic API (if enabled)

    Returns:
        Health status for each service

    Raises:
        HTTPException: 503 if critical services are unavailable
    """
    service = await get_hybrid_rag_service()

    health_status = {
        "overall": "healthy",
        "services": {},
    }

    # Check Nemotron LLM
    try:
        llm_healthy = await service.nemotron_llm.health_check()
        health_status["services"]["nemotron_llm"] = "healthy" if llm_healthy else "unhealthy"
    except Exception as e:
        health_status["services"]["nemotron_llm"] = f"error: {str(e)}"
        health_status["overall"] = "degraded"

    # Check NeMo Embedder
    try:
        embedder_response = await service.nemo_embedder.client.get(
            f"{service.nemo_embedder.base_url}/health"
        )
        health_status["services"]["nemo_embedder"] = "healthy" if embedder_response.status_code == 200 else "unhealthy"
    except Exception as e:
        health_status["services"]["nemo_embedder"] = f"error: {str(e)}"
        health_status["overall"] = "degraded"

    # Check NeMo Reranker
    try:
        reranker_response = await service.nemo_reranker.client.get(
            f"{service.nemo_reranker.base_url}/health"
        )
        health_status["services"]["nemo_reranker"] = "healthy" if reranker_response.status_code == 200 else "unhealthy"
    except Exception as e:
        health_status["services"]["nemo_reranker"] = f"error: {str(e)}"
        health_status["overall"] = "degraded"

    # Check OpenAI (if enabled)
    if service.use_gpt4_for_evaluation:
        try:
            # Simple API check
            models = await service.openai_client.models.list()
            health_status["services"]["openai"] = "healthy"
        except Exception as e:
            health_status["services"]["openai"] = f"error: {str(e)}"
            health_status["overall"] = "degraded"

    # Check Anthropic (if enabled)
    if service.use_claude_for_evaluation:
        # Anthropic doesn't have a simple health endpoint
        # We'll just mark as configured
        health_status["services"]["anthropic"] = "configured"

    # Determine overall status
    unhealthy_count = sum(
        1 for status in health_status["services"].values()
        if status not in ["healthy", "configured"]
    )

    if unhealthy_count >= 3:
        health_status["overall"] = "critical"
        raise HTTPException(
            status_code=503,
            detail=health_status
        )
    elif unhealthy_count > 0:
        health_status["overall"] = "degraded"

    return health_status

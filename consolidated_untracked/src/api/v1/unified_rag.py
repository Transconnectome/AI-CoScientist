"""
Unified RAG API Endpoints

Implementation for: Unified RAG API endpoints
Created: 2025-12-05

Acceptance Criteria:
- RESTful endpoints for unified search
- Strategy selection and override API
- Performance metrics and health check endpoints
- OpenAPI documentation complete

This module provides REST API endpoints for the unified RAG system with
comprehensive search capabilities, strategy management, and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
import asyncio

# Import unified RAG components
from src.services.rag.unified_rag_orchestrator import (
    UnifiedRAGOrchestrator, QueryContext, RAGResponse, RAGStrategy,
    QueryComplexity, QueryDomain, get_orchestrator
)
from src.services.rag.advanced_query_classifier import (
    MLQueryClassifier, QueryIntent, get_query_classifier
)
from src.monitoring.rag_metrics import get_metrics_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/rag", tags=["Unified RAG"])

# Pydantic models for API
class SearchRequest(BaseModel):
    """Request model for unified search"""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    strategy_override: Optional[RAGStrategy] = Field(None, description="Force specific RAG strategy")
    enable_fallback: bool = Field(True, description="Enable fallback to other strategies on failure")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for search")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional query metadata")

class ClassificationRequest(BaseModel):
    """Request model for query classification"""
    query: str = Field(..., description="Query to classify", min_length=1, max_length=1000)

class BatchSearchRequest(BaseModel):
    """Request model for batch search"""
    queries: List[str] = Field(..., description="List of queries to search", min_items=1, max_items=20)
    strategy_override: Optional[RAGStrategy] = Field(None, description="Force specific RAG strategy for all queries")
    max_concurrent: int = Field(5, description="Maximum concurrent searches", ge=1, le=20)
    enable_fallback: bool = Field(True, description="Enable fallback to other strategies on failure")

class SearchResponse(BaseModel):
    """Response model for search results"""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    confidence: float = Field(..., description="Answer confidence score", ge=0.0, le=1.0)
    strategy_used: RAGStrategy = Field(..., description="RAG strategy that generated the answer")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    classification: Optional[Dict[str, Any]] = Field(None, description="Query classification results")
    processing_time: float = Field(..., description="Total processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")

class ClassificationResponse(BaseModel):
    """Response model for query classification"""
    complexity: QueryComplexity = Field(..., description="Query complexity level")
    domain: QueryDomain = Field(..., description="Query domain")
    intent: QueryIntent = Field(..., description="Query intent")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each prediction")
    overall_confidence: float = Field(..., description="Overall classification confidence")
    features: Dict[str, Any] = Field(..., description="Extracted features")

class StrategyHealthResponse(BaseModel):
    """Response model for strategy health status"""
    strategies: Dict[str, Dict[str, Any]] = Field(..., description="Health status of each strategy")
    total_requests: int = Field(..., description="Total requests processed")
    active_strategies: int = Field(..., description="Number of active strategies")
    system_health: str = Field(..., description="Overall system health status")

class PerformanceResponse(BaseModel):
    """Response model for performance metrics"""
    strategies: Dict[str, Dict[str, Any]] = Field(..., description="Strategy performance data")
    metrics_summary: Dict[str, Any] = Field(..., description="Overall metrics summary")
    timestamp: datetime = Field(..., description="Response timestamp")

# Dependency injection
async def get_rag_orchestrator() -> UnifiedRAGOrchestrator:
    """Dependency to get RAG orchestrator"""
    return get_orchestrator()

async def get_classifier() -> MLQueryClassifier:
    """Dependency to get query classifier"""
    return get_query_classifier()

# API Endpoints

@router.post("/search", response_model=SearchResponse, summary="Execute unified RAG search")
async def search(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    orchestrator: UnifiedRAGOrchestrator = Depends(get_rag_orchestrator),
    classifier: MLQueryClassifier = Depends(get_classifier)
):
    """
    Execute a unified RAG search with automatic strategy selection.

    This endpoint provides intelligent query routing across all available RAG strategies
    with automatic fallback and comprehensive performance monitoring.

    **Features:**
    - Automatic query classification (complexity, domain, intent)
    - Intelligent strategy selection based on query characteristics
    - Fallback mechanisms for robustness
    - Real-time performance metrics
    - Strategy override capability

    **Query Classification:**
    - **Complexity**: simple, medium, complex
    - **Domain**: neuroscience, quantum_ml, developmental_disorders, general
    - **Intent**: factual, comparative, synthesis, procedural, causal
    """
    start_time = datetime.now()

    try:
        # Classify query
        classification_result = await classifier.classify(request.query)

        # Create query context
        query_context = QueryContext(
            query=request.query,
            complexity=classification_result.complexity,
            domain=classification_result.domain,
            intent=classification_result.intent.value,
            confidence=classification_result.overall_confidence,
            metadata=request.metadata or {},
            user_preferences=request.user_preferences
        )

        # Execute search
        rag_response = await orchestrator.search(
            query_context=query_context,
            strategy_override=request.strategy_override,
            enable_fallback=request.enable_fallback
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Prepare response
        response = SearchResponse(
            answer=rag_response.answer,
            sources=rag_response.sources,
            confidence=rag_response.confidence,
            strategy_used=rag_response.strategy_used,
            performance_metrics=rag_response.performance_metrics.__dict__ if rag_response.performance_metrics else None,
            classification={
                "complexity": classification_result.complexity.value,
                "domain": classification_result.domain.value,
                "intent": classification_result.intent.value,
                "confidence_scores": classification_result.confidence_scores,
                "overall_confidence": classification_result.overall_confidence
            },
            processing_time=processing_time,
            metadata=rag_response.metadata or {}
        )

        # Log successful request
        background_tasks.add_task(
            log_search_request,
            request.query,
            rag_response.strategy_used.value,
            processing_time,
            True
        )

        return response

    except Exception as e:
        # Log error
        error_time = (datetime.now() - start_time).total_seconds()
        background_tasks.add_task(
            log_search_request,
            request.query,
            "error",
            error_time,
            False,
            str(e)
        )

        logger.error(f"Search failed for query '{request.query}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.post("/classify", response_model=ClassificationResponse, summary="Classify query characteristics")
async def classify_query(
    request: ClassificationRequest,
    classifier: MLQueryClassifier = Depends(get_classifier)
):
    """
    Classify query characteristics for optimal strategy selection.

    This endpoint analyzes a query to determine its complexity, domain,
    and intent, which can be used to select the most appropriate RAG strategy.

    **Classification Output:**
    - **Complexity**: Determines computational requirements
      - simple: Basic fact retrieval
      - medium: Moderate analysis and comparison
      - complex: Deep synthesis and reasoning

    - **Domain**: Identifies subject area expertise
      - neuroscience: Brain and neural systems
      - quantum_ml: Quantum machine learning
      - developmental_disorders: Autism and developmental conditions
      - general: General scientific knowledge

    - **Intent**: Understands query purpose
      - factual: Simple fact retrieval
      - comparative: Comparison between concepts
      - synthesis: Complex analysis and integration
      - procedural: How-to questions
      - causal: Cause-effect relationships
    """
    try:
        result = await classifier.classify(request.query)

        return ClassificationResponse(
            complexity=result.complexity,
            domain=result.domain,
            intent=result.intent,
            confidence_scores=result.confidence_scores,
            overall_confidence=result.overall_confidence,
            features=result.features
        )

    except Exception as e:
        logger.error(f"Classification failed for query '{request.query}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )

@router.post("/search/batch", response_model=List[Union[SearchResponse, Dict[str, str]]], summary="Execute batch RAG search")
async def batch_search(
    request: BatchSearchRequest,
    background_tasks: BackgroundTasks,
    orchestrator: UnifiedRAGOrchestrator = Depends(get_rag_orchestrator),
    classifier: MLQueryClassifier = Depends(get_classifier)
):
    """
    Execute multiple RAG searches in parallel for improved efficiency.

    This endpoint processes multiple queries concurrently with intelligent
    load balancing and error handling. Failed queries return error objects
    instead of failing the entire batch.

    **Features:**
    - Parallel processing with configurable concurrency
    - Independent error handling per query
    - Automatic query classification for each query
    - Strategy override for all queries
    - Comprehensive performance monitoring
    """
    start_time = datetime.now()

    try:
        # Classify all queries first
        query_contexts = []
        for query in request.queries:
            classification_result = await classifier.classify(query)

            query_context = QueryContext(
                query=query,
                complexity=classification_result.complexity,
                domain=classification_result.domain,
                intent=classification_result.intent.value,
                confidence=classification_result.overall_confidence,
                metadata={"batch": True}
            )
            query_contexts.append(query_context)

        # Execute batch search
        responses = await orchestrator.search_parallel(
            query_contexts, max_concurrent=request.max_concurrent
        )

        # Process responses
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                # Handle individual query failure
                error_response = {
                    "error": str(response),
                    "query": request.queries[i],
                    "status": "failed"
                }
                processed_responses.append(error_response)
            else:
                # Convert successful response
                search_response = SearchResponse(
                    answer=response.answer,
                    sources=response.sources,
                    confidence=response.confidence,
                    strategy_used=response.strategy_used,
                    performance_metrics=response.performance_metrics.__dict__ if response.performance_metrics else None,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    metadata=response.metadata or {}
                )
                processed_responses.append(search_response)

        # Log batch request
        total_time = (datetime.now() - start_time).total_seconds()
        success_count = sum(1 for r in processed_responses if not isinstance(r, dict) or "error" not in r)

        background_tasks.add_task(
            log_batch_request,
            len(request.queries),
            success_count,
            total_time
        )

        return processed_responses

    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch search failed: {str(e)}"
        )

@router.get("/health", response_model=StrategyHealthResponse, summary="Get system health status")
async def get_health(
    orchestrator: UnifiedRAGOrchestrator = Depends(get_rag_orchestrator)
):
    """
    Get comprehensive health status of the RAG system.

    This endpoint provides detailed information about the availability
    and performance of all RAG strategies, system load, and overall health.

    **Health Metrics:**
    - Strategy availability and configuration
    - Request counts and performance scores
    - System load and active strategy count
    - Overall system health assessment
    """
    try:
        health_data = orchestrator.get_strategy_health()
        performance_summary = orchestrator.get_performance_summary()

        # Determine overall system health
        active_count = performance_summary["active_strategies"]
        total_strategies = len(health_data)

        if active_count == 0:
            system_health = "critical"
        elif active_count < total_strategies * 0.5:
            system_health = "degraded"
        elif active_count < total_strategies * 0.8:
            system_health = "warning"
        else:
            system_health = "healthy"

        return StrategyHealthResponse(
            strategies=health_data,
            total_requests=performance_summary["total_requests"],
            active_strategies=active_count,
            system_health=system_health
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/metrics", response_model=PerformanceResponse, summary="Get performance metrics")
async def get_metrics(
    strategy: Optional[RAGStrategy] = Query(None, description="Filter metrics by strategy"),
    hours: int = Query(24, description="Time window in hours", ge=1, le=168),
    orchestrator: UnifiedRAGOrchestrator = Depends(get_rag_orchestrator)
):
    """
    Get detailed performance metrics for the RAG system.

    This endpoint provides comprehensive performance data including
    latency statistics, quality scores, throughput metrics, and
    strategy-specific performance analysis.

    **Metrics Include:**
    - Request latency and throughput
    - Quality score distributions
    - Strategy performance comparisons
    - Error rates and success rates
    - Resource utilization statistics
    """
    try:
        metrics_manager = get_metrics_manager()

        if strategy:
            # Get strategy-specific metrics
            strategy_performance = metrics_manager.get_strategy_performance(strategy.value, hours)
            metrics_data = {
                "strategies": {strategy.value: strategy_performance},
                "metrics_summary": {"filtered_by_strategy": strategy.value}
            }
        else:
            # Get all metrics
            metrics_data = orchestrator.get_performance_summary()

        return PerformanceResponse(
            strategies=metrics_data["strategies"],
            metrics_summary=metrics_data.get("metrics_summary", {}),
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics retrieval failed: {str(e)}"
        )

@router.get("/strategies", summary="List available RAG strategies")
async def list_strategies(
    orchestrator: UnifiedRAGOrchestrator = Depends(get_rag_orchestrator)
):
    """
    List all available RAG strategies with their configurations.

    This endpoint provides information about all registered RAG strategies,
    their availability status, configuration parameters, and capabilities.

    **Strategy Information:**
    - Strategy name and description
    - Availability status
    - Supported domains and complexity levels
    - Configuration parameters
    - Performance characteristics
    """
    try:
        strategies_info = {}

        for strategy in RAGStrategy:
            config = orchestrator.config.get_config(strategy)
            is_available = strategy in orchestrator.strategies

            strategies_info[strategy.value] = {
                "available": is_available,
                "enabled": config.get("enabled", False),
                "priority": config.get("priority", 999),
                "domains": [d.value for d in config.get("domains", [])],
                "complexity_range": [c.value for c in config.get("complexity_range", [])],
                "max_concurrent": config.get("max_concurrent", 1),
                "description": _get_strategy_description(strategy)
            }

        return {
            "strategies": strategies_info,
            "total_available": len(orchestrator.strategies),
            "total_configured": len([s for s in strategies_info.values() if s["enabled"]])
        }

    except Exception as e:
        logger.error(f"Strategy listing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Strategy listing failed: {str(e)}"
        )

@router.post("/warmup", summary="Warm up RAG strategies")
async def warmup_strategies(
    background_tasks: BackgroundTasks,
    orchestrator: UnifiedRAGOrchestrator = Depends(get_rag_orchestrator)
):
    """
    Warm up all RAG strategies to improve initial response times.

    This endpoint triggers warmup procedures for all strategies,
    which can improve performance for subsequent requests by
    pre-loading models and initializing connections.
    """
    try:
        # Run warmup in background
        background_tasks.add_task(orchestrator.warmup)

        return {
            "status": "warmup_started",
            "message": "RAG strategies warmup initiated in background",
            "strategies_count": len(orchestrator.strategies)
        }

    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Warmup failed: {str(e)}"
        )

# Helper functions

async def log_search_request(
    query: str,
    strategy: str,
    processing_time: float,
    success: bool,
    error: Optional[str] = None
):
    """Log search request for analytics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query_length": len(query),
        "strategy": strategy,
        "processing_time": processing_time,
        "success": success,
        "error": error
    }

    # In production, this would be sent to a proper logging/analytics system
    logger.info(f"Search request: {log_entry}")

async def log_batch_request(
    total_queries: int,
    successful_queries: int,
    total_time: float
):
    """Log batch request for analytics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "success_rate": successful_queries / total_queries,
        "total_time": total_time,
        "avg_time_per_query": total_time / total_queries
    }

    logger.info(f"Batch request: {log_entry}")

def _get_strategy_description(strategy: RAGStrategy) -> str:
    """Get human-readable description of strategy"""
    descriptions = {
        RAGStrategy.HYBRID: "Balanced approach combining multiple retrieval methods for optimal performance",
        RAGStrategy.ENHANCED_DD_RAPTOR: "Specialized RAPTOR implementation for developmental disorder research",
        RAGStrategy.GRAPH_RAG: "Knowledge graph-based retrieval for complex relationship queries",
        RAGStrategy.GOLDEN_REFERENCE: "High-quality reference paper matching for authoritative answers",
        RAGStrategy.SIMPLE_RAG: "Fast, lightweight retrieval for simple factual queries",
        RAGStrategy.MULTIMODAL_RAG: "Advanced multimodal retrieval supporting text, images, and structured data"
    }

    return descriptions.get(strategy, "Advanced RAG strategy")

# Add error handlers
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Export router
__all__ = ["router"]
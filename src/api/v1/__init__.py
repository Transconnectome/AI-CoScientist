"""API v1 package."""

from fastapi import APIRouter

from src.api.v1 import projects, health, literature, hypotheses, experiments, papers, improvements, monitoring, multi_agent, research, rag_evaluation, hybrid_rag

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(literature.router, prefix="/literature", tags=["literature"])
api_router.include_router(hypotheses.router, tags=["hypotheses"])
api_router.include_router(experiments.router, tags=["experiments"])
api_router.include_router(papers.router, prefix="/papers", tags=["papers"])
api_router.include_router(improvements.router, prefix="/improvements", tags=["improvements"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(multi_agent.router, prefix="/multi-agent", tags=["multi-agent"])
api_router.include_router(research.router, tags=["research"])  # GPT Researcher endpoints
api_router.include_router(rag_evaluation.router, tags=["rag-evaluation"])  # RAG Evaluation endpoints
api_router.include_router(hybrid_rag.router, prefix="/hybrid-rag", tags=["hybrid-rag"])  # Hybrid RAG endpoints

__all__ = ["api_router"]

#!/usr/bin/env python3
"""FastAPI server for ensemble paper quality scoring.

Provides REST API for paper quality assessment using ensemble of:
- GPT-4 (qualitative analysis)
- Hybrid model (fast scoring)
- Multi-task model (multi-dimensional feedback)

API Endpoints:
- POST /score - Score a paper
- GET /health - Health check
- GET /models - Model status
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("❌ FastAPI not installed. Install with:")
    print("   pip install fastapi uvicorn pydantic")
    sys.exit(1)

from src.services.paper.ensemble_scorer import EnsemblePaperScorer


# Request/Response models
class PaperScoreRequest(BaseModel):
    """Paper scoring request."""
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    content: Optional[str] = Field(None, description="Paper full text (optional)")
    return_individual: bool = Field(False, description="Return individual model scores")


class PaperScoreResponse(BaseModel):
    """Paper scoring response."""
    overall: float = Field(..., description="Overall quality score (1-10)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    model_type: str = Field(..., description="Model type (ensemble)")
    num_models: int = Field(..., description="Number of models used")
    dimensions: Optional[dict] = Field(None, description="Quality dimensions")
    individual_scores: Optional[dict] = Field(None, description="Individual model scores")
    agreement: Optional[dict] = Field(None, description="Model agreement analysis")
    gpt4_analysis: Optional[str] = Field(None, description="GPT-4 qualitative analysis")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict


class ModelStatusResponse(BaseModel):
    """Model status response."""
    hybrid: dict
    multitask: dict
    gpt4: dict
    ensemble_weights: dict


# Initialize FastAPI app
app = FastAPI(
    title="Paper Quality Scorer API",
    description="Ensemble-based scientific paper quality assessment",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ensemble scorer (lazy initialized)
ensemble_scorer: Optional[EnsemblePaperScorer] = None


def get_ensemble_scorer() -> EnsemblePaperScorer:
    """Get or create ensemble scorer."""
    global ensemble_scorer

    if ensemble_scorer is None:
        ensemble_scorer = EnsemblePaperScorer(
            gpt4_weight=0.4,
            hybrid_weight=0.3,
            multitask_weight=0.3,
            use_gpt4=True
        )

    return ensemble_scorer


@app.post("/score", response_model=PaperScoreResponse)
async def score_paper(request: PaperScoreRequest):
    """Score a scientific paper.

    Args:
        request: Paper scoring request with title, abstract, content

    Returns:
        Paper quality score with confidence and analysis
    """
    try:
        # Get ensemble scorer
        scorer = get_ensemble_scorer()

        # Combine paper text
        paper_text = f"{request.title}\n\n{request.abstract}"
        if request.content:
            paper_text += f"\n\n{request.content}"

        # Score
        result = await scorer.score_paper(
            paper_text=paper_text,
            return_individual=request.return_individual
        )

        return PaperScoreResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns:
        Service health status
    """
    try:
        scorer = get_ensemble_scorer()

        # Check model availability
        models_loaded = {
            "hybrid": scorer._hybrid_model is not None,
            "multitask": scorer._multitask_model is not None,
            "gpt4": scorer._gpt4_client is not None
        }

        return HealthResponse(
            status="healthy",
            models_loaded=models_loaded
        )

    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            models_loaded={}
        )


@app.get("/models", response_model=ModelStatusResponse)
async def model_status():
    """Get model status and configuration.

    Returns:
        Detailed model status
    """
    try:
        scorer = get_ensemble_scorer()

        return ModelStatusResponse(
            hybrid={
                "loaded": scorer._hybrid_model is not None,
                "path": "models/hybrid/best_model.pt",
                "weight": scorer.hybrid_weight
            },
            multitask={
                "loaded": scorer._multitask_model is not None,
                "path": "models/multitask/best_model.pt",
                "weight": scorer.multitask_weight
            },
            gpt4={
                "enabled": scorer.use_gpt4,
                "loaded": scorer._gpt4_client is not None,
                "weight": scorer.gpt4_weight
            },
            ensemble_weights={
                "gpt4": scorer.gpt4_weight,
                "hybrid": scorer.hybrid_weight,
                "multitask": scorer.multitask_weight
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Paper Quality Scorer API",
        "version": "2.0.0",
        "endpoints": {
            "POST /score": "Score a paper",
            "GET /health": "Health check",
            "GET /models": "Model status",
            "GET /docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start ensemble scoring API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print("=" * 80)
    print("PAPER QUALITY SCORER API SERVER")
    print("=" * 80)
    print()
    print(f"🌐 Starting server at http://{args.host}:{args.port}")
    print(f"📚 API docs at http://{args.host}:{args.port}/docs")
    print()
    print("Ensemble Configuration:")
    print("  - GPT-4 (40%): Qualitative analysis")
    print("  - Hybrid (30%): Fast RoBERTa + linguistic features")
    print("  - Multi-task (30%): Multi-dimensional quality scores")
    print()
    print("=" * 80)
    print()

    uvicorn.run(
        "start_ensemble_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

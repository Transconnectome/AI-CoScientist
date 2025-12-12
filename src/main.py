"""Main FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.api.v1 import api_router
from src.core.config import settings
from src.core.database import close_db, init_db
from src.core.redis import redis_client
from src.core.rl_system import rl_lifespan_manager, add_rl_health_endpoints


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    # Startup
    await init_db()

    # Initialize RL system with enhanced agent pool integration
    async with rl_lifespan_manager(app):
        yield

    # Shutdown
    await close_db()
    await redis_client.close()


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Add RL system health endpoints
add_rl_health_endpoints(app)


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus metrics endpoint."""
    # Placeholder: Will integrate with PrometheusMetricsExporter in Phase 2.2
    return """# HELP rag_evaluation_requests_total Total RAG evaluation requests
# TYPE rag_evaluation_requests_total counter
rag_evaluation_requests_total 0

# HELP rag_evaluation_latency_seconds RAG evaluation latency
# TYPE rag_evaluation_latency_seconds histogram
rag_evaluation_latency_seconds_bucket{le="0.1"} 0
rag_evaluation_latency_seconds_bucket{le="0.5"} 0
rag_evaluation_latency_seconds_bucket{le="1.0"} 0
rag_evaluation_latency_seconds_bucket{le="+Inf"} 0
rag_evaluation_latency_seconds_sum 0
rag_evaluation_latency_seconds_count 0
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=1 if settings.api_reload else settings.api_workers
    )

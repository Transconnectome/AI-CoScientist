"""
Integration test fixtures for Phase 2 RAG Evaluation System.

These fixtures provide:
- Real database connections (PostgreSQL)
- Async session management
- Test data factories
- Cleanup between tests

Following TDD methodology (RED → GREEN → REFACTOR):
- RED: Tests initially fail without services running
- GREEN: Tests pass once services deployed
- REFACTOR: Optimize and document
"""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.core.database import get_db
from src.main import app
from src.models.base import Base
from src.models.rag_evaluation import (
    RAGABTest,
    RAGABTestResult,
    RAGCostBudget,
    RAGEvaluation,
    RAGPerformanceMetric,
)

# Test database URL (separate from production)
# Uses adjusted port to avoid conflicts with existing PostgreSQL instances
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:local_dev_password@localhost:5434/ai_coscientist_test"
)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    # Create test database schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop test database schema after all tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session with automatic rollback."""
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        # Start transaction
        await session.begin()

        yield session

        # Rollback transaction (cleanup)
        await session.rollback()


@pytest_asyncio.fixture
async def clean_database(db_session: AsyncSession):
    """Clean all tables before each test."""
    # Delete in reverse order to respect foreign keys
    await db_session.execute(text("TRUNCATE TABLE rag_ab_test_results CASCADE"))
    await db_session.execute(text("TRUNCATE TABLE rag_ab_tests CASCADE"))
    await db_session.execute(text("TRUNCATE TABLE rag_cost_budgets CASCADE"))
    await db_session.execute(text("TRUNCATE TABLE rag_performance_metrics CASCADE"))
    await db_session.execute(text("TRUNCATE TABLE rag_evaluations CASCADE"))
    await db_session.commit()


@pytest_asyncio.fixture
async def api_client() -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client for API endpoints."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_evaluation_data():
    """Factory for creating sample RAG evaluation data."""
    return {
        "dataset_id": "test_dataset_001",
        "evaluation_type": "ragas",
        "metrics": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.90,
            "context_recall": 0.88
        },
        "evaluation_metadata": {
            "test": True,
            "created_by": "integration_test"
        }
    }


@pytest.fixture
def sample_performance_metric_data():
    """Factory for creating sample performance metric data."""
    return {
        "operation": "test_retrieval",
        "latency": 0.5,
        "token_usage": {
            "prompt": 100,
            "completion": 50
        },
        "cost": 0.01
    }


@pytest.fixture
def sample_budget_data():
    """Factory for creating sample budget data."""
    return {
        "name": "Test Budget",
        "total_budget": 100.0,
        "spent": 25.0,
        "warning_threshold": 0.8,
        "critical_threshold": 0.95,
        "expenses": {
            "retrieval": 10.0,
            "generation": 15.0
        }
    }


@pytest.fixture
def sample_ab_test_data():
    """Factory for creating sample A/B test data."""
    return {
        "name": "Test A/B Experiment",
        "config": {
            "variants": ["variant_a", "variant_b"],
            "traffic_split": 0.5,
            "metrics": ["accuracy", "latency"]
        },
        "status": "active"
    }


@pytest.fixture
def sample_ab_test_result_data():
    """Factory for creating sample A/B test result data."""
    return {
        "variant_name": "variant_a",
        "metrics": {
            "accuracy": 0.92,
            "latency": 0.45
        },
        "cost": 0.02
    }


# Database record factories
@pytest_asyncio.fixture
async def create_evaluation(db_session: AsyncSession, sample_evaluation_data):
    """Create and persist a sample RAG evaluation."""
    evaluation = RAGEvaluation(**sample_evaluation_data)
    db_session.add(evaluation)
    await db_session.commit()
    await db_session.refresh(evaluation)
    return evaluation


@pytest_asyncio.fixture
async def create_performance_metric(db_session: AsyncSession, sample_performance_metric_data):
    """Create and persist a sample performance metric."""
    metric = RAGPerformanceMetric(**sample_performance_metric_data)
    db_session.add(metric)
    await db_session.commit()
    await db_session.refresh(metric)
    return metric


@pytest_asyncio.fixture
async def create_budget(db_session: AsyncSession, sample_budget_data):
    """Create and persist a sample budget."""
    budget = RAGCostBudget(**sample_budget_data)
    db_session.add(budget)
    await db_session.commit()
    await db_session.refresh(budget)
    return budget


@pytest_asyncio.fixture
async def create_ab_test(db_session: AsyncSession, sample_ab_test_data):
    """Create and persist a sample A/B test."""
    ab_test = RAGABTest(**sample_ab_test_data)
    db_session.add(ab_test)
    await db_session.commit()
    await db_session.refresh(ab_test)
    return ab_test


@pytest_asyncio.fixture
async def create_ab_test_with_results(db_session: AsyncSession, sample_ab_test_data, sample_ab_test_result_data):
    """Create A/B test with sample results."""
    # Create test
    ab_test = RAGABTest(**sample_ab_test_data)
    db_session.add(ab_test)
    await db_session.commit()
    await db_session.refresh(ab_test)

    # Create results for both variants
    for variant in ["variant_a", "variant_b"]:
        result_data = sample_ab_test_result_data.copy()
        result_data["variant_name"] = variant
        result_data["test_id"] = ab_test.id

        # Vary metrics slightly
        if variant == "variant_b":
            result_data["metrics"]["accuracy"] = 0.88
            result_data["metrics"]["latency"] = 0.52

        result = RAGABTestResult(**result_data)
        db_session.add(result)

    await db_session.commit()
    return ab_test

"""
Tests for RAG evaluation database models.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import pytest
from datetime import datetime
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.models.base import Base
from src.models.rag_evaluation import (
    RAGEvaluation,
    RAGPerformanceMetric,
    RAGCostBudget,
    RAGABTest,
    RAGABTestResult
)


@pytest.fixture
def engine():
    """Create in-memory SQLite engine for testing"""
    engine = create_engine("sqlite:///:memory:")
    # Only create our RAG evaluation tables, not all tables
    RAGEvaluation.__table__.create(engine, checkfirst=True)
    RAGPerformanceMetric.__table__.create(engine, checkfirst=True)
    RAGCostBudget.__table__.create(engine, checkfirst=True)
    RAGABTest.__table__.create(engine, checkfirst=True)
    RAGABTestResult.__table__.create(engine, checkfirst=True)
    return engine


@pytest.fixture
def session(engine):
    """Create database session"""
    with Session(engine) as session:
        yield session


class TestRAGEvaluation:
    """RAG evaluation model tests"""

    def test_model_creation(self, session):
        """Test creating RAG evaluation record"""
        evaluation = RAGEvaluation(
            dataset_id="test-dataset-001",
            evaluation_type="ragas",
            metrics={"faithfulness": 0.85, "answer_relevancy": 0.80},
            evaluation_metadata={"model": "gpt-4", "temperature": 0.7}
        )

        session.add(evaluation)
        session.commit()

        assert evaluation.id is not None
        assert isinstance(evaluation.id, UUID)
        assert evaluation.dataset_id == "test-dataset-001"
        assert evaluation.evaluation_type == "ragas"
        assert evaluation.metrics["faithfulness"] == 0.85
        assert isinstance(evaluation.created_at, datetime)

    def test_query_by_dataset_id(self, session):
        """Test querying evaluations by dataset ID"""
        eval1 = RAGEvaluation(
            dataset_id="dataset-001",
            evaluation_type="ragas",
            metrics={}
        )
        eval2 = RAGEvaluation(
            dataset_id="dataset-001",
            evaluation_type="baseline",
            metrics={}
        )
        eval3 = RAGEvaluation(
            dataset_id="dataset-002",
            evaluation_type="ragas",
            metrics={}
        )

        session.add_all([eval1, eval2, eval3])
        session.commit()

        results = session.scalars(
            select(RAGEvaluation).where(RAGEvaluation.dataset_id == "dataset-001")
        ).all()

        assert len(results) == 2
        assert all(r.dataset_id == "dataset-001" for r in results)


class TestRAGPerformanceMetric:
    """RAG performance metric model tests"""

    def test_metric_creation(self, session):
        """Test creating performance metric record"""
        metric = RAGPerformanceMetric(
            operation="retrieval",
            latency=0.5,
            token_usage={"prompt": 100, "completion": 50},
            cost=0.025
        )

        session.add(metric)
        session.commit()

        assert metric.id is not None
        assert metric.operation == "retrieval"
        assert metric.latency == 0.5
        assert metric.token_usage["prompt"] == 100
        assert metric.cost == 0.025

    def test_query_by_operation(self, session):
        """Test querying metrics by operation type"""
        metrics = [
            RAGPerformanceMetric(operation="retrieval", latency=0.3, cost=0.01),
            RAGPerformanceMetric(operation="retrieval", latency=0.4, cost=0.015),
            RAGPerformanceMetric(operation="generation", latency=1.2, cost=0.05)
        ]

        session.add_all(metrics)
        session.commit()

        results = session.scalars(
            select(RAGPerformanceMetric).where(
                RAGPerformanceMetric.operation == "retrieval"
            )
        ).all()

        assert len(results) == 2
        assert all(r.operation == "retrieval" for r in results)


class TestRAGCostBudget:
    """RAG cost budget model tests"""

    def test_budget_creation(self, session):
        """Test creating cost budget record"""
        budget = RAGCostBudget(
            name="Monthly Budget",
            total_budget=100.0,
            spent=25.50,
            warning_threshold=0.8,
            critical_threshold=0.95,
            expenses={"retrieval": 10.0, "generation": 15.50}
        )

        session.add(budget)
        session.commit()

        assert budget.id is not None
        assert budget.name == "Monthly Budget"
        assert budget.total_budget == 100.0
        assert budget.spent == 25.50
        assert budget.expenses["retrieval"] == 10.0

    def test_budget_status_calculation(self, session):
        """Test budget status properties"""
        budget = RAGCostBudget(
            name="Test Budget",
            total_budget=100.0,
            spent=85.0,
            warning_threshold=0.8,
            critical_threshold=0.95
        )

        session.add(budget)
        session.commit()

        assert budget.remaining == 15.0
        assert budget.usage_ratio == 0.85
        # Status should be 'warning' (between 80% and 95%)
        assert budget.status == "warning"


class TestRAGABTest:
    """RAG A/B test model tests"""

    def test_ab_test_creation(self, session):
        """Test creating A/B test record"""
        ab_test = RAGABTest(
            name="Model Comparison",
            config={
                "variants": [
                    {"name": "gpt4", "config": {"model": "gpt-4"}},
                    {"name": "gpt35", "config": {"model": "gpt-3.5-turbo"}}
                ],
                "traffic_split": {"gpt4": 0.5, "gpt35": 0.5}
            },
            status="active"
        )

        session.add(ab_test)
        session.commit()

        assert ab_test.id is not None
        assert ab_test.name == "Model Comparison"
        assert ab_test.status == "active"
        assert len(ab_test.config["variants"]) == 2

    def test_query_active_tests(self, session):
        """Test querying active A/B tests"""
        tests = [
            RAGABTest(name="Test 1", config={}, status="active"),
            RAGABTest(name="Test 2", config={}, status="completed"),
            RAGABTest(name="Test 3", config={}, status="active")
        ]

        session.add_all(tests)
        session.commit()

        results = session.scalars(
            select(RAGABTest).where(RAGABTest.status == "active")
        ).all()

        assert len(results) == 2
        assert all(r.status == "active" for r in results)


class TestRAGABTestResult:
    """RAG A/B test result model tests"""

    def test_result_creation(self, session):
        """Test creating A/B test result record"""
        # Create test first
        ab_test = RAGABTest(name="Test", config={}, status="active")
        session.add(ab_test)
        session.commit()

        # Create result
        result = RAGABTestResult(
            test_id=ab_test.id,
            variant_name="gpt4",
            metrics={"faithfulness": 0.85, "latency": 0.5},
            cost=0.05
        )

        session.add(result)
        session.commit()

        assert result.id is not None
        assert result.test_id == ab_test.id
        assert result.variant_name == "gpt4"
        assert result.metrics["faithfulness"] == 0.85
        assert result.cost == 0.05

    def test_query_results_by_test(self, session):
        """Test querying results for specific A/B test"""
        # Create test
        ab_test = RAGABTest(name="Test", config={}, status="active")
        session.add(ab_test)
        session.commit()

        # Create results
        results = [
            RAGABTestResult(
                test_id=ab_test.id,
                variant_name="gpt4",
                metrics={"score": 0.9},
                cost=0.05
            ),
            RAGABTestResult(
                test_id=ab_test.id,
                variant_name="gpt35",
                metrics={"score": 0.8},
                cost=0.01
            )
        ]

        session.add_all(results)
        session.commit()

        query_results = session.scalars(
            select(RAGABTestResult).where(RAGABTestResult.test_id == ab_test.id)
        ).all()

        assert len(query_results) == 2
        assert all(r.test_id == ab_test.id for r in query_results)

    def test_relationship_with_ab_test(self, session):
        """Test foreign key relationship"""
        # Create test
        ab_test = RAGABTest(name="Test", config={}, status="active")
        session.add(ab_test)
        session.commit()

        # Create result with test relationship
        result = RAGABTestResult(
            test_id=ab_test.id,
            variant_name="variant_a",
            metrics={},
            cost=0.01
        )

        session.add(result)
        session.commit()

        # Verify relationship
        assert result.test_id == ab_test.id
        # Foreign key constraint should be enforced

"""
Integration tests for Phase 2 database schema and operations.

TDD RED Phase: These tests document expected database behavior.
Tests will initially fail without PostgreSQL running.

Tests cover:
- Schema creation (5 tables)
- Foreign key relationships
- Index existence
- CRUD operations
- Computed properties
- CASCADE deletion
"""

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.rag_evaluation import (
    RAGABTest,
    RAGABTestResult,
    RAGCostBudget,
    RAGEvaluation,
    RAGPerformanceMetric,
)


class TestDatabaseSchema:
    """Test database schema creation and structure."""

    @pytest.mark.asyncio
    async def test_all_tables_created(self, test_engine):
        """Verify all 5 RAG evaluation tables exist."""
        async with test_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'rag_%'"
            ))
            tables = [row[0] for row in result]

        expected_tables = {
            'rag_evaluations',
            'rag_performance_metrics',
            'rag_cost_budgets',
            'rag_ab_tests',
            'rag_ab_test_results'
        }

        assert set(tables) >= expected_tables, f"Missing tables: {expected_tables - set(tables)}"

    @pytest.mark.asyncio
    async def test_rag_evaluations_indexes(self, test_engine):
        """Verify indexes on rag_evaluations table."""
        async with test_engine.connect() as conn:
            def get_indexes(connection):
                inspector = inspect(connection)
                return inspector.get_indexes('rag_evaluations')

            indexes = await conn.run_sync(get_indexes)
            index_names = [idx['name'] for idx in indexes]

        assert 'ix_rag_evaluations_dataset_id' in index_names
        assert 'ix_rag_evaluations_evaluation_type' in index_names

    @pytest.mark.asyncio
    async def test_rag_performance_metrics_indexes(self, test_engine):
        """Verify indexes on rag_performance_metrics table."""
        async with test_engine.connect() as conn:
            def get_indexes(connection):
                inspector = inspect(connection)
                return inspector.get_indexes('rag_performance_metrics')

            indexes = await conn.run_sync(get_indexes)
            index_names = [idx['name'] for idx in indexes]

        assert 'ix_rag_performance_metrics_operation' in index_names

    @pytest.mark.asyncio
    async def test_foreign_key_relationship(self, test_engine):
        """Verify foreign key from rag_ab_test_results to rag_ab_tests."""
        async with test_engine.connect() as conn:
            def get_foreign_keys(connection):
                inspector = inspect(connection)
                return inspector.get_foreign_keys('rag_ab_test_results')

            foreign_keys = await conn.run_sync(get_foreign_keys)

        assert len(foreign_keys) > 0
        fk = foreign_keys[0]
        assert fk['referred_table'] == 'rag_ab_tests'
        assert 'test_id' in fk['constrained_columns']


class TestRAGEvaluationOperations:
    """Test RAG evaluation CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_and_retrieve_evaluation(self, db_session: AsyncSession, clean_database):
        """Test creating and retrieving RAG evaluation."""
        # Create
        evaluation = RAGEvaluation(
            dataset_id="integration_test_001",
            evaluation_type="ragas",
            metrics={"faithfulness": 0.95, "relevancy": 0.90},
            evaluation_metadata={"test": True}
        )
        db_session.add(evaluation)
        await db_session.commit()
        await db_session.refresh(evaluation)

        # Retrieve
        result = await db_session.execute(
            select(RAGEvaluation).where(RAGEvaluation.dataset_id == "integration_test_001")
        )
        retrieved = result.scalar_one()

        assert retrieved.id == evaluation.id
        assert retrieved.metrics["faithfulness"] == 0.95
        assert retrieved.created_at is not None

    @pytest.mark.asyncio
    async def test_query_by_evaluation_type(self, db_session: AsyncSession, clean_database):
        """Test querying evaluations by type."""
        # Create multiple evaluations
        for i in range(3):
            eval_type = "ragas" if i < 2 else "baseline"
            evaluation = RAGEvaluation(
                dataset_id=f"dataset_{i}",
                evaluation_type=eval_type,
                metrics={"score": 0.8 + (i * 0.05)}
            )
            db_session.add(evaluation)
        await db_session.commit()

        # Query RAGAS evaluations
        result = await db_session.execute(
            select(RAGEvaluation).where(RAGEvaluation.evaluation_type == "ragas")
        )
        ragas_evals = result.scalars().all()

        assert len(ragas_evals) == 2


class TestRAGPerformanceMetricOperations:
    """Test performance metric CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_performance_metric(self, db_session: AsyncSession, clean_database):
        """Test creating performance metric with all fields."""
        metric = RAGPerformanceMetric(
            operation="integration_test_retrieval",
            latency=0.45,
            token_usage={"prompt": 150, "completion": 75},
            cost=0.015
        )
        db_session.add(metric)
        await db_session.commit()
        await db_session.refresh(metric)

        assert metric.id is not None
        assert metric.operation == "integration_test_retrieval"
        assert metric.token_usage["prompt"] == 150

    @pytest.mark.asyncio
    async def test_query_metrics_by_operation(self, db_session: AsyncSession, clean_database):
        """Test querying metrics by operation type."""
        # Create metrics for different operations
        for op in ["retrieval", "generation", "retrieval"]:
            metric = RAGPerformanceMetric(
                operation=op,
                latency=0.5,
                cost=0.01
            )
            db_session.add(metric)
        await db_session.commit()

        # Query retrieval metrics
        result = await db_session.execute(
            select(RAGPerformanceMetric).where(RAGPerformanceMetric.operation == "retrieval")
        )
        retrieval_metrics = result.scalars().all()

        assert len(retrieval_metrics) == 2


class TestRAGCostBudgetOperations:
    """Test cost budget CRUD operations and computed properties."""

    @pytest.mark.asyncio
    async def test_budget_computed_properties(self, db_session: AsyncSession, clean_database):
        """Test budget remaining, usage_ratio, and status properties."""
        budget = RAGCostBudget(
            name="Integration Test Budget",
            total_budget=100.0,
            spent=85.0,
            warning_threshold=0.8,
            critical_threshold=0.95
        )
        db_session.add(budget)
        await db_session.commit()
        await db_session.refresh(budget)

        # Test computed properties
        assert budget.remaining == 15.0
        assert budget.usage_ratio == 0.85
        assert budget.status == "warning"  # 0.85 >= 0.8, < 0.95

    @pytest.mark.asyncio
    async def test_budget_status_thresholds(self, db_session: AsyncSession, clean_database):
        """Test budget status changes based on thresholds."""
        # Normal status
        budget_normal = RAGCostBudget(
            name="Normal Budget",
            total_budget=100.0,
            spent=70.0
        )
        db_session.add(budget_normal)

        # Warning status
        budget_warning = RAGCostBudget(
            name="Warning Budget",
            total_budget=100.0,
            spent=85.0
        )
        db_session.add(budget_warning)

        # Critical status
        budget_critical = RAGCostBudget(
            name="Critical Budget",
            total_budget=100.0,
            spent=96.0
        )
        db_session.add(budget_critical)

        await db_session.commit()
        await db_session.refresh(budget_normal)
        await db_session.refresh(budget_warning)
        await db_session.refresh(budget_critical)

        assert budget_normal.status == "normal"
        assert budget_warning.status == "warning"
        assert budget_critical.status == "critical"


class TestRAGABTestOperations:
    """Test A/B test CRUD operations and relationships."""

    @pytest.mark.asyncio
    async def test_create_ab_test(self, db_session: AsyncSession, clean_database):
        """Test creating A/B test with configuration."""
        ab_test = RAGABTest(
            name="Integration Test Experiment",
            config={"variants": ["A", "B"], "split": 0.5},
            status="active"
        )
        db_session.add(ab_test)
        await db_session.commit()
        await db_session.refresh(ab_test)

        assert ab_test.id is not None
        assert ab_test.config["variants"] == ["A", "B"]
        assert ab_test.status == "active"

    @pytest.mark.asyncio
    async def test_ab_test_with_results(self, db_session: AsyncSession, clean_database):
        """Test creating A/B test with multiple results."""
        # Create test
        ab_test = RAGABTest(
            name="Test with Results",
            config={"variants": ["variant_a", "variant_b"]},
            status="active"
        )
        db_session.add(ab_test)
        await db_session.commit()
        await db_session.refresh(ab_test)

        # Create results
        for variant in ["variant_a", "variant_b"]:
            result = RAGABTestResult(
                test_id=ab_test.id,
                variant_name=variant,
                metrics={"score": 0.9 if variant == "variant_a" else 0.85},
                cost=0.02
            )
            db_session.add(result)
        await db_session.commit()

        # Query results
        results_query = await db_session.execute(
            select(RAGABTestResult).where(RAGABTestResult.test_id == ab_test.id)
        )
        results = results_query.scalars().all()

        assert len(results) == 2
        assert {r.variant_name for r in results} == {"variant_a", "variant_b"}

    @pytest.mark.asyncio
    async def test_cascade_delete_ab_test(self, db_session: AsyncSession, clean_database):
        """Test CASCADE deletion when A/B test is deleted."""
        # Create test with results
        ab_test = RAGABTest(
            name="Test CASCADE",
            config={"test": True},
            status="active"
        )
        db_session.add(ab_test)
        await db_session.commit()
        await db_session.refresh(ab_test)

        # Create result
        result = RAGABTestResult(
            test_id=ab_test.id,
            variant_name="test_variant",
            metrics={"score": 0.9},
            cost=0.01
        )
        db_session.add(result)
        await db_session.commit()
        result_id = result.id

        # Delete A/B test
        await db_session.delete(ab_test)
        await db_session.commit()

        # Verify result was also deleted (CASCADE)
        result_check = await db_session.execute(
            select(RAGABTestResult).where(RAGABTestResult.id == result_id)
        )
        assert result_check.scalar_one_or_none() is None

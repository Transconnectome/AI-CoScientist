"""
RAG Evaluation database models.

Models for storing RAG evaluation data, performance metrics,
cost tracking, and A/B testing results.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Float, ForeignKey, JSON, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

# Use JSON for cross-database compatibility (works with SQLite and PostgreSQL)
# PostgreSQL will upgrade to JSONB automatically when possible
JSONType = JSON().with_variant(JSONB(), "postgresql")

from src.models.base import Base, TimestampMixin, UUIDMixin


class RAGEvaluation(Base, UUIDMixin, TimestampMixin):
    """RAG evaluation results table.

    Stores evaluation results from RAGAS and baseline evaluations.
    """

    __tablename__ = "rag_evaluations"

    dataset_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Dataset identifier for the evaluation"
    )

    evaluation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of evaluation (ragas, baseline, etc.)"
    )

    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONType,
        nullable=False,
        comment="Evaluation metrics as JSON"
    )

    evaluation_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONType,
        nullable=True,
        comment="Additional metadata as JSON"
    )

    def __repr__(self) -> str:
        return f"<RAGEvaluation(id={self.id}, type={self.evaluation_type}, dataset={self.dataset_id})>"


class RAGPerformanceMetric(Base, UUIDMixin, TimestampMixin):
    """RAG performance metrics table.

    Stores latency, token usage, and cost metrics for RAG operations.
    """

    __tablename__ = "rag_performance_metrics"

    operation: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Operation name (retrieval, generation, etc.)"
    )

    latency: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Operation latency in seconds"
    )

    token_usage: Mapped[dict[str, Any] | None] = mapped_column(
        JSONType,
        nullable=True,
        comment="Token usage details as JSON"
    )

    cost: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Estimated cost in USD"
    )

    def __repr__(self) -> str:
        return f"<RAGPerformanceMetric(id={self.id}, operation={self.operation}, latency={self.latency}s)>"


class RAGCostBudget(Base, UUIDMixin, TimestampMixin):
    """RAG cost budget table.

    Stores cost budgets and spending tracking.
    """

    __tablename__ = "rag_cost_budgets"

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Budget name"
    )

    total_budget: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Total budget amount in USD"
    )

    spent: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Amount spent in USD"
    )

    warning_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.8,
        comment="Warning threshold as ratio (0-1)"
    )

    critical_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.95,
        comment="Critical threshold as ratio (0-1)"
    )

    expenses: Mapped[dict[str, float] | None] = mapped_column(
        JSONType,
        nullable=True,
        comment="Expense breakdown by category"
    )

    @property
    def remaining(self) -> float:
        """Calculate remaining budget"""
        return float(self.total_budget - self.spent)

    @property
    def usage_ratio(self) -> float:
        """Calculate budget usage ratio"""
        if self.total_budget == 0:
            return 0.0
        return float(self.spent / self.total_budget)

    @property
    def status(self) -> str:
        """Get budget status based on thresholds"""
        ratio = self.usage_ratio
        if ratio >= self.critical_threshold:
            return "critical"
        elif ratio >= self.warning_threshold:
            return "warning"
        else:
            return "normal"

    def __repr__(self) -> str:
        return f"<RAGCostBudget(id={self.id}, name={self.name}, spent={self.spent}/{self.total_budget})>"


class RAGABTest(Base, UUIDMixin, TimestampMixin):
    """RAG A/B test table.

    Stores A/B test configurations and metadata.
    """

    __tablename__ = "rag_ab_tests"

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Test name"
    )

    config: Mapped[dict[str, Any]] = mapped_column(
        JSONType,
        nullable=False,
        comment="Test configuration (variants, traffic split, etc.)"
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        index=True,
        comment="Test status (active, completed, paused)"
    )

    def __repr__(self) -> str:
        return f"<RAGABTest(id={self.id}, name={self.name}, status={self.status})>"


class RAGABTestResult(Base, UUIDMixin, TimestampMixin):
    """RAG A/B test result table.

    Stores individual results for A/B tests.
    """

    __tablename__ = "rag_ab_test_results"

    test_id: Mapped[UUID] = mapped_column(
        ForeignKey("rag_ab_tests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to parent A/B test"
    )

    variant_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Variant name"
    )

    metrics: Mapped[dict[str, float]] = mapped_column(
        JSONType,
        nullable=False,
        comment="Result metrics as JSON"
    )

    cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Cost for this result in USD"
    )

    def __repr__(self) -> str:
        return f"<RAGABTestResult(id={self.id}, test_id={self.test_id}, variant={self.variant_name})>"

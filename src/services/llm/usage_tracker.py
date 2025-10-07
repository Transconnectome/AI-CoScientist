"""LLM usage tracking and metrics."""

from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass, field

from src.services.llm.types import LLMRequest, LLMResponse, TaskType, ModelProvider


@dataclass
class UsageMetrics:
    """Track LLM usage metrics."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    requests_by_task: Dict[TaskType, int] = field(default_factory=dict)
    tokens_by_provider: Dict[ModelProvider, int] = field(default_factory=dict)
    errors: int = 0


class UsageTracker:
    """Track and report LLM usage."""

    def __init__(self, metrics_store: Any = None):
        """Initialize usage tracker."""
        self.store = metrics_store
        self.metrics = UsageMetrics()

    async def record_request(
        self,
        request: LLMRequest,
        response: LLMResponse
    ) -> None:
        """Record request metrics."""
        self.metrics.total_requests += 1
        self.metrics.total_tokens += response.tokens_used
        self.metrics.total_cost += response.cost

        # By task type
        task_count = self.metrics.requests_by_task.get(request.task_type, 0)
        self.metrics.requests_by_task[request.task_type] = task_count + 1

        # By provider
        provider_tokens = self.metrics.tokens_by_provider.get(
            response.provider, 0
        )
        self.metrics.tokens_by_provider[response.provider] = (
            provider_tokens + response.tokens_used
        )

        # Persist to database if store available
        if self.store:
            await self.store.save_metrics({
                "timestamp": datetime.utcnow(),
                "task_type": request.task_type.value,
                "provider": response.provider.value,
                "model": response.model,
                "tokens": response.tokens_used,
                "cost": response.cost,
                "latency_ms": response.latency_ms
            })

    async def record_error(self, error: Exception) -> None:
        """Record error."""
        self.metrics.errors += 1
        if self.store:
            await self.store.save_error({
                "timestamp": datetime.utcnow(),
                "error_type": type(error).__name__,
                "message": str(error)
            })

    def get_metrics(self) -> UsageMetrics:
        """Get current metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.metrics = UsageMetrics()

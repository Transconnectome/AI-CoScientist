"""Anthropic adapter for LLM service."""

import time
from typing import AsyncIterator, List

from anthropic import AsyncAnthropic

from src.services.llm.interface import LLMServiceInterface
from src.services.llm.types import (
    LLMConfig,
    LLMRequest,
    LLMResponse,
    ModelProvider,
    TaskType
)


class AnthropicAdapter(LLMServiceInterface):
    """Anthropic Claude adapter."""

    def __init__(self, api_key: str):
        """Initialize Anthropic adapter."""
        self.client = AsyncAnthropic(api_key=api_key)

        # Model pricing (per 1M tokens)
        self.pricing = {
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}
        }

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Anthropic."""
        config = request.config or self._get_default_config(request.task_type)

        start_time = time.time()

        response = await self.client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=request.system_message or "",
            messages=[{"role": "user", "content": request.prompt}]
        )

        latency_ms = (time.time() - start_time) * 1000

        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        cost = self.get_cost(tokens_used, config.model)

        return LLMResponse(
            content=response.content[0].text,
            model=config.model,
            provider=ModelProvider.ANTHROPIC,
            tokens_used=tokens_used,
            cost=cost,
            latency_ms=latency_ms,
            finish_reason=response.stop_reason or "complete",
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        )

    async def stream_complete(
        self,
        request: LLMRequest
    ) -> AsyncIterator[str]:
        """Stream completion using Anthropic."""
        config = request.config or self._get_default_config(request.task_type)

        async with self.client.messages.stream(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=request.system_message or "",
            messages=[{"role": "user", "content": request.prompt}]
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings (not natively supported by Anthropic)."""
        raise NotImplementedError(
            "Anthropic does not provide native embedding API. "
            "Use OpenAI or local models for embeddings."
        )

    def get_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for Anthropic API usage."""
        if model not in self.pricing:
            return 0.0

        # Simplified: assume 50/50 input/output split
        input_tokens = tokens // 2
        output_tokens = tokens - input_tokens

        pricing = self.pricing[model]
        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        return round(cost, 6)

    def _get_default_config(self, task_type: TaskType) -> LLMConfig:
        """Get default config for task type."""
        return LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=2000
        )

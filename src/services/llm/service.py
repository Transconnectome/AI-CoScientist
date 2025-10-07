"""Main LLM service orchestrator."""

from typing import AsyncIterator, Dict, Optional, Any

from redis.asyncio import Redis

from src.core.config import settings
from src.services.llm.interface import LLMServiceInterface
from src.services.llm.types import (
    LLMConfig,
    LLMRequest,
    LLMResponse,
    ModelProvider,
    TaskType
)
from src.services.llm.adapters import OpenAIAdapter, AnthropicAdapter
from src.services.llm.prompt_manager import PromptManager
from src.services.llm.cache import LLMCache
from src.services.llm.usage_tracker import UsageTracker


class LLMService:
    """Main LLM service orchestrator."""

    def __init__(
        self,
        redis_client: Redis,
        metrics_store: Optional[Any] = None
    ):
        """Initialize LLM service."""
        # Initialize adapters
        self.adapters: Dict[ModelProvider, LLMServiceInterface] = {
            ModelProvider.OPENAI: OpenAIAdapter(settings.openai_api_key),
            ModelProvider.ANTHROPIC: AnthropicAdapter(settings.anthropic_api_key)
        }

        # Initialize supporting services
        self.prompt_manager = PromptManager()
        self.cache = LLMCache(redis_client, ttl=settings.llm_cache_ttl)
        self.usage_tracker = UsageTracker(metrics_store)

        # Configuration
        self.primary_provider = ModelProvider(settings.llm_primary_provider)
        self.fallback_provider = ModelProvider(settings.llm_fallback_provider)
        self.cache_enabled = settings.llm_cache_enabled
        self.max_retries = settings.llm_max_retries

    async def complete(
        self,
        request: LLMRequest,
        use_cache: bool = True
    ) -> LLMResponse:
        """Generate completion with caching and fallback."""

        # Get config
        config = request.config or self._get_default_config(request.task_type)

        # Check cache if enabled
        if use_cache and self.cache_enabled:
            cached_response = await self.cache.get(request.prompt, config)
            if cached_response:
                return LLMResponse(
                    content=cached_response,
                    model="cached",
                    provider=config.provider,
                    tokens_used=0,
                    cost=0.0,
                    latency_ms=0.0,
                    finish_reason="cached",
                    metadata={"cached": True}
                )

        # Get adapter
        adapter = self.adapters.get(config.provider)
        if not adapter:
            raise ValueError(f"Unknown provider: {config.provider}")

        # Attempt completion with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Generate completion
                response = await adapter.complete(request)

                # Cache response if enabled
                if use_cache and self.cache_enabled:
                    await self.cache.set(
                        request.prompt,
                        config,
                        response.content
                    )

                # Track usage
                await self.usage_tracker.record_request(request, response)

                return response

            except Exception as e:
                last_error = e
                await self.usage_tracker.record_error(e)

                # If max retries exhausted, try fallback provider
                if attempt == self.max_retries - 1:
                    if config.provider == self.primary_provider:
                        # Try fallback provider
                        fallback_config = LLMConfig(
                            provider=self.fallback_provider,
                            model=self._get_fallback_model(self.fallback_provider),
                            temperature=config.temperature,
                            max_tokens=config.max_tokens
                        )
                        request.config = fallback_config
                        return await self.complete(request, use_cache=False)

        # If all retries failed, raise last error
        if last_error:
            raise last_error
        raise RuntimeError("LLM completion failed with unknown error")

    async def stream_complete(
        self,
        request: LLMRequest
    ) -> AsyncIterator[str]:
        """Stream completion from LLM."""
        config = request.config or self._get_default_config(request.task_type)
        adapter = self.adapters.get(config.provider)

        if not adapter:
            raise ValueError(f"Unknown provider: {config.provider}")

        async for chunk in adapter.stream_complete(request):
            yield chunk

    async def complete_with_template(
        self,
        template_name: str,
        context: Dict[str, Any],
        task_type: TaskType,
        config: Optional[LLMConfig] = None,
        system_message: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion using template."""

        # Render prompt
        prompt = self.prompt_manager.render_prompt(template_name, context)

        # Validate
        if not self.prompt_manager.validate_prompt(prompt):
            raise ValueError("Prompt exceeds token limit")

        # Optimize
        prompt = self.prompt_manager.optimize_prompt(prompt)

        # Create request
        request = LLMRequest(
            prompt=prompt,
            task_type=task_type,
            config=config,
            context=context,
            system_message=system_message
        )

        return await self.complete(request)

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using primary provider."""
        adapter = self.adapters.get(self.primary_provider)
        if not adapter:
            raise ValueError(f"Unknown provider: {self.primary_provider}")
        return await adapter.embed(text)

    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics."""
        metrics = self.usage_tracker.get_metrics()
        return {
            "total_requests": metrics.total_requests,
            "total_tokens": metrics.total_tokens,
            "total_cost": metrics.total_cost,
            "requests_by_task": {
                task.value: count
                for task, count in metrics.requests_by_task.items()
            },
            "tokens_by_provider": {
                provider.value: tokens
                for provider, tokens in metrics.tokens_by_provider.items()
            },
            "errors": metrics.errors
        }

    def _get_default_config(self, task_type: TaskType) -> LLMConfig:
        """Get default config for task type."""
        # Delegate to primary adapter
        adapter = self.adapters.get(self.primary_provider)
        if hasattr(adapter, '_get_default_config'):
            return adapter._get_default_config(task_type)

        # Fallback to generic config
        return LLMConfig(
            provider=self.primary_provider,
            model=settings.openai_model,
            temperature=0.7,
            max_tokens=2000
        )

    def _get_fallback_model(self, provider: ModelProvider) -> str:
        """Get fallback model for provider."""
        if provider == ModelProvider.OPENAI:
            return settings.openai_model
        elif provider == ModelProvider.ANTHROPIC:
            return settings.anthropic_model
        return "gpt-4"

"""OpenAI adapter for LLM service."""

import time
from typing import AsyncIterator, Dict, List, Optional

import tiktoken
from openai import AsyncOpenAI

from src.services.llm.interface import LLMServiceInterface
from src.services.llm.types import (
    LLMConfig,
    LLMRequest,
    LLMResponse,
    ModelProvider,
    TaskType
)


class OpenAIAdapter(LLMServiceInterface):
    """OpenAI GPT adapter."""

    def __init__(self, api_key: str):
        """Initialize OpenAI adapter."""
        self.client = AsyncOpenAI(api_key=api_key)
        self.encoder = tiktoken.encoding_for_model("gpt-4")

        # Model pricing (per 1K tokens)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI."""
        config = request.config or self._get_default_config(request.task_type)

        messages = self._build_messages(request)

        start_time = time.time()

        response = await self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            stop=config.stop_sequences,
            timeout=config.timeout
        )

        latency_ms = (time.time() - start_time) * 1000

        tokens_used = response.usage.total_tokens
        cost = self.get_cost(tokens_used, config.model)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=config.model,
            provider=ModelProvider.OPENAI,
            tokens_used=tokens_used,
            cost=cost,
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason,
            metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
        )

    async def stream_complete(
        self,
        request: LLMRequest
    ) -> AsyncIterator[str]:
        """Stream completion using OpenAI."""
        config = request.config or self._get_default_config(request.task_type)
        messages = self._build_messages(request)

        stream = await self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def get_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for OpenAI API usage."""
        if model not in self.pricing:
            return 0.0

        # Simplified: assume 50/50 input/output split
        input_tokens = tokens // 2
        output_tokens = tokens - input_tokens

        pricing = self.pricing[model]
        cost = (
            (input_tokens / 1000) * pricing["input"] +
            (output_tokens / 1000) * pricing["output"]
        )
        return round(cost, 6)

    def _build_messages(self, request: LLMRequest) -> List[Dict]:
        """Build message array for OpenAI."""
        messages = []

        # System message
        if request.system_message:
            messages.append({
                "role": "system",
                "content": request.system_message
            })

        # Few-shot examples
        if request.examples:
            for example in request.examples:
                messages.append({
                    "role": "user",
                    "content": example["input"]
                })
                messages.append({
                    "role": "assistant",
                    "content": example["output"]
                })

        # User prompt
        messages.append({
            "role": "user",
            "content": request.prompt
        })

        return messages

    def _get_default_config(self, task_type: TaskType) -> LLMConfig:
        """Get default config for task type."""
        configs = {
            TaskType.HYPOTHESIS_GENERATION: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4-turbo-preview",
                temperature=0.8,
                max_tokens=1000
            ),
            TaskType.LITERATURE_ANALYSIS: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4",
                temperature=0.3,
                max_tokens=2000
            ),
            TaskType.EXPERIMENT_DESIGN: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4",
                temperature=0.5,
                max_tokens=1500
            ),
            TaskType.DATA_ANALYSIS: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4",
                temperature=0.2,
                max_tokens=2000
            ),
            TaskType.PAPER_WRITING: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4-turbo-preview",
                temperature=0.6,
                max_tokens=3000
            ),
            TaskType.PEER_REVIEW: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4",
                temperature=0.4,
                max_tokens=2000
            )
        }
        return configs.get(task_type, LLMConfig(
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000
        ))

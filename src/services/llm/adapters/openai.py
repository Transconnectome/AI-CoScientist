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

        # Model pricing (per 1K tokens) - Updated 2025
        # Model pricing (per 1K tokens) - Updated 2025
        self.pricing = {
            "gpt-5-pro": {"input": 0.015, "output": 0.120},  # GPT-5 Pro
            "gpt-5-codex": {"input": 0.015, "output": 0.120},
            "gpt-5": {"input": 0.003, "output": 0.015},
            "gpt-5-mini": {"input": 0.0005, "output": 0.002},
            "gpt-5-nano": {"input": 0.0001, "output": 0.0005},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI."""
        config = request.config or self._get_default_config(request.task_type)
        messages = self._build_messages(request)
        start_time = time.time()

        # Check for models requiring 'responses' API
        use_responses_api = config.model in ["gpt-5-pro", "gpt-5-codex", "gpt-5-pro-2025-10-06"]

        if use_responses_api:
            # Extract the actual user prompt (last user message) for straightforward 'input' usage
            # Or use 'conversation' param if supported/needed (omitted for simplicity based on inspection)
            # We'll use the last user message content as the primary 'input'.
            prompt_content = next((m["content"] for m in reversed(messages) if m["role"] == "user"), request.prompt)
            
            try:
                response = await self.client.responses.create(
                    model=config.model,
                    input=prompt_content,
                    max_output_tokens=config.max_tokens if config.max_tokens >= 16 else 16
                )
                # Map response content
                content_text = ""
                if hasattr(response, 'output_text'):
                    content_text = response.output_text
                elif hasattr(response, 'output'):
                    content_text = response.output
                else: 
                     # Fallback inspection
                     content_text = str(response)

                finish_reason = "stop" # Default for responses API usually
                usage_prompt = 0
                usage_completion = 0
                # Try to extract usage if available
                if hasattr(response, 'usage'):
                    usage_prompt = getattr(response.usage, 'input_tokens', 0)
                    usage_completion = getattr(response.usage, 'output_tokens', 0)

            except Exception as e:
                # Fallback or error re-raise
                raise e
        else:
             # Standard Chat Completions API
            response = await self.client.chat.completions.create(
                model=config.model,
                messages=messages,
                # temperature=config.temperature,
                max_completion_tokens=config.max_tokens, 
                timeout=config.timeout
            )
            content_text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            usage_prompt = response.usage.prompt_tokens
            usage_completion = response.usage.completion_tokens

        latency_ms = (time.time() - start_time) * 1000
        tokens_used = usage_prompt + usage_completion
        cost = self.get_cost(tokens_used, config.model)

        return LLMResponse(
            content=content_text,
            model=config.model,
            provider=ModelProvider.OPENAI,
            tokens_used=tokens_used,
            cost=cost,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            metadata={
                "input_tokens": usage_prompt,
                "output_tokens": usage_completion
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
        from src.core.config import get_settings
        settings = get_settings()
        
        # Use gpt-5-pro or gpt-5 based on task importance if not explicitly overridden by OPENAI_MODEL
        # Actually, if the user set OPENAI_MODEL, we should probably prefer it.
        default_model = settings.openai_model
        
        configs = {
            TaskType.HYPOTHESIS_GENERATION: LLMConfig(
                provider=ModelProvider.OPENAI,
                model=default_model,
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens
            ),
            TaskType.LITERATURE_ANALYSIS: LLMConfig(
                provider=ModelProvider.OPENAI,
                model=default_model,
                temperature=0.3,
                max_tokens=settings.openai_max_tokens
            ),
            TaskType.EXPERIMENT_DESIGN: LLMConfig(
                provider=ModelProvider.OPENAI,
                model=default_model,
                temperature=0.5,
                max_tokens=settings.openai_max_tokens
            ),
            TaskType.DATA_ANALYSIS: LLMConfig(
                provider=ModelProvider.OPENAI,
                model=default_model,
                temperature=0.2,
                max_tokens=settings.openai_max_tokens
            ),
            TaskType.PAPER_WRITING: LLMConfig(
                provider=ModelProvider.OPENAI,
                model=default_model,
                temperature=0.6,
                max_tokens=settings.openai_max_tokens
            ),
            TaskType.PEER_REVIEW: LLMConfig(
                provider=ModelProvider.OPENAI,
                model=default_model,
                temperature=0.4,
                max_tokens=settings.openai_max_tokens
            )
        }
        return configs.get(task_type, LLMConfig(
            provider=ModelProvider.OPENAI,
            model=default_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens
        ))

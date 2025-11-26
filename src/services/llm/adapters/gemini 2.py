"""Google Gemini adapter for LLM service."""

import time
from typing import AsyncIterator, List

import google.generativeai as genai

from src.services.llm.interface import LLMServiceInterface
from src.services.llm.types import (
    LLMConfig,
    LLMRequest,
    LLMResponse,
    ModelProvider,
    TaskType
)


class GeminiAdapter(LLMServiceInterface):
    """Google Gemini adapter."""

    def __init__(self, api_key: str):
        """Initialize Gemini adapter."""
        genai.configure(api_key=api_key)

        # Model pricing (per 1M tokens) - Updated 2025
        self.pricing = {
            "gemini-2.5-pro": {"input": 1.25, "output": 5.0},  # Gemini 2.5 Pro (Mar 2025)
            "gemini-2.0-flash": {"input": 0.075, "output": 0.3},  # Gemini 2.0 Flash
            "gemini-1.5-pro": {"input": 1.25, "output": 5.0},  # Gemini 1.5 Pro
            "gemini-1.5-flash": {"input": 0.075, "output": 0.3},  # Gemini 1.5 Flash
        }

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Gemini."""
        config = request.config or self._get_default_config(request.task_type)

        # Build messages
        messages = self._build_messages(request)

        start_time = time.time()

        # Initialize model
        model = genai.GenerativeModel(
            model_name=config.model,
            generation_config=genai.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                top_p=config.top_p,
            )
        )

        # Generate response
        response = await model.generate_content_async(messages)

        latency_ms = (time.time() - start_time) * 1000

        # Calculate tokens (Gemini provides usage metadata)
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        tokens_used = input_tokens + output_tokens

        cost = self.get_cost(tokens_used, config.model)

        return LLMResponse(
            content=response.text,
            model=config.model,
            provider=ModelProvider.GOOGLE,
            tokens_used=tokens_used,
            cost=cost,
            latency_ms=latency_ms,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else "complete",
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        )

    async def stream_complete(
        self,
        request: LLMRequest
    ) -> AsyncIterator[str]:
        """Stream completion using Gemini."""
        config = request.config or self._get_default_config(request.task_type)
        messages = self._build_messages(request)

        # Initialize model
        model = genai.GenerativeModel(
            model_name=config.model,
            generation_config=genai.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                top_p=config.top_p,
            )
        )

        # Stream response
        response = await model.generate_content_async(
            messages,
            stream=True
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using Gemini."""
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def get_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for Gemini API usage."""
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

    def _build_messages(self, request: LLMRequest) -> str:
        """Build message for Gemini.

        Gemini uses a simpler message format - just concatenate system + prompt.
        """
        parts = []

        # System message
        if request.system_message:
            parts.append(f"System: {request.system_message}\n")

        # Few-shot examples
        if request.examples:
            for example in request.examples:
                parts.append(f"User: {example['input']}\n")
                parts.append(f"Assistant: {example['output']}\n")

        # User prompt
        parts.append(f"User: {request.prompt}")

        return "\n".join(parts)

    def _get_default_config(self, task_type: TaskType) -> LLMConfig:
        """Get default config for task type."""
        configs = {
            TaskType.HYPOTHESIS_GENERATION: LLMConfig(
                provider=ModelProvider.GOOGLE,
                model="gemini-2.5-pro",  # Gemini 2.5 Pro for creative tasks
                temperature=0.8,
                max_tokens=1000
            ),
            TaskType.LITERATURE_ANALYSIS: LLMConfig(
                provider=ModelProvider.GOOGLE,
                model="gemini-2.5-pro",  # Gemini 2.5 Pro for analytical tasks
                temperature=0.3,
                max_tokens=2000
            ),
            TaskType.EXPERIMENT_DESIGN: LLMConfig(
                provider=ModelProvider.GOOGLE,
                model="gemini-2.5-pro",  # Gemini 2.5 Pro for structured design
                temperature=0.5,
                max_tokens=1500
            ),
            TaskType.DATA_ANALYSIS: LLMConfig(
                provider=ModelProvider.GOOGLE,
                model="gemini-2.5-pro",  # Gemini 2.5 Pro for precision
                temperature=0.2,
                max_tokens=2000
            ),
            TaskType.PAPER_WRITING: LLMConfig(
                provider=ModelProvider.GOOGLE,
                model="gemini-2.5-pro",  # Gemini 2.5 Pro for long-form writing
                temperature=0.6,
                max_tokens=3000
            ),
            TaskType.PEER_REVIEW: LLMConfig(
                provider=ModelProvider.GOOGLE,
                model="gemini-2.5-pro",  # Gemini 2.5 Pro for critical review
                temperature=0.4,
                max_tokens=2000
            )
        }
        return configs.get(task_type, LLMConfig(
            provider=ModelProvider.GOOGLE,
            model="gemini-2.5-pro",  # Default to Gemini 2.5 Pro
            temperature=0.7,
            max_tokens=2000
        ))

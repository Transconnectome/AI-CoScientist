"""
Nemotron LLM Service

Wrapper for NVIDIA Nemotron Nano 9B V2 model with OpenAI-compatible API.
"""

import os
import httpx
import asyncio
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    """Result from completion request."""
    text: str
    model: str
    tokens_generated: int
    tokens_prompt: int
    finish_reason: str
    latency_ms: float


@dataclass
class ChatMessage:
    """Chat message."""
    role: str  # 'system', 'user', or 'assistant'
    content: str


class NemotronLLM:
    """
    Nemotron LLM Service.

    Provides access to Nemotron Nano 9B V2 model via OpenAI-compatible API.
    """

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        timeout: float = 60.0
    ):
        """
        Initialize Nemotron LLM service.

        Args:
            base_url: Base URL for Nemotron service (default: env NEMOTRON_BASE_URL)
            model: Model identifier (default: env NEMOTRON_MODEL)
            temperature: Sampling temperature (default: env NEMOTRON_TEMPERATURE or 0.7)
            max_tokens: Maximum tokens to generate (default: env NEMOTRON_MAX_TOKENS or 2048)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv(
            "NEMOTRON_BASE_URL",
            "http://localhost:8000/v1"
        )
        self.model = model or os.getenv(
            "NEMOTRON_MODEL",
            "nvidia/nvidia-nemotron-nano-9b-v2"
        )
        self.temperature = temperature or float(os.getenv("NEMOTRON_TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens or int(os.getenv("NEMOTRON_MAX_TOKENS", "2048"))
        self.client = httpx.AsyncClient(timeout=timeout)

        logger.info(
            f"Initialized Nemotron LLM: {self.base_url}, "
            f"model={self.model}, temp={self.temperature}"
        )

    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> CompletionResult:
        """
        Generate completion for a prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            stop: List of stop sequences
            **kwargs: Additional parameters for the API

        Returns:
            CompletionResult object
        """
        start_time = time.time()

        try:
            response = await self.client.post(
                f"{self.base_url}/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature or self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                    "stop": stop,
                    **kwargs
                }
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            choice = data["choices"][0]
            usage = data.get("usage", {})

            return CompletionResult(
                text=choice["text"],
                model=data.get("model", self.model),
                tokens_generated=usage.get("completion_tokens", 0),
                tokens_prompt=usage.get("prompt_tokens", 0),
                finish_reason=choice.get("finish_reason", "unknown"),
                latency_ms=latency_ms
            )

        except httpx.HTTPError as e:
            logger.error(f"Completion request failed: {e}")
            raise

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> CompletionResult:
        """
        Generate chat completion.

        Args:
            messages: List of ChatMessage objects
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            stop: List of stop sequences
            **kwargs: Additional parameters for the API

        Returns:
            CompletionResult object
        """
        start_time = time.time()

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": msg.role, "content": msg.content}
                        for msg in messages
                    ],
                    "temperature": temperature or self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                    "stop": stop,
                    **kwargs
                }
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            choice = data["choices"][0]
            message = choice["message"]
            usage = data.get("usage", {})

            return CompletionResult(
                text=message["content"],
                model=data.get("model", self.model),
                tokens_generated=usage.get("completion_tokens", 0),
                tokens_prompt=usage.get("prompt_tokens", 0),
                finish_reason=choice.get("finish_reason", "unknown"),
                latency_ms=latency_ms
            )

        except httpx.HTTPError as e:
            logger.error(f"Chat request failed: {e}")
            raise

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream chat completion.

        Args:
            messages: List of ChatMessage objects
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            **kwargs: Additional parameters for the API

        Yields:
            Token strings as they are generated
        """
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": msg.role, "content": msg.content}
                        for msg in messages
                    ],
                    "temperature": temperature or self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                    "stream": True,
                    **kwargs
                }
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str == "[DONE]":
                            break

                        try:
                            import json
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]

                            if "content" in delta:
                                yield delta["content"]

                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPError as e:
            logger.error(f"Stream request failed: {e}")
            raise

    async def summarize(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise"
    ) -> str:
        """
        Summarize text (convenience method).

        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            style: Summary style ('concise', 'detailed', 'bullet_points')

        Returns:
            Summary text
        """
        style_prompts = {
            "concise": "Provide a brief, concise summary.",
            "detailed": "Provide a detailed summary covering key points.",
            "bullet_points": "Summarize using bullet points for key information."
        }

        prompt = f"""
{style_prompts.get(style, style_prompts["concise"])}

Text to summarize:
{text}

Summary (max {max_length} words):
"""

        result = await self.complete(prompt, max_tokens=max_length * 2)
        return result.text.strip()

    async def extract_information(
        self,
        text: str,
        fields: List[str]
    ) -> Dict[str, str]:
        """
        Extract specific information fields from text.

        Args:
            text: Source text
            fields: List of field names to extract

        Returns:
            Dictionary mapping field names to extracted values
        """
        fields_str = ", ".join(fields)

        prompt = f"""
Extract the following information from the text:
{fields_str}

Text:
{text}

Provide the extracted information as JSON in this format:
{{
  "field1": "value1",
  "field2": "value2"
}}
"""

        result = await self.complete(prompt, max_tokens=500)

        # Parse JSON response
        import json
        try:
            extracted = json.loads(result.text)
            return extracted
        except json.JSONDecodeError:
            logger.warning("Failed to parse extraction result as JSON")
            return {field: "" for field in fields}

    async def classify(
        self,
        text: str,
        categories: List[str],
        return_confidence: bool = False
    ) -> str | Dict[str, float]:
        """
        Classify text into one of the given categories.

        Args:
            text: Text to classify
            categories: List of category names
            return_confidence: If True, return confidence scores for all categories

        Returns:
            Category name (str) or dict of category->confidence if return_confidence=True
        """
        categories_str = "\n".join([f"- {cat}" for cat in categories])

        prompt = f"""
Classify the following text into ONE of these categories:

{categories_str}

Text:
{text}

{"Return the category name and confidence scores (0-1) for each category as JSON." if return_confidence else "Return only the category name."}
"""

        result = await self.complete(prompt, max_tokens=100)
        response_text = result.text.strip()

        if return_confidence:
            import json
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: return category with 1.0 confidence
                return {response_text: 1.0}
        else:
            return response_text

    async def batch_process(
        self,
        prompts: List[str],
        max_concurrent: int = 5,
        **kwargs
    ) -> List[CompletionResult]:
        """
        Process multiple prompts in parallel.

        Args:
            prompts: List of prompts to process
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters for completion

        Returns:
            List of CompletionResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(prompt: str) -> CompletionResult:
            async with semaphore:
                return await self.complete(prompt, **kwargs)

        tasks = [process_one(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}")
                # Create a placeholder result
                valid_results.append(CompletionResult(
                    text="",
                    model=self.model,
                    tokens_generated=0,
                    tokens_prompt=0,
                    finish_reason="error",
                    latency_ms=0.0
                ))
            else:
                valid_results.append(result)

        return valid_results

    async def health_check(self) -> bool:
        """
        Check if the service is healthy and responding.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Convenience function for quick usage
async def create_nemotron_service() -> NemotronLLM:
    """Create a Nemotron LLM service with environment configuration."""
    return NemotronLLM()

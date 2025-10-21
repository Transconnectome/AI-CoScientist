"""Simple LLM service for MCP server."""

import os
import anthropic
from typing import Dict, Any


class SimpleLLMService:
    """Simplified LLM service for MCP server usage."""

    def __init__(self):
        """Initialize with Anthropic API."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5"  # Claude Sonnet 4.5 (Sep 2025)

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate completion using Anthropic API.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with content and metadata
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                "content": response.content[0].text,
                "model": response.model,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "finish_reason": response.stop_reason
            }
        except Exception as e:
            raise RuntimeError(f"LLM completion failed: {str(e)}")

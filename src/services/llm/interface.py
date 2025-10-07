"""LLM service interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, List

from src.services.llm.types import LLMRequest, LLMResponse


class LLMServiceInterface(ABC):
    """Abstract interface for LLM service."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from LLM."""
        pass

    @abstractmethod
    async def stream_complete(
        self,
        request: LLMRequest
    ) -> AsyncIterator[str]:
        """Stream completion from LLM."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings."""
        pass

    @abstractmethod
    def get_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for token usage."""
        pass

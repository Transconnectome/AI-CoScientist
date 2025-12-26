from abc import ABC, abstractmethod
from typing import AsyncIterator, List

from src.services.llm.types import LLMRequest, LLMResponse

class LLMServiceInterface(ABC):
    """Abstract base class for LLM service adapters."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a complete response."""
        pass

    @abstractmethod
    async def stream_complete(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream the response."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings."""
        pass

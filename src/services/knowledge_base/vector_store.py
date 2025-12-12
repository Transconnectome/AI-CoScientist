"""Minimal vector store stub for testing Foundation Models."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass


class VectorStoreManager:
    """Vector store manager for testing."""

    def __init__(self):
        self.vector_store: Optional[VectorStore] = None

    def get_vector_store(self, collection_name: str = "default") -> Optional[VectorStore]:
        """Get vector store instance."""
        return self.vector_store

    async def initialize(self) -> None:
        """Initialize vector store (placeholder)."""
        pass
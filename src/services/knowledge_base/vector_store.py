"""Vector store service using ChromaDB."""

from typing import Any, Dict, List, Optional
from uuid import UUID
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings


class VectorStore:
    """ChromaDB vector store wrapper."""

    def __init__(self):
        """Initialize vector store."""
        self.client = chromadb.HttpClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection_name = settings.chromadb_collection

    def get_or_create_collection(self, name: str | None = None):
        """Get or create a collection."""
        collection_name = name or self.collection_name
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        collection_name: str | None = None
    ) -> None:
        """Add documents to the collection."""
        collection = self.get_or_create_collection(collection_name)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        collection_name: str | None = None
    ) -> Dict[str, Any]:
        """Query the collection."""
        collection = self.get_or_create_collection(collection_name)
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )

    def get_document(
        self,
        document_id: str,
        collection_name: str | None = None
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        collection = self.get_or_create_collection(collection_name)
        result = collection.get(ids=[document_id])

        if not result['ids']:
            return None

        return {
            'id': result['ids'][0],
            'document': result['documents'][0] if result['documents'] else None,
            'metadata': result['metadatas'][0] if result['metadatas'] else None,
            'embedding': result['embeddings'][0] if result['embeddings'] else None
        }

    def update_document(
        self,
        document_id: str,
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: str | None = None
    ) -> None:
        """Update a document."""
        collection = self.get_or_create_collection(collection_name)
        collection.update(
            ids=[document_id],
            documents=[document] if document else None,
            embeddings=[embedding] if embedding else None,
            metadatas=[metadata] if metadata else None
        )

    def delete_document(
        self,
        document_id: str,
        collection_name: str | None = None
    ) -> None:
        """Delete a document."""
        collection = self.get_or_create_collection(collection_name)
        collection.delete(ids=[document_id])

    def count_documents(
        self,
        collection_name: str | None = None
    ) -> int:
        """Count documents in collection."""
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    def reset_collection(
        self,
        collection_name: str | None = None
    ) -> None:
        """Reset (delete all documents) in collection."""
        collection_name = collection_name or self.collection_name
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        self.get_or_create_collection(collection_name)

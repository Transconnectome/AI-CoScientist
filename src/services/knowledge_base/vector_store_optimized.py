"""Optimized vector store with connection pooling and batching."""

from typing import Any, Dict, List, Optional
import asyncio
from contextlib import asynccontextmanager
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings


class VectorStoreConnectionPool:
    """Connection pool for ChromaDB clients."""

    def __init__(
        self,
        max_connections: int = 10,
        host: str | None = None,
        port: int | None = None
    ):
        """Initialize connection pool.

        Args:
            max_connections: Maximum number of connections in pool
            host: ChromaDB host
            port: ChromaDB port
        """
        self.max_connections = max_connections
        self.host = host or settings.chromadb_host
        self.port = port or settings.chromadb_port

        self._pool: List[chromadb.HttpClient] = []
        self._available: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._lock = asyncio.Lock()
        self._created_connections = 0

    async def get_client(self) -> chromadb.HttpClient:
        """Get a client from the pool.

        Returns:
            ChromaDB HTTP client
        """
        # Try to get available client
        try:
            client = self._available.get_nowait()
            return client
        except asyncio.QueueEmpty:
            pass

        # Create new client if under limit
        async with self._lock:
            if self._created_connections < self.max_connections:
                client = self._create_client()
                self._pool.append(client)
                self._created_connections += 1
                return client

        # Wait for available client
        return await self._available.get()

    async def return_client(self, client: chromadb.HttpClient) -> None:
        """Return a client to the pool.

        Args:
            client: Client to return
        """
        await self._available.put(client)

    def _create_client(self) -> chromadb.HttpClient:
        """Create new ChromaDB client.

        Returns:
            ChromaDB HTTP client
        """
        return chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    async def close_all(self) -> None:
        """Close all connections in pool."""
        # Note: ChromaDB HttpClient doesn't have explicit close,
        # but we clear the pool
        self._pool.clear()
        while not self._available.empty():
            self._available.get_nowait()
        self._created_connections = 0


class OptimizedVectorStore:
    """Optimized ChromaDB vector store with connection pooling."""

    def __init__(
        self,
        connection_pool: Optional[VectorStoreConnectionPool] = None,
        collection_name: str | None = None
    ):
        """Initialize optimized vector store.

        Args:
            connection_pool: Connection pool for ChromaDB clients
            collection_name: Name of the collection
        """
        self.connection_pool = connection_pool or VectorStoreConnectionPool()
        self.collection_name = collection_name or settings.chromadb_collection

        # Performance metrics
        self._query_count = 0
        self._add_count = 0
        self._total_query_time = 0.0

    @asynccontextmanager
    async def _get_client(self):
        """Context manager for getting and returning client."""
        client = await self.connection_pool.get_client()
        try:
            yield client
        finally:
            await self.connection_pool.return_client(client)

    async def get_or_create_collection(
        self,
        name: str | None = None,
        client: Optional[chromadb.HttpClient] = None
    ):
        """Get or create a collection.

        Args:
            name: Collection name
            client: Optional client to use

        Returns:
            ChromaDB collection
        """
        collection_name = name or self.collection_name

        if client:
            return client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        async with self._get_client() as client:
            return client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        collection_name: str | None = None,
        batch_size: int = 100
    ) -> None:
        """Add documents to the collection in batches.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            collection_name: Collection name
            batch_size: Batch size for adding documents
        """
        async with self._get_client() as client:
            collection = await self.get_or_create_collection(
                collection_name,
                client
            )

            # Process in batches to avoid overwhelming ChromaDB
            total = len(documents)
            for i in range(0, total, batch_size):
                end = min(i + batch_size, total)

                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: collection.add(
                        documents=documents[i:end],
                        embeddings=embeddings[i:end],
                        metadatas=metadatas[i:end],
                        ids=ids[i:end]
                    )
                )

                self._add_count += (end - i)

    async def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        collection_name: str | None = None
    ) -> Dict[str, Any]:
        """Query the collection.

        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return
            where: Optional metadata filter
            collection_name: Collection name

        Returns:
            Query results
        """
        import time
        start_time = time.time()

        async with self._get_client() as client:
            collection = await self.get_or_create_collection(
                collection_name,
                client
            )

            # Run query in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where
                )
            )

        # Update metrics
        elapsed = time.time() - start_time
        self._query_count += 1
        self._total_query_time += elapsed

        return result

    async def query_batch(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        collection_name: str | None = None
    ) -> List[Dict[str, Any]]:
        """Query with multiple embeddings efficiently.

        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results per query
            where: Optional metadata filter
            collection_name: Collection name

        Returns:
            List of query results
        """
        # Process queries in parallel with connection pool
        tasks = [
            self.query([emb], n_results, where, collection_name)
            for emb in query_embeddings
        ]

        results = await asyncio.gather(*tasks)
        return results

    async def get_document(
        self,
        document_id: str,
        collection_name: str | None = None
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID.

        Args:
            document_id: Document ID
            collection_name: Collection name

        Returns:
            Document data or None
        """
        async with self._get_client() as client:
            collection = await self.get_or_create_collection(
                collection_name,
                client
            )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: collection.get(ids=[document_id])
            )

            if not result['ids']:
                return None

            return {
                'id': result['ids'][0],
                'document': result['documents'][0] if result['documents'] else None,
                'metadata': result['metadatas'][0] if result['metadatas'] else None,
                'embedding': result['embeddings'][0] if result['embeddings'] else None
            }

    async def update_document(
        self,
        document_id: str,
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: str | None = None
    ) -> None:
        """Update a document.

        Args:
            document_id: Document ID
            document: Document text
            embedding: Embedding vector
            metadata: Metadata dictionary
            collection_name: Collection name
        """
        async with self._get_client() as client:
            collection = await self.get_or_create_collection(
                collection_name,
                client
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: collection.update(
                    ids=[document_id],
                    documents=[document] if document else None,
                    embeddings=[embedding] if embedding else None,
                    metadatas=[metadata] if metadata else None
                )
            )

    async def delete_document(
        self,
        document_id: str,
        collection_name: str | None = None
    ) -> None:
        """Delete a document.

        Args:
            document_id: Document ID
            collection_name: Collection name
        """
        async with self._get_client() as client:
            collection = await self.get_or_create_collection(
                collection_name,
                client
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: collection.delete(ids=[document_id])
            )

    async def count_documents(
        self,
        collection_name: str | None = None
    ) -> int:
        """Count documents in collection.

        Args:
            collection_name: Collection name

        Returns:
            Number of documents
        """
        async with self._get_client() as client:
            collection = await self.get_or_create_collection(
                collection_name,
                client
            )

            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                None,
                lambda: collection.count()
            )

            return count

    async def reset_collection(
        self,
        collection_name: str | None = None
    ) -> None:
        """Reset (delete all documents) in collection.

        Args:
            collection_name: Collection name
        """
        collection_name = collection_name or self.collection_name

        async with self._get_client() as client:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: client.delete_collection(name=collection_name)
                )
            except Exception:
                pass

            await self.get_or_create_collection(collection_name, client)

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary with statistics
        """
        avg_query_time = (
            self._total_query_time / self._query_count
            if self._query_count > 0
            else 0.0
        )

        return {
            "query_count": self._query_count,
            "add_count": self._add_count,
            "total_query_time": self._total_query_time,
            "avg_query_time_ms": avg_query_time * 1000,
            "connection_pool_size": self.connection_pool._created_connections,
            "max_connections": self.connection_pool.max_connections
        }

    async def close(self) -> None:
        """Close connection pool."""
        await self.connection_pool.close_all()

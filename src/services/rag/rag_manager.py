"""
RAG Manager for Paper Improvement

Manages ChromaDB collections for storing and retrieving
successful improvement patterns.
"""

import os
import time
import subprocess
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.services.embeddings import EmbeddingService
from src.services.knowledge_base.vector_store import VectorStore
from src.services.rag.graph_index_store import GraphIndexStore
from src.services.rag.graph_seed_selector import GraphSeedSelector
from src.services.rag.graph_rag_pipeline import GraphRAGPipeline, GraphRAGPipelineResult
from src.services.rag.multi_agent_orchestrator import (
    AgentDefinition,
    MultiAgentOrchestrator,
)


class RAGManager:
    """Manages RAG system for paper improvement."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chromadb_mode: str = "auto",
        chromadb_host: str = "localhost",
        chromadb_port: int = 8001,
        chromadb_path: str = "./chromadb_data",
        graph_store: Optional[GraphIndexStore] = None,
        graph_seed_selector: Optional[GraphSeedSelector] = None,
        graph_orchestrator: Optional[MultiAgentOrchestrator] = None,
        graph_pipeline: Optional[GraphRAGPipeline] = None,
    ):
        """
        Initialize RAG Manager.

        Args:
            embedding_service: Service for generating embeddings
            chromadb_mode: "auto", "docker", "local", or "disabled"
            chromadb_host: ChromaDB host
            chromadb_port: ChromaDB port
            chromadb_path: Path for local persistent storage
        """
        self.embedding_service = embedding_service
        self.chromadb_mode = chromadb_mode
        self.chromadb_host = chromadb_host
        self.chromadb_port = chromadb_port
        self.chromadb_path = chromadb_path

        self.client = None
        self.client_type = None
        self.enabled = False

        # Graph-based RAG components
        self.graph_store = graph_store
        if graph_seed_selector is not None:
            self.graph_seed_selector = graph_seed_selector
        elif graph_store is not None:
            self.graph_seed_selector = GraphSeedSelector(graph_store)
        else:
            self.graph_seed_selector = None

        self.graph_orchestrator = graph_orchestrator

        if graph_pipeline is not None:
            self.graph_pipeline = graph_pipeline
        elif self.graph_store and self.graph_seed_selector and self.graph_orchestrator:
            self.graph_pipeline = GraphRAGPipeline(
                graph_store=self.graph_store,
                seed_selector=self.graph_seed_selector,
                orchestrator=self.graph_orchestrator,
            )
        else:
            self.graph_pipeline = None

        # Collection names
        self.IMPROVEMENT_PATTERNS = "improvement_patterns"
        self.SUCCESSFUL_PHRASES = "successful_phrases"
        self.SECTION_TEMPLATES = "section_templates"

        # Initialize ChromaDB
        self._initialize_chromadb()

    def _initialize_chromadb(self):
        """Initialize ChromaDB client with fallback strategy."""
        if self.chromadb_mode == "disabled":
            print("⚠️  RAG disabled by configuration")
            return

        # Try HTTP first (Docker/Server)
        if self.chromadb_mode in ["auto", "docker"]:
            if self._try_http_client():
                return

            # Try to start Docker
            if self.chromadb_mode == "auto":
                if self._start_docker_chromadb():
                    if self._try_http_client():
                        return

        # Fallback to local persistent
        if self.chromadb_mode in ["auto", "local"]:
            if self._try_local_client():
                return

        print("⚠️  ChromaDB not available, RAG features disabled")

    def _try_http_client(self) -> bool:
        """Try to connect to HTTP ChromaDB."""
        try:
            client = chromadb.HttpClient(
                host=self.chromadb_host,
                port=self.chromadb_port,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            client.heartbeat()

            self.client = client
            self.client_type = "http"
            self.enabled = True
            print(f"✅ Connected to ChromaDB at {self.chromadb_host}:{self.chromadb_port}")
            return True

        except Exception as e:
            print(f"   ChromaDB HTTP not available: {e}")
            return False

    def _start_docker_chromadb(self) -> bool:
        """Try to start ChromaDB via Docker."""
        try:
            print("🐳 Starting ChromaDB Docker container...")

            # Check if container already exists
            check_result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=ai-coscientist-chromadb", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )

            if "ai-coscientist-chromadb" in check_result.stdout:
                # Container exists, try to start it
                subprocess.run(
                    ["docker", "start", "ai-coscientist-chromadb"],
                    capture_output=True
                )
            else:
                # Create new container
                subprocess.run([
                    "docker", "run", "-d",
                    "--name", "ai-coscientist-chromadb",
                    "-p", f"{self.chromadb_port}:8000",
                    "chromadb/chroma"
                ], capture_output=True)

            # Wait for startup
            time.sleep(3)
            print("✅ ChromaDB Docker container started")
            return True

        except Exception as e:
            print(f"   Failed to start Docker: {e}")
            return False

    def _try_local_client(self) -> bool:
        """Try to use local persistent ChromaDB."""
        try:
            # Create directory if needed
            os.makedirs(self.chromadb_path, exist_ok=True)

            client = chromadb.PersistentClient(
                path=self.chromadb_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            self.client = client
            self.client_type = "local"
            self.enabled = True
            print(f"✅ Using local ChromaDB at {self.chromadb_path}")
            return True

        except Exception as e:
            print(f"   Failed to create local ChromaDB: {e}")
            return False

    def is_enabled(self) -> bool:
        """Check if RAG is enabled."""
        return self.enabled

    async def run_graph_rag(
        self,
        query: str,
        agents: List[AgentDefinition],
        seed_limit: int = 5,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> GraphRAGPipelineResult:
        """Execute graph-aware multi-agent RAG pipeline."""

        if self.graph_pipeline is None:
            raise RuntimeError("Graph RAG pipeline is not configured")

        return await self.graph_pipeline.run(
            query=query,
            agents=agents,
            seed_limit=seed_limit,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

    async def store_improvement_pattern(
        self,
        paper_id: str,
        section: str,
        before_text: str,
        after_text: str,
        scores_before: Dict[str, float],
        scores_after: Dict[str, float],
        strategy: str,
        field: str = "unknown",
        paper_type: str = "unknown"
    ) -> str:
        """
        Store a successful improvement pattern.

        Args:
            paper_id: Paper identifier
            section: Section name (Abstract, Introduction, etc.)
            before_text: Original text
            after_text: Improved text
            scores_before: Scores before improvement
            scores_after: Scores after improvement
            strategy: Strategy applied
            field: Research field
            paper_type: Paper type

        Returns:
            Improvement ID
        """
        if not self.enabled:
            return ""

        improvement_id = str(uuid4())

        # Generate embedding for improved text
        embedding = await self.embedding_service.embed_text(after_text)

        # Prepare metadata
        metadata = {
            "improvement_id": improvement_id,
            "paper_id": paper_id,
            "section": section,
            "timestamp": datetime.now().isoformat(),

            # Scores
            "before_overall": scores_before.get("overall_quality", 0),
            "after_overall": scores_after.get("overall_quality", 0),
            "before_novelty": scores_before.get("novelty", 0),
            "after_novelty": scores_after.get("novelty", 0),
            "before_methodology": scores_before.get("methodology", 0),
            "after_methodology": scores_after.get("methodology", 0),
            "before_clarity": scores_before.get("clarity", 0),
            "after_clarity": scores_after.get("clarity", 0),
            "before_significance": scores_before.get("significance", 0),
            "after_significance": scores_after.get("significance", 0),

            # Improvement metrics
            "overall_gain": scores_after.get("overall_quality", 0) - scores_before.get("overall_quality", 0),
            "clarity_gain": scores_after.get("clarity", 0) - scores_before.get("clarity", 0),
            "novelty_gain": scores_after.get("novelty", 0) - scores_before.get("novelty", 0),

            # Strategy
            "strategy_applied": strategy,
            "field": field,
            "paper_type": paper_type
        }

        # Store in ChromaDB
        collection = self.client.get_or_create_collection(
            name=self.IMPROVEMENT_PATTERNS,
            metadata={"hnsw:space": "cosine"}
        )

        collection.add(
            documents=[after_text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[improvement_id]
        )

        print(f"✅ Stored improvement pattern: {improvement_id}")
        return improvement_id

    async def search_similar_improvements(
        self,
        section: str,
        problem_description: str,
        current_scores: Dict[str, float],
        n_results: int = 3
    ) -> List[Dict]:
        """
        Search for similar successful improvements.

        Args:
            section: Section to improve
            problem_description: Description of the problem
            current_scores: Current scores
            n_results: Number of results to return

        Returns:
            List of similar improvement patterns
        """
        if not self.enabled:
            return []

        # Generate query embedding
        query_text = f"{section} section: {problem_description}"
        query_embedding = await self.embedding_service.embed_text(query_text)

        # Build where filter
        where_filter = {"section": section}

        # Add score filters (find improvements with similar starting scores)
        # This helps find relevant patterns

        # Search collection
        try:
            collection = self.client.get_or_create_collection(
                name=self.IMPROVEMENT_PATTERNS,
                metadata={"hnsw:space": "cosine"}
            )

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )

            # Format results
            patterns = []
            for i in range(len(results['ids'][0])):
                pattern = {
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                patterns.append(pattern)

            return patterns

        except Exception as e:
            print(f"⚠️  Error searching improvements: {e}")
            return []

    def format_rag_context(self, patterns: List[Dict]) -> str:
        """
        Format RAG search results as context for Claude.

        Args:
            patterns: List of improvement patterns

        Returns:
            Formatted context string
        """
        if not patterns:
            return ""

        context = "\n\n📚 SIMILAR SUCCESSFUL IMPROVEMENTS:\n\n"

        for i, pattern in enumerate(patterns, 1):
            meta = pattern['metadata']

            context += f"**Example {i}:**\n"
            context += f"Strategy: {meta.get('strategy_applied', 'unknown')}\n"
            context += f"Score improvement: {meta.get('before_overall', 0):.2f} → {meta.get('after_overall', 0):.2f} "
            context += f"(+{meta.get('overall_gain', 0):.2f})\n"

            # Show specific improvements
            if meta.get('clarity_gain', 0) > 0.1:
                context += f"  - Clarity: {meta.get('before_clarity', 0):.2f} → {meta.get('after_clarity', 0):.2f}\n"
            if meta.get('novelty_gain', 0) > 0.1:
                context += f"  - Novelty: {meta.get('before_novelty', 0):.2f} → {meta.get('after_novelty', 0):.2f}\n"

            # Show excerpt of improved text
            text = pattern['text']
            excerpt = text[:300] + "..." if len(text) > 300 else text
            context += f"\nImproved text excerpt:\n{excerpt}\n\n"

        context += "Use these successful patterns as inspiration for your improvement strategy.\n"
        return context

    def get_statistics(self) -> Dict:
        """Get RAG system statistics."""
        if not self.enabled:
            return {"enabled": False}

        try:
            collection = self.client.get_or_create_collection(
                name=self.IMPROVEMENT_PATTERNS
            )

            return {
                "enabled": True,
                "client_type": self.client_type,
                "total_patterns": collection.count(),
                "embedding_model": self.embedding_service.get_model_info()
            }

        except Exception as e:
            return {
                "enabled": True,
                "error": str(e)
            }

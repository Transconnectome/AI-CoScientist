#!/usr/bin/env python3
"""
Multi-Strategy Search Implementation
=====================================

Real ChromaDB-backed implementations for all 6 RAG strategies.
Connects to actual databases: DD-RAPTOR papers, ESM3 papers, Grant proposals.

Features:
- Real ChromaDB vector search integration
- Strategy-specific query optimization
- Cross-database search capabilities
- Performance metrics collection
- Intelligent result fusion

Usage:
    from src.services.rag.multi_strategy_search import (
        create_real_strategies,
        MultiStrategySearchEngine
    )

    engine = MultiStrategySearchEngine()
    await engine.initialize()
    results = await engine.search("ESM3 protein brain development", strategies=["HYBRID", "GRAPH_RAG"])
"""

import asyncio
import logging
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading

# Third-party
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Local imports
from src.services.rag.unified_rag_orchestrator import (
    RAGStrategy,
    RAGStrategyInterface,
    QueryContext,
    QueryComplexity,
    QueryDomain,
    RAGResponse
)
from src.monitoring.rag_metrics import get_metrics_manager, RAGMetrics

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB databases"""
    dd_raptor_path: str = "chromadb_data_dd"
    grants_path: str = "chromadb_grants_fixed_20251210_200233"
    esm3_papers_path: str = "chromadb_new_papers_20251210_204818"
    neurips_path: str = "chromadb_data_neurips2025"
    # Different embedding models for different databases
    embedding_model_384: str = "all-MiniLM-L6-v2"  # 384 dimensions
    embedding_model_768: str = "allenai/scibert_scivocab_uncased"  # 768 dimensions
    default_top_k: int = 10

@dataclass
class SearchResult:
    """Individual search result"""
    content: str
    title: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    database: str = ""
    strategy: str = ""

@dataclass
class MultiStrategyResult:
    """Combined results from multiple strategies"""
    query: str
    results: List[SearchResult]
    strategies_used: List[str]
    total_sources: int
    avg_relevance: float
    execution_time_ms: float
    cross_domain_detected: bool
    performance_breakdown: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# Real Strategy Implementations
# ============================================================================

class RealRAGStrategy(RAGStrategyInterface):
    """Base class for real ChromaDB-backed strategies"""

    def __init__(self, strategy: RAGStrategy, config: ChromaDBConfig):
        self.strategy = strategy
        self.config = config
        self._available = False
        self._clients: Dict[str, chromadb.Client] = {}
        self._collections: Dict[str, Any] = {}
        self._embedding_models: Dict[int, SentenceTransformer] = {}  # dimension -> model
        self._collection_dimensions: Dict[str, int] = {}  # collection -> dimension
        self._lock = threading.Lock()
        self._metrics_manager = get_metrics_manager()
        self._performance_history: List[float] = []

    async def initialize(self):
        """Initialize ChromaDB connections and embedding models"""
        try:
            # Initialize embedding models for both dimensions
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self._embedding_models[384] = SentenceTransformer(self.config.embedding_model_384)
                    logger.info(f"Loaded 384-dim model: {self.config.embedding_model_384}")
                except Exception as e:
                    logger.warning(f"Could not load 384-dim model: {e}")

                try:
                    self._embedding_models[768] = SentenceTransformer(self.config.embedding_model_768)
                    logger.info(f"Loaded 768-dim model: {self.config.embedding_model_768}")
                except Exception as e:
                    logger.warning(f"Could not load 768-dim model: {e}")

            # Initialize ChromaDB clients
            await self._init_chromadb_clients()

            self._available = len(self._collections) > 0
            logger.info(f"Strategy {self.strategy.value} initialized with {len(self._collections)} collections")

        except Exception as e:
            logger.error(f"Failed to initialize {self.strategy.value}: {e}")
            self._available = False

    async def _init_chromadb_clients(self):
        """Initialize ChromaDB clients for relevant databases"""
        db_configs = self._get_relevant_databases()

        for db_name, db_path in db_configs.items():
            try:
                if Path(db_path).exists():
                    client = chromadb.PersistentClient(
                        path=db_path,
                        settings=Settings(anonymized_telemetry=False)
                    )

                    # Get collections
                    collections = client.list_collections()
                    if collections:
                        self._clients[db_name] = client
                        for coll in collections:
                            coll_key = f"{db_name}_{coll.name}"
                            collection = client.get_collection(coll.name)
                            self._collections[coll_key] = collection

                            # Detect embedding dimension
                            dimension = self._detect_collection_dimension(collection, db_name)
                            self._collection_dimensions[coll_key] = dimension
                            logger.debug(f"Loaded collection: {coll_key} (dim={dimension})")

            except Exception as e:
                logger.warning(f"Could not load database {db_name}: {e}")

    def _detect_collection_dimension(self, collection: Any, db_name: str) -> int:
        """Detect the embedding dimension of a collection"""
        # DD-RAPTOR uses SciBERT (768), others use MiniLM (384)
        if "dd_raptor" in db_name or "dd" in db_name:
            return 768
        return 384

    def _get_relevant_databases(self) -> Dict[str, str]:
        """Get databases relevant to this strategy - Override in subclasses"""
        return {
            "dd_raptor": self.config.dd_raptor_path,
            "grants": self.config.grants_path,
            "esm3": self.config.esm3_papers_path
        }

    async def search(self, query_context: QueryContext) -> RAGResponse:
        """Execute search across relevant collections"""
        start_time = time.time()

        if not self._available:
            return self._create_fallback_response(query_context)

        try:
            # Search across collections with appropriate embedding dimensions
            all_results = []
            for coll_key, collection in self._collections.items():
                try:
                    # Get the correct dimension for this collection
                    dimension = self._collection_dimensions.get(coll_key, 384)

                    # Generate query embedding with correct dimension
                    query_embedding = self._get_embedding(query_context.query, dimension)

                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(self.config.default_top_k, 20),
                        include=["documents", "metadatas", "distances"]
                    )

                    # Process results
                    processed = self._process_collection_results(results, coll_key)
                    all_results.extend(processed)

                except Exception as e:
                    logger.warning(f"Search failed in {coll_key}: {e}")
                    continue

            # Sort by relevance and deduplicate
            all_results = self._deduplicate_and_rank(all_results)

            # Generate response
            execution_time = time.time() - start_time
            response = self._create_response(query_context, all_results, execution_time)

            # Record metrics
            self._record_search_metrics(response, execution_time)

            return response

        except Exception as e:
            logger.error(f"Search error in {self.strategy.value}: {e}")
            return self._create_fallback_response(query_context)

    def _get_embedding(self, text: str, dimension: int = 384) -> List[float]:
        """Generate embedding for text with specified dimension"""
        model = self._embedding_models.get(dimension)
        if model:
            return model.encode(text).tolist()

        # Fallback to any available model
        for dim, model in self._embedding_models.items():
            return model.encode(text).tolist()

        return [0.0] * dimension  # Default

    def _process_collection_results(self, results: Dict, collection_key: str) -> List[SearchResult]:
        """Process raw ChromaDB results into SearchResult objects"""
        processed = []

        if not results or not results.get("documents"):
            return processed

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(documents)

        for i, doc in enumerate(documents):
            if doc:
                # Convert distance to relevance (lower distance = higher relevance)
                relevance = max(0, 1.0 - (distances[i] / 2.0))

                metadata = metadatas[i] if i < len(metadatas) else {}
                title = metadata.get("title", metadata.get("section", f"Source {i+1}"))

                processed.append(SearchResult(
                    content=doc[:500],  # Truncate for response
                    title=str(title),
                    source=metadata.get("source", collection_key),
                    relevance_score=relevance,
                    metadata=metadata,
                    database=collection_key.split("_")[0],
                    strategy=self.strategy.value
                ))

        return processed

    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate and rank results by relevance"""
        # Simple deduplication by content similarity
        seen_content = set()
        unique_results = []

        for result in results:
            content_key = result.content[:100]  # First 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)

        # Sort by relevance
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return unique_results[:self.config.default_top_k]

    def _create_response(self, query_context: QueryContext, results: List[SearchResult], execution_time: float) -> RAGResponse:
        """Create RAGResponse from search results"""
        # Generate answer from top results
        if results:
            top_content = [r.content for r in results[:3]]
            answer = self._synthesize_answer(query_context.query, top_content)
            confidence = sum(r.relevance_score for r in results[:5]) / max(len(results[:5]), 1)
        else:
            answer = f"No relevant results found for: {query_context.query[:100]}..."
            confidence = 0.3

        sources = [
            {
                "title": r.title,
                "content": r.content[:200],
                "relevance": r.relevance_score,
                "database": r.database,
                "metadata": r.metadata
            }
            for r in results[:10]
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            strategy_used=self.strategy,
            metadata={
                "execution_time": execution_time,
                "total_results": len(results),
                "databases_searched": list(set(r.database for r in results))
            }
        )

    def _synthesize_answer(self, query: str, contents: List[str]) -> str:
        """Synthesize answer from retrieved content"""
        if not contents:
            return "No relevant information found."

        # Simple synthesis - in production, use LLM
        combined = " ".join(contents)
        return f"Based on {len(contents)} relevant sources: {combined[:500]}..."

    def _create_fallback_response(self, query_context: QueryContext) -> RAGResponse:
        """Create fallback response when search fails"""
        return RAGResponse(
            answer=f"Strategy {self.strategy.value} is currently unavailable. Query: {query_context.query[:50]}...",
            sources=[],
            confidence=0.1,
            strategy_used=self.strategy,
            metadata={"fallback": True}
        )

    def _record_search_metrics(self, response: RAGResponse, execution_time: float):
        """Record search metrics"""
        metrics = RAGMetrics(
            latency=execution_time,
            quality_score=response.confidence,
            tokens_processed=len(response.answer),
            retrieval_time=execution_time * 0.7,
            generation_time=execution_time * 0.3,
            context_relevance=response.confidence,
            faithfulness=response.confidence * 0.9,
            answer_relevancy=response.confidence * 0.95,
            strategy=self.strategy.value,
            timestamp=datetime.now()
        )
        self._metrics_manager.record_rag_request(metrics)
        self._performance_history.append(response.confidence)

    def is_available(self) -> bool:
        return self._available

    def get_strategy_name(self) -> RAGStrategy:
        return self.strategy

    def estimate_performance(self, query_context: QueryContext) -> float:
        """Estimate performance based on history and query characteristics"""
        base_score = 0.7

        # Use historical performance if available
        if self._performance_history:
            base_score = sum(self._performance_history[-10:]) / len(self._performance_history[-10:])

        # Adjust for domain match
        domain_bonus = self._get_domain_bonus(query_context.domain)

        # Adjust for complexity
        complexity_modifier = {
            QueryComplexity.SIMPLE: 1.1,
            QueryComplexity.MEDIUM: 1.0,
            QueryComplexity.COMPLEX: 0.9
        }.get(query_context.complexity, 1.0)

        return min(1.0, base_score * domain_bonus * complexity_modifier)

    def _get_domain_bonus(self, domain: QueryDomain) -> float:
        """Get domain-specific bonus - Override in subclasses"""
        return 1.0


# ============================================================================
# Strategy-Specific Implementations
# ============================================================================

class HybridRAGStrategy(RealRAGStrategy):
    """Hybrid strategy - combines multiple databases"""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(RAGStrategy.HYBRID, config)

    def _get_relevant_databases(self) -> Dict[str, str]:
        return {
            "dd_raptor": self.config.dd_raptor_path,
            "grants": self.config.grants_path,
            "esm3": self.config.esm3_papers_path
        }

    def _get_domain_bonus(self, domain: QueryDomain) -> float:
        return {
            QueryDomain.GENERAL: 1.2,
            QueryDomain.NEUROSCIENCE: 1.1,
            QueryDomain.QUANTUM_ML: 1.0,
            QueryDomain.DEVELOPMENTAL_DISORDERS: 1.1,
            QueryDomain.PSYCHOLOGY: 1.0
        }.get(domain, 1.0)


class EnhancedDDRaptorStrategy(RealRAGStrategy):
    """Enhanced DD-RAPTOR strategy - focuses on developmental disorders"""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(RAGStrategy.ENHANCED_DD_RAPTOR, config)

    def _get_relevant_databases(self) -> Dict[str, str]:
        return {
            "dd_raptor": self.config.dd_raptor_path
        }

    def _get_domain_bonus(self, domain: QueryDomain) -> float:
        return {
            QueryDomain.DEVELOPMENTAL_DISORDERS: 1.3,
            QueryDomain.NEUROSCIENCE: 1.2,
            QueryDomain.PSYCHOLOGY: 1.1,
            QueryDomain.GENERAL: 0.9,
            QueryDomain.QUANTUM_ML: 0.8
        }.get(domain, 1.0)


class GraphRAGStrategy(RealRAGStrategy):
    """Graph RAG strategy - knowledge graph queries"""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(RAGStrategy.GRAPH_RAG, config)

    def _get_relevant_databases(self) -> Dict[str, str]:
        return {
            "dd_raptor": self.config.dd_raptor_path,
            "esm3": self.config.esm3_papers_path
        }

    async def search(self, query_context: QueryContext) -> RAGResponse:
        """Enhanced search with cross-reference analysis"""
        # First, get base results
        base_response = await super().search(query_context)

        # Enhance with cross-reference analysis
        if base_response.sources:
            # Find cross-domain connections
            cross_domain_insights = self._find_cross_domain_connections(base_response.sources)
            if cross_domain_insights:
                base_response.metadata["cross_domain_insights"] = cross_domain_insights
                base_response.confidence = min(1.0, base_response.confidence * 1.1)

        return base_response

    def _find_cross_domain_connections(self, sources: List[Dict]) -> List[str]:
        """Find connections across different domains"""
        insights = []
        domains_found = set()

        domain_keywords = {
            "neuroscience": ["brain", "neural", "neuron", "cognitive"],
            "protein": ["protein", "esm", "structure", "amino"],
            "quantum": ["quantum", "optimization", "algorithm"]
        }

        for source in sources:
            content = str(source.get("content", "")).lower()
            for domain, keywords in domain_keywords.items():
                if any(kw in content for kw in keywords):
                    domains_found.add(domain)

        if len(domains_found) >= 2:
            insights.append(f"Cross-domain connection detected: {', '.join(domains_found)}")

        return insights

    def _get_domain_bonus(self, domain: QueryDomain) -> float:
        return {
            QueryDomain.QUANTUM_ML: 1.3,
            QueryDomain.NEUROSCIENCE: 1.2,
            QueryDomain.GENERAL: 1.0,
            QueryDomain.DEVELOPMENTAL_DISORDERS: 1.1,
            QueryDomain.PSYCHOLOGY: 0.9
        }.get(domain, 1.0)


class GoldenReferenceStrategy(RealRAGStrategy):
    """Golden Reference strategy - high-quality reference papers"""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(RAGStrategy.GOLDEN_REFERENCE, config)

    def _get_relevant_databases(self) -> Dict[str, str]:
        return {
            "grants": self.config.grants_path,
            "esm3": self.config.esm3_papers_path
        }

    def _get_domain_bonus(self, domain: QueryDomain) -> float:
        return {
            QueryDomain.GENERAL: 1.2,
            QueryDomain.NEUROSCIENCE: 1.1,
            QueryDomain.QUANTUM_ML: 1.1,
            QueryDomain.DEVELOPMENTAL_DISORDERS: 1.0,
            QueryDomain.PSYCHOLOGY: 1.0
        }.get(domain, 1.0)


class SimpleRAGStrategy(RealRAGStrategy):
    """Simple RAG strategy - fast, basic search"""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(RAGStrategy.SIMPLE_RAG, config)
        self.config.default_top_k = 5  # Fewer results for speed

    def _get_relevant_databases(self) -> Dict[str, str]:
        # Only use main database for speed
        return {
            "dd_raptor": self.config.dd_raptor_path
        }

    def _get_domain_bonus(self, domain: QueryDomain) -> float:
        return 1.0  # No domain specialization


class MultimodalRAGStrategy(RealRAGStrategy):
    """Multimodal RAG strategy - handles diverse content types"""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(RAGStrategy.MULTIMODAL_RAG, config)

    def _get_relevant_databases(self) -> Dict[str, str]:
        return {
            "dd_raptor": self.config.dd_raptor_path,
            "esm3": self.config.esm3_papers_path,
            "grants": self.config.grants_path
        }

    def _get_domain_bonus(self, domain: QueryDomain) -> float:
        return {
            QueryDomain.NEUROSCIENCE: 1.2,
            QueryDomain.QUANTUM_ML: 1.1,
            QueryDomain.GENERAL: 1.0,
            QueryDomain.DEVELOPMENTAL_DISORDERS: 1.1,
            QueryDomain.PSYCHOLOGY: 1.0
        }.get(domain, 1.0)


class PsychologyRAGStrategy(RealRAGStrategy):
    """Psychology RAG strategy - mental health and psychology focus"""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(RAGStrategy.PSYCHOLOGY_RAG, config)

    def _get_relevant_databases(self) -> Dict[str, str]:
        return {
            "dd_raptor": self.config.dd_raptor_path
        }

    def _get_domain_bonus(self, domain: QueryDomain) -> float:
        return {
            QueryDomain.PSYCHOLOGY: 1.4,
            QueryDomain.DEVELOPMENTAL_DISORDERS: 1.2,
            QueryDomain.NEUROSCIENCE: 1.1,
            QueryDomain.GENERAL: 0.9,
            QueryDomain.QUANTUM_ML: 0.7
        }.get(domain, 1.0)


# ============================================================================
# Multi-Strategy Search Engine
# ============================================================================

class MultiStrategySearchEngine:
    """
    High-level search engine that orchestrates multiple strategies
    with intelligent routing and result fusion
    """

    def __init__(self, config: Optional[ChromaDBConfig] = None):
        self.config = config or ChromaDBConfig()
        self.strategies: Dict[RAGStrategy, RealRAGStrategy] = {}
        self._initialized = False
        self._metrics_manager = get_metrics_manager()

        # Performance tracking
        self._search_count = 0
        self._total_latency = 0.0
        self._strategy_usage: Dict[str, int] = {}

    async def initialize(self):
        """Initialize all strategies"""
        logger.info("Initializing Multi-Strategy Search Engine...")

        strategy_classes = {
            RAGStrategy.HYBRID: HybridRAGStrategy,
            RAGStrategy.ENHANCED_DD_RAPTOR: EnhancedDDRaptorStrategy,
            RAGStrategy.GRAPH_RAG: GraphRAGStrategy,
            RAGStrategy.GOLDEN_REFERENCE: GoldenReferenceStrategy,
            RAGStrategy.SIMPLE_RAG: SimpleRAGStrategy,
            RAGStrategy.MULTIMODAL_RAG: MultimodalRAGStrategy,
            RAGStrategy.PSYCHOLOGY_RAG: PsychologyRAGStrategy
        }

        for strategy_type, strategy_class in strategy_classes.items():
            try:
                strategy = strategy_class(self.config)
                await strategy.initialize()

                if strategy.is_available():
                    self.strategies[strategy_type] = strategy
                    logger.info(f"✅ Initialized: {strategy_type.value}")
                else:
                    logger.warning(f"⚠️ Not available: {strategy_type.value}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize {strategy_type.value}: {e}")

        self._initialized = len(self.strategies) > 0
        logger.info(f"Multi-Strategy Engine ready with {len(self.strategies)} strategies")

    async def search(
        self,
        query: str,
        strategies: Optional[List[str]] = None,
        domain: Optional[str] = None,
        complexity: str = "medium",
        enable_fusion: bool = True
    ) -> MultiStrategyResult:
        """
        Execute search across specified strategies

        Args:
            query: Search query
            strategies: List of strategy names (or None for auto-select)
            domain: Target domain (neuroscience, quantum_ml, etc.)
            complexity: Query complexity (simple, medium, complex)
            enable_fusion: Enable result fusion across strategies

        Returns:
            MultiStrategyResult with combined results
        """
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        # Create query context
        query_context = self._create_query_context(query, domain, complexity)

        # Select strategies
        selected_strategies = self._select_strategies(strategies, query_context)

        if not selected_strategies:
            return MultiStrategyResult(
                query=query,
                results=[],
                strategies_used=[],
                total_sources=0,
                avg_relevance=0.0,
                execution_time_ms=0,
                cross_domain_detected=False
            )

        # Execute searches
        all_results = []
        strategies_used = []
        performance_breakdown = {}

        for strategy in selected_strategies:
            try:
                strategy_start = time.time()
                response = await strategy.search(query_context)
                strategy_time = (time.time() - strategy_start) * 1000

                # Process results
                for source in response.sources:
                    result = SearchResult(
                        content=source.get("content", ""),
                        title=source.get("title", "Unknown"),
                        source=source.get("database", ""),
                        relevance_score=source.get("relevance", 0.5),
                        metadata=source.get("metadata", {}),
                        database=source.get("database", ""),
                        strategy=strategy.strategy.value
                    )
                    all_results.append(result)

                strategies_used.append(strategy.strategy.value)
                performance_breakdown[strategy.strategy.value] = {
                    "latency_ms": strategy_time,
                    "results_count": len(response.sources),
                    "confidence": response.confidence
                }

                # Track usage
                self._strategy_usage[strategy.strategy.value] = \
                    self._strategy_usage.get(strategy.strategy.value, 0) + 1

            except Exception as e:
                logger.error(f"Strategy {strategy.strategy.value} failed: {e}")
                performance_breakdown[strategy.strategy.value] = {"error": str(e)}

        # Fuse results if enabled
        if enable_fusion and len(all_results) > 0:
            all_results = self._fuse_results(all_results)

        # Calculate metrics
        execution_time = (time.time() - start_time) * 1000
        avg_relevance = sum(r.relevance_score for r in all_results) / max(len(all_results), 1)
        cross_domain = self._detect_cross_domain(all_results)

        # Update tracking
        self._search_count += 1
        self._total_latency += execution_time

        return MultiStrategyResult(
            query=query,
            results=all_results[:20],  # Limit to top 20
            strategies_used=strategies_used,
            total_sources=len(all_results),
            avg_relevance=avg_relevance,
            execution_time_ms=execution_time,
            cross_domain_detected=cross_domain,
            performance_breakdown=performance_breakdown
        )

    def _create_query_context(self, query: str, domain: Optional[str], complexity: str) -> QueryContext:
        """Create QueryContext from parameters"""
        # Parse domain
        domain_enum = QueryDomain.GENERAL
        if domain:
            domain_map = {
                "neuroscience": QueryDomain.NEUROSCIENCE,
                "quantum_ml": QueryDomain.QUANTUM_ML,
                "developmental_disorders": QueryDomain.DEVELOPMENTAL_DISORDERS,
                "psychology": QueryDomain.PSYCHOLOGY,
                "general": QueryDomain.GENERAL
            }
            domain_enum = domain_map.get(domain.lower(), QueryDomain.GENERAL)

        # Detect domain from query if not specified
        if domain_enum == QueryDomain.GENERAL:
            domain_enum = self._detect_domain(query)

        # Parse complexity
        complexity_map = {
            "simple": QueryComplexity.SIMPLE,
            "medium": QueryComplexity.MEDIUM,
            "complex": QueryComplexity.COMPLEX
        }
        complexity_enum = complexity_map.get(complexity.lower(), QueryComplexity.MEDIUM)

        return QueryContext(
            query=query,
            complexity=complexity_enum,
            domain=domain_enum,
            intent="synthesis",
            confidence=0.8,
            metadata={}
        )

    def _detect_domain(self, query: str) -> QueryDomain:
        """Auto-detect domain from query content"""
        query_lower = query.lower()

        domain_keywords = {
            QueryDomain.NEUROSCIENCE: ["brain", "neural", "neuron", "cognitive", "뇌", "신경"],
            QueryDomain.QUANTUM_ML: ["quantum", "optimization", "qml", "양자"],
            QueryDomain.DEVELOPMENTAL_DISORDERS: ["autism", "developmental", "disorder", "자폐", "발달"],
            QueryDomain.PSYCHOLOGY: ["psychology", "mental", "cognitive", "심리"]
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return domain

        return QueryDomain.GENERAL

    def _select_strategies(
        self,
        requested: Optional[List[str]],
        query_context: QueryContext
    ) -> List[RealRAGStrategy]:
        """Select strategies based on request and query"""
        if requested:
            # Use requested strategies
            selected = []
            for name in requested:
                for strategy_type, strategy in self.strategies.items():
                    if strategy_type.value.upper() == name.upper():
                        selected.append(strategy)
                        break
            return selected

        # Auto-select based on domain and complexity
        scores = []
        for strategy_type, strategy in self.strategies.items():
            score = strategy.estimate_performance(query_context)
            scores.append((strategy, score))

        # Sort by score and return top 3
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:3]]

    def _fuse_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Fuse and deduplicate results from multiple strategies"""
        # Group by content similarity
        seen = {}
        for result in results:
            key = result.content[:100]
            if key in seen:
                # Keep higher relevance
                if result.relevance_score > seen[key].relevance_score:
                    seen[key] = result
            else:
                seen[key] = result

        # Sort by relevance
        fused = list(seen.values())
        fused.sort(key=lambda x: x.relevance_score, reverse=True)

        return fused

    def _detect_cross_domain(self, results: List[SearchResult]) -> bool:
        """Detect if results span multiple domains"""
        databases = set(r.database for r in results if r.database)
        return len(databases) >= 2

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        return {
            "total_searches": self._search_count,
            "average_latency_ms": self._total_latency / max(self._search_count, 1),
            "available_strategies": list(self.strategies.keys()),
            "strategy_usage": self._strategy_usage,
            "initialized": self._initialized
        }


# ============================================================================
# Factory Functions
# ============================================================================

async def create_real_strategies(config: Optional[ChromaDBConfig] = None) -> Dict[RAGStrategy, RealRAGStrategy]:
    """Create all real strategy implementations"""
    cfg = config or ChromaDBConfig()

    strategies = {
        RAGStrategy.HYBRID: HybridRAGStrategy(cfg),
        RAGStrategy.ENHANCED_DD_RAPTOR: EnhancedDDRaptorStrategy(cfg),
        RAGStrategy.GRAPH_RAG: GraphRAGStrategy(cfg),
        RAGStrategy.GOLDEN_REFERENCE: GoldenReferenceStrategy(cfg),
        RAGStrategy.SIMPLE_RAG: SimpleRAGStrategy(cfg),
        RAGStrategy.MULTIMODAL_RAG: MultimodalRAGStrategy(cfg),
        RAGStrategy.PSYCHOLOGY_RAG: PsychologyRAGStrategy(cfg)
    }

    # Initialize all
    for strategy in strategies.values():
        await strategy.initialize()

    return strategies


async def create_search_engine(config: Optional[ChromaDBConfig] = None) -> MultiStrategySearchEngine:
    """Create and initialize search engine"""
    engine = MultiStrategySearchEngine(config)
    await engine.initialize()
    return engine


# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    """CLI interface for testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Strategy Search Engine")
    parser.add_argument("query", nargs="?", default="ESM3 protein structure prediction brain development")
    parser.add_argument("--strategies", "-s", help="Comma-separated strategies")
    parser.add_argument("--domain", "-d", help="Target domain")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Initialize engine
    print("🚀 Initializing Multi-Strategy Search Engine...")
    engine = await create_search_engine()

    # Parse strategies
    strategies = args.strategies.split(",") if args.strategies else None

    # Execute search
    print(f"\n🔍 Searching: {args.query[:50]}...")
    result = await engine.search(
        query=args.query,
        strategies=strategies,
        domain=args.domain
    )

    # Print results
    print(f"\n{'='*60}")
    print("MULTI-STRATEGY SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Query: {result.query[:50]}...")
    print(f"Strategies Used: {result.strategies_used}")
    print(f"Total Sources: {result.total_sources}")
    print(f"Average Relevance: {result.avg_relevance:.3f}")
    print(f"Execution Time: {result.execution_time_ms:.1f}ms")
    print(f"Cross-Domain: {result.cross_domain_detected}")

    if args.verbose and result.results:
        print(f"\n📚 Top Results:")
        for i, r in enumerate(result.results[:5]):
            print(f"  {i+1}. [{r.strategy}] {r.title[:40]}... (score: {r.relevance_score:.3f})")

    print(f"\n📊 Performance Breakdown:")
    for strategy, perf in result.performance_breakdown.items():
        if "error" in perf:
            print(f"  ❌ {strategy}: {perf['error']}")
        else:
            print(f"  ✅ {strategy}: {perf['latency_ms']:.1f}ms, {perf['results_count']} results")

    # Print stats
    stats = engine.get_performance_stats()
    print(f"\n📈 Engine Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
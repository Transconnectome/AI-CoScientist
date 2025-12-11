#!/usr/bin/env python3
"""
Hybrid DD Search System
Combines DD papers (clinical research) with NeurIPS 2025 FM papers (technical innovations)
for comprehensive developmental disorder research with AI/ML integration
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from .query_classifier import QueryClassifier, QueryType, QueryClassification

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Individual search result from a single database"""
    document: str
    metadata: Dict
    distance: float
    score: float  # Normalized similarity score (0-1)
    source: str  # 'DD' or 'FM'
    level: str  # 'L0', 'L1', or 'L2'


@dataclass
class HybridSearchResult:
    """Combined search result with attribution and scoring"""
    document: str
    metadata: Dict
    dd_score: float
    fm_score: float
    combined_score: float
    source: str
    level: str
    rank: int
    reasoning: str


@dataclass
class HybridSearchResponse:
    """Complete hybrid search response"""
    query: str
    query_classification: QueryClassification
    results: List[HybridSearchResult]
    dd_count: int
    fm_count: int
    total_time_ms: float
    dd_search_time_ms: float
    fm_search_time_ms: float
    merge_time_ms: float
    performance_stats: Dict[str, Any]


class HybridDDSearch:
    """
    Hybrid search system combining DD papers with NeurIPS 2025 foundation model papers.

    Features:
    - Dual database querying (DD + FM papers)
    - Intelligent query classification
    - Adaptive search weighting
    - Cross-encoder reranking
    - Source attribution and provenance
    """

    def __init__(
        self,
        dd_path: str = "chromadb_data_dd",
        fm_path: str = "chromadb_data_neurips2025",
        config: Optional[Dict] = None
    ):
        self.dd_path = Path(dd_path)
        self.fm_path = Path(fm_path)
        self.config = config or self._default_config()

        # Initialize query classifier
        self.classifier = QueryClassifier()

        # Initialize models
        self._init_models()

        # Connect to both ChromaDB instances
        self._init_databases()

        # Performance tracking
        self.performance_metrics = {
            "total_queries": 0,
            "clinical_queries": 0,
            "technical_queries": 0,
            "mixed_queries": 0,
            "average_latency_ms": 0.0,
            "dd_result_ratio": [],
            "fm_result_ratio": []
        }

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "embedding_model": "allenai/scibert_scivocab_uncased",  # 768 dimensions, same as collections
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_results_per_db": 20,
            "final_top_k": 10,
            "use_reranking": True,
            "min_similarity_threshold": 0.5,
            "layer_weights": {
                "L0": 1.0,  # Leaf nodes (chunks)
                "L1": 1.2,  # Intermediate summaries
                "L2": 1.5   # Top-level summaries
            }
        }

    def _init_models(self):
        """Initialize embedding and reranking models"""
        logger.info("Initializing models...")

        # Embedding model for query encoding
        self.embedding_model = SentenceTransformer(
            self.config["embedding_model"]
        )

        # Cross-encoder for reranking
        if self.config["use_reranking"]:
            self.cross_encoder = CrossEncoder(
                self.config["cross_encoder_model"]
            )
        else:
            self.cross_encoder = None

        logger.info("Models initialized successfully")

    def _init_databases(self):
        """Initialize connections to both ChromaDB instances"""
        logger.info("Connecting to ChromaDB instances...")

        # DD papers database
        self.dd_client = chromadb.PersistentClient(path=str(self.dd_path))
        self.dd_collections = {
            'L0': self.dd_client.get_collection("dd_papers_L0"),
            'L1': self.dd_client.get_collection("dd_papers_L1"),
            'L2': self.dd_client.get_collection("dd_papers_L2")
        }

        # NeurIPS 2025 FM papers database
        self.fm_client = chromadb.PersistentClient(path=str(self.fm_path))
        self.fm_collections = {
            'L0': self.fm_client.get_collection("neurips_2025_L0"),
            'L1': self.fm_client.get_collection("neurips_2025_L1"),
            'L2': self.fm_client.get_collection("neurips_2025_L2")
        }

        # Log collection statistics
        dd_counts = {level: coll.count() for level, coll in self.dd_collections.items()}
        fm_counts = {level: coll.count() for level, coll in self.fm_collections.items()}

        logger.info(f"DD Collections: {dd_counts}")
        logger.info(f"FM Collections: {fm_counts}")
        logger.info("Database connections established")

    def search(
        self,
        query: str,
        top_k: int = None,
        layers: List[str] = None,
        enable_classification: bool = True
    ) -> HybridSearchResponse:
        """
        Perform hybrid search across DD and FM databases.

        Args:
            query: Search query string
            top_k: Number of final results to return (default: config value)
            layers: Which RAPTOR layers to search (default: all)
            enable_classification: Use query classification for adaptive weighting

        Returns:
            HybridSearchResponse with ranked results and performance stats
        """
        start_time = time.time()

        if top_k is None:
            top_k = self.config["final_top_k"]

        if layers is None:
            layers = ['L0', 'L1', 'L2']

        # Step 1: Classify query
        classification = self.classifier.classify(query) if enable_classification else None

        if classification:
            dd_weight, fm_weight = self.classifier.get_search_weights(classification)
            logger.info(f"Query classified as {classification.query_type.value}")
            logger.info(f"Search weights: DD={dd_weight:.2f}, FM={fm_weight:.2f}")
        else:
            dd_weight, fm_weight = 1.0, 1.0

        # Step 2: Search DD database
        dd_start = time.time()
        dd_results = self._search_database(
            query=query,
            collections=self.dd_collections,
            layers=layers,
            source='DD',
            max_results=self.config["max_results_per_db"]
        )
        dd_time = (time.time() - dd_start) * 1000

        # Step 3: Search FM database
        fm_start = time.time()
        fm_results = self._search_database(
            query=query,
            collections=self.fm_collections,
            layers=layers,
            source='FM',
            max_results=self.config["max_results_per_db"]
        )
        fm_time = (time.time() - fm_start) * 1000

        # Step 4: Merge and rerank results
        merge_start = time.time()
        merged_results = self._merge_and_rerank(
            query=query,
            dd_results=dd_results,
            fm_results=fm_results,
            dd_weight=dd_weight,
            fm_weight=fm_weight,
            top_k=top_k
        )
        merge_time = (time.time() - merge_start) * 1000

        total_time = (time.time() - start_time) * 1000

        # Count results by source
        dd_count = sum(1 for r in merged_results if r.source == 'DD')
        fm_count = sum(1 for r in merged_results if r.source == 'FM')

        # Update performance metrics
        self._update_metrics(classification, total_time, dd_count, fm_count, len(merged_results))

        # Build response
        response = HybridSearchResponse(
            query=query,
            query_classification=classification,
            results=merged_results,
            dd_count=dd_count,
            fm_count=fm_count,
            total_time_ms=total_time,
            dd_search_time_ms=dd_time,
            fm_search_time_ms=fm_time,
            merge_time_ms=merge_time,
            performance_stats=self._get_performance_summary()
        )

        return response

    def _search_database(
        self,
        query: str,
        collections: Dict[str, Any],
        layers: List[str],
        source: str,
        max_results: int
    ) -> List[SearchResult]:
        """Search a single database (DD or FM) across multiple layers"""
        all_results = []

        # Generate query embedding using SciBERT
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        for layer in layers:
            if layer not in collections:
                continue

            collection = collections[layer]
            layer_weight = self.config["layer_weights"].get(layer, 1.0)

            # Query the collection with embedding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max_results // len(layers), collection.count())
            )

            # Process results
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    # Convert distance to similarity score (lower distance = higher similarity)
                    # ChromaDB uses L2 distance, convert to similarity
                    similarity = 1 / (1 + distance)

                    # Apply layer weight
                    weighted_score = similarity * layer_weight

                    all_results.append(SearchResult(
                        document=doc,
                        metadata=metadata,
                        distance=distance,
                        score=weighted_score,
                        source=source,
                        level=layer
                    ))

        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:max_results]

    def _merge_and_rerank(
        self,
        query: str,
        dd_results: List[SearchResult],
        fm_results: List[SearchResult],
        dd_weight: float,
        fm_weight: float,
        top_k: int
    ) -> List[HybridSearchResult]:
        """
        Merge results from both databases and rerank using cross-encoder.

        Merging strategy:
        1. Apply source-specific weights to scores
        2. Normalize scores across both sources
        3. Optional: Use cross-encoder for final reranking
        4. Return top-k results
        """
        # Combine all results
        all_results = []

        # Process DD results
        for result in dd_results:
            all_results.append({
                'document': result.document,
                'metadata': result.metadata,
                'dd_score': result.score * dd_weight,
                'fm_score': 0.0,
                'source': result.source,
                'level': result.level
            })

        # Process FM results
        for result in fm_results:
            all_results.append({
                'document': result.document,
                'metadata': result.metadata,
                'dd_score': 0.0,
                'fm_score': result.score * fm_weight,
                'source': result.source,
                'level': result.level
            })

        # Calculate combined scores
        for result in all_results:
            result['combined_score'] = result['dd_score'] + result['fm_score']

        # Apply cross-encoder reranking if enabled
        if self.config["use_reranking"] and self.cross_encoder:
            all_results = self._apply_cross_encoder_reranking(query, all_results)
        else:
            # Sort by combined score
            all_results.sort(key=lambda x: x['combined_score'], reverse=True)

        # Convert to HybridSearchResult objects
        final_results = []
        for rank, result in enumerate(all_results[:top_k], 1):
            reasoning = self._generate_reasoning(result, rank)

            final_results.append(HybridSearchResult(
                document=result['document'],
                metadata=result['metadata'],
                dd_score=result['dd_score'],
                fm_score=result['fm_score'],
                combined_score=result['combined_score'],
                source=result['source'],
                level=result['level'],
                rank=rank,
                reasoning=reasoning
            ))

        return final_results

    def _apply_cross_encoder_reranking(
        self,
        query: str,
        results: List[Dict]
    ) -> List[Dict]:
        """Apply cross-encoder reranking to results"""
        # Prepare query-document pairs
        pairs = [[query, result['document']] for result in results]

        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)

        # Combine cross-encoder scores with existing scores
        for result, ce_score in zip(results, ce_scores):
            # Weighted combination: 70% cross-encoder, 30% original score
            result['combined_score'] = (
                0.7 * ce_score + 0.3 * result['combined_score']
            )

        # Sort by new combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)

        return results

    def _generate_reasoning(self, result: Dict, rank: int) -> str:
        """Generate reasoning for why this result was ranked this way"""
        source_name = "DD papers (clinical)" if result['source'] == 'DD' else "NeurIPS 2025 (technical)"

        reasoning_parts = [
            f"Ranked #{rank}",
            f"Source: {source_name}",
            f"Layer: {result['level']}"
        ]

        if result['dd_score'] > result['fm_score']:
            reasoning_parts.append(f"Strong clinical relevance (DD score: {result['dd_score']:.3f})")
        elif result['fm_score'] > result['dd_score']:
            reasoning_parts.append(f"Strong technical relevance (FM score: {result['fm_score']:.3f})")
        else:
            reasoning_parts.append("Balanced clinical-technical relevance")

        return " | ".join(reasoning_parts)

    def _update_metrics(
        self,
        classification: Optional[QueryClassification],
        total_time: float,
        dd_count: int,
        fm_count: int,
        total_count: int
    ):
        """Update performance metrics"""
        self.performance_metrics["total_queries"] += 1

        if classification:
            if classification.query_type == QueryType.CLINICAL:
                self.performance_metrics["clinical_queries"] += 1
            elif classification.query_type == QueryType.TECHNICAL:
                self.performance_metrics["technical_queries"] += 1
            else:
                self.performance_metrics["mixed_queries"] += 1

        # Update average latency (running average)
        n = self.performance_metrics["total_queries"]
        current_avg = self.performance_metrics["average_latency_ms"]
        self.performance_metrics["average_latency_ms"] = (
            (current_avg * (n - 1) + total_time) / n
        )

        # Track result distribution
        if total_count > 0:
            dd_ratio = dd_count / total_count
            fm_ratio = fm_count / total_count
            self.performance_metrics["dd_result_ratio"].append(dd_ratio)
            self.performance_metrics["fm_result_ratio"].append(fm_ratio)

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        metrics = self.performance_metrics

        summary = {
            "total_queries": metrics["total_queries"],
            "query_distribution": {
                "clinical": metrics["clinical_queries"],
                "technical": metrics["technical_queries"],
                "mixed": metrics["mixed_queries"]
            },
            "average_latency_ms": round(metrics["average_latency_ms"], 2)
        }

        if metrics["dd_result_ratio"]:
            summary["average_dd_ratio"] = round(
                np.mean(metrics["dd_result_ratio"]), 3
            )
            summary["average_fm_ratio"] = round(
                np.mean(metrics["fm_result_ratio"]), 3
            )

        return summary

    def search_async(self, query: str, **kwargs) -> asyncio.Future:
        """Async wrapper for search method"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: self.search(query, **kwargs))

    def _convert_to_python_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_python_types(item) for item in obj]
        else:
            return obj

    def export_results(
        self,
        response: HybridSearchResponse,
        output_file: str
    ):
        """Export search results to JSON file"""
        export_data = {
            "query": response.query,
            "query_classification": {
                "type": response.query_classification.query_type.value,
                "clinical_score": response.query_classification.clinical_score,
                "technical_score": response.query_classification.technical_score,
                "confidence": response.query_classification.confidence,
                "reasoning": response.query_classification.reasoning
            } if response.query_classification else None,
            "results": [
                {
                    "rank": r.rank,
                    "document": r.document[:500] + "..." if len(r.document) > 500 else r.document,
                    "metadata": self._convert_to_python_types(r.metadata),
                    "scores": {
                        "dd_score": round(float(r.dd_score), 4),
                        "fm_score": round(float(r.fm_score), 4),
                        "combined_score": round(float(r.combined_score), 4)
                    },
                    "source": r.source,
                    "level": r.level,
                    "reasoning": r.reasoning
                }
                for r in response.results
            ],
            "statistics": {
                "dd_count": response.dd_count,
                "fm_count": response.fm_count,
                "total_results": len(response.results),
                "timing": {
                    "total_ms": round(response.total_time_ms, 2),
                    "dd_search_ms": round(response.dd_search_time_ms, 2),
                    "fm_search_ms": round(response.fm_search_time_ms, 2),
                    "merge_rerank_ms": round(response.merge_time_ms, 2)
                }
            },
            "performance_stats": response.performance_stats
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results exported to {output_file}")


def format_search_response(response: HybridSearchResponse) -> str:
    """Format search response for display"""
    lines = [
        "=" * 80,
        f"HYBRID SEARCH RESULTS",
        "=" * 80,
        f"\nQuery: {response.query}",
        f"\nQuery Classification:"
    ]

    if response.query_classification:
        lines.extend([
            f"  Type: {response.query_classification.query_type.value.upper()}",
            f"  Clinical Score: {response.query_classification.clinical_score:.2%}",
            f"  Technical Score: {response.query_classification.technical_score:.2%}",
            f"  Confidence: {response.query_classification.confidence:.2%}",
            f"  Reasoning: {response.query_classification.reasoning}"
        ])

    lines.extend([
        f"\nResults: {len(response.results)} total (DD: {response.dd_count}, FM: {response.fm_count})",
        f"Timing: Total={response.total_time_ms:.0f}ms (DD={response.dd_search_time_ms:.0f}ms, FM={response.fm_search_time_ms:.0f}ms, Merge={response.merge_time_ms:.0f}ms)",
        "\n" + "-" * 80,
        "TOP RESULTS",
        "-" * 80
    ])

    for result in response.results:
        lines.extend([
            f"\n[{result.rank}] Source: {result.source} | Level: {result.level} | Score: {result.combined_score:.4f}",
            f"Reasoning: {result.reasoning}",
            f"Document: {result.document[:300]}...",
            f"Metadata: {json.dumps(result.metadata, indent=2)}"
        ])

    lines.append("\n" + "=" * 80)

    return '\n'.join(lines)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    search = HybridDDSearch()

    query = "foundation models for autism diagnosis using EEG"
    response = search.search(query, top_k=5)

    print(format_search_response(response))

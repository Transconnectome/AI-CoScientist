"""
NRF Proposal RAG Strategy

Integrates NRF Mid-Career proposal samples into the Unified RAG Orchestrator.
Provides Golden Reference strategy for Korean grant proposal optimization.

Collections:
- nrf_midcareer_samples_L0: Chunks (659 documents)
- nrf_midcareer_samples_L1: Section summaries (24 documents)
- nrf_midcareer_samples_L2: Document summaries (4 documents)

Usage:
    from src.services.rag.nrf_proposal_strategy import NRFProposalRAGStrategy

    strategy = NRFProposalRAGStrategy()
    results = await strategy.search(query_context)
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Import from unified orchestrator
try:
    from src.services.rag.unified_rag_orchestrator import (
        RAGStrategyInterface,
        RAGStrategy,
        QueryContext,
        RAGResponse,
        QueryDomain,
        QueryComplexity
    )
    from src.monitoring.rag_metrics import RAGMetrics
except ImportError:
    # Fallback for standalone testing
    RAGStrategyInterface = object
    RAGStrategy = None
    QueryContext = None
    RAGResponse = None

logger = logging.getLogger(__name__)

# Configuration
CHROMADB_PATH = "chromadb_data"
COLLECTION_L0 = "nrf_midcareer_samples_L0"
COLLECTION_L1 = "nrf_midcareer_samples_L1"
COLLECTION_L2 = "nrf_midcareer_samples_L2"


@dataclass
class NRFSearchResult:
    """Search result from NRF proposal collection."""
    content: str
    score: float
    section: str
    proposal_type: str
    proposal_id: str
    level: int  # 0=chunk, 1=section, 2=document
    metadata: Dict[str, Any]


class NRFProposalRAGStrategy(RAGStrategyInterface):
    """
    RAG Strategy for NRF Mid-Career Proposal samples.

    Implements hierarchical RAPTOR-style retrieval:
    - L0: Detailed chunks for specific content
    - L1: Section summaries for structural patterns
    - L2: Document summaries for overall approach
    """

    def __init__(
        self,
        chromadb_path: str = CHROMADB_PATH,
        embedding_model: str = "allenai/scibert_scivocab_uncased"
    ):
        self.chromadb_path = chromadb_path
        self.embedding_model_name = embedding_model

        # Lazy initialization
        self._client = None
        self._collections = {}
        self._embedding_model = None
        self._initialized = False

        logger.info("NRFProposalRAGStrategy created (lazy initialization)")

    def _initialize(self):
        """Initialize ChromaDB and embedding model."""
        if self._initialized:
            return

        try:
            # Initialize ChromaDB
            self._client = chromadb.PersistentClient(path=self.chromadb_path)

            # Get collections
            self._collections = {
                'L0': self._client.get_collection(COLLECTION_L0),
                'L1': self._client.get_collection(COLLECTION_L1),
                'L2': self._client.get_collection(COLLECTION_L2)
            }

            # Initialize embedding model
            self._embedding_model = SentenceTransformer(self.embedding_model_name)

            self._initialized = True

            # Log stats
            stats = self.get_collection_stats()
            logger.info(f"NRFProposalRAGStrategy initialized: {stats}")

        except Exception as e:
            logger.error(f"Failed to initialize NRFProposalRAGStrategy: {e}")
            raise

    def is_available(self) -> bool:
        """Check if strategy is available."""
        try:
            if not self._initialized:
                self._initialize()
            return self._initialized and all(
                c.count() > 0 for c in self._collections.values()
            )
        except Exception:
            return False

    def get_strategy_name(self):
        """Get strategy identifier."""
        if RAGStrategy:
            return RAGStrategy.GOLDEN_REFERENCE
        return "nrf_proposal"

    def get_collection_stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        if not self._initialized:
            self._initialize()

        return {
            'L0_chunks': self._collections['L0'].count(),
            'L1_sections': self._collections['L1'].count(),
            'L2_documents': self._collections['L2'].count()
        }

    def estimate_performance(self, query_context) -> float:
        """Estimate performance score for this query."""
        if query_context is None:
            return 0.5

        query = query_context.query if hasattr(query_context, 'query') else str(query_context)

        # NRF proposal keywords boost score
        nrf_keywords = [
            '연구', '제안서', '과제', '방법론', '추진', '전략', '목표',
            '창의성', '도전성', '기대효과', '연구비', '마일스톤',
            'proposal', 'grant', 'methodology', 'aim', 'objective',
            'foundation model', 'brain', 'neuroscience', 'AI'
        ]

        score = 0.5  # Base score
        query_lower = query.lower()

        for keyword in nrf_keywords:
            if keyword.lower() in query_lower:
                score += 0.05

        return min(score, 1.0)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self._initialized:
            self._initialize()

        embedding = self._embedding_model.encode([text])[0]
        return embedding.tolist()

    async def search(
        self,
        query_context,
        n_results: int = 10,
        levels: List[str] = None
    ) -> RAGResponse:
        """
        Execute hierarchical search across NRF proposal collections.

        Args:
            query_context: Query context with query string and metadata
            n_results: Number of results per level
            levels: Which levels to search ['L0', 'L1', 'L2']. Default: all

        Returns:
            RAGResponse with aggregated results
        """
        import time
        start_time = time.time()

        if not self._initialized:
            self._initialize()

        # Extract query
        if hasattr(query_context, 'query'):
            query = query_context.query
        else:
            query = str(query_context)

        levels = levels or ['L0', 'L1', 'L2']

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Search each level
        all_results = []

        for level in levels:
            if level not in self._collections:
                continue

            collection = self._collections[level]

            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )

                # Parse results
                level_int = int(level[1])  # 'L0' -> 0

                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    all_results.append(NRFSearchResult(
                        content=doc,
                        score=1.0 - dist,  # Convert distance to similarity
                        section=meta.get('section', 'unknown'),
                        proposal_type=meta.get('proposal_type', 'unknown'),
                        proposal_id=meta.get('proposal_id', 'unknown'),
                        level=level_int,
                        metadata=meta
                    ))

            except Exception as e:
                logger.warning(f"Error searching {level}: {e}")

        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Build response
        elapsed_time = time.time() - start_time

        # Format sources
        sources = []
        for r in all_results[:n_results * len(levels)]:
            sources.append({
                'content': r.content[:500] + '...' if len(r.content) > 500 else r.content,
                'score': r.score,
                'section': r.section,
                'proposal_type': r.proposal_type,
                'level': f'L{r.level}',
                'metadata': r.metadata
            })

        # Build answer summary
        if all_results:
            top_result = all_results[0]
            answer = f"Found {len(all_results)} relevant passages from NRF proposal samples. " \
                     f"Top match (score: {top_result.score:.3f}) from {top_result.proposal_type} " \
                     f"proposal, section: {top_result.section}."
        else:
            answer = "No relevant passages found in NRF proposal samples."

        # Create metrics
        metrics = None
        if 'RAGMetrics' in dir():
            try:
                from datetime import datetime
                metrics = RAGMetrics(
                    latency=elapsed_time,
                    quality_score=all_results[0].score if all_results else 0,
                    tokens_processed=len(query.split()),
                    retrieval_time=elapsed_time,
                    generation_time=0,
                    context_relevance=all_results[0].score if all_results else 0,
                    faithfulness=0.9,
                    answer_relevancy=all_results[0].score if all_results else 0,
                    strategy='nrf_proposal',
                    timestamp=datetime.now()
                )
            except Exception:
                pass

        if RAGResponse:
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=all_results[0].score if all_results else 0,
                strategy_used=self.get_strategy_name(),
                performance_metrics=metrics,
                metadata={
                    'total_results': len(all_results),
                    'levels_searched': levels,
                    'query': query
                }
            )
        else:
            # Fallback dict response
            return {
                'answer': answer,
                'sources': sources,
                'confidence': all_results[0].score if all_results else 0,
                'total_results': len(all_results)
            }

    async def search_by_section(
        self,
        query: str,
        section_filter: List[str] = None,
        n_results: int = 5
    ) -> List[NRFSearchResult]:
        """
        Search with section filtering.

        Args:
            query: Search query
            section_filter: List of sections to include (e.g., ['Methods', '추진전략'])
            n_results: Number of results

        Returns:
            List of NRFSearchResult
        """
        if not self._initialized:
            self._initialize()

        query_embedding = self._generate_embedding(query)

        results = self._collections['L0'].query(
            query_embeddings=[query_embedding],
            n_results=n_results * 3,  # Get more, then filter
            include=['documents', 'metadatas', 'distances']
        )

        search_results = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            section = meta.get('section', '')

            # Apply section filter
            if section_filter:
                if not any(s.lower() in section.lower() for s in section_filter):
                    continue

            search_results.append(NRFSearchResult(
                content=doc,
                score=1.0 - dist,
                section=section,
                proposal_type=meta.get('proposal_type', 'unknown'),
                proposal_id=meta.get('proposal_id', 'unknown'),
                level=0,
                metadata=meta
            ))

        return search_results[:n_results]

    async def search_by_proposal_type(
        self,
        query: str,
        proposal_types: List[str] = None,
        n_results: int = 5
    ) -> List[NRFSearchResult]:
        """
        Search with proposal type filtering.

        Args:
            query: Search query
            proposal_types: List of types (e.g., ['INCITE', 'Samsung', 'BrainLink'])
            n_results: Number of results

        Returns:
            List of NRFSearchResult
        """
        if not self._initialized:
            self._initialize()

        query_embedding = self._generate_embedding(query)

        results = self._collections['L0'].query(
            query_embeddings=[query_embedding],
            n_results=n_results * 3,
            include=['documents', 'metadatas', 'distances']
        )

        search_results = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            proposal_type = meta.get('proposal_type', '')

            if proposal_types:
                if not any(t.lower() in proposal_type.lower() for t in proposal_types):
                    continue

            search_results.append(NRFSearchResult(
                content=doc,
                score=1.0 - dist,
                section=meta.get('section', 'unknown'),
                proposal_type=proposal_type,
                proposal_id=meta.get('proposal_id', 'unknown'),
                level=0,
                metadata=meta
            ))

        return search_results[:n_results]

    async def get_proposal_patterns(self, proposal_type: str = None) -> Dict[str, Any]:
        """
        Extract common patterns from proposals.

        Args:
            proposal_type: Optional filter by type

        Returns:
            Dict with extracted patterns
        """
        if not self._initialized:
            self._initialize()

        # Get all document summaries
        results = self._collections['L2'].get(
            include=['documents', 'metadatas']
        )

        patterns = {
            'proposal_types': [],
            'common_sections': [],
            'total_documents': len(results['documents'])
        }

        for meta in results['metadatas']:
            ptype = meta.get('proposal_type', 'unknown')
            if ptype not in patterns['proposal_types']:
                patterns['proposal_types'].append(ptype)

        # Get section patterns from L1
        section_results = self._collections['L1'].get(
            include=['metadatas']
        )

        sections = set()
        for meta in section_results['metadatas']:
            sections.add(meta.get('section', 'unknown'))

        patterns['common_sections'] = list(sections)

        return patterns


# Factory function for UPE integration
def create_nrf_proposal_strategy() -> NRFProposalRAGStrategy:
    """Create NRF Proposal RAG Strategy instance."""
    return NRFProposalRAGStrategy()


# Demo and testing
async def demo():
    """Demo the NRF proposal strategy."""
    print("\n" + "=" * 70)
    print("NRF PROPOSAL RAG STRATEGY DEMO")
    print("=" * 70)

    strategy = NRFProposalRAGStrategy()

    # Check availability
    print(f"\nStrategy available: {strategy.is_available()}")
    print(f"Collection stats: {strategy.get_collection_stats()}")

    # Test queries
    queries = [
        "연구과제의 창의성과 도전성",
        "4D Swin Transformer architecture",
        "발달장애 진단 AI 모델",
        "연구 추진 전략과 방법론"
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = await strategy.search(query, n_results=3)

        if isinstance(response, dict):
            print(f"Answer: {response['answer']}")
            print(f"Top {len(response['sources'])} sources:")
            for src in response['sources'][:2]:
                print(f"  - [{src['level']}] {src['section']} ({src['proposal_type']})")
                print(f"    Score: {src['score']:.4f}")
        else:
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.4f}")

    # Test section filtering
    print("\n--- Section Filtered Search: Methods ---")
    results = await strategy.search_by_section(
        "뇌영상 분석 방법",
        section_filter=['Methods', '연구 방법', '추진전략'],
        n_results=3
    )
    for r in results:
        print(f"  - {r.section} (score: {r.score:.4f})")

    # Get patterns
    print("\n--- Proposal Patterns ---")
    patterns = await strategy.get_proposal_patterns()
    print(f"Types: {patterns['proposal_types']}")
    print(f"Common sections: {patterns['common_sections'][:5]}...")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())

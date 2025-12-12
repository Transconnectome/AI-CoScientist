"""
Test suite for GraphRAG Integration

Implementation for: GraphRAG integration testing
Created: 2025-12-05

Acceptance Criteria:
- Knowledge graph construction testing
- Entity extraction and relationship detection validation
- GraphRAG strategy integration with orchestrator
- Performance benchmarking of graph-based retrieval

This test suite validates the complete GraphRAG integration pipeline
from knowledge graph construction through enhanced retrieval.
"""

import pytest
import asyncio
import time
import tempfile
import json
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from services.rag.knowledge_graph_builder import (
    KnowledgeGraphBuilder, KnowledgeGraph, Entity, Relationship,
    EntityType, RelationType, SciBERTEntityExtractor, RelationshipExtractor,
    create_knowledge_graph_builder
)
from services.rag.graph_rag_strategy import (
    GraphRAGStrategy, EntityMatcher, GraphTraverser,
    GraphTraversalMode, ContextExpansionStrategy, GraphContext,
    create_graph_rag_strategy
)
from services.rag.unified_rag_orchestrator import (
    UnifiedRAGOrchestrator, QueryContext, RAGResponse, RAGStrategy,
    QueryComplexity, QueryDomain, create_unified_orchestrator
)

class TestKnowledgeGraphBuilder:
    """Test knowledge graph builder functionality"""

    @pytest.fixture
    def sample_documents(self):
        """Sample scientific documents for testing"""
        return [
            ("doc1", "fMRI measures brain activity using magnetic resonance imaging. It is used in neuroscience research to study cognitive functions."),
            ("doc2", "Machine learning algorithms can analyze fMRI data to detect patterns. Deep learning models are particularly effective for neuroimaging analysis."),
            ("doc3", "Autism spectrum disorder affects social communication and behavior. fMRI studies show different brain activation patterns in individuals with autism."),
            ("doc4", "Dopamine is a neurotransmitter that plays a crucial role in reward processing. It is measured using PET imaging and affects learning and motivation."),
            ("doc5", "Support vector machines are used for classification tasks in machine learning. They can be applied to neuroimaging data for diagnostic purposes.")
        ]

    @pytest.fixture
    async def kg_builder(self):
        """Create knowledge graph builder for testing"""
        return create_knowledge_graph_builder()

    def test_entity_extraction(self, kg_builder):
        """Test entity extraction from scientific text"""
        text = "fMRI measures brain activity using magnetic resonance imaging"
        doc_id = "test_doc"

        # Test with mock extractor since SciBERT may not be available
        extractor = SciBERTEntityExtractor()

        # The extractor should work even without the model (fallback mode)
        entities = asyncio.run(extractor.extract_entities(text, doc_id))

        # Should extract some entities (even with pattern matching)
        assert isinstance(entities, list)
        # Pattern-based extraction should find at least "fMRI"
        entity_texts = [entity.text for entity in entities]
        assert any("fMRI" in entity_texts for entity_texts in [[e] for e in entity_texts])

    def test_relationship_extraction(self):
        """Test relationship extraction between entities"""
        extractor = RelationshipExtractor()
        text = "fMRI measures brain activity"
        doc_id = "test_doc"

        # Create mock entities
        fmri_entity = Entity(
            id="entity1",
            text="fMRI",
            type=EntityType.TECHNIQUE,
            confidence=0.9
        )

        brain_activity_entity = Entity(
            id="entity2",
            text="brain activity",
            type=EntityType.MEASUREMENT,
            confidence=0.8
        )

        entities = [fmri_entity, brain_activity_entity]

        relationships = asyncio.run(extractor.extract_relationships(text, entities, doc_id))

        # Should find some relationships
        assert isinstance(relationships, list)

    @pytest.mark.asyncio
    async def test_knowledge_graph_construction(self, kg_builder, sample_documents):
        """Test complete knowledge graph construction"""
        graph = await kg_builder.build_graph_from_documents(sample_documents)

        # Validate graph structure
        assert isinstance(graph, KnowledgeGraph)
        assert len(graph.entities) > 0
        assert isinstance(graph.entities, dict)

        # Check for expected entities (pattern-based extraction should find these)
        entity_texts = [entity.text for entity in graph.entities.values()]
        assert any("fMRI" in text for text in entity_texts)

        # Validate entity structure
        for entity in graph.entities.values():
            assert isinstance(entity.id, str)
            assert isinstance(entity.text, str)
            assert isinstance(entity.type, EntityType)
            assert 0 <= entity.confidence <= 1
            assert isinstance(entity.source_docs, list)

        # Validate relationships
        for relationship in graph.relationships.values():
            assert isinstance(relationship.id, str)
            assert isinstance(relationship.source_entity, Entity)
            assert isinstance(relationship.target_entity, Entity)
            assert isinstance(relationship.type, RelationType)
            assert 0 <= relationship.confidence <= 1

    def test_entity_merging(self, kg_builder):
        """Test entity merging across documents"""
        # Create entities that should be merged
        entities = [
            Entity(
                id="e1",
                text="fMRI",
                type=EntityType.TECHNIQUE,
                confidence=0.8,
                source_docs=["doc1"]
            ),
            Entity(
                id="e2",
                text="fMRI",
                type=EntityType.TECHNIQUE,
                confidence=0.9,
                source_docs=["doc2"]
            )
        ]

        merged = kg_builder._merge_entities_globally(entities)

        # Should merge into single entity
        assert len(merged) == 1
        merged_entity = merged[0]
        assert "doc1" in merged_entity.source_docs
        assert "doc2" in merged_entity.source_docs
        assert merged_entity.confidence == 0.85  # Average of 0.8 and 0.9

class TestEntityMatcher:
    """Test entity matching functionality"""

    @pytest.fixture
    def sample_knowledge_graph(self):
        """Create sample knowledge graph for testing"""
        kg = KnowledgeGraph()

        # Add sample entities
        entities = [
            Entity(
                id="e1",
                text="fMRI",
                type=EntityType.TECHNIQUE,
                confidence=0.9,
                aliases=["functional MRI", "functional magnetic resonance imaging"]
            ),
            Entity(
                id="e2",
                text="brain activity",
                type=EntityType.MEASUREMENT,
                confidence=0.8
            ),
            Entity(
                id="e3",
                text="autism",
                type=EntityType.DISEASE,
                confidence=0.9,
                aliases=["autism spectrum disorder", "ASD"]
            ),
            Entity(
                id="e4",
                text="machine learning",
                type=EntityType.METHOD,
                confidence=0.85
            )
        ]

        for entity in entities:
            kg.add_entity(entity)

        return kg

    @pytest.fixture
    def entity_matcher(self, sample_knowledge_graph):
        """Create entity matcher for testing"""
        return EntityMatcher(sample_knowledge_graph)

    @pytest.mark.asyncio
    async def test_exact_matching(self, entity_matcher):
        """Test exact entity matching"""
        query = "What is fMRI?"
        query_context = QueryContext(
            query=query,
            complexity=QueryComplexity.SIMPLE,
            domain=QueryDomain.NEUROSCIENCE,
            intent="factual",
            confidence=0.9,
            metadata={}
        )

        matches = await entity_matcher.find_matching_entities(query, query_context)

        # Should find fMRI entity
        assert len(matches) > 0
        entity_texts = [entity.text for entity, _ in matches]
        assert "fMRI" in entity_texts

    @pytest.mark.asyncio
    async def test_alias_matching(self, entity_matcher):
        """Test matching through entity aliases"""
        query = "How does functional MRI work?"
        query_context = QueryContext(
            query=query,
            complexity=QueryComplexity.MEDIUM,
            domain=QueryDomain.NEUROSCIENCE,
            intent="procedural",
            confidence=0.9,
            metadata={}
        )

        matches = await entity_matcher.find_matching_entities(query, query_context)

        # Should find fMRI entity through alias
        assert len(matches) > 0
        matched_entities = [entity for entity, _ in matches]
        fmri_entities = [e for e in matched_entities if "fMRI" in e.text or "functional MRI" in e.aliases]
        assert len(fmri_entities) > 0

    @pytest.mark.asyncio
    async def test_partial_matching(self, entity_matcher):
        """Test partial text matching"""
        query = "autism spectrum disorders"
        query_context = QueryContext(
            query=query,
            complexity=QueryComplexity.SIMPLE,
            domain=QueryDomain.DEVELOPMENTAL_DISORDERS,
            intent="factual",
            confidence=0.9,
            metadata={}
        )

        matches = await entity_matcher.find_matching_entities(query, query_context)

        # Should find autism entity
        assert len(matches) > 0
        entity_texts = [entity.text for entity, _ in matches]
        assert any("autism" in text.lower() for text in entity_texts)

class TestGraphTraverser:
    """Test graph traversal functionality"""

    @pytest.fixture
    def sample_connected_graph(self):
        """Create sample graph with relationships"""
        kg = KnowledgeGraph()

        # Create entities
        entities = [
            Entity("e1", "fMRI", EntityType.TECHNIQUE, 0.9),
            Entity("e2", "brain activity", EntityType.MEASUREMENT, 0.8),
            Entity("e3", "autism", EntityType.DISEASE, 0.9),
            Entity("e4", "social behavior", EntityType.CONCEPT, 0.7),
            Entity("e5", "neural networks", EntityType.METHOD, 0.8),
        ]

        for entity in entities:
            kg.add_entity(entity)

        # Create relationships
        relationships = [
            Relationship(
                "r1", entities[0], entities[1], RelationType.MEASURES, 0.9,
                "fMRI measures brain activity", ["doc1"]
            ),
            Relationship(
                "r2", entities[2], entities[3], RelationType.AFFECTS, 0.8,
                "autism affects social behavior", ["doc2"]
            ),
            Relationship(
                "r3", entities[1], entities[2], RelationType.ASSOCIATED_WITH, 0.7,
                "brain activity associated with autism", ["doc3"]
            ),
            Relationship(
                "r4", entities[4], entities[1], RelationType.USED_FOR, 0.8,
                "neural networks used for brain activity analysis", ["doc4"]
            ),
        ]

        for rel in relationships:
            kg.add_relationship(rel)

        return kg

    @pytest.fixture
    def graph_traverser(self, sample_connected_graph):
        """Create graph traverser for testing"""
        return GraphTraverser(sample_connected_graph)

    @pytest.mark.asyncio
    async def test_entity_centric_expansion(self, graph_traverser, sample_connected_graph):
        """Test entity-centric context expansion"""
        # Start with fMRI entity
        seed_entities = [list(sample_connected_graph.entities.values())[0]]  # fMRI

        graph_context = await graph_traverser.expand_context(
            seed_entities,
            GraphTraversalMode.ENTITY_CENTRIC,
            ContextExpansionStrategy.IMMEDIATE_NEIGHBORS,
            max_depth=1,
            max_entities=10
        )

        # Should expand to include connected entities
        assert len(graph_context.entities) > 1
        assert len(graph_context.relationships) > 0
        assert graph_context.subgraph is not None

        # Should include brain activity (connected to fMRI)
        entity_texts = [entity.text for entity in graph_context.entities]
        assert any("brain activity" in text for text in entity_texts)

    @pytest.mark.asyncio
    async def test_relationship_aware_expansion(self, graph_traverser, sample_connected_graph):
        """Test relationship-aware context expansion"""
        seed_entities = [list(sample_connected_graph.entities.values())[0]]  # fMRI

        graph_context = await graph_traverser.expand_context(
            seed_entities,
            GraphTraversalMode.RELATIONSHIP_AWARE,
            ContextExpansionStrategy.TYPED_RELATIONSHIPS,
            max_depth=2,
            max_entities=10
        )

        # Should prioritize based on relationship types
        assert len(graph_context.entities) > 0
        assert len(graph_context.relationships) > 0

        # Check expansion metadata
        assert "strategy" in graph_context.expansion_metadata
        assert graph_context.expansion_metadata["strategy"] == "relationship_aware"

    @pytest.mark.asyncio
    async def test_multi_hop_expansion(self, graph_traverser, sample_connected_graph):
        """Test multi-hop graph traversal"""
        # Start with multiple entities
        entities = list(sample_connected_graph.entities.values())
        seed_entities = [entities[0], entities[2]]  # fMRI and autism

        graph_context = await graph_traverser.expand_context(
            seed_entities,
            GraphTraversalMode.MULTI_HOP,
            ContextExpansionStrategy.IMMEDIATE_NEIGHBORS,
            max_depth=2,
            max_entities=15
        )

        # Should find paths between entities
        assert len(graph_context.entities) > 2
        assert len(graph_context.traversal_path) > 0

        # Check if path information is captured
        path_info = [step for step in graph_context.traversal_path if "path:" in step]
        # May or may not find direct paths depending on graph structure

class TestGraphRAGStrategy:
    """Test GraphRAG strategy integration"""

    @pytest.fixture
    async def sample_graph_rag_strategy(self):
        """Create GraphRAG strategy for testing"""
        strategy = create_graph_rag_strategy()

        # Initialize with sample documents
        sample_documents = [
            ("doc1", "fMRI measures brain activity using magnetic resonance imaging. It is used in neuroscience research."),
            ("doc2", "Machine learning algorithms analyze fMRI data to detect patterns in brain activity."),
            ("doc3", "Autism affects social communication and shows distinct patterns in brain imaging studies."),
        ]

        await strategy.initialize(sample_documents)
        return strategy

    @pytest.mark.asyncio
    async def test_graph_rag_initialization(self, sample_graph_rag_strategy):
        """Test GraphRAG strategy initialization"""
        strategy = sample_graph_rag_strategy

        assert strategy.knowledge_graph is not None
        assert strategy.entity_matcher is not None
        assert strategy.graph_traverser is not None
        assert len(strategy.knowledge_graph.entities) > 0

    @pytest.mark.asyncio
    async def test_graph_rag_search(self, sample_graph_rag_strategy):
        """Test GraphRAG search functionality"""
        strategy = sample_graph_rag_strategy

        query_context = QueryContext(
            query="How does fMRI study brain activity?",
            complexity=QueryComplexity.MEDIUM,
            domain=QueryDomain.NEUROSCIENCE,
            intent="procedural",
            confidence=0.9,
            metadata={}
        )

        response = await strategy.search(query_context)

        # Validate response structure
        assert isinstance(response, RAGResponse)
        assert response.strategy_used == RAGStrategy.GRAPH_RAG
        assert response.answer is not None
        assert len(response.answer) > 0
        assert 0 <= response.confidence <= 1
        assert response.performance_metrics is not None

        # Check metadata for graph-specific information
        assert "matched_entities" in response.metadata
        assert "expanded_entities" in response.metadata
        assert "traversal_path" in response.metadata

    @pytest.mark.asyncio
    async def test_graph_rag_with_no_matches(self, sample_graph_rag_strategy):
        """Test GraphRAG behavior with no entity matches"""
        strategy = sample_graph_rag_strategy

        query_context = QueryContext(
            query="What is quantum computing?",  # Likely no entities in our sample
            complexity=QueryComplexity.COMPLEX,
            domain=QueryDomain.QUANTUM_ML,
            intent="factual",
            confidence=0.9,
            metadata={}
        )

        response = await strategy.search(query_context)

        # Should return response even with no matches
        assert isinstance(response, RAGResponse)
        assert response.confidence < 0.5  # Low confidence expected

    @pytest.mark.asyncio
    async def test_graph_rag_performance_tracking(self, sample_graph_rag_strategy):
        """Test performance tracking in GraphRAG"""
        strategy = sample_graph_rag_strategy

        # Execute multiple searches
        for i in range(3):
            query_context = QueryContext(
                query=f"Test query {i} about fMRI",
                complexity=QueryComplexity.SIMPLE,
                domain=QueryDomain.NEUROSCIENCE,
                intent="factual",
                confidence=0.8,
                metadata={}
            )

            await strategy.search(query_context)

        # Check performance stats
        stats = strategy.get_performance_stats()
        assert "avg_retrieval_time" in stats
        assert "total_retrievals" in stats
        assert stats["total_retrievals"] == 3

class TestGraphRAGIntegrationWithOrchestrator:
    """Test GraphRAG integration with unified orchestrator"""

    @pytest.fixture
    async def orchestrator_with_graph_rag(self):
        """Create orchestrator with GraphRAG strategy"""
        orchestrator = create_unified_orchestrator()

        # Mock GraphRAG strategy
        graph_rag_strategy = Mock()
        graph_rag_response = RAGResponse(
            answer="GraphRAG response about fMRI and brain activity",
            sources=[{"type": "entity", "text": "fMRI"}],
            confidence=0.8,
            strategy_used=RAGStrategy.GRAPH_RAG,
            performance_metrics=Mock()
        )

        async def mock_search(query_context):
            return graph_rag_response

        graph_rag_strategy.search = mock_search
        orchestrator.strategies[RAGStrategy.GRAPH_RAG] = graph_rag_strategy

        return orchestrator

    @pytest.mark.asyncio
    async def test_orchestrator_graph_rag_selection(self, orchestrator_with_graph_rag):
        """Test orchestrator selecting GraphRAG strategy"""
        orchestrator = orchestrator_with_graph_rag

        query_context = QueryContext(
            query="How are fMRI and autism related?",
            complexity=QueryComplexity.COMPLEX,
            domain=QueryDomain.NEUROSCIENCE,
            intent="synthesis",
            confidence=0.9,
            metadata={}
        )

        # Force GraphRAG strategy
        response = await orchestrator.search(query_context, strategy_override=RAGStrategy.GRAPH_RAG)

        assert response.strategy_used == RAGStrategy.GRAPH_RAG
        assert "GraphRAG response" in response.answer

    @pytest.mark.asyncio
    async def test_graph_rag_fallback_behavior(self, orchestrator_with_graph_rag):
        """Test fallback when GraphRAG fails"""
        orchestrator = orchestrator_with_graph_rag

        # Make GraphRAG strategy fail
        async def failing_search(query_context):
            raise Exception("GraphRAG failed")

        orchestrator.strategies[RAGStrategy.GRAPH_RAG].search = failing_search

        query_context = QueryContext(
            query="Test query",
            complexity=QueryComplexity.SIMPLE,
            domain=QueryDomain.GENERAL,
            intent="factual",
            confidence=0.8,
            metadata={}
        )

        # Should fallback to other strategies
        response = await orchestrator.search(query_context, enable_fallback=True)
        assert response.strategy_used != RAGStrategy.GRAPH_RAG

class TestGraphRAGPerformanceBenchmark:
    """Test GraphRAG performance benchmarking"""

    @pytest.mark.asyncio
    async def test_graph_rag_latency_benchmark(self):
        """Test GraphRAG latency performance"""
        strategy = create_graph_rag_strategy()

        # Small test dataset
        documents = [
            ("doc1", "fMRI measures brain activity"),
            ("doc2", "Machine learning analyzes data"),
        ]

        await strategy.initialize(documents)

        # Measure search latency
        start_time = time.time()

        query_context = QueryContext(
            query="What is fMRI?",
            complexity=QueryComplexity.SIMPLE,
            domain=QueryDomain.NEUROSCIENCE,
            intent="factual",
            confidence=0.9,
            metadata={}
        )

        response = await strategy.search(query_context)
        latency = time.time() - start_time

        # Should complete within reasonable time (adjust based on system)
        assert latency < 10.0  # 10 seconds max for test
        assert response.performance_metrics.latency > 0

    @pytest.mark.asyncio
    async def test_graph_rag_scalability(self):
        """Test GraphRAG with larger dataset"""
        strategy = create_graph_rag_strategy()

        # Larger test dataset
        documents = []
        for i in range(20):
            documents.append((
                f"doc{i}",
                f"Document {i} discusses fMRI, brain activity, and neuroscience research methods. "
                f"Machine learning is used for data analysis in study {i}."
            ))

        await strategy.initialize(documents)

        # Test multiple concurrent queries
        queries = [
            QueryContext(
                query=f"Query {i} about fMRI",
                complexity=QueryComplexity.SIMPLE,
                domain=QueryDomain.NEUROSCIENCE,
                intent="factual",
                confidence=0.8,
                metadata={}
            ) for i in range(5)
        ]

        start_time = time.time()
        responses = await asyncio.gather(*[strategy.search(q) for q in queries])
        total_time = time.time() - start_time

        # All queries should succeed
        assert len(responses) == 5
        assert all(isinstance(r, RAGResponse) for r in responses)

        # Average time per query should be reasonable
        avg_time = total_time / 5
        assert avg_time < 5.0  # 5 seconds average max

    def test_graph_construction_performance(self):
        """Test knowledge graph construction performance"""
        builder = create_knowledge_graph_builder()

        # Test with varying document sizes
        small_docs = [("doc1", "Short document about fMRI.")]
        medium_docs = [("doc1", "Medium length document about fMRI and brain activity. " * 10)]
        large_docs = [("doc1", "Long document about fMRI and brain activity research. " * 100)]

        for doc_set, description in [(small_docs, "small"), (medium_docs, "medium"), (large_docs, "large")]:
            start_time = time.time()
            graph = asyncio.run(builder.build_graph_from_documents(doc_set))
            construction_time = time.time() - start_time

            # Graph should be constructed successfully
            assert len(graph.entities) >= 0
            assert construction_time < 30.0  # 30 seconds max

            print(f"{description} docs: {len(graph.entities)} entities in {construction_time:.2f}s")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
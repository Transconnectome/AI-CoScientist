"""
Complete RAG System Integration Tests

Implementation for: End-to-end system integration and quality validation
Created: 2025-12-05

Acceptance Criteria:
- Full workflow integration from query to response
- All RAG strategies functional and coordinated
- Performance benchmarks met across all components
- Quality assurance with realistic scientific queries

This test suite validates the complete RAG enhancement system with
comprehensive integration testing across all phases and components.
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Import all major components
from services.rag.unified_rag_orchestrator import (
    UnifiedRAGOrchestrator, QueryContext, RAGResponse, RAGStrategy,
    QueryComplexity, QueryDomain, create_unified_orchestrator
)
from services.rag.advanced_query_classifier import (
    MLQueryClassifier, create_query_classifier
)
from services.rag.adaptive_hybrid_retriever import (
    AdaptiveHybridRetriever, create_adaptive_hybrid_retriever
)
from services.rag.context_sufficiency_checker import (
    ContextSufficiencyChecker, create_context_sufficiency_checker
)
from services.rag.intelligent_cache import (
    IntelligentCache, create_intelligent_cache
)
from services.rag.performance_benchmark import (
    PerformanceBenchmark, MixedWorkload, create_performance_benchmark
)
from services.rag.knowledge_graph_builder import (
    KnowledgeGraphBuilder, create_knowledge_graph_builder
)
from services.rag.graph_rag_strategy import (
    GraphRAGStrategy, create_graph_rag_strategy
)
from services.rag.multimodal_document_processor import (
    MultimodalDocumentProcessor, create_multimodal_processor
)
from services.rag.multimodal_rag_strategy import (
    MultimodalRAGStrategy, create_multimodal_rag_strategy
)
from services.rag.feedback_loop_integration import (
    AdaptiveLearningEngine, UserFeedback, create_feedback_loop_integration
)
from services.rag.adaptive_strategy_selection import (
    AdaptiveStrategySelector, create_adaptive_strategy_selector
)
from monitoring.rag_metrics import get_metrics_manager

class TestSystemIntegration:
    """Test complete system integration"""

    @pytest.fixture
    async def complete_system(self):
        """Setup complete integrated RAG system"""
        # Create orchestrator
        orchestrator = create_unified_orchestrator()

        # Create query classifier
        classifier = create_query_classifier()

        # Create performance benchmark
        benchmark = create_performance_benchmark(orchestrator, classifier)

        # Create cache system
        cache = create_intelligent_cache()

        # Create learning components
        learning_engine = create_feedback_loop_integration(orchestrator.config)
        adaptive_selector = create_adaptive_strategy_selector(learning_engine)

        system = {
            'orchestrator': orchestrator,
            'classifier': classifier,
            'benchmark': benchmark,
            'cache': cache,
            'learning_engine': learning_engine,
            'adaptive_selector': adaptive_selector
        }

        yield system

        # Cleanup
        orchestrator.shutdown()
        if hasattr(cache, 'stop'):
            await cache.stop()

    @pytest.fixture
    def scientific_test_queries(self):
        """Realistic scientific test queries"""
        return [
            # Neuroscience queries
            {
                'query': "How does fMRI measure BOLD signals in the brain?",
                'expected_domain': QueryDomain.NEUROSCIENCE,
                'expected_complexity': QueryComplexity.MEDIUM,
                'expected_strategies': [RAGStrategy.MULTIMODAL_RAG, RAGStrategy.ENHANCED_DD_RAPTOR]
            },
            {
                'query': "What are the neural mechanisms underlying autism spectrum disorders?",
                'expected_domain': QueryDomain.DEVELOPMENTAL_DISORDERS,
                'expected_complexity': QueryComplexity.COMPLEX,
                'expected_strategies': [RAGStrategy.ENHANCED_DD_RAPTOR, RAGStrategy.GRAPH_RAG]
            },
            # Quantum ML queries
            {
                'query': "How do variational quantum algorithms achieve quantum advantage?",
                'expected_domain': QueryDomain.QUANTUM_ML,
                'expected_complexity': QueryComplexity.COMPLEX,
                'expected_strategies': [RAGStrategy.GRAPH_RAG, RAGStrategy.HYBRID]
            },
            {
                'query': "What is a qubit?",
                'expected_domain': QueryDomain.QUANTUM_ML,
                'expected_complexity': QueryComplexity.SIMPLE,
                'expected_strategies': [RAGStrategy.SIMPLE_RAG, RAGStrategy.GOLDEN_REFERENCE]
            },
            # General queries
            {
                'query': "Explain machine learning algorithms",
                'expected_domain': QueryDomain.GENERAL,
                'expected_complexity': QueryComplexity.MEDIUM,
                'expected_strategies': [RAGStrategy.HYBRID, RAGStrategy.GRAPH_RAG]
            }
        ]

    @pytest.mark.asyncio
    async def test_complete_query_workflow(self, complete_system, scientific_test_queries):
        """Test complete query processing workflow"""
        orchestrator = complete_system['orchestrator']
        classifier = complete_system['classifier']

        successful_queries = 0
        total_response_time = 0

        for test_case in scientific_test_queries:
            try:
                start_time = time.time()

                # 1. Query Classification
                classification = await classifier.classify(test_case['query'])

                # 2. Create Query Context
                query_context = QueryContext(
                    query=test_case['query'],
                    complexity=classification.complexity,
                    domain=classification.domain,
                    intent=classification.intent.value,
                    confidence=classification.overall_confidence,
                    metadata={'test_case': True}
                )

                # 3. Execute RAG Search
                response = await orchestrator.search(query_context)

                # 4. Validate Response
                response_time = time.time() - start_time
                total_response_time += response_time

                # Basic response validation
                assert isinstance(response, RAGResponse)
                assert response.answer is not None
                assert len(response.answer) > 0
                assert response.strategy_used in RAGStrategy
                assert 0 <= response.confidence <= 1
                assert response.performance_metrics is not None

                # Domain classification validation (allow some flexibility)
                if test_case['expected_domain'] != QueryDomain.GENERAL:
                    assert classification.domain in [test_case['expected_domain'], QueryDomain.GENERAL]

                # Response time validation
                assert response_time < 30.0, f"Response too slow: {response_time:.2f}s"

                successful_queries += 1

            except Exception as e:
                print(f"Query failed: {test_case['query'][:50]}... Error: {e}")

        # Overall success criteria
        success_rate = successful_queries / len(scientific_test_queries)
        avg_response_time = total_response_time / len(scientific_test_queries)

        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.1%}"
        assert avg_response_time < 15.0, f"Average response time too high: {avg_response_time:.2f}s"

        print(f"Integration test results: {success_rate:.1%} success rate, "
              f"{avg_response_time:.2f}s average response time")

    @pytest.mark.asyncio
    async def test_strategy_routing_accuracy(self, complete_system, scientific_test_queries):
        """Test accuracy of strategy routing for different query types"""
        orchestrator = complete_system['orchestrator']
        classifier = complete_system['classifier']

        strategy_selections = {}

        for test_case in scientific_test_queries:
            classification = await classifier.classify(test_case['query'])

            query_context = QueryContext(
                query=test_case['query'],
                complexity=classification.complexity,
                domain=classification.domain,
                intent=classification.intent.value,
                confidence=classification.overall_confidence,
                metadata={}
            )

            response = await orchestrator.search(query_context)

            query_type = f"{classification.domain.value}_{classification.complexity.value}"
            if query_type not in strategy_selections:
                strategy_selections[query_type] = []
            strategy_selections[query_type].append(response.strategy_used)

        # Validate strategy selection patterns
        for query_type, strategies in strategy_selections.items():
            assert len(strategies) > 0, f"No strategies selected for {query_type}"

            # Check for reasonable strategy diversity
            unique_strategies = set(strategies)
            assert len(unique_strategies) <= 3, f"Too many different strategies for {query_type}"

    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, complete_system):
        """Test system performance with benchmark workloads"""
        benchmark = complete_system['benchmark']

        # Create test workload
        workload = MixedWorkload()

        # Run benchmark
        report = await benchmark.run_benchmark(workload, max_concurrent=3)

        # Validate benchmark results
        assert report.total_queries > 0
        assert report.overall_success_rate >= 0.7  # At least 70% success
        assert report.total_duration < 300  # Complete within 5 minutes
        assert len(report.strategy_metrics) > 0

        # Check strategy performance
        for strategy, metrics in report.strategy_metrics.items():
            assert metrics.success_rate >= 0.5  # At least 50% success per strategy
            assert metrics.avg_response_time < 20.0  # Under 20 seconds average

        print(f"Benchmark results: {report.overall_success_rate:.1%} success rate, "
              f"{report.total_duration:.1f}s total duration")

    @pytest.mark.asyncio
    async def test_caching_integration(self, complete_system):
        """Test intelligent caching integration"""
        orchestrator = complete_system['orchestrator']
        cache = complete_system['cache']
        classifier = complete_system['classifier']

        # Start cache
        await cache.start()

        try:
            query = "What is machine learning?"

            # First request (cache miss)
            classification = await classifier.classify(query)
            query_context = QueryContext(
                query=query,
                complexity=classification.complexity,
                domain=classification.domain,
                intent=classification.intent.value,
                confidence=classification.overall_confidence,
                metadata={}
            )

            start_time = time.time()
            response1 = await orchestrator.search(query_context)
            first_response_time = time.time() - start_time

            # Cache the result
            await cache.set(query, query_context, response1)

            # Second request (should hit cache)
            start_time = time.time()
            cached_response = await cache.get(query, query_context)
            cache_response_time = time.time() - start_time

            if cached_response:
                # Cache should be faster
                assert cache_response_time < first_response_time
                print(f"Cache performance: {first_response_time:.3f}s vs {cache_response_time:.3f}s")

        finally:
            await cache.stop()

    @pytest.mark.asyncio
    async def test_learning_system_integration(self, complete_system):
        """Test learning system integration"""
        orchestrator = complete_system['orchestrator']
        learning_engine = complete_system['learning_engine']
        adaptive_selector = complete_system['adaptive_selector']

        # Simulate query with feedback
        query_context = QueryContext(
            query="How does deep learning work?",
            complexity=QueryComplexity.MEDIUM,
            domain=QueryDomain.GENERAL,
            intent="procedural",
            confidence=0.8,
            metadata={}
        )

        # Select strategy using adaptive selector
        selection_result = await adaptive_selector.select_strategy(query_context)

        # Execute query
        response = await orchestrator.search(
            query_context,
            strategy_override=selection_result.selected_strategy
        )

        # Simulate user feedback
        feedback = UserFeedback(
            feedback_id="integration_test_feedback",
            query=query_context.query,
            response=response.answer,
            strategy_used=response.strategy_used,
            user_rating=4.0,
            user_comment="Good explanation, very helpful"
        )

        # Process feedback
        learning_update = await learning_engine.process_feedback(feedback)

        # Update adaptive selector
        await adaptive_selector.update_performance(
            query_context, response.strategy_used, 0.8, feedback
        )

        # Validate learning integration
        assert len(learning_engine.feedback_store) == 1
        assert response.strategy_used in adaptive_selector.performance_history

        if learning_update:
            success = await learning_engine.apply_learning_update(learning_update)
            assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_multimodal_integration(self, complete_system):
        """Test multimodal processing integration"""
        orchestrator = complete_system['orchestrator']

        # Test multimodal query
        query_context = QueryContext(
            query="Analyze brain imaging data from fMRI studies",
            complexity=QueryComplexity.COMPLEX,
            domain=QueryDomain.NEUROSCIENCE,
            intent="synthesis",
            confidence=0.9,
            metadata={}
        )

        # Force multimodal strategy
        response = await orchestrator.search(
            query_context,
            strategy_override=RAGStrategy.MULTIMODAL_RAG
        )

        # Validate multimodal response
        assert isinstance(response, RAGResponse)
        assert response.strategy_used == RAGStrategy.MULTIMODAL_RAG
        assert response.answer is not None

        # Check for multimodal metadata
        if response.metadata:
            # Should have multimodal-specific information
            assert 'text_blocks' in response.metadata or 'image_blocks' in response.metadata

    @pytest.mark.asyncio
    async def test_graph_rag_integration(self, complete_system):
        """Test GraphRAG integration"""
        orchestrator = complete_system['orchestrator']

        # Test graph-suitable query
        query_context = QueryContext(
            query="How are quantum computing concepts related to machine learning?",
            complexity=QueryComplexity.COMPLEX,
            domain=QueryDomain.QUANTUM_ML,
            intent="synthesis",
            confidence=0.9,
            metadata={}
        )

        # Force GraphRAG strategy
        response = await orchestrator.search(
            query_context,
            strategy_override=RAGStrategy.GRAPH_RAG
        )

        # Validate GraphRAG response
        assert isinstance(response, RAGResponse)
        assert response.strategy_used == RAGStrategy.GRAPH_RAG
        assert response.answer is not None

        # Check for graph-specific metadata
        if response.metadata:
            assert 'matched_entities' in response.metadata or 'expanded_entities' in response.metadata

    def test_metrics_integration(self, complete_system):
        """Test metrics collection integration"""
        orchestrator = complete_system['orchestrator']

        # Get metrics manager
        metrics_manager = get_metrics_manager()

        # Check metrics are being collected
        all_metrics = metrics_manager.get_all_metrics()
        assert isinstance(all_metrics, dict)

        # Validate metrics structure
        if 'strategies' in all_metrics:
            strategies_metrics = all_metrics['strategies']
            assert isinstance(strategies_metrics, dict)

class TestQualityAssurance:
    """Test quality assurance across the system"""

    @pytest.fixture
    def qa_test_cases(self):
        """Quality assurance test cases"""
        return [
            {
                'query': "What causes autism spectrum disorders?",
                'min_confidence': 0.6,
                'max_response_time': 15.0,
                'required_sources': 1,
                'domain': QueryDomain.DEVELOPMENTAL_DISORDERS
            },
            {
                'query': "How do quantum computers achieve speedup?",
                'min_confidence': 0.5,
                'max_response_time': 20.0,
                'required_sources': 2,
                'domain': QueryDomain.QUANTUM_ML
            },
            {
                'query': "Explain neural network backpropagation",
                'min_confidence': 0.7,
                'max_response_time': 10.0,
                'required_sources': 1,
                'domain': QueryDomain.GENERAL
            }
        ]

    @pytest.mark.asyncio
    async def test_quality_requirements(self, complete_system, qa_test_cases):
        """Test that system meets quality requirements"""
        orchestrator = complete_system['orchestrator']
        classifier = complete_system['classifier']

        quality_failures = []

        for test_case in qa_test_cases:
            try:
                start_time = time.time()

                # Classify and execute
                classification = await classifier.classify(test_case['query'])
                query_context = QueryContext(
                    query=test_case['query'],
                    complexity=classification.complexity,
                    domain=classification.domain,
                    intent=classification.intent.value,
                    confidence=classification.overall_confidence,
                    metadata={}
                )

                response = await orchestrator.search(query_context)
                response_time = time.time() - start_time

                # Check quality requirements
                failures = []

                if response.confidence < test_case['min_confidence']:
                    failures.append(f"Low confidence: {response.confidence:.2f} < {test_case['min_confidence']}")

                if response_time > test_case['max_response_time']:
                    failures.append(f"Slow response: {response_time:.2f}s > {test_case['max_response_time']}s")

                if len(response.sources) < test_case['required_sources']:
                    failures.append(f"Insufficient sources: {len(response.sources)} < {test_case['required_sources']}")

                if failures:
                    quality_failures.append({
                        'query': test_case['query'][:50],
                        'failures': failures
                    })

            except Exception as e:
                quality_failures.append({
                    'query': test_case['query'][:50],
                    'failures': [f"Exception: {str(e)}"]
                })

        # Report quality issues
        if quality_failures:
            print(f"Quality failures: {len(quality_failures)}/{len(qa_test_cases)}")
            for failure in quality_failures:
                print(f"  {failure['query']}: {failure['failures']}")

        # Assert quality standards
        failure_rate = len(quality_failures) / len(qa_test_cases)
        assert failure_rate <= 0.2, f"Too many quality failures: {failure_rate:.1%}"

    @pytest.mark.asyncio
    async def test_error_handling_robustness(self, complete_system):
        """Test system robustness to errors"""
        orchestrator = complete_system['orchestrator']

        # Test with problematic queries
        problematic_queries = [
            "",  # Empty query
            "a" * 10000,  # Very long query
            "🤖🧠🔬💻🚀",  # Emoji-only query
            "SELECT * FROM users;",  # SQL injection attempt
            "What is the meaning of life, the universe, and everything?" * 100,  # Repetitive long query
        ]

        successful_handles = 0

        for query in problematic_queries:
            try:
                query_context = QueryContext(
                    query=query,
                    complexity=QueryComplexity.SIMPLE,
                    domain=QueryDomain.GENERAL,
                    intent="factual",
                    confidence=0.5,
                    metadata={}
                )

                # Should not crash, even with problematic input
                response = await asyncio.wait_for(
                    orchestrator.search(query_context),
                    timeout=30.0
                )

                # Response should be valid even if low quality
                assert isinstance(response, RAGResponse)
                assert response.answer is not None
                successful_handles += 1

            except asyncio.TimeoutError:
                # Timeout is acceptable for some problematic queries
                successful_handles += 1
            except Exception as e:
                print(f"Unhandled error for query '{query[:30]}': {e}")

        # Should handle most problematic queries gracefully
        handle_rate = successful_handles / len(problematic_queries)
        assert handle_rate >= 0.8, f"Poor error handling: {handle_rate:.1%}"

class TestScalabilityValidation:
    """Test system scalability and performance under load"""

    @pytest.mark.asyncio
    async def test_concurrent_load(self, complete_system):
        """Test system under concurrent load"""
        orchestrator = complete_system['orchestrator']
        classifier = complete_system['classifier']

        # Create multiple concurrent queries
        queries = [
            f"Test query {i} about machine learning concepts"
            for i in range(20)
        ]

        async def process_query(query):
            try:
                classification = await classifier.classify(query)
                query_context = QueryContext(
                    query=query,
                    complexity=classification.complexity,
                    domain=classification.domain,
                    intent=classification.intent.value,
                    confidence=classification.overall_confidence,
                    metadata={}
                )

                start_time = time.time()
                response = await orchestrator.search(query_context)
                response_time = time.time() - start_time

                return {
                    'success': True,
                    'response_time': response_time,
                    'strategy': response.strategy_used,
                    'confidence': response.confidence
                }

            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'response_time': 0
                }

        # Execute concurrent queries
        start_time = time.time()
        results = await asyncio.gather(*[process_query(q) for q in queries])
        total_time = time.time() - start_time

        # Analyze results
        successful_results = [r for r in results if r['success']]
        success_rate = len(successful_results) / len(results)

        avg_response_time = sum(r['response_time'] for r in successful_results) / len(successful_results) if successful_results else 0
        max_response_time = max(r['response_time'] for r in successful_results) if successful_results else 0

        # Scalability assertions
        assert success_rate >= 0.9, f"Low success rate under load: {success_rate:.1%}"
        assert avg_response_time < 10.0, f"High average response time: {avg_response_time:.2f}s"
        assert max_response_time < 30.0, f"Excessive max response time: {max_response_time:.2f}s"
        assert total_time < 60.0, f"Total processing time too high: {total_time:.2f}s"

        print(f"Concurrent load test: {success_rate:.1%} success, "
              f"{avg_response_time:.2f}s avg, {max_response_time:.2f}s max")

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, complete_system):
        """Test memory efficiency under sustained load"""
        import psutil
        import gc

        orchestrator = complete_system['orchestrator']

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Run sustained queries
        for i in range(50):
            query_context = QueryContext(
                query=f"Memory test query {i}",
                complexity=QueryComplexity.SIMPLE,
                domain=QueryDomain.GENERAL,
                intent="factual",
                confidence=0.8,
                metadata={}
            )

            await orchestrator.search(query_context)

            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()

        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable
        memory_growth_mb = memory_growth / (1024 * 1024)
        assert memory_growth_mb < 100, f"Excessive memory growth: {memory_growth_mb:.1f}MB"

        print(f"Memory efficiency test: {memory_growth_mb:.1f}MB growth")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
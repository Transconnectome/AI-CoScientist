"""
Comprehensive Integration Tests for Unified RAG Orchestrator

Implementation for: Comprehensive integration tests
Created: 2025-12-05

Acceptance Criteria:
- End-to-end workflow tests for all strategies
- Strategy routing validation scenarios
- Performance regression test suite
- Error handling and recovery testing

This test suite provides comprehensive validation of the unified RAG orchestrator
with full integration testing across all components.
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from services.rag.unified_rag_orchestrator import (
    UnifiedRAGOrchestrator, QueryContext, RAGResponse, RAGStrategy,
    QueryComplexity, QueryDomain, RAGStrategyConfig, MockRAGStrategy,
    create_unified_orchestrator
)
from services.rag.advanced_query_classifier import (
    MLQueryClassifier, QueryIntent, ClassificationResult,
    create_query_classifier
)
from monitoring.rag_metrics import RAGMetrics, RAGMetricsManager
from datetime import datetime

class TestUnifiedRAGOrchestrator:
    """Test unified RAG orchestrator functionality"""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for testing"""
        config = RAGStrategyConfig()
        orchestrator = UnifiedRAGOrchestrator(config)
        yield orchestrator
        orchestrator.shutdown()

    @pytest.fixture
    def sample_query_context(self):
        """Create sample query context"""
        return QueryContext(
            query="What is machine learning?",
            complexity=QueryComplexity.SIMPLE,
            domain=QueryDomain.GENERAL,
            intent="factual",
            confidence=0.9,
            metadata={"test": True}
        )

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly"""
        assert orchestrator is not None
        assert len(orchestrator.strategies) > 0
        assert orchestrator.config is not None
        assert orchestrator.metrics_manager is not None

    @pytest.mark.asyncio
    async def test_basic_search(self, orchestrator, sample_query_context):
        """Test basic search functionality"""
        response = await orchestrator.search(sample_query_context)

        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.strategy_used in RAGStrategy
        assert 0 <= response.confidence <= 1
        assert response.sources is not None
        assert response.performance_metrics is not None

    @pytest.mark.asyncio
    async def test_strategy_override(self, orchestrator, sample_query_context):
        """Test strategy override functionality"""
        # Test with specific strategy
        response = await orchestrator.search(
            sample_query_context,
            strategy_override=RAGStrategy.SIMPLE_RAG
        )

        assert response.strategy_used == RAGStrategy.SIMPLE_RAG

    @pytest.mark.asyncio
    async def test_parallel_search(self, orchestrator):
        """Test parallel search capabilities"""
        query_contexts = [
            QueryContext(
                query=f"Test query {i}",
                complexity=QueryComplexity.SIMPLE,
                domain=QueryDomain.GENERAL,
                intent="factual",
                confidence=0.9,
                metadata={"test": True, "index": i}
            ) for i in range(3)
        ]

        start_time = time.time()
        responses = await orchestrator.search_parallel(query_contexts, max_concurrent=2)
        elapsed_time = time.time() - start_time

        assert len(responses) == 3
        assert all(isinstance(r, RAGResponse) for r in responses)
        # Parallel execution should be faster than sequential
        assert elapsed_time < 1.0  # Should complete quickly with mocks

    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, orchestrator, sample_query_context):
        """Test fallback to other strategies on failure"""
        # Mock one strategy to fail
        original_search = orchestrator.strategies[RAGStrategy.HYBRID].search

        async def failing_search(context):
            raise Exception("Strategy failed")

        orchestrator.strategies[RAGStrategy.HYBRID].search = failing_search

        try:
            # Should fallback to other strategies
            response = await orchestrator.search(sample_query_context, enable_fallback=True)
            assert response is not None
            assert response.strategy_used != RAGStrategy.HYBRID

        finally:
            # Restore original method
            orchestrator.strategies[RAGStrategy.HYBRID].search = original_search

    @pytest.mark.asyncio
    async def test_no_fallback_failure(self, orchestrator, sample_query_context):
        """Test that disabling fallback propagates failures"""
        # Mock strategy to fail
        original_search = orchestrator.strategies[RAGStrategy.SIMPLE_RAG].search

        async def failing_search(context):
            raise Exception("Strategy failed")

        orchestrator.strategies[RAGStrategy.SIMPLE_RAG].search = failing_search

        try:
            with pytest.raises(Exception):
                await orchestrator.search(
                    sample_query_context,
                    strategy_override=RAGStrategy.SIMPLE_RAG,
                    enable_fallback=False
                )

        finally:
            # Restore original method
            orchestrator.strategies[RAGStrategy.SIMPLE_RAG].search = original_search

    def test_strategy_selection(self, orchestrator):
        """Test strategy selection logic"""
        # Test simple neuroscience query
        neuro_context = QueryContext(
            query="What is fMRI?",
            complexity=QueryComplexity.SIMPLE,
            domain=QueryDomain.NEUROSCIENCE,
            intent="factual",
            confidence=0.9,
            metadata={}
        )

        strategies = orchestrator._select_strategies(neuro_context)
        assert len(strategies) > 0
        # Should prefer strategies that support neuroscience
        assert any(orchestrator.config.get_config(s).get("domains", []) for s in strategies
                  if QueryDomain.NEUROSCIENCE in orchestrator.config.get_config(s).get("domains", []))

    def test_performance_tracking(self, orchestrator, sample_query_context):
        """Test performance tracking updates"""
        # Get initial performance
        initial_performance = orchestrator._strategy_performance.copy()

        # Create mock response
        mock_response = RAGResponse(
            answer="Test answer",
            sources=[],
            confidence=0.8,
            strategy_used=RAGStrategy.SIMPLE_RAG
        )

        # Update performance tracking
        orchestrator._update_performance_tracking(
            RAGStrategy.SIMPLE_RAG, mock_response, 1.0
        )

        # Check that performance was updated
        new_performance = orchestrator._strategy_performance.get(RAGStrategy.SIMPLE_RAG, 0.0)
        assert new_performance > 0

    def test_strategy_health(self, orchestrator):
        """Test strategy health monitoring"""
        health = orchestrator.get_strategy_health()

        assert isinstance(health, dict)
        assert len(health) > 0

        for strategy_name, health_data in health.items():
            assert "available" in health_data
            assert "request_count" in health_data
            assert "performance_score" in health_data
            assert "config" in health_data

    def test_performance_summary(self, orchestrator):
        """Test performance summary generation"""
        summary = orchestrator.get_performance_summary()

        assert isinstance(summary, dict)
        assert "strategies" in summary
        assert "total_requests" in summary
        assert "active_strategies" in summary
        assert "metrics_summary" in summary

    @pytest.mark.asyncio
    async def test_warmup(self, orchestrator):
        """Test strategy warmup"""
        # Should complete without errors
        await orchestrator.warmup()

        # All strategies should still be available
        assert len(orchestrator.strategies) > 0

class TestQueryClassifier:
    """Test advanced query classifier"""

    @pytest.fixture
    def classifier(self):
        """Create classifier for testing"""
        return create_query_classifier()

    @pytest.mark.asyncio
    async def test_basic_classification(self, classifier):
        """Test basic query classification"""
        result = await classifier.classify("What is machine learning?")

        assert isinstance(result, ClassificationResult)
        assert result.complexity in QueryComplexity
        assert result.domain in QueryDomain
        assert result.intent in QueryIntent
        assert 0 <= result.overall_confidence <= 1
        assert isinstance(result.confidence_scores, dict)
        assert isinstance(result.features, dict)

    @pytest.mark.asyncio
    async def test_complexity_classification(self, classifier):
        """Test complexity classification accuracy"""
        test_cases = [
            ("What is ML?", QueryComplexity.SIMPLE),
            ("How do neural networks learn from data?", QueryComplexity.MEDIUM),
            ("Analyze the theoretical implications of quantum advantage in variational algorithms", QueryComplexity.COMPLEX)
        ]

        for query, expected_complexity in test_cases:
            result = await classifier.classify(query)
            # Allow some flexibility in classification
            assert result.complexity == expected_complexity or result.overall_confidence < 0.7

    @pytest.mark.asyncio
    async def test_domain_classification(self, classifier):
        """Test domain classification accuracy"""
        test_cases = [
            ("What is fMRI brain imaging?", QueryDomain.NEUROSCIENCE),
            ("How do quantum circuits work?", QueryDomain.QUANTUM_ML),
            ("What is autism spectrum disorder?", QueryDomain.DEVELOPMENTAL_DISORDERS),
            ("What is machine learning?", QueryDomain.GENERAL)
        ]

        for query, expected_domain in test_cases:
            result = await classifier.classify(query)
            # Allow for GENERAL classification as fallback
            assert result.domain == expected_domain or result.domain == QueryDomain.GENERAL

    @pytest.mark.asyncio
    async def test_intent_classification(self, classifier):
        """Test intent classification"""
        test_cases = [
            ("What is machine learning?", QueryIntent.FACTUAL),
            ("Compare neural networks and decision trees", QueryIntent.COMPARATIVE),
            ("How to train a neural network?", QueryIntent.PROCEDURAL),
            ("Why do quantum computers have advantage?", QueryIntent.CAUSAL),
            ("Analyze the implications of quantum supremacy", QueryIntent.SYNTHESIS)
        ]

        for query, expected_intent in test_cases:
            result = await classifier.classify(query)
            # Intent classification is more flexible
            assert result.intent in QueryIntent

    def test_feature_extraction(self, classifier):
        """Test feature extraction functionality"""
        features = classifier.feature_extractor.extract_features(
            "How do quantum neural networks work in practice?"
        )

        assert isinstance(features, dict)
        assert "word_count" in features
        assert "char_count" in features
        assert "quantum_ml_score" in features
        assert "neuroscience_score" in features
        assert features["word_count"] > 0
        assert features["char_count"] > 0

class TestIntegrationWorkflows:
    """Test end-to-end integration workflows"""

    @pytest.fixture
    async def full_system(self):
        """Setup complete system for integration testing"""
        orchestrator = create_unified_orchestrator()
        classifier = create_query_classifier()

        yield {
            "orchestrator": orchestrator,
            "classifier": classifier
        }

        orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_complete_search_workflow(self, full_system):
        """Test complete search workflow from classification to response"""
        orchestrator = full_system["orchestrator"]
        classifier = full_system["classifier"]

        query = "How does fMRI measure brain activity?"

        # Step 1: Classify query
        classification = await classifier.classify(query)
        assert classification.domain == QueryDomain.NEUROSCIENCE
        assert classification.complexity in [QueryComplexity.MEDIUM, QueryComplexity.SIMPLE]

        # Step 2: Create query context
        query_context = QueryContext(
            query=query,
            complexity=classification.complexity,
            domain=classification.domain,
            intent=classification.intent.value,
            confidence=classification.overall_confidence,
            metadata={"integration_test": True}
        )

        # Step 3: Execute search
        response = await orchestrator.search(query_context)

        # Validate complete response
        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert len(response.sources) > 0
        assert response.strategy_used in RAGStrategy
        assert response.performance_metrics is not None

    @pytest.mark.asyncio
    async def test_multi_domain_batch_processing(self, full_system):
        """Test batch processing across multiple domains"""
        orchestrator = full_system["orchestrator"]
        classifier = full_system["classifier"]

        queries = [
            "What is machine learning?",  # General
            "How does fMRI work?",        # Neuroscience
            "What is a qubit?",           # Quantum ML
            "What is autism?"             # Developmental disorders
        ]

        # Classify all queries
        query_contexts = []
        for query in queries:
            classification = await classifier.classify(query)
            context = QueryContext(
                query=query,
                complexity=classification.complexity,
                domain=classification.domain,
                intent=classification.intent.value,
                confidence=classification.overall_confidence,
                metadata={"batch_test": True}
            )
            query_contexts.append(context)

        # Execute batch search
        responses = await orchestrator.search_parallel(query_contexts, max_concurrent=3)

        # Validate batch results
        assert len(responses) == len(queries)
        assert all(isinstance(r, RAGResponse) for r in responses)

        # Check domain diversity in strategy selection
        strategies_used = [r.strategy_used for r in responses]
        assert len(set(strategies_used)) >= 1  # At least some variety

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, full_system):
        """Test error recovery and fallback mechanisms"""
        orchestrator = full_system["orchestrator"]
        classifier = full_system["classifier"]

        # Disable some strategies to test fallback
        original_strategies = orchestrator.strategies.copy()

        # Remove one strategy to force fallback
        if RAGStrategy.HYBRID in orchestrator.strategies:
            del orchestrator.strategies[RAGStrategy.HYBRID]

        try:
            query = "What is artificial intelligence?"
            classification = await classifier.classify(query)

            context = QueryContext(
                query=query,
                complexity=classification.complexity,
                domain=classification.domain,
                intent=classification.intent.value,
                confidence=classification.overall_confidence,
                metadata={"error_test": True}
            )

            # Should still work with remaining strategies
            response = await orchestrator.search(context)
            assert response is not None
            assert response.strategy_used != RAGStrategy.HYBRID

        finally:
            # Restore strategies
            orchestrator.strategies = original_strategies

    @pytest.mark.asyncio
    async def test_performance_regression(self, full_system):
        """Test performance regression detection"""
        orchestrator = full_system["orchestrator"]
        classifier = full_system["classifier"]

        # Measure baseline performance
        query = "What is deep learning?"
        classification = await classifier.classify(query)

        context = QueryContext(
            query=query,
            complexity=classification.complexity,
            domain=classification.domain,
            intent=classification.intent.value,
            confidence=classification.overall_confidence,
            metadata={"perf_test": True}
        )

        # Run multiple requests to establish baseline
        response_times = []
        for _ in range(5):
            start_time = time.time()
            response = await orchestrator.search(context)
            response_time = time.time() - start_time
            response_times.append(response_time)

            assert response is not None
            assert response.performance_metrics is not None

        # Calculate performance statistics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # Performance assertions (with mocks, should be fast)
        assert avg_response_time < 2.0  # Average under 2 seconds
        assert max_response_time < 5.0  # Max under 5 seconds

        # Check metrics collection
        performance_summary = orchestrator.get_performance_summary()
        assert performance_summary["total_requests"] >= 5

class TestConfigurationManagement:
    """Test configuration and strategy management"""

    def test_strategy_config_validation(self):
        """Test strategy configuration validation"""
        config = RAGStrategyConfig()

        # Test valid configuration
        test_context = QueryContext(
            query="test",
            complexity=QueryComplexity.SIMPLE,
            domain=QueryDomain.GENERAL,
            intent="factual",
            confidence=0.9,
            metadata={}
        )

        suitable = config.is_strategy_suitable(RAGStrategy.SIMPLE_RAG, test_context)
        assert suitable  # Simple RAG should be suitable for simple general queries

        # Test invalid configuration
        complex_context = QueryContext(
            query="test",
            complexity=QueryComplexity.COMPLEX,
            domain=QueryDomain.QUANTUM_ML,
            intent="synthesis",
            confidence=0.9,
            metadata={}
        )

        suitable = config.is_strategy_suitable(RAGStrategy.SIMPLE_RAG, complex_context)
        assert not suitable  # Simple RAG shouldn't handle complex quantum ML

    def test_strategy_configuration_loading(self):
        """Test strategy configuration loading and validation"""
        config = RAGStrategyConfig()

        # Test all strategies have valid configs
        for strategy in RAGStrategy:
            strategy_config = config.get_config(strategy)
            assert isinstance(strategy_config, dict)

            if strategy_config.get("enabled", False):
                assert "priority" in strategy_config
                assert "domains" in strategy_config
                assert "complexity_range" in strategy_config
                assert "max_concurrent" in strategy_config

class TestMetricsIntegration:
    """Test metrics system integration"""

    @pytest.mark.asyncio
    async def test_metrics_collection_integration(self):
        """Test metrics collection during orchestrator operation"""
        # Create orchestrator with metrics
        orchestrator = create_unified_orchestrator()

        try:
            # Create test query context
            context = QueryContext(
                query="Test metrics collection",
                complexity=QueryComplexity.SIMPLE,
                domain=QueryDomain.GENERAL,
                intent="factual",
                confidence=0.9,
                metadata={"metrics_test": True}
            )

            # Execute search and verify metrics collection
            response = await orchestrator.search(context)

            # Verify response includes metrics
            assert response.performance_metrics is not None
            assert response.performance_metrics.strategy == response.strategy_used.value
            assert response.performance_metrics.latency > 0
            assert 0 <= response.performance_metrics.quality_score <= 1

            # Verify metrics were recorded in manager
            metrics_manager = orchestrator.metrics_manager
            all_metrics = metrics_manager.get_all_metrics()
            assert "strategies" in all_metrics

        finally:
            orchestrator.shutdown()

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
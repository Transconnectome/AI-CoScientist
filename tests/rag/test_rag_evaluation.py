"""
Comprehensive test suite for RAG evaluation system

Tests for: Create comprehensive test suite for evaluation
Created: 2025-12-04

Acceptance Criteria:
- Unit tests for all RAGAS metrics (coverage ≥ 80%)
- Integration tests with mock data scenarios
- Performance benchmarks established
- Error handling validation
"""

import pytest
import asyncio
import sys
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from services.rag.rag_evaluator import (
    RAGEvaluator,
    RAGEvaluationResult,
    EvaluationInput,
    create_rag_evaluator,
    evaluate_rag_pipeline
)


class TestRAGEvaluationResult:
    """Test RAGEvaluationResult dataclass"""

    def test_result_creation(self):
        """Test basic result creation"""
        result = RAGEvaluationResult(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7
        )
        assert result.faithfulness == 0.8
        assert result.answer_relevancy == 0.9
        assert result.context_precision == 0.7
        assert result.context_recall is None
        assert result.overall_score is None

    def test_result_with_optional_fields(self):
        """Test result creation with all fields"""
        metadata = {"model": "gpt-4", "version": "1.0"}
        result = RAGEvaluationResult(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.85,
            overall_score=0.82,
            evaluation_time=0.5,
            metadata=metadata
        )
        assert result.context_recall == 0.85
        assert result.overall_score == 0.82
        assert result.evaluation_time == 0.5
        assert result.metadata == metadata


class TestEvaluationInput:
    """Test EvaluationInput dataclass"""

    def test_input_creation(self):
        """Test basic input creation"""
        input_data = EvaluationInput(
            query="What is AI?",
            contexts=["AI is artificial intelligence."],
            answer="AI stands for artificial intelligence."
        )
        assert input_data.query == "What is AI?"
        assert input_data.contexts == ["AI is artificial intelligence."]
        assert input_data.answer == "AI stands for artificial intelligence."
        assert input_data.ground_truth is None

    def test_input_with_ground_truth(self):
        """Test input creation with ground truth"""
        input_data = EvaluationInput(
            query="What is AI?",
            contexts=["AI is artificial intelligence."],
            answer="AI stands for artificial intelligence.",
            ground_truth="Artificial Intelligence (AI) refers to machines that can think."
        )
        assert input_data.ground_truth == "Artificial Intelligence (AI) refers to machines that can think."


class TestRAGEvaluator:
    """Test RAGEvaluator class"""

    def test_evaluator_initialization_default(self):
        """Test default evaluator initialization"""
        evaluator = RAGEvaluator()
        assert evaluator.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert evaluator.fallback_to_simple is True
        # RAGAS availability depends on environment
        assert isinstance(evaluator.enable_ragas, bool)

    def test_evaluator_initialization_custom(self):
        """Test custom evaluator initialization"""
        evaluator = RAGEvaluator(
            embedding_model="custom-model",
            enable_ragas=False,
            fallback_to_simple=False
        )
        assert evaluator.embedding_model_name == "custom-model"
        assert evaluator.enable_ragas is False
        assert evaluator.fallback_to_simple is False

    @pytest.mark.asyncio
    async def test_evaluate_single_basic(self):
        """Test single evaluation with basic metrics"""
        evaluator = RAGEvaluator(enable_ragas=False)  # Force fallback mode

        result = await evaluator.evaluate_single(
            query="What is the capital of France?",
            contexts=["Paris is the capital of France.", "France is in Europe."],
            answer="The capital of France is Paris."
        )

        assert isinstance(result, RAGEvaluationResult)
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevancy <= 1.0
        assert 0.0 <= result.context_precision <= 1.0
        assert result.overall_score is not None
        assert result.evaluation_time is not None
        assert result.evaluation_time > 0

    @pytest.mark.asyncio
    async def test_evaluate_single_with_ground_truth(self):
        """Test single evaluation with ground truth"""
        evaluator = RAGEvaluator(enable_ragas=False)

        result = await evaluator.evaluate_single(
            query="What is machine learning?",
            contexts=["Machine learning is a subset of AI."],
            answer="ML is artificial intelligence.",
            ground_truth="Machine learning is a method of data analysis."
        )

        assert isinstance(result, RAGEvaluationResult)
        assert result.overall_score is not None

    @pytest.mark.asyncio
    async def test_evaluate_batch(self):
        """Test batch evaluation"""
        evaluator = RAGEvaluator(enable_ragas=False)

        inputs = [
            EvaluationInput(
                query="What is AI?",
                contexts=["AI is artificial intelligence."],
                answer="AI means artificial intelligence."
            ),
            EvaluationInput(
                query="What is ML?",
                contexts=["ML is machine learning."],
                answer="ML stands for machine learning."
            )
        ]

        results = await evaluator.evaluate_batch(inputs)

        assert len(results) == 2
        assert all(isinstance(r, RAGEvaluationResult) for r in results)
        assert all(r.overall_score is not None for r in results)
        assert all(r.overall_score >= 0.0 for r in results)
        assert all(r.overall_score <= 1.0 for r in results)

    def test_calculate_word_overlap(self):
        """Test word overlap calculation"""
        evaluator = RAGEvaluator()

        # Perfect overlap
        overlap = evaluator._calculate_word_overlap("hello world", "hello world")
        assert overlap == 1.0

        # Partial overlap
        overlap = evaluator._calculate_word_overlap("hello world", "hello there")
        assert overlap == 0.5

        # No overlap
        overlap = evaluator._calculate_word_overlap("hello", "goodbye")
        assert overlap == 0.0

        # Empty text
        overlap = evaluator._calculate_word_overlap("", "hello")
        assert overlap == 0.0

    def test_calculate_context_precision_heuristic(self):
        """Test context precision heuristic calculation"""
        evaluator = RAGEvaluator()

        query = "machine learning algorithms"
        contexts = [
            "Machine learning uses algorithms",
            "Artificial intelligence is different",
            "Algorithms are used in machine learning"
        ]

        precision = evaluator._calculate_context_precision_heuristic(query, contexts)
        assert 0.0 <= precision <= 1.0

        # Empty query
        precision = evaluator._calculate_context_precision_heuristic("", contexts)
        assert precision == 0.0

        # Empty contexts
        precision = evaluator._calculate_context_precision_heuristic(query, [])
        assert precision == 0.0

    def test_calculate_overall_score(self):
        """Test overall score calculation"""
        evaluator = RAGEvaluator()

        # Perfect scores
        result = RAGEvaluationResult(
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_precision=1.0
        )
        overall = evaluator._calculate_overall_score(result)
        assert abs(overall - 1.0) < 0.001  # Use epsilon for floating point comparison

        # Zero scores
        result = RAGEvaluationResult(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0
        )
        overall = evaluator._calculate_overall_score(result)
        assert overall == 0.0

        # With context recall
        result = RAGEvaluationResult(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.85
        )
        overall = evaluator._calculate_overall_score(result)
        assert 0.0 <= overall <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_from_dict(self):
        """Test evaluation from dictionary input"""
        evaluator = RAGEvaluator(enable_ragas=False)

        data = {
            'query': "What is Python?",
            'contexts': ["Python is a programming language."],
            'answer': "Python is a programming language used for development.",
            'ground_truth': "Python is a high-level programming language.",
            'metadata': {"source": "test"}
        }

        result = await evaluator.evaluate_from_dict(data)
        assert isinstance(result, RAGEvaluationResult)
        assert result.metadata == {"source": "test"}

    def test_get_evaluation_summary(self):
        """Test evaluation summary generation"""
        evaluator = RAGEvaluator()

        results = [
            RAGEvaluationResult(
                faithfulness=0.8,
                answer_relevancy=0.9,
                context_precision=0.7,
                overall_score=0.82
            ),
            RAGEvaluationResult(
                faithfulness=0.9,
                answer_relevancy=0.8,
                context_precision=0.8,
                overall_score=0.85
            )
        ]

        summary = evaluator.get_evaluation_summary(results)

        assert 'faithfulness_mean' in summary
        assert 'answer_relevancy_mean' in summary
        assert 'context_precision_mean' in summary
        assert 'overall_score_mean' in summary

        assert abs(summary['faithfulness_mean'] - 0.85) < 0.001
        assert abs(summary['answer_relevancy_mean'] - 0.85) < 0.001
        assert abs(summary['context_precision_mean'] - 0.75) < 0.001

        # Test empty results
        summary = evaluator.get_evaluation_summary([])
        assert summary == {}

    @pytest.mark.asyncio
    async def test_error_handling_missing_embedding_model(self):
        """Test error handling when embedding model is not available"""
        with patch.object(RAGEvaluator, '__init__') as mock_init:
            # Create a mock evaluator with no embedding model
            mock_init.return_value = None
            evaluator = RAGEvaluator.__new__(RAGEvaluator)
            evaluator.embedding_model = None
            evaluator.enable_ragas = False
            evaluator.fallback_to_simple = True
            evaluator.logger = Mock()

            # Should still work with word overlap fallback
            result = await evaluator.evaluate_single(
                query="test query",
                contexts=["test context"],
                answer="test answer"
            )
            assert isinstance(result, RAGEvaluationResult)

    @pytest.mark.asyncio
    async def test_ragas_fallback_behavior(self):
        """Test fallback behavior when RAGAS fails"""
        # Create evaluator that will try RAGAS (even though not available)
        evaluator = RAGEvaluator(enable_ragas=False, fallback_to_simple=True)

        # Force enable_ragas for this test
        evaluator.enable_ragas = True

        # Mock RAGAS to fail
        with patch.object(evaluator, '_evaluate_with_ragas') as mock_ragas:
            mock_ragas.side_effect = Exception("RAGAS failed")

            result = await evaluator.evaluate_single(
                query="test query",
                contexts=["test context"],
                answer="test answer"
            )

            assert isinstance(result, RAGEvaluationResult)
            mock_ragas.assert_called_once()


class TestFactoryFunction:
    """Test factory function"""

    def test_create_rag_evaluator_default(self):
        """Test factory function with defaults"""
        evaluator = create_rag_evaluator()
        assert isinstance(evaluator, RAGEvaluator)

    def test_create_rag_evaluator_custom(self):
        """Test factory function with custom parameters"""
        evaluator = create_rag_evaluator(
            embedding_model="custom-model",
            enable_ragas=False
        )
        assert isinstance(evaluator, RAGEvaluator)
        assert evaluator.embedding_model_name == "custom-model"
        assert evaluator.enable_ragas is False


class TestPipelineEvaluation:
    """Test high-level pipeline evaluation function"""

    @pytest.mark.asyncio
    async def test_evaluate_rag_pipeline(self):
        """Test complete pipeline evaluation"""
        queries = ["What is AI?", "What is ML?"]
        contexts_list = [
            ["AI is artificial intelligence."],
            ["ML is machine learning."]
        ]
        answers = [
            "AI stands for artificial intelligence.",
            "ML means machine learning."
        ]

        report = await evaluate_rag_pipeline(queries, contexts_list, answers)

        assert 'results' in report
        assert 'summary' in report
        assert 'total_evaluated' in report
        assert 'ragas_enabled' in report
        assert 'evaluation_timestamp' in report

        assert len(report['results']) == 2
        assert report['total_evaluated'] == 2
        assert isinstance(report['ragas_enabled'], bool)

    @pytest.mark.asyncio
    async def test_evaluate_rag_pipeline_with_ground_truth(self):
        """Test pipeline evaluation with ground truth"""
        queries = ["What is Python?"]
        contexts_list = [["Python is a programming language."]]
        answers = ["Python is a language for coding."]
        ground_truths = ["Python is a programming language."]

        report = await evaluate_rag_pipeline(
            queries, contexts_list, answers, ground_truths
        )

        assert len(report['results']) == 1
        assert report['total_evaluated'] == 1


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    @pytest.mark.asyncio
    async def test_single_evaluation_performance(self):
        """Benchmark single evaluation performance"""
        evaluator = create_rag_evaluator(enable_ragas=False)

        import time
        start_time = time.time()

        result = await evaluator.evaluate_single(
            query="What is the performance test?",
            contexts=["This is a performance test context."],
            answer="This is a performance test answer."
        )

        end_time = time.time()
        evaluation_time = end_time - start_time

        assert evaluation_time < 1.0  # Should complete in less than 1 second
        assert result.evaluation_time is not None
        assert result.evaluation_time > 0

    @pytest.mark.asyncio
    async def test_batch_evaluation_performance(self):
        """Benchmark batch evaluation performance"""
        evaluator = create_rag_evaluator(enable_ragas=False)

        # Create 10 evaluation inputs
        inputs = []
        for i in range(10):
            inputs.append(EvaluationInput(
                query=f"Test query {i}",
                contexts=[f"Test context {i}"],
                answer=f"Test answer {i}"
            ))

        import time
        start_time = time.time()

        results = await evaluator.evaluate_batch(inputs)

        end_time = time.time()
        evaluation_time = end_time - start_time

        assert len(results) == 10
        assert evaluation_time < 5.0  # Should complete in less than 5 seconds


class TestIntegrationScenarios:
    """Integration test scenarios"""

    @pytest.mark.asyncio
    async def test_neuroscience_domain_scenario(self):
        """Test evaluation with neuroscience domain content"""
        evaluator = create_rag_evaluator(enable_ragas=False)

        query = "What are the key differences between ASD and ADHD in fMRI studies?"
        contexts = [
            "ASD shows reduced connectivity in the default mode network in fMRI studies.",
            "ADHD typically shows hyperconnectivity in attention networks.",
            "Both conditions affect brain connectivity but in different patterns."
        ]
        answer = "ASD and ADHD show different connectivity patterns in fMRI: ASD has reduced default mode network connectivity while ADHD shows attention network hyperconnectivity."

        result = await evaluator.evaluate_single(query, contexts, answer)

        assert isinstance(result, RAGEvaluationResult)
        assert result.faithfulness > 0.5  # Should have reasonable faithfulness
        assert result.answer_relevancy > 0.5  # Should be relevant to query

    @pytest.mark.asyncio
    async def test_quantum_ml_domain_scenario(self):
        """Test evaluation with quantum ML domain content"""
        evaluator = create_rag_evaluator(enable_ragas=False)

        query = "How does quantum advantage work in machine learning?"
        contexts = [
            "Quantum advantage in ML comes from quantum superposition and entanglement.",
            "Variational quantum algorithms can solve certain ML problems faster.",
            "Quantum circuits can represent complex probability distributions."
        ]
        answer = "Quantum advantage in ML leverages superposition and entanglement through variational algorithms to solve problems more efficiently than classical methods."

        result = await evaluator.evaluate_single(query, contexts, answer)

        assert isinstance(result, RAGEvaluationResult)
        assert result.overall_score > 0.4  # Should have decent overall score

    @pytest.mark.asyncio
    async def test_edge_case_empty_contexts(self):
        """Test handling of edge case with empty contexts"""
        evaluator = create_rag_evaluator(enable_ragas=False)

        result = await evaluator.evaluate_single(
            query="Test query",
            contexts=[],  # Empty contexts
            answer="Test answer"
        )

        assert isinstance(result, RAGEvaluationResult)
        assert result.context_precision == 0.0  # No contexts = no precision

    @pytest.mark.asyncio
    async def test_edge_case_very_long_texts(self):
        """Test handling of very long texts"""
        evaluator = create_rag_evaluator(enable_ragas=False)

        long_context = "This is a test. " * 1000  # Very long context
        long_answer = "This is a long answer. " * 100

        result = await evaluator.evaluate_single(
            query="Short query",
            contexts=[long_context],
            answer=long_answer
        )

        assert isinstance(result, RAGEvaluationResult)
        assert result.evaluation_time is not None


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
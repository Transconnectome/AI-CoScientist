"""
RAG Evaluator Tests (Phase 1)

TDD Cycle: Red → Green → Refactor
"""

import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock

# Phase 1: 평가 프레임워크 테스트


@pytest.mark.asyncio
class TestFaithfulnessMetric:
    """Faithfulness (신뢰성) 메트릭 테스트"""
    
    async def test_faithfulness_high_score(self):
        """답변이 컨텍스트에 완전히 기반할 때 높은 점수"""
        from src.services.rag.rag_evaluator import FaithfulnessMetric
        
        # Mock LLM response
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.95"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        metric = FaithfulnessMetric(llm_client=mock_client)
        
        context = ["The study found that X causes Y."]
        answer = "According to the study, X causes Y."
        
        score = await metric.evaluate(answer, context)
        assert score >= 0.9
        assert score <= 1.0
    
    async def test_faithfulness_low_score(self):
        """답변이 컨텍스트에 없는 정보를 포함할 때 낮은 점수"""
        from src.services.rag.rag_evaluator import FaithfulnessMetric
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.3"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        metric = FaithfulnessMetric(llm_client=mock_client)
        
        context = ["The study found that X causes Y."]
        answer = "The study found that X causes Y, and also Z causes W."
        
        score = await metric.evaluate(answer, context)
        assert score < 0.5
    
    async def test_faithfulness_empty_context(self):
        """컨텍스트가 없을 때 처리"""
        from src.services.rag.rag_evaluator import FaithfulnessMetric
        
        metric = FaithfulnessMetric()
        
        context: List[str] = []
        answer = "Some answer"
        
        score = await metric.evaluate(answer, context)
        assert score == 0.0
    
    async def test_faithfulness_multiple_contexts(self):
        """여러 컨텍스트에 대한 평가"""
        from src.services.rag.rag_evaluator import FaithfulnessMetric
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.85"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        metric = FaithfulnessMetric(llm_client=mock_client)
        
        context = [
            "The study found that X causes Y.",
            "Previous research showed similar results."
        ]
        answer = "Research indicates that X causes Y."
        
        score = await metric.evaluate(answer, context)
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
class TestAnswerRelevancyMetric:
    """Answer Relevancy (답변 관련성) 메트릭 테스트"""
    
    async def test_answer_relevancy_high_score(self):
        """답변이 쿼리를 잘 해결할 때 높은 점수"""
        from src.services.rag.rag_evaluator import AnswerRelevancyMetric
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.9"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        metric = AnswerRelevancyMetric(llm_client=mock_client)
        
        query = "What causes X?"
        answer = "X is caused by Y and Z factors."
        
        score = await metric.evaluate(query, answer)
        assert score >= 0.8
        assert score <= 1.0
    
    async def test_answer_relevancy_low_score(self):
        """답변이 쿼리와 관련 없을 때 낮은 점수"""
        from src.services.rag.rag_evaluator import AnswerRelevancyMetric
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.2"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        metric = AnswerRelevancyMetric(llm_client=mock_client)
        
        query = "What causes X?"
        answer = "The weather is nice today."
        
        score = await metric.evaluate(query, answer)
        assert score < 0.3
    
    async def test_answer_relevancy_partial_match(self):
        """부분적으로 관련된 답변"""
        from src.services.rag.rag_evaluator import AnswerRelevancyMetric
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.6"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        metric = AnswerRelevancyMetric(llm_client=mock_client)
        
        query = "What are the side effects of drug X?"
        answer = "Drug X is used for treatment."
        
        score = await metric.evaluate(query, answer)
        assert 0.4 <= score <= 0.7


@pytest.mark.asyncio
class TestContextPrecisionMetric:
    """Context Precision (컨텍스트 정밀도) 메트릭 테스트"""
    
    async def test_context_precision_high_score(self):
        """모든 컨텍스트가 관련 있을 때 높은 점수"""
        from src.services.rag.rag_evaluator import ContextPrecisionMetric
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.95"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        metric = ContextPrecisionMetric(llm_client=mock_client)
        
        query = "machine learning"
        contexts = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks."
        ]
        
        score = await metric.evaluate(query, contexts)
        assert score >= 0.8
    
    async def test_context_precision_mixed_relevance(self):
        """일부 컨텍스트만 관련 있을 때 중간 점수"""
        from src.services.rag.rag_evaluator import ContextPrecisionMetric
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.5"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        metric = ContextPrecisionMetric(llm_client=mock_client)
        
        query = "machine learning"
        contexts = [
            "Machine learning is a subset of AI.",
            "The weather forecast predicts rain."
        ]
        
        score = await metric.evaluate(query, contexts)
        assert 0.4 <= score <= 0.6
    
    async def test_context_precision_empty_contexts(self):
        """컨텍스트가 없을 때"""
        from src.services.rag.rag_evaluator import ContextPrecisionMetric
        
        metric = ContextPrecisionMetric()
        
        query = "test"
        contexts: List[str] = []
        
        score = await metric.evaluate(query, contexts)
        assert score == 0.0


@pytest.mark.asyncio
class TestContextRecallMetric:
    """Context Recall (컨텍스트 재현율) 메트릭 테스트"""
    
    async def test_context_recall_with_ground_truth(self):
        """Ground truth가 있을 때 재현율 계산"""
        from src.services.rag.rag_evaluator import ContextRecallMetric
        
        metric = ContextRecallMetric()
        
        retrieved_contexts = [
            "Machine learning uses algorithms.",
            "Deep learning uses neural networks."
        ]
        ground_truth_contexts = [
            "Machine learning uses algorithms.",
            "Deep learning uses neural networks.",
            "Reinforcement learning uses rewards."
        ]
        
        recall = await metric.evaluate(retrieved_contexts, ground_truth_contexts)
        # 2 out of 3 retrieved
        assert 0.6 <= recall <= 0.7
    
    async def test_context_recall_perfect_match(self):
        """모든 필요한 컨텍스트가 검색되었을 때"""
        from src.services.rag.rag_evaluator import ContextRecallMetric
        
        metric = ContextRecallMetric()
        
        retrieved_contexts = [
            "Machine learning uses algorithms.",
            "Deep learning uses neural networks."
        ]
        ground_truth_contexts = [
            "Machine learning uses algorithms.",
            "Deep learning uses neural networks."
        ]
        
        recall = await metric.evaluate(retrieved_contexts, ground_truth_contexts)
        assert recall == 1.0


@pytest.mark.asyncio
class TestRAGEvaluatorIntegration:
    """RAG Evaluator 통합 테스트"""
    
    async def test_complete_evaluation_pipeline(self):
        """전체 평가 파이프라인 테스트"""
        from src.services.rag.rag_evaluator import RAGEvaluator
        
        # Mock LLM clients
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.85"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        evaluator = RAGEvaluator(llm_client=mock_client)
        
        result = await evaluator.evaluate(
            query="What is RAG?",
            retrieved_context=["RAG is Retrieval-Augmented Generation..."],
            generated_answer="RAG stands for Retrieval-Augmented Generation..."
        )
        
        assert result.faithfulness >= 0.0
        assert result.faithfulness <= 1.0
        assert result.answer_relevancy >= 0.0
        assert result.answer_relevancy <= 1.0
        assert result.context_precision >= 0.0
        assert result.context_precision <= 1.0
    
    async def test_evaluation_with_ground_truth(self):
        """Ground truth가 있을 때 평가"""
        from src.services.rag.rag_evaluator import RAGEvaluator
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.9"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        evaluator = RAGEvaluator(llm_client=mock_client)
        
        result = await evaluator.evaluate(
            query="What is machine learning?",
            retrieved_context=["Machine learning is..."],
            generated_answer="Machine learning is a subset of AI.",
            ground_truth="Machine learning is a method of data analysis."
        )
        
        assert result.answer_correctness is not None
        assert 0.0 <= result.answer_correctness <= 1.0


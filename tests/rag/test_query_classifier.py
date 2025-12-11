"""
Query Classifier Tests (Phase 3)

TDD Cycle: Red → Green → Refactor
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
class TestQueryClassifier:
    """쿼리 분류기 테스트"""
    
    async def test_classify_factual_query(self):
        """사실적 쿼리 분류"""
        from src.services.rag.query_classifier import QueryClassifier, QueryType
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"type": "factual", "confidence": 0.9, "reasoning": "Direct factual question"}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        classifier = QueryClassifier(llm_client=mock_client)
        
        query = "What is machine learning?"
        result = await classifier.classify(query)
        
        assert result.query_type == QueryType.FACTUAL
        assert result.confidence >= 0.8
        assert len(result.reasoning) > 0
    
    async def test_classify_multi_hop_query(self):
        """다중 홉 쿼리 분류"""
        from src.services.rag.query_classifier import QueryClassifier, QueryType
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"type": "multi_hop", "confidence": 0.85, "reasoning": "Requires multiple steps"}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        classifier = QueryClassifier(llm_client=mock_client)
        
        query = "What methodologies are used in papers that cite X?"
        result = await classifier.classify(query)
        
        assert result.query_type == QueryType.MULTI_HOP
        assert result.confidence >= 0.7
    
    async def test_classify_hierarchical_query(self):
        """계층적 쿼리 분류"""
        from src.services.rag.query_classifier import QueryClassifier, QueryType
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"type": "hierarchical", "confidence": 0.8, "reasoning": "Requires high-level abstraction"}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        classifier = QueryClassifier(llm_client=mock_client)
        
        query = "What are the main themes across this research program?"
        result = await classifier.classify(query)
        
        assert result.query_type == QueryType.HIERARCHICAL
    
    async def test_classify_comparative_query(self):
        """비교적 쿼리 분류"""
        from src.services.rag.query_classifier import QueryClassifier, QueryType
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"type": "comparative", "confidence": 0.9, "reasoning": "Compares multiple items"}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        classifier = QueryClassifier(llm_client=mock_client)
        
        query = "Compare machine learning and deep learning approaches"
        result = await classifier.classify(query)
        
        assert result.query_type == QueryType.COMPARATIVE
    
    async def test_classify_unknown_query(self):
        """알 수 없는 쿼리 타입"""
        from src.services.rag.query_classifier import QueryClassifier, QueryType
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"type": "unknown", "confidence": 0.5, "reasoning": "Unclear query type"}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        classifier = QueryClassifier(llm_client=mock_client)
        
        query = "Random text that doesn't make sense"
        result = await classifier.classify(query)
        
        assert result.query_type == QueryType.UNKNOWN
    
    async def test_classify_empty_query(self):
        """빈 쿼리 처리"""
        from src.services.rag.query_classifier import QueryClassifier, QueryType
        
        classifier = QueryClassifier()
        
        query = ""
        result = await classifier.classify(query)
        
        assert result.query_type == QueryType.UNKNOWN
        assert result.confidence < 0.5


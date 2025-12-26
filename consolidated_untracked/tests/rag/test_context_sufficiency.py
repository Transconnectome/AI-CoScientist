"""
Context Sufficiency Tests (Phase 2)

TDD Cycle: Red → Green → Refactor
"""

import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
class TestContextSufficiencyChecker:
    """컨텍스트 충분성 검사 테스트"""
    
    async def test_sufficient_context(self):
        """충분한 컨텍스트일 때 True"""
        from src.services.rag.context_sufficiency import ContextSufficiencyChecker
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.9"  # 높은 충분성 점수
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        checker = ContextSufficiencyChecker(llm_client=mock_client)
        
        query = "What is the main finding?"
        context = ["The main finding is that X causes Y in 80% of cases."]
        
        is_sufficient = await checker.check(query, context)
        assert is_sufficient is True
    
    async def test_insufficient_context(self):
        """부족한 컨텍스트일 때 False"""
        from src.services.rag.context_sufficiency import ContextSufficiencyChecker
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.3"  # 낮은 충분성 점수
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        checker = ContextSufficiencyChecker(llm_client=mock_client)
        
        query = "What is the main finding?"
        context = ["The study was conducted."]
        
        is_sufficient = await checker.check(query, context)
        assert is_sufficient is False
    
    async def test_sufficiency_with_expansion(self):
        """부족할 때 확장 제안"""
        from src.services.rag.context_sufficiency import ContextSufficiencyChecker
        
        mock_client = AsyncMock()
        
        # 충분성 검사 응답
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = "0.4"  # 부족
        
        # 확장 제안 응답
        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = '["What are the side effects?", "What is the dosage?"]'
        
        mock_client.chat.completions.create = AsyncMock(side_effect=[mock_response1, mock_response2])
        
        checker = ContextSufficiencyChecker(llm_client=mock_client)
        
        query = "What are the side effects?"
        context = ["The drug was tested."]
        
        result = await checker.check_with_expansion(query, context)
        assert result.is_sufficient is False
        assert len(result.suggested_queries) > 0
    
    async def test_sufficiency_threshold(self):
        """임계값 테스트"""
        from src.services.rag.context_sufficiency import ContextSufficiencyChecker
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.7"  # 임계값과 동일
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        checker = ContextSufficiencyChecker(llm_client=mock_client, threshold=0.7)
        
        query = "test"
        context = ["Some context"]
        
        is_sufficient = await checker.check(query, context)
        assert is_sufficient is True  # >= threshold
    
    async def test_empty_context(self):
        """빈 컨텍스트 처리"""
        from src.services.rag.context_sufficiency import ContextSufficiencyChecker
        
        checker = ContextSufficiencyChecker()
        
        query = "test"
        context: List[str] = []
        
        is_sufficient = await checker.check(query, context)
        assert is_sufficient is False
    
    async def test_multiple_contexts(self):
        """여러 컨텍스트에 대한 충분성 검사"""
        from src.services.rag.context_sufficiency import ContextSufficiencyChecker
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.85"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        checker = ContextSufficiencyChecker(llm_client=mock_client)
        
        query = "What are the results?"
        context = [
            "The study included 100 participants.",
            "Results showed 80% improvement.",
            "Side effects were minimal."
        ]
        
        is_sufficient = await checker.check(query, context)
        assert is_sufficient is True


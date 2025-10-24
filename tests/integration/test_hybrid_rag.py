"""
Integration tests for Hybrid RAG pipeline.

Tests the complete hybrid system including:
- Task routing between GPT-4, Claude, and Nemotron
- Ensemble scoring and provider coordination
- API endpoint integration
- Error handling and fallback strategies
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.main import app
from src.services.hybrid_rag_service import (
    HybridRAGService,
    ModelProvider,
    EvaluationResult,
    EnsembleResult,
)
from src.services.nemotron_llm import NemotronLLM, CompletionResult
from src.services.nemo_retriever import NeMoEmbedder, NeMoReranker


@pytest.fixture
def test_client():
    """Create test client for API integration tests"""
    return TestClient(app)


@pytest.fixture
async def mock_hybrid_service():
    """Create HybridRAGService with all external APIs mocked"""
    with patch('httpx.AsyncClient'), \
         patch('openai.AsyncOpenAI'), \
         patch('anthropic.AsyncAnthropic'):

        service = HybridRAGService(
            hybrid_mode=True,
            use_gpt4_for_evaluation=True,
            use_claude_for_evaluation=True,
            use_nemotron_for_summarization=True,
            use_nemotron_for_extraction=True,
            ensemble_weight_gpt4=0.4,
            ensemble_weight_claude=0.3,
            ensemble_weight_nemotron=0.3,
        )

        return service


class TestHybridRAGTaskRouting:
    """Test task routing logic"""

    @pytest.mark.asyncio
    async def test_evaluation_routes_to_ensemble(self, mock_hybrid_service):
        """Test that evaluation tasks route to GPT-4 + Claude ensemble"""
        # Mock GPT-4 evaluation
        mock_hybrid_service._evaluate_with_gpt4 = AsyncMock(
            return_value=EvaluationResult(
                overall_quality=8.5,
                novelty=8.0,
                methodology=9.0,
                clarity=8.5,
                significance=8.0,
                feedback="Strong methodology",
                provider=ModelProvider.GPT4,
                confidence=0.9,
                latency_ms=1500,
            )
        )

        # Mock Claude evaluation
        mock_hybrid_service._evaluate_with_claude = AsyncMock(
            return_value=EvaluationResult(
                overall_quality=8.3,
                novelty=8.2,
                methodology=8.8,
                clarity=8.0,
                significance=8.5,
                feedback="Good clarity",
                provider=ModelProvider.CLAUDE,
                confidence=0.9,
                latency_ms=1800,
            )
        )

        # Test
        result = await mock_hybrid_service.evaluate_paper(
            paper_text="Sample paper content",
            section="abstract",
            use_ensemble=True,
        )

        # Assertions
        assert isinstance(result, EnsembleResult)
        assert 8.0 <= result.overall_quality <= 9.0  # Weighted average
        assert len(result.provider_scores) >= 2  # GPT-4 and Claude
        assert ModelProvider.GPT4.value in result.provider_scores
        assert ModelProvider.CLAUDE.value in result.provider_scores

        # Verify both models were called
        mock_hybrid_service._evaluate_with_gpt4.assert_called_once()
        mock_hybrid_service._evaluate_with_claude.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarization_routes_to_nemotron(self, mock_hybrid_service):
        """Test that summarization routes to Nemotron"""
        # Mock Nemotron summarization
        mock_hybrid_service.nemotron_llm.summarize = AsyncMock(
            return_value="Concise summary of the paper"
        )

        # Test
        summary = await mock_hybrid_service.summarize_paper(
            paper_text="Long paper content...",
            max_length=200,
            style="concise",
        )

        # Assertions
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Verify Nemotron was called
        mock_hybrid_service.nemotron_llm.summarize.assert_called_once()

    @pytest.mark.asyncio
    async def test_extraction_routes_to_nemotron(self, mock_hybrid_service):
        """Test that extraction routes to Nemotron"""
        # Mock Nemotron extraction
        mock_hybrid_service.nemotron_llm.extract_information = AsyncMock(
            return_value={
                "methodology": "Transformer-based architecture",
                "results": "95% accuracy on benchmark",
                "limitations": "Requires large training data"
            }
        )

        # Test
        extracted = await mock_hybrid_service.extract_information(
            paper_text="Paper with methodology and results",
            fields=["methodology", "results", "limitations"],
        )

        # Assertions
        assert isinstance(extracted, dict)
        assert "methodology" in extracted
        assert "results" in extracted
        assert "limitations" in extracted

        # Verify Nemotron was called
        mock_hybrid_service.nemotron_llm.extract_information.assert_called_once()


class TestEnsembleScoring:
    """Test ensemble scoring and weight management"""

    @pytest.mark.asyncio
    async def test_ensemble_weights_sum_to_one(self, mock_hybrid_service):
        """Test that ensemble weights are normalized to 1.0"""
        total_weight = (
            mock_hybrid_service.ensemble_weight_gpt4 +
            mock_hybrid_service.ensemble_weight_claude +
            mock_hybrid_service.ensemble_weight_nemotron
        )

        assert abs(total_weight - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_ensemble_computes_weighted_average(self, mock_hybrid_service):
        """Test that ensemble correctly computes weighted averages"""
        # Create mock evaluation results
        provider_scores = {
            ModelProvider.GPT4.value: EvaluationResult(
                overall_quality=9.0,
                novelty=9.0,
                methodology=9.0,
                clarity=9.0,
                significance=9.0,
                feedback="Excellent",
                provider=ModelProvider.GPT4,
                confidence=0.9,
                latency_ms=1000,
            ),
            ModelProvider.CLAUDE.value: EvaluationResult(
                overall_quality=8.0,
                novelty=8.0,
                methodology=8.0,
                clarity=8.0,
                significance=8.0,
                feedback="Good",
                provider=ModelProvider.CLAUDE,
                confidence=0.9,
                latency_ms=1000,
            ),
        }

        # Compute ensemble
        result = mock_hybrid_service._compute_ensemble(provider_scores)

        # Expected: 0.4 * 9.0 + 0.6 * 8.0 = 3.6 + 4.8 = 8.4
        # (normalizing weights: 0.4/0.7 ≈ 0.571, 0.3/0.7 ≈ 0.429)
        assert isinstance(result, EnsembleResult)
        assert 8.0 <= result.overall_quality <= 9.0

    @pytest.mark.asyncio
    async def test_ensemble_handles_partial_provider_failure(self, mock_hybrid_service):
        """Test ensemble when one provider fails"""
        # Mock GPT-4 success
        mock_hybrid_service._evaluate_with_gpt4 = AsyncMock(
            return_value=EvaluationResult(
                overall_quality=8.5,
                novelty=8.0,
                methodology=9.0,
                clarity=8.5,
                significance=8.0,
                feedback="Strong",
                provider=ModelProvider.GPT4,
                confidence=0.9,
                latency_ms=1500,
            )
        )

        # Mock Claude failure
        mock_hybrid_service._evaluate_with_claude = AsyncMock(
            side_effect=Exception("Claude API unavailable")
        )

        # Test - should still succeed with GPT-4 only
        result = await mock_hybrid_service.evaluate_paper(
            paper_text="Sample paper",
            use_ensemble=True,
        )

        # Assertions
        assert isinstance(result, EnsembleResult)
        assert len(result.provider_scores) >= 1  # At least GPT-4
        assert result.overall_quality > 0  # Valid score


class TestFallbackStrategies:
    """Test fallback strategies when services fail"""

    @pytest.mark.asyncio
    async def test_summarization_falls_back_to_gpt4(self, mock_hybrid_service):
        """Test summarization falls back to GPT-4 when Nemotron unavailable"""
        # Disable Nemotron
        mock_hybrid_service.use_nemotron_for_summarization = False

        # Mock GPT-4 fallback
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="GPT-4 summary"))]
        mock_hybrid_service.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Test
        summary = await mock_hybrid_service.summarize_paper("Paper text")

        # Should use GPT-4
        assert summary == "GPT-4 summary"
        mock_hybrid_service.openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_extraction_falls_back_to_gpt4(self, mock_hybrid_service):
        """Test extraction falls back to GPT-4 when Nemotron unavailable"""
        # Disable Nemotron
        mock_hybrid_service.use_nemotron_for_extraction = False

        # Mock GPT-4 fallback
        import json
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({"field1": "value1"})))
        ]
        mock_hybrid_service.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Test
        result = await mock_hybrid_service.extract_information(
            "Paper text",
            fields=["field1"]
        )

        # Should use GPT-4
        assert result == {"field1": "value1"}


class TestAPIEndpoints:
    """Test API endpoint integration"""

    @pytest.mark.asyncio
    async def test_evaluate_endpoint(self, test_client):
        """Test /hybrid-rag/evaluate endpoint"""
        # Mock the global service instance
        with patch('src.api.v1.hybrid_rag.get_hybrid_rag_service') as mock_get_service:
            mock_service = MagicMock()

            # Mock evaluation result
            mock_service.evaluate_paper = AsyncMock(
                return_value=EnsembleResult(
                    overall_quality=8.5,
                    novelty=8.0,
                    methodology=9.0,
                    clarity=8.5,
                    significance=8.0,
                    feedback="Test feedback",
                    provider_scores={
                        "gpt4": EvaluationResult(
                            overall_quality=8.5,
                            novelty=8.0,
                            methodology=9.0,
                            clarity=8.5,
                            significance=8.0,
                            feedback="GPT-4 feedback",
                            provider=ModelProvider.GPT4,
                            confidence=0.9,
                            latency_ms=1500,
                        )
                    },
                    ensemble_confidence=0.9,
                    total_latency_ms=1500,
                )
            )

            mock_get_service.return_value = mock_service

            # Test
            response = test_client.post(
                "/api/v1/hybrid-rag/evaluate",
                json={
                    "paper_text": "Sample paper content",
                    "section": "abstract",
                    "use_ensemble": True
                }
            )

            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert "overall_quality" in data
            assert "provider_scores" in data

    @pytest.mark.asyncio
    async def test_summarize_endpoint(self, test_client):
        """Test /hybrid-rag/summarize endpoint"""
        with patch('src.api.v1.hybrid_rag.get_hybrid_rag_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.summarize_paper = AsyncMock(
                return_value="Test summary"
            )
            mock_get_service.return_value = mock_service

            # Test
            response = test_client.post(
                "/api/v1/hybrid-rag/summarize",
                json={
                    "paper_text": "Long paper content",
                    "max_length": 200,
                    "style": "concise"
                }
            )

            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert data["summary"] == "Test summary"
            assert data["provider"] == "nemotron"

    @pytest.mark.asyncio
    async def test_status_endpoint(self, test_client):
        """Test /hybrid-rag/status endpoint"""
        with patch('src.api.v1.hybrid_rag.get_hybrid_rag_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.hybrid_mode = True
            mock_service.use_gpt4_for_evaluation = True
            mock_service.use_claude_for_evaluation = True
            mock_service.use_nemotron_for_summarization = True
            mock_service.use_nemotron_for_extraction = True
            mock_service.ensemble_weight_gpt4 = 0.4
            mock_service.ensemble_weight_claude = 0.3
            mock_service.ensemble_weight_nemotron = 0.3
            mock_service.nemotron_confidence_threshold = 0.75

            # Mock service URLs
            mock_service.nemotron_llm.base_url = "http://localhost:8000/v1"
            mock_service.nemo_embedder.base_url = "http://localhost:8001/v1"
            mock_service.nemo_reranker.base_url = "http://localhost:8002/v1"

            mock_get_service.return_value = mock_service

            # Test
            response = test_client.get("/api/v1/hybrid-rag/status")

            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert data["hybrid_mode"] is True
            assert "gpt4" in data["enabled_providers"]
            assert "claude" in data["enabled_providers"]
            assert data["ensemble_weights"]["gpt4"] == 0.4


class TestPerformanceMetrics:
    """Test performance tracking and metrics"""

    @pytest.mark.asyncio
    async def test_latency_tracking(self, mock_hybrid_service):
        """Test that latency is tracked for all operations"""
        # Mock evaluations with known latencies
        mock_hybrid_service._evaluate_with_gpt4 = AsyncMock(
            return_value=EvaluationResult(
                overall_quality=8.5,
                novelty=8.0,
                methodology=9.0,
                clarity=8.5,
                significance=8.0,
                feedback="Test",
                provider=ModelProvider.GPT4,
                confidence=0.9,
                latency_ms=1000,
            )
        )

        mock_hybrid_service._evaluate_with_claude = AsyncMock(
            return_value=EvaluationResult(
                overall_quality=8.3,
                novelty=8.2,
                methodology=8.8,
                clarity=8.0,
                significance=8.5,
                feedback="Test",
                provider=ModelProvider.CLAUDE,
                confidence=0.9,
                latency_ms=1500,
            )
        )

        # Test
        result = await mock_hybrid_service.evaluate_paper("Test paper")

        # Assertions
        assert result.total_latency_ms > 0
        assert result.total_latency_ms >= 1000  # At least GPT-4's latency

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, mock_hybrid_service):
        """Test confidence scoring aggregation"""
        # Test ensemble confidence calculation
        provider_scores = {
            ModelProvider.GPT4.value: EvaluationResult(
                overall_quality=8.5,
                novelty=8.0,
                methodology=9.0,
                clarity=8.5,
                significance=8.0,
                feedback="Test",
                provider=ModelProvider.GPT4,
                confidence=0.9,
                latency_ms=1000,
            ),
            ModelProvider.CLAUDE.value: EvaluationResult(
                overall_quality=8.3,
                novelty=8.2,
                methodology=8.8,
                clarity=8.0,
                significance=8.5,
                feedback="Test",
                provider=ModelProvider.CLAUDE,
                confidence=0.85,
                latency_ms=1500,
            ),
        }

        result = mock_hybrid_service._compute_ensemble(provider_scores)

        # Ensemble confidence should be weighted average
        assert 0.85 <= result.ensemble_confidence <= 0.9

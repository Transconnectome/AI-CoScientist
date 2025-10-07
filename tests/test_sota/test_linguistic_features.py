"""Unit tests for linguistic feature extractor."""

import pytest
import torch
from src.services.paper.linguistic_features import LinguisticFeatureExtractor


class TestLinguisticFeatureExtractor:
    """Test suite for linguistic feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return LinguisticFeatureExtractor()

    @pytest.fixture
    def sample_text(self):
        """Sample academic text for testing."""
        return """
        Deep Learning for Natural Language Processing: A Comprehensive Survey

        Abstract:
        This paper presents a comprehensive survey of deep learning methods for NLP.
        We analyze recent advances in neural architectures, including transformers,
        attention mechanisms, and pre-trained language models. Our methodology involves
        systematic review of over 200 papers published between 2020 and 2024.

        Introduction:
        Natural language processing has undergone significant transformation with deep learning.
        Traditional feature-based methods have been superseded by end-to-end neural approaches.
        This survey examines key developments and their implications for future research.
        """

    def test_extract_returns_correct_shape(self, extractor, sample_text):
        """Test that extract returns 20-dimensional tensor."""
        features = extractor.extract(sample_text)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (20,)
        assert features.dtype == torch.float32

    def test_extract_features_normalized(self, extractor, sample_text):
        """Test that all features are normalized to 0-1 range."""
        features = extractor.extract(sample_text)

        # All features should be between 0 and 1
        assert torch.all(features >= 0.0)
        assert torch.all(features <= 1.0)

    def test_readability_features(self, extractor, sample_text):
        """Test readability feature extraction."""
        features = extractor._extract_readability(sample_text)

        assert len(features) == 4
        assert all(0 <= f <= 1 for f in features)

    def test_vocabulary_features(self, extractor, sample_text):
        """Test vocabulary richness features."""
        features = extractor._extract_vocabulary(sample_text)

        assert len(features) == 4
        # TTR should be reasonable for academic text
        assert 0 < features[0] < 1

    def test_syntax_features(self, extractor, sample_text):
        """Test syntactic complexity features."""
        features = extractor._extract_syntax(sample_text)

        assert len(features) == 4
        assert all(0 <= f <= 1 for f in features)

    def test_academic_features(self, extractor, sample_text):
        """Test academic writing indicators."""
        features = extractor._extract_academic(sample_text)

        assert len(features) == 4
        # Should detect some technical terms
        assert features[1] > 0  # tech_ratio

    def test_coherence_features(self, extractor, sample_text):
        """Test discourse and coherence features."""
        features = extractor._extract_coherence(sample_text)

        assert len(features) == 4
        assert all(0 <= f <= 1 for f in features)

    def test_empty_text_handling(self, extractor):
        """Test handling of empty or very short text."""
        features = extractor.extract("")

        assert features.shape == (20,)
        # Should return zeros for empty text
        assert torch.all(features == 0.0)

    def test_academic_word_detection(self, extractor):
        """Test detection of academic vocabulary."""
        academic_text = "This research analyzes significant evidence from the methodology."
        casual_text = "This thing is cool and awesome and great."

        academic_features = extractor._extract_vocabulary(academic_text)
        casual_features = extractor._extract_vocabulary(casual_text)

        # Academic text should have higher academic word ratio
        assert academic_features[2] > casual_features[2]

    def test_citation_detection(self, extractor):
        """Test citation density calculation."""
        text_with_citations = """
        Previous work (Smith, 2020) showed that transformers [1] are effective.
        Recent studies (Jones et al., 2023) confirmed these findings [2][3].
        """
        text_without_citations = "This is text without any citations."

        cited_features = extractor._extract_academic(text_with_citations)
        uncited_features = extractor._extract_academic(text_without_citations)

        # Text with citations should have higher citation density
        assert cited_features[0] > uncited_features[0]

    def test_consistency_across_calls(self, extractor, sample_text):
        """Test that same text produces same features."""
        features1 = extractor.extract(sample_text)
        features2 = extractor.extract(sample_text)

        assert torch.allclose(features1, features2)

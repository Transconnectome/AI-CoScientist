"""Unit tests for paper quality metrics."""

import pytest
import numpy as np
from src.services.paper.metrics import PaperMetrics


class TestPaperMetrics:
    """Test suite for evaluation metrics."""

    def test_quadratic_weighted_kappa_perfect_agreement(self):
        """Test QWK with perfect agreement."""
        human_scores = [7, 8, 9, 6, 10]
        ai_scores = [7, 8, 9, 6, 10]

        qwk = PaperMetrics.quadratic_weighted_kappa(human_scores, ai_scores)

        # Perfect agreement should give QWK = 1.0
        assert 0.95 <= qwk <= 1.0

    def test_quadratic_weighted_kappa_no_agreement(self):
        """Test QWK with random predictions."""
        human_scores = [10, 10, 10, 10, 10]
        ai_scores = [1, 1, 1, 1, 1]

        qwk = PaperMetrics.quadratic_weighted_kappa(human_scores, ai_scores)

        # Complete disagreement should give low QWK
        assert qwk < 0.5

    def test_quadratic_weighted_kappa_partial_agreement(self):
        """Test QWK with partial agreement (off by 1)."""
        human_scores = [7, 8, 9, 6, 10]
        ai_scores = [8, 9, 10, 7, 9]  # All off by +1 or -1

        qwk = PaperMetrics.quadratic_weighted_kappa(human_scores, ai_scores)

        # Close predictions should give good QWK
        assert 0.5 <= qwk <= 0.9

    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        true_scores = [7.0, 8.0, 9.0, 6.0, 10.0]
        pred_scores = [7.5, 8.2, 8.8, 6.3, 9.5]

        mae = PaperMetrics.mean_absolute_error(true_scores, pred_scores)

        # MAE should be small for close predictions
        assert 0 <= mae <= 1.0

    def test_mean_absolute_error_perfect(self):
        """Test MAE with perfect predictions."""
        scores = [7.0, 8.0, 9.0]
        mae = PaperMetrics.mean_absolute_error(scores, scores)

        assert mae == 0.0

    def test_root_mean_squared_error(self):
        """Test RMSE calculation."""
        true_scores = [7.0, 8.0, 9.0]
        pred_scores = [7.5, 8.5, 9.5]

        rmse = PaperMetrics.root_mean_squared_error(true_scores, pred_scores)

        assert 0 < rmse < 1.0

    def test_calculate_correlation_pearson(self):
        """Test Pearson correlation."""
        scores1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        scores2 = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect linear relationship

        corr = PaperMetrics.calculate_correlation(scores1, scores2, method="pearson")

        # Perfect positive correlation
        assert 0.95 <= corr <= 1.0

    def test_calculate_correlation_negative(self):
        """Test negative correlation."""
        scores1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        scores2 = [5.0, 4.0, 3.0, 2.0, 1.0]  # Perfect negative

        corr = PaperMetrics.calculate_correlation(scores1, scores2)

        # Perfect negative correlation
        assert -1.0 <= corr <= -0.95

    def test_calculate_accuracy_exact(self):
        """Test accuracy with exact matches."""
        true_scores = [7, 8, 9, 6, 10]
        pred_scores = [7, 8, 9, 6, 10]

        accuracy = PaperMetrics.calculate_accuracy(true_scores, pred_scores, tolerance=0)

        assert accuracy == 1.0

    def test_calculate_accuracy_with_tolerance(self):
        """Test accuracy with tolerance."""
        true_scores = [7, 8, 9, 6, 10]
        pred_scores = [8, 9, 10, 7, 9]  # All off by 1

        accuracy = PaperMetrics.calculate_accuracy(true_scores, pred_scores, tolerance=1)

        # All predictions within ±1
        assert accuracy == 1.0

    def test_confusion_matrix_shape(self):
        """Test confusion matrix dimensions."""
        true_scores = [7, 8, 9, 6, 10]
        pred_scores = [7, 8, 9, 6, 10]

        matrix = PaperMetrics.confusion_matrix(true_scores, pred_scores, num_classes=10)

        assert matrix.shape == (10, 10)
        assert np.sum(matrix) == len(true_scores)

    def test_calculate_f1_score_perfect(self):
        """Test F1 score with perfect predictions for a class."""
        true_scores = [8, 8, 8, 7, 9]
        pred_scores = [8, 8, 8, 7, 9]

        f1 = PaperMetrics.calculate_f1_score(true_scores, pred_scores, target_class=8)

        # Perfect precision and recall for class 8
        assert f1 == 1.0

    def test_calculate_f1_score_no_predictions(self):
        """Test F1 when target class is never predicted."""
        true_scores = [7, 7, 7]
        pred_scores = [8, 8, 8]

        f1 = PaperMetrics.calculate_f1_score(true_scores, pred_scores, target_class=7)

        # Zero recall
        assert f1 == 0.0

    def test_manual_pearson_fallback(self):
        """Test manual Pearson calculation (fallback)."""
        scores1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        scores2 = [2.0, 4.0, 6.0, 8.0, 10.0]

        corr = PaperMetrics._manual_pearson(scores1, scores2)

        assert 0.95 <= corr <= 1.0

    def test_fallback_similarity(self):
        """Test fallback similarity when BERTScore unavailable."""
        improved = {"intro": "deep learning neural networks"}
        original = {"intro": "neural networks learning"}

        result = PaperMetrics._fallback_similarity(improved, original)

        assert "intro" in result
        assert 0 < result["intro"]["f1"] <= 1.0

    def test_empty_lists_handling(self):
        """Test handling of empty score lists."""
        with pytest.raises(ZeroDivisionError):
            PaperMetrics._manual_pearson([], [])

    def test_mismatched_lengths(self):
        """Test error on mismatched list lengths."""
        true_scores = [7.0, 8.0, 9.0]
        pred_scores = [7.0, 8.0]

        with pytest.raises(ValueError):
            PaperMetrics.mean_absolute_error(true_scores, pred_scores)

"""Paper quality evaluation metrics."""

from typing import Dict, List, Optional
import numpy as np


class PaperMetrics:
    """Calculate various quality metrics for papers."""

    @staticmethod
    async def compute_bertscore(
        improved_sections: Dict[str, str],
        original_sections: Dict[str, str],
        model_type: str = "microsoft/deberta-xlarge-mnli"
    ) -> Dict[str, Dict[str, float]]:
        """Compare improved vs original sections using BERTScore.

        Args:
            improved_sections: Dict mapping section names to improved content
            original_sections: Dict mapping section names to original content
            model_type: BERT model to use for scoring

        Returns:
            Dict mapping section names to precision, recall, F1 scores
        """
        try:
            from bert_score import score as bertscore
        except ImportError:
            # Fallback to simple similarity if bert-score not installed
            return PaperMetrics._fallback_similarity(improved_sections, original_sections)

        results = {}

        for section_name in improved_sections:
            if section_name in original_sections:
                # Compute BERTScore
                P, R, F1 = bertscore(
                    [improved_sections[section_name]],
                    [original_sections[section_name]],
                    lang="en",
                    model_type=model_type,
                    verbose=False
                )

                results[section_name] = {
                    "precision": round(P.item(), 4),
                    "recall": round(R.item(), 4),
                    "f1": round(F1.item(), 4)
                }

        return results

    @staticmethod
    def _fallback_similarity(
        improved_sections: Dict[str, str],
        original_sections: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """Fallback similarity metric when BERTScore unavailable.

        Uses simple word overlap (Jaccard similarity).
        """
        results = {}

        for section_name in improved_sections:
            if section_name in original_sections:
                improved_words = set(improved_sections[section_name].lower().split())
                original_words = set(original_sections[section_name].lower().split())

                intersection = len(improved_words & original_words)
                union = len(improved_words | original_words)

                similarity = intersection / union if union > 0 else 0.0

                results[section_name] = {
                    "precision": round(similarity, 4),
                    "recall": round(similarity, 4),
                    "f1": round(similarity, 4)
                }

        return results

    @staticmethod
    def quadratic_weighted_kappa(
        human_scores: List[int],
        ai_scores: List[int],
        min_rating: int = 1,
        max_rating: int = 10
    ) -> float:
        """Calculate Quadratic Weighted Kappa between human and AI scores.

        QWK measures agreement between raters, accounting for degree of disagreement.

        Args:
            human_scores: List of human expert scores
            ai_scores: List of AI-generated scores
            min_rating: Minimum possible rating
            max_rating: Maximum possible rating

        Returns:
            QWK score between -1 and 1 (1 = perfect agreement)
        """
        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:
            # Fallback to correlation if sklearn not available
            return PaperMetrics.calculate_correlation(
                [float(x) for x in human_scores],
                [float(x) for x in ai_scores],
                method="pearson"
            )

        # Convert to numpy arrays
        human = np.array(human_scores)
        ai = np.array(ai_scores)

        # Calculate QWK
        qwk = cohen_kappa_score(
            human,
            ai,
            weights='quadratic',
            labels=list(range(min_rating, max_rating + 1))
        )

        return round(qwk, 4)

    @staticmethod
    def calculate_correlation(
        scores1: List[float],
        scores2: List[float],
        method: str = "pearson"
    ) -> float:
        """Calculate correlation between two sets of scores.

        Args:
            scores1: First set of scores
            scores2: Second set of scores
            method: "pearson" or "spearman"

        Returns:
            Correlation coefficient
        """
        try:
            from scipy.stats import pearsonr, spearmanr

            if method == "pearson":
                corr, _ = pearsonr(scores1, scores2)
            else:  # spearman
                corr, _ = spearmanr(scores1, scores2)

            return round(corr, 4)
        except ImportError:
            # Manual Pearson correlation as fallback
            return PaperMetrics._manual_pearson(scores1, scores2)

    @staticmethod
    def _manual_pearson(scores1: List[float], scores2: List[float]) -> float:
        """Manual Pearson correlation calculation."""
        n = len(scores1)
        if n == 0:
            return 0.0

        mean1 = sum(scores1) / n
        mean2 = sum(scores2) / n

        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(scores1, scores2))
        denom1 = sum((x - mean1) ** 2 for x in scores1) ** 0.5
        denom2 = sum((y - mean2) ** 2 for y in scores2) ** 0.5

        if denom1 == 0 or denom2 == 0:
            return 0.0

        return round(numerator / (denom1 * denom2), 4)

    @staticmethod
    def mean_absolute_error(
        true_scores: List[float],
        predicted_scores: List[float]
    ) -> float:
        """Calculate MAE between true and predicted scores.

        Args:
            true_scores: Ground truth scores
            predicted_scores: Predicted scores

        Returns:
            Mean absolute error
        """
        if len(true_scores) != len(predicted_scores):
            raise ValueError("Score lists must have same length")

        errors = [abs(t - p) for t, p in zip(true_scores, predicted_scores)]
        return round(np.mean(errors), 4)

    @staticmethod
    def root_mean_squared_error(
        true_scores: List[float],
        predicted_scores: List[float]
    ) -> float:
        """Calculate RMSE between true and predicted scores.

        Args:
            true_scores: Ground truth scores
            predicted_scores: Predicted scores

        Returns:
            Root mean squared error
        """
        if len(true_scores) != len(predicted_scores):
            raise ValueError("Score lists must have same length")

        squared_errors = [(t - p) ** 2 for t, p in zip(true_scores, predicted_scores)]
        return round(np.sqrt(np.mean(squared_errors)), 4)

    @staticmethod
    def calculate_accuracy(
        true_scores: List[int],
        predicted_scores: List[int],
        tolerance: int = 1
    ) -> float:
        """Calculate accuracy with tolerance.

        Args:
            true_scores: Ground truth scores
            predicted_scores: Predicted scores
            tolerance: Acceptable difference (default: ±1)

        Returns:
            Accuracy as percentage (0-1)
        """
        if len(true_scores) != len(predicted_scores):
            raise ValueError("Score lists must have same length")

        correct = sum(
            1 for t, p in zip(true_scores, predicted_scores)
            if abs(t - p) <= tolerance
        )

        return round(correct / len(true_scores), 4)

    @staticmethod
    def confusion_matrix(
        true_scores: List[int],
        predicted_scores: List[int],
        num_classes: int = 10
    ) -> np.ndarray:
        """Generate confusion matrix for score predictions.

        Args:
            true_scores: Ground truth scores
            predicted_scores: Predicted scores
            num_classes: Number of score classes (1-10)

        Returns:
            Confusion matrix as numpy array
        """
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true, pred in zip(true_scores, predicted_scores):
            # Convert to 0-indexed
            true_idx = min(max(true - 1, 0), num_classes - 1)
            pred_idx = min(max(pred - 1, 0), num_classes - 1)
            matrix[true_idx, pred_idx] += 1

        return matrix

    @staticmethod
    def calculate_f1_score(
        true_scores: List[int],
        predicted_scores: List[int],
        target_class: int
    ) -> float:
        """Calculate F1 score for a specific quality class.

        Args:
            true_scores: Ground truth scores
            predicted_scores: Predicted scores
            target_class: Target quality level (e.g., 8 for "high quality")

        Returns:
            F1 score for the target class
        """
        true_positive = sum(
            1 for t, p in zip(true_scores, predicted_scores)
            if t == target_class and p == target_class
        )

        false_positive = sum(
            1 for t, p in zip(true_scores, predicted_scores)
            if t != target_class and p == target_class
        )

        false_negative = sum(
            1 for t, p in zip(true_scores, predicted_scores)
            if t == target_class and p != target_class
        )

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1, 4)

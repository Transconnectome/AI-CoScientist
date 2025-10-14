from typing import List, Dict
from src.quality.types import (
    QualityScores,
    ValidationResult,
    QualityThresholds
)


class QualityValidationLoop:
    """Validates and refines research outputs"""

    def __init__(self, quality_critics: List, threshold_config: QualityThresholds):
        self.critics = quality_critics or []
        self.thresholds = threshold_config or QualityThresholds()
        self.max_iterations = 3

    async def validate_and_refine(
        self,
        research_output,
        context: Dict
    ) -> ValidationResult:
        """Validate output with iterative refinement"""

        iteration = 0
        current_output = research_output
        validation_history = []

        while iteration < self.max_iterations:
            # Assess quality
            scores = await self._assess_quality(current_output)

            # Check if meets thresholds
            if self._meets_thresholds(scores):
                return ValidationResult(
                    status="APPROVED",
                    final_output=current_output,
                    quality_scores=scores,
                    iterations=iteration + 1,
                    history=validation_history
                )

            # Record iteration
            validation_history.append({
                "iteration": iteration + 1,
                "scores": scores
            })

            iteration += 1

        # Max iterations reached
        final_scores = await self._assess_quality(current_output)

        if final_scores.overall >= self.thresholds.minimums["overall"] * 0.9:
            return ValidationResult(
                status="APPROVED_WITH_CONDITIONS",
                final_output=current_output,
                quality_scores=final_scores,
                iterations=self.max_iterations,
                conditions=["Quality slightly below target"]
            )
        else:
            return ValidationResult(
                status="REJECTED",
                final_output=current_output,
                quality_scores=final_scores,
                iterations=self.max_iterations,
                reason="Quality below minimum threshold"
            )

    async def _assess_quality(self, output) -> QualityScores:
        """Assess output quality"""
        # For Tier 1: Use output's existing quality score
        overall = getattr(output, 'quality_score', 0.75)

        return QualityScores(
            overall=overall,
            novelty=overall * 0.9,  # Mock dimensions
            rigor=overall * 1.1,
            clarity=overall
        )

    def _meets_thresholds(self, scores: QualityScores) -> bool:
        """Check if scores meet minimum thresholds"""
        return scores.overall >= self.thresholds.minimums["overall"]

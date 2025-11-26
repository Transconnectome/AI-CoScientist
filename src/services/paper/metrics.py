"""Paper quality evaluation metrics."""

from typing import Dict, List, Optional
import json
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

    @staticmethod
    def score_reproducibility(content: str, code_snippets: List[str]) -> float:
        """Calculate reproducibility and robustness score (0-10).
        
        Criteria:
        1. Code/Data Availability (3.0)
        2. Statistical Rigor (Power analysis, Effect size) (4.0)
        3. Verifiability (Code snippets present) (3.0)
        """
        import re
        score = 0.0
        
        # 1. Asset Availability (3.0)
        if re.search(r"(github\.com|gitlab\.com|zenodo\.org|figshare\.com)", content, re.IGNORECASE):
            score += 3.0
        elif re.search(r"(data availability|code availability)", content, re.IGNORECASE):
            score += 1.5
            
        # 2. Statistical Rigor (4.0)
        # Power analysis
        if re.search(r"(power analysis|sample size calculation|G\*Power)", content, re.IGNORECASE):
            score += 2.0
        # Effect size / CI
        if re.search(r"(Cohen's d|effect size|confidence interval|95% CI)", content, re.IGNORECASE):
            score += 2.0
            
        # 3. Verifiability (3.0)
        if code_snippets and len(code_snippets) > 0:
            score += 3.0
            
        return min(10.0, score)

    @staticmethod
    async def score_narrative_arc(
        abstract: str,
        introduction: str,
        use_llm: bool = False,
        llm_service=None
    ) -> Dict:
        """Score the narrative arc of a paper (hook, tension, resolution).
        
        Args:
            abstract: Paper abstract
            introduction: Paper introduction
            use_llm: Whether to use LLM-based analysis (default: False for heuristic)
            llm_service: LLM service instance (required if use_llm=True)
            
        Returns:
            Dict with hook_score (0-10), tension_curve, and feedback
        """
        if use_llm:
            if llm_service is None:
                raise ValueError("llm_service required when use_llm=True")
            
            # Use LLM-based analysis
            from src.services.paper.narrative_analyzer import NarrativeAnalyzer
            analyzer = NarrativeAnalyzer()
            return await analyzer.analyze_narrative(abstract, introduction, llm_service)
        
        # Fallback to heuristic analysis (original implementation)
        hook_score = 5.0  # Start at middle
        
        # Evaluate Hook (first sentence)
        first_sentence = abstract.split('.')[0] if abstract else ""
        
        # Short, punchy opening is good
        if len(first_sentence) < 100:
            hook_score += 1.0
        
        # Power words boost hook
        power_words = ["novel", "first", "breakthrough", "critical", "fundamental"]
        for word in power_words:
            if word.lower() in first_sentence.lower():
                hook_score += 1.0
                break
        
        # Evaluate Gap Definition (tension keywords)
        combined_text = (abstract + " " + introduction).lower()
        gap_indicators = ["however", "although", "remains unknown", "yet", "challenge"]
        
        tension_score = 0.0
        for indicator in gap_indicators:
            if indicator in combined_text:
                tension_score += 0.4
        
        tension_score = min(1.0, tension_score)
        hook_score += tension_score * 2.0  # Up to +2.0 for good gap definition
        
        # Simple tension curve (heuristic)
        # [opening, problem, climax, resolution]
        tension_curve = [0.2, tension_score * 0.8, 0.9 if tension_score > 0.5 else 0.4, 0.5]
        
        return {
            "hook_score": min(10.0, hook_score),
            "tension_curve": tension_curve,
            "story_elements": {
                "hook": first_sentence,
                "gap": "Detected" if tension_score > 0 else "Weak",
                "resolution": "See full text"
            },
            "feedback": f"Heuristic analysis. Hook: {hook_score:.1f}/10. Upgrade to LLM for deeper analysis."
        }

    @staticmethod
    async def retrieve_golden_references(
        topic: str,
        journal_tier: str = "top",
        n_references: int = 3
    ) -> Dict:
        """Retrieve golden reference papers from top-tier journals.
        
        Args:
            topic: Research topic or paper description
            journal_tier: Journal tier ("top", "high", "medium")
            n_references: Number of references to retrieve
            
        Returns:
            Dict with:
                - references: List of reference papers
                - metadata: Search metadata
        """
        # For now, use heuristic-based retrieval
        # In production, this would query ChromaDB or external APIs
        
        # Simulated top-tier references (in production, query ChromaDB)
        golden_papers = [
            {
                "title": "Deep learning for fMRI analysis: A comprehensive review",
                "journal": "Nature Neuroscience",
                "year": 2023,
                "similarity_score": 0.92,
                "impact_factor": 28.7
            },
            {
                "title": "Transformer architectures for brain state decoding",
                "journal": "Nature Methods",
                "year": 2024,
                "similarity_score": 0.88,
                "impact_factor": 25.3
            },
            {
                "title": "Multimodal brain imaging with deep neural networks",
                "journal": "Science",
                "year": 2023,
                "similarity_score": 0.85,
                "impact_factor": 47.7
            }
        ]
        
        # Filter by journal tier
        tier_impact_thresholds = {
            "top": 20.0,
            "high": 10.0,
            "medium": 5.0
        }
        threshold = tier_impact_thresholds.get(journal_tier, 20.0)
        filtered_papers = [
            p for p in golden_papers 
            if p["impact_factor"] >= threshold
        ]
        
        # Sort by similarity and limit
        filtered_papers.sort(key=lambda x: x["similarity_score"], reverse=True)
        references = filtered_papers[:n_references]
        
        return {
            "references": references,
            "metadata": {
                "query_topic": topic,
                "journal_tier": journal_tier,
                "total_found": len(references),
                "search_method": "heuristic"
            }
        }

    @staticmethod
    async def score_against_golden_references(
        paper_content: str,
        target_journal_tier: str = "top"
    ) -> Dict:
        """Score paper against golden reference benchmarks.
        
        Args:
            paper_content: The paper content to evaluate
            target_journal_tier: Target journal tier for comparison
            
        Returns:
            Dict with:
                - benchmark_score: How close to top-tier quality (0-10)
                - gap_analysis: Specific areas below benchmark
                - reference_papers: Top similar papers for learning
        """
        # Analyze paper content
        content_lower = paper_content.lower()
        
        # Benchmark criteria for top-tier papers
        benchmark_criteria = {
            "methodology": {
                "power_analysis": "power analysis" in content_lower,
                "effect_size": "effect size" in content_lower or "cohen" in content_lower,
                "reproducibility": "reproducible" in content_lower or "code" in content_lower,
                "statistical_rigor": "statistical" in content_lower and ("p <" in content_lower or "p-value" in content_lower)
            },
            "clarity": {
                "structured": any(section in content_lower for section in ["introduction", "methods", "results"]),
                "clear_hypothesis": "hypothesis" in content_lower or "hypothesize" in content_lower,
                "well_organized": len(content_lower.split(".")) > 5  # More than 5 sentences
            },
            "novelty": {
                "novel_contribution": "novel" in content_lower or "first" in content_lower,
                "gap_identification": "however" in content_lower or "gap" in content_lower
            },
            "significance": {
                "impact_discussion": "impact" in content_lower or "significance" in content_lower,
                "practical_application": "application" in content_lower or "clinical" in content_lower
            }
        }
        
        # Calculate benchmark score
        dimension_scores = {}
        for dimension, criteria in benchmark_criteria.items():
            met_criteria = sum(1 for criterion, met in criteria.items() if met)
            total_criteria = len(criteria)
            dimension_scores[dimension] = (met_criteria / total_criteria) * 10
        
        # Overall benchmark score (weighted average)
        weights = {"methodology": 0.35, "clarity": 0.25, "novelty": 0.20, "significance": 0.20}
        benchmark_score = sum(
            dimension_scores[dim] * weights[dim] 
            for dim in dimension_scores
        )
        
        # Generate gap analysis
        gap_analysis = []
        for dimension, score in dimension_scores.items():
            target_level = 8.5 if target_journal_tier == "top" else 7.0
            if score < target_level:
                gap = {
                    "dimension": dimension,
                    "current_level": round(score, 2),
                    "target_level": target_level,
                    "gap": round(target_level - score, 2),
                    "suggestions": PaperMetrics._generate_gap_suggestions(dimension, benchmark_criteria[dimension])
                }
                gap_analysis.append(gap)
        
        # Sort gaps by size (largest first)
        gap_analysis.sort(key=lambda x: x["gap"], reverse=True)
        
        return {
            "benchmark_score": round(benchmark_score, 2),
            "dimension_scores": {k: round(v, 2) for k, v in dimension_scores.items()},
            "gap_analysis": gap_analysis,
            "tier_assessment": PaperMetrics._assess_tier(benchmark_score)
        }
    
    @staticmethod
    def _generate_gap_suggestions(dimension: str, criteria: Dict[str, bool]) -> List[str]:
        """Generate improvement suggestions based on unmet criteria.
        
        Args:
            dimension: Quality dimension
            criteria: Dict of criteria and whether they were met
            
        Returns:
            List of specific suggestions
        """
        suggestions = []
        
        unmet = [criterion for criterion, met in criteria.items() if not met]
        
        suggestion_map = {
            "power_analysis": "Add statistical power analysis (G*Power recommended)",
            "effect_size": "Calculate and report effect sizes (Cohen's d, eta-squared)",
            "reproducibility": "Provide code/data availability statement with repository link",
            "statistical_rigor": "Include comprehensive statistical analysis with p-values and confidence intervals",
            "structured": "Follow IMRaD structure (Introduction, Methods, Results, Discussion)",
            "clear_hypothesis": "State clear, testable hypotheses upfront",
            "well_organized": "Improve organization with clear paragraph structure and transitions",
            "novel_contribution": "Explicitly state novel contributions and distinguish from prior work",
            "gap_identification": "Identify and articulate research gap being addressed",
            "impact_discussion": "Discuss broader impact and significance of findings",
            "practical_application": "Describe practical applications or clinical implications"
        }
        
        for criterion in unmet:
            if criterion in suggestion_map:
                suggestions.append(suggestion_map[criterion])
        
        return suggestions
    
    @staticmethod
    def _assess_tier(benchmark_score: float) -> str:
        """Assess journal tier based on benchmark score.
        
        Args:
            benchmark_score: Score from 0-10
            
        Returns:
            Tier assessment string
        """
        if benchmark_score >= 8.5:
            return "Top-tier (Nature, Science, Cell level)"
        elif benchmark_score >= 7.5:
            return "High-tier (Strong specialty journals)"
        elif benchmark_score >= 6.5:
            return "Mid-tier (Good specialty journals)"
        else:
            return "Needs significant improvement for top-tier publication"


class NatureMetrics:
    """Advanced metrics for Nature-level paper evaluation."""

    @staticmethod
    async def score_conceptual_novelty(
        abstract: str,
        introduction: str,
        llm_service
    ) -> Dict:
        """Score conceptual novelty (0-10)."""
        prompt = f"""
        Evaluate the conceptual novelty of this research for a top-tier journal (Nature/Science).
        
        Abstract: {abstract}
        Introduction: {introduction}
        
        Score from 0-10 based on:
        1. Paradigm Shift (0-4): Does it challenge existing dogma or introduce a new field?
        2. Incrementalism (0-3): Is it just "X applied to Y" (low score) or a fundamental advance?
        3. Surprise Factor (0-3): Is the finding unexpected?
        
        Return JSON: {{ "score": float, "reasoning": str }}
        """
        try:
            response = await llm_service.complete(prompt, max_tokens=200)
            # Handle potential markdown code blocks in response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            return {"score": 5.0, "reasoning": f"Failed to parse LLM response: {str(e)}"}

    @staticmethod
    async def score_cross_disciplinary_impact(
        abstract: str,
        discussion: str,
        llm_service
    ) -> Dict:
        """Score cross-disciplinary impact (0-10)."""
        prompt = f"""
        Evaluate the cross-disciplinary impact of this research.
        
        Abstract: {abstract}
        Discussion: {discussion}
        
        Score from 0-10 based on:
        1. Broad Relevance (0-5): Is this interesting to scientists outside the immediate sub-field?
        2. Accessibility (0-5): Is the significance clear to a general scientific audience?
        
        Return JSON: {{ "score": float, "reasoning": str }}
        """
        try:
            response = await llm_service.complete(prompt, max_tokens=200)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            return {"score": 5.0, "reasoning": f"Failed to parse LLM response: {str(e)}"}

    @staticmethod
    def score_methodological_rigor(content: str) -> float:
        """Score methodological rigor (0-10) based on keywords."""
        score = 0.0
        content_lower = content.lower()
        
        # Pre-registration / Open Science (3.0)
        if any(x in content_lower for x in ["preregist", "pre-regist", "osf.io", "clinicaltrials.gov"]):
            score += 3.0
            
        # Statistical Rigor (4.0)
        if "power analysis" in content_lower:
            score += 1.0
        if "multiple comparison" in content_lower or "bonferroni" in content_lower or "fdr" in content_lower:
            score += 1.0
        if "blind" in content_lower and "random" in content_lower:
            score += 1.0
        if "ablation" in content_lower or "sensitivity analysis" in content_lower:
            score += 1.0
            
        # Data/Code Availability (3.0)
        if "code availab" in content_lower or "github" in content_lower:
            score += 1.5
        if "data availab" in content_lower or "zenodo" in content_lower:
            score += 1.5
            
        return min(10.0, score)


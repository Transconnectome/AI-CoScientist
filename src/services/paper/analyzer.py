"""Paper analysis service for quality assessment and feedback."""

import json
from typing import Optional, Dict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.services.llm.service import LLMService
from src.models.project import Paper, PaperSection


class PaperAnalyzer:
    """Service for analyzing academic paper quality with SOTA methods."""

    def __init__(self, llm_service: LLMService, db: AsyncSession):
        """Initialize analyzer.

        Args:
            llm_service: LLM service for intelligent analysis
            db: Database session
        """
        self.llm = llm_service
        self.db = db

        # Lazy-load SOTA components (Phase 1 & 2 enhancement)
        self._scibert_scorer = None
        self._hybrid_scorer = None
        self._metrics = None

    def _get_scibert_scorer(self):
        """Lazy load SciBERT scorer."""
        if self._scibert_scorer is None:
            try:
                from src.services.paper.scibert_scorer import SciBERTQualityScorer
                self._scibert_scorer = SciBERTQualityScorer()
            except ImportError:
                # Fallback if dependencies not installed
                self._scibert_scorer = None
        return self._scibert_scorer

    def _get_hybrid_scorer(self):
        """Lazy load hybrid scorer (Phase 2 enhancement)."""
        if self._hybrid_scorer is None:
            try:
                from src.services.paper.hybrid_scorer import HybridPaperScorer
                import torch
                from pathlib import Path

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._hybrid_scorer = HybridPaperScorer(device=device)

                # Load trained weights if available
                model_path = Path("models/hybrid/best_model.pt")
                if model_path.exists():
                    self._hybrid_scorer.load_weights(str(model_path))
            except Exception:
                # Fallback if hybrid model not available
                self._hybrid_scorer = None
        return self._hybrid_scorer

    def _get_metrics(self):
        """Lazy load metrics calculator."""
        if self._metrics is None:
            from src.services.paper.metrics import PaperMetrics
            self._metrics = PaperMetrics()
        return self._metrics

    async def analyze_quality(
        self,
        paper_id: UUID,
        use_scibert: bool = True,
        use_ensemble: bool = True,
        use_hybrid: bool = False
    ) -> dict:
        """Analyze overall paper quality.

        Provides comprehensive quality assessment including:
        - Overall quality score (0-10)
        - Strengths and weaknesses
        - Section-specific suggestions
        - Coherence and clarity scores

        Args:
            paper_id: Paper UUID
            use_scibert: Enable SciBERT scoring (Phase 1)
            use_ensemble: Enable ensemble of GPT-4 + SciBERT (Phase 1)
            use_hybrid: Enable hybrid model scoring (Phase 2, requires training)

        Returns:
            Analysis results dictionary

        Example:
            {
                "quality_score": 7.5,
                "strengths": ["Clear methodology", "Strong results"],
                "weaknesses": ["Introduction too long", "Missing related work"],
                "suggestions": [
                    {"section": "introduction", "suggestion": "Reduce length..."}
                ],
                "coherence_score": 8.0,
                "clarity_score": 7.0
            }
        """
        # Get paper with sections
        paper = await self._get_paper_with_sections(paper_id)

        if not paper:
            raise ValueError(f"Paper {paper_id} not found")

        # Build analysis prompt
        sections_text = ""
        if hasattr(paper, 'sections') and paper.sections:
            for section in sorted(paper.sections, key=lambda s: s.order):
                sections_text += f"\n\n## {section.name.upper()}\n{section.content}"
        else:
            sections_text = paper.content or ""

        prompt = f"""
        Analyze this academic paper and provide a comprehensive quality assessment.

        Title: {paper.title}
        Abstract: {paper.abstract or "Not provided"}

        Content:
        {sections_text[:6000]}  # Limit to avoid token overflow

        Provide analysis as JSON with these fields:
        {{
            "quality_score": <float 0-10>,
            "strengths": [<list of 2-4 strengths>],
            "weaknesses": [<list of 2-4 weaknesses>],
            "suggestions": [
                {{"section": "<section_name>", "suggestion": "<improvement suggestion>"}},
                ...
            ],
            "coherence_score": <float 0-10, logical flow between sections>,
            "clarity_score": <float 0-10, writing clarity and readability>
        }}

        Be specific and constructive in feedback.
        Return only valid JSON.
        """

        # GPT-4 Analysis (original qualitative method)
        gpt4_analysis = {}
        try:
            result = await self.llm.complete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3
            )

            gpt4_analysis = json.loads(result.content)

            # Validate required fields
            required_fields = ["quality_score", "strengths", "weaknesses", "suggestions"]
            for field in required_fields:
                if field not in gpt4_analysis:
                    gpt4_analysis[field] = [] if field.endswith('s') else 0.0

        except json.JSONDecodeError:
            # Default structure
            gpt4_analysis = {
                "quality_score": 0.0,
                "strengths": [],
                "weaknesses": ["Unable to analyze paper"],
                "suggestions": [],
                "coherence_score": 0.0,
                "clarity_score": 0.0
            }

        # Full text for SOTA methods
        full_text = f"{paper.title}\n\n{paper.abstract or ''}\n\n{sections_text}"

        # Phase 2 Enhancement: Use hybrid model if available and enabled
        if use_hybrid:
            hybrid_scorer = self._get_hybrid_scorer()
            if hybrid_scorer and hybrid_scorer.is_trained:
                try:
                    hybrid_scores = await hybrid_scorer.score_paper(full_text)
                    gpt4_analysis["hybrid_scores"] = hybrid_scores

                    # Use hybrid as primary quality score (most accurate when trained)
                    gpt4_analysis["quality_score"] = hybrid_scores.get("overall_quality", gpt4_analysis.get("quality_score", 0.0))
                    gpt4_analysis["analysis_methods"] = ["gpt4", "hybrid"]
                except Exception as e:
                    gpt4_analysis["hybrid_error"] = str(e)
                    gpt4_analysis["analysis_methods"] = ["gpt4"]
            else:
                gpt4_analysis["analysis_methods"] = ["gpt4"]

        # Phase 1 Enhancement: Add SciBERT scoring if enabled (and hybrid not used)
        elif use_scibert:
            scibert_scorer = self._get_scibert_scorer()
            if scibert_scorer:
                try:
                    # Get SciBERT scores
                    scibert_scores = await scibert_scorer.score_paper(full_text)
                    gpt4_analysis["scibert_scores"] = scibert_scores

                    # Ensemble scoring if enabled
                    if use_ensemble and "quality_score" in gpt4_analysis:
                        ensemble_score = self._compute_ensemble_score(
                            gpt4_analysis["quality_score"],
                            scibert_scores.get("overall_quality", 7.0)
                        )
                        gpt4_analysis["ensemble_score"] = ensemble_score
                        gpt4_analysis["quality_score"] = ensemble_score  # Use ensemble as primary

                    gpt4_analysis["analysis_methods"] = ["gpt4", "scibert"]
                except Exception as e:
                    # Fallback to GPT-4 only
                    gpt4_analysis["scibert_error"] = str(e)
                    gpt4_analysis["analysis_methods"] = ["gpt4"]
            else:
                gpt4_analysis["analysis_methods"] = ["gpt4"]
        else:
            gpt4_analysis["analysis_methods"] = ["gpt4"]

        return gpt4_analysis

    def _compute_ensemble_score(
        self,
        gpt4_score: float,
        scibert_score: float,
        gpt4_weight: float = 0.4,
        scibert_weight: float = 0.6
    ) -> float:
        """Compute weighted ensemble of GPT-4 and SciBERT scores.

        Args:
            gpt4_score: GPT-4 quality score
            scibert_score: SciBERT quality score
            gpt4_weight: Weight for GPT-4 (default: 0.4)
            scibert_weight: Weight for SciBERT (default: 0.6)

        Returns:
            Weighted ensemble score
        """
        ensemble = gpt4_weight * gpt4_score + scibert_weight * scibert_score
        return round(ensemble, 2)

    async def check_section_coherence(self, paper_id: UUID) -> dict:
        """Check coherence between paper sections.

        Analyzes logical flow and consistency across sections.

        Args:
            paper_id: Paper UUID

        Returns:
            Coherence analysis results
        """
        paper = await self._get_paper_with_sections(paper_id)

        if not paper or not hasattr(paper, 'sections'):
            return {"coherence_score": 0.0, "issues": ["No sections found"]}

        sections = sorted(paper.sections, key=lambda s: s.order)

        if len(sections) < 2:
            return {"coherence_score": 10.0, "issues": []}

        # Build section summaries for comparison
        sections_summary = "\n\n".join([
            f"## {s.name}\n{s.content[:500]}..."  # First 500 chars
            for s in sections
        ])

        prompt = f"""
        Analyze the coherence and logical flow between these paper sections.

        {sections_summary}

        Check for:
        - Logical progression from introduction to conclusion
        - Consistency between methods and results
        - Alignment between research questions and findings
        - Proper transitions between sections

        Return JSON:
        {{
            "coherence_score": <float 0-10>,
            "issues": [<list of coherence issues>],
            "recommendations": [<list of recommendations>]
        }}
        """

        try:
            result = await self.llm.complete(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.3
            )

            return json.loads(result.content)

        except json.JSONDecodeError:
            return {
                "coherence_score": 5.0,
                "issues": [],
                "recommendations": []
            }

    async def identify_gaps(self, paper_id: UUID) -> list[dict]:
        """Identify missing or underdeveloped content.

        Args:
            paper_id: Paper UUID

        Returns:
            List of identified gaps with recommendations
        """
        paper = await self._get_paper_with_sections(paper_id)

        if not paper:
            return []

        # Get section names
        section_names = []
        if hasattr(paper, 'sections'):
            section_names = [s.name for s in paper.sections]

        prompt = f"""
        Analyze this academic paper for missing or underdeveloped content.

        Title: {paper.title}
        Existing sections: {', '.join(section_names)}

        Common academic paper sections:
        - Abstract, Introduction, Related Work, Methods, Results, Discussion, Conclusion, References

        Identify:
        1. Missing sections that should be added
        2. Underdeveloped sections that need expansion
        3. Missing elements (literature review, statistical analysis, limitations, etc.)

        Return JSON array:
        [
            {{
                "gap_type": "missing_section" | "underdeveloped" | "missing_element",
                "description": "<what's missing>",
                "recommendation": "<what to add>",
                "priority": "high" | "medium" | "low"
            }},
            ...
        ]
        """

        try:
            result = await self.llm.complete(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.3
            )

            gaps = json.loads(result.content)
            return gaps if isinstance(gaps, list) else []

        except json.JSONDecodeError:
            return []

    async def _get_paper_with_sections(self, paper_id: UUID) -> Optional[Paper]:
        """Get paper with eagerly loaded sections.

        Args:
            paper_id: Paper UUID

        Returns:
            Paper object with sections or None
        """
        query = select(Paper).where(Paper.id == paper_id)
        query = query.options(selectinload(Paper.sections))

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

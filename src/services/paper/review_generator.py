"""Automated review generation system inspired by REVIEWER2.

Generates structured, actionable reviews combining:
1. Multi-task model scores (5 dimensions)
2. GPT-4 qualitative analysis
3. Citation analysis and related work assessment
4. Structured feedback templates

Output format similar to conference peer reviews.
"""

from typing import Dict, List, Optional
from uuid import UUID
import json

from sqlalchemy.ext.asyncio import AsyncSession

from src.services.llm.service import LLMService
from src.services.paper.analyzer import PaperAnalyzer


class AutomatedReviewGenerator:
    """Generate comprehensive paper reviews combining SOTA scoring and LLM analysis."""

    def __init__(
        self,
        llm_service: LLMService,
        db: AsyncSession,
        use_multitask: bool = True
    ):
        """Initialize review generator.

        Args:
            llm_service: LLM service for qualitative analysis
            db: Database session
            use_multitask: Use multi-task model for 5-dimensional scoring
        """
        self.llm = llm_service
        self.db = db
        self.use_multitask = use_multitask

        # Lazy-load components
        self._analyzer = None
        self._multitask_scorer = None

    def _get_analyzer(self) -> PaperAnalyzer:
        """Get paper analyzer instance."""
        if self._analyzer is None:
            self._analyzer = PaperAnalyzer(self.llm, self.db)
        return self._analyzer

    def _get_multitask_scorer(self):
        """Lazy load multi-task scorer."""
        if self._multitask_scorer is None:
            try:
                from src.services.paper.multitask_scorer import MultiTaskPaperScorer
                import torch
                from pathlib import Path

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._multitask_scorer = MultiTaskPaperScorer(device=device)

                # Load trained weights if available
                model_path = Path("models/multitask/best_model.pt")
                if model_path.exists():
                    self._multitask_scorer.load_weights(str(model_path))
            except Exception:
                self._multitask_scorer = None
        return self._multitask_scorer

    async def generate_review(
        self,
        paper_id: UUID,
        review_type: str = "conference",
        include_recommendations: bool = True
    ) -> Dict:
        """Generate comprehensive review for paper.

        Args:
            paper_id: Paper UUID
            review_type: Type of review ("conference", "journal", "workshop")
            include_recommendations: Include specific improvement recommendations

        Returns:
            Structured review dictionary
        """
        analyzer = self._get_analyzer()

        # Get paper content
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        from src.models.project import Paper

        query = select(Paper).where(Paper.id == paper_id)
        query = query.options(selectinload(Paper.sections))

        result = await self.db.execute(query)
        paper = result.scalar_one_or_none()

        if not paper:
            raise ValueError(f"Paper {paper_id} not found")

        # Build full text
        sections_text = ""
        if hasattr(paper, 'sections') and paper.sections:
            for section in sorted(paper.sections, key=lambda s: s.order):
                sections_text += f"\n\n## {section.name.upper()}\n{section.content}"
        else:
            sections_text = paper.content or ""

        full_text = f"{paper.title}\n\n{paper.abstract or ''}\n\n{sections_text}"

        # Step 1: Get multi-dimensional quality scores
        multitask_scores = None
        if self.use_multitask:
            multitask_scorer = self._get_multitask_scorer()
            if multitask_scorer and multitask_scorer.is_trained:
                try:
                    multitask_scores = await multitask_scorer.score_paper(full_text)
                except Exception:
                    pass

        # Step 2: Get qualitative analysis from GPT-4
        qualitative_analysis = await analyzer.analyze_quality(paper_id)

        # Step 3: Generate structured review sections
        review = await self._generate_review_sections(
            paper=paper,
            full_text=full_text,
            multitask_scores=multitask_scores,
            qualitative_analysis=qualitative_analysis,
            review_type=review_type,
            include_recommendations=include_recommendations
        )

        return review

    async def _generate_review_sections(
        self,
        paper,
        full_text: str,
        multitask_scores: Optional[Dict],
        qualitative_analysis: Dict,
        review_type: str,
        include_recommendations: bool
    ) -> Dict:
        """Generate structured review sections.

        Args:
            paper: Paper model instance
            full_text: Full paper text
            multitask_scores: Multi-dimensional quality scores
            qualitative_analysis: Qualitative analysis from GPT-4
            review_type: Review type
            include_recommendations: Include recommendations

        Returns:
            Structured review
        """
        # Extract scores
        if multitask_scores:
            overall_score = multitask_scores.get("overall_quality", 0)
            novelty_score = multitask_scores.get("novelty_quality", 0)
            methodology_score = multitask_scores.get("methodology_quality", 0)
            clarity_score = multitask_scores.get("clarity_quality", 0)
            significance_score = multitask_scores.get("significance_quality", 0)
        else:
            overall_score = qualitative_analysis.get("quality_score", 0)
            novelty_score = overall_score
            methodology_score = overall_score
            clarity_score = qualitative_analysis.get("clarity_score", overall_score)
            significance_score = overall_score

        # Generate detailed review sections using LLM
        review_prompt = f"""
        Generate a structured peer review for this academic paper.

        Title: {paper.title}
        Abstract: {paper.abstract or "Not provided"}

        Content Preview:
        {full_text[:3000]}

        Quality Scores (1-10 scale):
        - Overall Quality: {overall_score:.1f}
        - Novelty: {novelty_score:.1f}
        - Methodology: {methodology_score:.1f}
        - Clarity: {clarity_score:.1f}
        - Significance: {significance_score:.1f}

        Qualitative Analysis:
        - Strengths: {', '.join(qualitative_analysis.get('strengths', []))}
        - Weaknesses: {', '.join(qualitative_analysis.get('weaknesses', []))}

        Generate a {review_type} review with these sections:

        {{
            "summary": "<1-2 paragraph summary of paper's contribution>",
            "strengths": [<list of 3-5 specific strengths with examples>],
            "weaknesses": [<list of 3-5 specific weaknesses with examples>],
            "detailed_comments": {{
                "novelty": "<assessment of originality and contribution>",
                "methodology": "<assessment of technical approach and rigor>",
                "clarity": "<assessment of writing quality and organization>",
                "significance": "<assessment of impact and importance>"
            }},
            "questions_for_authors": [<2-4 clarifying questions>],
            "recommendation": "<accept|weak accept|borderline|weak reject|reject>",
            "confidence": "<high|medium|low>"
        }}

        Be specific, constructive, and professional. Cite specific sections when providing feedback.
        Return only valid JSON.
        """

        try:
            result = await self.llm.complete(
                prompt=review_prompt,
                max_tokens=3000,
                temperature=0.4
            )

            review_sections = json.loads(result.content)

        except json.JSONDecodeError:
            # Fallback structure
            review_sections = {
                "summary": "Unable to generate detailed review.",
                "strengths": qualitative_analysis.get("strengths", []),
                "weaknesses": qualitative_analysis.get("weaknesses", []),
                "detailed_comments": {},
                "questions_for_authors": [],
                "recommendation": "borderline",
                "confidence": "low"
            }

        # Add recommendations if requested
        if include_recommendations:
            recommendations = await self._generate_recommendations(
                paper_id=paper.id,
                weaknesses=review_sections.get("weaknesses", []),
                qualitative_analysis=qualitative_analysis
            )
            review_sections["improvement_recommendations"] = recommendations

        # Compile final review
        final_review = {
            "paper_id": str(paper.id),
            "paper_title": paper.title,
            "review_type": review_type,
            "scores": {
                "overall": round(overall_score, 1),
                "novelty": round(novelty_score, 1),
                "methodology": round(methodology_score, 1),
                "clarity": round(clarity_score, 1),
                "significance": round(significance_score, 1)
            },
            "review": review_sections,
            "analysis_methods": qualitative_analysis.get("analysis_methods", ["gpt4"])
        }

        return final_review

    async def _generate_recommendations(
        self,
        paper_id: UUID,
        weaknesses: List[str],
        qualitative_analysis: Dict
    ) -> List[Dict]:
        """Generate specific improvement recommendations.

        Args:
            paper_id: Paper UUID
            weaknesses: List of identified weaknesses
            qualitative_analysis: Qualitative analysis results

        Returns:
            List of actionable recommendations
        """
        # Combine weaknesses and suggestions
        all_issues = weaknesses + [
            s.get("suggestion", "")
            for s in qualitative_analysis.get("suggestions", [])
        ]

        if not all_issues:
            return []

        recommendation_prompt = f"""
        Based on these identified issues in a scientific paper:

        {chr(10).join(f"- {issue}" for issue in all_issues[:10])}

        Generate 3-5 specific, actionable improvement recommendations.

        Return as JSON array:
        [
            {{
                "priority": "high" | "medium" | "low",
                "category": "methodology" | "writing" | "analysis" | "structure",
                "issue": "<brief description of the issue>",
                "recommendation": "<specific action to address it>",
                "example": "<concrete example or guidance>"
            }},
            ...
        ]

        Be specific and actionable. Focus on improvements that would most increase paper quality.
        Return only valid JSON array.
        """

        try:
            result = await self.llm.complete(
                prompt=recommendation_prompt,
                max_tokens=1500,
                temperature=0.3
            )

            recommendations = json.loads(result.content)
            return recommendations if isinstance(recommendations, list) else []

        except json.JSONDecodeError:
            return []

    def format_review_markdown(self, review: Dict) -> str:
        """Format review as markdown for easy reading.

        Args:
            review: Review dictionary

        Returns:
            Formatted markdown string
        """
        md = []

        # Header
        md.append(f"# Peer Review: {review['paper_title']}\n")
        md.append(f"**Review Type**: {review['review_type'].capitalize()}\n")

        # Scores
        md.append("## Quality Scores\n")
        scores = review["scores"]
        md.append(f"- **Overall Quality**: {scores['overall']:.1f} / 10")
        md.append(f"- **Novelty**: {scores['novelty']:.1f} / 10")
        md.append(f"- **Methodology**: {scores['methodology']:.1f} / 10")
        md.append(f"- **Clarity**: {scores['clarity']:.1f} / 10")
        md.append(f"- **Significance**: {scores['significance']:.1f} / 10\n")

        review_content = review["review"]

        # Summary
        md.append("## Summary\n")
        md.append(review_content.get("summary", "") + "\n")

        # Strengths
        md.append("## Strengths\n")
        for strength in review_content.get("strengths", []):
            md.append(f"- {strength}")
        md.append("")

        # Weaknesses
        md.append("## Weaknesses\n")
        for weakness in review_content.get("weaknesses", []):
            md.append(f"- {weakness}")
        md.append("")

        # Detailed Comments
        detailed = review_content.get("detailed_comments", {})
        if detailed:
            md.append("## Detailed Comments\n")
            for aspect, comment in detailed.items():
                md.append(f"### {aspect.capitalize()}\n")
                md.append(f"{comment}\n")

        # Questions
        questions = review_content.get("questions_for_authors", [])
        if questions:
            md.append("## Questions for Authors\n")
            for i, question in enumerate(questions, 1):
                md.append(f"{i}. {question}")
            md.append("")

        # Recommendations
        recommendations = review_content.get("improvement_recommendations", [])
        if recommendations:
            md.append("## Improvement Recommendations\n")
            for rec in recommendations:
                priority = rec.get("priority", "medium").upper()
                category = rec.get("category", "general").capitalize()
                md.append(f"### [{priority}] {category}: {rec.get('issue', '')}\n")
                md.append(f"**Recommendation**: {rec.get('recommendation', '')}\n")
                if rec.get("example"):
                    md.append(f"**Example**: {rec.get('example')}\n")

        # Final Recommendation
        md.append("## Recommendation\n")
        md.append(f"**Decision**: {review_content.get('recommendation', 'borderline').upper()}")
        md.append(f"**Confidence**: {review_content.get('confidence', 'medium').capitalize()}\n")

        # Metadata
        md.append("---")
        md.append(f"*Analysis methods: {', '.join(review['analysis_methods'])}*")

        return "\n".join(md)

"""Hypothesis generation service using LLM with GPT Researcher integration."""

from typing import Any, Dict, List, Optional
from uuid import UUID
import json
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.project import Hypothesis, Project
from src.services.llm import LLMService
from src.services.llm.types import LLMRequest, TaskType
from src.services.knowledge_base.search import KnowledgeBaseSearch
from src.services.external.gpt_researcher_service import GPTResearcherService


logger = logging.getLogger(__name__)


class HypothesisGenerator:
    """Generate scientific hypotheses using LLM with systematic literature review."""

    def __init__(
        self,
        llm_service: LLMService,
        knowledge_base: KnowledgeBaseSearch,
        db: AsyncSession,
        gpt_researcher: Optional[GPTResearcherService] = None
    ):
        """
        Initialize hypothesis generator.

        Args:
            llm_service: LLM service for generation
            knowledge_base: Local knowledge base search
            db: Database session
            gpt_researcher: GPT Researcher service (optional, created if not provided)
        """
        self.llm_service = llm_service
        self.knowledge_base = knowledge_base
        self.db = db

        # Initialize GPT Researcher
        try:
            self.gpt_researcher = gpt_researcher or GPTResearcherService()
            self.use_gpt_researcher = True
            logger.info("GPT Researcher enabled for systematic literature review")
        except Exception as e:
            logger.warning(f"GPT Researcher not available: {e}")
            self.gpt_researcher = None
            self.use_gpt_researcher = False

    async def generate_hypotheses(
        self,
        project_id: UUID,
        research_question: str,
        num_hypotheses: int = 5,
        creativity_level: float = 0.7,
        literature_context: Optional[List[str]] = None,
        use_systematic_review: bool = True
    ) -> List[Hypothesis]:
        """
        Generate hypotheses for a research question with systematic literature review.

        Args:
            project_id: Project identifier
            research_question: Research question
            num_hypotheses: Number of hypotheses to generate
            creativity_level: Creativity level (0-1)
            literature_context: Optional literature context
            use_systematic_review: Use GPT Researcher for systematic review

        Returns:
            List of generated hypotheses
        """
        # Get project
        project_query = select(Project).where(Project.id == project_id)
        project_result = await self.db.execute(project_query)
        project = project_result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Get existing hypotheses to avoid duplication
        existing_query = select(Hypothesis).where(Hypothesis.project_id == project_id)
        existing_result = await self.db.execute(existing_query)
        existing_hypotheses = existing_result.scalars().all()

        # Perform systematic literature review with GPT Researcher
        literature_summary = ""
        gpt_research_data = None

        if use_systematic_review and self.use_gpt_researcher:
            logger.info("Performing systematic literature review with GPT Researcher")
            try:
                gpt_research_data = await self.gpt_researcher.systematic_literature_review(
                    research_question=research_question,
                    domain=project.domain,
                    depth="medium"
                )

                if gpt_research_data["success"]:
                    literature_summary = gpt_research_data["report"]
                    logger.info(f"GPT Researcher found {gpt_research_data['num_sources']} sources")
                else:
                    logger.warning("GPT Researcher failed, falling back to basic search")

            except Exception as e:
                logger.error(f"GPT Researcher error: {e}", exc_info=True)

        # Fallback to basic literature search if GPT Researcher not available
        if not literature_summary:
            logger.info("Using basic knowledge base search")
            literature_summary = await self._get_literature_summary(
                research_question,
                project.domain,
                literature_context
            )

        # Prepare enhanced context for prompt
        context = {
            "domain": project.domain,
            "research_question": research_question,
            "literature_summary": literature_summary,
            "num_hypotheses": num_hypotheses,
            "existing_hypotheses": [h.content for h in existing_hypotheses],
            "systematic_review_used": gpt_research_data is not None,
            "num_sources_reviewed": gpt_research_data.get("num_sources", 0) if gpt_research_data else 0
        }

        # Generate hypotheses using LLM
        response = await self.llm_service.complete_with_template(
            template_name="hypothesis_generation",
            context=context,
            task_type=TaskType.HYPOTHESIS_GENERATION,
            system_message=(
                "You are a scientific research assistant specialized in "
                "generating novel, testable hypotheses based on comprehensive literature analysis. "
                f"{'A systematic literature review with ' + str(context['num_sources_reviewed']) + ' sources has been conducted.' if context['systematic_review_used'] else 'Basic literature search performed.'}"
            )
        )

        # Parse hypotheses from response
        hypotheses = self._parse_hypotheses_response(response.content)

        # Create and save hypotheses
        saved_hypotheses = []
        for hyp_data in hypotheses:
            hypothesis = Hypothesis(
                project_id=project_id,
                content=hyp_data["statement"],
                rationale=hyp_data.get("rationale"),
                novelty_score=hyp_data.get("novelty_score", 0.5),
                testability_score=hyp_data.get("testability_score")
            )
            self.db.add(hypothesis)
            saved_hypotheses.append(hypothesis)

        await self.db.commit()

        # Refresh to get IDs
        for hyp in saved_hypotheses:
            await self.db.refresh(hyp)

        logger.info(f"Generated {len(saved_hypotheses)} hypotheses with {'systematic' if gpt_research_data else 'basic'} literature review")
        return saved_hypotheses

    async def validate_hypothesis(
        self,
        hypothesis_id: UUID
    ) -> Dict[str, Any]:
        """Validate hypothesis for novelty and testability."""
        # Get hypothesis
        query = select(Hypothesis).where(Hypothesis.id == hypothesis_id)
        result = await self.db.execute(query)
        hypothesis = result.scalar_one_or_none()

        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")

        # Check for similar work in literature
        similar_papers = await self.knowledge_base.semantic_search(
            query=hypothesis.content,
            top_k=10,
            min_score=0.6
        )

        # Analyze novelty
        is_novel = len(similar_papers) < 5 or all(
            paper.score < 0.8 for paper in similar_papers
        )

        novelty_score = 1.0 - (
            max([paper.score for paper in similar_papers]) if similar_papers else 0.0
        )

        # Build validation prompt
        validation_prompt = f"""
        Analyze the following scientific hypothesis for testability and feasibility:

        **Hypothesis**: {hypothesis.content}

        **Rationale**: {hypothesis.rationale or "Not provided"}

        **Similar Work Found**:
        {self._format_similar_work(similar_papers)}

        Provide analysis on:
        1. **Testability**: Can this be tested experimentally? How?
        2. **Feasibility**: Is it feasible with current technology?
        3. **Novelty Assessment**: How does it differ from similar work?
        4. **Suggested Methods**: What experimental methods would test this?

        Format your response as JSON with keys: testability_score (0-1),
        is_testable (bool), feasibility (high/medium/low), suggested_methods (array),
        novelty_assessment (string).
        """

        # Get LLM analysis
        request = LLMRequest(
            prompt=validation_prompt,
            task_type=TaskType.HYPOTHESIS_GENERATION,
            system_message="You are a scientific methodology expert."
        )

        response = await self.llm_service.complete(request)

        # Parse response
        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback if not valid JSON
            analysis = {
                "testability_score": 0.5,
                "is_testable": True,
                "feasibility": "unknown",
                "suggested_methods": [],
                "novelty_assessment": response.content[:200]
            }

        # Update hypothesis
        hypothesis.novelty_score = novelty_score
        hypothesis.testability_score = analysis.get("testability_score", 0.5)

        if hypothesis.novelty_score > 0.7 and hypothesis.testability_score > 0.7:
            hypothesis.status = "validated"

        await self.db.commit()

        return {
            "hypothesis_id": str(hypothesis_id),
            "is_novel": is_novel,
            "novelty_score": novelty_score,
            "similar_work": [
                {
                    "title": paper.title,
                    "similarity": paper.score,
                    "abstract": paper.abstract[:200] + "..."
                }
                for paper in similar_papers[:5]
            ],
            **analysis
        }

    async def _get_literature_summary(
        self,
        research_question: str,
        domain: str,
        literature_ids: Optional[List[str]] = None
    ) -> str:
        """Get summary of relevant literature."""
        if literature_ids:
            # Use specified literature
            papers = []
            for lit_id in literature_ids:
                result = await self.knowledge_base.semantic_search(
                    query=lit_id,
                    top_k=1
                )
                if result:
                    papers.extend(result)
        else:
            # Search for relevant papers
            papers = await self.knowledge_base.semantic_search(
                query=f"{domain} {research_question}",
                top_k=10,
                min_score=0.5
            )

        if not papers:
            return "No relevant literature found."

        # Summarize papers
        summary_parts = []
        for i, paper in enumerate(papers[:5], 1):
            summary_parts.append(
                f"{i}. {paper.title}\n"
                f"   {paper.abstract[:200]}..."
            )

        return "\n\n".join(summary_parts)

    def _parse_hypotheses_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse hypotheses from LLM response."""
        hypotheses = []

        # Try to extract structured data
        lines = response.split("\n")
        current_hyp: Dict[str, Any] = {}

        for line in lines:
            line = line.strip()

            if line.startswith("**Hypothesis"):
                if current_hyp:
                    hypotheses.append(current_hyp)
                current_hyp = {}

            elif "**Statement**:" in line:
                current_hyp["statement"] = line.split("**Statement**:")[-1].strip()

            elif "**Rationale**:" in line:
                current_hyp["rationale"] = line.split("**Rationale**:")[-1].strip()

            elif "**Testability**:" in line:
                current_hyp["testability"] = line.split("**Testability**:")[-1].strip()

            elif "**Expected Outcome**:" in line:
                current_hyp["expected_outcome"] = line.split("**Expected Outcome**:")[-1].strip()

            elif "**Novelty Score**:" in line:
                try:
                    score_str = line.split("**Novelty Score**:")[-1].strip()
                    current_hyp["novelty_score"] = float(score_str)
                except ValueError:
                    current_hyp["novelty_score"] = 0.5

        # Add last hypothesis
        if current_hyp and "statement" in current_hyp:
            hypotheses.append(current_hyp)

        # Fallback: if parsing failed, create simple hypotheses
        if not hypotheses:
            # Split by numbered list
            parts = response.split("\n\n")
            for part in parts:
                if part.strip():
                    hypotheses.append({
                        "statement": part.strip(),
                        "rationale": "Generated from research question",
                        "novelty_score": 0.5
                    })

        return hypotheses

    def _format_similar_work(self, papers: List[Any]) -> str:
        """Format similar papers for display."""
        if not papers:
            return "No similar work found."

        formatted = []
        for i, paper in enumerate(papers[:5], 1):
            formatted.append(
                f"{i}. {paper.title} (similarity: {paper.score:.2f})\n"
                f"   {paper.abstract[:150]}..."
            )

        return "\n\n".join(formatted)

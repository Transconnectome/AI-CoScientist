"""
GPT Researcher Service Integration

Provides systematic literature review and hypothesis validation
using GPT Researcher framework.
"""

import os
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import logging

from gpt_researcher import GPTResearcher


logger = logging.getLogger(__name__)


class GPTResearcherService:
    """Wrapper for GPT Researcher with AI-CoScientist integration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        report_type: str = "research_report",
        max_iterations: int = 3
    ):
        """
        Initialize GPT Researcher service.

        Args:
            api_key: OpenAI API key (uses env var if not provided)
            report_type: Type of research report to generate
            max_iterations: Maximum search iterations
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required for GPT Researcher")

        self.report_type = report_type
        self.max_iterations = max_iterations

    async def systematic_literature_review(
        self,
        research_question: str,
        domain: Optional[str] = None,
        depth: str = "medium"
    ) -> Dict[str, Any]:
        """
        Perform systematic literature review with query decomposition.

        Args:
            research_question: Main research question
            domain: Research domain (neuroscience, biology, etc.)
            depth: Search depth (quick, medium, deep)

        Returns:
            Structured literature review results
        """
        logger.info(f"Starting systematic literature review: {research_question}")

        # Enhance query with domain context
        if domain:
            enhanced_query = f"{domain}: {research_question}"
        else:
            enhanced_query = research_question

        # Configure researcher
        researcher = GPTResearcher(
            query=enhanced_query,
            report_type=self.report_type,
            config_path=None
        )

        # Run research
        try:
            # Conduct research
            await researcher.conduct_research()

            # Generate report
            report = await researcher.write_report()

            # Get sources and context
            sources = researcher.get_source_urls()
            context = researcher.get_research_context()

            return {
                "success": True,
                "research_question": research_question,
                "domain": domain,
                "report": report,
                "sources": sources,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "num_sources": len(sources)
            }

        except Exception as e:
            logger.error(f"Literature review failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "research_question": research_question
            }

    async def decompose_research_question(
        self,
        question: str,
        num_subquestions: int = 5
    ) -> List[str]:
        """
        Decompose complex research question into sub-questions.

        Args:
            question: Complex research question
            num_subquestions: Number of sub-questions to generate

        Returns:
            List of focused sub-questions
        """
        decomposition_query = f"""
        Decompose this research question into {num_subquestions} focused sub-questions
        for systematic literature review:

        Main Question: {question}

        Generate specific, answerable sub-questions that cover:
        1. Current state of knowledge
        2. Existing methodologies
        3. Known limitations
        4. Open problems
        5. Recent developments

        Return only the sub-questions, one per line.
        """

        researcher = GPTResearcher(
            query=decomposition_query,
            report_type="outline_report"
        )

        try:
            await researcher.conduct_research()
            report = await researcher.write_report()

            # Parse sub-questions from report
            lines = report.strip().split('\n')
            subquestions = [
                line.strip().lstrip('0123456789.-) ')
                for line in lines
                if line.strip() and len(line.strip()) > 20
            ][:num_subquestions]

            logger.info(f"Generated {len(subquestions)} sub-questions")
            return subquestions

        except Exception as e:
            logger.error(f"Question decomposition failed: {e}")
            return [question]  # Fallback to original question

    async def multi_hop_literature_search(
        self,
        initial_query: str,
        max_hops: int = 3,
        entities_per_hop: int = 3
    ) -> Dict[str, Any]:
        """
        Perform multi-hop literature search following entities and concepts.

        Args:
            initial_query: Starting research question
            max_hops: Maximum search depth
            entities_per_hop: Entities to follow per hop

        Returns:
            Comprehensive search results with hop history
        """
        logger.info(f"Starting multi-hop search: {initial_query}")

        all_sources = []
        hop_history = []
        visited_topics = set()

        current_query = initial_query

        for hop in range(max_hops):
            logger.info(f"Hop {hop + 1}/{max_hops}: {current_query}")

            # Research current query
            researcher = GPTResearcher(
                query=current_query,
                report_type="research_report"
            )

            try:
                await researcher.conduct_research()
                report = await researcher.write_report()
                sources = researcher.get_source_urls()

                hop_data = {
                    "hop": hop + 1,
                    "query": current_query,
                    "sources": sources,
                    "num_sources": len(sources)
                }
                hop_history.append(hop_data)
                all_sources.extend(sources)

                # Extract entities for next hop (simplified)
                # In production, use NER or entity extraction
                if hop < max_hops - 1:
                    # Generate follow-up query based on findings
                    next_query = f"Recent developments related to: {current_query}"
                    current_query = next_query
                    visited_topics.add(current_query)

            except Exception as e:
                logger.error(f"Hop {hop + 1} failed: {e}")
                break

        return {
            "initial_query": initial_query,
            "total_hops": len(hop_history),
            "total_sources": len(set(all_sources)),
            "hop_history": hop_history,
            "all_sources": list(set(all_sources)),
            "timestamp": datetime.now().isoformat()
        }

    async def validate_hypothesis_against_literature(
        self,
        hypothesis: str,
        domain: str
    ) -> Dict[str, Any]:
        """
        Validate hypothesis against existing literature.

        Args:
            hypothesis: Scientific hypothesis to validate
            domain: Research domain

        Returns:
            Validation results with novelty and evidence assessment
        """
        validation_query = f"""
        Analyze this scientific hypothesis against existing literature:

        Hypothesis: {hypothesis}
        Domain: {domain}

        Provide:
        1. Similar existing work (if any)
        2. Supporting evidence from literature
        3. Contradicting evidence (if any)
        4. Novelty assessment (paradigm shift vs incremental)
        5. Testability and feasibility
        """

        researcher = GPTResearcher(
            query=validation_query,
            report_type="research_report"
        )

        try:
            await researcher.conduct_research()
            report = await researcher.write_report()
            sources = researcher.get_source_urls()

            # Parse validation results (simplified)
            # In production, use structured extraction
            novelty_score = self._assess_novelty_from_sources(len(sources), report)

            return {
                "hypothesis": hypothesis,
                "validation_report": report,
                "supporting_sources": sources,
                "novelty_score": novelty_score,
                "is_novel": novelty_score > 0.7,
                "num_related_papers": len(sources),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Hypothesis validation failed: {e}")
            return {
                "hypothesis": hypothesis,
                "error": str(e),
                "novelty_score": 0.5  # Unknown
            }

    async def generate_research_proposal(
        self,
        hypothesis: str,
        background: str,
        domain: str
    ) -> str:
        """
        Generate comprehensive research proposal.

        Args:
            hypothesis: Main hypothesis
            background: Background context
            domain: Research domain

        Returns:
            Structured research proposal
        """
        proposal_query = f"""
        Generate a comprehensive research proposal for:

        Hypothesis: {hypothesis}
        Background: {background}
        Domain: {domain}

        Include:
        1. Literature review and gap analysis
        2. Research objectives
        3. Proposed methodology
        4. Expected outcomes
        5. Potential impact
        6. Required resources
        """

        researcher = GPTResearcher(
            query=proposal_query,
            report_type="research_report"
        )

        try:
            await researcher.conduct_research()
            proposal = await researcher.write_report()
            return proposal

        except Exception as e:
            logger.error(f"Proposal generation failed: {e}")
            return f"Error generating proposal: {e}"

    def _assess_novelty_from_sources(
        self,
        num_sources: int,
        report_text: str
    ) -> float:
        """
        Assess novelty based on literature search results.

        Args:
            num_sources: Number of related sources found
            report_text: Research report text

        Returns:
            Novelty score (0-1)
        """
        # Heuristic: fewer similar sources = more novel
        # In production, use semantic similarity and claim matching

        if num_sources == 0:
            return 0.95  # Highly novel (or poorly researched!)

        if num_sources < 3:
            return 0.85

        if num_sources < 10:
            return 0.7

        if num_sources < 20:
            return 0.6

        # Many similar sources = incremental work
        return 0.4

    async def get_research_trends(
        self,
        domain: str,
        timeframe: str = "recent"
    ) -> Dict[str, Any]:
        """
        Identify research trends in a domain.

        Args:
            domain: Research domain
            timeframe: "recent" (1-2 years) or "current" (current year)

        Returns:
            Research trends and hot topics
        """
        trends_query = f"""
        Identify current research trends and hot topics in {domain}.

        Focus on:
        1. Emerging methodologies
        2. Novel applications
        3. Recent breakthroughs
        4. Open challenges
        5. Future directions

        Timeframe: {timeframe}
        """

        researcher = GPTResearcher(
            query=trends_query,
            report_type="research_report"
        )

        try:
            await researcher.conduct_research()
            report = await researcher.write_report()
            sources = researcher.get_source_urls()

            return {
                "domain": domain,
                "timeframe": timeframe,
                "trends_report": report,
                "sources": sources,
                "num_sources": len(sources),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Trends analysis failed: {e}")
            return {
                "domain": domain,
                "error": str(e)
            }

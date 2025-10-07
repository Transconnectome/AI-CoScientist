"""Paper improvement service for content enhancement."""

import json
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.llm.service import LLMService
from src.models.project import Paper, PaperSection


class PaperImprover:
    """Service for improving academic paper content."""

    def __init__(self, llm_service: LLMService, db: AsyncSession):
        """Initialize improver.

        Args:
            llm_service: LLM service for content improvement
            db: Database session
        """
        self.llm = llm_service
        self.db = db

    async def improve_section(
        self,
        paper_id: UUID,
        section_name: str,
        feedback: Optional[str] = None
    ) -> dict:
        """Improve a specific paper section.

        Args:
            paper_id: Paper UUID
            section_name: Name of section to improve
            feedback: Optional specific feedback to address

        Returns:
            Improvement results with improved content and change summary

        Example:
            {
                "improved_content": "Revised section text...",
                "changes_summary": "Made text more concise, improved clarity...",
                "improvement_score": 8.5
            }
        """
        # Get section
        section = await self._get_section(paper_id, section_name)

        if not section:
            raise ValueError(f"Section '{section_name}' not found in paper {paper_id}")

        # Build improvement prompt
        feedback_text = f"\n\nSpecific feedback to address:\n{feedback}" if feedback else ""

        prompt = f"""
        Improve the following academic paper section.

        Section: {section_name}

        Original content:
        {section.content}
        {feedback_text}

        Improvement guidelines:
        - Enhance clarity and readability
        - Improve logical flow and structure
        - Remove redundancy and unnecessary content
        - Strengthen arguments and evidence
        - Maintain academic tone and rigor
        - Keep approximately the same length unless feedback specifies otherwise

        Return JSON:
        {{
            "improved_content": "<full improved section text>",
            "changes_summary": "<brief summary of key changes made>",
            "improvement_score": <float 0-10, estimated improvement>
        }}

        Return only valid JSON.
        """

        try:
            result = await self.llm.complete(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.4  # Slightly higher for creative improvements
            )

            improvement = json.loads(result.content)

            # Validate structure
            if "improved_content" not in improvement:
                improvement["improved_content"] = section.content
            if "changes_summary" not in improvement:
                improvement["changes_summary"] = "No changes made"
            if "improvement_score" not in improvement:
                improvement["improvement_score"] = 5.0

            return improvement

        except json.JSONDecodeError:
            return {
                "improved_content": section.content,
                "changes_summary": "Unable to generate improvements",
                "improvement_score": 0.0
            }

    async def generate_improvements(self, paper_id: UUID) -> list[dict]:
        """Generate improvement suggestions for all sections.

        Args:
            paper_id: Paper UUID

        Returns:
            List of section-specific improvement suggestions
        """
        # Get paper with sections
        query = select(Paper).where(Paper.id == paper_id)
        result = await self.db.execute(query)
        paper = result.scalar_one_or_none()

        if not paper:
            raise ValueError(f"Paper {paper_id} not found")

        # Get all sections
        sections_query = select(PaperSection).where(
            PaperSection.paper_id == paper_id
        ).order_by(PaperSection.order)

        sections_result = await self.db.execute(sections_query)
        sections = sections_result.scalars().all()

        if not sections:
            return []

        improvements = []

        for section in sections:
            # Generate improvement for each section
            try:
                improvement = await self.improve_section(
                    paper_id,
                    section.name,
                    feedback=None
                )

                improvements.append({
                    "section_name": section.name,
                    "section_order": section.order,
                    **improvement
                })

            except Exception as e:
                improvements.append({
                    "section_name": section.name,
                    "section_order": section.order,
                    "improved_content": section.content,
                    "changes_summary": f"Error generating improvement: {str(e)}",
                    "improvement_score": 0.0
                })

        return improvements

    async def rewrite_for_clarity(
        self,
        paper_id: UUID,
        section_name: str,
        target_length: Optional[str] = None
    ) -> dict:
        """Rewrite section specifically for clarity improvement.

        Args:
            paper_id: Paper UUID
            section_name: Section to rewrite
            target_length: Optional "shorter", "longer", or None to maintain length

        Returns:
            Rewritten section content
        """
        section = await self._get_section(paper_id, section_name)

        if not section:
            raise ValueError(f"Section '{section_name}' not found")

        length_instruction = ""
        if target_length == "shorter":
            length_instruction = "\n- Make the text more concise (reduce by 20-30%)"
        elif target_length == "longer":
            length_instruction = "\n- Expand with more detail and examples (increase by 20-30%)"

        prompt = f"""
        Rewrite this academic paper section for maximum clarity.

        Section: {section_name}

        Original content:
        {section.content}

        Rewriting guidelines:
        - Use clear, direct language
        - Break down complex ideas into simpler components
        - Use appropriate transitions
        - Define technical terms when first used
        - Improve sentence structure and flow
        {length_instruction}

        Return JSON:
        {{
            "rewritten_content": "<clear, improved text>",
            "clarity_improvements": [<list of specific clarity improvements made>]
        }}
        """

        try:
            result = await self.llm.complete(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.4
            )

            return json.loads(result.content)

        except json.JSONDecodeError:
            return {
                "rewritten_content": section.content,
                "clarity_improvements": []
            }

    async def _get_section(
        self,
        paper_id: UUID,
        section_name: str
    ) -> Optional[PaperSection]:
        """Get specific paper section.

        Args:
            paper_id: Paper UUID
            section_name: Section name

        Returns:
            PaperSection or None
        """
        query = select(PaperSection).where(
            PaperSection.paper_id == paper_id,
            PaperSection.name == section_name
        )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

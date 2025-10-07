"""Paper generation service for creating papers from project data."""

import json
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.services.llm.service import LLMService
from src.services.knowledge_base.search import KnowledgeBaseSearch
from src.models.project import Project, Paper, PaperSection, Hypothesis, Experiment, PaperStatus


class PaperGenerator:
    """Service for generating academic papers from research project data."""

    def __init__(
        self,
        llm_service: LLMService,
        knowledge_base: KnowledgeBaseSearch,
        db: AsyncSession
    ):
        """Initialize generator.

        Args:
            llm_service: LLM service for content generation
            knowledge_base: Knowledge base for literature context
            db: Database session
        """
        self.llm = llm_service
        self.kb = knowledge_base
        self.db = db

    async def generate_from_project(
        self,
        project_id: UUID,
        include_hypotheses: bool = True,
        include_experiments: bool = True
    ) -> Paper:
        """Generate paper from project data.

        Creates a complete academic paper draft including:
        - Title and abstract
        - Introduction with literature context
        - Methods from experiment protocols
        - Results from experiment data
        - Discussion synthesizing findings

        Args:
            project_id: UUID of project
            include_hypotheses: Include hypothesis data
            include_experiments: Include experiment data

        Returns:
            Generated Paper object with sections

        Raises:
            ValueError: If project not found or insufficient data
        """
        # Get project with all data
        project = await self._get_project_data(
            project_id,
            include_hypotheses,
            include_experiments
        )

        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Generate paper structure
        structure = await self._generate_structure(project)

        # Generate title
        title = await self._generate_title(project)

        # Generate sections
        abstract_content = await self._generate_abstract(project, structure)
        intro_content = await self._generate_introduction(project, structure)
        methods_content = await self._generate_methods(project, structure)
        results_content = await self._generate_results(project, structure)
        discussion_content = await self._generate_discussion(project, structure)

        # Combine all sections for paper content
        full_content = f"""
# {title}

## Abstract
{abstract_content}

## Introduction
{intro_content}

## Methods
{methods_content}

## Results
{results_content}

## Discussion
{discussion_content}
        """.strip()

        # Create Paper object
        paper = Paper(
            project_id=project_id,
            title=title,
            abstract=abstract_content,
            content=full_content,
            status=PaperStatus.DRAFT,
            version=1
        )

        self.db.add(paper)
        await self.db.flush()

        # Create PaperSection objects
        sections_data = [
            ("abstract", abstract_content, 0),
            ("introduction", intro_content, 1),
            ("methods", methods_content, 2),
            ("results", results_content, 3),
            ("discussion", discussion_content, 4),
        ]

        for name, content, order in sections_data:
            section = PaperSection(
                paper_id=paper.id,
                name=name,
                content=content,
                order=order,
                version=1
            )
            self.db.add(section)

        await self.db.commit()
        await self.db.refresh(paper)

        return paper

    async def _generate_structure(self, project: Project) -> dict:
        """Generate paper structure outline.

        Args:
            project: Project with loaded data

        Returns:
            Dictionary with section structures
        """
        prompt = f"""
        Create a structure outline for an academic paper based on this research project.

        Project: {project.name}
        Research Question: {project.research_question}
        Domain: {project.domain}

        Return a JSON structure with these sections:
        {{
            "abstract": ["key point 1", "key point 2", ...],
            "introduction": ["subsection 1", "subsection 2", ...],
            "methods": ["subsection 1", ...],
            "results": ["subsection 1", ...],
            "discussion": ["subsection 1", ...]
        }}

        Keep it concise and focused on the research question.
        """

        try:
            result = await self.llm.complete(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.3
            )

            return json.loads(result.content)

        except json.JSONDecodeError:
            # Return default structure
            return {
                "abstract": ["Background", "Methods", "Results", "Conclusion"],
                "introduction": ["Background", "Research gap", "Objectives"],
                "methods": ["Design", "Data collection", "Analysis"],
                "results": ["Main findings", "Secondary findings"],
                "discussion": ["Interpretation", "Implications", "Limitations"]
            }

    async def _generate_title(self, project: Project) -> str:
        """Generate paper title from project.

        Args:
            project: Project object

        Returns:
            Generated title string
        """
        prompt = f"""
        Generate an academic paper title for this research project.

        Project name: {project.name}
        Research question: {project.research_question}
        Domain: {project.domain}

        Requirements:
        - Concise (10-15 words)
        - Descriptive and specific
        - Academic tone
        - Include key concepts

        Return only the title, no additional text.
        """

        result = await self.llm.complete(
            prompt=prompt,
            max_tokens=100,
            temperature=0.4
        )

        return result.content.strip().strip('"')

    async def _generate_abstract(self, project: Project, structure: dict) -> str:
        """Generate abstract section.

        Args:
            project: Project with data
            structure: Paper structure outline

        Returns:
            Abstract text
        """
        # Collect hypotheses if available
        hypotheses_text = ""
        if hasattr(project, 'hypotheses') and project.hypotheses:
            hypotheses_text = "\n".join([
                f"- {h.content}" for h in project.hypotheses[:3]
            ])

        prompt = f"""
        Write an academic abstract for this research.

        Research Question: {project.research_question}
        Domain: {project.domain}

        Hypotheses:
        {hypotheses_text if hypotheses_text else "Not specified"}

        Structure points: {structure.get('abstract', [])}

        Requirements:
        - 150-250 words
        - Background, methods, results, conclusion structure
        - Clear and concise
        - Academic tone

        Return only the abstract text.
        """

        result = await self.llm.complete(
            prompt=prompt,
            max_tokens=500,
            temperature=0.4
        )

        return result.content.strip()

    async def _generate_introduction(self, project: Project, structure: dict) -> str:
        """Generate introduction section.

        Args:
            project: Project with data
            structure: Paper structure outline

        Returns:
            Introduction text
        """
        # Get relevant literature context
        literature_context = ""
        if project.research_question:
            try:
                papers = await self.kb.search(
                    query=project.research_question,
                    n_results=5
                )
                if papers:
                    literature_context = "\n\n".join([
                        f"- {p.get('title', 'Unknown')}: {p.get('abstract', '')[:200]}..."
                        for p in papers[:3]
                    ])
            except Exception:
                literature_context = ""

        prompt = f"""
        Write the Introduction section for this academic paper.

        Research Question: {project.research_question}
        Domain: {project.domain}
        Description: {project.description or "Not provided"}

        Related literature:
        {literature_context if literature_context else "Limited literature available"}

        Structure: {structure.get('introduction', [])}

        Requirements:
        - Provide background and context
        - Identify research gap
        - State objectives clearly
        - 3-4 paragraphs
        - Academic tone with proper flow

        Return only the introduction text.
        """

        result = await self.llm.complete(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.5
        )

        return result.content.strip()

    async def _generate_methods(self, project: Project, structure: dict) -> str:
        """Generate methods section.

        Args:
            project: Project with experiments
            structure: Paper structure outline

        Returns:
            Methods text
        """
        # Collect experiment protocols
        methods_data = ""
        if hasattr(project, 'hypotheses') and project.hypotheses:
            for hyp in project.hypotheses:
                if hasattr(hyp, 'experiments') and hyp.experiments:
                    for exp in hyp.experiments:
                        if exp.protocol:
                            methods_data += f"\n\nExperiment: {exp.title}\n{exp.protocol[:500]}..."
                            methods_data += f"\nSample size: {exp.sample_size}, Power: {exp.power}"

        if not methods_data:
            methods_data = "Experimental protocols to be determined based on research design."

        prompt = f"""
        Write the Methods section for this academic paper.

        Research Question: {project.research_question}

        Experimental details:
        {methods_data}

        Structure: {structure.get('methods', [])}

        Requirements:
        - Describe experimental design
        - Detail data collection procedures
        - Explain analysis methods
        - Include statistical approaches
        - 2-3 paragraphs
        - Sufficient detail for replication

        Return only the methods text.
        """

        result = await self.llm.complete(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.4
        )

        return result.content.strip()

    async def _generate_results(self, project: Project, structure: dict) -> str:
        """Generate results section.

        Args:
            project: Project with experiments
            structure: Paper structure outline

        Returns:
            Results text
        """
        # Collect experiment results
        results_data = ""
        if hasattr(project, 'hypotheses') and project.hypotheses:
            for hyp in project.hypotheses:
                if hasattr(hyp, 'experiments') and hyp.experiments:
                    for exp in hyp.experiments:
                        if exp.results_summary:
                            results_data += f"\n\n{exp.title}:\n{exp.results_summary}"
                        if exp.statistical_results:
                            results_data += f"\nStatistics: {exp.statistical_results[:200]}..."

        if not results_data:
            results_data = "Results will be presented based on experimental outcomes."

        prompt = f"""
        Write the Results section for this academic paper.

        Research Question: {project.research_question}

        Experimental results:
        {results_data}

        Structure: {structure.get('results', [])}

        Requirements:
        - Present findings clearly
        - Include key statistics
        - Organize logically
        - Objective tone (no interpretation)
        - 2-3 paragraphs

        Return only the results text.
        """

        result = await self.llm.complete(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.4
        )

        return result.content.strip()

    async def _generate_discussion(self, project: Project, structure: dict) -> str:
        """Generate discussion section.

        Args:
            project: Project with data
            structure: Paper structure outline

        Returns:
            Discussion text
        """
        prompt = f"""
        Write the Discussion section for this academic paper.

        Research Question: {project.research_question}
        Domain: {project.domain}

        Structure: {structure.get('discussion', [])}

        Requirements:
        - Interpret main findings
        - Relate to research question
        - Discuss implications
        - Note limitations
        - Suggest future research
        - 3-4 paragraphs
        - Balanced and thoughtful

        Return only the discussion text.
        """

        result = await self.llm.complete(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.5
        )

        return result.content.strip()

    async def _get_project_data(
        self,
        project_id: UUID,
        include_hypotheses: bool,
        include_experiments: bool
    ) -> Optional[Project]:
        """Get project with all related data.

        Args:
            project_id: Project UUID
            include_hypotheses: Load hypotheses
            include_experiments: Load experiments (via hypotheses)

        Returns:
            Project with loaded relationships or None
        """
        query = select(Project).where(Project.id == project_id)

        if include_hypotheses:
            query = query.options(selectinload(Project.hypotheses))

            if include_experiments:
                query = query.options(
                    selectinload(Project.hypotheses).selectinload(Hypothesis.experiments)
                )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

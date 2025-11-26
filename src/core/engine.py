from typing import Optional, List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.project import Project, ProjectStatus
from src.services.hypothesis.generator import HypothesisGenerator
from src.services.experiment.design import ExperimentDesigner
from src.services.paper.generator import PaperGenerator

class CoScientistEngine:
    """Unified orchestrator for the AI-CoScientist research lifecycle."""

    def __init__(
        self,
        hypothesis_generator: HypothesisGenerator,
        experiment_designer: ExperimentDesigner,
        paper_generator: PaperGenerator,
        db: AsyncSession
    ):
        self.hypothesis_generator = hypothesis_generator
        self.experiment_designer = experiment_designer
        self.paper_generator = paper_generator
        self.db = db

    async def start_project(self, topic: str, description: Optional[str] = None) -> Project:
        """Initialize a new research project."""
        project = Project(
            research_question=topic,
            description=description,
            domain="Scientific Research", # Default, could be inferred
            status=ProjectStatus.ACTIVE
        )
        self.db.add(project)
        await self.db.commit()
        await self.db.refresh(project)
        return project

    async def run_research_phase(self, project_id: UUID) -> None:
        """Execute the research phase (hypothesis generation)."""
        project = await self._get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        await self.hypothesis_generator.generate_hypotheses(
            project_id=project_id,
            research_question=project.research_question,
            num_hypotheses=3
        )

        project.status = ProjectStatus.ACTIVE
        await self.db.commit()

    async def run_experiment_phase(self, project_id: UUID) -> None:
        """Execute the experiment phase (design and simulation)."""
        # Load project with hypotheses
        query = select(Project).where(Project.id == project_id).options(selectinload(Project.hypotheses))
        result = await self.db.execute(query)
        project = result.scalar_one_or_none()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")

        for hypothesis in project.hypotheses:
            await self.experiment_designer.design_experiment(
                hypothesis_id=hypothesis.id,
                research_question=project.research_question,
                hypothesis_content=hypothesis.content
            )
            # In a real system, we would also RUN the experiment here.
            # For now, we assume design implies readiness or simulation.

        project.status = ProjectStatus.ACTIVE
        await self.db.commit()

    async def run_paper_phase(self, project_id: UUID) -> None:
        """Execute the paper generation phase."""
        project = await self._get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        await self.paper_generator.generate_from_project(project_id=project_id)

        project.status = ProjectStatus.ACTIVE
        await self.db.commit()

    async def run_discovery_loop(self, topic: str) -> Project:
        """Run the full scientific discovery loop from idea to paper."""
        # 1. Start Project
        project = await self.start_project(topic)
        
        # 2. Research Phase
        await self.run_research_phase(project.id)
        
        # 3. Experiment Phase
        await self.run_experiment_phase(project.id)
        
        # 4. Paper Phase
        await self.run_paper_phase(project.id)
        
        return project

    async def _get_project(self, project_id: UUID) -> Optional[Project]:
        """Helper to get project by ID."""
        result = await self.db.execute(select(Project).where(Project.id == project_id))
        return result.scalar_one_or_none()

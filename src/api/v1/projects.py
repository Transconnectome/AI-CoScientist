"""Project management endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.database import get_db
from src.models.project import Project, Hypothesis, Experiment, Paper
from src.schemas.project import (
    Project as ProjectSchema,
    ProjectCreate,
    ProjectUpdate,
    ProjectList,
    ProjectStats,
    Hypothesis as HypothesisSchema,
    HypothesisCreate,
    Experiment as ExperimentSchema,
    ExperimentCreate,
    Paper as PaperSchema,
    PaperCreate
)
from src.services.paper import PaperGenerator
from src.services.llm.service import LLMService
from src.services.knowledge_base.search import KnowledgeBaseSearch
from src.core.config import settings

router = APIRouter()


@router.get("", response_model=ProjectList)
async def list_projects(
    status: str | None = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
) -> ProjectList:
    """List all projects with pagination."""
    # Build query
    query = select(Project)

    if status:
        query = query.where(Project.status == status)

    # Get total count
    count_query = select(func.count()).select_from(Project)
    if status:
        count_query = count_query.where(Project.status == status)
    total = await db.scalar(count_query) or 0

    # Apply pagination
    query = query.offset((page - 1) * limit).limit(limit)
    query = query.order_by(Project.created_at.desc())

    result = await db.execute(query)
    projects = result.scalars().all()

    # Calculate stats for each project
    projects_with_stats = []
    for project in projects:
        # Get counts
        hyp_count = await db.scalar(
            select(func.count()).where(Hypothesis.project_id == project.id)
        ) or 0
        exp_count = await db.scalar(
            select(func.count())
            .select_from(Experiment)
            .join(Hypothesis)
            .where(Hypothesis.project_id == project.id)
        ) or 0
        paper_count = await db.scalar(
            select(func.count()).where(Paper.project_id == project.id)
        ) or 0

        project_dict = ProjectSchema.model_validate(project).model_dump()
        project_dict["stats"] = ProjectStats(
            hypotheses_count=hyp_count,
            experiments_count=exp_count,
            papers_count=paper_count
        )
        projects_with_stats.append(ProjectSchema(**project_dict))

    return ProjectList(
        projects=projects_with_stats,
        total=total,
        page=page,
        limit=limit,
        pages=(total + limit - 1) // limit
    )


@router.post("", response_model=ProjectSchema, status_code=201)
async def create_project(
    project_data: ProjectCreate,
    db: AsyncSession = Depends(get_db)
) -> ProjectSchema:
    """Create a new project."""
    project = Project(**project_data.model_dump())
    db.add(project)
    await db.commit()
    await db.refresh(project)

    return ProjectSchema.model_validate(project)


@router.get("/{project_id}", response_model=ProjectSchema)
async def get_project(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> ProjectSchema:
    """Get project by ID."""
    query = select(Project).where(Project.id == project_id)
    query = query.options(
        selectinload(Project.hypotheses),
        selectinload(Project.papers)
    )

    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get stats
    hyp_count = len(project.hypotheses)

    exp_count = await db.scalar(
        select(func.count())
        .select_from(Experiment)
        .join(Hypothesis)
        .where(Hypothesis.project_id == project.id)
    ) or 0

    paper_count = len(project.papers)

    project_dict = ProjectSchema.model_validate(project).model_dump()
    project_dict["stats"] = ProjectStats(
        hypotheses_count=hyp_count,
        experiments_count=exp_count,
        papers_count=paper_count
    )

    return ProjectSchema(**project_dict)


@router.patch("/{project_id}", response_model=ProjectSchema)
async def update_project(
    project_id: UUID,
    project_data: ProjectUpdate,
    db: AsyncSession = Depends(get_db)
) -> ProjectSchema:
    """Update project."""
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Update fields
    update_data = project_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)

    await db.commit()
    await db.refresh(project)

    return ProjectSchema.model_validate(project)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> None:
    """Delete project."""
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    await db.delete(project)
    await db.commit()


# Hypothesis endpoints
@router.get("/{project_id}/hypotheses", response_model=List[HypothesisSchema])
async def list_hypotheses(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> List[HypothesisSchema]:
    """List all hypotheses for a project."""
    query = select(Hypothesis).where(Hypothesis.project_id == project_id)
    query = query.order_by(Hypothesis.created_at.desc())

    result = await db.execute(query)
    hypotheses = result.scalars().all()

    return [HypothesisSchema.model_validate(h) for h in hypotheses]


@router.post("/{project_id}/hypotheses", response_model=HypothesisSchema, status_code=201)
async def create_hypothesis(
    project_id: UUID,
    hypothesis_data: HypothesisCreate,
    db: AsyncSession = Depends(get_db)
) -> HypothesisSchema:
    """Create a new hypothesis."""
    # Verify project exists
    project_query = select(Project).where(Project.id == project_id)
    project_result = await db.execute(project_query)
    if not project_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    hypothesis = Hypothesis(**hypothesis_data.model_dump())
    hypothesis.project_id = project_id

    db.add(hypothesis)
    await db.commit()
    await db.refresh(hypothesis)

    return HypothesisSchema.model_validate(hypothesis)


# Experiment endpoints
@router.post("/hypotheses/{hypothesis_id}/experiments", response_model=ExperimentSchema, status_code=201)
async def create_experiment(
    hypothesis_id: UUID,
    experiment_data: ExperimentCreate,
    db: AsyncSession = Depends(get_db)
) -> ExperimentSchema:
    """Create a new experiment."""
    # Verify hypothesis exists
    hyp_query = select(Hypothesis).where(Hypothesis.id == hypothesis_id)
    hyp_result = await db.execute(hyp_query)
    if not hyp_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    experiment = Experiment(**experiment_data.model_dump())
    experiment.hypothesis_id = hypothesis_id

    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)

    return ExperimentSchema.model_validate(experiment)


# Paper endpoints
@router.post("/{project_id}/papers", response_model=PaperSchema, status_code=201)
async def create_paper(
    project_id: UUID,
    paper_data: PaperCreate,
    db: AsyncSession = Depends(get_db)
) -> PaperSchema:
    """Create a new paper."""
    # Verify project exists
    project_query = select(Project).where(Project.id == project_id)
    project_result = await db.execute(project_query)
    if not project_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    paper = Paper(**paper_data.model_dump())
    paper.project_id = project_id

    db.add(paper)
    await db.commit()
    await db.refresh(paper)

    return PaperSchema.model_validate(paper)


@router.post("/{project_id}/papers/generate", response_model=PaperSchema, status_code=201)
async def generate_paper_from_project(
    project_id: UUID,
    include_hypotheses: bool = Query(True, description="Include hypotheses in paper"),
    include_experiments: bool = Query(True, description="Include experiments in paper"),
    db: AsyncSession = Depends(get_db)
) -> PaperSchema:
    """Generate paper from project data.

    Creates a complete academic paper draft from:
    - Research question and objectives
    - Generated hypotheses
    - Experiment protocols and results
    - Relevant literature context
    """
    # Initialize services
    llm_service = LLMService(
        primary_provider=settings.llm_primary_provider,
        fallback_provider=settings.llm_fallback_provider
    )
    knowledge_base = KnowledgeBaseSearch()

    # Generate paper
    generator = PaperGenerator(llm_service, knowledge_base, db)

    try:
        paper = await generator.generate_from_project(
            project_id,
            include_hypotheses=include_hypotheses,
            include_experiments=include_experiments
        )

        return PaperSchema.model_validate(paper)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

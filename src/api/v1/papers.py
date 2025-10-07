"""Paper editing and improvement endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.models.project import Paper, PaperSection
from src.schemas.paper import (
    PaperAnalyzeRequest,
    PaperAnalysisResponse,
    PaperImproveRequest,
    PaperImprovementResponse,
    MultipleSectionImprovementsResponse,
    PaperSectionSchema,
    SectionUpdateRequest,
    CoherenceCheckResponse,
    GapAnalysisResponse,
)
from src.services.paper import PaperParser, PaperAnalyzer, PaperImprover, PaperExporter
from src.services.llm.service import LLMService
from src.core.config import settings

router = APIRouter()


def get_llm_service() -> LLMService:
    """Get LLM service instance."""
    return LLMService(
        primary_provider=settings.llm_primary_provider,
        fallback_provider=settings.llm_fallback_provider
    )


@router.post("/{paper_id}/parse", response_model=List[PaperSectionSchema])
async def parse_paper(
    paper_id: UUID,
    db: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Parse paper text into structured sections.

    Extracts and identifies standard academic paper sections
    (Abstract, Introduction, Methods, Results, Discussion, etc.)
    from the paper content.

    Args:
        paper_id: UUID of the paper to parse

    Returns:
        List of parsed sections with order

    Raises:
        HTTPException: 404 if paper not found or no content
    """
    # Get paper
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id)
    )
    paper = result.scalar_one_or_none()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if not paper.content:
        raise HTTPException(status_code=400, detail="Paper has no content to parse")

    # Parse using LLM
    parser = PaperParser(llm_service)
    sections_data = await parser.extract_sections(paper.content)

    # Save sections to database
    sections = []
    for data in sections_data:
        section = PaperSection(
            paper_id=paper_id,
            name=data["name"],
            content=data["content"],
            order=data["order"]
        )
        db.add(section)
        sections.append(section)

    await db.commit()

    # Refresh to get IDs
    for section in sections:
        await db.refresh(section)

    return [PaperSectionSchema.model_validate(s) for s in sections]


@router.post("/{paper_id}/analyze", response_model=PaperAnalysisResponse)
async def analyze_paper(
    paper_id: UUID,
    request: PaperAnalyzeRequest = PaperAnalyzeRequest(),
    db: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Analyze paper quality and provide feedback.

    Provides comprehensive quality assessment including:
    - Overall quality score
    - Strengths and weaknesses
    - Section-specific improvement suggestions
    - Coherence and clarity scores

    Args:
        paper_id: UUID of paper to analyze
        request: Analysis configuration options

    Returns:
        Detailed analysis results

    Raises:
        HTTPException: 404 if paper not found
    """
    # Check paper exists
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id)
    )
    paper = result.scalar_one_or_none()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Perform analysis
    analyzer = PaperAnalyzer(llm_service, db)
    analysis = await analyzer.analyze_quality(paper_id)

    return PaperAnalysisResponse(**analysis)


@router.post("/{paper_id}/improve", response_model=PaperImprovementResponse | MultipleSectionImprovementsResponse)
async def improve_paper(
    paper_id: UUID,
    request: PaperImproveRequest,
    db: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Generate improvement suggestions for paper.

    Can improve a specific section or generate improvements for all sections.

    Args:
        paper_id: UUID of paper to improve
        request: Improvement configuration (section, feedback)

    Returns:
        Improvement suggestions and improved content

    Raises:
        HTTPException: 404 if paper or section not found
    """
    # Check paper exists
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id)
    )
    paper = result.scalar_one_or_none()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    improver = PaperImprover(llm_service, db)

    if request.section_name:
        # Improve specific section
        try:
            improvement = await improver.improve_section(
                paper_id,
                request.section_name,
                request.feedback
            )
            return PaperImprovementResponse(
                section_name=request.section_name,
                **improvement
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        # Improve all sections
        improvements = await improver.generate_improvements(paper_id)
        return MultipleSectionImprovementsResponse(
            improvements=improvements,
            total_sections=len(improvements)
        )


@router.patch("/{paper_id}/sections/{section_name}", response_model=PaperSectionSchema)
async def update_section(
    paper_id: UUID,
    section_name: str,
    request: SectionUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Update a specific paper section.

    Updates section content and increments version number.

    Args:
        paper_id: UUID of paper
        section_name: Name of section to update
        request: New section content

    Returns:
        Updated section with new version

    Raises:
        HTTPException: 404 if section not found
    """
    # Get section
    query = select(PaperSection).where(
        PaperSection.paper_id == paper_id,
        PaperSection.name == section_name
    )
    result = await db.execute(query)
    section = result.scalar_one_or_none()

    if not section:
        raise HTTPException(
            status_code=404,
            detail=f"Section '{section_name}' not found in paper {paper_id}"
        )

    # Update section
    section.content = request.content
    section.version += 1

    await db.commit()
    await db.refresh(section)

    return PaperSectionSchema.model_validate(section)


@router.get("/{paper_id}/sections", response_model=List[PaperSectionSchema])
async def list_sections(
    paper_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """List all sections of a paper.

    Args:
        paper_id: UUID of paper

    Returns:
        List of paper sections in order

    Raises:
        HTTPException: 404 if paper not found
    """
    # Check paper exists
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id)
    )
    paper = result.scalar_one_or_none()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Get sections
    query = select(PaperSection).where(
        PaperSection.paper_id == paper_id
    ).order_by(PaperSection.order)

    result = await db.execute(query)
    sections = result.scalars().all()

    return [PaperSectionSchema.model_validate(s) for s in sections]


@router.post("/{paper_id}/coherence", response_model=CoherenceCheckResponse)
async def check_coherence(
    paper_id: UUID,
    db: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Check coherence between paper sections.

    Analyzes logical flow and consistency across sections.

    Args:
        paper_id: UUID of paper

    Returns:
        Coherence analysis with score and recommendations

    Raises:
        HTTPException: 404 if paper not found
    """
    # Check paper exists
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id)
    )
    paper = result.scalar_one_or_none()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    analyzer = PaperAnalyzer(llm_service, db)
    coherence = await analyzer.check_section_coherence(paper_id)

    return CoherenceCheckResponse(**coherence)


@router.post("/{paper_id}/gaps", response_model=GapAnalysisResponse)
async def identify_gaps(
    paper_id: UUID,
    db: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Identify missing or underdeveloped content.

    Args:
        paper_id: UUID of paper

    Returns:
        List of identified gaps with recommendations

    Raises:
        HTTPException: 404 if paper not found
    """
    # Check paper exists
    result = await db.execute(
        select(Paper).where(Paper.id == paper_id)
    )
    paper = result.scalar_one_or_none()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    analyzer = PaperAnalyzer(llm_service, db)
    gaps = await analyzer.identify_gaps(paper_id)

    return GapAnalysisResponse(
        gaps=gaps,
        total_gaps=len(gaps)
    )


@router.get("/{paper_id}/export/word")
async def export_paper_to_word(
    paper_id: UUID,
    include_metadata: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Export paper to Microsoft Word format (.docx).

    Creates a professionally formatted Word document with:
    - Title page (if include_metadata=True)
    - Abstract
    - All sections in order
    - Proper formatting (Times New Roman, 12pt, 1.5 line spacing)

    Args:
        paper_id: Paper ID to export
        include_metadata: Include title page with version/status info
        db: Database session

    Returns:
        File download response with .docx document

    Raises:
        HTTPException: 404 if paper not found or has no content
    """
    from fastapi.responses import FileResponse
    import os

    exporter = PaperExporter(db)

    try:
        # Export to Word
        output_path = await exporter.export_to_word(
            paper_id,
            include_metadata=include_metadata
        )

        # Return file
        return FileResponse(
            path=output_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=os.path.basename(output_path),
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{paper_id}/export/word-improved")
async def export_improved_paper_to_word(
    paper_id: UUID,
    improvements: dict,
    db: AsyncSession = Depends(get_db)
):
    """Export paper with AI improvements to Word format.

    Creates a Word document with improved content and notes
    indicating which sections were enhanced by AI.

    Args:
        paper_id: Paper ID to export
        improvements: Dict mapping section names to improved content
        db: Database session

    Returns:
        File download response with improved .docx document

    Raises:
        HTTPException: 404 if paper not found
    """
    from fastapi.responses import FileResponse
    import os

    exporter = PaperExporter(db)

    try:
        # Export with improvements
        output_path = await exporter.export_improved_paper(
            paper_id,
            improvements
        )

        return FileResponse(
            path=output_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=os.path.basename(output_path),
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

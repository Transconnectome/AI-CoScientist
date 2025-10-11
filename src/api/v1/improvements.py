"""Phase 4: Improvement API endpoints for intelligent paper enhancement."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from src.core.database import get_db
from src.schemas.improvement import (
    ApplyImprovementRequest,
    ApplyImprovementResponse,
    VersionRollbackRequest,
    VersionComparisonResponse,
    VersionHistoryResponse,
    SmartSuggestionResponse,
    IterativeImprovementRequest,
    IterativeImprovementResponse,
    AnalyticsDashboardResponse,
)
from src.services.paper.improvement_service import ImprovementService

router = APIRouter()


@router.post("/{paper_id}/apply", response_model=ApplyImprovementResponse)
async def apply_improvement(
    paper_id: UUID,
    request: ApplyImprovementRequest,
    db: AsyncSession = Depends(get_db),
):
    """One-click application of AI improvement to paper section.

    Creates new version, applies improvement, updates ChromaDB patterns.
    Rollback available through version system.

    Args:
        paper_id: Paper UUID
        request: Improvement details (section, content, metadata)
        db: Database session

    Returns:
        Improvement application results with new version

    Raises:
        HTTPException: 404 if paper/section not found, 400 on validation error
    """
    service = ImprovementService(db)

    try:
        result = await service.apply_improvement(
            paper_id=paper_id,
            section_name=request.section_name,
            improved_content=request.improved_content,
            improvement_metadata=request.metadata,
        )
        return ApplyImprovementResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to apply improvement: {str(e)}")


@router.post("/{paper_id}/versions/{version}/rollback")
async def rollback_to_version(
    paper_id: UUID,
    version: str,
    request: VersionRollbackRequest = VersionRollbackRequest(target_version=""),
    db: AsyncSession = Depends(get_db),
):
    """Rollback paper to a previous version.

    Creates new version (doesn't delete history) with content
    from specified previous version.

    Args:
        paper_id: Paper UUID
        version: Target version string (e.g., "1.2.0")
        request: Rollback configuration
        db: Database session

    Returns:
        Rollback status and new version info

    Raises:
        HTTPException: 404 if paper/version not found, 400 on error
    """
    service = ImprovementService(db)

    # Use path parameter version if request doesn't specify
    target_version = request.target_version or version

    try:
        result = await service.rollback_to_version(
            paper_id=paper_id,
            target_version=target_version,
            create_backup=request.create_backup,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Rollback failed: {str(e)}")


@router.get("/{paper_id}/versions/compare", response_model=VersionComparisonResponse)
async def compare_versions(
    paper_id: UUID,
    version_a: str,
    version_b: str,
    db: AsyncSession = Depends(get_db),
):
    """Compare two versions of a paper with diff visualization.

    Returns:
    - Side-by-side content comparison
    - Quality score changes
    - Improvement summary
    - Diff visualization data

    Args:
        paper_id: Paper UUID
        version_a: First version (e.g., "1.0.0")
        version_b: Second version (e.g., "1.2.0")
        db: Database session

    Returns:
        Version comparison with diffs

    Raises:
        HTTPException: 404 if paper/versions not found
    """
    service = ImprovementService(db)

    try:
        comparison = await service.compare_versions(
            paper_id=paper_id, version_a=version_a, version_b=version_b
        )
        return VersionComparisonResponse(**comparison)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")


@router.get("/{paper_id}/versions", response_model=VersionHistoryResponse)
async def get_version_history(
    paper_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get complete version history for a paper.

    Returns:
    - All versions with timestamps
    - Quality scores per version
    - Change summaries
    - Current version indicator

    Args:
        paper_id: Paper UUID
        db: Database session

    Returns:
        Complete version history

    Raises:
        HTTPException: 404 if paper not found
    """
    service = ImprovementService(db)

    try:
        history = await service.get_version_history(paper_id)
        return VersionHistoryResponse(**history)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get history: {str(e)}")


@router.get("/{paper_id}/suggestions/smart", response_model=SmartSuggestionResponse)
async def get_smart_suggestions(
    paper_id: UUID,
    section_name: str = None,
    db: AsyncSession = Depends(get_db),
):
    """Get RAG-powered smart suggestions using historical patterns.

    Uses ChromaDB to find similar successful improvements
    and generate contextual suggestions.

    Args:
        paper_id: Paper UUID
        section_name: Optional specific section to improve
        db: Database session

    Returns:
        Smart suggestions with RAG metadata

    Raises:
        HTTPException: 404 if paper not found
    """
    service = ImprovementService(db)

    try:
        suggestions = await service.generate_smart_suggestions(
            paper_id=paper_id, section_name=section_name
        )
        return SmartSuggestionResponse(**suggestions)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to generate suggestions: {str(e)}"
        )


@router.post("/{paper_id}/iterate", response_model=IterativeImprovementResponse)
async def start_iterative_improvement(
    paper_id: UUID,
    request: IterativeImprovementRequest,
    db: AsyncSession = Depends(get_db),
):
    """Start iterative improvement session with target quality score.

    Runs multiple rounds of analysis → improvement → application
    until target score reached or max iterations hit.

    Args:
        paper_id: Paper UUID
        request: Iteration configuration (target_score, max_iterations, focus_areas)
        db: Database session

    Returns:
        Iteration results with final metrics

    Raises:
        HTTPException: 404 if paper not found, 400 on error
    """
    service = ImprovementService(db)

    try:
        result = await service.run_iterative_improvement(
            paper_id=paper_id,
            target_score=request.target_score,
            max_iterations=request.max_iterations,
            focus_areas=request.focus_areas,
        )
        return IterativeImprovementResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Iterative improvement failed: {str(e)}"
        )


@router.get("/{paper_id}/analytics", response_model=AnalyticsDashboardResponse)
async def get_analytics_dashboard(
    paper_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get comprehensive analytics dashboard for a paper.

    Returns:
    - Quality progression metrics
    - Section improvement statistics
    - Improvement type effectiveness
    - Iteration efficiency metrics
    - Version timeline
    - Learning system statistics
    - Intelligent recommendations

    Args:
        paper_id: Paper UUID
        db: Database session

    Returns:
        Complete analytics dashboard data

    Raises:
        HTTPException: 404 if paper not found, 400 on error
    """
    service = ImprovementService(db)

    try:
        analytics = await service.get_analytics(paper_id)
        return AnalyticsDashboardResponse(**analytics)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate analytics: {str(e)}")

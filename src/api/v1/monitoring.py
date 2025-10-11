"""Literature monitoring API endpoints.

REST API for managing literature sources, alerts, and triggering syncs.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from src.core.database import get_db
from src.schemas.monitoring import (
    LiteratureSourceCreate,
    LiteratureSourceResponse,
    LiteratureSourceUpdate,
    MonitoringAlertCreate,
    MonitoringAlertResponse,
    MonitoringAlertUpdate,
    SourceStatisticsResponse,
    SyncTriggerResponse,
)
from src.services.monitoring.orchestrator import MonitoringOrchestrator

logger = get_logger(__name__)

router = APIRouter()


@router.post("/sources", response_model=LiteratureSourceResponse, status_code=201)
async def create_literature_source(
    source: LiteratureSourceCreate,
    db: AsyncSession = Depends(get_db),
) -> LiteratureSourceResponse:
    """Create a new literature monitoring source.

    **ArXiv Source Example**:
    ```json
    {
        "source_type": "arxiv",
        "category": "cs.AI,cs.LG,q-bio.NC",
        "sync_frequency": "daily",
        "status": "active"
    }
    ```

    **PubMed Source Example**:
    ```json
    {
        "source_type": "pubmed",
        "query": "neuroscience[MeSH] AND machine learning",
        "sync_frequency": "weekly",
        "status": "active"
    }
    ```
    """
    orchestrator = MonitoringOrchestrator(db)

    try:
        if source.source_type == "arxiv":
            if not source.category:
                raise HTTPException(
                    status_code=400,
                    detail="ArXiv source requires 'category' field",
                )
            categories = [c.strip() for c in source.category.split(",")]
            result = await orchestrator.create_arxiv_source(
                categories=categories,
                sync_frequency=source.sync_frequency,
                status=source.status,
            )

        elif source.source_type == "pubmed":
            if not source.query:
                raise HTTPException(
                    status_code=400,
                    detail="PubMed source requires 'query' field",
                )
            result = await orchestrator.create_pubmed_source(
                query=source.query,
                sync_frequency=source.sync_frequency,
                status=source.status,
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source_type: {source.source_type}",
            )

        logger.info("literature_source_created", source_id=result.id, source_type=source.source_type)
        return LiteratureSourceResponse.model_validate(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("create_source_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources", response_model=List[LiteratureSourceResponse])
async def list_literature_sources(
    source_type: Optional[str] = Query(None, description="Filter by source type (arxiv, pubmed)"),
    status: Optional[str] = Query(None, description="Filter by status (active, paused, disabled)"),
    db: AsyncSession = Depends(get_db),
) -> List[LiteratureSourceResponse]:
    """List all literature monitoring sources with optional filtering."""
    orchestrator = MonitoringOrchestrator(db)

    try:
        sources = await orchestrator.list_sources(source_type=source_type, status=status)
        return [LiteratureSourceResponse.model_validate(s) for s in sources]

    except Exception as e:
        logger.error("list_sources_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_id}", response_model=LiteratureSourceResponse)
async def get_literature_source(
    source_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> LiteratureSourceResponse:
    """Get a specific literature monitoring source."""
    orchestrator = MonitoringOrchestrator(db)

    try:
        source = await orchestrator.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        return LiteratureSourceResponse.model_validate(source)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_source_error", source_id=source_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sources/{source_id}", response_model=LiteratureSourceResponse)
async def update_literature_source(
    source_id: UUID,
    update: LiteratureSourceUpdate,
    db: AsyncSession = Depends(get_db),
) -> LiteratureSourceResponse:
    """Update a literature monitoring source.

    Can update configuration, frequency, or status.
    """
    orchestrator = MonitoringOrchestrator(db)

    try:
        # Get existing source
        source = await orchestrator.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Update fields
        if update.category is not None:
            source.category = update.category
        if update.query is not None:
            source.query = update.query
        if update.sync_frequency is not None:
            source.sync_frequency = update.sync_frequency
        if update.status is not None:
            source.status = update.status

        await db.commit()
        await db.refresh(source)

        logger.info("source_updated", source_id=source_id)
        return LiteratureSourceResponse.model_validate(source)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_source_error", source_id=source_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sources/{source_id}/sync", response_model=SyncTriggerResponse)
async def trigger_source_sync(
    source_id: UUID,
    api_key: Optional[str] = Query(None, description="NCBI API key (for PubMed sources)"),
    db: AsyncSession = Depends(get_db),
) -> SyncTriggerResponse:
    """Trigger a manual sync for a specific source.

    Returns immediately with task ID. Check task status using Celery tools.
    """
    orchestrator = MonitoringOrchestrator(db)

    try:
        result = await orchestrator.trigger_manual_sync(source_id, api_key=api_key)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        logger.info("manual_sync_triggered", source_id=source_id, task_id=result["task_id"])
        return SyncTriggerResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("trigger_sync_error", source_id=source_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/all", response_model=SyncTriggerResponse)
async def trigger_full_sync(
    db: AsyncSession = Depends(get_db),
) -> SyncTriggerResponse:
    """Trigger a full sync of all active sources.

    Dispatches individual sync tasks for each active source.
    Returns immediately with task ID.
    """
    orchestrator = MonitoringOrchestrator(db)

    try:
        result = await orchestrator.trigger_full_sync()
        logger.info("full_sync_triggered", task_id=result["task_id"])
        return SyncTriggerResponse(**result)

    except Exception as e:
        logger.error("trigger_full_sync_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_id}/statistics", response_model=SourceStatisticsResponse)
async def get_source_statistics(
    source_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> SourceStatisticsResponse:
    """Get statistics and status for a literature source."""
    orchestrator = MonitoringOrchestrator(db)

    try:
        stats = await orchestrator.get_source_statistics(source_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Source not found")

        return SourceStatisticsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_statistics_error", source_id=source_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts", response_model=MonitoringAlertResponse, status_code=201)
async def create_monitoring_alert(
    alert: MonitoringAlertCreate,
    db: AsyncSession = Depends(get_db),
) -> MonitoringAlertResponse:
    """Create a new monitoring alert.

    **Example**:
    ```json
    {
        "topic": "AI in Neuroscience",
        "keywords": ["machine learning", "brain imaging", "fMRI"],
        "frequency": "weekly"
    }
    ```
    """
    orchestrator = MonitoringOrchestrator(db)

    try:
        result = await orchestrator.create_alert(
            topic=alert.topic,
            keywords=alert.keywords,
            frequency=alert.frequency,
            user_id=alert.user_id,
        )

        logger.info("alert_created", alert_id=result.id, topic=alert.topic)
        return MonitoringAlertResponse.model_validate(result)

    except Exception as e:
        logger.error("create_alert_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[MonitoringAlertResponse])
async def list_monitoring_alerts(
    user_id: Optional[UUID] = Query(None, description="Filter by user ID"),
    active_only: bool = Query(True, description="Only return active alerts"),
    db: AsyncSession = Depends(get_db),
) -> List[MonitoringAlertResponse]:
    """List monitoring alerts with optional filtering."""
    orchestrator = MonitoringOrchestrator(db)

    try:
        alerts = await orchestrator.list_alerts(user_id=user_id, active_only=active_only)
        return [MonitoringAlertResponse.model_validate(a) for a in alerts]

    except Exception as e:
        logger.error("list_alerts_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/alerts/{alert_id}", response_model=MonitoringAlertResponse)
async def update_monitoring_alert(
    alert_id: UUID,
    update: MonitoringAlertUpdate,
    db: AsyncSession = Depends(get_db),
) -> MonitoringAlertResponse:
    """Update a monitoring alert.

    Can update topic, keywords, frequency, or active status.
    """
    orchestrator = MonitoringOrchestrator(db)

    try:
        # Get existing alert
        from sqlalchemy import select
        from src.models.paper_version import MonitoringAlert

        result = await db.execute(select(MonitoringAlert).where(MonitoringAlert.id == alert_id))
        alert = result.scalar_one_or_none()

        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Update fields
        if update.topic is not None:
            alert.topic = update.topic
        if update.keywords is not None:
            alert.keywords = update.keywords
        if update.frequency is not None:
            alert.frequency = update.frequency
        if update.active is not None:
            alert.active = update.active

        await db.commit()
        await db.refresh(alert)

        logger.info("alert_updated", alert_id=alert_id)
        return MonitoringAlertResponse.model_validate(alert)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_alert_error", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts/{alert_id}", status_code=204)
async def delete_monitoring_alert(
    alert_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a monitoring alert (deactivate instead of hard delete)."""
    orchestrator = MonitoringOrchestrator(db)

    try:
        alert = await orchestrator.update_alert_status(alert_id, active=False)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        logger.info("alert_deleted", alert_id=alert_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_alert_error", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

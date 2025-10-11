"""Literature Monitoring Orchestrator Service.

High-level service for managing literature monitoring sources and triggering syncs.
Provides a clean API for creating, updating, and managing monitoring sources.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from src.models.paper_version import LiteratureSource, MonitoringAlert
from src.tasks.monitoring_tasks import (
    sync_arxiv_papers_task,
    sync_pubmed_papers_task,
    sync_all_literature_sources_task,
)

logger = get_logger(__name__)


class MonitoringOrchestrator:
    """Orchestrates literature monitoring operations.

    Provides high-level API for:
    - Creating and managing monitoring sources
    - Triggering manual syncs
    - Managing monitoring alerts
    - Viewing sync history and statistics
    """

    def __init__(self, db: AsyncSession):
        """Initialize orchestrator.

        Args:
            db: Database session
        """
        self.db = db

    async def create_arxiv_source(
        self,
        categories: List[str],
        sync_frequency: str = "daily",
        status: str = "active",
    ) -> LiteratureSource:
        """Create a new ArXiv monitoring source.

        Args:
            categories: List of ArXiv categories (e.g., ["cs.AI", "cs.LG"])
            sync_frequency: How often to sync (daily, weekly)
            status: Initial status (active, paused, disabled)

        Returns:
            Created LiteratureSource

        Example:
            source = await orchestrator.create_arxiv_source(
                categories=["cs.AI", "cs.LG", "q-bio.NC"],
                sync_frequency="daily"
            )
        """
        source = LiteratureSource(
            id=uuid4(),
            source_type="arxiv",
            category=",".join(categories),
            query=None,
            last_sync_time=None,
            status=status,
            sync_frequency=sync_frequency,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self.db.add(source)
        await self.db.commit()
        await self.db.refresh(source)

        logger.info(
            "arxiv_source_created",
            source_id=source.id,
            categories=categories,
            frequency=sync_frequency,
        )

        return source

    async def create_pubmed_source(
        self,
        query: str,
        sync_frequency: str = "daily",
        status: str = "active",
    ) -> LiteratureSource:
        """Create a new PubMed monitoring source.

        Args:
            query: PubMed search query (supports MeSH terms, field tags, etc.)
            sync_frequency: How often to sync (daily, weekly)
            status: Initial status (active, paused, disabled)

        Returns:
            Created LiteratureSource

        Example:
            source = await orchestrator.create_pubmed_source(
                query="neuroscience[MeSH] AND machine learning",
                sync_frequency="daily"
            )
        """
        source = LiteratureSource(
            id=uuid4(),
            source_type="pubmed",
            category=None,
            query=query,
            last_sync_time=None,
            status=status,
            sync_frequency=sync_frequency,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self.db.add(source)
        await self.db.commit()
        await self.db.refresh(source)

        logger.info(
            "pubmed_source_created",
            source_id=source.id,
            query=query,
            frequency=sync_frequency,
        )

        return source

    async def get_source(self, source_id: UUID) -> Optional[LiteratureSource]:
        """Get a monitoring source by ID.

        Args:
            source_id: UUID of source

        Returns:
            LiteratureSource or None if not found
        """
        result = await self.db.execute(
            select(LiteratureSource).where(LiteratureSource.id == source_id)
        )
        return result.scalar_one_or_none()

    async def list_sources(
        self,
        source_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[LiteratureSource]:
        """List monitoring sources with optional filtering.

        Args:
            source_type: Filter by type (arxiv, pubmed)
            status: Filter by status (active, paused, disabled)

        Returns:
            List of LiteratureSource
        """
        query = select(LiteratureSource)

        if source_type:
            query = query.where(LiteratureSource.source_type == source_type)
        if status:
            query = query.where(LiteratureSource.status == status)

        query = query.order_by(LiteratureSource.created_at.desc())

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update_source_status(
        self,
        source_id: UUID,
        status: str,
    ) -> Optional[LiteratureSource]:
        """Update monitoring source status.

        Args:
            source_id: UUID of source
            status: New status (active, paused, disabled)

        Returns:
            Updated LiteratureSource or None if not found
        """
        source = await self.get_source(source_id)
        if not source:
            return None

        source.status = status
        source.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(source)

        logger.info("source_status_updated", source_id=source_id, status=status)

        return source

    async def trigger_manual_sync(
        self,
        source_id: UUID,
        api_key: Optional[str] = None,
    ) -> Dict:
        """Trigger a manual sync for a specific source.

        Args:
            source_id: UUID of source to sync
            api_key: Optional NCBI API key (for PubMed sources)

        Returns:
            Dict with task information

        Example:
            result = await orchestrator.trigger_manual_sync(source_id)
            # Returns: {"task_id": "...", "source_type": "arxiv", "status": "pending"}
        """
        source = await self.get_source(source_id)
        if not source:
            return {"error": "Source not found"}

        if source.source_type == "arxiv":
            task = sync_arxiv_papers_task.delay(str(source_id))
            logger.info("manual_arxiv_sync_triggered", source_id=source_id, task_id=task.id)

        elif source.source_type == "pubmed":
            task = sync_pubmed_papers_task.delay(str(source_id), api_key)
            logger.info("manual_pubmed_sync_triggered", source_id=source_id, task_id=task.id)

        else:
            return {"error": f"Unknown source type: {source.source_type}"}

        return {
            "task_id": task.id,
            "source_id": str(source_id),
            "source_type": source.source_type,
            "status": "pending",
        }

    async def trigger_full_sync(self) -> Dict:
        """Trigger a full sync of all active sources.

        Returns:
            Dict with task information
        """
        task = sync_all_literature_sources_task.delay()
        logger.info("full_sync_triggered", task_id=task.id)

        return {
            "task_id": task.id,
            "status": "pending",
            "message": "Full sync dispatched for all active sources",
        }

    async def create_alert(
        self,
        topic: str,
        keywords: List[str],
        frequency: str = "daily",
        user_id: Optional[UUID] = None,
    ) -> MonitoringAlert:
        """Create a new monitoring alert.

        Args:
            topic: Alert topic/title
            keywords: List of keywords to monitor
            frequency: Alert frequency (daily, weekly, monthly)
            user_id: Optional user ID for personalized alerts

        Returns:
            Created MonitoringAlert

        Example:
            alert = await orchestrator.create_alert(
                topic="AI in Neuroscience",
                keywords=["machine learning", "brain imaging", "fMRI"],
                frequency="weekly"
            )
        """
        alert = MonitoringAlert(
            id=uuid4(),
            user_id=user_id,
            topic=topic,
            keywords=keywords,
            frequency=frequency,
            last_alert_sent=None,
            active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self.db.add(alert)
        await self.db.commit()
        await self.db.refresh(alert)

        logger.info(
            "monitoring_alert_created",
            alert_id=alert.id,
            topic=topic,
            keywords=keywords,
            frequency=frequency,
        )

        return alert

    async def list_alerts(
        self,
        user_id: Optional[UUID] = None,
        active_only: bool = True,
    ) -> List[MonitoringAlert]:
        """List monitoring alerts with optional filtering.

        Args:
            user_id: Filter by user ID
            active_only: Only return active alerts

        Returns:
            List of MonitoringAlert
        """
        query = select(MonitoringAlert)

        if user_id:
            query = query.where(MonitoringAlert.user_id == user_id)
        if active_only:
            query = query.where(MonitoringAlert.active == True)

        query = query.order_by(MonitoringAlert.created_at.desc())

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update_alert_status(
        self,
        alert_id: UUID,
        active: bool,
    ) -> Optional[MonitoringAlert]:
        """Update monitoring alert status.

        Args:
            alert_id: UUID of alert
            active: New active status

        Returns:
            Updated MonitoringAlert or None if not found
        """
        result = await self.db.execute(
            select(MonitoringAlert).where(MonitoringAlert.id == alert_id)
        )
        alert = result.scalar_one_or_none()

        if not alert:
            return None

        alert.active = active
        alert.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(alert)

        logger.info("alert_status_updated", alert_id=alert_id, active=active)

        return alert

    async def get_source_statistics(self, source_id: UUID) -> Optional[Dict]:
        """Get statistics for a monitoring source.

        Args:
            source_id: UUID of source

        Returns:
            Dict with statistics or None if source not found
        """
        source = await self.get_source(source_id)
        if not source:
            return None

        # Calculate time since last sync
        time_since_sync = None
        if source.last_sync_time:
            time_since_sync = (datetime.utcnow() - source.last_sync_time).total_seconds()

        # TODO: Add actual paper count from database
        # For now, return basic statistics
        return {
            "source_id": str(source.id),
            "source_type": source.source_type,
            "status": source.status,
            "sync_frequency": source.sync_frequency,
            "last_sync_time": source.last_sync_time.isoformat() if source.last_sync_time else None,
            "time_since_sync_seconds": time_since_sync,
            "configuration": {
                "categories": source.category.split(",") if source.category else None,
                "query": source.query,
            },
        }

"""Literature monitoring Celery tasks.

Automated periodic tasks for real-time literature monitoring from ArXiv and PubMed.
Supports daily/weekly sync schedules with error handling and retry logic.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from celery import Task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from src.core.celery_app import celery_app
from src.core.database import get_db
from src.models.project import Paper
from src.models.paper_version import LiteratureSource, MonitoringAlert
from src.services.external.arxiv_monitor import ArXivMonitor, ArXivPaper
from src.services.external.pubmed_monitor import PubMedMonitor, PubMedPaper
from src.services.knowledge_base.ingestion import LiteratureIngestion

logger = get_logger(__name__)


class MonitoringTask(Task):
    """Base task with error handling and retries."""

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 60}  # Retry after 1 minute
    retry_backoff = True
    retry_backoff_max = 600  # Max 10 minutes
    retry_jitter = True


@celery_app.task(base=MonitoringTask, name="sync_arxiv_papers")
def sync_arxiv_papers_task(source_id: str) -> Dict:
    """Sync papers from ArXiv for a specific source.

    Args:
        source_id: UUID of LiteratureSource record

    Returns:
        Dict with sync results (papers_found, papers_ingested, errors)
    """
    return asyncio.run(_sync_arxiv_papers(UUID(source_id)))


async def _sync_arxiv_papers(source_id: UUID) -> Dict:
    """Async implementation of ArXiv sync."""
    async for db in get_db():
        try:
            # Get source configuration
            result = await db.execute(
                select(LiteratureSource).where(LiteratureSource.id == source_id)
            )
            source = result.scalar_one_or_none()

            if not source:
                logger.error("literature_source_not_found", source_id=source_id)
                return {"error": "Source not found"}

            if source.status != "active":
                logger.info("literature_source_inactive", source_id=source_id, status=source.status)
                return {"skipped": "Source inactive"}

            # Parse categories
            categories = source.category.split(",") if source.category else []
            if not categories:
                logger.error("no_categories_configured", source_id=source_id)
                return {"error": "No categories configured"}

            # Calculate days_back based on last sync
            days_back = 1  # Default: last 24 hours
            if source.last_sync_time:
                delta = datetime.utcnow() - source.last_sync_time
                days_back = max(1, min(7, delta.days + 1))  # Between 1-7 days

            logger.info(
                "starting_arxiv_sync",
                source_id=source_id,
                categories=categories,
                days_back=days_back,
            )

            # Fetch papers from ArXiv
            async with ArXivMonitor() as monitor:
                papers = await monitor.fetch_recent_papers(
                    categories=categories,
                    days_back=days_back,
                    max_results=100,
                )

            logger.info("arxiv_papers_fetched", count=len(papers), source_id=source_id)

            # Ingest papers into knowledge base
            ingestion_service = LiteratureIngestion(db)
            ingested_count = 0
            errors = []

            for paper in papers:
                try:
                    # Check if paper already exists (by ArXiv ID)
                    existing = await db.execute(
                        select(Paper).where(Paper.arxiv_id == paper.arxiv_id)
                    )
                    if existing.scalar_one_or_none():
                        logger.debug("paper_already_exists", arxiv_id=paper.arxiv_id)
                        continue

                    # Ingest paper
                    paper_metadata = {
                        "title": paper.title,
                        "abstract": paper.abstract,
                        "authors": paper.authors,
                        "publication_date": paper.published.isoformat(),
                        "arxiv_id": paper.arxiv_id,
                        "categories": paper.categories,
                        "pdf_url": paper.pdf_url,
                        "doi": paper.doi,
                    }

                    await ingestion_service.ingest_paper(paper_metadata)
                    ingested_count += 1

                except Exception as e:
                    logger.error("paper_ingestion_failed", arxiv_id=paper.arxiv_id, error=str(e))
                    errors.append({"arxiv_id": paper.arxiv_id, "error": str(e)})

            # Update source last_sync_time
            source.last_sync_time = datetime.utcnow()
            await db.commit()

            logger.info(
                "arxiv_sync_completed",
                source_id=source_id,
                papers_found=len(papers),
                papers_ingested=ingested_count,
                errors=len(errors),
            )

            return {
                "source_id": str(source_id),
                "papers_found": len(papers),
                "papers_ingested": ingested_count,
                "errors": errors,
                "sync_time": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("arxiv_sync_error", source_id=source_id, error=str(e))
            raise


@celery_app.task(base=MonitoringTask, name="sync_pubmed_papers")
def sync_pubmed_papers_task(source_id: str, api_key: Optional[str] = None) -> Dict:
    """Sync papers from PubMed for a specific source.

    Args:
        source_id: UUID of LiteratureSource record
        api_key: Optional NCBI API key for higher rate limits

    Returns:
        Dict with sync results (papers_found, papers_ingested, errors)
    """
    return asyncio.run(_sync_pubmed_papers(UUID(source_id), api_key))


async def _sync_pubmed_papers(source_id: UUID, api_key: Optional[str] = None) -> Dict:
    """Async implementation of PubMed sync."""
    async for db in get_db():
        try:
            # Get source configuration
            result = await db.execute(
                select(LiteratureSource).where(LiteratureSource.id == source_id)
            )
            source = result.scalar_one_or_none()

            if not source:
                logger.error("literature_source_not_found", source_id=source_id)
                return {"error": "Source not found"}

            if source.status != "active":
                logger.info("literature_source_inactive", source_id=source_id, status=source.status)
                return {"skipped": "Source inactive"}

            # Get query
            query = source.query
            if not query:
                logger.error("no_query_configured", source_id=source_id)
                return {"error": "No query configured"}

            # Calculate days_back based on last sync
            days_back = 1  # Default: last 24 hours
            if source.last_sync_time:
                delta = datetime.utcnow() - source.last_sync_time
                days_back = max(1, min(7, delta.days + 1))  # Between 1-7 days

            logger.info(
                "starting_pubmed_sync",
                source_id=source_id,
                query=query,
                days_back=days_back,
            )

            # Fetch papers from PubMed
            async with PubMedMonitor(api_key=api_key) as monitor:
                papers = await monitor.fetch_recent_papers(
                    query=query,
                    days_back=days_back,
                    max_results=100,
                )

            logger.info("pubmed_papers_fetched", count=len(papers), source_id=source_id)

            # Ingest papers into knowledge base
            ingestion_service = LiteratureIngestion(db)
            ingested_count = 0
            errors = []

            for paper in papers:
                try:
                    # Check if paper already exists (by PMID)
                    existing = await db.execute(
                        select(Paper).where(Paper.pubmed_id == paper.pmid)
                    )
                    if existing.scalar_one_or_none():
                        logger.debug("paper_already_exists", pmid=paper.pmid)
                        continue

                    # Ingest paper
                    paper_metadata = {
                        "title": paper.title,
                        "abstract": paper.abstract,
                        "authors": paper.authors,
                        "publication_date": paper.pub_date.isoformat(),
                        "pubmed_id": paper.pmid,
                        "journal": paper.journal,
                        "doi": paper.doi,
                        "pmc_id": paper.pmc_id,
                        "mesh_terms": paper.mesh_terms,
                    }

                    await ingestion_service.ingest_paper(paper_metadata)
                    ingested_count += 1

                except Exception as e:
                    logger.error("paper_ingestion_failed", pmid=paper.pmid, error=str(e))
                    errors.append({"pmid": paper.pmid, "error": str(e)})

            # Update source last_sync_time
            source.last_sync_time = datetime.utcnow()
            await db.commit()

            logger.info(
                "pubmed_sync_completed",
                source_id=source_id,
                papers_found=len(papers),
                papers_ingested=ingested_count,
                errors=len(errors),
            )

            return {
                "source_id": str(source_id),
                "papers_found": len(papers),
                "papers_ingested": ingested_count,
                "errors": errors,
                "sync_time": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("pubmed_sync_error", source_id=source_id, error=str(e))
            raise


@celery_app.task(name="sync_all_literature_sources")
def sync_all_literature_sources_task() -> Dict:
    """Sync all active literature sources (ArXiv + PubMed).

    Scheduled to run periodically (default: daily).
    Dispatches individual sync tasks for each source.

    Returns:
        Dict with overall sync results
    """
    return asyncio.run(_sync_all_literature_sources())


async def _sync_all_literature_sources() -> Dict:
    """Async implementation of full sync."""
    async for db in get_db():
        try:
            logger.info("starting_full_literature_sync")

            # Get all active sources
            result = await db.execute(
                select(LiteratureSource).where(LiteratureSource.status == "active")
            )
            sources = result.scalars().all()

            logger.info("active_sources_found", count=len(sources))

            # Dispatch sync tasks
            arxiv_tasks = []
            pubmed_tasks = []

            for source in sources:
                if source.source_type == "arxiv":
                    task = sync_arxiv_papers_task.delay(str(source.id))
                    arxiv_tasks.append(task)
                    logger.info("arxiv_sync_dispatched", source_id=source.id)

                elif source.source_type == "pubmed":
                    task = sync_pubmed_papers_task.delay(str(source.id))
                    pubmed_tasks.append(task)
                    logger.info("pubmed_sync_dispatched", source_id=source.id)

            return {
                "sources_total": len(sources),
                "arxiv_tasks": len(arxiv_tasks),
                "pubmed_tasks": len(pubmed_tasks),
                "dispatch_time": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("full_sync_error", error=str(e))
            raise


@celery_app.task(name="check_monitoring_alerts")
def check_monitoring_alerts_task() -> Dict:
    """Check for monitoring alerts and send notifications.

    Scheduled to run periodically (default: hourly).
    Checks all active alerts and sends notifications if new papers match.

    Returns:
        Dict with alert check results
    """
    return asyncio.run(_check_monitoring_alerts())


async def _check_monitoring_alerts() -> Dict:
    """Async implementation of alert checking."""
    async for db in get_db():
        try:
            logger.info("starting_alert_check")

            # Get all active alerts
            result = await db.execute(
                select(MonitoringAlert).where(MonitoringAlert.active == True)
            )
            alerts = result.scalars().all()

            logger.info("active_alerts_found", count=len(alerts))

            alerts_triggered = 0

            for alert in alerts:
                # Check if it's time to send alert based on frequency
                if alert.last_alert_sent:
                    time_since_last = datetime.utcnow() - alert.last_alert_sent

                    if alert.frequency == "daily" and time_since_last < timedelta(days=1):
                        continue
                    elif alert.frequency == "weekly" and time_since_last < timedelta(weeks=1):
                        continue
                    elif alert.frequency == "monthly" and time_since_last < timedelta(days=30):
                        continue

                # Search for papers matching alert keywords
                # TODO: Implement actual search logic with knowledge base
                # For now, just log the alert
                logger.info(
                    "alert_check",
                    alert_id=alert.id,
                    topic=alert.topic,
                    keywords=alert.keywords,
                )

                # Update last_alert_sent time
                alert.last_alert_sent = datetime.utcnow()
                alerts_triggered += 1

            await db.commit()

            logger.info(
                "alert_check_completed",
                alerts_checked=len(alerts),
                alerts_triggered=alerts_triggered,
            )

            return {
                "alerts_checked": len(alerts),
                "alerts_triggered": alerts_triggered,
                "check_time": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("alert_check_error", error=str(e))
            raise


# Celery Beat schedule configuration
celery_app.conf.beat_schedule = {
    "sync-all-literature-daily": {
        "task": "sync_all_literature_sources",
        "schedule": 86400.0,  # 24 hours in seconds
        "options": {"expires": 3600},  # Expire after 1 hour if not picked up
    },
    "check-alerts-hourly": {
        "task": "check_monitoring_alerts",
        "schedule": 3600.0,  # 1 hour in seconds
        "options": {"expires": 600},  # Expire after 10 minutes if not picked up
    },
}

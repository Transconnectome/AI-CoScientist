"""Celery tasks package."""

from src.tasks.experiment_tasks import (
    design_experiment_task,
    analyze_experiment_task
)
from src.tasks.hypothesis_tasks import (
    generate_hypotheses_task,
    validate_hypothesis_task
)
from src.tasks.literature_tasks import ingest_literature_task
from src.tasks.monitoring_tasks import (
    sync_arxiv_papers_task,
    sync_pubmed_papers_task,
    sync_all_literature_sources_task,
    check_monitoring_alerts_task,
)

__all__ = [
    "design_experiment_task",
    "analyze_experiment_task",
    "generate_hypotheses_task",
    "validate_hypothesis_task",
    "ingest_literature_task",
    "sync_arxiv_papers_task",
    "sync_pubmed_papers_task",
    "sync_all_literature_sources_task",
    "check_monitoring_alerts_task",
]

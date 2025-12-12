import argparse
import asyncio
import logging
import sys
from uuid import UUID

from dotenv import load_dotenv

# Load environment variables before importing settings
load_dotenv()

from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from src.core.engine import CoScientistEngine
from src.models.project import Project, ProjectStatus
from src.services.llm import LLMService
from src.services.knowledge_base.search import KnowledgeBaseSearch
from src.services.hypothesis.generator import HypothesisGenerator
from src.services.experiment.design import ExperimentDesigner
from src.services.paper.generator import PaperGenerator
from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

async def get_db_session():
    """Create database session."""
    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    return async_session()

async def get_engine(db: AsyncSession) -> CoScientistEngine:
    """Initialize CoScientistEngine with dependencies."""
    llm_service = LLMService()
    kb_search = KnowledgeBaseSearch()
    
    hypothesis_generator = HypothesisGenerator(
        llm_service=llm_service,
        knowledge_base=kb_search,
        db=db
    )
    
    experiment_designer = ExperimentDesigner(
        llm_service=llm_service,
        knowledge_base=kb_search,
        db=db
    )
    
    paper_generator = PaperGenerator(
        llm_service=llm_service,
        knowledge_base=kb_search,
        db=db
    )
    
    return CoScientistEngine(
        hypothesis_generator=hypothesis_generator,
        experiment_designer=experiment_designer,
        paper_generator=paper_generator,
        db=db
    )

async def start_project(args):
    """Start a new project."""
    console.print(f"[bold green]Starting new project:[/bold green] {args.topic}")
    
    db = await get_db_session()
    try:
        engine = await get_engine(db)
        project = await engine.run_discovery_loop(args.topic)
        console.print(f"[bold blue]Project Started![/bold blue] ID: {project.id}")
        console.print(f"Status: {project.status}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Failed to start project")
    finally:
        await db.close()

async def list_projects(args):
    """List all projects."""
    db = await get_db_session()
    try:
        result = await db.execute(select(Project))
        projects = result.scalars().all()
        
        table = Table(title="Research Projects")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Topic", style="magenta")
        table.add_column("Status", style="green")
        
        for p in projects:
            table.add_row(str(p.id), p.research_question[:50], p.status)
            
        console.print(table)
    finally:
        await db.close()

async def status_project(args):
    """Get status of a project."""
    db = await get_db_session()
    try:
        project_id = UUID(args.project_id)
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        
        if not project:
            console.print(f"[bold red]Project {project_id} not found[/bold red]")
            return

        console.print(f"[bold]Project:[/bold] {project.research_question}")
        console.print(f"[bold]Status:[/bold] {project.status}")
        # Add more details here (hypotheses, experiments, etc.)
        
    except ValueError:
        console.print("[bold red]Invalid Project ID[/bold red]")
    finally:
        await db.close()

def main():
    parser = argparse.ArgumentParser(description="AI-CoScientist CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start a new research project")
    start_parser.add_argument("topic", help="Research topic or question")
    
    # List command
    subparsers.add_parser("list", help="List all projects")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get project status")
    status_parser.add_argument("project_id", help="Project ID")
    
    args = parser.parse_args()
    
    if args.command == "start":
        asyncio.run(start_project(args))
    elif args.command == "list":
        asyncio.run(list_projects(args))
    elif args.command == "status":
        asyncio.run(status_project(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

"""Integration tests for database operations."""

import pytest
from uuid import uuid4
from datetime import datetime, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.project import (
    Project, Hypothesis, Experiment, Literature,
    ProjectStatus, HypothesisStatus, ExperimentStatus
)
from src.core.database import get_db


@pytest.fixture
async def db_session():
    """Create database session for testing."""
    async for session in get_db():
        yield session


@pytest.fixture
async def test_project_db(db_session: AsyncSession):
    """Create a test project in database."""
    project = Project(
        id=uuid4(),
        name="DB Test Project",
        description="Database integration test",
        research_domain="Computer Science",
        status=ProjectStatus.ACTIVE,
        created_at=datetime.utcnow()
    )
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)
    return project


@pytest.mark.asyncio
class TestProjectOperations:
    """Tests for Project database operations."""

    async def test_create_project(self, db_session: AsyncSession):
        """Test creating a project in database."""
        project = Project(
            id=uuid4(),
            name="New Project",
            description="Test project creation",
            research_domain="Biology",
            status=ProjectStatus.ACTIVE
        )

        db_session.add(project)
        await db_session.commit()
        await db_session.refresh(project)

        assert project.id is not None
        assert project.created_at is not None
        assert project.status == ProjectStatus.ACTIVE

    async def test_query_projects(self, db_session: AsyncSession, test_project_db):
        """Test querying projects."""
        result = await db_session.execute(
            select(Project).where(Project.id == test_project_db.id)
        )
        project = result.scalar_one_or_none()

        assert project is not None
        assert project.id == test_project_db.id
        assert project.name == test_project_db.name

    async def test_update_project(self, db_session: AsyncSession, test_project_db):
        """Test updating a project."""
        test_project_db.status = ProjectStatus.COMPLETED
        test_project_db.updated_at = datetime.utcnow()

        await db_session.commit()
        await db_session.refresh(test_project_db)

        assert test_project_db.status == ProjectStatus.COMPLETED
        assert test_project_db.updated_at is not None

    async def test_delete_project(self, db_session: AsyncSession):
        """Test deleting a project."""
        project = Project(
            id=uuid4(),
            name="Delete Me",
            description="Will be deleted",
            research_domain="Physics"
        )
        db_session.add(project)
        await db_session.commit()

        project_id = project.id

        await db_session.delete(project)
        await db_session.commit()

        result = await db_session.execute(
            select(Project).where(Project.id == project_id)
        )
        deleted_project = result.scalar_one_or_none()

        assert deleted_project is None

    async def test_filter_by_status(self, db_session: AsyncSession):
        """Test filtering projects by status."""
        # Create projects with different statuses
        active_project = Project(
            id=uuid4(),
            name="Active Project",
            status=ProjectStatus.ACTIVE
        )
        completed_project = Project(
            id=uuid4(),
            name="Completed Project",
            status=ProjectStatus.COMPLETED
        )

        db_session.add_all([active_project, completed_project])
        await db_session.commit()

        # Query active projects
        result = await db_session.execute(
            select(Project).where(Project.status == ProjectStatus.ACTIVE)
        )
        active_projects = result.scalars().all()

        assert len(active_projects) > 0
        assert all(p.status == ProjectStatus.ACTIVE for p in active_projects)

    async def test_filter_by_domain(self, db_session: AsyncSession):
        """Test filtering projects by research domain."""
        bio_project = Project(
            id=uuid4(),
            name="Biology Project",
            research_domain="Biology"
        )
        cs_project = Project(
            id=uuid4(),
            name="CS Project",
            research_domain="Computer Science"
        )

        db_session.add_all([bio_project, cs_project])
        await db_session.commit()

        result = await db_session.execute(
            select(Project).where(Project.research_domain == "Biology")
        )
        bio_projects = result.scalars().all()

        assert len(bio_projects) > 0
        assert all(p.research_domain == "Biology" for p in bio_projects)


@pytest.mark.asyncio
class TestHypothesisOperations:
    """Tests for Hypothesis database operations."""

    async def test_create_hypothesis(self, db_session: AsyncSession, test_project_db):
        """Test creating a hypothesis."""
        hypothesis = Hypothesis(
            id=uuid4(),
            project_id=test_project_db.id,
            content="Test hypothesis content",
            rationale="Test rationale",
            novelty_score=0.8,
            status=HypothesisStatus.PENDING
        )

        db_session.add(hypothesis)
        await db_session.commit()
        await db_session.refresh(hypothesis)

        assert hypothesis.id is not None
        assert hypothesis.project_id == test_project_db.id
        assert hypothesis.status == HypothesisStatus.PENDING

    async def test_query_hypotheses_by_project(
        self,
        db_session: AsyncSession,
        test_project_db
    ):
        """Test querying hypotheses by project."""
        # Create multiple hypotheses
        hypotheses = [
            Hypothesis(
                id=uuid4(),
                project_id=test_project_db.id,
                content=f"Hypothesis {i}",
                status=HypothesisStatus.PENDING
            )
            for i in range(3)
        ]

        db_session.add_all(hypotheses)
        await db_session.commit()

        result = await db_session.execute(
            select(Hypothesis).where(
                Hypothesis.project_id == test_project_db.id
            )
        )
        project_hypotheses = result.scalars().all()

        assert len(project_hypotheses) >= 3

    async def test_update_hypothesis_status(
        self,
        db_session: AsyncSession,
        test_project_db
    ):
        """Test updating hypothesis status."""
        hypothesis = Hypothesis(
            id=uuid4(),
            project_id=test_project_db.id,
            content="Status update test",
            status=HypothesisStatus.PENDING
        )

        db_session.add(hypothesis)
        await db_session.commit()

        hypothesis.status = HypothesisStatus.VALIDATED
        hypothesis.validation_score = 0.85
        await db_session.commit()
        await db_session.refresh(hypothesis)

        assert hypothesis.status == HypothesisStatus.VALIDATED
        assert hypothesis.validation_score == 0.85

    async def test_filter_by_novelty_score(
        self,
        db_session: AsyncSession,
        test_project_db
    ):
        """Test filtering hypotheses by novelty score."""
        high_novelty = Hypothesis(
            id=uuid4(),
            project_id=test_project_db.id,
            content="High novelty",
            novelty_score=0.9
        )
        low_novelty = Hypothesis(
            id=uuid4(),
            project_id=test_project_db.id,
            content="Low novelty",
            novelty_score=0.3
        )

        db_session.add_all([high_novelty, low_novelty])
        await db_session.commit()

        result = await db_session.execute(
            select(Hypothesis).where(Hypothesis.novelty_score >= 0.7)
        )
        high_novelty_hypotheses = result.scalars().all()

        assert all(h.novelty_score >= 0.7 for h in high_novelty_hypotheses)


@pytest.mark.asyncio
class TestExperimentOperations:
    """Tests for Experiment database operations."""

    async def test_create_experiment(self, db_session: AsyncSession, test_project_db):
        """Test creating an experiment."""
        # Create hypothesis first
        hypothesis = Hypothesis(
            id=uuid4(),
            project_id=test_project_db.id,
            content="Test hypothesis"
        )
        db_session.add(hypothesis)
        await db_session.commit()

        experiment = Experiment(
            id=uuid4(),
            hypothesis_id=hypothesis.id,
            title="Test Experiment",
            protocol="Test protocol",
            sample_size=100,
            power=0.8,
            effect_size=0.5,
            status=ExperimentStatus.DESIGNED
        )

        db_session.add(experiment)
        await db_session.commit()
        await db_session.refresh(experiment)

        assert experiment.id is not None
        assert experiment.hypothesis_id == hypothesis.id

    async def test_query_experiments_by_hypothesis(
        self,
        db_session: AsyncSession,
        test_project_db
    ):
        """Test querying experiments by hypothesis."""
        hypothesis = Hypothesis(
            id=uuid4(),
            project_id=test_project_db.id,
            content="Test"
        )
        db_session.add(hypothesis)
        await db_session.commit()

        experiments = [
            Experiment(
                id=uuid4(),
                hypothesis_id=hypothesis.id,
                title=f"Experiment {i}",
                protocol="Test",
                sample_size=50
            )
            for i in range(2)
        ]

        db_session.add_all(experiments)
        await db_session.commit()

        result = await db_session.execute(
            select(Experiment).where(
                Experiment.hypothesis_id == hypothesis.id
            )
        )
        hypothesis_experiments = result.scalars().all()

        assert len(hypothesis_experiments) >= 2


@pytest.mark.asyncio
class TestLiteratureOperations:
    """Tests for Literature database operations."""

    async def test_create_literature(self, db_session: AsyncSession):
        """Test creating a literature entry."""
        literature = Literature(
            id=uuid4(),
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            abstract="Test abstract",
            doi="10.1234/test.2024",
            publication_date=datetime.utcnow(),
            citations_count=10,
            source="arxiv"
        )

        db_session.add(literature)
        await db_session.commit()
        await db_session.refresh(literature)

        assert literature.id is not None
        assert literature.title == "Test Paper"

    async def test_search_by_title(self, db_session: AsyncSession):
        """Test searching literature by title."""
        literature = Literature(
            id=uuid4(),
            title="Machine Learning Applications",
            abstract="Test abstract",
            source="arxiv"
        )

        db_session.add(literature)
        await db_session.commit()

        result = await db_session.execute(
            select(Literature).where(
                Literature.title.ilike("%machine learning%")
            )
        )
        found = result.scalars().all()

        assert len(found) > 0
        assert any("Machine Learning" in lit.title for lit in found)

    async def test_order_by_citations(self, db_session: AsyncSession):
        """Test ordering literature by citations."""
        lit1 = Literature(
            id=uuid4(),
            title="Paper 1",
            citations_count=100,
            source="arxiv"
        )
        lit2 = Literature(
            id=uuid4(),
            title="Paper 2",
            citations_count=50,
            source="arxiv"
        )

        db_session.add_all([lit1, lit2])
        await db_session.commit()

        result = await db_session.execute(
            select(Literature).order_by(Literature.citations_count.desc())
        )
        papers = result.scalars().all()

        assert papers[0].citations_count >= papers[1].citations_count


@pytest.mark.asyncio
class TestComplexQueries:
    """Tests for complex database queries."""

    async def test_join_project_hypotheses(
        self,
        db_session: AsyncSession,
        test_project_db
    ):
        """Test joining projects with hypotheses."""
        # Create hypotheses
        hypotheses = [
            Hypothesis(
                id=uuid4(),
                project_id=test_project_db.id,
                content=f"Hypothesis {i}"
            )
            for i in range(3)
        ]
        db_session.add_all(hypotheses)
        await db_session.commit()

        # Query with join
        result = await db_session.execute(
            select(Project, Hypothesis)
            .join(Hypothesis)
            .where(Project.id == test_project_db.id)
        )
        rows = result.all()

        assert len(rows) >= 3

    async def test_aggregate_hypothesis_count(
        self,
        db_session: AsyncSession,
        test_project_db
    ):
        """Test aggregating hypothesis count per project."""
        # Create hypotheses
        for i in range(5):
            hypothesis = Hypothesis(
                id=uuid4(),
                project_id=test_project_db.id,
                content=f"Hypothesis {i}"
            )
            db_session.add(hypothesis)
        await db_session.commit()

        # Aggregate query
        result = await db_session.execute(
            select(Project.id, func.count(Hypothesis.id))
            .join(Hypothesis)
            .where(Project.id == test_project_db.id)
            .group_by(Project.id)
        )
        row = result.first()

        assert row is not None
        assert row[1] >= 5  # At least 5 hypotheses

    async def test_filter_recent_projects(self, db_session: AsyncSession):
        """Test filtering projects created recently."""
        cutoff_date = datetime.utcnow() - timedelta(days=7)

        result = await db_session.execute(
            select(Project).where(Project.created_at >= cutoff_date)
        )
        recent_projects = result.scalars().all()

        assert all(
            p.created_at >= cutoff_date
            for p in recent_projects
            if p.created_at
        )


@pytest.mark.asyncio
class TestTransactions:
    """Tests for database transaction handling."""

    async def test_rollback_on_error(self, db_session: AsyncSession, test_project_db):
        """Test transaction rollback on error."""
        try:
            # Create a hypothesis
            hypothesis = Hypothesis(
                id=uuid4(),
                project_id=test_project_db.id,
                content="Test hypothesis"
            )
            db_session.add(hypothesis)

            # Force an error (e.g., duplicate ID)
            duplicate = Hypothesis(
                id=hypothesis.id,  # Same ID
                project_id=test_project_db.id,
                content="Duplicate"
            )
            db_session.add(duplicate)

            await db_session.commit()

        except Exception:
            await db_session.rollback()

        # Verify nothing was committed
        result = await db_session.execute(
            select(Hypothesis).where(
                Hypothesis.content == "Test hypothesis"
            )
        )
        found = result.scalar_one_or_none()

        # Should be None due to rollback
        assert found is None or found.content != "Test hypothesis"

    async def test_commit_multiple_objects(
        self,
        db_session: AsyncSession,
        test_project_db
    ):
        """Test committing multiple objects in one transaction."""
        hypotheses = [
            Hypothesis(
                id=uuid4(),
                project_id=test_project_db.id,
                content=f"Batch hypothesis {i}"
            )
            for i in range(5)
        ]

        db_session.add_all(hypotheses)
        await db_session.commit()

        # Verify all were committed
        result = await db_session.execute(
            select(Hypothesis).where(
                Hypothesis.project_id == test_project_db.id
            )
        )
        saved = result.scalars().all()

        assert len(saved) >= 5


@pytest.mark.asyncio
class TestIndexPerformance:
    """Tests for database index performance."""

    async def test_indexed_query_performance(
        self,
        db_session: AsyncSession,
        test_project_db
    ):
        """Test that indexed queries perform well."""
        import time

        # Create many projects
        projects = [
            Project(
                id=uuid4(),
                name=f"Project {i}",
                status=ProjectStatus.ACTIVE,
                research_domain="Biology"
            )
            for i in range(100)
        ]
        db_session.add_all(projects)
        await db_session.commit()

        # Query with indexed column (status)
        start = time.time()
        result = await db_session.execute(
            select(Project).where(Project.status == ProjectStatus.ACTIVE)
        )
        active_projects = result.scalars().all()
        duration = time.time() - start

        # Should be fast due to index
        assert duration < 1.0  # Less than 1 second
        assert len(active_projects) > 0

    async def test_composite_index_usage(
        self,
        db_session: AsyncSession,
        test_project_db
    ):
        """Test composite index on project_id + status."""
        # Create hypotheses with different statuses
        hypotheses = [
            Hypothesis(
                id=uuid4(),
                project_id=test_project_db.id,
                content=f"Hypothesis {i}",
                status=HypothesisStatus.VALIDATED if i % 2 == 0 else HypothesisStatus.PENDING
            )
            for i in range(50)
        ]
        db_session.add_all(hypotheses)
        await db_session.commit()

        # Query using composite index
        import time
        start = time.time()
        result = await db_session.execute(
            select(Hypothesis).where(
                Hypothesis.project_id == test_project_db.id,
                Hypothesis.status == HypothesisStatus.VALIDATED
            )
        )
        validated = result.scalars().all()
        duration = time.time() - start

        # Should be fast due to composite index
        assert duration < 1.0
        assert len(validated) > 0

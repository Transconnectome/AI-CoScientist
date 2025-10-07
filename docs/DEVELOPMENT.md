# AI-CoScientist Development Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-05

Complete guide for setting up, developing, and contributing to AI-CoScientist.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [Development Workflow](#development-workflow)
5. [Code Standards](#code-standards)
6. [Testing Guide](#testing-guide)
7. [Debugging](#debugging)
8. [Contributing](#contributing)

---

## Prerequisites

### Required Software

```bash
# Python 3.11+
python --version  # Should be 3.11 or higher

# Poetry (Python dependency management)
curl -sSL https://install.python-poetry.org | python3 -

# PostgreSQL 14+
psql --version

# Redis 7+
redis-server --version

# Git
git --version
```

### Optional Tools

```bash
# Docker (for containerized development)
docker --version

# pgAdmin (PostgreSQL GUI)
# Redis Insight (Redis GUI)
# Postman/Insomnia (API testing)
```

---

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/AI-CoScientist.git
cd AI-CoScientist
```

### 2. Install Dependencies

```bash
# Install Python dependencies
poetry install

# Activate virtual environment
poetry shell

# Verify installation
python -c "import fastapi; print(fastapi.__version__)"
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
nano .env
```

**Required Environment Variables**:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ai_coscientist
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4-turbo-preview

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Application
APP_NAME=AI-CoScientist
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

### 4. Database Setup

```bash
# Create database
createdb ai_coscientist

# Run migrations
poetry run alembic upgrade head

# Verify database
psql ai_coscientist -c "SELECT tablename FROM pg_tables WHERE schemaname='public';"
```

### 5. Start Services

#### Option A: Local Services

```bash
# Terminal 1: PostgreSQL (if not running as service)
postgres -D /usr/local/var/postgres

# Terminal 2: Redis
redis-server

# Terminal 3: ChromaDB (optional, for vector search)
chroma run --host localhost --port 8000

# Terminal 4: Application
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

#### Option B: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### 6. Verify Installation

```bash
# Check health endpoint
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status":"healthy","timestamp":"...","version":"1.0.0"}

# Open API docs
open http://localhost:8000/docs
```

---

## Project Structure

```
AI-CoScientist/
├── src/                          # Source code
│   ├── api/                      # API layer
│   │   ├── routes/               # API endpoints
│   │   │   ├── health.py
│   │   │   ├── projects.py
│   │   │   ├── literature.py
│   │   │   ├── hypotheses.py
│   │   │   └── experiments.py
│   │   ├── middleware/           # Middleware
│   │   └── schemas/              # Pydantic models
│   │
│   ├── services/                 # Business logic
│   │   ├── research/             # Research engine
│   │   │   ├── hypothesis_generator.py
│   │   │   ├── literature_analyzer.py
│   │   │   └── novelty_scorer.py
│   │   ├── experiment/           # Experiment engine
│   │   │   ├── design.py
│   │   │   ├── power_analysis.py
│   │   │   └── protocol_builder.py
│   │   ├── literature/           # Literature engine
│   │   │   ├── arxiv_fetcher.py
│   │   │   ├── pubmed_fetcher.py
│   │   │   └── ingestion_service.py
│   │   ├── llm/                  # LLM service
│   │   │   ├── service.py
│   │   │   ├── adapters/
│   │   │   │   ├── openai.py
│   │   │   │   └── anthropic.py
│   │   │   ├── interface.py
│   │   │   └── types.py
│   │   └── knowledge_base/       # Vector store
│   │
│   ├── models/                   # Database models
│   │   └── project.py
│   │
│   ├── core/                     # Core utilities
│   │   ├── config.py             # Configuration
│   │   ├── database.py           # DB connection
│   │   ├── performance.py        # Metrics
│   │   ├── cache_manager.py      # Caching
│   │   └── connection_pool.py    # Connection pooling
│   │
│   └── main.py                   # Application entry point
│
├── tests/                        # Test suite
│   ├── test_services/            # Unit tests
│   ├── test_integration/         # Integration tests
│   └── test_e2e/                 # End-to-end tests
│
├── alembic/                      # Database migrations
│   └── versions/
│
├── docs/                         # Documentation
│   ├── INDEX.md
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md
│   └── DEVELOPMENT.md (this file)
│
├── scripts/                      # Utility scripts
│   ├── seed_data.py              # Database seeding
│   └── run_migrations.sh         # Migration helper
│
├── .env.example                  # Environment template
├── docker-compose.yml            # Docker composition
├── Dockerfile                    # Container definition
├── pyproject.toml                # Poetry configuration
├── pytest.ini                    # Test configuration
└── README.md                     # Project overview
```

---

## Development Workflow

### 1. Feature Development

#### Step 1: Create Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

#### Step 2: Implement Feature

```python
# Example: Add new API endpoint
# src/api/routes/new_feature.py

from fastapi import APIRouter, Depends
from src.api.schemas.new_feature import FeatureRequest, FeatureResponse

router = APIRouter(prefix="/api/v1/features", tags=["features"])

@router.post("/", response_model=FeatureResponse)
async def create_feature(
    request: FeatureRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create new feature."""
    # Implementation
    pass
```

#### Step 3: Add Tests

```python
# tests/test_integration/test_new_feature.py

import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_feature(client: AsyncClient):
    """Test feature creation."""
    response = await client.post(
        "/api/v1/features/",
        json={"name": "Test Feature"}
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Test Feature"
```

#### Step 4: Run Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_integration/test_new_feature.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

#### Step 5: Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type checking
poetry run mypy src/
```

#### Step 6: Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new feature endpoint

- Add POST /api/v1/features endpoint
- Implement feature creation logic
- Add integration tests
- Update API documentation"

# Push to remote
git push origin feature/your-feature-name
```

### 2. Database Changes

#### Create Migration

```bash
# Auto-generate migration
poetry run alembic revision --autogenerate -m "add new table"

# Review generated migration
cat alembic/versions/xxx_add_new_table.py

# Apply migration
poetry run alembic upgrade head

# Rollback if needed
poetry run alembic downgrade -1
```

#### Manual Migration

```python
# alembic/versions/xxx_add_feature.py

def upgrade() -> None:
    """Add feature table."""
    op.create_table(
        'features',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_index('idx_features_name', 'features', ['name'])

def downgrade() -> None:
    """Remove feature table."""
    op.drop_index('idx_features_name')
    op.drop_table('features')
```

### 3. API Development

#### Define Schema

```python
# src/api/schemas/feature.py

from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime

class FeatureBase(BaseModel):
    name: str = Field(..., description="Feature name")
    description: str | None = Field(None, description="Feature description")

class FeatureCreate(FeatureBase):
    pass

class FeatureResponse(FeatureBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True
```

#### Implement Endpoint

```python
# src/api/routes/features.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.api.schemas.feature import FeatureCreate, FeatureResponse
from src.models.feature import Feature

router = APIRouter(prefix="/api/v1/features", tags=["features"])

@router.post("/", response_model=FeatureResponse, status_code=201)
async def create_feature(
    feature_data: FeatureCreate,
    db: AsyncSession = Depends(get_db)
) -> FeatureResponse:
    """
    Create a new feature.

    Args:
        feature_data: Feature creation data
        db: Database session

    Returns:
        Created feature

    Raises:
        HTTPException: If feature already exists
    """
    # Check for duplicates
    existing = await db.execute(
        select(Feature).where(Feature.name == feature_data.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail="Feature with this name already exists"
        )

    # Create feature
    feature = Feature(**feature_data.model_dump())
    db.add(feature)
    await db.commit()
    await db.refresh(feature)

    return feature
```

---

## Code Standards

### Python Style Guide

Follow **PEP 8** with these additions:

```python
# Import order
import standard_library
import third_party
import local_app

# Type hints (required)
def process_data(text: str, limit: int = 10) -> list[dict]:
    pass

# Docstrings (Google style)
def complex_function(param1: str, param2: int) -> bool:
    """Brief description.

    Longer description explaining what this function does,
    its behavior, and any important notes.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
    """
    pass

# Async naming convention
async def fetch_data():  # async functions: use verbs
    pass
```

### File Organization

```python
# Standard module structure

"""Module docstring explaining purpose."""

# Imports
from typing import Optional
from fastapi import APIRouter

# Constants
MAX_ITEMS = 100
DEFAULT_TIMEOUT = 30

# Type definitions
ItemType = dict[str, any]

# Main code
class MyService:
    """Service class docstring."""

    def __init__(self):
        """Initialize service."""
        pass

    async def process(self, item: ItemType) -> ItemType:
        """Process an item."""
        pass
```

### Naming Conventions

```python
# Variables and functions: snake_case
user_name = "john"
def get_user_data(): pass

# Classes: PascalCase
class UserService: pass

# Constants: UPPER_CASE
MAX_RETRIES = 3

# Private members: _leading_underscore
def _internal_method(): pass

# Async functions: descriptive verbs
async def fetch_user(): pass
async def create_project(): pass
```

### Error Handling

```python
# Specific exceptions
from fastapi import HTTPException

# Good
raise HTTPException(
    status_code=404,
    detail="Project not found"
)

# Bad
raise Exception("Error")

# With logging
import structlog
logger = structlog.get_logger(__name__)

try:
    result = await process_data()
except ValueError as e:
    logger.error("data_processing_error", error=str(e))
    raise HTTPException(status_code=400, detail=str(e))
```

---

## Testing Guide

### Test Structure

```python
# tests/test_services/test_my_service.py

import pytest
from uuid import uuid4
from src.services.my_service import MyService

@pytest.fixture
def service():
    """Create service instance."""
    return MyService()

@pytest.fixture
async def db_session():
    """Create database session."""
    # Setup
    session = create_test_session()
    yield session
    # Teardown
    await session.close()

class TestMyService:
    """Test suite for MyService."""

    def test_sync_method(self, service):
        """Test synchronous method."""
        result = service.sync_method()
        assert result is not None

    @pytest.mark.asyncio
    async def test_async_method(self, service, db_session):
        """Test asynchronous method."""
        result = await service.async_method(db_session)
        assert result is not None
```

### Running Tests

```bash
# All tests
pytest

# Specific category
pytest -m unit
pytest -m integration
pytest -m e2e

# Specific file
pytest tests/test_services/test_my_service.py

# Specific test
pytest tests/test_services/test_my_service.py::TestMyService::test_sync_method

# With coverage
pytest --cov=src --cov-report=html --cov-report=term

# Parallel execution
pytest -n auto

# Verbose output
pytest -vv

# Stop on first failure
pytest -x

# Show print statements
pytest -s
```

### Writing Good Tests

```python
# AAA Pattern: Arrange, Act, Assert

@pytest.mark.asyncio
async def test_create_project(client, db_session):
    """Test project creation."""
    # Arrange
    project_data = {
        "name": "Test Project",
        "description": "Test description",
        "research_domain": "Biology"
    }

    # Act
    response = await client.post(
        "/api/v1/projects/",
        json=project_data
    )

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == project_data["name"]
    assert "id" in data
```

---

## Debugging

### Local Debugging

#### Using print/logging

```python
import structlog

logger = structlog.get_logger(__name__)

async def my_function(data: dict):
    logger.info("function_called", data=data)

    try:
        result = await process(data)
        logger.debug("processing_complete", result=result)
        return result
    except Exception as e:
        logger.error("processing_failed", error=str(e), data=data)
        raise
```

#### Using debugger

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or with async support
import ipdb; ipdb.set_trace()

# Or in VS Code: just click to add breakpoint
```

#### VS Code Configuration

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "src.main:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000"
            ],
            "jinja": true,
            "justMyCode": false
        },
        {
            "name": "Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "-v",
                "--tb=short"
            ]
        }
    ]
}
```

### Database Debugging

```python
# Enable SQL logging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Check connection pool
from src.core.database import engine
print(engine.pool.status())

# Analyze slow queries
# Enable in PostgreSQL config
ALTER DATABASE ai_coscientist SET log_min_duration_statement = 100;
```

### API Debugging

```bash
# Using curl
curl -X POST http://localhost:8000/api/v1/projects/ \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","domain":"Biology"}' \
  -v

# Using httpie
http POST localhost:8000/api/v1/projects/ name="Test" domain="Biology"

# Check OpenAPI docs
open http://localhost:8000/docs
```

---

## Contributing

### Pull Request Process

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/AI-CoScientist.git
   ```

2. **Create Branch**
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

4. **Test & Quality**
   ```bash
   pytest
   black src/ tests/
   ruff check src/ tests/
   mypy src/
   ```

5. **Commit**
   ```bash
   git commit -m "feat: add your feature

   - Detailed description
   - What changed
   - Why it changed"
   ```

6. **Push & PR**
   ```bash
   git push origin feature/your-feature
   # Create PR on GitHub
   ```

### Commit Message Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Example**:
```
feat(api): add batch project creation endpoint

- Add POST /api/v1/projects/batch endpoint
- Implement bulk creation with transaction support
- Add validation for batch size limits
- Update API documentation

Closes #123
```

---

## Additional Resources

### Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org/)
- [Pytest Docs](https://docs.pytest.org/)

### Internal Docs
- [Architecture](./ARCHITECTURE.md)
- [API Reference](./API_REFERENCE.md)
- [Deployment Guide](./DEPLOYMENT.md)

### Getting Help
- GitHub Issues: Bug reports and feature requests
- Discussions: General questions
- Email: dev@ai-coscientist.com

---

**Last Updated**: 2025-10-05
**Maintainers**: AI-CoScientist Development Team

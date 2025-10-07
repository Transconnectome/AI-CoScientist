# AI-CoScientist Testing Framework

Comprehensive testing suite for the AI-CoScientist platform.

## Test Structure

```
tests/
├── test_services/           # Unit tests
│   └── test_experiment_design.py
├── test_integration/        # Integration tests
│   ├── test_api_endpoints.py
│   ├── test_database_operations.py
│   └── test_external_apis.py
└── test_e2e/               # End-to-end tests
    ├── test_research_workflow.py
    └── test_complete_pipeline.py
```

## Test Categories

### 1. Unit Tests (`test_services/`)

Tests for individual components and services in isolation.

**Coverage**:
- Sample size calculation
- Power analysis
- Experiment design workflow
- Protocol generation
- Methodology search

**Run unit tests**:
```bash
pytest tests/test_services/ -v
```

### 2. Integration Tests (`test_integration/`)

Tests for API endpoints, database operations, and external service integrations.

**Coverage**:
- **API Endpoints**: All REST API endpoints (Projects, Literature, Hypotheses, Experiments)
- **Database Operations**: CRUD operations, queries, transactions, indexing
- **External APIs**: Claude API, ArXiv, PubMed, ChromaDB

**Run integration tests**:
```bash
pytest tests/test_integration/ -v -m integration
```

### 3. End-to-End Tests (`test_e2e/`)

Complete workflow tests simulating real research scenarios.

**Coverage**:
- Complete research pipeline (Project → Literature → Hypothesis → Experiment)
- Multi-project workflows
- Error recovery scenarios
- Performance under load
- Data integrity and consistency

**Run E2E tests**:
```bash
pytest tests/test_e2e/ -v -m e2e
```

## Running Tests

### All Tests
```bash
pytest
```

### By Category
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# E2E tests only
pytest -m e2e
```

### With Coverage
```bash
# All tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Specific test file with coverage
pytest tests/test_services/test_experiment_design.py --cov=src.services.experiment
```

### Parallel Execution
```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Specific Tests
```bash
# Run specific test class
pytest tests/test_services/test_experiment_design.py::TestSampleSizeCalculation

# Run specific test method
pytest tests/test_services/test_experiment_design.py::TestSampleSizeCalculation::test_calculate_sample_size_medium_effect

# Run tests matching pattern
pytest -k "sample_size"
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for system components
- `@pytest.mark.e2e`: End-to-end workflow tests
- `@pytest.mark.slow`: Tests that take significant time
- `@pytest.mark.asyncio`: Async tests (auto-detected)

## Test Configuration

### pytest.ini

Configuration file with:
- Test discovery patterns
- Markers definition
- Async mode configuration
- Coverage settings
- Logging configuration

### Fixtures

Common fixtures available:

```python
# Database fixtures
@pytest.fixture
async def db_session():
    """Database session for testing."""

# API client fixtures
@pytest.fixture
async def client():
    """HTTP client for API testing."""

# Service fixtures
@pytest.fixture
def claude_service():
    """Mocked Claude service."""

# Mock fixtures
@pytest.fixture
def mock_llm_service():
    """Mock LLM service with controlled responses."""
```

## Writing Tests

### Unit Test Example

```python
import pytest
from src.services.experiment.design import ExperimentDesigner

def test_calculate_sample_size(experiment_designer):
    """Test sample size calculation."""
    sample_size = experiment_designer._calculate_sample_size(
        effect_size=0.5,
        power=0.8,
        alpha=0.05
    )

    assert 60 <= sample_size <= 80
```

### Integration Test Example

```python
@pytest.mark.asyncio
async def test_create_project(client):
    """Test project creation API."""
    response = await client.post(
        "/api/v1/projects/",
        json={"name": "Test Project", "domain": "Biology"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Project"
```

### E2E Test Example

```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_workflow(client):
    """Test complete research workflow."""
    # Create project
    project_response = await client.post("/api/v1/projects/", json=project_data)
    project = project_response.json()

    # Ingest literature
    await client.post(f"/api/v1/projects/{project['id']}/literature/ingest", ...)

    # Generate hypotheses
    hypotheses = await client.post(f"/api/v1/projects/{project['id']}/hypotheses/generate", ...)

    # Design experiment
    experiment = await client.post(f"/api/v1/hypotheses/{hypothesis_id}/experiments/design", ...)

    # Verify complete chain
    assert experiment["hypothesis_id"] == hypothesis_id
```

## Test Data

### Mock Data

Tests use mocked data for:
- External API responses (Claude, ArXiv, PubMed)
- Database operations (with in-memory or test database)
- File system operations

### Test Database

Integration tests use a separate test database:

```python
# Set environment variable
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/ai_coscientist_test"

# Or use in-memory SQLite for faster tests
export DATABASE_URL="sqlite+aiosqlite:///:memory:"
```

## Coverage Goals

### Current Coverage

- **Unit Tests**: 85%+ coverage for core services
- **Integration Tests**: All API endpoints covered
- **E2E Tests**: Complete workflows covered

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

## Performance Testing

### Load Testing

```python
@pytest.mark.performance
async def test_concurrent_requests(client):
    """Test performance under load."""
    import asyncio

    # Create 100 concurrent requests
    responses = await asyncio.gather(*[
        client.get("/api/v1/projects/")
        for _ in range(100)
    ])

    # All should succeed
    assert all(r.status_code == 200 for r in responses)
```

### Benchmarking

Use pytest-benchmark for performance benchmarks:

```bash
pytest tests/ --benchmark-only
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install poetry
        poetry install

    - name: Run tests
      run: |
        poetry run pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

**Issue**: Tests failing with async errors
```bash
# Solution: Ensure pytest-asyncio is installed
poetry add --dev pytest-asyncio
```

**Issue**: Database connection errors
```bash
# Solution: Set correct DATABASE_URL or use in-memory DB
export DATABASE_URL="sqlite+aiosqlite:///:memory:"
```

**Issue**: Tests timing out
```bash
# Solution: Increase timeout in pytest.ini or use marker
@pytest.mark.timeout(300)
async def test_long_running():
    ...
```

### Debug Mode

```bash
# Run with debug output
pytest -vv --log-cli-level=DEBUG

# Run with PDB on failure
pytest --pdb

# Run and stop on first failure
pytest -x
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Descriptive Names**: Use clear, descriptive test names
3. **AAA Pattern**: Arrange, Act, Assert
4. **Mock External Services**: Don't call real APIs in tests
5. **Fast Tests**: Keep unit tests < 1 second
6. **Comprehensive Coverage**: Aim for 80%+ coverage
7. **Test Edge Cases**: Include boundary conditions and error cases

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [HTTPX](https://www.python-httpx.org/) - Async HTTP client for testing

---

**Last Updated**: 2025-10-05
**Framework Version**: 1.0.0

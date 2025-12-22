# Suggested Improvements for CLAUDE.md

Based on analysis of the codebase, here are specific improvements that could enhance the existing CLAUDE.md:

## 1. Add Single Test Running Commands

The current testing section is good but missing specific commands for running individual tests:

```bash
# Run single test file
poetry run pytest tests/rag/test_rag_evaluator.py -v

# Run single test method
poetry run pytest tests/rag/test_rag_evaluator.py::test_create_rag_evaluator -v

# Run tests with specific markers
poetry run pytest -m integration -v

# Run tests for specific module
poetry run pytest tests/agents/test_pool.py::TestAgentPool::test_agent_selection -v
```

## 2. Add Debugging Commands Section

```bash
# Debug with breakpoints
poetry run python -m debugpy --wait-for-client --listen 5678 src/main.py

# Interactive debugging
poetry run python -i scripts/proposal_optimizer_unified.py

# Celery worker debugging
poetry run celery -A src.core.celery_app worker --loglevel=debug --pool=solo
```

## 3. Add Error Recovery Commands

```bash
# Reset ChromaDB (DANGEROUS - only if corrupted)
rm -rf chromadb_data/
poetry run python scripts/initialize_chromadb.py

# Clear Redis cache
docker exec ai-coscientist-redis redis-cli FLUSHALL

# Database recovery
poetry run alembic downgrade -1
poetry run alembic upgrade head
```

## 4. Add Performance Monitoring Access

```bash
# Access monitoring dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
# API Health: http://localhost:8000/api/v1/health
# Metrics: http://localhost:8000/metrics
```

## 5. Technology Stack Update

Update the technology stack section to reflect latest versions from pyproject.toml:
- Add MCP (Model Context Protocol) 1.17.0
- Update specific versions for key dependencies
- Note Python 3.11+ requirement

## 6. Add Troubleshooting Section

Common issues and solutions:
- ChromaDB connection errors
- Celery worker not starting
- Database migration failures
- Memory issues with large embeddings

## 7. Add File Reference System

Include line references for key functions to help navigation:
```
src/agents/pool.py:123 - get_optimal_agent_team()
src/services/rag/unified_rag_orchestrator.py:45 - search_unified()
```

## 8. Add Development Workflow Examples

Concrete examples of common development tasks:
- Adding a new RAG strategy
- Creating a new agent type
- Adding API endpoints
- Writing integration tests

## 9. Environment Variables Validation

Add specific validation commands:
```bash
# Validate environment setup
poetry run python -c "from src.core.config import Settings; print('✅ Config valid')"

# Check API keys
poetry run python -c "from src.core.config import settings; print('OpenAI:', bool(settings.openai_api_key))"
```

## 10. Add Quick Health Check Script

```bash
# System health check
poetry run python scripts/health_check.py --all
```

These improvements would make the CLAUDE.md more actionable and practical for day-to-day development tasks.
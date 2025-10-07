# API Reference

**Base URL**: `http://localhost:8000/api/v1`
**Version**: 0.1.0
**Authentication**: JWT Bearer Token (when implemented)

---

## Table of Contents

- [Health API](#health-api)
- [Projects API](#projects-api)
- [Literature API](#literature-api)
- [Hypotheses API](#hypotheses-api)
- [Experiments API](#experiments-api)
- [Common Patterns](#common-patterns)
- [Error Responses](#error-responses)

---

## Health API

### GET /health

Health check endpoint to verify system status.

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "database": "connected",
    "redis": "connected",
    "vector_store": "ready"
  }
}
```

---

## Projects API

### POST /projects

Create a new research project.

**Request Body**:
```json
{
  "name": "string",
  "description": "string",
  "domain": "string",
  "research_question": "string"
}
```

**Response**: `201 Created`
```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "domain": "string",
  "research_question": "string",
  "status": "active",
  "created_at": "2025-10-05T00:00:00Z",
  "updated_at": "2025-10-05T00:00:00Z"
}
```

### GET /projects

List all projects.

**Query Parameters**:
- `skip` (int, optional): Number of records to skip (default: 0)
- `limit` (int, optional): Maximum records to return (default: 100)
- `status` (string, optional): Filter by status (active, completed, archived, paused)

**Response**: `200 OK`
```json
[
  {
    "id": "uuid",
    "name": "string",
    "description": "string",
    "domain": "string",
    "research_question": "string",
    "status": "active",
    "created_at": "2025-10-05T00:00:00Z",
    "updated_at": "2025-10-05T00:00:00Z"
  }
]
```

### GET /projects/{project_id}

Get project details.

**Path Parameters**:
- `project_id` (uuid): Project ID

**Response**: `200 OK`
```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "domain": "string",
  "research_question": "string",
  "status": "active",
  "hypotheses": [...],
  "papers": [...],
  "created_at": "2025-10-05T00:00:00Z",
  "updated_at": "2025-10-05T00:00:00Z"
}
```

---

## Literature API

### POST /literature/search

Search scientific literature.

**Request Body**:
```json
{
  "query": "string",
  "top_k": 10,
  "search_type": "hybrid",
  "filters": {
    "year_min": 2020,
    "year_max": 2025,
    "source_type": "journal"
  }
}
```

**Search Types**:
- `semantic`: Embedding-based similarity search
- `keyword`: PostgreSQL full-text search
- `hybrid`: Combined (70% semantic + 30% keyword)

**Response**: `200 OK`
```json
[
  {
    "document_id": "string",
    "title": "string",
    "abstract": "string",
    "score": 0.95,
    "metadata": {
      "year": 2024,
      "authors": ["Author 1", "Author 2"],
      "journal": "Journal Name",
      "doi": "10.xxxx/xxxxx"
    },
    "highlights": ["relevant snippet 1", "relevant snippet 2"]
  }
]
```

### POST /literature/ingest

Ingest literature from external sources.

**Request Body**:
```json
{
  "source_type": "doi",
  "source_value": "10.1038/s41586-023-xxxxx",
  "max_results": 50
}
```

**Source Types**:
- `doi`: Single paper by DOI
- `query`: Multiple papers by search query

**Response**: `202 Accepted`
```json
{
  "status": "completed",
  "paper_ids": ["uuid1", "uuid2"],
  "count": 2
}
```

### GET /literature/{paper_id}/similar

Find similar papers.

**Path Parameters**:
- `paper_id` (string): Paper ID

**Query Parameters**:
- `top_k` (int, optional): Number of results (default: 10)

**Response**: `200 OK`
```json
[
  {
    "document_id": "string",
    "title": "string",
    "abstract": "string",
    "score": 0.88,
    "metadata": {...},
    "highlights": [...]
  }
]
```

---

## Hypotheses API

### POST /projects/{project_id}/hypotheses/generate

Generate hypotheses for a project.

**Path Parameters**:
- `project_id` (uuid): Project ID

**Request Body**:
```json
{
  "research_question": "string",
  "num_hypotheses": 5,
  "creativity_level": 0.7,
  "literature_context": ["paper_id_1", "paper_id_2"]
}
```

**Parameters**:
- `num_hypotheses` (int): Number to generate (1-10)
- `creativity_level` (float): Temperature for LLM (0.1-1.0)
- `literature_context` (list, optional): Paper IDs for context

**Response**: `202 Accepted`
```json
{
  "status": "completed",
  "project_id": "uuid",
  "hypotheses_generated": 5,
  "hypothesis_ids": ["uuid1", "uuid2", "uuid3", "uuid4", "uuid5"]
}
```

### POST /hypotheses/{hypothesis_id}/validate

Validate hypothesis for novelty and testability.

**Path Parameters**:
- `hypothesis_id` (uuid): Hypothesis ID

**Response**: `200 OK`
```json
{
  "hypothesis_id": "uuid",
  "novelty_score": 0.85,
  "testability_score": 0.90,
  "similar_hypotheses": [
    {
      "hypothesis": "string",
      "similarity": 0.65
    }
  ],
  "assessment": {
    "is_novel": true,
    "is_testable": true,
    "reasoning": "string"
  },
  "suggested_methods": ["method1", "method2"]
}
```

---

## Experiments API

### POST /hypotheses/{hypothesis_id}/experiments/design

Design an experiment for a hypothesis.

**Path Parameters**:
- `hypothesis_id` (uuid): Hypothesis ID

**Request Body**:
```json
{
  "research_question": "string",
  "hypothesis_content": "string",
  "desired_power": 0.8,
  "significance_level": 0.05,
  "expected_effect_size": 0.5,
  "constraints": {
    "max_participants": 200,
    "max_duration_days": 90,
    "budget": 50000
  },
  "experimental_approach": "controlled laboratory experiment"
}
```

**Parameters**:
- `desired_power` (float): Statistical power (0.5-0.99, default: 0.8)
- `significance_level` (float): Alpha (0.01-0.1, default: 0.05)
- `expected_effect_size` (float): Cohen's d (0.1-2.0, optional)
- `constraints` (object, optional): Resource constraints
- `experimental_approach` (string, optional): Suggested approach

**Response**: `201 Created`
```json
{
  "experiment_id": "uuid",
  "title": "string",
  "protocol": "Detailed step-by-step protocol...",
  "sample_size": 88,
  "power": 0.8,
  "effect_size": 0.5,
  "significance_level": 0.05,
  "estimated_duration": "6 weeks",
  "resource_requirements": {
    "participants": 88,
    "materials": ["material1", "material2"],
    "equipment": ["equipment1"]
  },
  "suggested_methods": ["method1", "method2"],
  "potential_confounds": ["confound1", "confound2"],
  "mitigation_strategies": ["strategy1", "strategy2"]
}
```

### POST /experiments/{experiment_id}/analyze

Analyze experimental data.

**Path Parameters**:
- `experiment_id` (uuid): Experiment ID

**Request Body**:
```json
{
  "data": {
    "records": [
      {"group": "control", "variable1": 45.2, "variable2": 12.3},
      {"group": "treatment", "variable1": 68.5, "variable2": 15.1}
    ]
  },
  "analysis_types": ["descriptive", "inferential", "effect_size"],
  "visualization_types": ["distribution", "comparison", "correlation"]
}
```

**Data Formats**:
- Records format (list of dicts)
- Column format (dict of lists)

**Analysis Types**:
- `descriptive`: Mean, median, std, quartiles
- `inferential`: t-test, ANOVA
- `effect_size`: Cohen's d

**Visualization Types**:
- `distribution`: Histograms
- `comparison`: Box plots
- `correlation`: Heatmaps

**Response**: `200 OK`
```json
{
  "experiment_id": "uuid",
  "descriptive_statistics": {
    "variable1": {
      "mean": 56.8,
      "median": 55.0,
      "std": 12.3,
      "min": 35.0,
      "max": 85.0,
      "q25": 48.0,
      "q75": 65.0,
      "count": 100
    }
  },
  "statistical_tests": [
    {
      "test_name": "Independent t-test",
      "statistic": 4.52,
      "p_value": 0.0001,
      "degrees_of_freedom": 98,
      "confidence_interval": [5.2, 18.7],
      "effect_size": 0.65,
      "interpretation": "Significant difference between groups (p=0.0001)"
    }
  ],
  "visualizations": [
    {
      "visualization_type": "comparison",
      "url": "data:image/png;base64,iVBOR...",
      "description": "Box plots comparing groups",
      "format": "png"
    }
  ],
  "overall_interpretation": "The results show a statistically significant...",
  "confidence_level": 0.95,
  "recommendations": [
    "Significant differences were found. Consider replication studies."
  ],
  "limitations": [
    "Sample size limitations",
    "Potential confounding variables"
  ]
}
```

### POST /power-analysis

Calculate statistical power or required sample size.

**Request Body**:
```json
{
  "effect_size": 0.5,
  "sample_size": 64,
  "power": 0.8,
  "significance_level": 0.05,
  "test_type": "two_sample_t"
}
```

**Required Parameters** (provide either option):
- Option 1: `effect_size` + `sample_size` → calculates `power`
- Option 2: `effect_size` + `power` → calculates `sample_size`

**Response**: `200 OK`
```json
{
  "effect_size": 0.5,
  "sample_size": 64,
  "power": 0.8,
  "significance_level": 0.05,
  "recommendation": "To achieve 80% power with effect size 0.5, you need 64 participants per group."
}
```

### GET /experiments/{experiment_id}

Get experiment details.

**Path Parameters**:
- `experiment_id` (uuid): Experiment ID

**Response**: `200 OK`
```json
{
  "id": "uuid",
  "hypothesis_id": "uuid",
  "title": "string",
  "protocol": "string",
  "status": "designed",
  "sample_size": 88,
  "power": 0.8,
  "effect_size": 0.5,
  "significance_level": 0.05,
  "results_summary": "string",
  "statistical_results": "{...}",
  "visualization_urls": "[...]",
  "interpretation": "string",
  "created_at": "2025-10-05T00:00:00Z",
  "updated_at": "2025-10-05T00:00:00Z"
}
```

---

## Common Patterns

### Pagination

Most list endpoints support pagination:
```
GET /api/v1/projects?skip=0&limit=20
```

### Filtering

Use query parameters for filtering:
```
GET /api/v1/projects?status=active
```

### Async Tasks

Some operations return 202 Accepted for async processing:
```json
{
  "status": "pending",
  "task_id": "celery-task-uuid"
}
```

Check task status:
```
GET /api/v1/tasks/{task_id}
```

---

## Error Responses

### 400 Bad Request
Invalid request parameters.

```json
{
  "detail": "Validation error",
  "errors": [
    {
      "loc": ["body", "field_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 404 Not Found
Resource not found.

```json
{
  "detail": "Project not found"
}
```

### 500 Internal Server Error
Server error.

```json
{
  "detail": "Internal server error: <error message>"
}
```

---

## Rate Limiting

**Current**: 100 requests per minute (configurable)

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1633000000
```

---

## Examples

### Complete Research Workflow

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000/api/v1")

# 1. Create project
project = client.post("/projects", json={
    "name": "Enzyme Kinetics Study",
    "description": "Temperature effects",
    "domain": "Biochemistry",
    "research_question": "How does temperature affect enzyme activity?"
}).json()

# 2. Ingest literature
client.post("/literature/ingest", json={
    "source_type": "query",
    "source_value": "enzyme kinetics temperature",
    "max_results": 50
})

# 3. Search literature
papers = client.post("/literature/search", json={
    "query": "enzyme activity temperature",
    "search_type": "hybrid",
    "top_k": 10
}).json()

# 4. Generate hypotheses
hypotheses = client.post(
    f"/projects/{project['id']}/hypotheses/generate",
    json={
        "research_question": "How does temperature affect enzyme activity?",
        "num_hypotheses": 5,
        "creativity_level": 0.7
    }
).json()

# 5. Design experiment
experiment = client.post(
    f"/hypotheses/{hypotheses['hypothesis_ids'][0]}/experiments/design",
    json={
        "research_question": "How does temperature affect enzyme activity?",
        "hypothesis_content": "Enzyme activity peaks at 37°C",
        "desired_power": 0.8
    }
).json()

# 6. Analyze data
analysis = client.post(
    f"/experiments/{experiment['experiment_id']}/analyze",
    json={
        "data": {
            "records": [
                {"group": "25C", "activity": 45.2},
                {"group": "37C", "activity": 68.5},
                {"group": "45C", "activity": 52.1}
            ]
        }
    }
).json()

print(f"Interpretation: {analysis['overall_interpretation']}")
```

---

**Navigation**: [Documentation Index](./INDEX.md) | [Architecture](./ARCHITECTURE.md) | [Development](./DEVELOPMENT.md)

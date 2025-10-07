# AI CoScientist REST API Specification

## 🎯 API Overview

**Base URL**: `https://api.ai-coscientist.com/v1`
**Authentication**: Bearer Token (JWT)
**Content-Type**: `application/json`

## 🔐 Authentication

### POST /auth/login
로그인 및 JWT 토큰 발급

**Request**:
```json
{
  "email": "researcher@university.edu",
  "password": "securepassword123"
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "id": "user-123",
    "email": "researcher@university.edu",
    "name": "Dr. Jane Doe",
    "role": "researcher"
  }
}
```

### POST /auth/refresh
액세스 토큰 갱신

**Request**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}
```

---

## 📋 Projects API

### GET /projects
프로젝트 목록 조회

**Query Parameters**:
- `status`: (optional) `active`, `completed`, `archived`
- `page`: (optional, default: 1)
- `limit`: (optional, default: 20)

**Response** (200 OK):
```json
{
  "projects": [
    {
      "id": "proj-123",
      "name": "Drug Discovery for Alzheimer's",
      "description": "Investigating novel compounds for AD treatment",
      "domain": "pharmacology",
      "status": "active",
      "created_at": "2024-01-15T10:00:00Z",
      "updated_at": "2024-01-20T15:30:00Z",
      "stats": {
        "hypotheses_count": 15,
        "experiments_count": 8,
        "papers_count": 2
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 45,
    "pages": 3
  }
}
```

### POST /projects
새 프로젝트 생성

**Request**:
```json
{
  "name": "CRISPR Gene Editing Optimization",
  "description": "Improving CRISPR-Cas9 efficiency",
  "domain": "molecular_biology",
  "research_question": "How can we reduce off-target effects in CRISPR?"
}
```

**Response** (201 Created):
```json
{
  "id": "proj-456",
  "name": "CRISPR Gene Editing Optimization",
  "description": "Improving CRISPR-Cas9 efficiency",
  "domain": "molecular_biology",
  "research_question": "How can we reduce off-target effects in CRISPR?",
  "status": "active",
  "created_at": "2024-01-21T12:00:00Z",
  "updated_at": "2024-01-21T12:00:00Z"
}
```

### GET /projects/{project_id}
프로젝트 상세 조회

**Response** (200 OK):
```json
{
  "id": "proj-123",
  "name": "Drug Discovery for Alzheimer's",
  "description": "Investigating novel compounds for AD treatment",
  "domain": "pharmacology",
  "research_question": "What novel compounds can inhibit amyloid-beta aggregation?",
  "status": "active",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-20T15:30:00Z",
  "hypotheses": [
    {
      "id": "hyp-789",
      "content": "Compound X inhibits beta-amyloid aggregation via novel mechanism",
      "novelty_score": 0.85,
      "status": "validated"
    }
  ],
  "experiments": [
    {
      "id": "exp-321",
      "hypothesis_id": "hyp-789",
      "title": "In vitro testing of Compound X",
      "status": "completed",
      "results_summary": "70% reduction in aggregation"
    }
  ],
  "papers": [
    {
      "id": "paper-654",
      "title": "Novel Inhibitors of Amyloid-Beta Aggregation",
      "status": "draft",
      "version": 2
    }
  ]
}
```

### PATCH /projects/{project_id}
프로젝트 업데이트

**Request**:
```json
{
  "status": "completed",
  "description": "Updated description with final outcomes"
}
```

**Response** (200 OK):
```json
{
  "id": "proj-123",
  "name": "Drug Discovery for Alzheimer's",
  "description": "Updated description with final outcomes",
  "status": "completed",
  "updated_at": "2024-02-01T10:00:00Z"
}
```

---

## 💡 Hypotheses API

### GET /projects/{project_id}/hypotheses
프로젝트의 가설 목록

**Response** (200 OK):
```json
{
  "hypotheses": [
    {
      "id": "hyp-789",
      "project_id": "proj-123",
      "content": "Compound X inhibits beta-amyloid aggregation",
      "rationale": "Previous studies show similar compounds affect protein folding",
      "novelty_score": 0.85,
      "testability_score": 0.92,
      "status": "validated",
      "created_at": "2024-01-16T09:00:00Z",
      "validation": {
        "is_novel": true,
        "is_testable": true,
        "feasibility": "high",
        "expected_impact": "high"
      }
    }
  ]
}
```

### POST /projects/{project_id}/hypotheses/generate
AI 기반 가설 생성

**Request**:
```json
{
  "research_question": "How to reduce CRISPR off-target effects?",
  "literature_context": [
    "paper-id-1",
    "paper-id-2"
  ],
  "num_hypotheses": 5,
  "creativity_level": 0.8
}
```

**Response** (202 Accepted):
```json
{
  "task_id": "task-abc123",
  "status": "processing",
  "message": "Hypothesis generation in progress",
  "estimated_time_seconds": 30
}
```

### GET /tasks/{task_id}
비동기 작업 상태 조회

**Response** (200 OK):
```json
{
  "task_id": "task-abc123",
  "status": "completed",
  "result": {
    "hypotheses": [
      {
        "content": "Modified guide RNA structure reduces off-target binding",
        "rationale": "Structural modifications can increase specificity",
        "novelty_score": 0.78,
        "testability_score": 0.85
      },
      {
        "content": "Cas9 protein engineering enhances target discrimination",
        "rationale": "Protein variants with higher fidelity exist in nature",
        "novelty_score": 0.65,
        "testability_score": 0.90
      }
    ]
  },
  "completed_at": "2024-01-16T09:01:30Z"
}
```

### POST /hypotheses/{hypothesis_id}/validate
가설 검증 (novelty check)

**Response** (200 OK):
```json
{
  "hypothesis_id": "hyp-789",
  "validation": {
    "is_novel": true,
    "novelty_score": 0.85,
    "similar_work": [
      {
        "paper_id": "paper-999",
        "title": "Related work on protein folding",
        "similarity": 0.62,
        "difference": "Our approach uses different compound class"
      }
    ],
    "is_testable": true,
    "testability_score": 0.92,
    "feasibility": "high",
    "suggested_methods": [
      "In vitro aggregation assay",
      "Fluorescence spectroscopy",
      "Electron microscopy"
    ]
  }
}
```

---

## 🔬 Experiments API

### POST /projects/{project_id}/experiments/design
실험 설계 자동 생성

**Request**:
```json
{
  "hypothesis_id": "hyp-789",
  "experimental_constraints": {
    "budget": 50000,
    "timeline_weeks": 12,
    "available_equipment": ["PCR", "spectrophotometer", "centrifuge"],
    "safety_level": "BSL-2"
  },
  "statistical_power": 0.8,
  "significance_level": 0.05
}
```

**Response** (201 Created):
```json
{
  "experiment_id": "exp-321",
  "hypothesis_id": "hyp-789",
  "design": {
    "title": "In Vitro Testing of Compound X on Amyloid-Beta Aggregation",
    "objective": "Test inhibitory effect of Compound X on beta-amyloid aggregation",
    "variables": {
      "independent": [
        {
          "name": "Compound X concentration",
          "type": "continuous",
          "range": "0-100 µM",
          "levels": [0, 10, 25, 50, 100]
        }
      ],
      "dependent": [
        {
          "name": "Aggregation rate",
          "type": "continuous",
          "measurement": "Thioflavin T fluorescence",
          "unit": "AU/min"
        }
      ],
      "controlled": [
        "Temperature (37°C)",
        "pH (7.4)",
        "Incubation time (24h)"
      ]
    },
    "methodology": {
      "sample_size": 30,
      "groups": [
        {
          "name": "Control",
          "n": 6,
          "treatment": "Vehicle only"
        },
        {
          "name": "Low dose",
          "n": 6,
          "treatment": "10 µM Compound X"
        }
      ],
      "randomization": "Complete randomization",
      "blinding": "Double-blind",
      "replicates": 3
    },
    "procedure": [
      "Prepare beta-amyloid peptide solution (50 µM)",
      "Add Compound X at specified concentrations",
      "Incubate at 37°C with agitation",
      "Measure ThT fluorescence at 0, 6, 12, 24h",
      "Record fluorescence intensity at 482nm"
    ],
    "data_analysis": {
      "primary_analysis": "One-way ANOVA with post-hoc Tukey test",
      "secondary_analysis": "Dose-response curve fitting",
      "software": "GraphPad Prism / R"
    },
    "statistical_power": 0.82,
    "expected_effect_size": 0.65,
    "timeline": {
      "preparation": "1 week",
      "execution": "2 weeks",
      "analysis": "1 week",
      "total_weeks": 4
    },
    "estimated_cost": 8500,
    "ethics_approval_required": false,
    "safety_considerations": [
      "Handle beta-amyloid with care (aggregation hazard)",
      "Use appropriate PPE",
      "Dispose of solutions properly"
    ]
  }
}
```

### GET /experiments/{experiment_id}
실험 상세 조회

**Response** (200 OK):
```json
{
  "id": "exp-321",
  "project_id": "proj-123",
  "hypothesis_id": "hyp-789",
  "title": "In Vitro Testing of Compound X",
  "status": "in_progress",
  "design": { /* ... full design object ... */ },
  "execution": {
    "started_at": "2024-01-18T09:00:00Z",
    "progress": 0.60,
    "current_step": "Data collection - 24h timepoint",
    "notes": "All groups responding as expected"
  },
  "results": null,
  "created_at": "2024-01-17T10:00:00Z"
}
```

### POST /experiments/{experiment_id}/results
실험 결과 입력

**Request**:
```json
{
  "raw_data": {
    "measurements": [
      {
        "group": "Control",
        "replicate": 1,
        "timepoint_h": 24,
        "fluorescence_AU": 1250
      },
      {
        "group": "Low dose",
        "replicate": 1,
        "timepoint_h": 24,
        "fluorescence_AU": 450
      }
    ]
  },
  "observations": "Dose-dependent inhibition observed",
  "data_file_url": "s3://bucket/experiment-321/data.csv"
}
```

**Response** (200 OK):
```json
{
  "experiment_id": "exp-321",
  "status": "completed",
  "results": {
    "summary": "Compound X shows dose-dependent inhibition",
    "key_findings": [
      "70% reduction in aggregation at 50 µM",
      "IC50 = 35 µM",
      "Statistically significant (p < 0.001)"
    ],
    "statistical_analysis": {
      "test": "One-way ANOVA",
      "f_statistic": 45.2,
      "p_value": 0.0001,
      "effect_size": 0.68
    },
    "visualizations": [
      {
        "type": "bar_chart",
        "url": "/api/experiments/exp-321/figures/bar_chart.png",
        "caption": "Aggregation rates by treatment group"
      },
      {
        "type": "dose_response",
        "url": "/api/experiments/exp-321/figures/dose_response.png",
        "caption": "Dose-response curve for Compound X"
      }
    ]
  },
  "interpretation": {
    "hypothesis_supported": true,
    "confidence": 0.95,
    "limitations": [
      "In vitro only - needs in vivo validation",
      "Single peptide tested"
    ],
    "next_steps": [
      "Test in cell culture model",
      "Investigate mechanism of action",
      "Test related compounds"
    ]
  }
}
```

### POST /experiments/{experiment_id}/analyze
AI 기반 데이터 분석

**Request**:
```json
{
  "data_file_url": "s3://bucket/experiment-321/data.csv",
  "analysis_type": "comprehensive",
  "include_visualizations": true
}
```

**Response** (202 Accepted):
```json
{
  "task_id": "task-xyz789",
  "status": "processing",
  "message": "Data analysis in progress"
}
```

---

## 📝 Papers API

### POST /projects/{project_id}/papers/generate
논문 자동 생성

**Request**:
```json
{
  "paper_type": "research_article",
  "target_journal": "Nature Communications",
  "sections": ["introduction", "methods", "results", "discussion"],
  "hypothesis_ids": ["hyp-789"],
  "experiment_ids": ["exp-321", "exp-322"],
  "writing_style": "concise",
  "word_limit": 5000
}
```

**Response** (202 Accepted):
```json
{
  "task_id": "task-paper123",
  "status": "processing",
  "estimated_time_minutes": 10
}
```

### GET /papers/{paper_id}
논문 조회

**Response** (200 OK):
```json
{
  "id": "paper-654",
  "project_id": "proj-123",
  "title": "Novel Small Molecule Inhibitors of Amyloid-Beta Aggregation",
  "abstract": "Alzheimer's disease is characterized by...",
  "status": "draft",
  "version": 3,
  "sections": {
    "introduction": {
      "content": "Alzheimer's disease (AD) affects...",
      "word_count": 800,
      "citations": ["ref-1", "ref-2"]
    },
    "methods": {
      "content": "Compound X was synthesized...",
      "word_count": 1200,
      "citations": ["ref-5", "ref-6"]
    },
    "results": {
      "content": "Treatment with Compound X resulted in...",
      "word_count": 1500,
      "figures": ["fig-1", "fig-2"],
      "tables": ["table-1"]
    },
    "discussion": {
      "content": "Our findings demonstrate...",
      "word_count": 1000,
      "citations": ["ref-10", "ref-11"]
    }
  },
  "figures": [
    {
      "id": "fig-1",
      "caption": "Dose-response curve for Compound X",
      "url": "/api/papers/paper-654/figures/fig1.png"
    }
  ],
  "tables": [
    {
      "id": "table-1",
      "caption": "Summary of experimental results",
      "data": [/* ... */]
    }
  ],
  "references": [
    {
      "id": "ref-1",
      "citation": "Smith et al., Nature 2020",
      "doi": "10.1038/s41586-020-xxxx"
    }
  ],
  "metadata": {
    "total_word_count": 4500,
    "citation_count": 45,
    "created_at": "2024-01-22T10:00:00Z",
    "last_edited": "2024-01-25T15:30:00Z"
  }
}
```

### PATCH /papers/{paper_id}
논문 편집

**Request**:
```json
{
  "sections": {
    "introduction": {
      "content": "Updated introduction text..."
    }
  }
}
```

**Response** (200 OK):
```json
{
  "id": "paper-654",
  "version": 4,
  "updated_at": "2024-01-25T16:00:00Z"
}
```

### POST /papers/{paper_id}/review
AI 피어 리뷰 시뮬레이션

**Request**:
```json
{
  "review_aspects": [
    "methodology",
    "statistical_analysis",
    "novelty",
    "clarity",
    "citations"
  ],
  "strictness_level": "high"
}
```

**Response** (200 OK):
```json
{
  "paper_id": "paper-654",
  "review": {
    "overall_score": 7.5,
    "recommendation": "accept_with_minor_revisions",
    "strengths": [
      "Novel approach to inhibiting aggregation",
      "Rigorous experimental design",
      "Clear presentation of results"
    ],
    "weaknesses": [
      "Limited mechanistic insight",
      "Only in vitro validation",
      "Some citations outdated"
    ],
    "detailed_comments": {
      "methodology": {
        "score": 8,
        "comments": "Methods are sound and well-described. Consider adding controls for specificity."
      },
      "statistical_analysis": {
        "score": 9,
        "comments": "Appropriate statistical tests used. Power analysis is adequate."
      },
      "novelty": {
        "score": 7,
        "comments": "Compound is novel but approach is incremental. Discuss relationship to previous work more clearly."
      },
      "clarity": {
        "score": 8,
        "comments": "Well-written overall. Some figures could be clearer."
      }
    },
    "suggestions": [
      "Add mechanistic studies (e.g., binding assay)",
      "Include cell culture validation",
      "Update citations to include recent 2024 papers",
      "Improve figure 2 resolution"
    ],
    "estimated_revision_time": "2 weeks"
  }
}
```

### POST /papers/{paper_id}/export
논문 내보내기

**Request**:
```json
{
  "format": "latex",
  "include_figures": true,
  "citation_style": "nature"
}
```

**Response** (200 OK):
```json
{
  "download_url": "https://storage.ai-coscientist.com/exports/paper-654.zip",
  "expires_at": "2024-01-26T10:00:00Z",
  "contents": [
    "manuscript.tex",
    "references.bib",
    "figures/fig1.eps",
    "figures/fig2.eps"
  ]
}
```

---

## 📚 Literature/Knowledge Base API

### GET /literature/search
문헌 검색

**Query Parameters**:
- `q`: Search query (required)
- `filters`: JSON object with filters
- `page`: Page number
- `limit`: Results per page (default: 20, max: 100)

**Request**:
```
GET /literature/search?q=CRISPR+off-target&filters={"year_min":2020,"fields":["molecular_biology"]}&limit=10
```

**Response** (200 OK):
```json
{
  "results": [
    {
      "id": "paper-abc123",
      "doi": "10.1038/s41586-2023-xxxxx",
      "title": "Reducing CRISPR Off-Target Effects Through Modified Guide RNAs",
      "authors": ["Smith J", "Doe A", "Johnson B"],
      "abstract": "CRISPR-Cas9 gene editing...",
      "publication_date": "2023-06-15",
      "journal": "Nature",
      "citations_count": 45,
      "relevance_score": 0.92,
      "url": "https://doi.org/10.1038/s41586-2023-xxxxx"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 234,
    "pages": 24
  }
}
```

### POST /literature/ingest
새 논문 추가 (DOI or PDF)

**Request**:
```json
{
  "source_type": "doi",
  "source_value": "10.1038/s41586-2023-xxxxx"
}
```

**Response** (201 Created):
```json
{
  "paper_id": "paper-xyz789",
  "status": "indexed",
  "indexed_at": "2024-01-21T10:30:00Z"
}
```

### GET /literature/{paper_id}/similar
유사 논문 찾기

**Response** (200 OK):
```json
{
  "similar_papers": [
    {
      "paper_id": "paper-def456",
      "title": "Another relevant paper",
      "similarity_score": 0.85,
      "reason": "Similar methodology and research question"
    }
  ]
}
```

---

## 📊 Analytics API

### GET /projects/{project_id}/analytics
프로젝트 분석 대시보드

**Response** (200 OK):
```json
{
  "project_id": "proj-123",
  "summary": {
    "hypotheses_generated": 15,
    "hypotheses_validated": 12,
    "experiments_completed": 8,
    "papers_published": 1,
    "total_cost_usd": 45000,
    "timeline_weeks": 24
  },
  "hypothesis_pipeline": {
    "generated": 15,
    "validated": 12,
    "experimental_testing": 8,
    "confirmed": 5,
    "rejected": 3
  },
  "experiment_outcomes": {
    "successful": 5,
    "inconclusive": 2,
    "failed": 1
  },
  "literature_coverage": {
    "papers_reviewed": 150,
    "citations_used": 45,
    "key_papers": ["paper-1", "paper-2"]
  },
  "ai_usage": {
    "llm_calls": 1234,
    "total_tokens": 2500000,
    "cost_usd": 125
  }
}
```

### GET /analytics/trending
연구 트렌드 분석

**Response** (200 OK):
```json
{
  "trending_topics": [
    {
      "topic": "CRISPR gene editing",
      "paper_count": 1234,
      "growth_rate": 0.35,
      "avg_citations": 15
    }
  ],
  "emerging_methods": [
    {
      "method": "Single-cell RNA sequencing",
      "adoption_rate": 0.45,
      "key_papers": ["paper-1", "paper-2"]
    }
  ]
}
```

---

## ⚠️ Error Responses

### 400 Bad Request
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid request parameters",
    "details": {
      "field": "hypothesis_id",
      "reason": "Hypothesis ID does not exist"
    }
  }
}
```

### 401 Unauthorized
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Authentication required"
  }
}
```

### 403 Forbidden
```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "Insufficient permissions to access this resource"
  }
}
```

### 404 Not Found
```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Resource not found",
    "details": {
      "resource_type": "project",
      "resource_id": "proj-999"
    }
  }
}
```

### 429 Too Many Requests
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "window_seconds": 60,
      "retry_after_seconds": 45
    }
  }
}
```

### 500 Internal Server Error
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "request_id": "req-abc123"
  }
}
```

---

## 🔄 WebSocket API

### Connection
```
wss://api.ai-coscientist.com/v1/ws?token=<jwt_token>
```

### Subscribe to Project Updates
```json
{
  "action": "subscribe",
  "channel": "project:proj-123"
}
```

### Real-time Events
```json
{
  "event": "hypothesis.generated",
  "data": {
    "project_id": "proj-123",
    "hypothesis_id": "hyp-new",
    "content": "New hypothesis generated",
    "timestamp": "2024-01-21T10:30:00Z"
  }
}
```

---

## 📝 Rate Limits

| Endpoint Type | Limit | Window |
|--------------|-------|--------|
| Authentication | 5 requests | 1 minute |
| Read operations | 100 requests | 1 minute |
| Write operations | 20 requests | 1 minute |
| AI operations | 10 requests | 1 minute |
| File uploads | 5 requests | 5 minutes |

---

## 🔐 API Key Alternative

For programmatic access without user authentication:

**Header**: `X-API-Key: <your_api_key>`

API keys can be generated in user settings and have scoped permissions.

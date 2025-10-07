# AI-CoScientist Documentation Index

**Version**: 0.1.0
**Last Updated**: 2025-10-05
**Status**: Production Ready

---

## 📚 Quick Navigation

| Category | Document | Description | Status |
|----------|----------|-------------|--------|
| **Getting Started** | [Quick Start](../QUICK_START.md) | Fast setup and basic usage | ✅ |
| **Getting Started** | [Setup Complete](../SETUP_COMPLETE.md) | Environment configuration guide | ✅ |
| **Getting Started** | [환경설정 완료](../환경설정_완료.md) | Korean setup guide | ✅ |
| **Architecture** | [Architecture Guide](./ARCHITECTURE.md) | System design and patterns | ✅ |
| **Architecture** | [Implementation Summary](../IMPLEMENTATION_SUMMARY.md) | Complete system overview | ✅ |
| **Development** | [Development Guide](./DEVELOPMENT.md) | Contributing and development | ✅ |
| **Deployment** | [Deployment Guide](./DEPLOYMENT.md) | Production deployment | ✅ |
| **Features** | [Phase 2 Complete](../PHASE2_COMPLETE.md) | Research Engine documentation | ✅ |
| **Features** | [Phase 3 Complete](../PHASE3_COMPLETE.md) | Experiment Engine documentation | ✅ |
| **Features** | [Improvements Implemented](../IMPROVEMENTS_IMPLEMENTED.md) | Performance & testing improvements | ✅ |
| **API Reference** | [API Documentation](./API_REFERENCE.md) | Complete API endpoint reference (633 lines) | ✅ |
| **Testing** | [Testing Guide](../tests/README.md) | Complete testing framework documentation | ✅ |

---

## 🏗️ Project Structure

```
AI-CoScientist/
├── docs/                          📚 Documentation
│   ├── INDEX.md                   This file
│   ├── API_REFERENCE.md           API endpoint documentation
│   ├── ARCHITECTURE.md            System architecture
│   ├── DEVELOPMENT.md             Development guide
│   └── DEPLOYMENT.md              Deployment instructions
│
├── src/                           💻 Source code
│   ├── api/v1/                    🌐 API endpoints
│   │   ├── health.py              Health check endpoint
│   │   ├── projects.py            Project management
│   │   ├── literature.py          Literature search & ingestion
│   │   ├── hypotheses.py          Hypothesis generation
│   │   └── experiments.py         Experiment design & analysis
│   │
│   ├── core/                      🔧 Core infrastructure
│   │   ├── config.py              Configuration management
│   │   ├── database.py            Database setup
│   │   ├── redis.py               Redis client
│   │   └── celery_app.py          Celery configuration
│   │
│   ├── models/                    🗄️ Database models
│   │   ├── base.py                Base model classes
│   │   ├── user.py                User model
│   │   ├── project.py             Project, Hypothesis, Experiment
│   │   └── literature.py          Literature models
│   │
│   ├── schemas/                   📋 Pydantic schemas
│   │   ├── project.py             Project schemas
│   │   ├── literature.py          Literature schemas
│   │   └── experiment.py          Experiment schemas
│   │
│   ├── services/                  ⚙️ Business logic
│   │   ├── llm/                   🤖 LLM integration
│   │   ├── knowledge_base/        📚 Vector search
│   │   ├── hypothesis/            💡 Hypothesis generation
│   │   ├── experiment/            🧪 Experiment services
│   │   └── external/              🌐 External APIs
│   │
│   ├── tasks/                     ⚡ Celery tasks
│   │   ├── experiment_tasks.py    Background experiment tasks
│   │   ├── hypothesis_tasks.py    Background hypothesis tasks
│   │   └── literature_tasks.py    Background literature tasks
│   │
│   └── main.py                    🚀 Application entry point
│
├── tests/                         🧪 Test suite
├── scripts/                       🔧 Utility scripts
├── config/                        ⚙️ Configuration files
└── docker/                        🐳 Docker configurations

```

---

## 🎯 Feature Documentation

### Phase 1: Core Infrastructure ✅
- FastAPI backend with async support
- PostgreSQL database with SQLAlchemy ORM
- Redis caching and session management
- Multi-LLM support (OpenAI GPT-4 + Anthropic Claude)
- JWT authentication
- Prometheus metrics

**Documentation**: See [Architecture Guide](./ARCHITECTURE.md)

### Phase 2: Research Engine ✅
- **Vector Storage**: ChromaDB with SciBERT embeddings
- **Literature Search**: Semantic, keyword, and hybrid search
- **Literature Ingestion**: Semantic Scholar + CrossRef integration
- **Hypothesis Generation**: LLM-powered with novelty scoring
- **Knowledge Base**: Persistent vector storage

**Documentation**: See [PHASE2_COMPLETE.md](../PHASE2_COMPLETE.md)

### Phase 3: Experiment Engine ✅
- **Experiment Design**: Protocol generation with power analysis
- **Sample Size Calculation**: Statistical power optimization
- **Data Analysis**: Descriptive + inferential statistics
- **Visualizations**: Automatic graph generation
- **Result Interpretation**: AI-powered analysis

**Documentation**: See [PHASE3_COMPLETE.md](../PHASE3_COMPLETE.md)

---

## 📖 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoint Categories

| Category | Endpoints | Documentation |
|----------|-----------|---------------|
| **Health** | `/health` | [Health API](./API_REFERENCE.md#health) |
| **Projects** | `/projects/*` | [Projects API](./API_REFERENCE.md#projects) |
| **Literature** | `/literature/*` | [Literature API](./API_REFERENCE.md#literature) |
| **Hypotheses** | `/hypotheses/*` | [Hypotheses API](./API_REFERENCE.md#hypotheses) |
| **Experiments** | `/experiments/*` | [Experiments API](./API_REFERENCE.md#experiments) |

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔧 Development Resources

### Setup & Configuration
- [Quick Start Guide](../QUICK_START.md) - Get started in minutes
- [Setup Complete](../SETUP_COMPLETE.md) - Detailed setup instructions
- [Environment Configuration](./DEVELOPMENT.md#environment) - .env file reference

### Development Workflow
- [Development Guide](./DEVELOPMENT.md) - Coding standards and patterns
- [Testing Guide](./DEVELOPMENT.md#testing) - Unit and integration tests
- [Contributing Guidelines](./DEVELOPMENT.md#contributing) - How to contribute

### Deployment
- [Deployment Guide](./DEPLOYMENT.md) - Production deployment
- [Docker Guide](./DEPLOYMENT.md#docker) - Containerization
- [Scaling Guide](./DEPLOYMENT.md#scaling) - Performance optimization

---

## 🏛️ Architecture Documentation

### System Design
- [Architecture Overview](./ARCHITECTURE.md) - High-level system design
- [Database Schema](./ARCHITECTURE.md#database) - Data model relationships
- [Service Layer](./ARCHITECTURE.md#services) - Business logic organization
- [API Design](./ARCHITECTURE.md#api) - RESTful API patterns

### Technical Specifications
- [LLM Integration](./ARCHITECTURE.md#llm) - Multi-provider LLM strategy
- [Vector Search](./ARCHITECTURE.md#vector-search) - ChromaDB + SciBERT
- [Background Tasks](./ARCHITECTURE.md#celery) - Celery task queue
- [Caching Strategy](./ARCHITECTURE.md#caching) - Redis caching patterns

---

## 📊 Component Reference

### Core Services

#### LLM Service (`src/services/llm/`)
Multi-provider LLM integration with caching and fallback support.

**Key Components**:
- `service.py` - Main LLM service interface
- `adapters/openai.py` - OpenAI GPT-4 integration
- `adapters/anthropic.py` - Anthropic Claude integration
- `cache.py` - Redis-based response caching
- `usage_tracker.py` - Token usage monitoring

**Documentation**: [LLM Service Guide](./ARCHITECTURE.md#llm-service)

#### Knowledge Base Service (`src/services/knowledge_base/`)
Vector-based literature search and management.

**Key Components**:
- `vector_store.py` - ChromaDB wrapper
- `embedding.py` - SciBERT embeddings
- `search.py` - Search service (semantic/keyword/hybrid)
- `ingestion.py` - Literature ingestion pipeline

**Documentation**: [Knowledge Base Guide](./ARCHITECTURE.md#knowledge-base)

#### Experiment Service (`src/services/experiment/`)
Experiment design and data analysis.

**Key Components**:
- `design.py` - Protocol generation + power analysis
- `analysis.py` - Statistical testing + visualization

**Documentation**: [Experiment Service Guide](./ARCHITECTURE.md#experiment-service)

#### Hypothesis Service (`src/services/hypothesis/`)
AI-powered hypothesis generation and validation.

**Key Components**:
- `generator.py` - Hypothesis generation + validation

**Documentation**: [Hypothesis Service Guide](./ARCHITECTURE.md#hypothesis-service)

### API Endpoints

#### Projects API (`src/api/v1/projects.py`)
Research project management.

**Endpoints**:
- `POST /projects` - Create project
- `GET /projects` - List projects
- `GET /projects/{id}` - Get project details
- `PUT /projects/{id}` - Update project
- `DELETE /projects/{id}` - Delete project

#### Literature API (`src/api/v1/literature.py`)
Scientific literature search and ingestion.

**Endpoints**:
- `POST /literature/search` - Search papers
- `POST /literature/ingest` - Ingest papers
- `GET /literature/{id}/similar` - Find similar papers

#### Hypotheses API (`src/api/v1/hypotheses.py`)
Hypothesis generation and validation.

**Endpoints**:
- `POST /projects/{id}/hypotheses/generate` - Generate hypotheses
- `POST /hypotheses/{id}/validate` - Validate hypothesis

#### Experiments API (`src/api/v1/experiments.py`)
Experiment design and data analysis.

**Endpoints**:
- `POST /hypotheses/{id}/experiments/design` - Design experiment
- `POST /experiments/{id}/analyze` - Analyze data
- `POST /power-analysis` - Calculate power/sample size
- `GET /experiments/{id}` - Get experiment details

### Database Models

#### Project Model (`src/models/project.py`)
```python
Project (id, name, description, domain, research_question, status)
├── Hypothesis (id, project_id, content, novelty_score, status)
│   └── Experiment (id, hypothesis_id, title, protocol, sample_size, power)
└── Paper (id, project_id, title, abstract, content, status)
```

#### Literature Model (`src/models/literature.py`)
```python
Literature (id, doi, title, abstract, publication_date, citations_count)
├── Author (id, name, affiliation)
└── FieldOfStudy (id, name)
```

---

## 🔍 Cross-References

### Workflow Integration

**Complete Research Pipeline**:
```
1. Create Project (Projects API)
   ↓
2. Ingest Literature (Literature API)
   ↓
3. Search Papers (Literature API)
   ↓
4. Generate Hypotheses (Hypotheses API)
   ↓
5. Validate Hypothesis (Hypotheses API)
   ↓
6. Design Experiment (Experiments API)
   ↓
7. Analyze Data (Experiments API)
   ↓
8. Interpret Results (AI-powered)
```

### Service Dependencies

```
Experiments Service
├── LLM Service (protocol generation)
├── Knowledge Base (methodology search)
└── Database (experiment storage)

Hypotheses Service
├── LLM Service (hypothesis generation)
├── Knowledge Base (literature context)
└── Database (hypothesis storage)

Literature Service
├── External APIs (Semantic Scholar, CrossRef)
├── Vector Store (embeddings)
├── Embedding Service (SciBERT)
└── Database (metadata storage)
```

---

## 📚 Learning Resources

### Tutorials
1. [Getting Started Tutorial](../QUICK_START.md) - First steps
2. [Research Workflow Example](../QUICK_START.md#complete-research-workflow) - End-to-end example
3. [API Usage Examples](./API_REFERENCE.md#examples) - Code samples

### Concepts
- [Vector Search Explained](./ARCHITECTURE.md#vector-search) - Semantic search
- [Power Analysis Guide](../PHASE3_COMPLETE.md#power-analysis) - Statistical concepts
- [LLM Integration Patterns](./ARCHITECTURE.md#llm-patterns) - Multi-provider strategy
- [Async Task Processing](./ARCHITECTURE.md#celery) - Background jobs

### Best Practices
- [API Design Patterns](./DEVELOPMENT.md#api-patterns) - RESTful best practices
- [Error Handling](./DEVELOPMENT.md#error-handling) - Exception management
- [Testing Strategies](./DEVELOPMENT.md#testing) - Unit + integration tests
- [Performance Optimization](./DEVELOPMENT.md#performance) - Scaling tips

---

## 🛠️ Maintenance

### Regular Tasks
- **Database Migrations**: `poetry run alembic upgrade head`
- **Dependency Updates**: `poetry update`
- **Docker Cleanup**: `docker system prune`
- **Log Rotation**: Configure in production

### Monitoring
- **Health Check**: `GET /api/v1/health`
- **Metrics**: Prometheus endpoint (if enabled)
- **Logs**: `docker-compose logs -f`
- **Database**: Connection pool monitoring

### Backup & Recovery
- **Database Backup**: PostgreSQL backup procedures
- **Vector Store**: ChromaDB persistence
- **Redis**: RDB/AOF persistence configuration

---

## 📞 Support & Community

### Getting Help
- **Documentation**: Start with [Quick Start](../QUICK_START.md)
- **Troubleshooting**: See [QUICK_START.md#troubleshooting](../QUICK_START.md#troubleshooting)
- **Issues**: GitHub issue tracker (if applicable)

### Contributing
- **Development Guide**: [DEVELOPMENT.md](./DEVELOPMENT.md)
- **Code Standards**: PEP 8, type hints, docstrings
- **Pull Requests**: Follow contribution guidelines

---

## 🗺️ Roadmap

### Current Version (0.1.0)
- ✅ Complete research pipeline
- ✅ Multi-LLM integration
- ✅ Statistical analysis
- ✅ Background task processing

### Future Enhancements
- 📄 **Phase 4**: Paper generation (optional)
- 🔬 Advanced statistical methods (regression, Bayesian)
- 📊 Interactive visualizations (Plotly)
- 🌐 Multi-language support
- 🔐 Enhanced authentication
- 📱 Web UI frontend

---

## 📄 Document Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-05 | 1.0.0 | Initial documentation index created |

---

**Navigation**: [Top](#ai-coscientist-documentation-index) | [Quick Start](../QUICK_START.md) | [API Reference](./API_REFERENCE.md) | [Architecture](./ARCHITECTURE.md)

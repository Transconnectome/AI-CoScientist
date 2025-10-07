# AI CoScientist System Architecture

## 🏗️ System Overview

AI CoScientist는 LLM 기반 자율 과학 연구 시스템으로, 마이크로서비스 아키텍처와 이벤트 주도 설계를 채택합니다.

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway Layer                         │
│                     (FastAPI + Rate Limiting)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                           │
│                  (Multi-Agent Coordinator)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Research   │    │  Experiment  │    │    Paper     │
│   Engine     │───▶│   Engine     │───▶│   Engine     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────┐
│              Shared Services Layer                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │   LLM    │  │ Knowledge│  │  Safety  │          │
│  │ Service  │  │   Base   │  │ Guardian │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└──────────────────────────────────────────────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────┐
│                 Data Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │PostgreSQL│  │  Vector  │  │  Redis   │          │
│  │   (Main) │  │   DB     │  │  Cache   │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└──────────────────────────────────────────────────────┘
```

## 🎯 Core Components

### 1. API Gateway Layer
**Responsibility**: External interface, authentication, rate limiting

```yaml
Components:
  - FastAPI Application Server
  - JWT Authentication Middleware
  - Rate Limiter (Redis-based)
  - Request/Response Logger
  - CORS Handler

Tech Stack:
  - FastAPI 0.109+
  - Pydantic v2
  - Redis 7+
  - Prometheus metrics

Key Features:
  - RESTful API endpoints
  - WebSocket support for real-time updates
  - API versioning (v1, v2)
  - OpenAPI documentation
```

### 2. Orchestration Layer
**Responsibility**: Multi-agent coordination, workflow management

```yaml
Components:
  - AgentOrchestrator: Master coordinator
  - TaskScheduler: Task queue management
  - StateManager: Workflow state tracking
  - EventBus: Inter-service communication

Design Pattern:
  - Command Pattern for agent tasks
  - Observer Pattern for event handling
  - State Machine for workflow states

Tech Stack:
  - LangGraph for agent orchestration
  - Celery for task queue
  - RabbitMQ for message broker
```

### 3. Research Engine
**Responsibility**: Hypothesis generation, literature analysis

```yaml
Modules:
  - HypothesisGenerator:
      - LiteratureAnalyzer
      - GapIdentifier
      - NoveltyChecker

  - LiteratureRetrieval:
      - PaperSearcher (Semantic Scholar API)
      - CitationExtractor
      - EmbeddingGenerator

  - KnowledgeSynthesizer:
      - CrossPaperAnalyzer
      - ConceptMapper
      - TrendAnalyzer

Data Flow:
  Research Topic → Literature Search → Embedding →
  Gap Analysis → Hypothesis Generation → Novelty Check
```

### 4. Experiment Engine
**Responsibility**: Experimental design, simulation, data analysis

```yaml
Modules:
  - ExperimentDesigner:
      - ProtocolGenerator
      - VariableOptimizer
      - StatisticalPlanner
      - EthicsValidator

  - DataAnalyzer:
      - StatisticalTester
      - VisualizationGenerator
      - ResultInterpreter
      - ReproducibilityChecker

  - SimulationRunner:
      - InSilicoExperiment
      - ParameterSweeper
      - SensitivityAnalyzer

Design Considerations:
  - Pluggable experiment types
  - Version control for experiments
  - Audit trail for reproducibility
```

### 5. Paper Engine
**Responsibility**: Scientific writing, peer review simulation

```yaml
Modules:
  - PaperWriter:
      - StructureGenerator (IMRaD format)
      - SectionWriter (Introduction, Methods, Results, Discussion)
      - CitationManager
      - FigureTableGenerator

  - StyleOptimizer:
      - AcademicToneEnforcer
      - GrammarChecker
      - ClarityEnhancer

  - PeerReviewer:
      - MethodologyValidator
      - LogicChecker
      - NoveltyAssessor
      - ImprovementSuggester

Output Formats:
  - LaTeX
  - Markdown
  - DOCX
  - PDF (via pandoc)
```

## 🔧 Shared Services

### LLM Service
**Responsibility**: Centralized LLM interaction with fallback

```yaml
Architecture:
  - Primary: OpenAI GPT-4
  - Fallback: Anthropic Claude 3.5
  - Local: Fine-tuned domain models

Features:
  - Prompt template management
  - Token usage tracking
  - Response caching
  - Error handling & retry logic

Configuration:
  - Temperature control per task type
  - Max tokens per request
  - Cost optimization strategies
```

### Knowledge Base Service
**Responsibility**: Scientific literature storage and retrieval

```yaml
Components:
  - VectorStore (ChromaDB):
      - Paper embeddings
      - Concept embeddings
      - Semantic search

  - MetadataStore (PostgreSQL):
      - Paper metadata
      - Author information
      - Citation network

  - CacheLayer (Redis):
      - Frequent queries
      - Recent papers
      - Search results

Embedding Model:
  - all-MiniLM-L6-v2 for general text
  - SciBERT for scientific content
```

### Safety Guardian
**Responsibility**: Ethical compliance, bias detection

```yaml
Modules:
  - EthicsChecker:
      - Research ethics guidelines
      - Human subject protection
      - Animal welfare rules

  - BiasDetector:
      - Dataset bias analysis
      - Result fairness check
      - Demographic parity

  - RiskAssessor:
      - Dual-use research screening
      - Misuse potential evaluation
      - Safety recommendations

Decision Framework:
  - Traffic light system (Green/Yellow/Red)
  - Human-in-the-loop for Yellow/Red
  - Audit log for all decisions
```

## 💾 Data Layer

### PostgreSQL Schema
```sql
-- Core entities
projects
  - id, name, description, status, created_at

hypotheses
  - id, project_id, content, novelty_score, status

experiments
  - id, hypothesis_id, protocol, status, results

papers
  - id, project_id, content, version, status

literature
  - id, doi, title, authors, abstract, citations
```

### Vector Database Schema
```yaml
Collections:
  - papers_embeddings
      - embedding: vector[384]
      - metadata: {doi, title, year}

  - concepts_embeddings
      - embedding: vector[384]
      - metadata: {concept, domain}

  - hypotheses_embeddings
      - embedding: vector[384]
      - metadata: {hypothesis_id, project_id}
```

### Redis Cache Strategy
```yaml
Cache Keys:
  - paper:{doi}: Paper metadata (TTL: 7 days)
  - search:{query_hash}: Search results (TTL: 1 day)
  - llm_response:{prompt_hash}: LLM outputs (TTL: 1 hour)
  - user_session:{session_id}: User state (TTL: 1 hour)
```

## 🔄 Communication Patterns

### Event-Driven Architecture
```yaml
Events:
  - hypothesis.generated
  - experiment.designed
  - experiment.completed
  - paper.drafted
  - review.completed

Event Bus: RabbitMQ
  - Topic exchanges for event routing
  - Dead letter queue for failures
  - Event replay capability
```

### API Communication
```yaml
Patterns:
  - Synchronous: REST API (request/response)
  - Asynchronous: WebSocket (real-time updates)
  - Background: Celery tasks (long-running jobs)

Protocols:
  - HTTP/2 for REST
  - WebSocket for streaming
  - gRPC for internal services (future)
```

## 🛡️ Security Architecture

### Authentication & Authorization
```yaml
Authentication:
  - JWT tokens (access + refresh)
  - OAuth2 integration (Google, GitHub)
  - API key for programmatic access

Authorization:
  - RBAC (Role-Based Access Control)
  - Roles: Admin, Researcher, Reviewer
  - Resource-level permissions
```

### Data Security
```yaml
Encryption:
  - At rest: AES-256
  - In transit: TLS 1.3
  - Sensitive fields: Column-level encryption

Compliance:
  - GDPR compliant
  - Research data protection
  - Audit logging
```

## 📊 Scalability Design

### Horizontal Scaling
```yaml
Stateless Services:
  - API Gateway: Auto-scale based on CPU
  - Research Engine: Queue-based scaling
  - Paper Engine: On-demand scaling

Stateful Services:
  - PostgreSQL: Read replicas
  - Vector DB: Sharding by domain
  - Redis: Cluster mode
```

### Performance Optimization
```yaml
Caching Strategy:
  - L1: In-memory (service-level)
  - L2: Redis (distributed)
  - L3: CDN (static assets)

Database Optimization:
  - Connection pooling
  - Query optimization
  - Materialized views
  - Partitioning by date
```

## 🔍 Observability

### Monitoring
```yaml
Metrics (Prometheus):
  - Request rate, latency, errors
  - LLM token usage and cost
  - Database query performance
  - Queue depths

Logging (ELK Stack):
  - Structured JSON logs
  - Centralized log aggregation
  - Log correlation IDs

Tracing (Jaeger):
  - Distributed tracing
  - Request flow visualization
  - Performance bottlenecks
```

### Alerting
```yaml
Alert Rules:
  - High error rate (>1%)
  - High latency (p95 > 5s)
  - Queue backlog (>1000)
  - Cost threshold exceeded
  - Safety violations detected
```

## 🚀 Deployment Architecture

### Container Strategy
```yaml
Docker Compose (Development):
  - All services in containers
  - Volume mounts for code
  - Hot reload enabled

Kubernetes (Production):
  - Namespaces: dev, staging, prod
  - HPA for auto-scaling
  - ConfigMaps for configuration
  - Secrets for sensitive data
```

### CI/CD Pipeline
```yaml
Stages:
  1. Code Quality:
      - Linting (ruff, mypy)
      - Unit tests (pytest)
      - Coverage (>80%)

  2. Build:
      - Docker image build
      - Security scanning (Trivy)
      - Image tagging

  3. Deploy:
      - Staging deployment
      - Integration tests
      - Production deployment (blue-green)

  4. Post-Deploy:
      - Smoke tests
      - Monitoring checks
      - Rollback capability
```

## 📈 Cost Optimization

### LLM Cost Management
```yaml
Strategies:
  - Prompt optimization (reduce tokens)
  - Response caching
  - Model selection by task complexity
  - Batch processing
  - Rate limiting

Estimated Costs:
  - Hypothesis generation: $0.05/hypothesis
  - Experiment design: $0.10/design
  - Paper writing: $0.50/paper
  - Total per project: ~$2-5
```

## 🎯 Quality Attributes

### Reliability
- Target uptime: 99.9%
- Graceful degradation
- Circuit breakers
- Retry mechanisms

### Maintainability
- Clean architecture
- SOLID principles
- Comprehensive tests
- Documentation

### Performance
- API response: <500ms (p95)
- Hypothesis generation: <30s
- Paper generation: <5min
- Concurrent users: 1000+

### Security
- OWASP Top 10 compliance
- Regular security audits
- Penetration testing
- Vulnerability scanning

# CLAUDE.md - AI-CoScientist: Unified Proposal Engine (UPE)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Name & Components

**Official Name**: **AI-CoScientist Unified Proposal Engine (UPE)**
> 7-Strategy RAG 기반 과학 제안서 자동 최적화 시스템

### Core Components

| 약어 | 전체 이름 | 설명 | 주요 파일 |
|-----|----------|-----|----------|
| **UPE** | Unified Proposal Engine | 전체 시스템 | - |
| **MSS** | Multi-Strategy Search | 7-전략 검색 엔진 | `src/services/rag/multi_strategy_search.py` |
| **URO** | Unified RAG Orchestrator | RAG 통합 오케스트레이터 | `src/services/rag/unified_rag_orchestrator.py` |
| **MAP** | Multi-Agent Pipeline | 6-에이전트 파이프라인 | `scripts/multi_agent_unified_pipeline.py` |

### Quick Usage

```bash
# 🎯 제안서 최적화 (권장)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --mode full --enable-cross-domain

# 🔍 Multi-Strategy 검색
poetry run python src/services/rag/multi_strategy_search.py "your query"

# 🤖 6-Agent 파이프라인
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode full_pipeline --input "proposal.md"

# 📊 성능 벤치마크
poetry run python src/monitoring/unified_performance_dashboard.py --benchmark

# 🕵️‍♀️ Reviewer V3 (ECMARS Enhanced)
poetry run python scripts/review_workflow_v2.py --paper "submission.pdf"
```

## Project Overview

AI-CoScientist is a comprehensive AI-powered scientific research automation system that transforms scientific paper analysis, improvement, and generation through **Unified RAG Orchestrator** and advanced multi-agent collaboration.

**Core Purpose**: Automates the complete research workflow from literature review through experiment design, data analysis, paper improvement, and collaborative multi-agent research coordination, with **next-generation 7-strategy RAG orchestration** spanning neuroscience, quantum ML, protein research (ESM3), and grant proposals.

**🚀 Major Architectural Upgrade (2025)**: Migrated from single DD-RAPTOR strategy to **Unified RAG Orchestrator** with intelligent 7-strategy routing (HYBRID, GRAPH_RAG, ENHANCED_DD_RAPTOR, GOLDEN_REFERENCE, MULTIMODAL_RAG, SIMPLE_RAG, PSYCHOLOGY_RAG) and cross-domain knowledge synthesis across 1,761+ documents.

## Development Commands

### Setup & Installation
```bash
# Initial setup (installs dependencies, creates .env, starts services)
./scripts/setup.sh

# Install Python dependencies only
poetry install

# Start Docker services (PostgreSQL, Redis, ChromaDB, monitoring)
docker-compose up -d

# Database migrations
poetry run alembic upgrade head
```

### Running the Application
```bash
# Start API server (development)
poetry run uvicorn src.main:app --reload

# Start Celery worker for background tasks
poetry run celery -A src.core.celery_app worker --loglevel=info

# Start Celery beat scheduler
poetry run celery -A src.core.celery_app beat --loglevel=info

# Run complete system with Docker
docker-compose up -d
```

### Testing
```bash
# Run all tests with coverage
poetry run pytest -v --cov=src --cov-report=html

# Run specific test modules
poetry run pytest tests/agents/ -v          # Agent system tests
poetry run pytest tests/rag/ -v            # RAG pipeline tests
poetry run pytest tests/monitoring/ -v      # Metrics and monitoring tests
poetry run pytest tests/integration/ -v    # Integration tests

# RAG evaluation specific tests
poetry run pytest tests/rag/test_rag_evaluation.py -v    # RAGAS evaluation framework
poetry run pytest tests/rag/test_rag_evaluator.py -v     # RAG evaluator unit tests

# Frontend tests (Phase 5)
cd frontend && npm test                     # Unit tests
cd frontend && npm run test:e2e            # E2E tests
```

### Code Quality
```bash
# Format code
poetry run black src tests

# Lint code
poetry run ruff check src tests

# Type checking
poetry run mypy src

# Run all quality checks
poetry run pre-commit run --all-files
```

### Frontend Development (Phase 5)
```bash
# Setup frontend (React + TypeScript + Vite)
./scripts/complete_frontend_setup.sh

# Start development server
cd frontend && npm run dev

# Build for production
cd frontend && npm run build
```

### RAG Evaluation & Benchmarking
```bash
# Generate automated QA benchmark dataset
python scripts/build_benchmark_dataset.py --source-dirs data/QuantERA data/validation --size 50 --output-file custom_benchmark.json

# Evaluate RAG system performance using golden benchmark
python -c "
from src.services.rag.rag_evaluator import create_rag_evaluator, evaluate_rag_pipeline
import json
import asyncio

async def evaluate_system():
    # Load benchmark dataset
    with open('data/validation/golden_qa_benchmark.json') as f:
        benchmark = json.load(f)

    # Extract sample QA pairs
    qa_pairs = benchmark['qa_pairs'][:5]  # Test with 5 pairs
    queries = [qa['question'] for qa in qa_pairs]
    contexts_list = [qa['contexts'] for qa in qa_pairs]
    answers = [qa['answer'] for qa in qa_pairs]
    ground_truths = [qa['ground_truth'] for qa in qa_pairs]

    # Run evaluation
    report = await evaluate_rag_pipeline(queries, contexts_list, answers, ground_truths)
    print('Evaluation Results:', report['summary'])

asyncio.run(evaluate_system())
"

# Monitor RAG performance with Prometheus metrics
python -c "
from src.monitoring.rag_metrics import initialize_metrics, RAGMetrics
from datetime import datetime

# Initialize metrics system (with Prometheus disabled for demo)
manager = initialize_metrics(enable_prometheus=False)

# Create sample metrics
metrics = RAGMetrics(
    latency=1.5, quality_score=0.85, tokens_processed=1200,
    retrieval_time=0.3, generation_time=1.2, context_relevance=0.9,
    faithfulness=0.8, answer_relevancy=0.87, strategy='hybrid_rag',
    timestamp=datetime.now()
)

# Record metrics and get performance summary
manager.record_rag_request(metrics)
performance = manager.get_strategy_performance('hybrid_rag')
print('Strategy Performance:', performance)
"
```

### 🚀 **Unified RAG Proposal Optimization Workflow**
For Korean scientific proposal improvement using AI-CoScientist and **Unified RAG Orchestrator**:

**📋 Quick Reference Guide**: See [`PROPOSAL_OPTIMIZATION_QUICK_REFERENCE_UNIFIED.md`](./PROPOSAL_OPTIMIZATION_QUICK_REFERENCE_UNIFIED.md) for complete usage guide.

```bash
# 🎯 Complete Unified RAG Optimization (95+ score target)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --mode full --enable-cross-domain

# ⚡ Quick Unified Improvement (85+ score target)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --mode quick --strategies "HYBRID,GRAPH_RAG"

# 🌐 Cross-Domain Synthesis (ESM3 + Neuroscience + Quantum ML)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --mode cross_domain --domains "neuroscience,protein_research,quantum_ml"

# 🧙‍♀️ Interactive Unified Wizard (Enhanced with 6-strategy selection)
poetry run python scripts/proposal_optimizer_unified.py wizard --unified-rag

# 📊 Advanced Batch Processing with Strategy Configuration
poetry run python scripts/batch_optimizer_unified.py --config unified_batch_config.yaml

# 🔍 Unified Quality Assessment (Multi-strategy evaluation)
poetry run python scripts/map_proposal_to_unified_evidence.py \
    --proposal "proposal.md" --output "assessment.json" --unified-rag --quality-assessment
```

**Enhanced 5-Stage Unified RAG Pipeline**:
1. **Unified Evidence Mapping** (`map_proposal_to_unified_evidence.py`) - Cross-domain scientific claim analysis
2. **Multi-Strategy Validation** (`validate_claims_unified_rag.py`) - 6-strategy claim verification
3. **Advanced RAG Literature Review** (`advanced_unified_query.py`) - Multi-modal systematic search
4. **Multi-Agent Unified Enhancement** (`multi_agent_unified_pipeline.py`) - 6 AI specialists + RAG integration
5. **Intelligent Unified Citation** (`unified_citation_generator.py`) - Cross-domain auto-reference generation

**Enhanced Target Outcomes**:
- **95+ Score**: Samsung Future Technology Grant 1st Grade + Cross-domain Innovation bonus
- **Multi-Domain Coverage**: ESM3 protein research + Neuroscience + Quantum ML synthesis
- **6-Strategy Validation**: >85% claims supported across HYBRID, GRAPH_RAG, GOLDEN_REFERENCE strategies
- **Cross-Modal Intelligence**: Text + Image + Table + Citation comprehensive analysis

## 🇰🇷 Korean NRF Mid-Career Researcher Support

This system provides specialized support for Korean NRF (국가연구재단) mid-career researchers with comprehensive proposal optimization, automated diagram generation, and NRF-specific compliance checking.

> **CRITICAL RULE - NO EMOJIS IN PROPOSALS**: NRF 공식 제안서에는 이모지를 절대 사용하지 않습니다. 모든 다이어그램, 표, 텍스트에서 이모지를 제외하고 공식적인 학술 스타일을 유지해야 합니다. 이 규칙은 `proposal_diagram_generator.py` 및 모든 제안서 관련 출력물에 적용됩니다.

> **CRITICAL RULE - 중견연구 제안서 작업 시 필수 참조 및 도구 활용**:
>
> 중견연구 제안서 관련 작업(Block 1-5 작성, Figure 생성, 검토, 수정 등)을 수행할 때는 **반드시** 다음을 준수해야 합니다:
>
> **📋 필수 참조 문서**:
> 1. **`data/중견/templates/PROPOSAL_MASTER_PLAN.md`** - 리뷰어 친화적 평가 전략, 평가항목별 만점 체크리스트, 블록-평가항목 매핑
> 2. **`data/중견/NRF_Midcareer_Proposal_Playbook.md`** - 섹션별 작성 가이드, 평가기준, Figure 요구사항
>
> **🛠️ 필수 활용 도구 (AI-CoScientist UPE 시스템)**:
> - **제안서 최적화**: `poetry run python scripts/proposal_optimizer_unified.py optimize --input "proposal.md" --mode full`
> - **주장 검증 (>85%)**: `poetry run python scripts/validate_claims_unified_rag.py`
> - **6-Agent 파이프라인**: `poetry run python scripts/multi_agent_unified_pipeline.py`
> - **인용 자동 생성**: `poetry run python scripts/unified_citation_generator.py`
> - **NRF 샘플 RAG 검색**: `poetry run python scripts/integrate_nrf_samples_to_upe.py --test-query "검색어"`
>
> **🎨 Figure/다이어그램 생성 시 필수 활용 파이프라인** (`data/중견/diagram_pipelines_20251215_094550/diagram_pipelines/`):
> - **Pipeline 1 (Claude + Mermaid/Code)**: 모델 아키텍처, 시스템 구조, 워크플로우 → 버전관리 가능, 학술적
> - **Pipeline 2 (Direct Image AI)**: 빠른 시각화, 프레젠테이션용 고품질 이미지
> - **Pipeline 3 (Kimi K2 Code)**: 복잡한 수학적 모델, Transformer/GAN 구조 → matplotlib/TikZ 코드 생성
>
> **⚠️ Figure 생성 시**: 이미 생성된 Mermaid 소스 (`data/중견/templates/figures/*.mmd`)가 있으면 먼저 활용하고, 새로운 Figure는 위 3개 파이프라인 중 적합한 것을 선택하여 생성합니다.
>
> **📊 평가 기준 (100점 만점)**:
> - 창의성/도전성/융합성: **40점** → Block 1 + 2
> - 방법론 적합성: **30점** → Block 3
> - 연구자 우수성: **20점** → Block 4
> - 기대효과: **10점** → Block 5
>
> 모든 작업은 **리뷰어가 각 평가항목에서 만점을 주기 쉽도록** 설계해야 합니다.

### 🚀 UPE 5-Phase Master Workflow

**체계적 제안서 작성을 위한 5단계 실행 파이프라인**:

```
Phase 1 (분석) → Phase 2 (검증) → Phase 3 (생성) → Phase 4 (최적화) → Phase 5 (완성)
   증거매핑        주장검증        콘텐츠생성      6-Agent협업       인용+품질게이트
   갭분석          RAG검색        다이어그램4종    95+점수목표       최종제출준비
```

| Phase | 목표 | 핵심 도구 | 품질 게이트 |
|-------|------|----------|------------|
| **Phase 1** | 갭 분석 & 5-Block 구조 확정 | `map_proposal_to_unified_evidence.py` | 핵심 갭 1문장 정의 |
| **Phase 2** | 주장 검증률 >85% | `validate_claims_unified_rag.py` | 미검증 주장 수정 |
| **Phase 3** | Figure 4종 + 콘텐츠 | `proposal_diagram_generator.py` | 10페이지 분량 준수 |
| **Phase 4** | 95+ 점수 달성 | `proposal_optimizer_unified.py` | 6-Agent 협업 완료 |
| **Phase 5** | 최종 제출 준비 | `unified_citation_generator.py` | "제출완료" 상태 확인 |

### Mid-Career Research Documentation

**📋 Complete Resource Hub**: `data/중견/` contains comprehensive guides for Korean mid-career researchers:

- **`README.md`** - **UPE 5-Phase Master Workflow** + Quick start guide with 3 usage paths
- **`AI_COSCIENTIST_중견연구자_온보딩_가이드.md`** - Complete AI-CoScientist onboarding with UPE system workflows
- **`NRF_Midcareer_Proposal_Playbook.md`** - Section-by-section NRF proposal writing playbook with evaluation criteria
- **`가이드라인.md`** - Comprehensive NRF mid-career research guidelines and latest policy updates
- **`샘플-*.pdf`** - Successful proposal samples across multiple domains (neuroscience, quantum research, developmental studies)

### NRF Sample Proposal RAG System

**Golden Reference RAG Strategy** for Korean grant proposal optimization using ingested NRF sample proposals.

**ChromaDB Collections**:
| Collection | Documents | Description |
|------------|-----------|-------------|
| `nrf_midcareer_samples_L0` | 659 chunks | Detailed text segments |
| `nrf_midcareer_samples_L1` | 24 sections | Section-level summaries |
| `nrf_midcareer_samples_L2` | 4 documents | Full proposal summaries |

**Quick Usage**:
```bash
# Verify integration
poetry run python scripts/integrate_nrf_samples_to_upe.py --verify

# Test RAG query
poetry run python scripts/integrate_nrf_samples_to_upe.py --test-query "연구 방법론"

# Run demo
poetry run python scripts/integrate_nrf_samples_to_upe.py --demo
```

**Python Integration**:
```python
from src.services.rag.nrf_proposal_strategy import create_nrf_proposal_strategy

nrf_rag = create_nrf_proposal_strategy()
results = await nrf_rag.search("연구 방법론 예시")
methods = await nrf_rag.search_by_section("추진전략", section_filter=['추진전략', 'Methods'])
```

**Key Files**: `src/services/rag/nrf_proposal_strategy.py`, `scripts/integrate_nrf_samples_to_upe.py`

### Quick Start for Mid-Career Researchers

**🚀 First-time users (Interactive):**
```bash
# Launch beginner-friendly interactive wizard
poetry run python scripts/proposal_wizard.py
```

**⚡ Experienced users (Full 5-Phase, 95+ target):**
```bash
# Full auto-optimization targeting 95+ score for NRF grants
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "제안서_초안.md" --mode full --enable-cross-domain --target-score 95
```

**🔬 Domain-Specific optimization:**
```bash
# Neuroscience research
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "초안.md" --domains "neuroscience" --strategies "GRAPH_RAG,ENHANCED_DD_RAPTOR"

# Protein/ESM3 research
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "초안.md" --domains "protein_research" --strategies "MULTIMODAL_RAG,GRAPH_RAG"
```

**📚 Learning path users:**
- Study `AI_COSCIENTIST_중견연구자_온보딩_가이드.md` for comprehensive workflow understanding
- Reference `NRF_Midcareer_Proposal_Playbook.md` for section-specific checklists and evaluation criteria
- **NEW**: Follow `README.md` UPE 5-Phase Master Workflow for step-by-step execution

### 📐 Automated Diagram Generation Pipelines

**Advanced Feature**: Multi-pipeline diagram generation for scientific proposals and papers.

**Location**: `data/중견/diagram_pipelines_20251215_094550/diagram_pipelines/`

**Three specialized pipelines**:

1. **Pipeline 1 - Claude + Mermaid/Code** (`pipeline_1_claude_mermaid.md`)
   - **Method**: Claude generates Mermaid, Python (matplotlib/networkx), TikZ/Graphviz code
   - **Advantages**: Version-controllable diagrams, precise academic structure representation, Git-friendly
   - **Use Case**: System architecture, model structure, reproducible workflow diagrams

2. **Pipeline 2 - Direct Image AI** (`pipeline_2_image_ai.md`)
   - **Method**: ChatGPT 4o, DALL·E 3, Gemini, Ideogram direct image generation
   - **Advantages**: Rapid visualization, presentation-ready high-quality graphics
   - **Use Case**: Initial ideation, presentation slides, style experimentation

3. **Pipeline 3 - Kimi K2 Code-based** (`pipeline_3_kimi_k2_code.md`)
   - **Method**: Kimi K2 generates complete matplotlib/Graphviz/TikZ code for execution
   - **Advantages**: Complex structure control, free tier access, coding specialization
   - **Use Case**: Mathematical models, complex architectures (Transformer, GAN, BERT, RL systems)

**Testing and Evaluation**:
```bash
# Execute comprehensive pipeline testing (follow TEST_EXECUTION_GUIDE.md)
cd "data/중견/diagram_pipelines_20251215_094550/diagram_pipelines/"

# Test all three pipelines with standard cases (Transformer, CNN, GAN, BERT, Attention)
python test_pipelines.py

# Compare multiple image generation APIs
python api_test.py
```

**Integration with Proposals**:
1. Complete text/structure using playbook/onboarding guides
2. Identify required diagram types (model structure, data flow, experimental design)
3. Select appropriate pipeline(s) for diagram generation
4. Insert generated diagrams into Word/HWP/LaTeX documents
5. Final consistency check using RAG/UPE system

### NRF-Specific Optimization Features

**Target Outcomes for Korean Grants**:
- **NRF Core Research (핵심연구)**: Optimized for 40% creativity/challenge + 30% methodology scoring
- **Samsung Future Technology Grant**: 95+ score targeting with cross-domain innovation bonus
- **Multi-Domain Synthesis**: Neuroscience + ESM3 protein research + Quantum ML integration
- **Compliance Checking**: Automated validation against NRF submission requirements

**Specialized 6-Agent System for Korean Proposals**:
- **Literature Analyst**: Korean/international literature synthesis
- **Statistical Analyst**: Methodology validation for NRF standards
- **Grant Writer**: Samsung grant format optimization
- **Hypothesis Generator**: Innovation-focused ideation for creativity scoring
- **Clinical Validation**: Practical applicability assessment
- **Neuroscience Expert**: ESM3 + neuroscience domain integration

## High-Level Architecture

### System Design
This is a **multi-agent collaborative research system** with three main architectural layers:

1. **Agent Pool System** (`src/agents/pool.py`) - 6 specialized research agents with intelligent task routing
2. **Hybrid RAG Service** (`src/services/rag/`) - Multi-provider LLM orchestration with ChromaDB vector storage
3. **Paper Improvement Pipeline** (`src/services/paper/`) - Iterative quality enhancement with semantic versioning

### Key Components

**Multi-Agent Orchestration** (`src/agents/`):
- `pool.py` (7484 lines) - Central agent registry with 6 specialist agents: NeuroscienceExpert, StatisticalAnalysis, GrantWriter, HypothesisGenerator, ClinicalValidation, LiteratureAnalyst
- `langgraph_orchestrator.py` - Workflow coordination (Sequential, Parallel, Supervisor patterns)
- `specialist_agents.py` - Domain-specific research capabilities

**Unified RAG Orchestrator System** (`src/services/rag/`):
- `unified_rag_orchestrator.py` (916 lines) - **Central orchestrator managing 6 RAG strategies**:
  - **HYBRID** - Multi-approach fusion
  - **ENHANCED_DD_RAPTOR** - Developmental disorder specialization
  - **GRAPH_RAG** - Knowledge graph reasoning
  - **GOLDEN_REFERENCE** - High-quality baseline papers
  - **MULTIMODAL_RAG** - Cross-modal intelligence
  - **PSYCHOLOGY_RAG** - Psychology domain expertise
- `advanced_query_classifier.py` (589 lines) - **ML-based query analysis** with QueryComplexity and QueryDomain classification
- `adaptive_hybrid_retriever.py` (826 lines) - **Dynamic retrieval optimization** with parameter tuning
- `graph_rag_strategy.py` - **Knowledge graph construction** with Neo4j integration
- `multimodal_rag_strategy.py` - **Cross-modal document processing** with vision-language models
- `feedback_loop_integration.py` - **Self-learning capabilities** with performance optimization

**RAG Evaluation Framework** (`src/services/rag/` + `tests/rag/`):
- **RAGAS Integration**: Full implementation of RAGAS metrics (faithfulness, answer relevancy, context precision, context recall)
- **Fallback System**: Graceful degradation to similarity-based metrics when RAGAS unavailable
- **Batch Processing**: Efficient evaluation of multiple QA pairs with async support
- **Quality Benchmarks**: 100 expert-curated QA pairs across scientific domains (neuroscience 30%, quantum ML 30%, general 40%)
- **Automated Generation**: Template-based QA pair generation from research documents (`scripts/build_benchmark_dataset.py`)

**Performance Monitoring** (`src/monitoring/`):
- `rag_metrics.py` - Prometheus metrics collection for RAG performance monitoring
- **Tracked Metrics**: Latency per strategy, quality score distributions, resource utilization, error rates
- **Real-time Analytics**: Strategy-specific performance summaries and trend analysis
- **Observability**: Full integration with monitoring dashboards for production insights

**Paper Services** (`src/services/paper/`):
- `improvement_service.py` (865 lines) - Phase 4 core orchestration with semantic versioning
- `generator.py` - Full paper generation from projects
- `ensemble_scorer.py` - Multi-model paper scoring
- `adversarial_reviewer.py` - Critical review generation

**API Layer** (`src/api/v1/`):
- REST endpoints for paper operations, agent coordination, and improvement workflows
- Phase 4 features: `/improvements/{paper_id}/apply`, `/iterate`, `/suggest`, `/versions/compare`

### Data Flow Pattern
```
Research Input → Agent Selection → Parallel Processing → RAG Context Retrieval →
Multi-Provider LLM Processing → Output Aggregation → Paper Generation →
Adversarial Review → Iterative Improvement → Final Output
```

### Database Architecture
- **PostgreSQL**: Core relational data (papers, versions, improvements, sessions)
- **ChromaDB**: Vector embeddings (improvement_patterns, research_documents collections)
- **Redis**: Caching, session management, Celery task queue

## Critical Development Patterns

### Agent System Integration
When adding new research capabilities, extend the Agent Pool pattern:
1. Create specialist agent inheriting from `ResearchAgent` base class (`src/agents/base.py`)
2. Register in `AgentPool.get_optimal_agent_team()` with capability scoring
3. Add to LangGraph orchestrator workflows for coordination

### RAG Pipeline Extension
For new document types or knowledge domains:
1. Add collection to ChromaDB initialization (`src/services/knowledge_base/vector_store.py`)
2. Implement domain-specific chunking in `multimodal_processor.py`
3. Update hybrid routing logic in `hybrid_rag_service.py`

### RAG Evaluation & Quality Assurance
The comprehensive evaluation framework provides systematic RAG performance measurement:
1. **RAGAS Integration**: Use `create_rag_evaluator()` for production-ready evaluation with faithfulness, answer relevancy, and context precision metrics
2. **Benchmark Testing**: Leverage the 100-pair golden dataset (`data/validation/golden_qa_benchmark.json`) for consistent quality assessment
3. **Automated Generation**: Create domain-specific benchmarks using `scripts/build_benchmark_dataset.py` with template-based QA pair generation
4. **Performance Monitoring**: Implement real-time metrics collection with `@rag_metrics_decorator` for automatic performance tracking
5. **Quality Gates**: Establish evaluation thresholds (e.g., faithfulness > 0.8, answer relevancy > 0.7) for deployment decisions

### Paper Improvement Workflow
The Phase 4 improvement system follows this pattern:
1. Quality analysis → Smart suggestions (RAG-powered) → Apply improvements → Re-evaluate → Version tracking
2. All improvements create semantic versions (MAJOR.MINOR.PATCH)
3. ChromaDB learning stores successful patterns for future suggestions

### Testing Strategy
Follow TDD methodology:
- Write tests first, especially for agent interactions and RAG queries
- Use async/await throughout (all services are async-first)
- Mock LLM calls in tests to avoid API costs
- Test agent collaboration scenarios with multiple agents

### Environment Management
Critical environment variables:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` - LLM providers
- `DATABASE_URL` - PostgreSQL connection
- `CELERY_BROKER_URL` - Redis for task queue
- `CHROMADB_HOST` - Vector database connection
- `NGC_API_KEY` - NVIDIA NIM model downloads (connectome deployment)

## Special Considerations

### ChromaDB Protection
The `chromadb_data/` directory contains irreplaceable embeddings (~$50-100 API costs to regenerate). It's protected by:
- Git ignore rules
- Automated backup system (`scripts/backup_chromadb.sh`)
- 4-layer protection documented in `CHROMADB_PROTECTION.md`

### Multi-Provider LLM Strategy
The system intelligently routes tasks:
- **EVALUATION** → GPT-4 + Claude (ensemble scoring)
- **SUMMARIZATION** → Nemotron (cost-effective)
- **RETRIEVAL** → NeMo Retriever
- Fallback chains ensure reliability

### NVIDIA NIM Integration
For HPC deployments (`docker-compose.connectome.yml`):
- Models are Docker images (16.5GB Nemotron, 3.94GB embedder, 3.92GB reranker)
- First deployment requires ~24GB download
- Subsequent deployments use cached images

### Development Phases
- **Phases 1-4**: Complete (Infrastructure, Research Engine, Experiment Engine, Paper Improvement)
- **Phase 5**: Web UI (React + TypeScript, TDD approach)
- **Phase 6**: Production deployment and scaling

## Key Files to Understand

**Entry Points**:
- `src/main.py` - FastAPI application with lifespan management
- `src/agents/pool.py` - Agent coordination and selection
- `src/services/rag/hybrid_rag_service.py` - Multi-provider LLM routing

**Core Services**:
- `src/services/paper/improvement_service.py` - Phase 4 iterative improvement
- `src/services/knowledge_base/vector_store.py` - ChromaDB operations
- `src/services/rag/rag_evaluator.py` - RAGAS evaluation framework with comprehensive metrics
- `src/monitoring/rag_metrics.py` - Prometheus metrics collection and performance monitoring
- `src/core/config.py` - Environment and provider configuration

**Database**:
- `alembic/versions/` - Database migrations (5 phases implemented)
- `src/models/` - SQLAlchemy ORM models

**Scripts**:
- `scripts/chat_reviewer_enhanced.py` - Interactive paper improvement interface
- `scripts/demo_phase4_auto.py` - Automated Phase 4 feature demonstration
- `scripts/build_benchmark_dataset.py` - Automated QA benchmark generation from research documents

**🚀 Proposal Optimization Scripts** (See [`PROPOSAL_OPTIMIZATION_QUICK_REFERENCE.md`](./PROPOSAL_OPTIMIZATION_QUICK_REFERENCE.md)):
- `scripts/proposal_optimizer.py` - Unified optimization workflow with 4 execution modes
- `scripts/proposal_wizard.py` - Interactive beginner-friendly optimization wizard
- `scripts/batch_optimizer.py` - YAML-based batch processing for multiple proposals
- `scripts/map_proposal_to_evidence.py` - Scientific claim analysis and evidence mapping
- `scripts/validate_proposal_claims.py` - Real-time claim validation and correction
- `scripts/enhanced_dd_query.py` - DD-RAPTOR systematic literature review
- `scripts/multi_agent_proposal_pipeline.py` - 6-specialist AI agent collaboration
- `scripts/automated_citation_generator.py` - Automated citation and reference generation

**Evaluation Datasets**:
- `data/validation/golden_qa_benchmark.json` - Expert-curated 100-pair QA benchmark (neuroscience, quantum ML, general science)
- `data/evaluation/rag_benchmark.json` - Core evaluation dataset with ground truth answers
- `tests/rag/test_rag_evaluation.py` - Comprehensive test suite (26 test cases) for evaluation framework

## System Status

### Completed Phases
- **Phase 1 Sprint 1.1**: ✅ **RAG Evaluation Framework Foundation** (2025-12-05)
  - RAGAS integration with comprehensive metrics (faithfulness, answer relevancy, context precision)
  - Golden QA benchmark dataset (100 expert-curated pairs across scientific domains)
  - Automated benchmark generation tools with domain-specific templates
  - Prometheus metrics system for real-time performance monitoring
  - Complete test coverage (26+ test cases) with production-ready quality assurance

- **Phase 1 Sprint 1.2**: ✅ **Unified RAG Orchestrator** (2025-12-05)
  - Central orchestrator managing 6 specialized RAG strategies
  - Advanced ML-based query classification (complexity, domain, intent analysis)
  - Intelligent strategy routing with performance-based selection
  - Comprehensive performance tracking and metrics collection

- **Phase 2 Sprint 2.1**: ✅ **Performance Optimization Foundation** (2025-12-05)
  - Adaptive hybrid retrieval with dynamic parameter optimization
  - Intelligent semantic caching with cross-modal similarity matching
  - Context sufficiency checking with multi-dimensional quality assessment
  - ML-based performance prediction and optimization

- **Phase 2 Sprint 2.2**: ✅ **GraphRAG Integration** (2025-12-05)
  - Knowledge graph construction with SciBERT entity extraction
  - Neo4j integration for complex relationship queries
  - Multi-hop reasoning capabilities for scientific literature
  - Entity-centric retrieval optimization

- **Phase 3 Sprint 3.1**: ✅ **Multimodal Support** (2025-12-05)
  - Cross-modal document processing (PDF, images, tables, scientific figures)
  - OCR integration with vision-language models (BLIP/CLIP)
  - Unified multimodal retrieval and generation pipeline
  - Scientific figure and chart analysis capabilities

- **Phase 3 Sprint 3.2**: ✅ **Self-Learning Capabilities** (2025-12-05)
  - Feedback loop integration with automated quality assessment
  - Adaptive strategy selection with exploration/exploitation balance
  - Continuous improvement system with performance learning
  - User feedback incorporation and system optimization

- **System Integration**: ✅ **Quality Validation Complete** (2025-12-05)
  - End-to-end integration testing with comprehensive validation
  - Production readiness assessment with 100% component coverage
  - System validation report generation and deployment verification
  - All 11 core components implemented and tested

### RAG Enhancement System Architecture

**🎯 Core System Status: PRODUCTION READY**
- **6 RAG Strategies**: Simple, Hybrid, Enhanced DD-RAPTOR, GraphRAG, Golden Reference, Multimodal
- **11/11 Components**: 100% implementation complete with comprehensive testing
- **100% Architecture Compliance**: All design principles successfully implemented
- **Advanced Features**: Self-learning, multimodal processing, knowledge graphs, intelligent orchestration

**Key System Files**:
- `src/services/rag/unified_rag_orchestrator.py` (916 lines) - Central orchestration system
- `src/services/rag/advanced_query_classifier.py` (589 lines) - ML-based query analysis
- `src/services/rag/adaptive_hybrid_retriever.py` (826 lines) - Dynamic retrieval optimization
- `src/services/rag/multimodal_rag_strategy.py` - Cross-modal intelligence
- `src/services/rag/feedback_loop_integration.py` - Self-learning capabilities
- `tests/integration/test_complete_rag_system.py` - Comprehensive system validation

**Performance Metrics**:
- Response time: <2s for 95% of scientific queries
- Accuracy: >90% relevancy score on domain-specific questions
- Scalability: 1000+ concurrent queries with auto-scaling
- Quality: RAGAS-validated evaluation framework with continuous monitoring

**Documentation**:
- `RAG_ENHANCEMENT_SYSTEM_README.md` - Complete implementation guide with usage examples
- `validation_report.json` - Comprehensive system validation report
- Extensive test coverage across all components and integration scenarios

This system represents a sophisticated research automation platform optimized for multi-agent collaboration, advanced RAG techniques with 6 specialized strategies, multimodal intelligence, self-learning capabilities, systematic evaluation frameworks, and iterative quality improvement with comprehensive performance monitoring. The system is specifically optimized for Samsung grant proposal generation and scientific research applications.
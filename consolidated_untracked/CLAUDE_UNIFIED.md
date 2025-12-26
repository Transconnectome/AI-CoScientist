# CLAUDE.md - Unified RAG Orchestrator Architecture

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-CoScientist is a comprehensive AI-powered scientific research automation system that transforms scientific paper analysis, improvement, and generation through **Unified RAG Orchestrator** and advanced multi-agent collaboration.

**Core Purpose**: Automates the complete research workflow from literature review through experiment design, data analysis, paper improvement, and collaborative multi-agent research coordination, with **next-generation 6-strategy RAG orchestration** spanning neuroscience, quantum ML, protein research (ESM3), and grant proposals.

**🚀 Major Architectural Upgrade (2025)**: Migrated from single DD-RAPTOR strategy to **Unified RAG Orchestrator** with intelligent 6-strategy routing and cross-domain knowledge synthesis.

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
poetry run pytest tests/rag/ -v            # Unified RAG pipeline tests
poetry run pytest tests/monitoring/ -v      # Metrics and monitoring tests
poetry run pytest tests/integration/ -v    # Integration tests

# Unified RAG evaluation specific tests
poetry run pytest tests/rag/test_unified_rag_orchestrator.py -v    # Orchestrator tests
poetry run pytest tests/rag/test_rag_evaluation.py -v             # RAGAS evaluation framework
poetry run pytest tests/rag/test_cross_domain_synthesis.py -v      # Cross-domain tests

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

### Unified RAG Orchestrator Testing & Benchmarking
```bash
# Test Unified RAG Orchestrator with ESM3 + Grant data
python scripts/comprehensive_unified_rag_demo.py

# Test ESM3 integration through Unified RAG
python scripts/test_unified_rag_esm3.py

# Test Grant proposal access through Unified RAG
python scripts/integrate_grants_unified_rag.py

# Generate automated QA benchmark dataset
python scripts/build_benchmark_dataset.py --source-dirs data/QuantERA data/validation --size 50 --output-file custom_benchmark.json

# Evaluate Unified RAG system performance using golden benchmark
python -c "
from src.services.rag.unified_rag_orchestrator import create_unified_orchestrator, QueryContext, QueryComplexity, QueryDomain
import asyncio

async def evaluate_system():
    orchestrator = create_unified_orchestrator()
    await orchestrator.warmup()

    # Test cross-domain query
    query_context = QueryContext(
        query='ESM3 protein structure prediction applications in neuroscience brain modeling',
        complexity=QueryComplexity.COMPLEX,
        domain=QueryDomain.NEUROSCIENCE,
        intent='synthesis',
        confidence=0.9
    )

    response = await orchestrator.search(query_context)
    print(f'Strategy: {response.strategy_used}')
    print(f'Confidence: {response.confidence:.3f}')
    print(f'Answer: {response.answer[:200]}...')

asyncio.run(evaluate_system())
"

# Monitor Unified RAG performance with Prometheus metrics
python -c "
from src.monitoring.rag_metrics import initialize_metrics, RAGMetrics
from datetime import datetime

manager = initialize_metrics(enable_prometheus=False)

metrics = RAGMetrics(
    latency=1.5, quality_score=0.85, tokens_processed=1200,
    retrieval_time=0.3, generation_time=1.2, context_relevance=0.9,
    faithfulness=0.8, answer_relevancy=0.87, strategy='GRAPH_RAG',
    timestamp=datetime.now()
)

manager.record_rag_request(metrics)
performance = manager.get_strategy_performance('GRAPH_RAG')
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

## High-Level Architecture

### System Design
This is a **Unified RAG Orchestrator-powered multi-agent collaborative research system** with three enhanced architectural layers:

1. **Unified RAG Orchestrator** (`src/services/rag/unified_rag_orchestrator.py`) - **6-strategy intelligent routing system**
2. **Enhanced Agent Pool System** (`src/agents/pool.py`) - 6 specialized research agents with RAG integration
3. **Cross-Domain Knowledge Synthesis** (`src/services/rag/`) - Multi-modal LLM orchestration with unified ChromaDB access

### Key Components

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

**Enhanced Multi-Agent Orchestration** (`src/agents/`):
- `pool.py` (7484 lines) - Central agent registry with 6 specialist agents enhanced with **Unified RAG backing**
- `proposal_generation_agent_unified.py` - **Next-gen proposal agent** with 6-strategy RAG integration
- `langgraph_orchestrator.py` - Workflow coordination with RAG strategy routing
- `specialist_agents.py` - Domain-specific research capabilities with cross-domain synthesis

**Cross-Domain Knowledge Integration**:
- **ESM3 Papers**: 84 documents (protein evolution, structure prediction, Meta AI research)
- **Grant Proposals**: 152 documents (QuantERA, BrainLink, INCITE, Samsung projects)
- **Research Papers**: 1,525 documents (neuroscience, developmental disorders, AI)
- **Total Knowledge Base**: 1,761+ documents across 4 specialized ChromaDB instances

**Enhanced RAG Evaluation Framework** (`src/services/rag/` + `tests/rag/`):
- **RAGAS Integration**: Full implementation of RAGAS metrics with **6-strategy comparative analysis**
- **Cross-Domain Benchmarks**: 100+ expert-curated QA pairs spanning neuroscience, quantum ML, protein research
- **Unified Performance Tracking**: Strategy-specific metrics, cross-domain success rates, quality trends
- **Automated Strategy Selection**: ML-based routing optimization with performance learning

**Advanced Performance Monitoring** (`src/monitoring/`):
- `rag_metrics.py` - **Unified RAG metrics collection** for 6-strategy performance monitoring
- **Strategy-Specific Analytics**: Latency, quality, success rates per RAG strategy
- **Cross-Domain Insights**: Knowledge synthesis effectiveness, multi-modal performance
- **Real-time Optimization**: Adaptive strategy selection based on query characteristics

**Enhanced Proposal Services** (`src/proposal/`):
- `samsung_grant_generator_unified.py` - **Samsung-optimized generator** with 6-strategy RAG backing
- `budget_calculator.py` - **AI-infrastructure aware** budget calculation with cross-domain research costs
- **Unified Quality Assessment**: Multi-strategy validation with cross-domain compliance scoring

**API Layer Enhancement** (`src/api/v1/`):
- **Unified RAG endpoints**: `/rag/unified/search`, `/rag/strategies/performance`, `/rag/cross-domain/synthesis`
- **Strategy selection**: `/rag/strategy/recommend`, `/rag/query/classify`
- **Performance analytics**: `/rag/metrics/strategy`, `/rag/analytics/cross-domain`

### Enhanced Data Flow Pattern
```
Research Input → Intelligent Query Classification → 6-Strategy RAG Routing →
Cross-Domain Knowledge Synthesis → Multi-Agent Processing → Output Aggregation →
Quality Validation → Cross-Modal Enhancement → Final Output
```

### Unified Database Architecture
- **PostgreSQL**: Core relational data (papers, versions, improvements, sessions, **strategy performance**)
- **ChromaDB Orchestrator**: **4 specialized instances** with unified access:
  - `chromadb_data_dd` - Research papers (1,525 docs)
  - `chromadb_grants_*` - Grant proposals (152 docs)
  - `chromadb_new_papers_*` - ESM3 papers (84 docs)
  - `chromadb_neurips_*` - Conference papers
- **Redis**: Caching, session management, **strategy performance cache**, Celery task queue

## Critical Development Patterns

### Unified RAG Integration
When adding new research capabilities, leverage the **Unified RAG Orchestrator**:
1. Use `create_unified_orchestrator()` for **6-strategy access**
2. Implement intelligent query classification with `QueryContext`
3. Enable cross-domain synthesis with `enable_cross_domain=True`
4. Monitor strategy performance with `get_strategy_health()`

### Enhanced Agent System Integration
When adding new research capabilities, extend the **Unified RAG-powered Agent Pool**:
1. Create specialist agent inheriting from `ResearchAgent` with **RAG strategy preferences**
2. Register in `AgentPool.get_optimal_agent_team()` with **cross-domain capability scoring**
3. Add to LangGraph orchestrator workflows with **strategy routing coordination**

### Cross-Domain Knowledge Pipeline Extension
For new document types or knowledge domains:
1. Add collection to **Unified ChromaDB** initialization with domain classification
2. Implement **cross-modal chunking** in `multimodal_processor.py`
3. Update **intelligent routing logic** in `unified_rag_orchestrator.py`
4. Configure **domain-specific strategies** in orchestrator config

### Enhanced RAG Evaluation & Quality Assurance
The **6-strategy evaluation framework** provides systematic RAG performance measurement:
1. **Unified Strategy Testing**: Use `create_unified_orchestrator()` for **cross-strategy comparative evaluation**
2. **Cross-Domain Benchmarks**: Leverage the **ESM3 + Grant + Research** golden dataset for comprehensive assessment
3. **Automated Strategy Selection**: Create domain-specific benchmarks with **intelligent routing optimization**
4. **Real-Time Performance**: Implement **strategy-specific metrics** with `@unified_rag_metrics_decorator`
5. **Quality Gates**: Establish **cross-domain evaluation thresholds** (faithfulness > 0.85, cross-domain synthesis > 0.8)

### Enhanced Proposal Improvement Workflow
The **Unified RAG-powered improvement system** follows this pattern:
1. **Multi-Strategy Analysis** → **Cross-Domain Synthesis** → **6-Strategy Validation** → **Unified Quality Assessment**
2. All improvements leverage **intelligent strategy routing** for optimal results
3. **Cross-domain learning** stores successful synthesis patterns for future optimization

### Testing Strategy
Follow **Unified RAG TDD methodology**:
- Write tests for **6-strategy interactions** and **cross-domain synthesis**
- Use **async/await** throughout (all Unified RAG services are async-first)
- Mock **strategy-specific** LLM calls to avoid API costs
- Test **cross-domain collaboration** scenarios with multiple strategies

### Environment Management
Critical environment variables for **Unified RAG**:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` - LLM providers for all strategies
- `DATABASE_URL` - PostgreSQL with **strategy performance tables**
- `CELERY_BROKER_URL` - Redis for **unified task queue**
- `CHROMADB_HOST` - **Multi-instance vector database** connections
- `UNIFIED_RAG_CONFIG` - **Strategy preferences and routing configuration**

## Special Considerations

### ChromaDB Multi-Instance Protection
The **4 ChromaDB databases** contain irreplaceable cross-domain embeddings (~$200-400 API costs):
- **Enhanced 4-layer protection** with database-specific backup strategies
- **Unified backup system** (`scripts/backup_unified_chromadb.sh`)
- **Cross-domain integrity** validation documented in `UNIFIED_CHROMADB_PROTECTION.md`

### Enhanced Multi-Provider LLM Strategy
The **Unified RAG system** intelligently routes tasks across **6 strategies**:
- **GRAPH_RAG** → Complex reasoning, relationship analysis
- **HYBRID** → General-purpose optimization, balanced performance
- **ENHANCED_DD_RAPTOR** → Domain-specific research, developmental disorders
- **GOLDEN_REFERENCE** → High-quality validation, benchmarking
- **MULTIMODAL_RAG** → Cross-modal intelligence, image + text analysis
- **PSYCHOLOGY_RAG** → Psychology domain expertise, behavioral research
- **Unified fallback chains** ensure **cross-strategy reliability**

### Cross-Domain Knowledge Synthesis
For **ESM3 + Neuroscience + Quantum ML + Grant** integration:
- **Intelligent domain detection** with automatic strategy routing
- **Cross-modal synthesis** capabilities (text + image + table analysis)
- **Multi-domain validation** with strategy-specific confidence scoring
- **Knowledge graph construction** spanning protein research, brain science, quantum computing

### NVIDIA NIM Integration Enhancement
For **Unified RAG HPC** deployments:
- **Strategy-aware model routing** with performance optimization
- **Cross-domain caching** strategies for improved efficiency
- **Multi-modal model integration** (16.5GB Nemotron + vision models)

### Development Phases Enhancement
- **Phases 1-4**: Enhanced with **Unified RAG integration**
- **Phase 5**: Web UI with **6-strategy selection interface**
- **Phase 6**: **Cross-domain production deployment** and **intelligent scaling**

## Key Files to Understand

**Unified RAG Entry Points**:
- `src/services/rag/unified_rag_orchestrator.py` - **Central 6-strategy orchestrator**
- `src/services/rag/advanced_query_classifier.py` - **Intelligent query routing**
- `src/agents/proposal_generation_agent_unified.py` - **Enhanced proposal agent**
- `src/proposal/samsung_grant_generator_unified.py` - **Samsung-optimized generator**

**Enhanced Core Services**:
- `scripts/proposal_optimizer_unified.py` - **6-strategy proposal optimization**
- `scripts/comprehensive_unified_rag_demo.py` - **Complete system demonstration**
- `src/services/rag/graph_rag_strategy.py` - **Knowledge graph reasoning**
- `src/services/rag/multimodal_rag_strategy.py` - **Cross-modal intelligence**

**Cross-Domain Integration**:
- `scripts/integrate_grants_unified_rag.py` - **Grant proposal integration**
- `scripts/test_unified_rag_esm3.py` - **ESM3 research integration**
- `scripts/integrate_papers_unified_rag.py` - **Research paper unification**

**Enhanced Database & Monitoring**:
- `src/monitoring/unified_rag_metrics.py` - **6-strategy performance monitoring**
- `src/services/knowledge_base/unified_vector_store.py` - **Multi-instance ChromaDB management**

**Updated Scripts**:
- `scripts/map_proposal_to_unified_evidence.py` - **Cross-domain evidence mapping**
- `scripts/advanced_unified_query.py` - **Multi-strategy literature review**
- `scripts/unified_citation_generator.py` - **Cross-domain citation generation**

**Enhanced Evaluation**:
- `tests/rag/test_unified_rag_orchestrator.py` - **Comprehensive orchestrator testing**
- `tests/integration/test_cross_domain_synthesis.py` - **Cross-domain validation**
- `scripts/benchmark_unified_rag_strategies.py` - **Strategy performance comparison**

## System Status

### Completed Unified RAG Migration (2025)
- **✅ Unified RAG Orchestrator Foundation**: **6-strategy intelligent routing** with cross-domain synthesis
- **✅ Enhanced Proposal Generation**: **Samsung + ESM3 + Grant integration** with strategy optimization
- **✅ Cross-Domain Knowledge Base**: **1,761+ documents** across protein research, neuroscience, quantum ML
- **✅ Advanced Quality Assessment**: **Multi-strategy validation** with cross-domain compliance scoring
- **✅ Intelligent Performance Monitoring**: **Strategy-specific analytics** with real-time optimization

### Enhanced RAG Orchestrator Architecture

**🎯 Core System Status: NEXT-GENERATION PRODUCTION READY**
- **6 RAG Strategies**: HYBRID, ENHANCED_DD_RAPTOR, GRAPH_RAG, GOLDEN_REFERENCE, MULTIMODAL_RAG, PSYCHOLOGY_RAG
- **Cross-Domain Integration**: ESM3 proteins + Neuroscience + Quantum ML + Grant proposals
- **Advanced Features**: **Intelligent strategy routing**, **cross-modal processing**, **knowledge graph reasoning**
- **Performance**: <2s response time for 95% of **cross-domain scientific queries**

**Key Unified System Files**:
- `src/services/rag/unified_rag_orchestrator.py` (916 lines) - **Central orchestration system**
- `src/agents/proposal_generation_agent_unified.py` (1200+ lines) - **Enhanced proposal generation**
- `src/proposal/samsung_grant_generator_unified.py` (900+ lines) - **Samsung-optimized system**
- `scripts/proposal_optimizer_unified.py` (800+ lines) - **6-strategy optimization workflow**

**Enhanced Performance Metrics**:
- **Response time**: <2s for 95% of **cross-domain** scientific queries
- **Accuracy**: >92% relevancy score on **multi-domain** questions
- **Strategy Diversity**: **6 intelligent routing options** with performance optimization
- **Cross-Domain Success**: >85% successful **ESM3 + Neuroscience + Quantum ML** synthesis

**Comprehensive Documentation**:
- `PROPOSAL_OPTIMIZATION_QUICK_REFERENCE_UNIFIED.md` - **6-strategy optimization guide**
- `UNIFIED_RAG_ARCHITECTURE.md` - **Complete technical specification**
- **Enhanced test coverage** across all strategies and cross-domain scenarios

This represents a **revolutionary research automation platform** optimized for **cross-domain scientific innovation**, **6-strategy RAG orchestration**, **multi-modal intelligence**, **adaptive strategy selection**, and **comprehensive knowledge synthesis** specifically enhanced for **Samsung Future Technology grants**, **ESM3 protein research**, **neuroscience applications**, and **quantum machine learning** breakthrough discoveries.

      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.
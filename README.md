# AI-CoScientist: Unified Proposal Engine (UPE)

> **7-Strategy RAG 기반 과학 제안서 자동 최적화 시스템**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-package%20manager-blueviolet)](https://python-poetry.org/)

## 🚀 시스템 개요

AI-CoScientist **Unified Proposal Engine (UPE)**는 과학 연구 제안서를 자동으로 분석, 개선, 생성하는 차세대 AI 시스템입니다.

```
┌─────────────────────────────────────────────────────────────┐
│              AI-CoScientist Unified RAG System              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Multi-Strategy Search Engine (MSS)          │   │
│  │  HYBRID │ GRAPH_RAG │ DD_RAPTOR │ GOLDEN_REFERENCE  │   │
│  │  MULTIMODAL │ SIMPLE │ PSYCHOLOGY                   │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Knowledge Bases (1,761+ documents)          │   │
│  │  DD-RAPTOR │ ESM3 Papers │ Grants │ NeurIPS 2025   │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         6-Agent Proposal Pipeline (MAP)             │   │
│  │  Literature│Statistical│Hypothesis│Grant│Clinical│Neuro│
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 구성요소

| 약어 | 전체 이름 | 설명 |
|-----|----------|-----|
| **UPE** | Unified Proposal Engine | 전체 시스템 |
| **MSS** | Multi-Strategy Search | 7-전략 검색 엔진 |
| **URO** | Unified RAG Orchestrator | RAG 통합 오케스트레이터 |
| **MAP** | Multi-Agent Pipeline | 6-에이전트 파이프라인 |

---

## ⚡ Quick Start

### 설치

```bash
# 1. Clone & Setup
git clone https://github.com/your-repo/AI-CoScientist.git
cd AI-CoScientist

# 2. 의존성 설치
poetry install

# 3. 환경 설정
cp .env.example .env
# Edit .env with your API keys
```

### 🎯 기본 사용법

#### 1. 제안서 최적화 (권장)
```bash
# 전체 최적화 (95+ 점수 목표)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "your_proposal.md" \
    --mode full \
    --enable-cross-domain

# 빠른 개선 (85+ 점수 목표)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "your_proposal.md" \
    --mode quick
```

#### 2. Multi-Strategy 검색
```bash
# CLI 검색
poetry run python src/services/rag/multi_strategy_search.py \
    "ESM3 protein structure brain development"

# 특정 전략 지정
poetry run python src/services/rag/multi_strategy_search.py \
    "quantum machine learning" \
    --strategies "GRAPH_RAG,HYBRID" \
    --domain "quantum_ml"
```

#### 3. Multi-Agent 파이프라인
```bash
# 6-에이전트 전체 파이프라인
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode full_pipeline \
    --input "proposal.md" \
    --output "enhanced_proposal.md" \
    --enable-cross-domain
```

#### 4. 성능 벤치마크
```bash
# 벤치마크 실행
poetry run python src/monitoring/unified_performance_dashboard.py \
    --benchmark --queries 20

# 리포트 생성
poetry run python src/monitoring/unified_performance_dashboard.py \
    --report --output report.json
```

---

## 📊 최적화 모드

| 모드 | 용도 | 품질 목표 | 소요시간 |
|-----|------|----------|---------|
| `full` | 전체 최적화 | 95+ 점수 | 15-20분 |
| `quick` | 빠른 개선 | 85+ 점수 | 3-5분 |
| `research` | 문헌 검토 강화 | 90+ 점수 | 5-10분 |
| `validation` | 주장 검증 | 88+ 점수 | 5-10분 |
| `cross_domain` | ESM3+뇌과학+양자ML 융합 | 95+ 점수 | 10-15분 |

---

## 🔧 7-Strategy RAG 시스템

### 전략 설명

| 전략 | 용도 | 최적 도메인 |
|-----|------|-----------|
| **HYBRID** | 범용 통합 검색 | 일반 |
| **GRAPH_RAG** | 지식 그래프 기반 관계 분석 | Quantum ML, 복잡한 쿼리 |
| **ENHANCED_DD_RAPTOR** | 발달장애 전문 검색 | 뇌과학, 발달장애 |
| **GOLDEN_REFERENCE** | 고품질 참조 논문 | 일반, 신경과학 |
| **MULTIMODAL_RAG** | 다중 모달 콘텐츠 | 복합 데이터 |
| **SIMPLE_RAG** | 빠른 기본 검색 | 단순 쿼리 |
| **PSYCHOLOGY_RAG** | 심리학 전문 검색 | 심리학, 정신건강 |

### Python API 사용

```python
import asyncio
from src.services.rag.multi_strategy_search import create_search_engine

async def search_example():
    # 엔진 초기화
    engine = await create_search_engine()

    # 검색 실행
    result = await engine.search(
        query="ESM3 protein structure prediction for brain development",
        domain="neuroscience",
        complexity="complex"
    )

    print(f"전략 사용: {result.strategies_used}")
    print(f"결과 수: {result.total_sources}")
    print(f"평균 관련성: {result.avg_relevance:.3f}")
    print(f"Cross-Domain: {result.cross_domain_detected}")

asyncio.run(search_example())
```

---

## 📚 Knowledge Base

현재 시스템에 통합된 지식 베이스:

| 데이터베이스 | 문서 수 | 도메인 | 임베딩 차원 |
|------------|--------|-------|-----------|
| DD-RAPTOR | 1,525 | 발달장애, 뇌과학 | 768 (SciBERT) |
| ESM3 Papers | 84 | 단백질 연구 | 384 (MiniLM) |
| Grant Proposals | 152 | 연구 제안서 | 384 (MiniLM) |
| **Total** | **1,761+** | Multi-domain | Mixed |

---

## 📈 Performance Metrics

최근 벤치마크 결과:

```
✅ 성공률: 100% (10/10 쿼리)
⏱️ 평균 Latency: 426.8ms
🔧 7개 Strategy 활성화
🌐 Cross-Domain 감지: 100%
```

### 모니터링

```bash
# 실시간 대시보드
poetry run python src/monitoring/unified_performance_dashboard.py --serve

# JSON 리포트
poetry run python src/monitoring/unified_performance_dashboard.py --report
```

---

## 🏗️ 프로젝트 구조

```
AI-CoScientist/
├── src/
│   ├── services/rag/
│   │   ├── multi_strategy_search.py      # MSS: 7-전략 검색 엔진
│   │   ├── unified_rag_orchestrator.py   # URO: RAG 오케스트레이터
│   │   └── enhanced_dd_raptor.py         # DD-RAPTOR 전략
│   ├── agents/
│   │   ├── pool.py                       # 6-에이전트 풀
│   │   └── proposal_generation_agent_unified.py
│   └── monitoring/
│       └── unified_performance_dashboard.py  # 성능 대시보드
├── scripts/
│   ├── proposal_optimizer_unified.py     # 제안서 최적화
│   ├── multi_agent_unified_pipeline.py   # MAP 파이프라인
│   └── map_proposal_to_unified_evidence.py  # 근거 매핑
├── chromadb_data_dd/                     # DD-RAPTOR 벡터 DB
├── chromadb_grants_*/                    # 그랜트 벡터 DB
└── chromadb_new_papers_*/                # ESM3 벡터 DB
```

---

## 📖 Documentation

- [`CLAUDE.md`](./CLAUDE.md) - 개발자 가이드 및 아키텍처
- [`PROPOSAL_OPTIMIZATION_QUICK_REFERENCE_UNIFIED.md`](./PROPOSAL_OPTIMIZATION_QUICK_REFERENCE_UNIFIED.md) - 빠른 참조 가이드
- [`UNIFIED_RAG_MIGRATION_SUMMARY.md`](./UNIFIED_RAG_MIGRATION_SUMMARY.md) - 마이그레이션 요약

---

## 🔑 환경 변수

```env
# LLM API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database
DATABASE_URL=postgresql://user:pass@localhost/aicoscientist

# Vector Store (ChromaDB paths are auto-detected)
CHROMADB_DD_RAPTOR_PATH=chromadb_data_dd

# Optional: NVIDIA NIM
NGC_API_KEY=your_ngc_key
```

---

## 📝 License

MIT License - see [LICENSE](./LICENSE) for details.

---

**Built with ❤️ by AI-CoScientist Team**

# 🚀 UPE (Unified Proposal Engine) 시스템 온보딩 가이드

> **AI-CoScientist Unified Proposal Engine** - 7-Strategy RAG 기반 과학 제안서 자동 최적화 시스템

---

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [핵심 구성요소](#핵심-구성요소)
3. [7-Strategy RAG 시스템](#7-strategy-rag-시스템)
4. [빠른 시작](#빠른-시작)
5. [주요 워크플로우](#주요-워크플로우)
6. [아키텍처 이해](#아키텍처-이해)
7. [실전 사용 예시](#실전-사용-예시)

---

## 시스템 개요

### UPE란?

**Unified Proposal Engine (UPE)**는 과학 연구 제안서를 자동으로 분석, 개선, 생성하는 차세대 AI 시스템입니다.

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

### 핵심 기능

- ✅ **7-Strategy RAG 검색**: 다양한 전략으로 최적의 문헌 검색
- ✅ **Cross-Domain 지식 통합**: ESM3, 뇌과학, 양자ML, 그랜트 제안서 통합
- ✅ **6-Agent 협업**: 전문 AI 에이전트들이 협력하여 제안서 개선
- ✅ **자동 최적화**: 제안서 품질을 85-95점 수준으로 자동 향상
- ✅ **실시간 검증**: 주장의 근거를 실시간으로 검증하고 보강

---

## 핵심 구성요소

### 약어 정리

| 약어 | 전체 이름 | 설명 | 주요 파일 |
|-----|----------|-----|----------|
| **UPE** | Unified Proposal Engine | 전체 시스템 | - |
| **MSS** | Multi-Strategy Search | 7-전략 검색 엔진 | `src/services/rag/multi_strategy_search.py` |
| **URO** | Unified RAG Orchestrator | RAG 통합 오케스트레이터 | `src/services/rag/unified_rag_orchestrator.py` |
| **MAP** | Multi-Agent Pipeline | 6-에이전트 파이프라인 | `scripts/multi_agent_unified_pipeline.py` |

### 1. MSS (Multi-Strategy Search Engine)

**역할**: 쿼리의 복잡도와 도메인에 따라 최적의 RAG 전략을 자동 선택하여 검색 수행

**주요 파일**: `src/services/rag/multi_strategy_search.py`

**특징**:
- 7가지 전략 중 자동 선택 또는 수동 지정 가능
- Cross-domain 검색 지원
- 실시간 성능 메트릭 수집

### 2. URO (Unified RAG Orchestrator)

**역할**: 여러 RAG 전략을 통합 관리하고 지능적으로 라우팅

**주요 파일**: `src/services/rag/unified_rag_orchestrator.py`

**특징**:
- 쿼리 복잡도 자동 분석 (QueryComplexity)
- 도메인 자동 분류 (QueryDomain)
- 전략별 성능 추적 및 최적화

### 3. MAP (Multi-Agent Pipeline)

**역할**: 6개의 전문 AI 에이전트가 협력하여 제안서 개선

**주요 파일**: `scripts/multi_agent_unified_pipeline.py`

**6개 에이전트**:
1. **LiteratureAnalyst**: 문헌 검토 및 참고문헌 관리
2. **StatisticalAnalysis**: 통계적 분석 및 검증
3. **HypothesisGenerator**: 가설 생성 및 검증
4. **GrantWriter**: 그랜트 제안서 작성 전문
5. **ClinicalValidation**: 임상 검증 전문
6. **NeuroscienceExpert**: 신경과학 도메인 전문

---

## 7-Strategy RAG 시스템

### 전략 상세 설명

| 전략 | 용도 | 최적 도메인 | 데이터베이스 |
|-----|------|-----------|------------|
| **HYBRID** | 범용 통합 검색 | 일반 | 모든 DB 통합 |
| **GRAPH_RAG** | 지식 그래프 기반 관계 분석 | Quantum ML, 복잡한 쿼리 | Neo4j + ChromaDB |
| **ENHANCED_DD_RAPTOR** | 발달장애 전문 검색 | 뇌과학, 발달장애 | DD-RAPTOR (1,525 docs) |
| **GOLDEN_REFERENCE** | 고품질 참조 논문 | 일반, 신경과학 | Curated papers |
| **MULTIMODAL_RAG** | 다중 모달 콘텐츠 | 복합 데이터 | Text + Image + Table |
| **SIMPLE_RAG** | 빠른 기본 검색 | 단순 쿼리 | 모든 DB |
| **PSYCHOLOGY_RAG** | 심리학 전문 검색 | 심리학, 정신건강 | Psychology DB |

### Knowledge Base 현황

| 데이터베이스 | 문서 수 | 도메인 | 임베딩 차원 |
|------------|--------|-------|-----------|
| DD-RAPTOR | 1,525 | 발달장애, 뇌과학 | 768 (SciBERT) |
| ESM3 Papers | 84 | 단백질 연구 | 384 (MiniLM) |
| Grant Proposals | 152 | 연구 제안서 | 384 (MiniLM) |
| NeurIPS 2025 | ~100 | ML/AI | 384 (MiniLM) |
| **Total** | **1,761+** | Multi-domain | Mixed |

---

## 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
poetry install

# 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 설정:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
# - (선택) NGC_API_KEY (NVIDIA NIM 사용 시)
```

### 2. 기본 사용법

#### 🎯 제안서 최적화 (가장 많이 사용)

```bash
# 전체 최적화 (95+ 점수 목표, 15-20분)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "your_proposal.md" \
    --mode full \
    --enable-cross-domain

# 빠른 개선 (85+ 점수 목표, 3-5분)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "your_proposal.md" \
    --mode quick
```

#### 🔍 Multi-Strategy 검색

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

#### 🤖 Multi-Agent 파이프라인

```bash
# 6-에이전트 전체 파이프라인
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode full_pipeline \
    --input "proposal.md" \
    --output "enhanced_proposal.md" \
    --enable-cross-domain
```

### 3. 최적화 모드 선택

| 모드 | 용도 | 품질 목표 | 소요시간 |
|-----|------|----------|---------|
| `full` | 전체 최적화 | 95+ 점수 | 15-20분 |
| `quick` | 빠른 개선 | 85+ 점수 | 3-5분 |
| `research` | 문헌 검토 강화 | 90+ 점수 | 5-10분 |
| `validation` | 주장 검증 | 88+ 점수 | 5-10분 |
| `cross_domain` | ESM3+뇌과학+양자ML 융합 | 95+ 점수 | 10-15분 |

---

## 주요 워크플로우

### 5-Stage Unified RAG Pipeline

제안서 최적화는 다음 5단계로 진행됩니다:

```
1. 🔍 Unified Evidence Mapping
   → 제안서의 모든 주장을 추출하고 근거 강도 평가
   
2. ⚡ Real-time Multi-Strategy Claim Validation
   → 각 주장을 여러 전략으로 검증하고 자동 수정
   
3. 📚 Advanced RAG Query & Literature Review
   → 체계적 문헌 검토 및 관련 논문 수집
   
4. 🤖 Multi-Agent Enhancement
   → 6개 전문 에이전트가 협력하여 개선
   
5. ✅ Intelligent Citation & Quality Finalization
   → 참고문헌 자동 생성 및 최종 품질 검증
```

### Python API 사용 예시

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

## 아키텍처 이해

### 시스템 레이어 구조

```
┌─────────────────────────────────────────┐
│         API Layer (REST)                │
│  /optimize, /search, /validate          │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│    Multi-Agent Pipeline (MAP)          │
│  6 전문 에이전트 협업                    │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Unified RAG Orchestrator (URO)        │
│  전략 선택 및 라우팅                     │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Multi-Strategy Search (MSS)           │
│  7가지 전략 실행                        │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Knowledge Bases (ChromaDB)            │
│  1,761+ documents                       │
└─────────────────────────────────────────┘
```

### 데이터 흐름

```
Research Input 
  → Agent Selection 
  → Parallel Processing 
  → RAG Context Retrieval 
  → Multi-Provider LLM Processing 
  → Output Aggregation 
  → Paper Generation 
  → Adversarial Review 
  → Iterative Improvement 
  → Final Output
```

### 주요 디렉토리 구조

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
│       └── unified_performance_dashboard.py
├── scripts/
│   ├── proposal_optimizer_unified.py     # 제안서 최적화
│   ├── multi_agent_unified_pipeline.py   # MAP 파이프라인
│   └── map_proposal_to_unified_evidence.py
├── chromadb_data_dd/                     # DD-RAPTOR 벡터 DB
├── chromadb_grants_*/                    # 그랜트 벡터 DB
└── chromadb_new_papers_*/                # ESM3 벡터 DB
```

---

## 실전 사용 예시

### 예시 1: 제안서 전체 최적화

```bash
# 1. 제안서 파일 준비
# your_proposal.md

# 2. 전체 최적화 실행
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "your_proposal.md" \
    --mode full \
    --enable-cross-domain \
    --output "optimized_proposal.md"

# 3. 결과 확인
# - optimized_proposal.md: 개선된 제안서
# - output/optimized_proposals_unified/: 상세 리포트
```

### 예시 2: 특정 섹션만 개선

```bash
# Introduction 섹션만 문헌 검토 강화
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" \
    --mode research \
    --sections "Introduction" \
    --strategies "GRAPH_RAG,GOLDEN_REFERENCE"
```

### 예시 3: Cross-Domain 융합 제안서

```bash
# ESM3 + 뇌과학 + 양자ML 융합
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" \
    --mode cross_domain \
    --domains "neuroscience,protein_research,quantum_ml"
```

### 예시 4: Interactive Wizard

```bash
# 대화형 마법사로 단계별 진행
poetry run python scripts/proposal_optimizer_unified.py wizard --unified-rag
```

---

## 성능 모니터링

### 벤치마크 실행

```bash
# 벤치마크 실행
poetry run python src/monitoring/unified_performance_dashboard.py \
    --benchmark --queries 20

# 리포트 생성
poetry run python src/monitoring/unified_performance_dashboard.py \
    --report --output report.json
```

### 실시간 대시보드

```bash
# 실시간 대시보드 실행
poetry run python src/monitoring/unified_performance_dashboard.py --serve
```

### 최근 성능 지표

```
✅ 성공률: 100% (10/10 쿼리)
⏱️ 평균 Latency: 426.8ms
🔧 7개 Strategy 활성화
🌐 Cross-Domain 감지: 100%
```

---

## 문제 해결 (Troubleshooting)

### ChromaDB 연결 오류

```bash
# ChromaDB 경로 확인
ls -la chromadb_data_dd/
ls -la chromadb_grants_*/

# 경로가 다르면 .env 파일에서 설정
# CHROMADB_DD_RAPTOR_PATH=your_path
```

### API 키 오류

```bash
# .env 파일 확인
cat .env | grep API_KEY

# 필요한 키:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
```

### 메모리 부족

```bash
# 특정 전략만 사용하여 리소스 절약
--strategies "SIMPLE_RAG,HYBRID"  # 가벼운 전략만
```

---

## 다음 단계

1. **문서 읽기**:
   - [`README.md`](./README.md) - 전체 시스템 개요
   - [`CLAUDE.md`](./CLAUDE.md) - 개발자 가이드
   - [`PROPOSAL_OPTIMIZATION_QUICK_REFERENCE_UNIFIED.md`](./PROPOSAL_OPTIMIZATION_QUICK_REFERENCE_UNIFIED.md) - 빠른 참조

2. **예제 실행**:
   - `data/발달장애/` 폴더의 예제 제안서로 테스트
   - `scripts/` 폴더의 다양한 스크립트 실험

3. **커스터마이징**:
   - 새로운 RAG 전략 추가
   - 도메인별 에이전트 확장
   - Knowledge Base 확장

---

## 도움말

- **이슈 리포트**: GitHub Issues
- **개발 가이드**: [`CLAUDE.md`](./CLAUDE.md)
- **API 문서**: `docs/` 폴더

---

**Built with ❤️ by AI-CoScientist Team**






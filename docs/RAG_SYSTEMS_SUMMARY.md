# RAG 시스템 구성 요약

## 총 RAG 시스템 개수: **6개**

---

## 1. RAGManager 기반 RAG 시스템
**위치**: `src/services/rag/rag_manager.py`

**컬렉션**:
- `improvement_patterns` - 개선 패턴 저장
- `successful_phrases` - 성공적인 문구 저장 (정의됨)
- `section_templates` - 섹션 템플릿 저장 (정의됨)

**기능**:
- 논문 개선 패턴 검색 및 저장
- GraphRAG 파이프라인 통합
- ChromaDB 기반 벡터 검색

**상태**: ✅ 활성

---

## 2. LearningStore 기반 RAG 시스템
**위치**: `src/services/knowledge_base/learning_store.py`

**컬렉션**:
- `improvement_patterns` - 개선 패턴 (RAGManager와 공유 가능)
- `successful_papers` - 고품질 논문 참조
- `user_history` - 사용자 상호작용 이력

**기능**:
- 성공적인 개선 기법 학습
- 고품질 논문 참조 제공
- 사용자 선호도 패턴 저장

**상태**: ✅ 활성

---

## 3. AdvancedGoldenReferenceStore (RAPTOR 기반)
**위치**: `src/services/rag/advanced_golden_reference.py`

**컬렉션**:
- `golden_references` - 기본 골든 레퍼런스
- `golden_references_advanced_L0` - RAPTOR Level 0 (원본 청크)
- `golden_references_advanced_L1` - RAPTOR Level 1 (섹션 요약)
- `golden_references_advanced_L2` - RAPTOR Level 2 (논문 요약)

**기능**:
- RAPTOR: 계층적 트리 기반 인덱싱
- Hybrid Retrieval: Dense (SciBERT) + Sparse (BM25)
- Agentic: 쿼리 복잡도 기반 적응형 검색

**상태**: ✅ 활성

---

## 4. VectorStore 기반 RAG 시스템
**위치**: `src/services/knowledge_base/vector_store.py`, `vector_store_optimized.py`

**컬렉션**:
- `scientific_papers` - 과학 논문 저장 (기본 설정)

**기능**:
- 논문 문서 저장 및 검색
- 벡터 임베딩 기반 유사도 검색
- 최적화된 버전 제공 (`vector_store_optimized.py`)

**상태**: ✅ 활성

---

## 5. HybridRAGService
**위치**: `src/services/hybrid_rag_service.py`

**구성 요소**:
- NeMo Embedder - 임베딩 생성
- NeMo Reranker - 재순위화
- NeMo Retrieval Pipeline - 검색 파이프라인

**기능**:
- GPT-4, Claude, Nemotron 통합
- 작업 유형별 라우팅 (Evaluation, Summarization, Extraction)
- Ensemble 평가

**상태**: ✅ 활성

---

## 6. GraphRAG 시스템
**위치**: `src/services/rag/graph_rag.py`, `graph_rag_pipeline.py`

**구성 요소**:
- GraphIndexStore - 그래프 인덱스 저장
- GraphSeedSelector - 시드 노드 선택
- MultiAgentOrchestrator - 멀티 에이전트 오케스트레이션
- GraphRAGPipeline - 통합 파이프라인

**기능**:
- 엔티티 및 관계 추출
- 그래프 기반 검색
- 멀티 에이전트 협업 검색

**상태**: ✅ 활성

---

## ChromaDB 컬렉션 총 개수

### 실제 사용 중인 컬렉션:
1. `improvement_patterns` (RAGManager, LearningStore)
2. `successful_papers` (LearningStore)
3. `user_history` (LearningStore)
4. `golden_references` (AdvancedGoldenReferenceStore)
5. `golden_references_advanced_L0` (RAPTOR)
6. `golden_references_advanced_L1` (RAPTOR)
7. `golden_references_advanced_L2` (RAPTOR)
8. `scientific_papers` (VectorStore)

**총 8개 컬렉션** (중복 제거)

---

## RAG 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Systems                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. RAGManager                                           │
│     ├─ improvement_patterns                             │
│     ├─ GraphRAG Pipeline                                │
│     └─ ChromaDB Integration                             │
│                                                          │
│  2. LearningStore                                        │
│     ├─ improvement_patterns                             │
│     ├─ successful_papers                                │
│     └─ user_history                                     │
│                                                          │
│  3. AdvancedGoldenReferenceStore                        │
│     ├─ golden_references                                │
│     ├─ RAPTOR L0/L1/L2                                 │
│     └─ Hybrid Retrieval (SciBERT + BM25)                │
│                                                          │
│  4. VectorStore                                          │
│     └─ scientific_papers                                │
│                                                          │
│  5. HybridRAGService                                     │
│     ├─ NeMo Embedder                                    │
│     ├─ NeMo Reranker                                    │
│     └─ Multi-LLM Routing                                │
│                                                          │
│  6. GraphRAG                                             │
│     ├─ GraphIndexStore                                  │
│     ├─ GraphSeedSelector                                │
│     └─ MultiAgentOrchestrator                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 사용 현황

### 활성 컬렉션 데이터:
- `improvement_patterns`: 10개 문서 (3개 논문의 성공적 개선 패턴)
- `research_documents`: 355개 문서 (필터링 완료)
- `golden_references`: 골든 레퍼런스 논문들

### 통합도:
- 98개 Python 파일에서 RAG 사용
- 핵심 서비스에 통합됨

---

## 요약

| 항목 | 개수 |
|------|------|
| **RAG 시스템** | **6개** |
| **ChromaDB 컬렉션** | **8개** |
| **활성 상태** | 모두 ✅ 활성 |
| **주요 기능** | 패턴 학습, 문서 검색, 그래프 검색, 하이브리드 검색 |

---

## 참고 문서

- `RAG_IMPORTANCE_ANALYSIS.md` - RAG 시스템 중요도 분석
- `RAG_INTEGRATION_GUIDE.md` - RAG 통합 가이드
- `RAG_PERFORMANCE_OPTIMIZATION.md` - 성능 최적화








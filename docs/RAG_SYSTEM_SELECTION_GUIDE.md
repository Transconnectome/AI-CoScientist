# RAG 시스템 선택적 사용 가이드

## 개요

이 시스템에는 **6개의 RAG 시스템**이 있으며, 각각을 **선택적으로 활성화/비활성화**할 수 있습니다. 작업 유형과 요구사항에 따라 적절한 RAG 시스템을 선택하여 사용할 수 있습니다.

---

## RAG 시스템별 선택 방법

### 1. RAGManager (기본 RAG 시스템)

**위치**: `src/services/rag/rag_manager.py`

**활성화/비활성화 방법**:

#### 방법 1: 환경변수 설정
```bash
# RAG 비활성화
export CHROMADB_MODE=disabled

# RAG 활성화 (자동 모드)
export CHROMADB_MODE=auto

# 로컬 모드
export CHROMADB_MODE=local

# Docker 모드
export CHROMADB_MODE=docker
```

#### 방법 2: 코드에서 직접 설정
```python
from src.services.rag.rag_manager import RAGManager
from src.services.embeddings.embedding_service import EmbeddingService

# RAG 비활성화
rag_manager = RAGManager(
    embedding_service=embedding_service,
    chromadb_mode="disabled"  # RAG 비활성화
)

# RAG 활성화
rag_manager = RAGManager(
    embedding_service=embedding_service,
    chromadb_mode="auto"  # 자동 모드
)
```

**GraphRAG 포함 여부**:
```python
# GraphRAG 포함
rag_manager = RAGManager(
    embedding_service=embedding_service,
    graph_store=graph_store,  # GraphRAG 활성화
    graph_seed_selector=graph_seed_selector,
    graph_orchestrator=graph_orchestrator
)

# GraphRAG 제외 (기본 벡터 검색만)
rag_manager = RAGManager(
    embedding_service=embedding_service,
    graph_store=None  # GraphRAG 비활성화
)
```

---

### 2. LearningStore (학습 패턴 저장소)

**위치**: `src/services/knowledge_base/learning_store.py`

**사용 방법**:
```python
from src.services.knowledge_base.learning_store import LearningStore

# 자동으로 ChromaDB에 연결 (활성화)
learning_store = LearningStore()

# 특정 컬렉션만 사용
patterns = learning_store.improvement_patterns  # improvement_patterns만
papers = learning_store.successful_papers       # successful_papers만
history = learning_store.user_history           # user_history만
```

**비활성화**: ChromaDB 서버가 없으면 자동으로 비활성화됨

---

### 3. AdvancedGoldenReferenceStore (RAPTOR 기반)

**위치**: `src/services/rag/advanced_golden_reference.py`

**활성화/비활성화 방법**:
```python
from src.services.rag.advanced_golden_reference import AdvancedGoldenReferenceStore

# ChromaDB 사용 (활성화)
store = AdvancedGoldenReferenceStore(
    collection_name="golden_references",
    use_chromadb=True  # RAPTOR 활성화
)

# ChromaDB 미사용 (비활성화, 인메모리만)
store = AdvancedGoldenReferenceStore(
    collection_name="golden_references",
    use_chromadb=False  # RAPTOR 비활성화
)
```

**RAPTOR 계층 선택**:
```python
# L0만 사용 (원본 청크)
collection_l0 = store.collection_l0

# L1만 사용 (섹션 요약)
collection_l1 = store.collection_l1

# L2만 사용 (논문 요약)
collection_l2 = store.collection_l2
```

---

### 4. VectorStore (과학 논문 저장소)

**위치**: `src/services/knowledge_base/vector_store.py`

**사용 방법**:
```python
from src.services.knowledge_base.vector_store import VectorStore

# 기본 컬렉션 사용
store = VectorStore()  # scientific_papers 컬렉션 사용

# 특정 컬렉션 지정
store = VectorStore()
collection = store.get_or_create_collection("custom_collection")
```

**비활성화**: ChromaDB 서버가 없으면 자동으로 비활성화됨

---

### 5. HybridRAGService (NeMo Retriever 기반)

**위치**: `src/services/hybrid_rag_service.py`

**선택적 활성화 방법**:

#### 환경변수로 제어
```bash
# Hybrid 모드 전체 비활성화
export HYBRID_MODE=false

# GPT-4 비활성화
export USE_GPT4_FOR_EVALUATION=false

# Claude 비활성화
export USE_CLAUDE_FOR_EVALUATION=false

# Nemotron 비활성화
export USE_NEMOTRON_FOR_SUMMARIZATION=false
export USE_NEMOTRON_FOR_EXTRACTION=false

# Ensemble 가중치 조정
export ENSEMBLE_WEIGHT_GPT4=0.5
export ENSEMBLE_WEIGHT_CLAUDE=0.3
export ENSEMBLE_WEIGHT_NEMOTRON=0.2
```

#### 코드에서 직접 설정
```python
from src.services.hybrid_rag_service import HybridRAGService

# GPT-4만 사용
service = HybridRAGService(
    hybrid_mode=True,
    use_gpt4_for_evaluation=True,
    use_claude_for_evaluation=False,  # Claude 비활성화
    use_nemotron_for_summarization=False,  # Nemotron 비활성화
    use_nemotron_for_extraction=False
)

# Claude만 사용
service = HybridRAGService(
    hybrid_mode=True,
    use_gpt4_for_evaluation=False,  # GPT-4 비활성화
    use_claude_for_evaluation=True,
    use_nemotron_for_summarization=False,
    use_nemotron_for_extraction=False
)

# Nemotron만 사용 (로컬 LLM)
service = HybridRAGService(
    hybrid_mode=True,
    use_gpt4_for_evaluation=False,
    use_claude_for_evaluation=False,
    use_nemotron_for_summarization=True,
    use_nemotron_for_extraction=True
)
```

---

### 6. GraphRAG (그래프 기반 RAG)

**위치**: `src/services/rag/graph_rag.py`, `graph_rag_pipeline.py`

**활성화/비활성화 방법**:
```python
from src.services.rag.graph_rag_pipeline import GraphRAGPipeline
from src.services.rag.graph_index_store import GraphIndexStore
from src.services.rag.graph_seed_selector import GraphSeedSelector
from src.services.rag.multi_agent_orchestrator import MultiAgentOrchestrator

# GraphRAG 활성화
graph_store = GraphIndexStore()
graph_seed_selector = GraphSeedSelector(graph_store)
graph_orchestrator = MultiAgentOrchestrator(...)

graph_rag = GraphRAGPipeline(
    graph_store=graph_store,
    seed_selector=graph_seed_selector,
    orchestrator=graph_orchestrator
)

# GraphRAG 비활성화 (RAGManager에서)
rag_manager = RAGManager(
    embedding_service=embedding_service,
    graph_store=None,  # GraphRAG 비활성화
    graph_seed_selector=None,
    graph_orchestrator=None
)
```

---

## 사용 시나리오별 추천 구성

### 시나리오 1: 빠른 프로토타입 (최소 구성)
```python
# RAGManager만 사용 (가장 기본)
rag_manager = RAGManager(
    embedding_service=embedding_service,
    chromadb_mode="local"  # 로컬만 사용
)
```

### 시나리오 2: 고품질 논문 개선 (RAPTOR 활용)
```python
# AdvancedGoldenReferenceStore 사용
store = AdvancedGoldenReferenceStore(
    collection_name="golden_references",
    use_chromadb=True
)

# RAPTOR L2 (논문 요약) 사용
results = store.search(query, level="L2")
```

### 시나리오 3: 비용 절감 (로컬 LLM만)
```python
# HybridRAGService에서 Nemotron만 사용
service = HybridRAGService(
    hybrid_mode=True,
    use_gpt4_for_evaluation=False,  # 비용 절감
    use_claude_for_evaluation=False,  # 비용 절감
    use_nemotron_for_summarization=True,
    use_nemotron_for_extraction=True
)
```

### 시나리오 4: 최고 품질 (모든 시스템 활용)
```python
# 모든 RAG 시스템 활성화
rag_manager = RAGManager(
    embedding_service=embedding_service,
    chromadb_mode="auto",
    graph_store=graph_store,  # GraphRAG 포함
    graph_seed_selector=graph_seed_selector,
    graph_orchestrator=graph_orchestrator
)

learning_store = LearningStore()  # 학습 패턴
golden_store = AdvancedGoldenReferenceStore(use_chromadb=True)  # RAPTOR
hybrid_service = HybridRAGService(hybrid_mode=True)  # 하이브리드
```

### 시나리오 5: 특정 작업만 (선택적 사용)
```python
# 논문 개선 패턴만 필요
learning_store = LearningStore()
patterns = await learning_store.find_similar_improvements(query)

# 과학 논문 검색만 필요
vector_store = VectorStore()
results = await vector_store.search(query)

# 그래프 기반 검색만 필요
graph_rag = GraphRAGPipeline(...)
result = await graph_rag.run(query, agents, ...)
```

---

## API 엔드포인트에서 선택적 사용

### Hybrid RAG API
```python
# /api/v1/hybrid-rag/evaluate
# 환경변수로 제어되는 HybridRAGService 사용

# /api/v1/hybrid-rag/status
# 현재 활성화된 RAG 시스템 확인
GET /api/v1/hybrid-rag/status
```

### Paper Improvement API
```python
# RAG 사용 여부를 요청에서 지정 가능
POST /api/v1/papers/{paper_id}/improve
{
    "rag_enhanced": true,  # RAG 사용
    "rag_enhanced": false  # RAG 미사용
}
```

---

## 환경변수 전체 목록

```bash
# RAGManager
CHROMADB_MODE=auto|docker|local|disabled
CHROMADB_HOST=localhost
CHROMADB_PORT=8001
CHROMADB_PATH=./chromadb_data

# HybridRAGService
HYBRID_MODE=true|false
USE_GPT4_FOR_EVALUATION=true|false
USE_CLAUDE_FOR_EVALUATION=true|false
USE_NEMOTRON_FOR_SUMMARIZATION=true|false
USE_NEMOTRON_FOR_EXTRACTION=true|false
ENSEMBLE_WEIGHT_GPT4=0.40
ENSEMBLE_WEIGHT_CLAUDE=0.30
ENSEMBLE_WEIGHT_NEMOTRON=0.30

# VectorStore
CHROMADB_COLLECTION=scientific_papers
EMBEDDING_MODEL=allenai/scibert_scivocab_uncased
```

---

## 체크리스트: RAG 시스템 선택

### 어떤 RAG를 사용할지 결정:

- [ ] **논문 개선 패턴 학습** → `LearningStore` 또는 `RAGManager`
- [ ] **고품질 논문 참조** → `AdvancedGoldenReferenceStore` (RAPTOR)
- [ ] **과학 논문 검색** → `VectorStore`
- [ ] **멀티 LLM 라우팅** → `HybridRAGService`
- [ ] **엔티티/관계 검색** → `GraphRAG`
- [ ] **비용 절감** → `HybridRAGService` (Nemotron만)
- [ ] **최고 품질** → 모든 시스템 활성화

---

## 주의사항

1. **ChromaDB 의존성**: 대부분의 RAG 시스템은 ChromaDB가 필요합니다
2. **성능**: 여러 RAG 시스템을 동시에 사용하면 응답 시간이 증가할 수 있습니다
3. **비용**: HybridRAGService에서 GPT-4/Claude 사용 시 API 비용 발생
4. **메모리**: GraphRAG는 추가 메모리가 필요할 수 있습니다

---

## 요약

| RAG 시스템 | 활성화 방법 | 비활성화 방법 | 선택적 사용 |
|-----------|-----------|-------------|-----------|
| **RAGManager** | `chromadb_mode="auto"` | `chromadb_mode="disabled"` | ✅ 가능 |
| **LearningStore** | 자동 (ChromaDB 연결 시) | ChromaDB 미연결 | ✅ 가능 |
| **AdvancedGoldenReferenceStore** | `use_chromadb=True` | `use_chromadb=False` | ✅ 가능 |
| **VectorStore** | 자동 (ChromaDB 연결 시) | ChromaDB 미연결 | ✅ 가능 |
| **HybridRAGService** | 환경변수/코드 설정 | 환경변수로 각 기능 제어 | ✅ 가능 |
| **GraphRAG** | `graph_store` 제공 | `graph_store=None` | ✅ 가능 |

**결론**: 모든 RAG 시스템은 **선택적으로 활성화/비활성화**가 가능하며, 작업 요구사항에 따라 적절히 조합하여 사용할 수 있습니다.








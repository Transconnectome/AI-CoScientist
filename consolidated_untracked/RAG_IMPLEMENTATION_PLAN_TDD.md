# RAG 시스템 개선: TDD 기반 구현 계획 (2025)

**작성일**: 2025-01-XX  
**방법론**: Test-Driven Development (TDD)  
**우선순위**: P0 개선사항 중심  
**예상 기간**: 6주 (Phase 1-3)

---

## 📋 개요

이 문서는 AI-CoScientist RAG 시스템의 P0 개선사항을 **TDD(Test-Driven Development)** 방식으로 구현하는 상세 계획입니다.

### TDD 원칙
1. **Red**: 실패하는 테스트 먼저 작성
2. **Green**: 테스트를 통과하는 최소 코드 작성
3. **Refactor**: 코드 개선 및 최적화

### 구현 우선순위 (의존성 기반)

```
Phase 1: 평가 프레임워크 (독립적, 낮은 난이도)
  ↓
Phase 2: 컨텍스트 충분성 (평가 프레임워크 의존)
  ↓
Phase 3: 쿼리 분류기 (독립적)
  ↓
Phase 4: 적응형 검색 라우터 (쿼리 분류기 의존)
  ↓
Phase 5: RAPTOR (복잡하지만 독립적)
```

---

## 🎯 Phase 1: 평가 프레임워크 (Week 1-2)

### 목표
2025년 표준 RAG 평가 메트릭 구현: Faithfulness, Answer Relevancy, Context Precision/Recall, Context Sufficiency

### TDD 사이클

#### 1.1 Faithfulness Metric (신뢰성)

**Step 1: Red - 테스트 작성**
```python
# tests/rag/test_rag_evaluator.py

import pytest
from src.services.rag.rag_evaluator import FaithfulnessMetric

@pytest.mark.asyncio
async def test_faithfulness_high_score():
    """답변이 컨텍스트에 완전히 기반할 때 높은 점수"""
    metric = FaithfulnessMetric()
    
    context = ["The study found that X causes Y."]
    answer = "According to the study, X causes Y."
    
    score = await metric.evaluate(answer, context)
    assert score >= 0.9

@pytest.mark.asyncio
async def test_faithfulness_low_score():
    """답변이 컨텍스트에 없는 정보를 포함할 때 낮은 점수"""
    metric = FaithfulnessMetric()
    
    context = ["The study found that X causes Y."]
    answer = "The study found that X causes Y, and also Z causes W."
    
    score = await metric.evaluate(answer, context)
    assert score < 0.5

@pytest.mark.asyncio
async def test_faithfulness_empty_context():
    """컨텍스트가 없을 때 처리"""
    metric = FaithfulnessMetric()
    
    context = []
    answer = "Some answer"
    
    score = await metric.evaluate(answer, context)
    assert score == 0.0
```

**Step 2: Green - 최소 구현**
```python
# src/services/rag/rag_evaluator.py

from typing import List
from openai import AsyncOpenAI

class FaithfulnessMetric:
    """답변이 검색된 컨텍스트에 기반하는지 평가"""
    
    def __init__(self, llm_client: Optional[AsyncOpenAI] = None):
        self.llm_client = llm_client or AsyncOpenAI()
    
    async def evaluate(
        self,
        answer: str,
        context: List[str],
        threshold: float = 0.7
    ) -> float:
        """Faithfulness 점수 계산 (0.0-1.0)"""
        if not context:
            return 0.0
        
        # LLM-as-judge 방식
        prompt = self._build_prompt(answer, context)
        response = await self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        # 점수 파싱 (0.0-1.0)
        score = self._parse_score(response.choices[0].message.content)
        return score
    
    def _build_prompt(self, answer: str, context: List[str]) -> str:
        context_text = "\n".join(f"- {c}" for c in context)
        return f"""Evaluate if the answer is fully supported by the context.

Context:
{context_text}

Answer:
{answer}

Rate faithfulness from 0.0 to 1.0, where:
- 1.0: All claims in answer are directly supported by context
- 0.5: Some claims are supported, some are not
- 0.0: Answer contains claims not in context

Return only the score as a float."""
    
    def _parse_score(self, response: str) -> float:
        try:
            return float(response.strip())
        except:
            return 0.0
```

**Step 3: Refactor**
- 캐싱 추가
- 배치 처리 지원
- 에러 핸들링 강화

#### 1.2 Answer Relevancy Metric

**Step 1: Red**
```python
@pytest.mark.asyncio
async def test_answer_relevancy_high_score():
    """답변이 쿼리를 잘 해결할 때 높은 점수"""
    metric = AnswerRelevancyMetric()
    
    query = "What causes X?"
    answer = "X is caused by Y and Z factors."
    
    score = await metric.evaluate(query, answer)
    assert score >= 0.8

@pytest.mark.asyncio
async def test_answer_relevancy_low_score():
    """답변이 쿼리와 관련 없을 때 낮은 점수"""
    metric = AnswerRelevancyMetric()
    
    query = "What causes X?"
    answer = "The weather is nice today."
    
    score = await metric.evaluate(query, answer)
    assert score < 0.3
```

**Step 2: Green** (구현)

**Step 3: Refactor**

#### 1.3 Context Precision/Recall

**Step 1: Red**
```python
@pytest.mark.asyncio
async def test_context_precision():
    """검색된 컨텍스트의 관련성 평가"""
    metric = ContextPrecisionMetric()
    
    query = "machine learning"
    contexts = [
        "Machine learning is a subset of AI.",
        "The weather forecast predicts rain."
    ]
    
    score = await metric.evaluate(query, contexts)
    assert score < 0.6  # 하나는 관련, 하나는 무관
```

#### 1.4 통합 테스트

```python
# tests/rag/test_rag_evaluator_integration.py

@pytest.mark.asyncio
async def test_complete_evaluation_pipeline():
    """전체 평가 파이프라인 테스트"""
    evaluator = RAGEvaluator()
    
    result = await evaluator.evaluate(
        query="What is RAG?",
        retrieved_context=["RAG is Retrieval-Augmented Generation..."],
        generated_answer="RAG stands for Retrieval-Augmented Generation..."
    )
    
    assert result.faithfulness >= 0.7
    assert result.answer_relevancy >= 0.7
    assert result.context_precision >= 0.6
```

---

## 🎯 Phase 2: 컨텍스트 충분성 (Week 2)

### 목표
ICLR 2025 연구 기반: LLM이 답변하기에 충분한 컨텍스트인지 판단

### TDD 사이클

#### 2.1 Context Sufficiency Check

**Step 1: Red**
```python
# tests/rag/test_context_sufficiency.py

@pytest.mark.asyncio
async def test_sufficient_context():
    """충분한 컨텍스트일 때 True"""
    checker = ContextSufficiencyChecker()
    
    query = "What is the main finding?"
    context = ["The main finding is that X causes Y in 80% of cases."]
    
    is_sufficient = await checker.check(query, context)
    assert is_sufficient is True

@pytest.mark.asyncio
async def test_insufficient_context():
    """부족한 컨텍스트일 때 False"""
    checker = ContextSufficiencyChecker()
    
    query = "What is the main finding?"
    context = ["The study was conducted."]
    
    is_sufficient = await checker.check(query, context)
    assert is_sufficient is False

@pytest.mark.asyncio
async def test_sufficiency_with_expansion():
    """부족할 때 확장 제안"""
    checker = ContextSufficiencyChecker()
    
    query = "What are the side effects?"
    context = ["The drug was tested."]
    
    result = await checker.check_with_expansion(query, context)
    assert result.is_sufficient is False
    assert len(result.suggested_queries) > 0
```

**Step 2: Green**
```python
# src/services/rag/context_sufficiency.py

class ContextSufficiencyChecker:
    """컨텍스트 충분성 검사 (ICLR 2025)"""
    
    async def check(
        self,
        query: str,
        context: List[str],
        threshold: float = 0.7
    ) -> bool:
        """컨텍스트가 충분한지 판단"""
        # LLM 기반 충분성 판단
        prompt = self._build_sufficiency_prompt(query, context)
        response = await self.llm_client.chat.completions.create(...)
        
        score = self._parse_sufficiency_score(response)
        return score >= threshold
    
    async def check_with_expansion(
        self,
        query: str,
        context: List[str]
    ) -> SufficiencyResult:
        """충분성 검사 + 확장 제안"""
        is_sufficient = await self.check(query, context)
        
        if not is_sufficient:
            suggested_queries = await self._suggest_expansions(query, context)
            return SufficiencyResult(
                is_sufficient=False,
                confidence=0.0,
                suggested_queries=suggested_queries
            )
        
        return SufficiencyResult(is_sufficient=True, confidence=1.0)
```

---

## 🎯 Phase 3: 쿼리 분류기 (Week 3)

### 목표
쿼리 타입 분류: factual, multi-hop, hierarchical, comparative

### TDD 사이클

#### 3.1 Query Classifier

**Step 1: Red**
```python
# tests/rag/test_query_classifier.py

@pytest.mark.asyncio
async def test_classify_factual_query():
    """사실적 쿼리 분류"""
    classifier = QueryClassifier()
    
    query = "What is machine learning?"
    result = await classifier.classify(query)
    
    assert result.query_type == QueryType.FACTUAL
    assert result.confidence >= 0.8

@pytest.mark.asyncio
async def test_classify_multi_hop_query():
    """다중 홉 쿼리 분류"""
    classifier = QueryClassifier()
    
    query = "What methodologies are used in papers that cite X?"
    result = await classifier.classify(query)
    
    assert result.query_type == QueryType.MULTI_HOP
    assert result.confidence >= 0.7

@pytest.mark.asyncio
async def test_classify_hierarchical_query():
    """계층적 쿼리 분류"""
    classifier = QueryClassifier()
    
    query = "What are the main themes across this research program?"
    result = await classifier.classify(query)
    
    assert result.query_type == QueryType.HIERARCHICAL
```

**Step 2: Green**
```python
# src/services/rag/query_classifier.py

from enum import Enum
from dataclasses import dataclass

class QueryType(Enum):
    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    HIERARCHICAL = "hierarchical"
    COMPARATIVE = "comparative"
    UNKNOWN = "unknown"

@dataclass
class QueryClassification:
    query_type: QueryType
    confidence: float
    reasoning: str

class QueryClassifier:
    """쿼리 타입 분류"""
    
    async def classify(self, query: str) -> QueryClassification:
        """쿼리 분류"""
        prompt = self._build_classification_prompt(query)
        response = await self.llm_client.chat.completions.create(...)
        
        return self._parse_classification(response)
```

---

## 🎯 Phase 4: 적응형 검색 라우터 (Week 4-5)

### 목표
쿼리 타입에 따라 최적의 검색 전략 선택

### TDD 사이클

#### 4.1 Adaptive Retrieval Router

**Step 1: Red**
```python
# tests/rag/test_adaptive_router.py

@pytest.mark.asyncio
async def test_route_factual_query():
    """사실적 쿼리는 Dense 검색으로 라우팅"""
    router = AdaptiveRetrievalRouter()
    
    query = "What is machine learning?"
    strategy = await router.route(query)
    
    assert isinstance(strategy, DenseRetrievalStrategy)
    assert strategy.top_k == 5

@pytest.mark.asyncio
async def test_route_multi_hop_query():
    """다중 홉 쿼리는 Graph 검색으로 라우팅"""
    router = AdaptiveRetrievalRouter()
    
    query = "What methodologies are used in papers that cite X?"
    strategy = await router.route(query)
    
    assert isinstance(strategy, GraphRetrievalStrategy)
    assert strategy.max_depth == 3

@pytest.mark.asyncio
async def test_route_hierarchical_query():
    """계층적 쿼리는 RAPTOR 검색으로 라우팅"""
    router = AdaptiveRetrievalRouter()
    
    query = "What are the main themes?"
    strategy = await router.route(query)
    
    assert isinstance(strategy, RAPTORRetrievalStrategy)
    assert strategy.levels == [0, 1, 2]
```

**Step 2: Green**
```python
# src/services/rag/adaptive_router.py

class AdaptiveRetrievalRouter:
    """쿼리 의존적 검색 라우팅"""
    
    def __init__(
        self,
        classifier: QueryClassifier,
        dense_strategy: DenseRetrievalStrategy,
        graph_strategy: GraphRetrievalStrategy,
        raptor_strategy: RAPTORRetrievalStrategy,
        hybrid_strategy: HybridRetrievalStrategy
    ):
        self.classifier = classifier
        self.strategies = {
            QueryType.FACTUAL: dense_strategy,
            QueryType.MULTI_HOP: graph_strategy,
            QueryType.HIERARCHICAL: raptor_strategy,
            QueryType.COMPARATIVE: hybrid_strategy
        }
    
    async def route(self, query: str) -> RetrievalStrategy:
        """쿼리에 맞는 검색 전략 선택"""
        classification = await self.classifier.classify(query)
        strategy = self.strategies.get(
            classification.query_type,
            self.strategies[QueryType.FACTUAL]  # 기본값
        )
        return strategy
```

---

## 🎯 Phase 5: RAPTOR 구현 (Week 5-6)

### 목표
계층적 트리 구조: 재귀적 클러스터링 + 추상적 요약

### TDD 사이클

#### 5.1 RAPTOR Indexer

**Step 1: Red**
```python
# tests/rag/test_raptor_indexer.py

@pytest.mark.asyncio
async def test_build_raptor_tree():
    """RAPTOR 트리 구축"""
    indexer = RAPTORIndexer()
    
    documents = [
        Document(id="doc1", text="Long document about machine learning..."),
        Document(id="doc2", text="Another document about deep learning...")
    ]
    
    tree = await indexer.build_tree(documents, num_levels=3)
    
    assert tree.num_levels == 3
    assert len(tree.get_nodes(level=0)) > 0  # 원본 청크
    assert len(tree.get_nodes(level=1)) > 0  # 클러스터 요약
    assert len(tree.get_nodes(level=2)) > 0  # 추상 요약

@pytest.mark.asyncio
async def test_raptor_retrieval():
    """RAPTOR 계층적 검색"""
    retriever = RAPTORRetriever(tree)
    
    query = "What are the main themes?"
    results = await retriever.retrieve(query, levels=[0, 1, 2])
    
    assert len(results) > 0
    assert all(r.level in [0, 1, 2] for r in results)
```

**Step 2: Green**
```python
# src/services/rag/raptor_indexer.py

class RAPTORIndexer:
    """RAPTOR 계층적 트리 인덱서"""
    
    async def build_tree(
        self,
        documents: List[Document],
        num_levels: int = 3,
        cluster_size: int = 5
    ) -> RAPTORTree:
        """재귀적으로 트리 구축"""
        # Level 0: 원본 청크
        chunks = self._chunk_documents(documents)
        level_0_nodes = [TreeNode(level=0, content=c) for c in chunks]
        
        # Level 1+: 클러스터링 + 요약
        current_level = level_0_nodes
        tree = RAPTORTree()
        tree.add_level(0, current_level)
        
        for level in range(1, num_levels + 1):
            # 클러스터링
            clusters = await self._cluster_nodes(current_level, cluster_size)
            
            # 요약 생성
            summaries = await self._summarize_clusters(clusters)
            level_nodes = [TreeNode(level=level, content=s) for s in summaries]
            
            tree.add_level(level, level_nodes)
            current_level = level_nodes
        
        return tree
    
    async def _cluster_nodes(
        self,
        nodes: List[TreeNode],
        cluster_size: int
    ) -> List[List[TreeNode]]:
        """노드 클러스터링 (k-means 또는 hierarchical)"""
        # 임베딩 생성
        embeddings = await self._embed_nodes(nodes)
        
        # 클러스터링
        from sklearn.cluster import KMeans
        n_clusters = len(nodes) // cluster_size
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(embeddings)
        
        # 클러스터별로 그룹화
        clusters = [[] for _ in range(n_clusters)]
        for node, label in zip(nodes, labels):
            clusters[label].append(node)
        
        return clusters
    
    async def _summarize_clusters(
        self,
        clusters: List[List[TreeNode]]
    ) -> List[str]:
        """클러스터별 추상적 요약 생성"""
        summaries = []
        for cluster in clusters:
            cluster_text = "\n".join([n.content for n in cluster])
            summary = await self._generate_summary(cluster_text)
            summaries.append(summary)
        return summaries
```

---

## 📁 파일 구조

```
src/services/rag/
├── __init__.py
├── rag_evaluator.py          # Phase 1: 평가 프레임워크
│   ├── FaithfulnessMetric
│   ├── AnswerRelevancyMetric
│   ├── ContextPrecisionMetric
│   ├── ContextRecallMetric
│   └── RAGEvaluator (통합)
├── context_sufficiency.py    # Phase 2: 컨텍스트 충분성
│   ├── ContextSufficiencyChecker
│   └── SufficiencyResult
├── query_classifier.py       # Phase 3: 쿼리 분류
│   ├── QueryClassifier
│   ├── QueryType (Enum)
│   └── QueryClassification
├── adaptive_router.py         # Phase 4: 적응형 라우팅
│   ├── AdaptiveRetrievalRouter
│   ├── RetrievalStrategy (ABC)
│   ├── DenseRetrievalStrategy
│   ├── GraphRetrievalStrategy
│   ├── RAPTORRetrievalStrategy
│   └── HybridRetrievalStrategy
└── raptor_indexer.py         # Phase 5: RAPTOR
    ├── RAPTORIndexer
    ├── RAPTORTree
    ├── TreeNode
    └── RAPTORRetriever

tests/rag/
├── __init__.py
├── test_rag_evaluator.py              # Phase 1 테스트
├── test_rag_evaluator_integration.py  # Phase 1 통합 테스트
├── test_context_sufficiency.py        # Phase 2 테스트
├── test_query_classifier.py          # Phase 3 테스트
├── test_adaptive_router.py           # Phase 4 테스트
└── test_raptor_indexer.py            # Phase 5 테스트
```

---

## 🔄 TDD 워크플로우

### 각 Phase별 진행 순서

1. **테스트 작성** (Red)
   ```bash
   # 테스트 파일 생성
   touch tests/rag/test_new_feature.py
   
   # 테스트 작성 (실패하는 테스트)
   # pytest 실행 → 실패 확인
   pytest tests/rag/test_new_feature.py -v
   ```

2. **최소 구현** (Green)
   ```bash
   # 구현 파일 생성
   touch src/services/rag/new_feature.py
   
   # 최소 코드로 테스트 통과
   # pytest 실행 → 통과 확인
   pytest tests/rag/test_new_feature.py -v
   ```

3. **리팩토링** (Refactor)
   ```bash
   # 코드 개선
   # 테스트 계속 통과 확인
   pytest tests/rag/test_new_feature.py -v
   
   # 커버리지 확인
   pytest tests/rag/test_new_feature.py --cov=src/services/rag/new_feature
   ```

### 일일 TDD 사이클

```
Morning:
1. 오늘 구현할 기능 선택
2. 테스트 작성 (Red) - 30분
3. 최소 구현 (Green) - 1-2시간
4. 테스트 통과 확인

Afternoon:
5. 리팩토링 (Refactor) - 1-2시간
6. 통합 테스트 작성
7. 문서화
```

---

## 📊 테스트 커버리지 목표

- **Unit Tests**: 90%+ 커버리지
- **Integration Tests**: 모든 주요 통합 지점
- **E2E Tests**: 전체 파이프라인 검증

### 커버리지 확인

```bash
# 전체 커버리지
pytest --cov=src/services/rag --cov-report=html --cov-report=term

# 특정 모듈
pytest tests/rag/test_rag_evaluator.py --cov=src/services/rag/rag_evaluator

# 커버리지 리포트 확인
open htmlcov/index.html
```

---

## 🎯 성공 기준

### Phase 1 (평가 프레임워크)
- ✅ 모든 평가 메트릭 구현
- ✅ 테스트 커버리지 90%+
- ✅ 통합 테스트 통과

### Phase 2 (컨텍스트 충분성)
- ✅ 충분성 검사 구현
- ✅ 확장 제안 기능
- ✅ 평가 프레임워크와 통합

### Phase 3 (쿼리 분류)
- ✅ 4가지 쿼리 타입 분류
- ✅ 정확도 85%+
- ✅ 테스트 커버리지 90%+

### Phase 4 (적응형 라우팅)
- ✅ 모든 전략 구현
- ✅ 라우팅 정확도 90%+
- ✅ 성능 테스트 통과

### Phase 5 (RAPTOR)
- ✅ 3-level 트리 구축
- ✅ 계층적 검색 구현
- ✅ 벤치마크: +20% 검색 정확도

---

## 🚀 실행 계획

### Week 1: 평가 프레임워크
- Day 1-2: Faithfulness, Answer Relevancy
- Day 3-4: Context Precision/Recall
- Day 5: 통합 테스트 및 문서화

### Week 2: 컨텍스트 충분성
- Day 1-2: 충분성 검사 구현
- Day 3-4: 확장 제안 기능
- Day 5: 통합 및 테스트

### Week 3: 쿼리 분류기
- Day 1-2: 분류기 구현
- Day 3-4: 정확도 개선
- Day 5: 통합 테스트

### Week 4-5: 적응형 라우팅
- Week 4: 전략 구현
- Week 5: 라우팅 로직 및 통합

### Week 6: RAPTOR
- Day 1-3: 인덱서 구현
- Day 4-5: 검색 및 벤치마크

---

## 📝 체크리스트

### 각 Phase 완료 시
- [ ] 모든 단위 테스트 통과
- [ ] 통합 테스트 통과
- [ ] 테스트 커버리지 90%+
- [ ] 문서화 완료
- [ ] 코드 리뷰 완료
- [ ] 성능 벤치마크 통과

### 최종 완료 시
- [ ] 모든 Phase 완료
- [ ] E2E 테스트 통과
- [ ] 전체 시스템 통합
- [ ] 성능 목표 달성
- [ ] 문서화 완료

---

## 🔗 참고문헌

1. Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." arXiv:2401.18059
2. Google Research (2025). "Sufficient Context: A New Lens on Retrieval Augmented Generation Systems." ICLR 2025
3. Gan et al. (2025). "Retrieval Augmented Generation Evaluation in the Era of Large Language Models."

---

**다음 단계**: Phase 1 테스트 작성부터 시작!


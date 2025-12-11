# RAG 시스템 개선: TDD 기반 구현 계획 (한글 요약)

**작성일**: 2025-01-XX  
**방법론**: Test-Driven Development (TDD)  
**기간**: 6주 (Phase 1-5)

---

## 📋 개요

이 문서는 AI-CoScientist RAG 시스템의 P0 개선사항을 **TDD 방식**으로 구현하는 계획입니다.

### TDD 원칙
1. **Red**: 실패하는 테스트 먼저 작성 ✅ (완료)
2. **Green**: 테스트를 통과하는 최소 코드 작성 (다음 단계)
3. **Refactor**: 코드 개선 및 최적화

---

## 🎯 구현 우선순위

```
Phase 1: 평가 프레임워크 (Week 1-2) ← 현재 여기
  ↓
Phase 2: 컨텍스트 충분성 (Week 2)
  ↓
Phase 3: 쿼리 분류기 (Week 3)
  ↓
Phase 4: 적응형 검색 라우터 (Week 4-5)
  ↓
Phase 5: RAPTOR (Week 6)
```

---

## ✅ 완료된 작업

### 1. 문서화
- ✅ `RAG_IMPLEMENTATION_PLAN_TDD.md` - 상세 구현 계획 (영문)
- ✅ `RAG_TDD_QUICK_START.md` - 빠른 시작 가이드
- ✅ `RAG_TDD_구현_계획_한글.md` - 이 문서

### 2. 테스트 작성 (Red 단계) ✅
- ✅ `tests/rag/test_rag_evaluator.py` - 평가 프레임워크 테스트
- ✅ `tests/rag/test_context_sufficiency.py` - 컨텍스트 충분성 테스트
- ✅ `tests/rag/test_query_classifier.py` - 쿼리 분류기 테스트

**현재 상태**: 테스트는 실패합니다 (의도된 것 - Red 단계)

---

## 🚀 다음 단계: Green 단계 (구현)

### Phase 1: 평가 프레임워크 구현

#### Step 1: FaithfulnessMetric 구현

**파일 생성**:
```bash
touch src/services/rag/rag_evaluator.py
```

**최소 구현**:
```python
# src/services/rag/rag_evaluator.py

from typing import List, Optional
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
        
        score = self._parse_score(response.choices[0].message.content)
        return score
    
    def _build_prompt(self, answer: str, context: List[str]) -> str:
        context_text = "\n".join(f"- {c}" for c in context)
        return f"""Evaluate if the answer is fully supported by the context.

Context:
{context_text}

Answer:
{answer}

Rate faithfulness from 0.0 to 1.0.
Return only the score as a float."""
    
    def _parse_score(self, response: str) -> float:
        try:
            return float(response.strip())
        except:
            return 0.0
```

**테스트 실행**:
```bash
pytest tests/rag/test_rag_evaluator.py::TestFaithfulnessMetric -v
# → 통과해야 함
```

---

## 📁 생성된 파일 구조

```
프로젝트 루트/
├── RAG_IMPLEMENTATION_PLAN_TDD.md      # 상세 구현 계획 (영문)
├── RAG_TDD_QUICK_START.md              # 빠른 시작 가이드
├── RAG_TDD_구현_계획_한글.md           # 이 문서
├── IMPLEMENTATION_STATUS.md            # 구현 현황 추적
│
├── src/services/rag/
│   ├── rag_evaluator.py                # Phase 1: 구현 필요
│   ├── context_sufficiency.py          # Phase 2: 구현 필요
│   ├── query_classifier.py             # Phase 3: 구현 필요
│   ├── adaptive_router.py              # Phase 4: 구현 필요
│   └── raptor_indexer.py              # Phase 5: 구현 필요
│
└── tests/rag/
    ├── test_rag_evaluator.py           # ✅ 작성 완료
    ├── test_context_sufficiency.py    # ✅ 작성 완료
    └── test_query_classifier.py       # ✅ 작성 완료
```

---

## 🎯 구현 로드맵

### Week 1: 평가 프레임워크
- **Day 1-2**: FaithfulnessMetric, AnswerRelevancyMetric
- **Day 3-4**: ContextPrecisionMetric, ContextRecallMetric
- **Day 5**: 통합 테스트 및 문서화

### Week 2: 컨텍스트 충분성
- **Day 1-2**: ContextSufficiencyChecker 구현
- **Day 3-4**: 확장 제안 기능
- **Day 5**: 통합 및 테스트

### Week 3: 쿼리 분류기
- **Day 1-2**: QueryClassifier 구현
- **Day 3-4**: 정확도 개선
- **Day 5**: 통합 테스트

### Week 4-5: 적응형 라우팅
- **Week 4**: 전략 구현
- **Week 5**: 라우팅 로직 및 통합

### Week 6: RAPTOR
- **Day 1-3**: 인덱서 구현
- **Day 4-5**: 검색 및 벤치마크

---

## 📊 성공 기준

### Phase 1 완료 시
- ✅ 모든 평가 메트릭 구현
- ✅ 모든 단위 테스트 통과
- ✅ 테스트 커버리지 90%+
- ✅ 통합 테스트 통과

### 전체 완료 시
- ✅ Phase 1-5 모두 완료
- ✅ E2E 테스트 통과
- ✅ 성능 벤치마크: +20% 검색 정확도
- ✅ 전체 문서화 완료

---

## 🚀 시작하기

### 1. 현재 상태 확인
```bash
# 테스트 실행 (실패 예상 - Red 단계)
pytest tests/rag/test_rag_evaluator.py -v
```

### 2. 첫 번째 구현 시작
```bash
# 구현 파일 생성
touch src/services/rag/rag_evaluator.py

# FaithfulnessMetric 구현
# (위의 코드 참고)

# 테스트 통과 확인
pytest tests/rag/test_rag_evaluator.py::TestFaithfulnessMetric -v
```

### 3. 커버리지 확인
```bash
pytest tests/rag/ --cov=src/services/rag --cov-report=term
```

---

## 📝 체크리스트

### Phase 1 (평가 프레임워크)
- [ ] FaithfulnessMetric 구현
- [ ] AnswerRelevancyMetric 구현
- [ ] ContextPrecisionMetric 구현
- [ ] ContextRecallMetric 구현
- [ ] RAGEvaluator 통합 클래스
- [ ] 모든 테스트 통과
- [ ] 커버리지 90%+

### Phase 2 (컨텍스트 충분성)
- [ ] ContextSufficiencyChecker 구현
- [ ] 확장 제안 기능
- [ ] 통합 테스트

### Phase 3 (쿼리 분류기)
- [ ] QueryClassifier 구현
- [ ] 4가지 타입 분류
- [ ] 정확도 85%+

### Phase 4 (적응형 라우팅)
- [ ] AdaptiveRetrievalRouter 구현
- [ ] 모든 전략 구현
- [ ] 라우팅 정확도 90%+

### Phase 5 (RAPTOR)
- [ ] RAPTORIndexer 구현
- [ ] 3-level 트리 구축
- [ ] 계층적 검색
- [ ] 벤치마크 통과

---

## 🔗 관련 문서

1. **상세 계획**: `RAG_IMPLEMENTATION_PLAN_TDD.md`
2. **빠른 시작**: `RAG_TDD_QUICK_START.md`
3. **구현 현황**: `IMPLEMENTATION_STATUS.md`
4. **평가 요약**: `RAG_평가_요약_한글.md`

---

## 💡 팁

### TDD 사이클
1. **작은 단위로**: 한 번에 하나의 메트릭만 구현
2. **테스트 먼저**: 항상 테스트를 먼저 작성
3. **최소 구현**: 테스트를 통과하는 최소한의 코드만
4. **리팩토링**: 테스트가 통과한 후에만 개선

### 디버깅
```bash
# 특정 테스트만 실행
pytest tests/rag/test_rag_evaluator.py::TestFaithfulnessMetric::test_faithfulness_high_score -v

# 상세 출력
pytest tests/rag/test_rag_evaluator.py -v -s

# 실패한 테스트만 재실행
pytest tests/rag/test_rag_evaluator.py --lf
```

---

**다음 단계**: `src/services/rag/rag_evaluator.py` 파일을 생성하고 FaithfulnessMetric부터 구현 시작!


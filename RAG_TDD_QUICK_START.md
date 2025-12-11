# RAG TDD 구현 빠른 시작 가이드

## 🚀 시작하기

### 1. 테스트 실행 (현재 상태 확인)

```bash
# Phase 1 테스트 실행 (실패 예상 - Red 단계)
pytest tests/rag/test_rag_evaluator.py -v

# Phase 2 테스트 실행
pytest tests/rag/test_context_sufficiency.py -v

# Phase 3 테스트 실행
pytest tests/rag/test_query_classifier.py -v
```

### 2. 첫 번째 구현: Faithfulness Metric

**Step 1: 테스트 확인 (Red)**
```bash
pytest tests/rag/test_rag_evaluator.py::TestFaithfulnessMetric -v
# → 실패 예상 (아직 구현 안 됨)
```

**Step 2: 최소 구현 (Green)**
```python
# src/services/rag/rag_evaluator.py 생성
# 최소 코드로 테스트 통과하도록 구현
```

**Step 3: 테스트 통과 확인**
```bash
pytest tests/rag/test_rag_evaluator.py::TestFaithfulnessMetric -v
# → 통과 확인
```

**Step 4: 리팩토링**
- 코드 개선
- 테스트 계속 통과 확인

## 📋 구현 순서

### Week 1: 평가 프레임워크

**Day 1-2: Faithfulness & Answer Relevancy**
```bash
# 1. 테스트 작성 완료 ✅
# 2. 구현 시작
touch src/services/rag/rag_evaluator.py

# 3. 최소 구현
# 4. 테스트 통과 확인
pytest tests/rag/test_rag_evaluator.py::TestFaithfulnessMetric -v
pytest tests/rag/test_rag_evaluator.py::TestAnswerRelevancyMetric -v
```

**Day 3-4: Context Precision/Recall**
```bash
pytest tests/rag/test_rag_evaluator.py::TestContextPrecisionMetric -v
pytest tests/rag/test_rag_evaluator.py::TestContextRecallMetric -v
```

**Day 5: 통합 테스트**
```bash
pytest tests/rag/test_rag_evaluator.py::TestRAGEvaluatorIntegration -v
```

## 🎯 다음 단계

각 Phase 완료 후:
1. 테스트 커버리지 확인
2. 문서화 업데이트
3. 다음 Phase로 진행

```bash
# 커버리지 확인
pytest tests/rag/ --cov=src/services/rag --cov-report=term

# 전체 테스트 실행
pytest tests/rag/ -v
```


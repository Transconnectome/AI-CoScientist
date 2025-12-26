# 🚀 RAG Enhancement Workflow - Quick Start Guide

## 즉시 시작하기

### 1. 환경 설정 (5분)
```bash
# 워크플로우 환경 자동 설정
./start_workflow.sh

# 또는 수동 설정
poetry install --with dev
poetry add click rich pyyaml
chmod +x scripts/workflow_automation.py
```

### 2. 현재 상태 확인
```bash
# 전체 워크플로우 상태 보기
python3 scripts/workflow_automation.py status

# 설정 파일 확인
cat workflow_config.yaml
```

### 3. 첫 번째 작업 시작
```bash
# RAGAS 통합 작업 시작
python3 scripts/workflow_automation.py start ragas_integration

# 파일 구조 확인
ls -la src/services/rag/rag_evaluator.py
```

## 📋 Phase 1 실행 계획 (2주)

### Week 1: 평가 프레임워크 (Sprint 1.1)

#### Day 1-2: RAGAS 통합
```bash
# 1. 작업 시작
python3 scripts/workflow_automation.py start ragas_integration

# 2. 구현 파일 편집
code src/services/rag/rag_evaluator.py

# 3. 테스트 작성
code tests/rag/test_rag_evaluation.py

# 4. 실행 및 검증
poetry run pytest tests/rag/test_rag_evaluation.py -v

# 5. 작업 완료
python3 scripts/workflow_automation.py complete ragas_integration
```

**구현 체크리스트**:
- [ ] RAGAS 라이브러리 설치: `poetry add ragas`
- [ ] faithfulness 메트릭 구현
- [ ] answer_relevancy 메트릭 구현
- [ ] context_precision 메트릭 구현
- [ ] 기존 RAG 시스템과 통합
- [ ] 테스트 커버리지 ≥ 80%

#### Day 3-4: 벤치마크 데이터셋
```bash
# 1. 벤치마크 작업 시작
python3 scripts/workflow_automation.py start benchmark_dataset

# 2. 골든 QA 쌍 생성
code data/evaluation/rag_benchmark.json

# 3. 자동 생성 스크립트
code scripts/build_benchmark_dataset.py

# 4. 데이터셋 검증
python3 scripts/build_benchmark_dataset.py --validate
```

**데이터셋 구조 예시**:
```json
{
  "qa_pairs": [
    {
      "id": "neuro_001",
      "domain": "neuroscience",
      "difficulty": "complex",
      "question": "What are the key differences between ASD and ADHD connectivity patterns in fMRI studies?",
      "golden_answer": "Research shows distinct differences...",
      "context_documents": ["doc1.pdf", "doc2.pdf"],
      "keywords": ["fMRI", "connectivity", "ASD", "ADHD"]
    }
  ]
}
```

#### Day 5: 모니터링 인프라
```bash
# 1. 모니터링 작업 시작
python3 scripts/workflow_automation.py start monitoring_infrastructure

# 2. Prometheus 메트릭 구현
code src/monitoring/rag_metrics.py

# 3. Grafana 대시보드 설정
code monitoring/grafana/dashboards/rag_dashboard.json
```

### Week 2: 통합 오케스트레이터 (Sprint 1.2)

#### Day 1-3: 핵심 오케스트레이터
```bash
# 1. 오케스트레이터 작업 시작
python3 scripts/workflow_automation.py start unified_orchestrator

# 2. 메인 로직 구현
code src/services/rag/unified_rag_orchestrator.py

# 3. 쿼리 분류기 구현
python3 scripts/workflow_automation.py start query_classifier
code src/services/rag/advanced_query_classifier.py
```

**오케스트레이터 핵심 기능**:
```python
class UnifiedRAGOrchestrator:
    def __init__(self):
        self.strategies = {
            'basic': BasicRAG(),
            'raptor': AdvancedGoldenReference(),
            'graph': GraphRAG(),
            'dd_multimodal': EnhancedDDRaptor(),
            'qml_math': QuantERAAgent()
        }

    async def route_query(self, query: str, domain: str) -> RAGResult:
        complexity = await self.classify_complexity(query)
        strategy = self.select_strategy(domain, complexity)
        return await self.strategies[strategy].search(query)
```

#### Day 4-5: API 및 테스트
```bash
# 1. API 엔드포인트 구현
python3 scripts/workflow_automation.py start unified_api
code src/api/v1/unified_rag.py

# 2. 통합 테스트 작성
python3 scripts/workflow_automation.py start integration_tests
code tests/rag/test_unified_orchestrator.py

# 3. 전체 테스트 실행
poetry run pytest tests/rag/ -v --cov=src/services/rag
```

## 🎯 성공 기준 및 검증

### Sprint 1.1 완료 체크리스트
```bash
# 1. 평가 메트릭 검증
python3 -c "from src.services.rag.rag_evaluator import RAGEvaluator; print('✅ RAGAS 통합 성공')"

# 2. 벤치마크 데이터셋 검증
python3 scripts/build_benchmark_dataset.py --validate

# 3. 모니터링 동작 확인
curl http://localhost:8000/metrics | grep rag_

# 4. Sprint 검증 실행
python3 scripts/workflow_automation.py validate_sprint sprint1_1
```

### Sprint 1.2 완료 체크리스트
```bash
# 1. 오케스트레이터 기능 테스트
python3 -c "
from src.services.rag.unified_rag_orchestrator import UnifiedRAGOrchestrator
orch = UnifiedRAGOrchestrator()
result = orch.route_query('What is quantum advantage?', 'quantum_ml')
print(f'✅ 라우팅 성공: {result.strategy_used}')
"

# 2. API 엔드포인트 테스트
curl -X POST http://localhost:8000/api/v1/unified_rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "autism fMRI connectivity", "domain": "neuroscience"}'

# 3. 성능 개선 확인
python3 scripts/performance_benchmark.py --compare-baseline
```

## 🔍 문제 해결 가이드

### 일반적인 이슈들

#### 1. RAGAS 설치 오류
```bash
# RAGAS 의존성 충돌 해결
poetry remove ragas
poetry add ragas==0.3.7 --python "^3.11"

# 또는 조건부 설치
pip install ragas[all]
```

#### 2. ChromaDB 연결 오류
```bash
# ChromaDB 서비스 확인
docker ps | grep chroma

# 재시작 필요시
docker-compose restart chromadb
```

#### 3. 메모리 부족 오류
```bash
# 임베딩 배치 크기 조정
export EMBEDDING_BATCH_SIZE=16  # 기본값: 32

# 또는 설정 파일 수정
echo "embedding_batch_size: 16" >> config/rag_config.yaml
```

#### 4. 테스트 실패
```bash
# 특정 테스트만 실행
poetry run pytest tests/rag/test_rag_evaluation.py::test_faithfulness_metric -v

# 디버그 모드로 실행
poetry run pytest tests/rag/ -v -s --pdb
```

## 📊 진행 상황 추적

### 일일 체크인
```bash
# 매일 실행할 명령어들
python3 scripts/workflow_automation.py status
python3 scripts/workflow_automation.py generate_report
git status
```

### 주간 리뷰
```bash
# 주간 성과 리포트 생성
python3 scripts/workflow_automation.py generate_report > weekly_report.json

# 성능 벤치마크 실행
python3 scripts/performance_benchmark.py --full-suite

# 코드 품질 체크
poetry run ruff check src/
poetry run mypy src/
```

## 🎯 다음 단계 (Phase 2 준비)

Phase 1 완료 후:

```bash
# Phase 2 브랜치 생성
git checkout -b rag-enhancement-phase2

# Phase 2 워크플로우 설정 업데이트
python3 scripts/workflow_automation.py --phase phase2

# 적응형 검색 작업 시작
python3 scripts/workflow_automation.py start adaptive_retriever
```

## 🔗 참고 자료

- **RAGAS 문서**: https://docs.ragas.io/
- **ChromaDB 가이드**: https://docs.trychroma.com/
- **FastAPI 문서**: https://fastapi.tiangolo.com/
- **Neo4j Python 드라이버**: https://neo4j.com/docs/python-manual/

## 💬 지원 및 질문

문제가 발생하면:

1. **워크플로우 상태 확인**: `python3 scripts/workflow_automation.py status`
2. **로그 확인**: `tail -f logs/rag_enhancement.log`
3. **이슈 생성**: GitHub Issues에 상세한 오류 정보와 함께 보고
4. **문서 업데이트**: 해결책을 찾으면 이 가이드 업데이트

---

**시작하려면**: `./start_workflow.sh`를 실행하고 가이드를 따라하세요!

🚀 **성공적인 RAG 향상을 위해 화이팅!**
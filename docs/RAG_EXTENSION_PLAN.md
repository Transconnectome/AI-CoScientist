# 심리학과 RAG 시스템 확장 계획

## 🎯 기본 전략: 기존 시스템 최대한 활용!

### ✅ 그대로 사용할 기존 컴포넌트

#### 1. UnifiedRAGOrchestrator (핵심 엔진)
```python
# 기존 오케스트레이터에 새 전략만 추가
class UnifiedRAGOrchestrator:
    strategies = {
        'hybrid': HybridRAGStrategy,           # ✅ 그대로 사용
        'enhanced_dd_raptor': DDRAPTORStrategy, # ✅ 그대로 사용
        'graph_rag': GraphRAGStrategy,         # ✅ 그대로 사용
        'golden_reference': GoldenRefStrategy,  # ✅ 그대로 사용
        'multimodal_rag': MultimodalStrategy,  # ✅ 그대로 사용
        'psychology_specialized': PsychologyRAGStrategy  # 🆕 새로 추가
    }
```

#### 2. ChromaDB Collections (기존 DB 확장)
```python
# 기존 ChromaDB에 새로운 컬렉션만 추가
existing_collections = [
    "research_documents",      # ✅ 기존
    "improvement_patterns",    # ✅ 기존
    "dd_papers"               # ✅ 기존 (발달장애 논문)
]

new_psychology_collections = [
    "psychology_papers",       # 🆕 심리학과 논문
    "psychology_metadata",     # 🆕 메타데이터
    "psychology_glossary"      # 🆕 전문용어
]
```

#### 3. RAGAS 평가 시스템 (그대로 활용)
```python
# 기존 평가 시스템을 심리학 데이터로 확장
from src.services.rag.rag_evaluator import create_rag_evaluator

# 심리학용 QA 벤치마크만 새로 생성
psychology_qa_benchmark = [
    {
        "question": "인지편향 연구의 최신 동향은?",
        "ground_truth": "안우영 교수의 2023년 연구에서...",
        "contexts": ["관련 논문들..."]
    }
    # ... 더 많은 QA 쌍들
]

# 기존 evaluator로 그대로 평가 가능
evaluator = create_rag_evaluator(enable_ragas=True)
results = await evaluator.evaluate(psychology_qa_benchmark)
```

### 🆕 새로 개발할 심리학 특화 부분

#### 1. Psychology RAG Strategy (신규)
```python
# src/services/rag/psychology_rag_strategy.py
from src.services.rag.base import RAGStrategy  # 기존 베이스 클래스 활용

class PsychologyRAGStrategy(RAGStrategy):
    """심리학 특화 RAG 전략 - 기존 베이스 클래스 상속"""

    name = "psychology_specialized"

    def __init__(self):
        super().__init__()  # 기존 베이스 기능 모두 활용
        self.korean_embedder = self._load_korean_model()
        self.psychology_terms = self._load_psych_glossary()

    async def retrieve(self, query: str, **kwargs):
        # 1. 기존 하이브리드 검색 활용
        base_results = await super().hybrid_retrieve(query)

        # 2. 심리학 특화 후처리만 추가
        enhanced_results = self._enhance_psychology_results(base_results)

        return enhanced_results

    def _enhance_psychology_results(self, results):
        """심리학 특화 결과 향상 - 기존 결과를 개선"""
        # 한국어 용어 매핑
        # 연구 분야별 가중치
        # 교수진별 전문성 고려
        pass
```

#### 2. Korean NLP Pipeline (신규)
```python
# src/services/psychology/korean_nlp.py
class KoreanPsychologyNLP:
    """한국어 심리학 전문 NLP - 완전히 새로운 모듈"""

    def __init__(self):
        self.kobert = AutoModel.from_pretrained("monologg/kobert")
        self.psychology_dict = self._load_psychology_dictionary()

    def enhance_query(self, query: str) -> str:
        """한국어 쿼리를 영어로 확장"""
        # "인지편향" → "cognitive bias" 추가
        # "학습이론" → "learning theory" 추가
        pass
```

#### 3. Psychology Document Processor (신규)
```python
# src/services/psychology/document_processor.py
class PsychologyDocumentProcessor:
    """심리학 논문 전용 처리기"""

    def __init__(self):
        # 기존 multimodal processor 활용
        from src.services.rag.multimodal_processor import MultimodalProcessor
        self.base_processor = MultimodalProcessor()

    async def process_psychology_pdf(self, pdf_path: str):
        # 1. 기존 PDF 처리기 활용
        base_result = await self.base_processor.process_pdf(pdf_path)

        # 2. 심리학 특화 메타데이터만 추가
        psychology_metadata = self._extract_psychology_metadata(base_result)

        return {**base_result, **psychology_metadata}
```

## 🔄 통합 아키텍처

### 기존 시스템과의 관계
```
🏗️ AI-CoScientist (기존)
├── UnifiedRAGOrchestrator ✅ 그대로 사용
│   ├── 6개 기존 전략 ✅ 그대로 유지
│   └── psychology_specialized 🆕 추가
├── ChromaDB ✅ 그대로 사용
│   ├── 3개 기존 컬렉션 ✅ 유지
│   └── 3개 심리학 컬렉션 🆕 추가
├── RAGAS 평가 ✅ 그대로 사용
└── Agent Pool ✅ 확장
    ├── 6개 기존 에이전트 ✅ 유지
    └── Psychology Agent 🆕 추가

🧠 Psychology Extension (신규)
├── Korean NLP Pipeline 🆕
├── Psychology RAG Strategy 🆕
├── Document Processor 🆕
└── Chat Interface 🆕
```

## 🚀 구현 전략: 점진적 확장

### Phase 1: 기존 시스템 활용 (1주)
```bash
# 기존 컬렉션에 심리학 데이터 추가
python scripts/add_psychology_collection.py

# 기존 오케스트레이터에 새 전략 등록
# src/services/rag/unified_rag_orchestrator.py 수정
```

### Phase 2: 심리학 특화 기능 개발 (2주)
```bash
# 새로운 모듈들 개발
mkdir src/services/psychology/
touch src/services/psychology/korean_nlp.py
touch src/services/psychology/document_processor.py
touch src/services/rag/psychology_rag_strategy.py
```

### Phase 3: 통합 테스트 (1주)
```bash
# 기존 테스트 프레임워크 활용
python -m pytest tests/rag/test_psychology_integration.py
```

## 💡 핵심 장점

### ✅ 기존 시스템 활용의 이점
1. **검증된 인프라**: 이미 운영 중인 안정적 시스템
2. **개발 시간 단축**: 90% 기존 코드 재사용
3. **성능 보장**: 이미 최적화된 벡터 검색
4. **모니터링**: 기존 메트릭스 시스템 그대로 사용

### 🆕 심리학 특화의 가치
1. **도메인 전문성**: 심리학 용어 및 개념 이해
2. **한국어 지원**: 서울대 맞춤 언어 처리
3. **교수진 특화**: 66편 논문 기반 전문 검색
4. **연구 지원**: 방법론, 윤리, 협업 매칭

## 🔧 기술적 구현 세부사항

### 기존 API 확장
```python
# src/api/v1/rag.py (기존 파일 확장)
@router.post("/search")
async def rag_search(request: SearchRequest):
    # 기존 코드 그대로
    orchestrator = UnifiedRAGOrchestrator()

    # 심리학 쿼리 감지 시 새 전략 사용
    if detect_psychology_query(request.query):
        strategy = "psychology_specialized"
    else:
        strategy = orchestrator.auto_select_strategy(request.query)

    results = await orchestrator.search(
        query=request.query,
        strategy=strategy
    )

    return results
```

### 설정 파일 확장
```python
# src/core/config.py (기존 파일에 추가)
class Settings(BaseSettings):
    # 기존 설정들...

    # 심리학 확장 설정만 추가
    psychology_enabled: bool = True
    korean_model_path: str = "models/korean/"
    psychology_collections: List[str] = [
        "psychology_papers",
        "psychology_metadata"
    ]
```

이 계획의 핵심은 **"바퀴를 재발명하지 않는다"**입니다. 이미 훌륭하게 구축된 AI-CoScientist의 RAG 시스템을 최대한 활용하면서, 심리학과만의 특별한 요구사항을 위한 최소한의 확장만 진행하는 것입니다.

**결과적으로:**
- 🕐 **개발 시간**: 9주 → 4주로 단축
- 💰 **개발 비용**: 70% 절감
- 🛡️ **안정성**: 검증된 시스템 기반
- 🚀 **성능**: 이미 최적화된 인프라 활용
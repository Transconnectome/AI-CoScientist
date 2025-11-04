# AI-CoScientist 온보딩 가이드

**신규 개발자를 위한 3일 학습 경로**

이 가이드는 AI-CoScientist 코드베이스를 처음 접하는 개발자가 **3일 안에** 시스템을 이해하고 기여할 수 있도록 설계되었습니다.

---

## 📋 시작하기 전에

### 필수 사전 지식
- **Python 3.11+**: 비동기 프로그래밍 (async/await)
- **FastAPI**: REST API 개발 경험
- **PostgreSQL**: 관계형 데이터베이스 기본
- **Docker**: 컨테이너 기반 개발 환경
- **LLM 기본 개념**: GPT, Claude, Prompt Engineering

### 권장 사전 학습
- Transformer 모델 구조 (BERT, SciBERT)
- Vector Database 개념 (ChromaDB, Embeddings)
- RAG (Retrieval-Augmented Generation) 기본 원리

---

## Day 1: 큰 그림 이해하기 (2-3시간)

### 1.1 시스템 개요 파악 (30분)

**필독 문서 순서:**
1. `README_KO.md` - 전체 시스템 비전과 아키텍처
2. `QUICK_START.md` - 빠른 설치 및 첫 실행
3. `docs/ARCHITECTURE.md` - 상세 아키텍처 설명

**핵심 개념 체크리스트:**
- [ ] 3대 엔진(연구/실험/논문) 역할 이해
- [ ] 4차원 평가 시스템 (Novelty, Methodology, Clarity, Significance)
- [ ] RAG 시스템이 해결하는 문제 (환각 방지, 근거 기반 제안)
- [ ] Nemotron 하이브리드의 비용 효율성 (59% 절감)

### 1.2 아키텍처 다이어그램 분석 (30분)

```
┌─────────────────────────────────────────────────────────────┐
│                      클라이언트 계층                           │
│  Claude Code, REST API, Python SDK, 웹 대시보드               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    API 게이트웨이 (FastAPI)                   │
│  라우팅, 인증, 검증, 속도 제한, CORS                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐  ┌───────▼────────┐  ┌──────▼──────┐
│  연구 엔진     │  │   실험 엔진     │  │  논문 엔진   │
│               │  │                │  │             │
│ • 가설 생성   │  │ • 프로토콜     │  │ • 파싱      │
│ • 문헌 분석   │  │ • 통계 분석   │  │ • 4D 평가   │
│ • 신규성 평가 │  │ • 시각화      │  │ • RAG 개선  │
└───────┬───────┘  └───────┬────────┘  └──────┬──────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      데이터 계층                              │
│                                                              │
│  PostgreSQL          ChromaDB          Redis                │
│  (구조화 데이터)      (벡터 검색)       (캐시/큐)             │
│                                                              │
│  • 프로젝트/논문     • 355개 문헌      • LLM 응답 캐시       │
│  • 버전 관리        • 10개 패턴        • Celery 작업 큐      │
│  • 평가 히스토리    • 하이브리드 검색  • 세션 데이터         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      외부 서비스                              │
│                                                              │
│  GPT-5              Claude Sonnet    Gemini 2.5            │
│  (주 평가 모델)      (폴백 모델)      (실험적)               │
│                                                              │
│  Semantic Scholar   CrossRef         arXiv                 │
│  (문헌 검색)        (메타데이터)      (논문 다운로드)         │
└─────────────────────────────────────────────────────────────┘
```

**실습 과제:**
```bash
# 1. 프로젝트 클론 및 환경 구성
git clone https://github.com/your-org/AI-CoScientist.git
cd AI-CoScientist

# 2. .env 파일 생성
cp .env.example .env
# 필수: OPENAI_API_KEY, ANTHROPIC_API_KEY 설정

# 3. Docker로 의존성 실행
docker-compose up -d postgres redis chromadb

# 4. Python 환경 설정 (Poetry 권장)
poetry install
poetry shell

# 5. 데이터베이스 마이그레이션
alembic upgrade head

# 6. 개발 서버 시작
uvicorn src.api.main:app --reload --port 8000
```

**확인 체크:**
- [ ] http://localhost:8000/docs 에서 Swagger UI 확인
- [ ] `/health` 엔드포인트가 정상 응답
- [ ] PostgreSQL, Redis, ChromaDB 모두 연결 성공

### 1.3 첫 API 호출 테스트 (1시간)

**시나리오 1: 논문 업로드 및 파싱**

```bash
# 테스트용 논문 파일 준비
cp paper.pdf input/test_paper.pdf

# API 호출 (cURL)
curl -X POST "http://localhost:8000/api/v1/papers/upload" \
  -F "file=@input/test_paper.pdf" \
  -F "title=Test Paper" \
  -F "project_id=your-project-uuid"
```

**시나리오 2: 논문 평가 요청**

```python
# simple_demo.py 실행
python simple_demo.py

# 예상 출력:
# ✅ Paper uploaded: <paper_id>
# ✅ Parsed into 6 sections
# ✅ Quality score: 7.2/10
#    - Novelty: 7.0
#    - Methodology: 7.5
#    - Clarity: 6.8
#    - Significance: 7.5
```

**시나리오 3: RAG 기반 개선 제안**

```python
# demo_paper_analysis.py 실행
python demo_paper_analysis.py

# RAG 시스템이 제공하는 개선 제안 확인:
# - 근거: 355개 논문 중 유사 사례 검색
# - 패턴: 10개 개선 패턴 매칭
# - 구체성: "Figure 2의 통계 검정력 명시 권장"
```

---

## Day 2: 코드 흐름 따라가기 (4-5시간)

### 2.1 논문 평가 요청의 전체 흐름 (2시간)

**Step-by-Step 코드 추적:**

#### 1️⃣ API 진입점 (`src/api/v1/papers.py:96`)

```python
@router.post("/{paper_id}/analyze", response_model=PaperAnalysisResponse)
async def analyze_paper(
    paper_id: UUID,
    request: PaperAnalyzeRequest = PaperAnalyzeRequest(),
    db: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
):
```

**이곳에서 일어나는 일:**
- 요청 검증: `paper_id` 유효성 확인
- 의존성 주입: `db` 세션, `llm_service` 인스턴스
- 라우팅: `PaperAnalyzer` 서비스 호출

#### 2️⃣ 분석 서비스 (`src/services/paper/analyzer.py:71`)

```python
async def analyze_quality(
    self,
    paper_id: UUID,
    use_scibert: bool = True,
    use_ensemble: bool = True,
    use_hybrid: bool = False
) -> dict:
```

**3가지 평가 모드:**

| 모드 | 설명 | 정확도 | 속도 |
|------|------|--------|------|
| **GPT-5 Only** | 기본 평가 | 85% | 빠름 |
| **SciBERT Ensemble** | GPT + SciBERT 앙상블 | 89% | 보통 |
| **Hybrid Model** | 학습된 하이브리드 | 92% | 느림 |

**코드 경로:**
```
analyzer.py:88 → _get_scibert_scorer()
              → scibert_scorer.py:45 (SciBERTQualityScorer)
              → metrics.py:23 (PaperMetrics.calculate_coherence)
```

#### 3️⃣ LLM 호출 (`src/services/llm/service.py:67`)

```python
async def complete(
    self,
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.3,
    use_cache: bool = True
) -> LLMResponse:
```

**폴백 로직:**
1. **Primary (GPT-5)** 시도
2. 실패 시 → **Fallback (Claude)** 시도
3. 모두 실패 → `LLMServiceError` 발생

**캐싱 최적화:**
- 동일 프롬프트 → Redis 캐시 조회 (TTL: 1시간)
- 캐시 미스 → API 호출 → 캐시 저장

#### 4️⃣ 데이터베이스 저장 (`src/api/v1/papers.py:135`)

```python
# 평가 결과 저장
evaluation = PaperEvaluation(
    paper_id=paper_id,
    overall_score=analysis["quality_score"],
    dimensions=analysis["dimensions"],
    feedback=analysis["suggestions"]
)
db.add(evaluation)
await db.commit()
```

**스키마 구조:**
```sql
CREATE TABLE paper_evaluations (
    id UUID PRIMARY KEY,
    paper_id UUID REFERENCES papers(id),
    overall_score FLOAT,
    dimensions JSONB,  -- {novelty: 7.0, methodology: 7.5, ...}
    feedback JSONB,    -- [{section: "intro", suggestion: "..."}]
    created_at TIMESTAMP
);
```

### 2.2 RAG 시스템 동작 원리 (1.5시간)

**RAG 파이프라인 상세:**

```
사용자 입력 (논문 초록)
    │
    ▼
┌───────────────────────────────────┐
│ 1. 쿼리 임베딩 생성               │
│    (SciBERT embeddings)           │
│    384차원 벡터                    │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│ 2. 하이브리드 검색                │
│                                   │
│   Semantic (70%):                 │
│   - Cosine similarity             │
│   - ChromaDB 벡터 검색            │
│                                   │
│   Keyword (30%):                  │
│   - BM25 알고리즘                 │
│   - 전문 용어 매칭                │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│ 3. 문서 재랭킹 (Reranking)        │
│                                   │
│   • 최신성 가중치 (0.2)           │
│   • 인용 횟수 (0.3)               │
│   • 유사도 점수 (0.5)             │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│ 4. 개선 패턴 매칭                 │
│                                   │
│   10개 패턴 중 Top-3 선택:        │
│   1. Abstract clarity enhancement │
│   2. Methodology strengthening    │
│   3. Results presentation         │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│ 5. GPT-5 개선 생성                │
│                                   │
│   Context:                        │
│   - 검색된 문헌 3편               │
│   - 적용 가능 패턴 3개            │
│   - 현재 논문 컨텍스트            │
│                                   │
│   Output:                         │
│   - 개선된 텍스트                 │
│   - 변경 사항 요약                │
│   - 근거 (어떤 논문/패턴 참조)     │
└───────────────────────────────────┘
```

**코드 위치:**

```python
# src/services/paper/improver.py:105
async def improve_with_rag(
    self,
    paper_id: UUID,
    section_name: str
) -> dict:
    # 1. 임베딩 생성
    query_embedding = await self.embedding_service.embed(
        section.content
    )

    # 2. 하이브리드 검색
    similar_docs = await self.vector_store.hybrid_search(
        query_embedding=query_embedding,
        query_text=section.content,
        semantic_weight=0.7,
        keyword_weight=0.3,
        top_k=10
    )

    # 3. 재랭킹
    reranked = self._rerank_documents(
        similar_docs,
        recency_weight=0.2,
        citation_weight=0.3,
        similarity_weight=0.5
    )

    # 4. 패턴 매칭
    patterns = await self._match_improvement_patterns(
        section.content,
        top_k=3
    )

    # 5. LLM 개선 생성
    improved = await self.llm.complete(
        prompt=self._build_rag_prompt(
            section=section.content,
            references=reranked[:3],
            patterns=patterns
        )
    )
```

**실습 과제:**

```bash
# RAG 데이터 확인
python -c "
from src.services.knowledge_base.vector_store import VectorStore
vs = VectorStore()

# 컬렉션 통계
stats = vs.get_collection_stats('research_documents')
print(f'문서 수: {stats['count']}')
print(f'임베딩 차원: {stats['dimension']}')

# 샘플 검색
results = vs.search(
    query='deep learning brain imaging',
    top_k=5
)
for r in results:
    print(f'- {r['metadata']['title']} (유사도: {r['score']:.3f})')
"
```

### 2.3 데이터베이스 스키마 이해 (30분)

**핵심 테이블 관계도:**

```
┌──────────────┐
│   projects   │  1:N
│──────────────│─────┐
│ id (PK)      │     │
│ name         │     │
│ description  │     │
└──────────────┘     │
                     │
                     ▼
┌──────────────┐  ┌──────────────────┐
│    papers    │  │ paper_versions   │
│──────────────│──│──────────────────│
│ id (PK)      │  │ id (PK)          │
│ project_id   │  │ paper_id (FK)    │
│ title        │  │ version          │
│ content      │  │ content          │
│ status       │  │ created_at       │
└──┬───────────┘  └──────────────────┘
   │
   │ 1:N
   ▼
┌──────────────────┐
│ paper_sections   │
│──────────────────│
│ id (PK)          │
│ paper_id (FK)    │
│ name             │
│ content          │
│ order            │
└──────────────────┘
   │
   │ 1:N
   ▼
┌─────────────────────┐
│ paper_evaluations   │
│─────────────────────│
│ id (PK)             │
│ paper_id (FK)       │
│ overall_score       │
│ dimensions (JSONB)  │
│ feedback (JSONB)    │
│ created_at          │
└─────────────────────┘
```

**중요 쿼리 패턴:**

```python
# 1. 최신 버전 조회 (src/models/project.py:123)
latest_version = (
    await db.execute(
        select(PaperVersion)
        .where(PaperVersion.paper_id == paper_id)
        .order_by(PaperVersion.version.desc())
        .limit(1)
    )
).scalar_one()

# 2. 평가 히스토리 조회
evaluations = (
    await db.execute(
        select(PaperEvaluation)
        .where(PaperEvaluation.paper_id == paper_id)
        .order_by(PaperEvaluation.created_at.desc())
    )
).scalars().all()

# 3. 섹션별 개선 추적
improvements = (
    await db.execute(
        select(SectionImprovement)
        .join(PaperSection)
        .where(PaperSection.paper_id == paper_id)
        .options(selectinload(SectionImprovement.section))
    )
).scalars().all()
```

### 2.4 설정 파일 및 환경 변수 (30분)

**`.env` 핵심 설정 설명:**

```bash
# LLM 제공자 우선순위
LLM_PRIMARY_PROVIDER=openai      # GPT-5 (정확도 우선)
LLM_FALLBACK_PROVIDER=anthropic  # Claude (안정성)

# 캐싱 전략
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL=3600  # 1시간 (동일 프롬프트 재사용)

# RAG 설정
CHROMADB_COLLECTION=research_documents
EMBEDDING_MODEL=allenai/scibert_scivocab_uncased

# 성능 튜닝
DATABASE_POOL_SIZE=5         # 동시 DB 연결
REDIS_MAX_CONNECTIONS=10     # Redis 연결 풀
API_WORKERS=4                # Uvicorn 워커 수
```

**환경별 설정 파일:**

| 파일 | 용도 | 주요 차이점 |
|------|------|------------|
| `.env.example` | 로컬 개발 | DEBUG=true, 작은 풀 크기 |
| `.env.production.template` | 프로덕션 | 큰 풀, 모니터링 활성화 |
| `.env.connectome.hybrid.template` | Connectome 서버 | GPU 활성화, Nemotron 사용 |

---

## Day 3: 실제 작업 시작하기 (3-4시간)

### 3.1 일반적인 개발 작업 시나리오

#### 시나리오 A: 새로운 평가 기준 추가하기

**요구사항:** "Impact" 차원 추가 (현재 4차원 → 5차원)

**Step 1: 스키마 업데이트**

```python
# src/schemas/paper.py:45
class PaperDimensions(BaseModel):
    """Paper quality dimensions."""
    novelty: float = Field(..., ge=0, le=10)
    methodology: float = Field(..., ge=0, le=10)
    clarity: float = Field(..., ge=0, le=10)
    significance: float = Field(..., ge=0, le=10)
    impact: float = Field(..., ge=0, le=10)  # ✅ 추가
```

**Step 2: 프롬프트 수정**

```python
# prompts/evaluation_prompt.txt:23
평가 차원:
1. Novelty (0-10): 연구의 독창성과 새로운 기여
2. Methodology (0-10): 방법론의 견고성과 적절성
3. Clarity (0-10): 명확성과 구조
4. Significance (0-10): 학문적 중요성
5. Impact (0-10): 실용적 영향력과 파급 효과  # ✅ 추가
```

**Step 3: 분석 로직 업데이트**

```python
# src/services/paper/analyzer.py:156
dimensions = {
    "novelty": scores.get("novelty", 0.0),
    "methodology": scores.get("methodology", 0.0),
    "clarity": scores.get("clarity", 0.0),
    "significance": scores.get("significance", 0.0),
    "impact": scores.get("impact", 0.0)  # ✅ 추가
}

# 평균 계산 로직 수정
overall_score = sum(dimensions.values()) / 5  # 4 → 5
```

**Step 4: 테스트 작성**

```python
# tests/test_services/test_paper_analyzer.py:89
async def test_five_dimension_evaluation():
    """Test new 5-dimension evaluation."""
    result = await analyzer.analyze_quality(
        paper_id=test_paper_id,
        use_ensemble=True
    )

    assert "impact" in result["dimensions"]
    assert 0 <= result["dimensions"]["impact"] <= 10
    assert len(result["dimensions"]) == 5
```

**Step 5: 마이그레이션 생성**

```bash
# 데이터베이스 스키마 변경 없음 (JSONB 필드 사용)
# 기존 데이터 마이그레이션 스크립트 작성

# scripts/migrate_add_impact.py
async def migrate_existing_evaluations():
    """Add impact dimension to existing evaluations."""
    evaluations = await db.execute(
        select(PaperEvaluation)
    )

    for eval in evaluations.scalars():
        if "impact" not in eval.dimensions:
            # 기존 4차원 평균으로 초기화
            avg = sum(eval.dimensions.values()) / 4
            eval.dimensions["impact"] = round(avg, 2)

    await db.commit()
```

#### 시나리오 B: RAG 문서 업데이트하기

**요구사항:** 새로운 논문 50편을 RAG 시스템에 추가

**Step 1: 논문 다운로드**

```bash
# scripts/download_papers.py 사용
python scripts/download_papers.py \
  --query "multimodal learning neuroscience" \
  --max-results 50 \
  --output-dir papers_collection/multimodal/
```

**Step 2: 품질 필터링**

```python
# scripts/filter_papers.py
from src.services.paper.analyzer import PaperAnalyzer

async def filter_high_quality_papers():
    """Only ingest papers with quality score >= 7.0."""
    for paper_file in Path("papers_collection/multimodal/").glob("*.pdf"):
        # 빠른 평가
        score = await analyzer.quick_evaluate(paper_file)

        if score >= 7.0:
            approved_papers.append(paper_file)
        else:
            print(f"Filtered out: {paper_file.name} (score: {score})")
```

**Step 3: 임베딩 생성 및 수집**

```bash
# scripts/ingest_papers.py 실행
python scripts/ingest_papers.py \
  --input-dir papers_collection/multimodal/ \
  --collection research_documents \
  --batch-size 10 \
  --embedding-model allenai/scibert_scivocab_uncased
```

**내부 동작:**
```python
# src/services/knowledge_base/ingestion.py:67
async def ingest_batch(self, papers: List[Path]):
    for paper in papers:
        # 1. PDF 파싱
        text = await self.parser.extract_text(paper)

        # 2. 청크 분할 (512 토큰 단위)
        chunks = self.text_splitter.split(
            text,
            chunk_size=512,
            overlap=50
        )

        # 3. 임베딩 생성
        embeddings = await self.embedding_service.embed_batch(
            chunks
        )

        # 4. ChromaDB 저장
        await self.vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadatas=[{
                "source": paper.name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            } for i in range(len(chunks))]
        )
```

**Step 4: 검증**

```python
# 새로운 문서가 검색되는지 확인
results = await vector_store.search(
    query="multimodal learning brain",
    top_k=10
)

# 새로 추가된 문서 확인
new_docs = [r for r in results if "multimodal" in r["metadata"]["source"]]
print(f"Added {len(new_docs)} new multimodal papers to index")
```

#### 시나리오 C: 테스트 작성 및 실행

**테스트 계층 구조:**

```
tests/
├── test_unit/              # 단위 테스트 (개별 함수)
│   ├── test_metrics.py
│   └── test_embeddings.py
│
├── test_integration/       # 통합 테스트 (여러 컴포넌트)
│   ├── test_api_endpoints.py
│   └── test_rag_pipeline.py
│
└── test_e2e/              # E2E 테스트 (전체 워크플로우)
    └── test_complete_pipeline.py
```

**좋은 테스트 예시:**

```python
# tests/test_integration/test_rag_pipeline.py
import pytest
from src.services.paper.improver import PaperImprover

@pytest.mark.asyncio
async def test_rag_improvement_with_references(
    db_session,
    test_paper,
    mock_llm_service
):
    """Test RAG improvement returns referenced sources."""
    improver = PaperImprover(
        llm_service=mock_llm_service,
        db=db_session
    )

    result = await improver.improve_with_rag(
        paper_id=test_paper.id,
        section_name="Abstract"
    )

    # 개선 텍스트 생성 확인
    assert "improved_content" in result
    assert len(result["improved_content"]) > 0

    # 참조 문헌 포함 확인
    assert "references" in result
    assert len(result["references"]) >= 1
    assert all("title" in ref for ref in result["references"])

    # 변경 사항 요약 확인
    assert "changes_summary" in result
    assert "based on" in result["changes_summary"].lower()
```

**테스트 실행:**

```bash
# 전체 테스트
pytest

# 특정 파일
pytest tests/test_integration/test_rag_pipeline.py

# 커버리지 확인
pytest --cov=src --cov-report=html

# 느린 테스트 건너뛰기 (개발 중)
pytest -m "not slow"

# 병렬 실행 (빠른 피드백)
pytest -n auto
```

### 3.2 자주 발생하는 문제와 해결 방법

#### 문제 1: ChromaDB 연결 실패

**증상:**
```
chromadb.errors.ConnectionError: Could not connect to ChromaDB at localhost:8001
```

**해결:**
```bash
# 1. ChromaDB 컨테이너 상태 확인
docker ps | grep chromadb

# 2. 컨테이너 재시작
docker-compose restart chromadb

# 3. 로그 확인
docker-compose logs chromadb

# 4. 포트 충돌 확인
lsof -i :8001

# 5. 데이터 손상 시 재구축
docker-compose down -v
docker-compose up -d chromadb
python scripts/rebuild_vector_db.py
```

#### 문제 2: LLM API 속도 제한

**증상:**
```
openai.error.RateLimitError: Rate limit exceeded
```

**해결:**

```python
# src/services/llm/service.py:234
# 지수 백오프 재시도 로직 추가됨
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
async def _call_with_retry(self, provider, prompt):
    return await provider.complete(prompt)
```

**임시 대응:**
```bash
# .env 설정
LLM_FALLBACK_PROVIDER=anthropic  # Claude로 폴백
LLM_MAX_RETRIES=5
LLM_TIMEOUT=120  # 타임아웃 증가
```

#### 문제 3: 메모리 부족 (대용량 논문 처리)

**증상:**
```
MemoryError: Unable to allocate array
```

**해결:**

```python
# 청크 단위 처리로 변경
# src/services/paper/parser.py:89
async def parse_large_paper(self, content: str):
    """Parse large papers in chunks to avoid memory issues."""
    chunk_size = 100000  # 100KB 청크

    for i in range(0, len(content), chunk_size):
        chunk = content[i:i+chunk_size]
        sections = await self._parse_chunk(chunk)
        yield sections  # Generator로 메모리 효율화
```

#### 문제 4: 비동기 작업 데드락

**증상:**
```
asyncio.exceptions.TimeoutError: Task took longer than 60 seconds
```

**디버깅:**

```python
# 작업 추적 로깅 추가
import logging
logger = logging.getLogger(__name__)

async def analyze_paper(paper_id):
    logger.info(f"[START] Analyzing paper {paper_id}")

    try:
        result = await analyzer.analyze_quality(paper_id)
        logger.info(f"[SUCCESS] Analysis completed for {paper_id}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"[TIMEOUT] Analysis timed out for {paper_id}")
        # Celery 백그라운드 작업으로 전환
        from src.tasks.paper_tasks import analyze_paper_task
        analyze_paper_task.delay(str(paper_id))
```

### 3.3 코드 기여 워크플로우

**Git 브랜치 전략:**

```
main (프로덕션)
  │
  ├── develop (개발)
  │     │
  │     ├── feature/add-impact-dimension  ← 새 기능
  │     ├── fix/rag-search-bug           ← 버그 수정
  │     └── refactor/analyzer-cleanup    ← 리팩토링
  │
  └── hotfix/critical-api-fix  ← 긴급 수정
```

**작업 절차:**

```bash
# 1. 최신 코드 동기화
git checkout develop
git pull origin develop

# 2. 기능 브랜치 생성
git checkout -b feature/add-impact-dimension

# 3. 작업 수행 (TDD 권장)
# - 테스트 작성 → 구현 → 리팩토링

# 4. 커밋 (Conventional Commits)
git add .
git commit -m "feat(analyzer): add Impact dimension to evaluation

- Add impact field to PaperDimensions schema
- Update evaluation prompts
- Modify scoring logic to include 5th dimension
- Add migration script for existing evaluations

Closes #123"

# 5. 푸시 및 PR
git push origin feature/add-impact-dimension
gh pr create --title "Add Impact dimension to paper evaluation" \
  --body "Implements #123" \
  --base develop
```

**PR 체크리스트:**

- [ ] 모든 테스트 통과 (`pytest`)
- [ ] 코드 스타일 준수 (`ruff check`, `black --check`)
- [ ] 타입 체크 통과 (`mypy src/`)
- [ ] 문서 업데이트 (API 변경 시)
- [ ] CHANGELOG.md 업데이트
- [ ] 리뷰어 2명 이상 승인

### 3.4 디버깅 팁 및 유용한 도구

#### 로컬 디버깅 (VS Code)

**`.vscode/launch.json` 설정:**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI Server",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "src.api.main:app",
        "--reload",
        "--port", "8000"
      ],
      "jinja": true,
      "justMyCode": false
    },
    {
      "name": "Pytest: Current File",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": [
        "${file}",
        "-v",
        "-s"
      ],
      "console": "integratedTerminal"
    }
  ]
}
```

#### 프로파일링

```python
# 느린 함수 찾기
from cProfile import Profile
from pstats import Stats

profiler = Profile()
profiler.enable()

# 분석할 코드
result = await analyzer.analyze_quality(paper_id)

profiler.disable()
stats = Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 느린 함수
```

#### 로그 분석

```bash
# 에러만 필터링
docker-compose logs api | grep ERROR

# 특정 paper_id 추적
docker-compose logs api | grep "paper_id=123e4567"

# 성능 로그 분석
cat logs/performance.log | jq '.duration' | \
  awk '{sum+=$1; count++} END {print "Avg:", sum/count, "ms"}'
```

---

## 📚 유용한 리소스

### 프로젝트 문서
- **README_KO.md**: 시스템 개요 및 비전
- **docs/ARCHITECTURE.md**: 상세 아키텍처
- **docs/API_REFERENCE.md**: 전체 API 문서
- **docs/DEVELOPMENT.md**: 개발 가이드
- **CHANGELOG.md**: 변경 이력

### 외부 자료
- **FastAPI 공식 문서**: https://fastapi.tiangolo.com/
- **ChromaDB 가이드**: https://docs.trychroma.com/
- **Prompt Engineering**: https://platform.openai.com/docs/guides/prompt-engineering
- **SciBERT 논문**: https://arxiv.org/abs/1903.10676
- **RAG 논문**: https://arxiv.org/abs/2005.11401

### 커뮤니티
- **Discord**: #ai-coscientist-dev
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **주간 Sync**: 매주 화요일 10:00 (Zoom)

---

## ✅ 온보딩 완료 체크리스트

**Day 1:**
- [ ] 로컬 환경 구성 완료
- [ ] 첫 API 호출 성공
- [ ] 아키텍처 다이어그램 이해

**Day 2:**
- [ ] 논문 평가 흐름 전체 추적
- [ ] RAG 시스템 동작 원리 이해
- [ ] 데이터베이스 스키마 파악

**Day 3:**
- [ ] 새로운 평가 기준 추가 (실습)
- [ ] 테스트 작성 및 실행
- [ ] 첫 PR 제출

**추가 학습:**
- [ ] Celery 비동기 작업 이해
- [ ] Grafana 대시보드 활용
- [ ] 프로덕션 배포 프로세스

---

## 🆘 도움이 필요하면?

**빠른 답변:**
1. `docs/` 디렉토리의 관련 문서 확인
2. GitHub Issues에서 유사 문제 검색
3. Discord #questions 채널 질문

**멘토링:**
- **백엔드/API**: @backend-team
- **ML/RAG**: @ml-team
- **인프라/배포**: @devops-team

**긴급 상황:**
- 프로덕션 장애: @on-call-engineer
- 보안 이슈: security@ai-coscientist.com

---

**환영합니다! 🎉**

이 가이드를 완료하면 AI-CoScientist의 핵심 컴포넌트를 이해하고, 실제 기여를 시작할 수 있습니다.

**다음 단계:**
- 첫 이슈 선택하기 (Label: `good-first-issue`)
- 코드 리뷰 참여하기
- 주간 Sync 미팅 참석하기

**Happy Coding! 🚀**

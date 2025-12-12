# Golden Reference RAG Ingestion Plan

## 현재 상황 분석

### ✅ 준비된 것들
1. **53개의 고품질 PDF** (592MB)
   - Nature, Nature Medicine, Nature Biomedical Engineering 등
   - Foundation model 관련 최신 논문들
   - 98.1% 텍스트 추출 가능

2. **기존 인프라**
   - ChromaDB 연동 (`vector_store.py`)
   - SciBERT 임베딩 모델 (`allenai/scibert_scivocab_uncased`)
   - LLM 기반 섹션 파서 (`parser.py`)
   - Golden Reference Store 기본 구조

### ⚠️ 문제점: 기존 ingestion.py의 한계

**현재 구현 (ingestion.py:164-195):**
```python
async def _embed_paper(self, paper_id: UUID, paper_data: dict) -> None:
    # ❌ 문제: title + abstract만 임베딩
    text = f"{paper_data.get('title', '')}. {paper_data.get('abstract', '')}"

    # ❌ 문제: 단일 임베딩만 생성 (chunking 없음)
    embedding = await self.embedding_service.encode_async(text)

    # ❌ 문제: API 기반 (DOI/query), 로컬 PDF 지원 안함
```

**결과:**
- Full-text 검색 불가능 (Introduction, Methods, Results 등 누락)
- 긴 논문의 세부 내용 검색 실패
- 로컬 PDF 파일 인제스트 불가능

---

## 필수 고려사항

### 1. PDF 파싱 전략

**PyPDF2 vs LLM 기반 파싱:**

| 방법 | 장점 | 단점 |
|------|------|------|
| **PyPDF2** | ✅ 빠름<br>✅ 비용 없음<br>✅ 텍스트 추출 | ❌ 섹션 구분 어려움<br>❌ 레이아웃 문제 |
| **LLM 파싱** | ✅ 섹션 자동 인식<br>✅ 구조 파악 | ❌ 느림 (53개 × ~30초)<br>❌ API 비용 |
| **Hybrid** | ✅ PyPDF2 추출 + LLM 섹션화 | ✅ 속도/품질 균형 |

**추천: Hybrid 접근**
1. PyPDF2로 전체 텍스트 추출
2. LLM으로 섹션 구분 (Abstract, Intro, Methods 등)
3. 섹션별로 메타데이터 태깅

### 2. Chunking 전략

**Naive Chunking (현재 ingestion.py):**
```python
❌ text = title + abstract  # 전체 논문의 5%만 사용
❌ 단일 임베딩  # 검색 품질 저하
```

**Section-Aware Chunking (권장):**
```python
✅ chunks = [
    {"text": "Abstract: ...", "section": "abstract", "chunk_id": "paper1_abs_0"},
    {"text": "Intro paragraph 1...", "section": "introduction", "chunk_id": "paper1_intro_0"},
    {"text": "Intro paragraph 2...", "section": "introduction", "chunk_id": "paper1_intro_1"},
    {"text": "Methods subsection...", "section": "methods", "chunk_id": "paper1_meth_0"},
]
```

**핵심 파라미터:**
- **Chunk size**: 512-1024 tokens (SciBERT max 512, 여유 있게 512)
- **Overlap**: 50-100 tokens (문맥 유지)
- **Section boundary preservation**: 섹션 경계에서 자르지 않음

### 3. RAPTOR 계층 구조 (Advanced)

**`advanced_golden_reference.py`에 이미 구현됨:**

```
Level 2 (Paper Summary)
    └─ "This paper presents foundation model for..."

Level 1 (Section Summaries)
    ├─ "Abstract+Intro: Novel vision-language model..."
    ├─ "Methods: Architecture uses transformer..."
    └─ "Results: Achieved 95% accuracy..."

Level 0 (Original Chunks)
    ├─ Chunk 1: "Abstract: We present..."
    ├─ Chunk 2: "Introduction paragraph 1..."
    ├─ Chunk 3: "Introduction paragraph 2..."
    └─ ...
```

**장점:**
- 추상적 질문 → Level 2 검색
- 구체적 질문 → Level 0 검색
- 검색 품질 20-30% 향상 (RAPTOR 논문 기준)

**단점:**
- 구현 복잡도 증가
- LLM 요약 비용 (53개 × 섹션당 요약)

### 4. 메타데이터 설계

**필수 메타데이터:**
```python
metadata = {
    # 논문 정보
    "paper_id": "nature_2024_foundation_001",
    "title": "A pathology foundation model...",
    "journal": "Nature",
    "year": 2024,
    "doi": "10.1038/...",

    # 청크 정보
    "chunk_id": "nature_2024_foundation_001_intro_2",
    "section": "introduction",  # abstract, intro, methods, results, discussion
    "chunk_index": 2,
    "total_chunks": 45,

    # 검색 필터용
    "has_code": true,  # Code availability 여부
    "has_data": true,  # Data availability 여부
    "keywords": ["foundation model", "vision-language", "pathology"],

    # 품질 지표
    "citation_count": 150,
    "impact_factor": 42.8,
}
```

**활용:**
```python
# 최근 논문만 검색
results = search(query, where={"year": {"$gte": 2023}})

# Methods 섹션만 검색
results = search(query, where={"section": "methods"})

# Nature 논문만 검색
results = search(query, where={"journal": "Nature"})
```

---

## 권장 인제스션 전략 (3단계)

### Option A: 빠른 시작 (Simple)
**목적:** 빠르게 프로토타입 테스트

1. **PyPDF2로 전체 텍스트 추출**
2. **고정 크기 chunking** (512 tokens, 50 overlap)
3. **기본 메타데이터** (title, journal, year만)
4. **ChromaDB 저장**

**예상 시간:** 10-15분
**장점:** 빠른 검증
**단점:** 검색 품질 중간

### Option B: 균형잡힌 접근 (Recommended) ⭐
**목적:** 품질과 속도의 균형

1. **PyPDF2로 텍스트 추출**
2. **LLM으로 섹션 구분** (Abstract, Intro, Methods, Results, Discussion)
3. **Section-aware chunking** (섹션 경계 존중)
4. **풍부한 메타데이터** (섹션, 키워드, 품질 지표)
5. **ChromaDB 저장**

**예상 시간:** 30-45분 (53개 × ~40초)
**API 비용:** ~$2-3 (섹션 구분)
**장점:** 고품질 검색, 섹션 필터링
**단점:** LLM 비용, 다소 느림

### Option C: 최고 품질 (Advanced)
**목적:** 연구용 production-grade RAG

1. **PyPDF2로 텍스트 추출**
2. **LLM으로 섹션 구분 + 요약**
3. **RAPTOR 3-level 계층 구조**
   - Level 0: 원본 chunks
   - Level 1: 섹션 요약
   - Level 2: 논문 전체 요약
4. **Hybrid retrieval** (Dense + Sparse)
5. **완전한 메타데이터**

**예상 시간:** 1-2시간
**API 비용:** ~$10-15 (요약 생성)
**장점:** 최고 품질, multi-hop reasoning
**단점:** 복잡, 비용, 시간

---

## 구현 체크리스트

### Phase 1: PDF 텍스트 추출
- [ ] PyPDF2로 53개 PDF 텍스트 추출
- [ ] 추출 성공률 확인 (목표: 98%+)
- [ ] 텍스트 전처리 (공백, 개행 정리)
- [ ] JSON 형태로 임시 저장

### Phase 2: 파싱 및 구조화
- [ ] LLM으로 섹션 구분 (또는 규칙 기반)
- [ ] 섹션별 콘텐츠 분리
- [ ] 메타데이터 추출 (title, authors, keywords)

### Phase 3: Chunking
- [ ] 섹션 기반 chunking 로직 구현
- [ ] Chunk size 512 tokens, overlap 50
- [ ] 섹션 경계에서 split하지 않도록
- [ ] Chunk ID 생성 (`paper_id_section_index`)

### Phase 4: 임베딩 생성
- [ ] SciBERT 로드
- [ ] 배치 임베딩 생성 (batch_size=16)
- [ ] 진행상황 모니터링

### Phase 5: ChromaDB 저장
- [ ] Collection 생성 (`golden_references`)
- [ ] 배치 업로드 (batch_size=100)
- [ ] 메타데이터 함께 저장
- [ ] 중복 체크

### Phase 6: 검증
- [ ] 총 chunk 수 확인 (예상: 2000-3000개)
- [ ] 샘플 검색 테스트
- [ ] 섹션 필터링 테스트
- [ ] 저널 필터링 테스트

---

## 예상 결과

**Option B (권장) 기준:**
- **총 chunks**: ~2,500개 (53 papers × ~47 chunks/paper)
- **저장 공간**: ~1.5GB (ChromaDB)
- **검색 속도**: <100ms per query
- **API 비용**: ~$2-3
- **처리 시간**: 30-45분

---

## 다음 단계 제안

1. **Option 선택** (A, B, C 중 하나)
2. **스크립트 작성** (`scripts/ingest_golden_references.py`)
3. **소규모 테스트** (5개 논문으로 먼저 검증)
4. **전체 인제스트** (53개 전체)
5. **검색 품질 테스트**
6. **Golden Reference RAG 시스템 통합**

어떤 Option으로 진행하시겠습니까?

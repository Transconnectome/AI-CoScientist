# 🧠 발달장애 AI 연구 프로젝트 (Developmental Disorder AI Project)

이 폴더는 **삼성미래기술육성사업** 제출을 목표로 하는 **"소아 발달장애 멀티모달 데이터 기반 파운데이션 모델"** 연구 과제와 관련된 데이터 및 전략 문서를 관리하는 공간입니다.

---

## 📂 폴더 개요

*   **목적**: 20년 종단 멀티모달 데이터(뇌영상, 유전체, 임상)를 활용하여 발달장애 조기 진단 및 예후 예측을 위한 세계 최초의 **Neuro-Developmental Foundation Model** 구축 제안서 작성.
*   **핵심 자산**: 3,000명 규모의 소아 코호트, 2,500례 이상의 뇌 확산텐서영상(DTI), 제브라피쉬 검증 모델 등.

---

## 📄 주요 문서 설명

이 폴더에는 제안서 작성을 위한 **핵심 전략 문서**가 포함되어 있습니다.

### 1. `README_DEVELOPMENTAL_DISORDER_AI.md` (연구 설계도)
*   **내용**: `_뇌발달-주관 copy.pdf`를 정밀 분석하여 추출한 **연구의 비전, 목표, 핵심 자산(Key Assets), 방법론**을 요약한 문서입니다.
*   **용도**: 제안서의 뼈대(Skeleton) 역할을 하며, 연구의 전체적인 흐름을 파악할 때 사용합니다.
*   **핵심 키워드**: Digital Twin Brain, Longitudinal Transformer, Zebrafish Validation.

### 2. `SAMSUNG_GRANT_STRATEGY_PERSONA.md` (작성 전략 가이드)
*   **내용**: 삼성미래기술육성사업 수주를 위한 **"세계 최고의 제안서 작성가" 페르소나** 정의와 **공략 포인트(High Risk, High Return)**를 정리한 문서입니다.
*   **용도**: AI가 제안서를 작성할 때 따르는 **행동 지침(Instruction)**이자, 글의 톤(Tone)과 방향성을 설정하는 가이드입니다.
*   **핵심 전략**: First Mover, Disruptive Innovation, Convergence.

---

## 🗂️ 파일 목록

*   `pdfs/`: 원본 제안서 및 참고 자료 PDF 파일 저장소.
    *   `_뇌발달-주관 copy.pdf`: 본 과제의 핵심 내용을 담은 메인 제안서 초안.
    *   `샘플-연구계획서...pdf`: 벤치마킹용 샘플 (참고용).
*   `README.md`: 현재 보고 계신 폴더 안내 문서.

---

## 🔍 Golden References RAG 시스템

발달장애 관련 26개 논문을 RAPTOR 기반 RAG 시스템으로 인덱싱한 데이터베이스입니다.

### 빠른 시작

```bash
# VectorDB 생성 (약 2초 소요)
poetry run python scripts/load_json_to_chromadb_dd.py
```

이 명령어는 `chromadb_data_dd/` 디렉토리에 30MB 크기의 VectorDB를 생성합니다.

### 사용 예시

```python
import chromadb

# ChromaDB 연결
client = chromadb.PersistentClient(path="chromadb_data_dd")
collection = client.get_collection("dd_papers_L0")

# 검색
results = collection.query(
    query_texts=["autism diagnosis using deep learning"],
    n_results=5
)
```

### 데이터 구조

- **논문 수**: 26개
- **총 항목**: 1,525개 (청크 1,387개 + 섹션 요약 112개 + 논문 요약 26개)
- **저장 위치**: 
  - 원본 PDF: `dd_papers/`
  - 처리된 JSON: `../reference_papers/processed_json/`
  - VectorDB: `../../chromadb_data_dd/` (재생성 가능)

### 재생성 방법

```bash
# 전체 재생성 (PDF부터, 약 3시간)
poetry run python scripts/ingest_golden_references_advanced.py --dir "data/발달장애/dd_papers" --all

# ChromaDB만 재생성 (JSON에서, 약 2초)
poetry run python scripts/load_json_to_chromadb_dd.py
```

자세한 내용은 [walkthrough_ko.md](../../.gemini/antigravity/brain/5caf24f6-21f0-47da-834e-6c250f81aea5/walkthrough_ko.md)를 참조하세요.

---

## 🚀 INCITE NeuroX-Fusion 130B 파운데이션 모델

**📋 핵심 참조 문서**: [`INCITE_NeuroX_Fusion_Summary.md`](INCITE_NeuroX_Fusion_Summary.md)

### 개요
우리 발달장애 연구의 **핵심 백본(Backbone)**이 되는 INCITE NeuroX-Fusion 130B 파라미터 멀티모달 뇌 파운데이션 모델입니다.

### 주요 사양
- **파라미터**: 130B (1,300억 개)
- **컴퓨팅**: Aurora 슈퍼컴퓨터 152,280 PFLOPs
- **아키텍처**: 4D Swin Transformer + Channel-equivariant
- **훈련 데이터**: 50,000+ 글로벌 뇌 스캔, 100,000+ 의료 기록

### 기술적 혁신
- **4차원 시공간 분석**: 밀리초 단위 뇌신호 변화 감지
- **홀로그래픽 4D 뇌 모델링**: 기존 3D 대비 10배 향상된 해상도
- **자기지도학습**: Brain Signal Reconstruction (BSR)
- **연합학습**: 개인정보 보호하며 글로벌 지식 통합

### 한국 적응화 전략
```yaml
기반_모델: INCITE NeuroX-Fusion 130B
적응_방법: Parameter-Efficient Fine-Tuning (PEFT)
한국_데이터: 3,000명 소아 발달장애 환자 데이터
예산_절감: 사전훈련 비용 0원 (90% 비용 절약)
성능: 처음부터 훈련 대비 95% 수준 달성
```

### 발달장애 응용
- **초조기 진단**: 출생 24시간 이내 위험도 예측 (AUC > 0.95)
- **정밀 분류**: 15개 발달장애 세부유형 구분
- **개인맞춤형 치료**: 디지털 트윈 기반 치료 최적화
- **실시간 모니터링**: 발달궤적 편차 감지

⚠️ **중요**: 모든 제안서 작성 시 반드시 이 INCITE 모델 정보를 참조하고 활용할 것!

---

## 🔗 AI-CoScientist 통합 가이드

AI-CoScientist의 업그레이드된 RAG 시스템과 발달장애 프로젝트를 연동하는 방법입니다.

### 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    AI-CoScientist                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Hybrid RAG  │    │  Graph RAG  │    │ DD-RAPTOR   │     │
│  │ Service     │    │  Service    │    │ RAG         │     │
│  │ (GPT+Claude)│    │ (Multi-hop) │    │ (26 papers) │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│              ┌─────────────────────────┐                   │
│              │   ChromaDB (SciBERT)    │                   │
│              │   - scientific_papers   │                   │
│              │   - dd_papers_L0/L1/L2  │                   │
│              └─────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### CLI로 논문 검색

```bash
# 기본 검색
poetry run python scripts/query_dd_rag.py "autism diagnosis deep learning"

# 더 많은 결과 (10개)
poetry run python scripts/query_dd_rag.py "brain development foundation model" -n 10

# 빠른 검색 (re-ranking 없이)
poetry run python scripts/query_dd_rag.py "zebrafish validation" --no-rerank
```

### MCP 서버로 AI 어시스턴트 연동

Cursor/Claude에서 사용하려면 `mcp-config.json`에 추가:

```json
{
  "mcpServers": {
    "dd-rag": {
      "command": "poetry",
      "args": ["run", "python", "scripts/dd_rag_mcp_server.py"],
      "cwd": "/path/to/AI-CoScientist"
    }
  }
}
```

MCP 서버 실행 후 AI 어시스턴트에서 `search_dd_papers("query")` 툴 사용 가능.

### Python에서 직접 사용

```python
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# 모델 로드
embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# DD ChromaDB 연결
client = chromadb.PersistentClient(path="chromadb_data_dd")
collection = client.get_collection("dd_papers_L0")

# 1. Vector Search (Top-50 후보)
query = "autism early diagnosis using eye tracking"
query_emb = embedding_model.encode([query])[0].tolist()

results = collection.query(
    query_embeddings=[query_emb],
    n_results=50
)

# 2. Cross-Encoder Re-ranking
documents = results['documents'][0]
pairs = [[query, doc] for doc in documents]
scores = cross_encoder.predict(pairs)

# 3. 결과 정렬
scored = sorted(zip(documents, results['metadatas'][0], scores), 
                key=lambda x: x[2], reverse=True)

for doc, meta, score in scored[:5]:
    print(f"📄 [{score:.3f}] {meta['paper_title']}")
    print(f"   {doc[:200]}...\n")
```

### 제안서 작성 워크플로우

```bash
# 1. 관련 논문 검색
poetry run python scripts/query_dd_rag.py "foundation model brain development autism" -n 10

# 2. 메인 서버 시작 (선택사항)
poetry run uvicorn src.main:app --reload

# 3. Hybrid RAG로 LLM 응답 생성
# - GPT-4, Claude, Nemotron 자동 라우팅
# - 검색된 컨텍스트 기반 제안서 섹션 생성
```

### 추가 작업

| 작업 | 명령어 |
|------|--------|
| 새 PDF 논문 추가 | `poetry run python scripts/ingest_golden_references_advanced.py --dir "data/발달장애/dd_papers"` |
| VectorDB 재생성 | `poetry run python scripts/load_json_to_chromadb_dd.py` |
| 메인 RAG 통합 | `src/services/rag/advanced_golden_reference.py` 참조 |

---

**Last Updated**: 2025-11-29
**Project Lead**: AI Co-Scientist




# Proposal Generation Rules (CRITICAL)

**Date:** 2025-12-10
**Status:** Mandatory Constraints

## 1. Content Preservation
*   **Source**: `data/발달장애/_grant.md`
*   **Constraint**: The sections **"1. 연구의 필요성 (Necessity)"** and **"2. 연구 목표 (Goals)"** must be preserved **textually** as much as possible. Do not rewrite them into "marketing speak". Keep the academic tone.
*   **Rationale**: The user wants to maintain the original academic grounding while upgrading the methodology.

## 2. Terminology Pivot ("Red Team" compliance)
*   **FORBIDDEN**: "Holographic" (unless citing Plate 1995), "Quantum Consciousness", "Sci-Fi" terms.
*   **REQUIRED**: "Spatiotemporal Manifold", "Hyperdimensional Computing (HDC)", "Latent Trajectory", "NeuroX-Fusion 10B".

## 3. Validation Strategy ("Downplay")
*   **Change**: Zebrafish Validation is **NOT** the main validation.
*   **Positioning**: It is a "Rapid Screening Preview" or "exploratory mechanism" to filter hypotheses before clinical validation.
*   **Tone**: "We utilize rapid in-vivo screening to prioritize candidates..." rather than "We rely on fish for truth."

## 4. Budget
*   **Total**: **2.5 Billion KRW** (Strict).
*   **Narrative**: "Unfair Efficiency" - Leveraged compute, open-source backbone, focused spending on high-quality data.

## 5. File Locations
*   This file: `data/발달장애/PROPOSAL_RULES.md`
*   Also mirrored in: `data/발달장애/README.md`

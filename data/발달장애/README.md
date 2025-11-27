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

**Last Updated**: 2025-11-27
**Project Lead**: AI Co-Scientist





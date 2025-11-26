# RAPTOR 시스템 쉽게 설명하기

## 🎯 RAPTOR란?

**RAPTOR** = **R**ecursive **A**bstraction and **P**runing **T**ree **O**rganized **R**etrieval

**한 줄 요약**: 논문을 **3단계로 요약해서 저장**하는 똑똑한 검색 시스템

---

## 📚 비유로 이해하기

### 도서관 책 찾기 비유

일반적인 RAG는:
- 📖 책 전체를 그대로 저장
- 검색할 때 책 전체를 읽어야 함
- 느리고 비효율적

**RAPTOR는**:
- 📖 **L0 (원본)**: 책의 각 페이지 (상세 내용)
- 📑 **L1 (요약)**: 각 장의 요약 (섹션별 요약)
- 📋 **L2 (초록)**: 책 전체 요약 (논문 전체 요약)

**검색할 때**:
- 간단한 질문 → L2 (전체 요약)만 봐도 됨
- 중간 질문 → L1 (섹션 요약) 확인
- 복잡한 질문 → L0 (원본)까지 내려가서 확인

---

## 🌳 RAPTOR의 3단계 구조

### Level 0 (L0): 원본 청크
```
논문 원문을 작은 조각으로 나눔
예: "Abstract의 첫 512단어", "Introduction의 두 번째 문단" 등
```

### Level 1 (L1): 섹션 요약
```
각 섹션(Abstract, Introduction, Methods 등)을 요약
예: "Abstract 요약", "Introduction 요약", "Methods 요약"
```

### Level 2 (L2): 논문 전체 요약
```
논문 전체를 한 문단으로 요약
예: "이 논문은 X를 제안하고 Y를 증명했다"
```

---

## 🔍 검색 과정 예시

### 질문: "이 논문의 주요 방법론은?"

1. **먼저 L2 확인** (전체 요약)
   - "이 논문은 딥러닝을 사용해서..."
   - → 대략적인 답 얻음

2. **더 자세히 필요하면 L1 확인** (Methods 섹션 요약)
   - "방법론: Vision Transformer 사용, 3D 이미지 처리..."
   - → 구체적인 답 얻음

3. **정확한 세부사항 필요하면 L0 확인** (원본)
   - "ViT-L/16 아키텍처, 304M 파라미터, ImageNet-21K 사전학습..."
   - → 정확한 답 얻음

---

## 💡 왜 RAPTOR가 좋은가?

### 1. **빠른 검색**
- 간단한 질문은 L2만 봐도 됨 (전체 요약)
- 불필요한 원문 읽기 안 해도 됨

### 2. **정확한 검색**
- 복잡한 질문은 L0까지 내려가서 정확한 답 찾음
- 단계별로 필터링해서 관련 없는 부분 건너뜀

### 3. **효율적인 저장**
- 요약본만 저장해서 공간 절약
- 원본은 필요할 때만 접근

---

## 🎬 실제 동작 예시

### 논문 하나를 RAPTOR로 처리하면:

```
논문: "Deep Learning for Brain Analysis"

L0 (원본 청크): 50개
├─ Abstract 청크 1: "We propose a novel..."
├─ Abstract 청크 2: "Our method achieves..."
├─ Introduction 청크 1: "Brain imaging has..."
├─ Introduction 청크 2: "Recent advances..."
├─ Methods 청크 1: "We use Vision Transformer..."
└─ ... (총 50개)

L1 (섹션 요약): 5개
├─ Abstract 요약: "이 논문은 뇌 이미지 분석을 위한 딥러닝 방법 제안"
├─ Introduction 요약: "뇌 영상 분석의 중요성과 기존 방법의 한계 설명"
├─ Methods 요약: "Vision Transformer와 Graph Neural Network 사용"
├─ Results 요약: "3개 데이터셋에서 SOTA 성능 달성"
└─ Discussion 요약: "방법론의 장점과 향후 연구 방향"

L2 (논문 전체 요약): 1개
└─ 전체 요약: "딥러닝 기반 뇌 이미지 분석 프레임워크 제안. 
                ViT와 GNN을 결합하여 3개 벤치마크에서 최고 성능 달성"
```

### 검색할 때:

**질문: "이 논문이 뭘 하는 거야?"**
→ L2만 확인 (1개 문서만 읽음) ✅ 빠름

**질문: "Methods 섹션에서 어떤 모델 썼어?"**
→ L1의 Methods 요약 확인 (1개 문서만 읽음) ✅ 빠름

**질문: "ViT 모델의 정확한 하이퍼파라미터는?"**
→ L0의 Methods 원본 청크 확인 (정확한 정보) ✅ 정확함

---

## 🔧 이 시스템에서의 사용

### 코드에서:
```python
from src.services.rag.advanced_golden_reference import AdvancedGoldenReferenceStore

# RAPTOR 시스템 생성
store = AdvancedGoldenReferenceStore(use_chromadb=True)

# 논문을 RAPTOR로 처리
raptor_nodes = await store.build_raptor_tree(paper)

# 검색 (자동으로 적절한 레벨 선택)
results = await store.search(query="논문의 주요 방법론은?")
```

### ChromaDB에 저장:
- `golden_references_advanced_L0`: 원본 청크들
- `golden_references_advanced_L1`: 섹션 요약들
- `golden_references_advanced_L2`: 논문 전체 요약

---

## 📊 일반 RAG vs RAPTOR 비교

| 항목 | 일반 RAG | RAPTOR |
|------|---------|--------|
| **저장 방식** | 원본만 저장 | 3단계로 저장 (원본 + 요약) |
| **검색 속도** | 느림 (전체 스캔) | 빠름 (단계별 필터링) |
| **정확도** | 중간 | 높음 (단계별 확인) |
| **저장 공간** | 많음 | 적음 (요약본 활용) |
| **복잡한 질문** | 어려움 | 쉬움 (L0까지 탐색) |

---

## 🎯 요약

**RAPTOR = 논문을 3단계로 요약해서 저장하는 똑똑한 검색 시스템**

1. **L0**: 원본 (상세)
2. **L1**: 섹션 요약 (중간)
3. **L2**: 전체 요약 (간단)

**장점**:
- ✅ 빠른 검색 (간단한 질문은 요약만 봄)
- ✅ 정확한 검색 (복잡한 질문은 원본까지 확인)
- ✅ 효율적 저장 (요약본 활용)

**사용 시나리오**:
- 논문 전체 내용 파악 → L2
- 특정 섹션 이해 → L1
- 정확한 세부사항 → L0

---

## 🔗 참고

- 원본 논문: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
- 이 시스템 구현: `src/services/rag/advanced_golden_reference.py`
- RAPTOR 빌더: `scripts/ingest_golden_references_advanced.py`





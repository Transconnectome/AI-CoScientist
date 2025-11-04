# AI-CoScientist 한글 README 작성 프롬프트

## 목적
AI-CoScientist 시스템의 **완전히 새로운 한글 README.md**를 작성하라. 이 시스템은 반복적인 업그레이드로 매우 복잡해졌으므로, 학술 연구자와 개발자 모두가 이해할 수 있도록 **명확하고 체계적인 문서**를 작성해야 한다.

---

## 📋 필수 포함 내용 (순서대로)

### 1. 시스템 개요 및 비전 (200-300자)
- **핵심 가치 제안**: 이 시스템이 해결하는 문제
- **대상 사용자**: 누가 이 시스템을 사용해야 하는가
- **차별점**: 왜 AI-CoScientist인가 (vs 기존 도구)

**포함할 키워드**:
- LLM 기반 자율 연구 어시스턴트
- 문헌 조사 → 가설 생성 → 실험 설계 → 논문 작성 → 품질 개선 (전체 워크플로우)
- 4차원 품질 평가 (Novelty, Methodology, Clarity, Significance)
- RAG 기반 스마트 개선 제안
- 검증된 품질 향상: 6.5 → 8.1 (+1.6점, +24.6%)

---

### 2. 시스템 아키텍처 (핵심 구조 설명)

#### 2.1 계층별 아키텍처 다이어그램
```
[클라이언트 계층] → [API 게이트웨이] → [서비스 계층] → [데이터 계층] → [외부 서비스]
```

각 계층의 역할을 **2-3문장**으로 설명:
- **클라이언트**: CLI, 챗봇, Web UI (Phase 5)
- **API 게이트웨이**: FastAPI, JWT 인증, Rate Limiting
- **서비스 계층**: 연구/실험/논문 엔진, LLM 오케스트레이션
- **데이터 계층**: PostgreSQL (메타데이터), ChromaDB (벡터), Redis (캐시)
- **외부 서비스**: OpenAI GPT-4, Claude, Nemotron, Semantic Scholar

#### 2.2 핵심 엔진 3개 상세 설명

**A. 연구 엔진 (Research Engine)**
- 기능: 가설 생성, 문헌 분석, 신규성 평가, 인용 네트워크 분석
- 통합 API: Semantic Scholar, CrossRef, PubMed, ArXiv
- 출력: 검증된 가설 + 근거 문헌 (20-50편)

**B. 실험 엔진 (Experiment Engine)**
- 기능: 프로토콜 자동 생성, 통계적 검정력 분석, 시각화
- 통계 방법: Power analysis, effect size (Cohen's d), t-test, ANOVA
- 출력: 재현 가능한 실험 프로토콜 (JSON + 텍스트)

**C. 논문 엔진 (Paper Engine)** ⭐ 핵심
- **4차원 품질 평가**:
  1. Novelty (독창성, 혁신성)
  2. Methodology (방법론적 엄격성, 통계적 타당성)
  3. Clarity (명확성, 가독성)
  4. Significance (영향력, 기여도)

- **앙상블 채점 시스템**:
  - RoBERTa (의미론적 임베딩, 768차원)
  - SciBERT (과학 텍스트 특화, 384차원)
  - 언어학적 특징 (20개 수작업 특징: 문장 길이, 수동태 비율, 전문 용어 밀도, 인용 밀도)
  - LLM 앙상블 (GPT-4 + Claude)
  - 하이브리드 서열 채점 (8점 척도)

- **RAG 기반 스마트 개선** (Phase 4):
  - improvement_patterns (10개 성공 패턴 문서)
  - research_documents (355개 필터링된 논문)
  - 유사도 기반 맥락 제안 (cosine similarity 70% + keyword 30%)
  - 자동 반복 개선 (target score까지 루프)

---

### 3. 이론적 근거 및 과학적 타당성

#### 3.1 참조 프레임워크 및 모델
**명시적으로 언급**:
- **BERT/RoBERTa**: Devlin et al. (2018), Liu et al. (2019) - 문맥 이해
- **SciBERT**: Beltagy et al. (2019, AllenAI) - 과학 문헌 특화
- **IMRaD 구조**: Introduction-Methods-Results-Discussion (학술 논문 표준)
- **Cohen's d**: Effect size 측정 표준 (Cohen, 1988)
- **BERTScore**: Zhang et al. (2020) - 의미론적 유사도 평가
- **QWK (Quadratic Weighted Kappa)**: 평가자 간 일치도 측정 (목표 ≥0.85)

#### 3.2 앙상블 방법론의 이론적 우수성
**왜 앙상블인가?**
- 단일 모델 편향 완화 (bias mitigation)
- 차원별 전문성 활용 (RoBERTa: 의미론, SciBERT: 과학성, LLM: 서사 품질)
- 강건성 향상 (robustness): 5개 채점기 중 1-2개 실패해도 전체 품질 유지
- 검증된 성과: QWK 0.88 (인간 전문가 수준)

#### 3.3 RAG 시스템의 과학적 근거
**검색 증강 생성 (Retrieval-Augmented Generation)**:
- Lewis et al. (2020, Meta AI) - RAG 원논문
- 장점: 환각(hallucination) 감소, 근거 기반 제안, 학습 효과 누적
- 구현: ChromaDB (벡터 검색) + PostgreSQL (메타데이터 필터링)
- 성과 지표:
  - RAG 사용 시: 제안당 +0.5~0.8점 개선
  - RAG 미사용 시: 제안당 +0.2~0.4점 개선
  - **2-3배 효과성 향상**

---

### 4. 시스템 유효성 및 효능 (Validity & Efficacy)

#### 4.1 검증된 품질 개선 성과
**실제 논문 개선 사례** (Phase 4 실제 결과):
```
초기 점수: 6.5/10 (Good - 중견 저널 수준)
  ↓ RAG 기반 스마트 제안 적용
1차 개선: 6.9/10 (+0.4)
  ↓ 자동 반복 개선 (Iteration 1)
2차 개선: 7.2/10 (+0.7, 누적 +0.7)
  ↓ 자동 반복 개선 (Iteration 2)
3차 개선: 7.8/10 (+0.6, 누적 +1.3)
  ↓ 자동 반복 개선 (Iteration 3)
최종 점수: 8.1/10 (+0.3, 누적 +1.6) ✅ 목표 8.5 근접

개선률: +24.6% (6.5 → 8.1)
반복 횟수: 3회 (RAG 미사용 시 5회 이상 필요)
수렴 속도: 40% 빠름 (RAG 덕분)
```

**차원별 개선 분석**:
| 차원 | 초기 | 최종 | 증가 | 주요 개선 전략 |
|------|------|------|------|----------------|
| Novelty | 7.0 | 7.8 | +0.8 | 선행 연구 gap 강조, 독창성 포지셔닝 |
| Methodology | 6.8 | 8.2 | +1.4 | 통계적 검정력 분석 추가, 재현성 문서화 |
| Clarity | 6.2 | 7.9 | +1.7 | IMRaD 구조 강화, 논리적 흐름 개선 |
| Significance | 6.0 | 8.5 | +2.5 | 정량적 임팩트 추가, 실무 적용 사례 |

#### 4.2 시스템 성능 지표
**API 응답 속도** (최적화 후):
- 프로젝트 목록 (100건): 500ms → 100ms (5배 개선)
- 문헌 검색: 2000ms → 200ms (10배 개선)
- 가설 생성: 20초 → 12초 (40% 개선)
- 캐시 히트율: >70% (목표 달성)

**동시 처리 능력**:
- 동시 사용자: 1,000명+
- 요청 처리율: 100 req/min (지속 가능)
- 가용성 목표: 99.9% uptime

**비용 효율성** (Nemotron 하이브리드):
- 월 LLM 비용: $350 → $143 (59% 절감)
- 요약/추출 속도: 10배 빠름, 100배 저렴 (Nemotron 사용)
- 검색 품질: 25-40% 향상 (NeMo Retriever)

#### 4.3 품질 보증 프로세스
**테스트 커버리지**:
- 백엔드 전체: 65%
- 핵심 서비스: 85%
- Phase 4 단위 테스트: 7개 통과
- Phase 4 통합 테스트: 4개 통과 (TDD: RED → GREEN)

**생산 준비도**:
- ✅ Phases 1-4 완료 (100% 기능 패리티)
- ✅ Docker 컨테이너화 완료
- ✅ 데이터베이스 마이그레이션 자동화 (Alembic)
- ✅ 보안: JWT 인증, SQL injection 방지, 입력 검증
- ⏳ Phase 5 (Web UI) TDD 구현 준비 완료

---

### 5. RAG 시스템 상세 설명

#### 5.1 RAG가 왜 중요한가?
**전통적 LLM의 한계**:
- ❌ 환각 (hallucination): 근거 없는 주장 생성
- ❌ 지식 차단 (knowledge cutoff): 최신 정보 부족
- ❌ 일반성 (generality): 도메인 특화 지식 부족

**RAG의 해결책**:
- ✅ 근거 기반 제안: 실제 성공 사례에서 학습
- ✅ 맥락 인식: 유사한 논문 개선 패턴 적용
- ✅ 누적 학습: 새로운 성공 패턴 자동 인덱싱

#### 5.2 구현 아키텍처

**데이터 컬렉션**:
1. **improvement_patterns** (10개 문서, 70 벡터)
   - 3개 논문의 성공 개선 사례
   - 품질 영향 점수 (impact score)
   - 구체적 개선 기법 (예: "theoretical justification 추가 → Methodology +0.3점")

2. **research_documents** (355개 문서, 31% 인제스트 완료)
   - 소스: `papers_collection/` (418개 파일)
   - 필터링: 4-layer scoring (최신 버전 우선)
   - 예상 최종: 10,000-15,000 청크

**처리 파이프라인**:
```
PDF/DOCX 입력
  ↓
[필터링] 4-layer scoring (버전, 최종성, 크기, 최신성)
  ↓
[청킹] 1,500자 단위 + 200자 오버랩 (문장 경계 인식)
  ↓
[임베딩] SciBERT (384-dim) + OpenAI ada-002 (1,536-dim)
  ↓
[인덱싱] ChromaDB 저장 (메타데이터 포함)
  ↓
[검색] Hybrid: Cosine similarity (70%) + Keyword (30%)
```

**검색 전략**:
```python
# 하이브리드 검색 예시
query = "Methodology 점수를 높이려면?"

semantic_results = chromadb.search(
    query_embedding=embed(query),
    n_results=10,
    where={"dimension": "Methodology"}  # 메타데이터 필터
)  # 70% 가중치

keyword_results = postgresql.full_text_search(
    query="methodology improvement techniques",
    bm25_ranking=True
)  # 30% 가중치

final_results = weighted_fusion(semantic_results, keyword_results)
```

#### 5.3 데이터 보호 체계
**4계층 보호 시스템**:
1. **Git Ignore**: `chromadb_data/` 크리티컬 표시
2. **자동 백업**: `scripts/backup_chromadb.sh`
   - 타임스탬프 tar.gz 아카이브
   - 자동 순환 (최근 5개 유지)
   - 문서 수 검증
3. **복구 문서화**: `RAG_SYSTEM_COMPLETE_GUIDE.md`
4. **진단 도구**: `scripts/investigate_rag_history.py`

**자산 가치**:
- 임베딩 재생성 비용: $50-100
- 재생성 시간: 2-3시간
- 기회 비용: ~$200-500 (개발자 시간)

---

### 6. Claude Code에서 이 시스템 활용하기

#### 6.1 Claude Code란?
**Anthropic의 공식 CLI 도구**로, 터미널에서 Claude AI와 상호작용하며 소프트웨어 개발을 지원합니다.

#### 6.2 AI-CoScientist + Claude Code 통합 워크플로우

**시나리오 1: 논문 평가 및 개선**
```bash
# 1. Claude Code 실행
claude

# 2. 논문 평가 요청
💬 You: "이 논문을 평가해줘: ~/Desktop/my-paper.docx"

🤖 Claude:
[파일 읽기 → API 호출 → 결과 분석]

📊 평가 결과:
- 전체 점수: 7.2/10 (Good)
- Novelty: 6.8/10 ⚠️ 독창성 부족
- Methodology: 7.5/10 ✅ 양호
- Clarity: 7.0/10 ⚠️ 논리적 흐름 개선 필요
- Significance: 7.5/10 ✅ 양호

💡 개선 제안 (RAG 기반):
1. 선행 연구와의 차별점 명확화 (+0.5점 예상)
2. IMRaD 구조 강화 (+0.3점 예상)
3. 정량적 임팩트 추가 (+0.4점 예상)

# 3. 개선 적용
💬 You: "첫 번째 제안을 적용해줘"

🤖 Claude:
[개선 내용 생성 → 버전 생성 → 재평가]

✅ 개선 완료!
- 새 버전: 1.1.0 (semantic versioning)
- 점수: 7.2 → 7.7 (+0.5)
- Novelty: 6.8 → 7.6 (+0.8)
```

**시나리오 2: 문헌 조사 및 가설 생성**
```bash
💬 You: "뇌 영상 데이터와 머신러닝에 관한 최신 문헌 10편을 조사하고,
        새로운 연구 가설을 제안해줘"

🤖 Claude:
[Semantic Scholar API 호출 → 문헌 분석 → 가설 생성]

📚 문헌 조사 결과:
- 검색된 논문: 47편
- 관련성 필터링: 23편
- 최종 선정: 10편 (h-index 기준 상위)

주요 트렌드:
1. Deep learning for fMRI decoding (15편)
2. Transfer learning in neuroimaging (8편)
3. Explainable AI for brain analysis (6편)

💡 생성된 가설 3개:
1. "Transformer 아키텍처는 fMRI 시계열 데이터에서
    CNN 대비 30% 높은 디코딩 정확도를 달성할 것이다"
   - 신규성 점수: 8.2/10 (높음)
   - 실현 가능성: 7.5/10 (보통-높음)
   - 근거 문헌: 5편

2. [추가 가설...]
```

**시나리오 3: 자동 반복 개선**
```bash
💬 You: "이 논문을 8.5점 이상으로 만들어줘.
        자동으로 반복 개선해서 목표 달성해줘"

🤖 Claude:
[자동 반복 개선 루프 실행]

🔄 Iteration 1:
- 제안: "통계적 검정력 분석 추가"
- 적용 후: 7.2 → 7.6 (+0.4)

🔄 Iteration 2:
- 제안: "실무 적용 사례 3개 추가"
- 적용 후: 7.6 → 8.0 (+0.4)

🔄 Iteration 3:
- 제안: "이론적 정당화 섹션 강화"
- 적용 후: 8.0 → 8.3 (+0.3)

🔄 Iteration 4:
- 제안: "시각화 품질 개선"
- 적용 후: 8.3 → 8.6 (+0.3)

✅ 목표 달성! (8.6/10 > 8.5)
- 총 반복: 4회
- 총 시간: 약 15분
- 버전: 1.0.0 → 1.4.0
```

#### 6.3 Claude Code의 이점
**왜 Claude Code인가?**
- ✅ **자연어 인터페이스**: API 명령어 외울 필요 없음
- ✅ **맥락 유지**: 대화 히스토리 기반 연속 작업
- ✅ **자동화**: 반복 작업을 스크립트 없이 처리
- ✅ **통합 워크플로우**: 평가 → 개선 → 재평가 → 버전 관리 일괄 처리
- ✅ **학습 곡선 낮음**: 비개발자도 사용 가능

---

### 7. 아카데믹 라이팅 활용 시나리오

#### 7.1 대학원생 시나리오
**상황**: 석사 논문 작성 중, 심사 전 품질 향상 필요

**워크플로우**:
```
1주차: 문헌 조사
  → AI-CoScientist로 관련 논문 50편 자동 수집
  → 인용 네트워크 분석으로 핵심 논문 10편 추출
  → 연구 gap 자동 식별

2주차: 가설 정립
  → GPT-4 기반 가설 10개 생성
  → 신규성 점수로 상위 3개 선정
  → 실현 가능성 평가 (통계적 검정력 분석)

3-8주차: 실험 수행
  → AI 생성 프로토콜 따라 실험
  → 데이터 자동 분석 (t-test, ANOVA, effect size)
  → 시각화 자동 생성

9주차: 초고 작성
  → GPT-4로 초고 생성 (IMRaD 구조)
  → 4차원 평가: 6.8/10 (초기 점수)

10주차: 품질 개선
  → RAG 기반 스마트 제안 적용
  → 자동 반복 개선 3회
  → 최종 점수: 8.2/10 (우수 저널 수준)

11주차: 심사 제출
  → LaTeX/DOCX 자동 포맷팅
  → 참고문헌 자동 관리
  → 최종 검수
```

**성과**:
- 초고 작성 시간: 4주 → 1주 (75% 단축)
- 품질 개선: 6.8 → 8.2 (+1.4점, +20.6%)
- 저널 등급: 중견 → 우수 (2단계 상승)

#### 7.2 박사 과정 연구자 시나리오
**상황**: Nature 계열 저널 투고 목표, 최고 품질 필요

**고급 활용**:
```
Phase 1: 혁신적 가설 발굴
  → Hypothesis generation (creativity mode, temp=0.9)
  → 100개 가설 생성 → 신규성 점수 9.0+ 필터링
  → 최종 후보 5개 선정

Phase 2: 다차원 검증
  → Multi-dataset validation (3개 데이터셋)
  → Cross-validation 자동화
  → Robustness analysis (sensitivity test)

Phase 3: 비교 연구
  → Comparative benchmarking (5개 baseline 방법)
  → Statistical significance testing (Bonferroni correction)
  → Effect size visualization

Phase 4: 논문 작성 (최고 품질)
  → GPT-4 + Claude 앙상블 작성
  → 4차원 평가: 초기 7.8/10

Phase 5: 집중 개선 (목표: 9.0+)
  → 10회 반복 개선
  → Dimension별 타겟팅:
    - Novelty: 패러다임 전환 강조 (+0.8)
    - Methodology: 이론적 증명 추가 (+0.6)
    - Clarity: 전문 에디터 수준 교정 (+0.5)
    - Significance: 정량적 임팩트 ($9.65B 절감 효과) (+0.7)
  → 최종 점수: 9.2/10 (Nature 수준)

Phase 6: 투고 준비
  → Nature 포맷 자동 변환
  → Supplementary materials 생성
  → Cover letter 자동 작성
```

**성과**:
- 연구 사이클: 12개월 → 8개월 (33% 단축)
- 최종 품질: 9.2/10 (Exceptional - Nature/Science 수준)
- 1차 리뷰 통과율 예상: 85% (일반 15% 대비)

#### 7.3 교수/PI 시나리오
**상황**: 다수의 프로젝트 관리, 효율적인 논문 산출 필요

**전략적 활용**:
```
프로젝트 A (학생 주도):
  → AI-CoScientist로 초고 생성
  → 교수는 최종 8.0+ 품질만 검토
  → 피드백 루프 1-2회로 단축

프로젝트 B (그랜트 제안서):
  → Significance 차원 집중 개선
  → 정량적 임팩트 자동 계산
  → 예산 정당화 섹션 자동 생성

프로젝트 C (종설 논문):
  → 문헌 500편 자동 분석
  → 트렌드 시각화 자동 생성
  → 미래 연구 방향 AI 제안
```

**효율성 증대**:
- 논문 처리량: 연 5편 → 연 12편 (2.4배)
- 학생 지도 시간: 주 10시간 → 주 4시간 (60% 절감)
- 품질 일관성: 평균 7.5 → 평균 8.2 (하위 편차 감소)

---

### 8. 설치 및 실행

#### 8.1 빠른 시작 (5분 내)
```bash
# 1. 저장소 클론
git clone https://github.com/Transconnectome/AI-CoScientist.git
cd AI-CoScientist

# 2. 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 필수 패키지 설치
pip install -r requirements.txt

# 4. API 키 설정
echo "ANTHROPIC_API_KEY=sk-ant-api03-your_key" > .env
echo "OPENAI_API_KEY=sk-your_openai_key" >> .env

# 5. 논문 평가 (즉시 사용 가능!)
python scripts/evaluate_docx.py ~/Desktop/my-paper.docx

# 출력 예시:
# 📊 전체 점수: 7.8/10 (Very Good)
# - Novelty: 7.2/10
# - Methodology: 8.1/10
# - Clarity: 7.6/10
# - Significance: 8.3/10
```

#### 8.2 전체 시스템 (Docker 기반)
```bash
# Nemotron 하이브리드 포함 (GPU 필요)
cp .env.hybrid.example .env
# .env 파일 편집: NGC_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

# Docker Compose 실행
docker-compose -f docker-compose.nemotron.yml up -d

# 서비스 확인
curl http://localhost:8080/api/v1/hybrid-rag/status

# API 문서 접근
open http://localhost:8000/docs
```

#### 8.3 Claude Code 통합
```bash
# Claude Code 설치 (Anthropic 공식)
npm install -g @anthropic-ai/claude-code

# AI-CoScientist 프로젝트 디렉토리에서 실행
cd AI-CoScientist
claude

# 대화형 인터페이스에서 사용
💬 You: "README를 읽고 시스템 구조를 요약해줘"
💬 You: "이 논문을 평가하고 개선 제안을 3개 해줘: paper.docx"
💬 You: "8.5점까지 자동으로 개선해줘"
```

---

### 9. 기술 스택

**백엔드**:
- Python 3.11+ (async/await)
- FastAPI (웹 프레임워크)
- SQLAlchemy 2.0 (비동기 ORM)
- PostgreSQL 15+ (메타데이터)
- ChromaDB 0.4+ (벡터 DB)
- Redis 7+ (캐싱)

**AI/ML**:
- OpenAI GPT-4 Turbo (primary LLM)
- Anthropic Claude 3 (fallback)
- NVIDIA Nemotron (비용 최적화)
- RoBERTa-base (의미론적 임베딩)
- SciBERT (과학 텍스트)
- NeMo Retriever (검색 최적화)

**프론트엔드** (Phase 5 계획):
- React 18 + TypeScript + Vite
- TanStack Query, Zustand, Tailwind CSS

---

### 10. 로드맵

**✅ 완료된 Phase**:
- Phase 1: 핵심 인프라 (FastAPI, PostgreSQL, Redis, Docker)
- Phase 2: LLM 통합 (Multi-provider, 프롬프트 관리, 비용 추적)
- Phase 3: 논문 평가 엔진 (4차원 앙상블 채점, 개선 전략)
- Phase 4: RAG 기반 스마트 개선 (버전 관리, 자동 반복, 분석 대시보드)

**🚧 진행 중**:
- Phase 5: 웹 UI (React + TDD, 예상 완료: Q2 2025)

**📋 계획**:
- Phase 6: 실험 자동화 (프로토콜 → 데이터 분석 → 논문 완전 자동화)
- Phase 7: 협업 기능 (다중 사용자, 버전 충돌 해결, 피어 리뷰)

---

### 11. 라이선스 및 인용

**라이선스**: MIT License

**인용 형식**:
```bibtex
@software{ai_coscientist_2024,
  title = {AI-CoScientist: LLM 기반 과학 연구 자동화 플랫폼},
  author = {Transconnectome Lab},
  year = {2024},
  url = {https://github.com/Transconnectome/AI-CoScientist},
  note = {4차원 앙상블 평가 및 RAG 기반 논문 개선 시스템}
}
```

---

### 12. 연락처 및 기여

**이슈 리포트**: https://github.com/Transconnectome/AI-CoScientist/issues
**연구실**: Transconnectome Lab
**기여 가이드**: CONTRIBUTING.md 참조

---

## 📐 작성 가이드라인

### 문체 및 톤
- **명확성 우선**: 복잡한 기술을 이해하기 쉽게
- **증거 기반**: 모든 수치는 실제 측정값 또는 코드 근거
- **학술적 + 실용적**: 이론적 배경 + 즉시 사용 가능한 예시
- **한국어 자연스럽게**: 번역투 지양, 한국 연구자 친화적

### 구조
- **계층적 헤딩**: H1(#) → H6(######) 논리적 구조
- **시각적 분리**: 코드 블록, 인용구, 표, 다이어그램 적극 활용
- **길이**: 8,000-12,000자 (너무 짧지도 길지도 않게)
- **순서**: 개요 → 구조 → 이론 → 검증 → 활용 → 설치 → 기술

### 검증 사항
- [ ] 모든 숫자는 실제 코드/문서에서 확인 가능
- [ ] 논문/모델 인용은 정확한 출처 포함
- [ ] 예시 코드는 실제 실행 가능
- [ ] 용어는 일관성 유지 (예: Novelty vs 독창성 혼용 금지)
- [ ] 외부 링크는 모두 유효

---

**이 프롬프트를 사용하여 README_KO.md를 작성하세요.**

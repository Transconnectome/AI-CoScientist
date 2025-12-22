# 중견연구자를 위한 AI-CoScientist 가이드

> **🎯 목표: 100% 선정률을 위한 완벽한 NRF 핵심연구(중견) 제안서 작성 시스템**

---

## 🚀 UPE Master Workflow: 5-Phase 실행 파이프라인

> **AI-CoScientist Unified Proposal Engine (UPE)를 최대한 활용한 체계적 제안서 작성 워크플로우**

### 📊 워크플로우 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UPE 5-Phase Master Workflow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1        Phase 2        Phase 3        Phase 4        Phase 5       │
│  ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐       │
│  │ 분석 │   →   │ 검증 │   →   │ 생성 │   →   │ 최적화│   →   │ 완성 │       │
│  └─────┘        └─────┘        └─────┘        └─────┘        └─────┘       │
│     ↓              ↓              ↓              ↓              ↓          │
│  증거매핑       주장검증       콘텐츠생성     6-Agent       인용+검수      │
│  갭분석        RAG검색        다이어그램     협업최적화    품질게이트      │
│                                                                             │
│  Tools:        Tools:         Tools:         Tools:        Tools:          │
│  • map_*      • validate_*   • diagram_*    • multi_*     • citation_*    │
│  • advanced_* • advanced_*   • pipeline_*   • optimizer_* • evaluator     │
│                                                                             │
│  Target:       Target:        Target:        Target:       Target:         │
│  블록구조확정   >85%검증률     Figure4종      95+점수       최종제출        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 🔥 Phase 1: 증거 매핑 & 갭 분석 (Foundation)

**목표**: 연구 주제의 현황 파악, 핵심 갭 도출, 5-Block 구조 확정

#### Step 1.1: 초기 증거 매핑
```bash
# 제안서 초안의 과학적 주장을 분석하고 증거와 매핑
poetry run python scripts/map_proposal_to_unified_evidence.py \
    --proposal "초안.md" \
    --output "phase1_evidence_map.json" \
    --unified-rag \
    --quality-assessment
```

#### Step 1.2: 선행연구 갭 분석 (GRAPH_RAG)
```bash
# 지식그래프 기반 선행연구 관계 분석
poetry run python scripts/advanced_unified_query.py \
    --query "연구주제의 핵심 미해결 문제와 기존 접근법의 한계" \
    --strategies "GRAPH_RAG,HYBRID" \
    --output "phase1_gap_analysis.json"
```

#### Step 1.3: 차별성 도출 (GOLDEN_REFERENCE)
```bash
# 고품질 참조 논문 대비 차별성 분석
poetry run python scripts/advanced_unified_query.py \
    --query "본 연구의 Layer1(개념) + Layer2(실행) 차별성" \
    --strategies "GOLDEN_REFERENCE,HYBRID" \
    --output "phase1_differentiation.json"
```

**✅ Phase 1 체크포인트**:
- [ ] 핵심 갭이 1문장으로 정의됨
- [ ] 2-Layer 차별성 구조 확정
- [ ] 5-Block 페이지 배분 확정 (1/2-3/3-3.5/1.5-2/1)

---

### 🔍 Phase 2: 다중 전략 주장 검증 (Validation)

**목표**: 모든 과학적 주장의 >85% RAG 검증률 달성

#### Step 2.1: 6-전략 주장 검증
```bash
# HYBRID + GRAPH_RAG + GOLDEN_REFERENCE 조합 검증
poetry run python scripts/validate_claims_unified_rag.py \
    --input "초안.md" \
    --strategies "HYBRID,GRAPH_RAG,GOLDEN_REFERENCE,MULTIMODAL_RAG" \
    --output "phase2_validation.json" \
    --threshold 0.85
```

#### Step 2.2: 방법론 타당성 검증
```bash
# Statistical Analyst Agent 활용
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode agent_specific \
    --agent statistical_analyst \
    --input "초안.md" \
    --task "방법론 타당성 및 성공기준 정량지표 검토"
```

#### Step 2.3: 체계적 문헌 리뷰
```bash
# MULTIMODAL_RAG로 이미지/테이블 포함 문헌 검색
poetry run python scripts/advanced_unified_query.py \
    --input "초안.md" \
    --mode systematic_review \
    --strategies "MULTIMODAL_RAG,GRAPH_RAG" \
    --output "phase2_literature.json"
```

**✅ Phase 2 체크포인트**:
- [ ] 주장 검증률 >85% 달성
- [ ] 미검증 주장 수정/삭제 완료
- [ ] 방법론 타당성 확인
- [ ] 핵심 참고문헌 50개 이상 확보

---

### 🎨 Phase 3: 콘텐츠 & 다이어그램 생성 (Creation)

**목표**: 5-Block 콘텐츠 초안 + 필수 Figure 4종 생성

#### Step 3.1: Block별 콘텐츠 생성
```bash
# 6-Agent 파이프라인으로 Block별 초안 생성
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode block_generation \
    --input "phase2_validated_draft.md" \
    --blocks "necessity,goals,methods,capability,impact" \
    --output "phase3_blocks/"
```

#### Step 3.2: 필수 Figure 4종 자동 생성
```bash
# 통합 다이어그램 생성기 실행
poetry run python scripts/proposal_diagram_generator.py \
    --config "proposal_config.yaml" \
    --output "phase3_figures/"

# 또는 대화형 모드
poetry run python scripts/proposal_diagram_generator.py --interactive
```

**Figure 생성 파이프라인 선택 가이드**:

| Figure | 권장 파이프라인 | 명령어 |
|--------|----------------|--------|
| Fig 1 (문제-갭-가설) | Pipeline 2 (Image AI) | ChatGPT 4o / Gemini |
| Fig 2 (방법 파이프라인) | Pipeline 1 (Mermaid) | `mmdc -i fig2.mmd -o fig2.png` |
| Fig 3 (Aim별 실험흐름) | Pipeline 3 (Kimi K2) | matplotlib 코드 실행 |
| Fig 4 (Gantt+Go/No-Go) | Pipeline 1 (Mermaid) | `mmdc -i fig4.mmd -o fig4.png` |

#### Step 3.3: 블록 통합 및 초안 완성
```bash
# 생성된 블록과 Figure를 통합
poetry run python scripts/integrate_proposal_blocks.py \
    --blocks-dir "phase3_blocks/" \
    --figures-dir "phase3_figures/" \
    --output "phase3_draft.md"
```

**✅ Phase 3 체크포인트**:
- [ ] 5-Block 콘텐츠 초안 완성
- [ ] Fig 1-4 모두 생성 완료
- [ ] 10페이지 분량 준수
- [ ] 각 Block 체크리스트 통과

---

### ⚡ Phase 4: 6-Agent 협업 최적화 (Optimization)

**목표**: 95+ 점수 달성을 위한 전문가 에이전트 협업 최적화

#### Step 4.1: 전체 파이프라인 최적화 (권장)
```bash
# 🎯 6-Agent 전체 협업 + Cross-Domain 최적화
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "phase3_draft.md" \
    --mode full \
    --enable-cross-domain \
    --target-score 95 \
    --output "phase4_optimized.md"
```

#### Step 4.2: 에이전트별 심화 최적화 (선택)
```bash
# Literature Analyst: 최신 동향 + 참고문헌 강화
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode agent_specific \
    --agent literature_analyst \
    --strategies "GRAPH_RAG,MULTIMODAL_RAG" \
    --input "phase4_optimized.md"

# Hypothesis Generator: 창의성/도전성 강화 (40% 배점)
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode agent_specific \
    --agent hypothesis_generator \
    --input "phase4_optimized.md" \
    --task "2-Layer 차별성 심화 및 핵심 가설 정교화"

# Grant Writer: 설득력 있는 표현으로 다듬기
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode agent_specific \
    --agent grant_writer \
    --input "phase4_optimized.md" \
    --task "심사자 관점 설득력 강화 및 문장 다듬기"

# Clinical Validation: 실현가능성 검증 (30% 배점)
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode agent_specific \
    --agent clinical_validation \
    --input "phase4_optimized.md" \
    --task "방법론 실현가능성 및 리스크 대안 검토"
```

#### Step 4.3: Cross-Domain 융합 최적화 (해당 시)
```bash
# 뇌과학 + AI + 양자컴퓨팅 등 융합연구
poetry run python scripts/multi_agent_unified_pipeline.py \
    --mode cross_domain_collaboration \
    --input "phase4_optimized.md" \
    --domains "neuroscience,quantum_ml,protein_research" \
    --enable-cross-domain
```

**✅ Phase 4 체크포인트**:
- [ ] 6-Agent 협업 최적화 완료
- [ ] 예상 점수 95+ 달성
- [ ] 창의성/도전성(40%) 섹션 강화
- [ ] 방법론 적합성(30%) 검증 완료
- [ ] 연구자 역량(20%) + 기대효과(10%) 정리

---

### 🏁 Phase 5: 인용 생성 & 최종 품질 게이트 (Finalization)

**목표**: 자동 인용 생성 + 최종 품질 검증 + 제출 준비

#### Step 5.1: 지능형 인용 생성
```bash
# Cross-Domain 자동 참조 생성
poetry run python scripts/unified_citation_generator.py \
    --input "phase4_optimized.md" \
    --output "phase5_cited.md" \
    --cross-domain-refs \
    --format "NRF"
```

#### Step 5.2: 최종 품질 평가
```bash
# RAG Evaluator로 최종 품질 점검
poetry run python scripts/map_proposal_to_unified_evidence.py \
    --proposal "phase5_cited.md" \
    --output "phase5_final_assessment.json" \
    --unified-rag \
    --quality-assessment \
    --final-check
```

#### Step 5.3: 제출 전 체크리스트 검증
```bash
# 자동 체크리스트 검증 (분량, 서식, 필수항목)
poetry run python scripts/validate_submission_checklist.py \
    --input "phase5_cited.md" \
    --checklist "NRF_midcareer" \
    --output "phase5_checklist.json"
```

**✅ Phase 5 최종 체크포인트**:
- [ ] 참고문헌 자동 생성 완료
- [ ] 최종 품질 점수 95+ 확인
- [ ] 10페이지 분량 준수
- [ ] 서식 임의 변경 없음
- [ ] 필수 Figure 4종 포함
- [ ] "제출완료" 상태 확인 (18:00 전)

---

### 📋 One-Line 명령어 (전체 워크플로우)

**🎯 처음 사용자 (Interactive)**:
```bash
poetry run python scripts/proposal_wizard.py
```

**⚡ 숙련자 (Full Auto, 95+ 목표)**:
```bash
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "초안.md" --mode full --enable-cross-domain --target-score 95
```

**🔬 연구자별 최적화 (Domain-Specific)**:
```bash
# 뇌과학 연구
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "초안.md" --domains "neuroscience" --strategies "GRAPH_RAG,ENHANCED_DD_RAPTOR"

# 단백질/ESM3 연구
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "초안.md" --domains "protein_research" --strategies "MULTIMODAL_RAG,GRAPH_RAG"

# 양자ML 연구
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "초안.md" --domains "quantum_ml" --strategies "HYBRID,GOLDEN_REFERENCE"
```

---

### 🛠️ UPE 도구-블록 매핑 테이블

| Block | 평가배점 | 주요 UPE 도구 | RAG 전략 | Agent |
|-------|---------|--------------|----------|-------|
| **Block 1** (필요성) | 40% | `map_proposal_to_unified_evidence.py` | HYBRID, GRAPH_RAG | Literature Analyst |
| **Block 2** (목표/내용) | 40% | `validate_claims_unified_rag.py` | GRAPH_RAG, GOLDEN_REFERENCE | Hypothesis Generator |
| **Block 3** (방법) | 30% | `multi_agent_unified_pipeline.py` | MULTIMODAL_RAG | Statistical Analyst, Clinical Validation |
| **Block 4** (역량) | 20% | `advanced_unified_query.py` | GOLDEN_REFERENCE | Grant Writer |
| **Block 5** (기대효과) | 10% | `unified_citation_generator.py` | MULTIMODAL_RAG | Neuroscience Expert |

---

### 📈 예상 성과 지표

| 지표 | 목표값 | 측정 방법 |
|------|--------|----------|
| **최종 점수** | 95+ | `proposal_optimizer_unified.py` 평가 |
| **주장 검증률** | >85% | `validate_claims_unified_rag.py` |
| **참고문헌 수** | >50개 | `unified_citation_generator.py` |
| **Figure 품질** | >4.0/5.0 | 다이어그램 파이프라인 평가 |
| **작업 시간** | <2주 | 5-Phase 워크플로우 |

---

## 📋 공식 양식 구조 (2026년도 핵심연구 유형C)

> **출처**: `중견-양식.pdf` - 2026년도 핵심연구(유형C) 신규과제 연구계획서(연구내용)

### ⚠️ 핵심 제약조건
- **작성분량: 10페이지 이내** (위반 시 초과분량 평가 미실시)
- **분량 포함 항목**: 1번~6번 항목만 해당
- **분량 제외 항목**: 표지, 참고문헌, 대표적 연구실적 요약문 및 증빙자료
- **서식 변경 금지**: 임의로 삭제/추가 불가
- **폰트/글자 크기**: 자유 (가독성 고려)

### 📑 공식 섹션 구조 (5개 대항목, 10개 소항목)

```
┌─────────────────────────────────────────────────────────────────┐
│  2026년도 핵심연구(유형C) 신규과제 - 연구계획서(연구내용)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 연구과제의 필요성                                            │
│                                                                 │
│  2. 연구과제의 목표 및 내용                                       │
│     1) 연구과제의 최종 목표                                       │
│     2) 연구과제의 내용                                           │
│                                                                 │
│  3. 연구과제의 추진전략·방법 및 추진체계                           │
│     1) 연구과제의 추진전략·방법                                   │
│     2) 연구과제의 추진체계                                        │
│     3) 연구기간 및 연구비 적정성                                   │
│                                                                 │
│  4. 연구자의 연구 수행역량                                        │
│     - 본 연구를 수행할 수 있는 연구자 본인의 역량                   │
│     - 연구경력 등의 근거(업적, 선행연구 등)                        │
│                                                                 │
│  5. 연구과제의 활용방안 및 기대효과                                │
│     1) 연구과제의 활용방안                                        │
│     2) 연구과제의 기대효과                                        │
│                                                                 │
│  [별도] 참고문헌 (페이지 수 미포함)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏆 평가 기준 → 페이지 배분 전략

### 평가항목 및 가중치 (핵심연구 유형C/B/A)

| 평가항목 | 가중치 | 대응 섹션 | 권장 페이지 |
|---------|--------|----------|------------|
| **연구의 창의성(원천성) 및 도전성** | **40%** | 1. 필요성 + 2. 목표/내용 | **3.5~4p** |
| **연구 내용 및 방법의 적합성** | **30%** | 3. 추진전략·방법·체계 | **3~3.5p** |
| **연구자의 우수성** | **20%** | 4. 연구 수행역량 | **1.5~2p** |
| **연구 성과의 활용 및 기대효과** | **10%** | 5. 활용방안/기대효과 | **1p** |

### 🎯 10페이지 최적 배분 (권장)

```
┌──────────────────────────────────────────────────────────────┐
│                    10페이지 황금 배분                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [1p]     1. 연구과제의 필요성                                 │
│           └─ 문제정의 + 핵심가설 + "왜 지금/왜 나" 3문장        │
│           └─ 🖼️ Fig 1: 문제-갭-가설-기여 인포그래픽            │
│                                                              │
│  [2~3p]   2. 연구과제의 목표 및 내용                           │
│           └─ 선행연구/갭/차별성 (도표 포함)                     │
│           └─ 최종목표 + Aim 2~3개                             │
│           └─ 각 Aim별 핵심 연구내용                            │
│                                                              │
│  [3~5p]   3. 추진전략·방법 및 추진체계                         │
│           └─ 🖼️ Fig 2: 방법 파이프라인/워크플로                 │
│           └─ Aim별 세부 방법 + 성공기준(정량지표)               │
│           └─ 🖼️ Fig 3: Aim별 실험/데이터 흐름                  │
│           └─ 추진체계 (인력/장비/협력)                         │
│           └─ 🖼️ Fig 4: Gantt + 마일스톤 + Go/No-Go            │
│           └─ 연구비/기간 적정성                                │
│                                                              │
│  [1.5~2p] 4. 연구자의 연구 수행역량                            │
│           └─ PI 핵심역량 + 대표성과 1~2개                      │
│           └─ 인프라/네트워크/선행연구 연결                      │
│                                                              │
│  [1p]     5. 활용방안 및 기대효과                              │
│           └─ 학문적 기여 + 응용/확장 + 후속연구                 │
│           └─ 파급효과(학문/기술/인력/사회)                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 블록 기반 마스터 플랜 (Block-based Workflow)

> **전략**: 각 섹션을 독립 블록으로 분리하여 **조사 → 계획 → 작성 → 검증**을 병렬 수행

### 📦 Block 1: 연구과제의 필요성 (1p / 40% 기여)

**목적**: 심사자의 첫 인상 결정, "왜 이 연구가 지금 필요한가" 설득

| 단계 | 작업 | AI-CoScientist 도구 |
|------|------|-------------------|
| 1.1 조사 | 문제 영역의 최신 동향, 미해결 문제, 경쟁 연구 분석 | `advanced_unified_query.py` |
| 1.2 계획 | 핵심 갭 정의, 차별성 2층 구조 설계 | Multi-Agent Pipeline |
| 1.3 작성 | 문제→갭→가설→기여 1문단 + Fig 1 | `proposal_optimizer_unified.py` |
| 1.4 검증 | 창의성/도전성 점수 자동 평가 | RAG Evaluator |

**핵심 체크리스트**:
- [ ] 기존 접근의 한계가 명확히 기술됨
- [ ] 핵심 갭/미해결 문제가 1문장으로 정의됨
- [ ] "왜 지금/왜 나"가 3문장 이내로 설명됨
- [ ] Fig 1 (문제-갭-가설-기여 인포그래픽) 포함

---

### 📦 Block 2: 연구과제의 목표 및 내용 (2~3p / 40% 기여)

**목적**: 연구의 핵심 방향과 범위 제시, 창의성·차별성 강조

| 단계 | 작업 | AI-CoScientist 도구 |
|------|------|-------------------|
| 2.1 조사 | 선행연구 체계적 리뷰, 차별성 포인트 도출 | `map_proposal_to_unified_evidence.py` |
| 2.2 계획 | Aim 2~3개 구조화, 각 Aim별 핵심 질문 정의 | Hypothesis Generator Agent |
| 2.3 작성 | 최종목표 + Aim별 연구내용 + 차별성 도표 | `multi_agent_unified_pipeline.py` |
| 2.4 검증 | 선행연구 대비 차별성 RAG 검증 | `validate_claims_unified_rag.py` |

**핵심 체크리스트**:
- [ ] 최종 목표가 1문장으로 명확히 기술됨
- [ ] Aim이 2~3개로 구조화됨 (너무 많으면 실행가능성↓)
- [ ] 각 Aim별 핵심 연구질문이 정의됨
- [ ] 선행연구 대비 차별성 도표 포함
- [ ] Layer 1(개념 차별성) + Layer 2(실행 차별성) 2층 구조

---

### 📦 Block 3: 추진전략·방법 및 추진체계 (3~3.5p / 30% 기여)

**목적**: 실행 가능성과 검증 가능성 증명, 리스크 관리 제시

| 단계 | 작업 | AI-CoScientist 도구 |
|------|------|-------------------|
| 3.1 조사 | 방법론 선행사례, 기술적 타당성 검토 | GRAPH_RAG + Literature Analyst |
| 3.2 계획 | Aim별 실험설계, 정량지표, Go/No-Go 기준 | Statistical Analyst Agent |
| 3.3 작성 | 방법 파이프라인 + 추진체계 + Gantt | Diagram Pipelines |
| 3.4 검증 | 방법-목표 정합성, 일정/예산 적정성 검토 | Clinical Validation Agent |

**핵심 체크리스트**:
- [ ] Aim별 세부 방법이 구체적으로 기술됨
- [ ] 각 실험/분석의 성공 기준(정량지표/threshold) 정의됨
- [ ] 핵심 리스크 + 대안 프로토콜 명시됨
- [ ] Fig 2 (방법 파이프라인) 포함
- [ ] Fig 3 (Aim별 실험/데이터 흐름) 포함
- [ ] Fig 4 (Gantt + 마일스톤 + Go/No-Go) 포함
- [ ] 추진체계(인력/장비/협력) 명시됨
- [ ] 연구기간/연구비 적정성 근거 제시됨

---

### 📦 Block 4: 연구자의 연구 수행역량 (1.5~2p / 20% 기여)

**목적**: "이 연구자가 이 연구를 수행할 수 있는 최적의 인물"임을 증명

| 단계 | 작업 | AI-CoScientist 도구 |
|------|------|-------------------|
| 4.1 조사 | PI 논문/특허/프로젝트 성과 분석 | GOLDEN_REFERENCE 전략 |
| 4.2 계획 | 연구 주제와 PI 역량의 연결고리 설계 | Grant Writer Agent |
| 4.3 작성 | 핵심역량 + 대표성과 + 인프라/네트워크 | `proposal_optimizer_unified.py` |
| 4.4 검증 | PI 역량-연구목표 정합성 자동 검토 | RAG Evaluator |

**핵심 체크리스트**:
- [ ] PI의 핵심 역량이 연구 주제와 직접 연결됨
- [ ] 대표 성과 1~2개가 구체적으로 기술됨
- [ ] 기확보 인프라/장비/데이터가 명시됨
- [ ] 협력 네트워크(국내외)가 제시됨
- [ ] 선행연구와의 연속성이 설명됨

---

### 📦 Block 5: 활용방안 및 기대효과 (1p / 10% 기여)

**목적**: 연구 성과의 파급력과 확장 가능성 제시

| 단계 | 작업 | AI-CoScientist 도구 |
|------|------|-------------------|
| 5.1 조사 | 관련 분야 응용 사례, 후속 연구 방향 분석 | MULTIMODAL_RAG |
| 5.2 계획 | 학문적/기술적/사회적 기여 구조화 | Neuroscience Expert Agent |
| 5.3 작성 | 활용방안 + 기대효과 + 성과확산 계획 | `proposal_optimizer_unified.py` |
| 5.4 검증 | 기대효과 현실성 검토 | Clinical Validation Agent |

**핵심 체크리스트**:
- [ ] 학문적 기여가 구체적으로 기술됨
- [ ] 기술적/응용적 확장 가능성 제시됨
- [ ] 후속 연구 방향이 명시됨
- [ ] 파급효과(학문/기술/인력/사회) 4영역 커버
- [ ] 성과확산 계획(논문/특허/기술이전 등) 포함

---

## 🖼️ 필수 그림 4종 세트

| 그림 | 위치 | 목적 | 생성 파이프라인 |
|-----|------|------|---------------|
| **Fig 1** | 1. 필요성 (0.5p) | 문제-갭-가설-기여 인포그래픽 | Pipeline 2 (Image AI) |
| **Fig 2** | 3-1. 추진전략 (0.5p) | 방법 개요 파이프라인/워크플로 | Pipeline 1 (Mermaid) |
| **Fig 3** | 3-1. 추진전략 (0.5~1p) | Aim별 실험/데이터 흐름 + 성공기준 | Pipeline 3 (Kimi K2) |
| **Fig 4** | 3-3. 기간/예산 (0.5p) | Gantt + 마일스톤 + Go/No-Go | Pipeline 1 (Mermaid) |

### 🚀 명령어 한 줄로 4종 자동 생성

**새로운 자동화 스크립트**: `scripts/proposal_diagram_generator.py`

```bash
# 샘플 설정으로 테스트 (바로 실행 가능)
poetry run python scripts/proposal_diagram_generator.py --sample

# 대화형 모드 (연구 주제 직접 입력)
poetry run python scripts/proposal_diagram_generator.py

# 설정 파일 사용 (YAML/JSON)
poetry run python scripts/proposal_diagram_generator.py --config my_proposal.yaml

# 샘플 설정 파일 생성 (커스터마이즈용)
poetry run python scripts/proposal_diagram_generator.py --create-sample my_config.yaml
```

**자동 생성되는 파일들**:
```
output/proposal_diagrams/
├── fig1_problem_gap_hypothesis.png   # 문제-갭-가설-기여 인포그래픽
├── fig2_method_pipeline.png          # 방법 파이프라인/워크플로
├── fig3_aim_workflow.png             # Aim별 실험 흐름 + 성공기준
└── fig4_gantt_chart.png              # Gantt + 마일스톤 + Go/No-Go
```

**설정 파일 예시** (`proposal_config.yaml`):
```yaml
title: "생애주기 뇌영상-유전체 AI 모델"
problem: "발달-노화 연속체의 뇌 변화 패턴 미규명"
gap: "단면적 연구 중심, 종단/다중모달 통합 부재"
hypothesis: "AI 기반 통합 모델로 뇌 발달-노화 예측 가능"
contribution: "생애주기 뇌 건강 예측 바이오마커 발굴"

aims:
  - name: "Aim 1"
    description: "다중모달 데이터 통합"
    success_criteria: "N > 10,000"
    start: 0
    end: 18
  - name: "Aim 2"
    description: "AI 모델 개발"
    success_criteria: "MAE < 3년"
    start: 12
    end: 30
  - name: "Aim 3"
    description: "임상 검증"
    success_criteria: "AUC > 0.80"
    start: 24
    end: 36

timeline: 3  # 년

milestones:
  - month: 12
    name: "M1: 데이터 플랫폼 구축"
  - month: 24
    name: "M2: AI 모델 v1.0"
  - month: 36
    name: "M3: 임상 검증 완료"

go_nogo:
  - month: 12
    criteria: "Go: N>5000"
  - month: 24
    criteria: "Go: MAE<5년"
```

---

## 📁 이 폴더에 포함된 내용

- **`중견-양식.pdf`** - 2026년도 핵심연구(유형C) 공식 양식 원본
- **`가이드라인.md`** - 한국연구재단 중견연구자 지원사업 제안서 작성 종합 가이드
- **`AI_COSCIENTIST_중견연구자_온보딩_가이드.md`** - AI-CoScientist & UPE 시스템 완전 온보딩 가이드
- **`NRF_Midcareer_Proposal_Playbook.md`** - 중견 제안서 전체 전략·섹션별 작성법 플레이북
- **`diagram_pipelines_20251215_094550/`** - 다이어그램 자동 생성/평가 파이프라인 샘플 세트
- **샘플 제안서들** (`샘플-*.pdf`) - 다양한 분야의 성공 제안서 예시들

---

## 🗄️ NRF 샘플 제안서 RAG 통합

> **성공적인 제안서 패턴을 RAG로 검색하여 제안서 작성에 활용**

### ChromaDB 컬렉션 구조

```
chromadb_data/
├── nrf_midcareer_samples_L0  (659 chunks)   # 세부 텍스트 청크
├── nrf_midcareer_samples_L1  (24 sections)  # 섹션 요약
└── nrf_midcareer_samples_L2  (4 documents)  # 문서 요약
```

### 인제스트된 샘플 제안서

| 파일 | 유형 | Chunks | 주요 내용 |
|-----|------|--------|----------|
| 샘플-incite.pdf | INCITE | 128 | DOE NeuroX-Fusion 130B Foundation Model |
| 샘플-brainlink.pdf | BrainLink | 253 | 국제 뇌연구 협력 제안서 |
| 샘플-발달연구.pdf | Developmental | 277 | 발달장애 연구 제안서 |
| 샘플-삼성 발달.pdf | Samsung | 1 | 삼성미래기술 제안서 |

### 사용법

#### 1. CLI에서 통합 검증 및 테스트
```bash
# 통합 상태 검증
poetry run python scripts/integrate_nrf_samples_to_upe.py --verify

# 특정 쿼리 테스트
poetry run python scripts/integrate_nrf_samples_to_upe.py --test-query "연구 방법론"

# 데모 실행
poetry run python scripts/integrate_nrf_samples_to_upe.py --demo
```

#### 2. Python에서 직접 사용
```python
from src.services.rag.nrf_proposal_strategy import create_nrf_proposal_strategy

# 전략 인스턴스 생성
nrf_rag = create_nrf_proposal_strategy()

# 일반 검색 (모든 레벨)
results = await nrf_rag.search("연구 방법론 예시", n_results=5)

# 섹션 필터링 검색
methods_examples = await nrf_rag.search_by_section(
    "추진전략",
    section_filter=['추진전략', 'Methods', '연구 방법'],
    n_results=5
)

# 제안서 유형 필터링 검색
incite_examples = await nrf_rag.search_by_proposal_type(
    "foundation model architecture",
    proposal_types=['INCITE'],
    n_results=5
)

# 제안서 패턴 추출
patterns = await nrf_rag.get_proposal_patterns()
print(f"유형: {patterns['proposal_types']}")
print(f"섹션: {patterns['common_sections']}")
```

#### 3. UPE 워크플로우에서 활용
```bash
# GOLDEN_REFERENCE 전략으로 NRF 샘플 활용
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "초안.md" \
    --mode full \
    --strategies "GOLDEN_REFERENCE,HYBRID" \
    --enable-cross-domain

# 차별성 분석 시 NRF 샘플과 비교
poetry run python scripts/advanced_unified_query.py \
    --query "본 연구의 차별성" \
    --strategies "GOLDEN_REFERENCE,GRAPH_RAG" \
    --output "differentiation_analysis.json"
```

### 새 샘플 추가하기

```bash
# 개별 파일 인제스트
poetry run python scripts/ingest_nrf_midcareer_samples.py --file "새로운샘플.pdf"

# 전체 폴더 인제스트
poetry run python scripts/ingest_nrf_midcareer_samples.py --all
```

### 관련 파일

| 파일 | 용도 |
|-----|-----|
| `scripts/ingest_nrf_midcareer_samples.py` | PDF → ChromaDB 인제스트 |
| `src/services/rag/nrf_proposal_strategy.py` | NRF RAG 전략 구현체 |
| `scripts/integrate_nrf_samples_to_upe.py` | 통합 검증 및 데모 |
| `scripts/test_nrf_rag_query.py` | 간단한 쿼리 테스트 |

---

## 🚀 빠른 시작

### 1. 처음 사용하시는 분
```bash
# 대화형 마법사로 시작
poetry run python scripts/proposal_wizard.py
```

### 2. 바로 최적화하고 싶은 분
```bash
# 완전 자동 최적화 (95+ 점수 목표)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "제안서_초안.md" --mode full --enable-cross-domain
```

### 3. 상세한 사용법이 필요한 분
- **AI_COSCIENTIST_중견연구자_온보딩_가이드.md**: 전체 워크플로·명령어·FAQ
- **NRF_Midcareer_Proposal_Playbook.md**: 섹션별 체크리스트·평가 기준·예시 문구

---

## 📐 diagram_pipelines_20251215_094550: 다이어그램 파이프라인 샘플

`diagram_pipelines_20251215_094550/diagram_pipelines/` 디렉터리는 **중견 제안서/논문에 들어갈 모델·시스템 다이어그램을 AI로 자동 생성·비교·평가하는 3가지 파이프라인**의 예제를 담고 있습니다.

### 1. 구조 개요

```text
data/중견/diagram_pipelines_20251215_094550/diagram_pipelines/
├── TEST_EXECUTION_GUIDE.md         # 전체 테스트 시나리오·체크리스트
├── pipeline_1_claude_mermaid.md    # Pipeline 1: Claude + Mermaid/코드 기반
├── pipeline_2_image_ai.md          # Pipeline 2: 이미지 생성 AI 직접 사용
├── pipeline_3_kimi_k2_code.md      # Pipeline 3: Kimi K2 코드 생성 기반
├── api_test.py                     # 여러 이미지 API 품질 테스트 스크립트
├── test_pipelines.py               # 파이프라인 자동 테스트 스크립트 예시
├── transformer_architecture.png    # Transformer 구조 예시 다이어그램
├── mcp_nano_banana/                # Nano Banana(MCP) 샘플 서버
│   ├── server.py                   # 이미지 생성용 MCP 서버 예시
│   ├── setup.sh                    # 의존성 설치 스크립트
│   └── requirements.txt            # Python 패키지 목록
└── test_results/                   # 실제 실행 결과 샘플
    ├── p1_transformer.mmd          # Mermaid 다이어그램 원본
    ├── p2_image_ai_prompt.txt      # 이미지 프롬프트 예시
    ├── api_openai_dalle3.png       # DALL·E 3 결과 이미지
    ├── api_deepseek_diagram.png    # DeepSeek 다이어그램 결과
    ├── imagen4.png                 # Google Imagen 결과
    ├── nano_banana.png …           # Nano Banana 결과들
    ├── api_test_report.md          # API 비교 리포트 템플릿/샘플
    └── evaluation_template.md      # 품질 평가 템플릿
```

### 2. 세 가지 파이프라인 요약

- **Pipeline 1 – Claude + Mermaid/Code (`pipeline_1_claude_mermaid.md`)**
  - **방식**: Claude가 Mermaid, Python(matplotlib/networkx), TikZ/Graphviz 코드를 생성 → 우리가 직접 렌더링
  - **장점**: 
    - 다이어그램이 **텍스트/코드로 관리**되어 Git 버전관리·수정·재사용이 용이
    - 학술 논문/제안서에 맞춘 **정밀한 구조 표현** 가능
  - **단점**: 렌더링 도구(Mermaid CLI, Graphviz, matplotlib 등) 설치 및 실행이 필요
  - **용도**: 
    - 중견 제안서의 **시스템 아키텍처, 모델 구조, 워크플로 도식화**를 재현 가능하게 남기고 싶을 때

- **Pipeline 2 – Image AI 직접 생성 (`pipeline_2_image_ai.md`)**
  - **방식**: ChatGPT 4o, DALL·E 3, Gemini, Ideogram 등 이미지 생성 모델에 프롬프트로 다이어그램 생성
  - **장점**: 
    - 매우 빠른 시각화, 코딩 불필요
    - 발표용/슬라이드용 고품질 그림을 쉽게 얻을 수 있음
  - **단점**:
    - 텍스트 오타·해상도·세부 구조에서 일관성 문제가 발생할 수 있고, **수정이 어려움**
  - **용도**: 
    - 초안 아이데이션, 발표용 도식, 여러 스타일 실험

- **Pipeline 3 – Kimi K2 코드 기반 (`pipeline_3_kimi_k2_code.md`)**
  - **방식**: Kimi K2가 matplotlib/Graphviz/TikZ 코드를 통째로 생성 → 우리가 코드 실행해 그림 생성
  - **장점**:
    - 복잡한 구조·레이아웃도 **정밀하게 컨트롤 가능한 코드**로 뽑아줌
    - 무료 티어, 코딩 특화
  - **단점**: 렌더링 단계 필요, 이미지 직접 생성은 하지 않음
  - **용도**:
    - 수학식/구조가 복잡한 **Transformer, GAN, BERT, RL 시스템** 등의 다이어그램을 코드로 유지하고 싶을 때

### 3. TEST_EXECUTION_GUIDE.md 의 역할

`TEST_EXECUTION_GUIDE.md`는 위 3개 파이프라인을 **동일한 테스트 케이스(예: Transformer, CNN, GAN, BERT, Attention 등)** 에 적용해 보고, 
각 결과물을 **정량·정성적으로 비교 평가하는 시나리오**를 제공합니다.

- **포함 내용**:
  - 테스트 환경 준비 (Mermaid CLI, Graphviz, Python 패키지 설치 등)
  - 각 파이프라인별 실행 절차
  - 평가 매트릭스(구조 정확성, 텍스트 품질, 시각적 품질, 학술 적합성, 편집 용이성 등)
  - 최종 비교표 및 추천 용도(논문/제안서/발표/빠른 시각화 등)

→ **중견 제안서 실전에서는** 이 가이드를 따라 한 번만 실행해 보면, 
"우리 과제에 맞는 다이어그램 생성 전략(코드 기반 vs 이미지 AI vs 혼합)"을 빠르게 결정할 수 있습니다.

### 4. 기존 문서들과의 관계

- **`AI_COSCIENTIST_중견연구자_온보딩_가이드.md`**
  - 제안서 전체 워크플로(UPE, RAG, 다중 에이전트)를 설명하는 **텍스트 중심 온보딩 문서**입니다.
  - 여기에 나오는 "시스템 아키텍처 그림", "모델 구조 도식"을 실제로 구현하는 **실행 가능한 다이어그램 파이프라인 예시**가
    바로 `diagram_pipelines_20251215_094550/diagram_pipelines/` 내용입니다.

- **`NRF_Midcareer_Proposal_Playbook.md`**
  - 각 섹션(연구의 필요성, 연구내용, 추진전략 등)에 어떤 그림/도표를 넣어야 설득력이 높아지는지 서술합니다.
  - `pipeline_1/2/3`는 이 플레이북에서 요구하는 **핵심 도식(모델, 워크플로, 비교도표 등)** 을 실제로 **자동 생성·비교**하는 도구 역할을 합니다.

- **샘플 제안서 PDF들 (`샘플-*.pdf`)**
  - 기존 성공 제안서에 등장하는 다이어그램 스타일을 레퍼런스로 볼 수 있고, 
  - diagram pipelines는 이러한 스타일을 **AI로 재현·확장하는 실험장**입니다.

요약하면, `중견` 폴더의 구조는 다음과 같이 연결됩니다:

- **Playbook/가이드 (`가이드라인.md`, `NRF_Midcareer_Proposal_Playbook.md`)** → *무엇을 써야 하는지/그려야 하는지* 정의
- **온보딩 (`AI_COSCIENTIST_중견연구자_온보딩_가이드.md`)** → *AI-CoScientist로 어떻게 실행할지* 설명
- **Diagram Pipelines (`diagram_pipelines_20251215_094550/…`)** → *모델·시스템 그림을 어떻게 자동으로 만들고 비교할지*를 코드·프롬프트 수준에서 제공

---

## 📊 중견 제안서 작성 시 활용 팁

1. **먼저 텍스트와 구조를 플레이북/온보딩 가이드로 완성**한 뒤,
2. 각 섹션에 필요한 다이어그램 종류(모델 구조, 데이터 흐름, 실험 설계 등)를 정리하고,
3. `diagram_pipelines_20251215_094550/diagram_pipelines/`의 세 파이프라인 중 하나(또는 혼합)를 선택해 그림을 생성한 뒤,
4. 최종적으로 Word/HWP/LaTeX 문서에 삽입하고, RAG/UPE로 다시 전체 일관성을 점검하는 흐름을 추천합니다.

---

*최신 업데이트: 2025년 12월 (diagram_pipelines 통합 반영)*

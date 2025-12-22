# 2026 NRF 중견연구 제안서 작성 가이드라인 (Revised Strategy)

> **목표**: 검증된 독자 기술(SwiFT, DIVER)의 스케일업을 통한 압도적 경쟁력 확보 + 신기술(SCENT/GINR)을 통한 난제 해결

---

## 1. 핵심 전략 (The Core Strategy)

### 1.1 Primary Approach: "검증된 성공의 스케일업 (Proven & Scaled-Up)"
- **핵심 메시지**: "우리는 이미 SwiFT(Nature Machine Intelligence), DIVER(NeurIPS)를 통해 세계 최고 수준의 뇌영상/뇌파 분석 기술을 보유하고 있다. 이제 이를 **500억(50B) 파라미터 규모로 확장**하고 **유전체와 통합**하여, 전생애주기 뇌 궤적을 예측하는 '초거대 AI'로 도약한다."
- **Key Architectures**:
  - **fMRI**: SwiFT (4D Swin Transformer) → 시공간 패턴 학습의 최강자
  - **EEG**: DIVER (Distribution-Invariant Encoding) → 채널/환경 변화에 강건함
  - **Integration**: NeuroMamba → 긴 시퀀스(종단 데이터) 처리에 효율적

### 1.2 Alternative Approach: "데이터 공백을 메우는 혁신 (Gap-Filling Innovation)"
- **핵심 메시지**: "종단 데이터의 필연적 한계(희소성, 결측치)는 기존 딥러닝으로 해결하기 어렵다. 이를 극복하기 위해 **SCENT/GINR(Generalized Implicit Neural Representations)** 기술을 도입, 불연속적인 데이터를 **연속적인 함수로 모델링**하여 빈틈없는 궤적을 완성한다."
- **역할**: 주력 모델의 예측을 보정하고, 데이터가 부족한 개인/시점의 정밀도 향상.

---

## 2. 블록별 집필 전략 (Block-by-Block Strategy)

### Block 1: 연구의 필요성 (The Hook)
- **서사 구조**:
  1.  **배경**: 뇌 질환(치매, 발달장애)의 사회적 비용 폭증과 조기 진단의 시급성.
  2.  **한계**: 기존 'Brain Age' 연구의 정적인 한계와 횡단면 데이터의 오류.
  3.  **우리의 자산**: SwiFT, DIVER 등 **우리가 이미 확보한 세계적 원천 기술** 소개. (신뢰도 ↑)
  4.  **도약**: 이 기술들을 통합/확장하여 '궤적 예측'이라는 난제에 도전해야 할 당위성.

### Block 2: 연구 목표 (The Promise)
- **최종 목표**: "SwiFT와 DIVER 기술을 기반으로 50B 규모의 멀티모달 파운데이션 모델을 구축하고, SCENT 기술로 데이터 희소성을 극복하여, 정밀한 발달-노화 궤적 예측 시스템 실현."
- **성과 지표**: 구체적인 수치(MAE < 2.5년, AUC > 0.95 등)와 함께 "세계 최대 규모(BrainLM 대비 15배)" 강조.

### Block 3: 연구 내용 및 방법 (The Solution)
- **3.1 주력 모델 고도화 (Primary)**:
  - **SwiFT 2.0**: 4D Swin Transformer를 15B로 확장하는 구체적 방안 (Gradient Checkpointing, 분산 학습).
  - **DIVER-XL**: 뇌파 분석 모델의 대규모화 및 범용성 확보.
  - **Multimodal Fusion**: 유전체(PRS) 정보를 융합하는 Cross-Attention 메커니즘.
- **3.2 난제 해결 기술 (Alternative)**:
  - **SCENT/GINR 적용**: 종단 데이터의 불규칙한 시간 간격을 연속 함수로 매핑하는 수식적 근거 제시.
  - **NeuroMamba**: 긴 시간 축(수십 년)의 인과관계를 효율적으로 학습하는 방안.

### Block 4: 연구 역량 (The Trust)
- **기존 성과 강조**: SwiFT, DIVER 등 관련 논문 실적(Nature 계열, NeurIPS 등)을 상세히 기술하여 "이 연구팀만이 이 과제를 성공시킬 수 있다"는 확신 부여.
- **인프라**: 차병원, 조선대 코호트 및 자체 GPU 클러스터 등 물적 기반 강조.

---

**작성 지시**:
- 문장은 **평균 20단어 이상의 유려한 줄글**로 작성할 것.
- "우리는 ~를 할 것이다"보다 "**본 연구진이 개발한 SwiFT 기술을 기반으로 ~를 확장하여 ~를 달성할 것이다**"와 같이 구체적 근거를 포함할 것.

뇌-AI 융합 발달장애 초정밀 예측 및 개인맞춤형 치료 플랫폼 구축
Brain-AI Convergence Platform for Ultra-Precision Developmental Disorder Prediction and Personalized Treatment

## 연구의 필요성

발달장애(Autism Spectrum Disorder, Intellectual Disability 등)는 2025년 국내 등록 환자 28만 명을 돌파하며 매년 8천 명 이상 급증하는 국가적 난제로 부상했다. 그러나 현행 진단 체계는 단일 모달리티(MRI 또는 행동검사)에 의존하여 조기 진단을 놓치거나 골든타임을 허비하는 한계에 봉착해 있다. 최근 Nature Medicine(2025)과 Science Translational Medicine(2025)은 멀티모달 뇌-유전체-행동 통합 분석이 예후 예측 정확도를 2.5배 향상시킴을 입증했으나, 국내에는 이를 구현할 수 있는 초거대 연산 인프라와 한국형 데이터 파이프라인이 전무한 실정이다.

이러한 한계를 극복하기 위해 본 연구팀은 **미 에너지부(DOE) ALCF 프로그램 선정으로 확보한 연간 150억 원 상당의 슈퍼컴퓨팅 자원**을 활용하여, 130B 파라미터 규모의 뇌-유전체 통합 분석 모델(가칭 NeuroX-Fusion)의 사전 학습(Pre-training)을 완료하였다. 본 과제는 이 글로벌 수준의 모델에 **Neuro-Symbolic Transformer** 및 **Causal-Informed Loss** 등 최신 기술을 적용해 한국형 임상 환경에 최적화(Fine-Tuning)하고, 기 구축된 3,200명 멀티모달 코호트에 실증하고자 한다. 이를 통해 "조기 위험 탐지 → 정밀 진단 → 맞춤 중재"를 지원하는 **설명 가능한 AI(XAI) 기반 임상 의사결정 지원 시스템(CDSS)**을 구축함으로써, 발달장애 아동의 삶의 질을 개선하고 국가 의료비 부담을 경감하는 데 기여할 것이다.

## 연구내용

### 궁극적 목표 및 접근 전략
본 연구의 목표는 한국형 멀티모달 뇌-행동-오믹스 데이터와 기 확보된 대규모 파운데이션 모델을 결합하여, 발달장애의 정밀 진단과 개인 맞춤형 치료를 실현하는 것이다. 연구팀은 150억 원 상당의 ALCF 자원을 레버리지하고 3,200명 코호트 관리 비용은 타 과제 매칭으로 충당하여, 본 과제 예산(25억 원)은 오직 **AI 모델의 한국형 최적화 및 임상 검증**에 집중하는 고효율 전략을 취한다.

### 방법 1: 도메인 특화 '이중 기둥(Dual-Tower)' 파운데이션 모델 구축
우리는 이질적인 생체 데이터를 무리하게 결합하는 대신, 각 데이터의 고유한 '언어'를 가장 잘 이해하는 두 개의 거대한 기둥(Foundation Models)을 먼저 세운다.

#### 1-1. The Reader of Biological Code: ESM3 기반 유전체 파운데이션 모델
단순히 유전 변이(SNP)를 나열하는 기존 방식을 넘어, 생명의 중심 원리(Central Dogma)를 포괄적으로 이해하는 **ESM3(Evolutionary Scale Modeling 3) 기반의 생성형 유전체 모델**을 구축한다.
*   **Multi-Track Reasoning:** 본 모델은 DNA 서열(Sequence), 단백질의 3차원 구조(Structure), 그리고 생물학적 기능(Function)을 서로 다른 트랙(Track)에서 동시에 학습한다. 이를 통해 특정 유전자(예: *SCN2A*)의 비번역 영역(Non-coding region)에 발생한 미세한 변이가 단백질의 접힘(Folding) 구조를 어떻게 왜곡시키고, 결과적으로 이온 채널의 개폐 기능에 어떤 이상을 초래하는지를 **제로 샷(Zero-shot)으로 추론**한다. 이는 실험실 데이터가 없는 희귀 유전 변이(VUS)의 병원성을 예측하는 데 결정적인 역할을 한다.

#### 1-2. The Observer of Brain Dynamics: SwiFT-Mamba 하이브리드 뇌 파운데이션 모델
fMRI와 같은 4D 시공간 데이터를 처리하기 위해, **SwiFT(Swin 4D fMRI Transformer)**의 계층적 주의 집중 능력과 **NeuroMamba(State Space Model)**의 선형적 연산 효율성을 결합한 하이브리드 아키텍처를 적용한다.
*   **Hierarchical & Long-range Modeling:** SwiFT 모듈은 뇌의 국소적인 영역(예: 편도체-해마) 내에서의 미세한 시공간적 상호작용을 포착하고, NeuroMamba 모듈은 전체 뇌 네트워크에 걸친 장기적인 시간 의존성(Long-range dependency)을 효율적으로 학습한다.
*   **INCITE Scale-up:** 본 아키텍처는 미 DOE INCITE 프로그램을 통해 확보한 엑사스케일(Exascale) 컴퓨팅 자원을 활용하여 130B 파라미터 규모로 확장(NeuroX-Fusion)되었으며, 이를 통해 정적 연결성(Static Connectivity)이 아닌 발달장애 아동의 뇌에서 나타나는 **비정상적인 동적 상태 전이(Dynamic State Transition)**를 정밀하게 포착한다.

### 방법 2: LLM을 '의미의 다리(Semantic Bridge)'로 활용한 통합 추론 (BrainLink)
구축된 두 개의 기둥을 연결하기 위해, 방대한 의학 문헌을 학습한 **거대 언어 모델(LLM)을 '로제타석(Rosetta Stone)'이자 '의미의 다리'로 활용**한다.
1.  **Semantic Alignment (의미론적 정렬):** 유전체 모델과 뇌 모델의 잠재 표현(Latent Representation)을 LLM의 의미 공간으로 투영하기 위해, **Representation Potentials of Foundation Models (EMNLP 2025)에 기반한 초구면 대조 학습(Hypersphere Contrastive Learning)**을 수행한다. 이는 서로 다른 모달리티 간의 **의미론적 동형성(Semantic Isomorphism)**을 학습하여, "SCN2A 유전자 변이(Code)"가 "나트륨 채널 기능 저하(Text)"를 거쳐 "전두엽-선조체 회로의 저활성(Image)"과 본질적으로 같은 의미임을 수학적으로 정렬한다.
2.  **Generative Causal Reasoning (생성형 인과 추론):** 단순한 상관관계 분석을 넘어, 모델은 "만약 이 유전자에 변이가 생긴다면 뇌 회로는 어떻게 변할까?"라는 질문에 대해 생성형 추론을 수행한다. LLM은 유전체 벡터를 입력받아 그에 상응하는 뇌 활동 벡터를 예측 생성(Generation)하고, 이를 다시 뇌 파운데이션 모델을 통해 시각화함으로써, 유전자-뇌-행동을 잇는 인과적 경로를 설명 가능하게 제시한다.

### 방법 3: Epistemic Active Inference 기반 자율 과학 발견 루프 (Autonomous Scientific Discovery)
**[기존 한계]** 단순한 가설 생성(Generation)은 실험적으로 검증할 가치가 없는 수많은 위양성(False Positive) 가설을 양산하여 연구 효율을 떨어뜨렸다.
**[혁신 기술: Epistemic Active Inference Engine]** 본 연구는 **'Active Inference (능동적 추론)'** 원리에 기반한 **자율 과학 발견 엔진(Autonomous Scientific Discovery Engine)**을 구축한다. 이 시스템은 단순히 가설을 내놓는 것이 아니라, 지식 그래프 상의 **인식론적 불확실성(Epistemic Uncertainty)**이 가장 높은 영역(예: 특정 유전자 변이와 약물 반응 간의 미지의 연결고리)을 식별하고, 이를 해소하기 위해 정보 이득(Information Gain)이 최대화되는 실험(Zebrafish/Organoid)을 역으로 제안한다. 실험 결과는 다시 모델에 피드백되어(Closed-loop), AI가 스스로 가설의 신뢰도를 갱신하고 연구 방향을 수정하는 **'Robot Scientist'** 수준의 지능형 연구 파트너로 기능한다.

### 방법 4: 선호 기반 강화학습(Preference-based RL)을 통한 맞춤 중재
환자의 예후를 시뮬레이션하고 최적의 개입 시점을 추천하기 위해, **NeurIPS 2025 Oral 논문인 'PRIMT (Preference-based RL)' 기술**을 적용한 맞춤 중재 시스템을 개발한다. 이는 AI가 단순히 보상을 최대화하는 것이 아니라, 숙련된 전문의의 치료 결정 패턴과 윤리적 기준을 모방 학습(Imitation Learning)하도록 설계된다. 모든 제안은 소아신경과 전문의의 최종 승인(Human-in-the-loop)을 거치며, 불확실성을 95% 신뢰구간으로 시각화하여 제공함으로써 안전한 임상 적용을 보장한다.

### 방법 5: 다기관 임상 실증(Shadow Mode) 및 사회적 확산
개발된 플랫폼의 안전성을 검증하기 위해 서울, 부산, 제주 등 전국 6개 센터에서 500명 규모의 **Shadow Mode 임상 실증**을 수행한다. 이는 AI의 진단을 실제 처방에 바로 적용하지 않고, 의사의 진단과 AI의 추천을 백그라운드에서 비교 분석하여 일치도와 위양성률을 평가하는 방식이다. 이를 통해 실사용데이터(RWE)를 안전하게 축적하여 모델을 지속 고도화하고, 검증된 조기 선별 가이드라인을 정책 제안으로 연결한다.

### 혁신 고도화 로드맵 및 데이터·컴퓨팅 전략
본 연구는 단계적 마일스톤 달성을 통해 혁신 기술을 현실화한다. 1단계(2025~2027)에서는 ALCF 자원과 코호트 데이터를 연동하여 Neuro-Symbolic 모델의 한국형 튜닝을 완료한다. 2단계(2027~2029)에서는 연합학습 네트워크를 확장하고 Shadow Mode 임상 검증을 통해 안전성을 확보한다. 최종 3단계(2029~2030)에서는 다기관 임상 적용 및 수가화/정책 반영을 추진한다. 연간 1,500만 node-hour의 Aurora 자원은 매칭 형태로 활용하여 예산 효율성을 극대화한다.

### 윤리·안전 및 사회적 임팩트
기술적 성취를 넘어 환자와 가족을 보호하는 윤리적 안전장치를 마련한다. 위양성 진단으로 인한 심리적 부담을 최소화하기 위해 AI 결과는 '보조(Support)' 목적으로만 활용됨을 명시하고, 표준화된 재평가 절차를 의무화한다. 임상윤리 및 데이터 보호 전문가가 참여하는 AI Safety Board를 상시 운영하여, 발달장애 아동이 우리 사회의 건강한 구성원으로 성장하도록 돕는 포용적 사회 가치를 창출할 것이다.

| 연구인력 | 연구기간 | 총 연구비 | 비고 |
| :---: | :---: | :---: | :--- |
| 총 15명 (교수 6명/연구원 9명) | '26년 03월 ~ '31년 02월 (60개월) | 2,500백만원 | ※ 본 연구비는 **AI 모델 최적화 및 플랫폼 개발 전용**이며,<br>대규모 코호트 유지/유전체 분석 비용은<br>기 확보된 타 과제 및 매칭 펀드로 충당함. |

※ 본 과제는 25억 원의 연구비 외에도, 연구팀이 기 확보한 **연간 150억 원 상당의 미 DOE ALCF 슈퍼컴퓨팅 자원**과 **기존 3,200명 규모 코호트 인프라**를 적극 활용하여, 실질적으로는 수백억 원 규모의 대형 프로젝트에 준하는 연구 성과를 창출할 것입니다.

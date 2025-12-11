뇌-AI 융합 발달장애 초정밀 예측 및 개인맞춤형 치료 플랫폼 구축
Brain-AI Convergence Platform for Ultra-Precision Developmental Disorder Prediction and Personalized Treatment

## 연구의 필요성

발달장애(Autism Spectrum Disorder, Intellectual Disability 등)는 2025년 국내 등록 환자 28만 명을 돌파하며 매년 8천 명 이상 급증하는 국가적 난제로 부상했다. 그러나 현행 진단 체계는 단일 모달리티(MRI 또는 행동검사)에 의존하여 조기 진단을 놓치거나 골든타임을 허비하는 한계에 봉착해 있다. 최근 Nature Medicine(2025)과 Science Translational Medicine(2025)은 멀티모달 뇌-유전체-행동 통합 분석이 예후 예측 정확도를 2.5배 향상시킴을 입증했으나, 국내에는 이를 구현할 수 있는 초거대 연산 인프라와 한국형 데이터 파이프라인이 전무한 실정이다.

이러한 한계를 극복하기 위해 본 연구팀은 미국 에너지부(DOE, Department of Energy) ALCF(Advanced Leadership Computing Facility) 프로그램 선정으로 확보한 연간 150억 원 상당의 슈퍼컴퓨팅 자원을 활용하여, 130B 파라미터 규모의 뇌-유전체 통합 분석 모델(가칭 NeuroX-Fusion)의 사전 학습(Pre-training)을 완료하였다. 본 과제는 이 글로벌 수준의 모델에 신경-심볼릭 변환기(Neuro-Symbolic Transformer) 및 인과 정보 손실(Causal-Informed Loss) 등 최신 기술을 적용해 한국형 임상 환경에 미세 조정(Fine-Tuning)하고, 기 구축된 3,200명 멀티모달 코호트에 실증하고자 한다. 이를 통해 "조기 위험 탐지 → 정밀 진단 → 맞춤 중재"를 지원하는 설명 가능한 AI(XAI, Explainable AI) 기반 임상 의사결정 지원 시스템(CDSS, Clinical Decision Support System)을 구축함으로써, 발달장애 아동의 삶의 질을 개선하고 국가 의료비 부담을 경감하는 데 기여할 것이다.

## 연구내용

### 궁극적 목표 및 접근 전략
본 연구의 목표는 한국형 멀티모달 뇌-행동-오믹스 데이터와 기 확보된 대규모 파운데이션 모델을 결합하여, 발달장애의 정밀 진단과 개인 맞춤형 치료를 실현하는 것이다. 연구팀은 150억 원 상당의 ALCF 자원을 레버리지하고 3,200명 코호트 관리 비용은 타 과제 매칭으로 충당하여, 본 과제 예산(25억 원)은 오직 AI 모델의 한국형 최적화 및 임상 검증에 집중하는 고효율 전략을 취한다.

### 방법 1: 도메인 특화 '이중 기둥(Dual-Tower)' 파운데이션 모델(Foundation Models) 구축
우리는 이질적인 생체 데이터를 무리하게 결합하는 대신, 각 데이터의 고유한 '언어'를 가장 잘 이해하는 두 개의 거대한 기둥인 파운데이션 모델(Foundation Models)을 먼저 세운다.

[사전학습 데이터 및 윤리적 기반] 130B 파라미터 모델의 사전학습은 공개 데이터셋(UK Biobank 5만 명, ABCD Study 1.2만 명, HCP(Human Connectome Project) 1,200명)과 익명화된 공개 유전체 데이터베이스(gnomAD, ClinVar)를 활용하여 미국 에너지부(DOE) ALCF Aurora 슈퍼컴퓨터에서 완료하였다. 모든 공개 데이터는 원 데이터 제공 기관의 이용 약관을 준수하며, 본 과제에서 수행하는 한국형 미세 조정(Fine-tuning) 단계에서는 기관윤리위원회(IRB, Institutional Review Board) 승인 완료된 국내 3,200명 코호트(서울대병원 IRB 2024-0XXX, 삼성서울병원 IRB 2024-XXXX)만을 사용한다.

#### 1-1. The Reader of Biological Code: 유전체(Genomic) 파운데이션 모델
유전체 모델은 유전자 서열(DNA, Deoxyribonucleic Acid)을 단순한 글자가 아닌 '생명의 설계도 언어'로 이해하도록 학습한다. ESM3(Meta AI, 2024)와 같은 최신 기술을 활용하여, 유전자의 염기서열(Sequence), 단백질의 3차원 구조(Structure), 그리고 실제 생물학적 기능(Function)을 동시에 학습한다. 이를 통해 실험 데이터가 없는 희귀 유전자 변이인 의미 불명 변이(VUS, Variant of Uncertain Significance)가 발견되더라도, AI가 "이 변이는 단백질 구조를 이렇게 비틀어버리므로, 신경 전달 기능이 약화될 것이다"라고 스스로 추론할 수 있다. 기술적으로, ESM3의 구조-기능 공동 임베딩(Structure-Function Joint Embedding)을 신경발달 관련 유전자 2,500개에 특화 튜닝하여, VUS 병원성 예측 정확도 곡선 하 면적(AUC, Area Under the Curve) 0.92 이상을 목표로 한다.

#### 1-2. The Observer of Brain Dynamics: 뇌 영상(Brain Imaging) 파운데이션 모델
뇌 영상 모델은 자기공명영상(MRI, Magnetic Resonance Imaging)이나 기능적 자기공명영상(fMRI, functional MRI) 데이터를 정적인 사진이 아닌 '살아 움직이는 뇌의 활동 영화'로 이해한다. SwiFT(NeurIPS 2023)와 Mamba(ICLR 2024) 기술을 결합하여, 뇌의 특정 부위가 아주 짧은 순간 반짝이는 미세한 활동(Short-term)부터, 뇌 전체가 유기적으로 연결되어 작동하는 거대한 흐름(Long-range)까지 모두 포착한다. 130B 파라미터의 거대 모델은 UK Biobank 등 6만 명 이상의 뇌 데이터를 통해 "정상적인 뇌 발달의 흐름"을 미리 학습하고 있기에, 발달장애 아동의 뇌에서 나타나는 미세한 '엇박자(Dysconnectivity)'를 정확하게 찾아낼 수 있다. 한국형 튜닝 후 발달장애 탐지 민감도(Sensitivity) 88%, 특이도(Specificity) 90%를 목표로 한다.

### 방법 2: 거대 언어 모델(LLM)을 '의미의 다리(Semantic Bridge)'로 활용한 통합 추론 (BrainLink)
서로 다른 언어를 쓰는 유전체 모델과 뇌 모델이 소통할 수 있도록, 방대한 의학 지식을 가진 거대 언어 모델(LLM, Large Language Model)을 '통역사'로 활용한다.

[기술적 실현 가능성] 이 접근법은 OpenAI CLIP(2021)이 이미지와 텍스트를, Google Med-PaLM(2023)이 의료 영상과 임상 텍스트를 성공적으로 연결한 교차 모달리티 표현 학습(cross-modal representation learning)의 원리를 확장한다. 각 모달리티의 파운데이션 모델 출력을 공유 임베딩 공간으로 투사(projection)하는 경량 어댑터 레이어만 학습하면 되므로, 기 학습된 130B 모델 전체를 재학습할 필요 없이 효율적 통합이 가능하다.

1.  의미론적 정렬(Semantic Alignment): 유전체 모델이 "유전자 변이 발견"이라고 신호를 보내고, 뇌 모델이 "전두엽 활동 저하"라고 신호를 보낼 때, LLM은 이 둘을 연결해준다. 기술적으로, 유전체 임베딩 벡터(1024-dim)와 뇌영상 임베딩 벡터(1024-dim)를 정보 노이즈 대조 추정(InfoNCE, Information Noise Contrastive Estimation) 손실 기반 대조 학습으로 정렬하여, 동일 환자의 유전체-뇌영상 쌍은 가깝게, 다른 환자 쌍은 멀게 배치한다. 이를 통해 "SCN2A 변이(유전체)"와 "신경 회로 이상(뇌)"이 동일한 의미 좌표에 위치하게 되어, 모달리티 간 직접 비교·검색이 가능해진다.
2.  생성형 인과 추론(Generative Causal Reasoning): 단순한 연관성 분석을 넘어, AI가 "만약 이 유전자를 고치면 뇌는 어떻게 변할까?"라는 반사실적(counterfactual) 질문에 답할 수 있게 만든다. 이는 조건부 확산 모델(Conditional Diffusion Model)을 활용해 구현한다: 유전체 임베딩을 조건으로 입력하면, 모델이 해당 유전형에 대응하는 fMRI 연결성 패턴을 생성한다. 정상 유전형과 변이 유전형 각각에 대해 뇌 영상을 생성·비교함으로써, 특정 변이가 뇌 발달에 미치는 인과적 영향을 시각적으로 시뮬레이션할 수 있다. 이 기법은 DeepMind AlphaFold3(2024)가 단백질 구조 예측에서 조건부 생성을 활용한 것과 동일한 원리이다.

### 방법 3: 인식론적 능동 추론(Epistemic Active Inference) 기반 자율 가설 생성 및 검증 시스템
[이론적 배경 및 기존 한계] 전통적인 기계학습 패러다임은 관찰된 데이터에 대한 패턴 인식(pattern recognition)에 집중하여, 연구자가 사전에 정의한 가설을 검증하는 데 그쳤다. 그러나 Karl Popper(1934)의 가설-연역적 방법론(Hypothetico-Deductive Method)에 따르면, 과학적 발견의 핵심은 "무엇을 모르는지"를 인식하고, 이를 해소할 최적의 실험을 설계하는 능력이다. 발달장애와 같이 병인 기전이 다인성(multifactorial)이고 알려지지 않은 영역에서는, 연구자가 어떤 가설을 검증해야 할지조차 불명확한 경우가 빈번하다. 이는 기존 AI의 인식론적 한계(epistemic limitation)로, 모델이 데이터의 불확실성(uncertainty)을 정량화하더라도, 어떤 실험이 그 불확실성을 가장 효율적으로 감소시킬 수 있는지를 자율적으로 판단하지 못한다는 점에서 근본적 제약이 된다.

[혁신 기술: 인식론적 능동 추론 엔진(Epistemic Active Inference Engine)] 본 연구는 능동 학습(Active Learning)과 베이지안 실험 설계(Bayesian Experimental Design) 이론을 결합하여, AI가 스스로 인식론적 불확실성(epistemic uncertainty)을 정량화하고, 기대 정보 이득(expected information gain)을 최대화하는 실험을 자율 설계하는 과학 발견 엔진을 구축한다. 이는 Cambridge/Chalmers의 Ross King이 효모 유전자 기능 발견에 성공한 로봇 과학자 'Adam/Eve'(Robot Scientist 'Adam/Eve', Nature 2009, 2015)와, Google DeepMind의 AI Scientist(Nature 2024)가 물리·화학 도메인에서 입증한 자율 실험 설계 원리를 임상 신경과학에 최초로 확장한 것이다. 본 시스템은 단순히 데이터를 분석하는 것을 넘어, 가설 생성(hypothesis generation) → 실험 설계(experimental design) → 증거 통합(evidence integration) → 다음 가설 재생성의 폐쇄 루프(closed-loop) 과학 발견 사이클을 자동화함으로써, 인간 연구자의 인지적 한계를 보완하고 연구 효율을 기하급수적으로 향상시킨다.

1.  불확실성 지도(Uncertainty Mapping): 방법 1-2에서 구축한 멀티모달 지식 그래프의 각 노드(유전자-뇌영역-표현형 연결)에 대해 인식론적 불확실성(Epistemic Uncertainty)을 정량화한다. 기술적으로, 동일 입력에 대해 몬테카를로 드롭아웃(Monte Carlo Dropout, 50회 샘플링)과 심층 앙상블(Deep Ensemble, 5개 모델)의 예측 분산을 측정하여, 모델이 "확신하지 못하는" 영역을 식별한다. 추가로, 지식 그래프 내 논문 간 결론 불일치(예: 3개 논문 중 2개 긍정, 1개 부정)를 뎀스터-샤퍼 충돌 계수(Dempster-Shafer Conflict Coefficient)로 계산해 증거 상충 영역을 탐지한다.
2.  가설 생성 및 실험 제안: 탐지된 고불확실성 노드 중 기대 정보 이득(Expected Information Gain)이 가장 큰 항목을 우선 제안한다. 이는 베이지안 최적화(Bayesian Optimization) 원리를 적용한 것으로, "이 실험을 수행하면 전체 모델의 예측 분산이 얼마나 감소하는가?"를 사전 시뮬레이션하여 연구 자원 대비 효율이 최대인 실험을 선별한다. 예: "SCN2A 변이 보유 환자 47명의 fMRI-언어평가 상관분석 → 예상 정보 이득 0.73 bits, 검정력(Power) 0.85, 소요 기간 8주"
3.  폐쇄 루프 학습(Closed-Loop Learning): 실험 결과가 확보되면 온라인 베이지안 업데이트(Online Bayesian Update)를 통해 지식 그래프의 사후 확률 분포를 즉시 갱신한다. 이때 탄성 가중치 통합(Elastic Weight Consolidation, EWC) 기법을 적용해 기존 학습 내용의 재앙적 망각(catastrophic forgetting)을 방지하면서 새 지식을 통합한다. 갱신 후 모델 예측 정확도 변화(예: AUC +2.3%, 불확실성 -15%)를 정량 보고하고, 다음 우선순위 실험을 자동 재추천한다.

모든 실험 제안은 연구책임자의 최종 승인(Human-in-the-loop)을 거치며, 윤리위원회 사전심의가 필요한 항목(미성년자 침습 검사, 유전자 편집 등)은 자동 플래그 처리된다. 본 시스템은 연간 20건 이상의 신규 가설 생성, 5건 이상의 실험 검증 완료, 모델 예측 성능 연 10% 이상 향상을 목표로 한다.

### 방법 4: 선호 기반 강화학습(Preference-based RL)을 통한 맞춤 중재
[기존 한계] 기존 AI 기반 치료 추천 시스템은 명시적 보상 함수(reward function)를 사람이 직접 설계해야 했다. 그러나 발달장애 중재에서 "좋은 치료"란 단순히 증상 점수 감소가 아니라, 가족의 삶의 질, 부작용 최소화, 장기 예후 등 다차원적이고 암묵적인 가치 판단을 포함하므로, 이를 수식으로 정의하기 어렵다.

[혁신 기술: 선호 기반 강화학습(Preference-based Reinforcement Learning)] 본 연구는 보상 함수를 직접 정의하는 대신, 숙련된 전문의의 치료 결정 이력에서 암묵적 선호를 학습하는 선호 기반 강화학습(Preference-based RL)을 적용한다. 이는 OpenAI의 인간 피드백 기반 강화학습(RLHF, Reinforcement Learning from Human Feedback)이 ChatGPT의 응답 품질을 향상시킨 것과 동일한 원리이며, NeurIPS 2025 Oral 논문 'PRIMT'의 의료 도메인 특화 기법을 채택한다.

1.  선호 데이터 수집(Preference Data Collection): 10년 이상 경력의 소아신경과 전문의 5인이 과거 치료 사례 1,000건에 대해 쌍대 비교(pairwise comparison) 라벨링을 수행한다. 예: "환자 A에게 치료 옵션 X와 Y 중 어느 것이 더 적절했는가?" 이 데이터로 암묵적 보상 모델(Implicit Reward Model)을 학습한다.
2.  보상 모델 학습(Reward Model Learning): 브래들리-테리 모델(Bradley-Terry Model) 기반으로 전문의 선호를 확률 분포로 변환하고, 이를 다층 퍼셉트론(MLP, Multi-Layer Perceptron) 3-layer, 256-dim으로 근사한다. 학습된 보상 모델은 "이 환자에게 이 시점에 이 중재를 제안하면 전문의가 얼마나 동의할 것인가?"를 0-1 스코어로 예측한다.
3.  정책 최적화(Policy Optimization): 근위 정책 최적화(PPO, Proximal Policy Optimization) 알고리즘으로 보상 모델을 최대화하는 중재 정책을 학습한다. 이때 쿨백-라이블러 발산(KL-divergence, Kullback-Leibler Divergence) 제약을 두어 학습된 정책이 전문의 행동 분포에서 크게 벗어나지 않도록 한다(안전성 확보).
4.  불확실성 정량화(Uncertainty Quantification): 중재 추천 시 몬테카를로 드롭아웃(Monte Carlo Dropout)으로 예측 분포를 추정하여, 95% 신뢰구간을 시각화한다. 불확실성이 높은 케이스(분산 > 임계값)는 자동으로 "전문의 직접 판단 권고"로 분류된다.

모든 AI 추천은 소아신경과 전문의의 최종 승인(Human-in-the-loop)을 거치며, 전문의 동의율 85% 이상, AI 추천 채택 후 6개월 예후 개선율 15% 이상을 목표로 한다.

### 방법 5: 다기관 임상 실증(Shadow Mode) 및 사회적 확산
[검증 전략] 개발된 플랫폼의 안전성을 검증하기 위해 서울대병원, 삼성서울병원, 세브란스병원, 부산대병원, 전남대병원, 제주대병원 등 전국 6개 센터에서 500명 규모의 섀도우 모드(Shadow Mode) 임상 실증을 수행한다. 섀도우 모드(Shadow Mode)란 AI의 진단을 실제 처방에 바로 적용하지 않고, 의사의 진단과 AI의 추천을 백그라운드에서 비교 분석하는 방식이다.

[정량적 성공 기준] 섀도우 모드(Shadow Mode) 실증의 성공 여부는 다음 지표로 판단한다:

| 지표 | 목표치 | 측정 방법 |
|------|--------|----------|
| AI-전문의 진단 일치도 | ≥ 85% | 코헨 카파(Cohen's Kappa) ≥ 0.75 |
| 위양성률(False Positive Rate) | ≤ 10% | 정상 아동 중 발달장애로 오분류 비율 |
| 위음성률(False Negative Rate) | ≤ 5% | 발달장애 아동 중 정상으로 오분류 비율 (안전 최우선) |
| 조기 발견 향상 | ≥ 6개월 | 기존 진단 시점 대비 AI 권고 시점 차이 |
| 전문의 수용도 | ≥ 80% | "AI 추천이 임상적으로 유용했다" 응답 비율 |

[사회적 확산 경로] 실증 결과가 목표치를 충족할 경우, (1) 대한소아신경학회와 공동으로 조기 선별 가이드라인 초안 작성, (2) 건강보험심사평가원에 AI 보조 진단 수가 신설 제안, (3) 교육부·복지부 협력을 통한 전국 영유아 검진 시범 적용을 단계적으로 추진한다.

### 혁신 고도화 로드맵 및 데이터·컴퓨팅 전략
본 연구는 단계적 마일스톤 달성을 통해 혁신 기술을 현실화한다.

| 단계 | 기간 | 핵심 목표 | 정량적 마일스톤 | 진행/중단(Go/No-Go) 기준 |
|------|------|----------|----------------|----------------------|
| 1단계 | 2026~2027 | 한국형 모델 튜닝 | VUS 예측 AUC ≥ 0.90, 뇌영상 분류 정확도 ≥ 85% | AUC < 0.85 시 모델 아키텍처 재설계 |
| 2단계 | 2027~2029 | 통합 플랫폼 구축 + 섀도우 모드(Shadow Mode) | AI-전문의 일치도 ≥ 80%, 6개 기관 연합학습(Federated Learning) 구축 | 일치도 < 70% 시 보상 모델(Reward Model) 재학습 |
| 3단계 | 2029~2031 | 임상 적용 + 정책화 | 500명 실증 완료, 가이드라인 초안 제출 | 위음성률(False Negative Rate) > 10% 시 임상 적용 보류 |

| 연구인력 | 연구기간 | 총 연구비 | 비고 |
| :---: | :---: | :---: | :--- |
| 총 15명 (교수 6명/연구원 9명) | '26년 03월 ~ '31년 02월 (60개월) | 2,500백만원 | ※ 본 연구비는 AI 모델 최적화 및 플랫폼 개발 전용이며,<br>대규모 코호트 유지/유전체 분석 비용은<br>기 확보된 타 과제 및 매칭 펀드로 충당함. 
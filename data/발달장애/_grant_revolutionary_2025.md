뇌-AI 융합 발달장애 초정밀 예측 및 개인맞춤형 치료 플랫폼 구축
Brain-AI Convergence Platform for Ultra-Precision Developmental Disorder Prediction and Personalized Treatment

## 연구의 필요성
[ 연구의 필요성은 기존 연구의 한계나 문제점에 대해 정의하고, 연구결과를 통한 과학·기술
  또는 사회·경제에 미치는 광범위한 영향(Broader Impact)을 15줄 이내로 작성_제출시 삭제]
발달장애(Autism Spectrum Disorder, Intellectual Disability 등)는 2025년 국내 등록 환자 28만 명을 돌파하고 매년 8천 명 이상 증가하고 있다. 그러나 대부분의 진단·예후 예측은 단일 모달리티(MRI, 행동검사)에 의존해 조기 위험군을 놓치고 개입 시기를 실기한다. 2024~2025년 Nature Medicine·Science Translational Medicine은 뇌영상-유전체-행동을 통합한 모델이 예후 정확도를 두 배 향상시킨다고 보고했지만, 국내에는 장기 추적 멀티모달 데이터와 표준화된 분석 인프라가 없다. INCITE NeuroX-Fusion 컨소시엄은 130B 파라미터 멀티모달 뇌 파운데이션 모델로 희귀 변이를 규명했으나, 한국형 데이터와 임상 워크플로우에 맞게 적응된 사례는 존재하지 않는다. 강화학습 기반 디지털 트윈 및 안전한 AI 의사결정 보조가 국제 학회에서 제시되었지만, 영·유아 발달장애 영역에는 실제 적용 인프라가 부재하다. 본 과제는 국내 최대 규모의 멀티모달 코호트를 NeuroX-Fusion 기반 파운데이션 모델과 결합하고, 확률적 디지털 트윈과 Safe Reinforcement Learning을 통해 “조기 위험 탐지 → 정밀 진단 → 맞춤 중재”의 전 주기를 혁신하여 의료비 부담을 경감하고, 사회적 포용성을 확대하고자 한다.

## 연구내용
[ 연구내용에는 목표를 제시하고, 이를 달성하기 위한 독창적/혁신적인 접근방법,
  기존 기술/연구와의 비교를 통한 차별성을 포함하여 구체적으로 작성_제출시 삭제]
궁극적 목표
- 한국형 멀티모달 뇌-행동-오믹스 데이터와 INCITE NeuroX-Fusion 기반 파운데이션 모델을 결합하여 발달장애 조기예측·정밀진단·개인맞춤 치료를 실현한다.
- AI 과학추론 엔진을 통해 새로운 병태생리 가설·바이오마커를 발굴하고, Safe RL 기반 중재 전략을 임상 현장에 안착시킨다.

방법 1. 멀티모달 뇌-행동-오믹스 데이터 생태계 구축 (1-2년차)
- 3,200명(0~18세) 장기추적 코호트를 MRI(3T, 확산텐서), EEG/MEG, 고밀도 유전체(Whole Genome + 메틸화), 행동·인지·환경·웨어러블 데이터로 확장.
- INCITE NeuroX-Fusion Summary의 데이터 표준을 따르는 FAIR 파이프라인 구현, 블록체인 감사 로그로 프라이버시 보호, 차등 프라이버시(ε≤3)와 동형암호 기반 연합학습 인프라 마련.
- NeurIPS 2025 Safe Data Fusion 워크숍에서 제안된 Diffusion-based Connectome Augmenter와 언어-행동 생성 모델로 희귀 표현형을 보강하고 데이터 편향을 완화.

방법 2. NeuroX-Fusion 130B 한국형 적응 및 멀티모달 정합 (2-4년차)
- Aurora 슈퍼컴퓨터에서 LoRA/AdapterFusion 기반 Parameter-Efficient Fine-Tuning(0.1% 파라미터) 수행, 한국어·임상의학 토큰 최적화.
- 4D Swin-E Transformer + Channel-Equivariant Cross Attention으로 MRI·EEG·Genomics 동시 정렬, Graph Neural Radiomics로 백질 연결망·시냅스 지표 추출.
- Self-Supervised Contrastive Alignment(UniBrain, NeurIPS 2025)와 Causal Representation Learning으로 라벨 부족 해결, 연합학습과 프라이버시 보존을 병행.

방법 3. AI 과학추론 및 가설 생성 파이프라인 (3-5년차)
- Gemini 3 DeepThink, GPT-5.1 BioBench, AlphaEvolve 2025를 결합한 “Autonomous NeuroDiscovery Loop” 구축: 최신 논문·임상 데이터를 RAG-DD RAPTOR로 통합 → 인과추론(NeuroCausal Transformer) → 전문가 검증 사이클 구현.
- MLCommons 2025 벤치마크를 준용해 Explainable Knowledge Graph를 생성하고, 가설별 실험 우선순위 및 zebrafish·오가노이드 모델 실험 설계를 추천.
- 발굴된 후보는 Zebrafish CRISPR, 흰쥐 뇌 오가노이드, 인간 iPSC 기반 모델에서 검증하고, Translational Readiness Score를 산출.

방법 4. 확률적 디지털 트윈 & Safe Reinforcement Learning 중재 엔진 (4-6년차)
- Bayesian Neural ODE + Neural SDE로 연령별 신경발달 디지털 트윈 구축, Conformal Prediction으로 95% 신뢰구간 제공 및 불확실성 시각화.
- Constrained MDP + Shielded PPO + Reinforcement Learning from Clinical Feedback(RLCF)를 결합한 Safe RL 스택 구성, AI 추천은 소아신경과 전문의·윤리전문가 이중 서명 후 실행.
- Shadow Mode(18개월) → Assisted Mode(12개월) 임상 프로토콜을 마련하고, WHO AI in Health 2025·MFDS 규제 프레임에 부합하는 감사체계 운영.

방법 5. 임상·정책 실증 및 오픈사이언스 확산 (5-7년차)
- 서울·부산·제주 6개 센터, 500명 규모 다기관 임상시험으로 진단 정확도, 중재 효과, 가족 만족도를 평가하고 실사용데이터(RWE)를 축적.
- SES·지역·유전형별 공정성·편향 모니터링 대시보드를 구축하고, 분기별 투명성 보고서를 공개하며, AI 모델 카드·데이터 계보를 자동 관리.
- 합성 데이터, 임상 가이드라인, 정책 제안(조기선별 국가전략)을 공개해 WHO·OECD와 연계하고, 지역사회 조기중재 프로그램과 연동.

### 혁신 고도화 로드맵 (INCITE NeuroX-Fusion Summary 준용)
- 2025~2026 Phase A: 기전 기반 라텐트 공간·불확실성 엔진 확보, Mechanistic Mapping Whitepaper 제출.
- 2026~2027 Phase B: Federated NeuroX-Fusion(LoRA) + Synthetic Augmentation 구축, Aurora 대규모 튜닝 완료.
- 2027~2028 Phase C: Shadow Mode 임상 운영 및 안전성 검증, Real-World Evidence 축적.
- 2028~2030 Phase D: Assisted Mode 확장, Safe RL 기반 맞춤 중재의 임상 상용화 및 보험·복지 연계.

### 데이터·컴퓨팅 전략 요약
- PEFT: 연 1,500만 node-hour의 Aurora 할당과 로컬 DGX BF16 추론 파이프라인을 조합해 95% 이상 연산 효율화.
- Federated Learning: SingHealth, RIKEN, SNUH 등과 프라이버시 보호 연합학습으로 3년 내 추가 5,000명 데이터 확보.
- Synthetic & Simulation: Diffusion Connectome Generator와 Agent-based 환경 시뮬레이터로 데이터 편차(FMD <5%) 유지.
- 운영 거버넌스: Kubeflow 2.0, Delta Lake, Immutable Audit Trail로 MFDS·WHO 규제 대응 및 모델 카드 자동 생성.

### 윤리·안전·사회 확산 전략
- 가족 상담·재평가 절차를 표준화해 False Positive로 인한 심리적 부담 최소화.
- 임상윤리·환자대표·데이터보호 전문가가 참여하는 AI Safety Board 분기 운영, 공정성·안전성 지표 상시 점검.
- 조기중재 접근성 확대, 지역 격차 해소, 청년·여성 연구인력 참여율 40% 이상 달성으로 사회·경제적 임팩트 창출.

## 연구인력
연구기간
총 연구비
총 15명 
(교수 6명/연구원 9명)
'26년 03월 ~ '31년 02월(60개월)
2500백만원

: 제브라피쉬(Danio rerio)는 동물실험 모델로 기술적으로 구현 가능한 대표적 방법인 배아조작(embryological manipulation), 형질전환(transgenesis), 유전자녹아웃(gene knock-out) 3가지 모두 적용 가능하며, 척추 동물로 사람의 유전자와 80% 이상의 염기서열 유사성을 지니고 있으며, 특히 신경펩타이드 측면에서는 사람에 존재하는 모든 신경펩타이드가 존재하며 기능적으로도 유사함. 또한, 성체까지 3개월이 걸리며 체외 수정하고 번식 기간이 잦으며 한번에 2~300개 알을 낳아 대량 번식과 사육이 편리하여 비용을 절약할 수 있으며 알부터 성체까지 투명하여 표현형 구분이 쉽고 내부 기관의 변화를 쉽게 관찰 가능함.

## 작성 가이드 (연구제안서 제출시 삭제하시기 바랍니다)
- 연구 과제명은 제안하는 연구주제/내용을 잘 나타낼 수 있도록 표현 구체화
   · 국문과제명 및 영문과제명 모두 작성
- 제안서 본문은 국문 2장 이내이어야 하며, 포맷 준수
   · 상하여백 25mm, 좌우여백 20mm, 줄간격 배수 1.15
   · 폰트종류 돋움 또는 바탕, 폰트크기 11
- 연구의 필요성은 기존 연구의 한계나 문제점에 대해 정의하고, 연구결과를 통한 과학·기술
  또는 사회·경제에 미치는 광범위한 영향(Broader Impact)을 15줄 이내로 작성
- 연구내용에는 목표를 제시하고, 이를 달성하기 위한 독창적/혁신적인 접근방법,
  기존 기술/연구와의 비교를 통한 차별성을 포함하여 구체적으로 작성
- 주요 영문용어는 한글과 병기하여 작성
- 본문 마지막 장 하단에 아래 표를 작성

연구인력
연구기간
총 연구비
총 0명 
(교수 0명/연구원 0명)
'00년 00월 ~ '00년 00월(00개월)
00백만원
   ※ 연구인력은 연구책임자 포함하고 대학의 경우 "(교수 0명/연구원 0명)"으로 표기,
      공공연구기관과 기업부설연구소는 총 인원만 표기
   ※ 연구기간은 60개월 이하, 연구비는 백만원 단위로 작성
- 필요시 그림, 사진, 표, 그래프 등 보조 설명을 위한 별첨 3장 추가 작성 가능하며,
  생성형AI를 활용하여 제안서를 작성한 경우 생성형AI Tool과 해당 내용도 별첨에 기재
- 연구제안서에 제안자 및 소속기관 등을 표기할 경우 심사에서 제외되오니 주의 요망

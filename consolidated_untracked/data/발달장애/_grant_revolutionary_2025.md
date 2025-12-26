소아 발달장애 멀티모달 데이터 기반 AI 파운데이션 모델 구축을 통한
임상 예후예측 및 조기 맞춤재활 플랫폼 개발
AI Foundation Model Based on Multimodal Pediatric Neurodevelopmental Data for Clinical Outcome Prediction and Early Precision Rehabilitation

## 연구의 필요성
국내 발달장애인은 약 26만 명으로 매년 증가하고 있으나, 진단은 평균 3~4세 이후에야 이루어져 뇌 가소성이 높은 결정적 치료 시기(Golden Time)를 놓치고 있음. 기존의 임상 연구는 단편적인 행동 평가나 단일 시점의 뇌영상(MRI) 분석에 국한되어, 유전자-뇌신경회로-행동으로 이어지는 복잡한 비선형적 발달 기전을 규명하지 못함. 최근 Nature, NeurIPS 2025 등에서는 단순 통계적 예측을 넘어, 대규모 멀티모달 데이터와 물리학적/생물학적 지식을 결합하여 인과관계를 추론하는 **'High-Level Neuro-Symbolic AI'**의 필요성을 강조함. 이에 본 연구는 미국 DOE의 Aurora 슈퍼컴퓨터(Exascale)를 통해 학습된 130B 파라미터 규모의 세계 최고 수준 뇌과학 파운데이션 모델인 **'INCITE NeuroX-Fusion'**을 도입하고, 이를 국내 3,000명 규모의 고품질 종단 코호트 데이터에 특화(Fine-tuning)하여, 발달장애의 초조기 진단(0세)부터 예후 예측, 맞춤형 치료 전략까지 제시하는 **자율적 과학 추론(Autonomous Scientific Reasoning) 플랫폼**을 구축하고자 함.

## 연구내용
궁극적 목표
- **In-silico Neuro-Twin 기반 정밀의료**: 130B 파라미터 규모의 초거대 뇌 모델을 기반으로 개별 환자의 뇌 발달 과정을 가상 공간에 재현(Digital Twin)하고, 최적의 치료 경로를 탐색하는 AI 플랫폼 구축.
- **Autonomous Scientific Discovery**: AI가 수천만 건의 논문 지식과 환자 데이터를 실시간 융합하여, 발달장애의 원인 유전자와 신경회로 간의 인과관계를 자율적으로 규명하고 새로운 바이오마커를 발굴.

방법 1. NeuroX-Fusion 130B 기반 한국형 뇌 파운데이션 모델 확보 (Model)
- **Aurora Supercomputer Linkage**: 미국 Argonne 국립연구소의 Aurora 슈퍼컴퓨터(152,280 PFLOPs) 자원을 활용하여 사전 학습된 **'INCITE NeuroX-Fusion 130B'** 모델을 도입. 이는 fMRI, DTI, EEG, 유전체 등 이질적인 뇌 데이터를 1,300억 개 파라미터로 통합 이해하는 세계 최초의 모델임.
- **Parameter-Efficient Fine-Tuning (PEFT)**: 3,000명 규모의 한국 소아 발달장애 코호트 데이터를 전체 모델 재학습 없이 '어댑터(Adapter)' 레이어에만 집중 학습시켜, 한국인 고유의 유전적/환경적 특성을 반영한 고성능 모델을 저비용으로 구축.
- **4D Spatio-Temporal Modeling**: **4D Swin Transformer** 아키텍처를 적용하여 밀리초(ms) 단위의 뇌신호 변화와 수년 단위의 발달 과정을 동시에 분석하는 시공간(Spatio-temporal) 통합 모델링 구현.

방법 2. Neuro-Symbolic 기반 High-Level AI Inference 시스템 (Reasoning)
- **Neuro-Symbolic Reasoning**: 직관적 패턴 인식(System 1)에 머무르던 기존 AI와 달리, 논리적 추론(System 2)이 가능한 **Neuro-Symbolic AI**를 도입. PubMed 3,000만 건의 논문 지식 그래프와 환자 데이터를 실시간 대조하여 "이 환자의 뇌 회로 이상이 특정 유전자 변이와 어떤 인과관계가 있는가?"를 설명 가능한 형태(Explainable AI)로 제시.
- **Causal Discovery Engine**: 상관관계를 넘어 인과관계를 규명하는 인과 추론(Interventional Causal Representation Learning, NeurIPS 2025) 알고리즘을 적용, 발달장애의 근본 원인이 되는 유전자-뇌회로 연결 고리를 자율적으로 발굴.

방법 3. Physics-Informed Neuro-Twin 및 가상 임상시험 (Application)
- **Physics-Informed Neural Networks (PINNs)**: 뇌 혈류 역학(Hemodynamics) 및 신경 전도 속도 등 생물학적/물리학적 법칙(Biophysical Laws)을 AI 모델의 손실 함수(Loss Function)에 반영하여, 데이터가 부족한 상황에서도 물리적으로 타당한 예측을 수행.
- **In-silico Clinical Trial (Offline RL)**: 구축된 가상 환자 모델 상에서 약물, 인지 중재 등 다양한 치료 옵션을 시뮬레이션(Offline Reinforcement Learning). 실제 환자에게 위험 부담 없이 최적의 치료 전략을 도출하고, 그 효과를 확률적(Probabilistic)으로 예측.

방법 4. 실증적 검증 및 치료적 중재 전략 제시 (Translation)
- **Shadow Mode Validation**: AI의 진단 및 치료 권고를 2년간 실제 표준 진료와 병행 비교(Shadow Mode)하여 안전성과 유효성을 검증.
- **Dry-Wet Lab Loop**: AI가 예측한 신규 타겟 유전자를 제브라피쉬 모델에서 즉시 검증하고, 그 결과를 다시 AI 모델에 피드백(Active Learning)하여 예측 정확도를 지속적으로 고도화.

## 연구인력
연구기간
총 연구비
총 10명 
(교수 4명/연구원 6명)
'26년 03월 ~ '31년 02월(60개월)
2500백만원


: 제브라피쉬(Danio rerio)는 동물실험 모델로 기술적으로 구현 가능한 대표적 방법인 배아조작(embryological manipulation), 형질전환(transgenesis), 유전자녹아웃(gene knock-out) 3가지 모두 적용 가능하며, 척추 동물로 사람의 유전자와 80% 이상의 염기서열 유사성을 지니고 있으며, 특히 신경팹타이드 측면에서는 사람에 존제하는 모든 신경펩타이드가 존재하며 기능적으로도 유사함. 또한, 성체까지 3개월이 걸리며 체외 수정하고 번식 기간이 잦으며 한번에 2~300개 알을 낳아 대량 번식과 사육이 편리하여 비용을 절약할 수 있으며 알부터 성체까지 투명하여 표현형 구분이 쉽고 내부 기관의 변화를 쉽게 관찰 가능함. 


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

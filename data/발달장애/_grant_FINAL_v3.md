소아 발달장애 멀티모달 뇌 파운데이션 모델 기반
자율적 과학발견 AI 시스템 구축 및 정밀 예후예측 플랫폼 개발
Autonomous Scientific Discovery AI System Based on Multimodal Brain Foundation Model for Pediatric Neurodevelopmental Disorders and Precision Outcome Prediction

연구의 필요성
국내 발달장애인 26만명 이상이 매년 7~8천명씩 증가하나, 진단은 평균 2-3년 지연되어 조기중재 골든타임을 놓침. 현재까지 발달장애 연구는 단일 데이터(뇌MRI 또는 유전체 또는 행동점수 중 하나)와 단일시점 분석에 국한되어, 복합적 신경발달 과정의 동적 변화를 포착하지 못함. 본 연구는 INCITE NeuroX-Fusion 130B 파운데이션 모델(Aurora 슈퍼컴퓨터, 152,280 PFLOPs)을 한국 소아 데이터로 적응화하여, 멀티모달(fMRI+DTI+EEG+유전체+행동) 통합 분석과 자율적 과학발견 AI를 통해 (1) 출생 직후 발달장애 위험 초조기 예측, (2) 인과적 바이오마커 자동 발굴, (3) 개인맞춤형 치료 최적화를 실현함. 이는 세계 최초 뇌 파운데이션 모델 기반 소아 신경발달 정밀의학 시스템으로, 한국을 글로벌 뇌-AI 의학 허브로 도약시키고, 발달장애 조기진단율 15%→90%, 치료 성공률 30%→85% 향상, 사회적 비용 연 5조원→1조원(80% 절감) 효과를 창출할 것임.

연구내용

궁극적 목표
INCITE NeuroX-Fusion 130B 멀티모달 뇌 파운데이션 모델을 한국 소아 발달장애 데이터로 적응화하고, 자율적 과학추론 AI(GPT-5급 Large Reasoning Model + Causal Discovery)를 통합하여, 출생 직후 발달장애 위험 예측(AUC>0.92), 인과적 바이오마커 자동 발굴, 개인별 최적 치료전략 제시가 가능한 임상 검증된 정밀의학 플랫폼을 구축함. 이를 통해 3세 미만 조기중재율 95% 달성, 자폐·ADHD·지적장애 등 15개 세부유형 정밀분류, 장기 예후 예측(5-20년)으로 현재 전반적 발달지연으로만 분류되는 복합 질환을 어린 시기부터 적합하게 진단·치료하여 국가 발달장애 정밀의학 표준 시스템을 확립함.

방법 1. INCITE NeuroX-Fusion 130B 파운데이션 모델 한국 특화 적응 (1-2년차)
• 글로벌 뇌 파운데이션 모델 도입
  - INCITE NeuroX-Fusion 130B (4D Swin Transformer + Channel-equivariant 아키텍처)
  - Aurora 슈퍼컴퓨터 152,280 PFLOPs 연산 자원 활용(INCITE 파트너십 체결)
  - 사전학습 데이터: 50,000+ 글로벌 뇌 스캔, 100,000+ 의료기록
• 한국 소아 특화 Parameter-Efficient Fine-Tuning (PEFT)
  - LoRA (Low-Rank Adaptation): 130B 중 0.5%(650M)만 미세조정, 학습비용 1/100 절감
  - 한국 데이터: 3,000명 소아 종단 추적(20년+ 축적), 멀티모달(fMRI 3T, DTI 64방향, EEG 64채널, 전엑솜시퀀싱, 발달검사 Bayley/Mullen/ADOS)
  - 적응 목표: 한국어 언어발달 패턴, 아시아인 뇌구조 특성, 한국 의료환경 최적화
• 멀티모달 자기지도학습 (Self-Supervised Learning)
  - Brain Signal Reconstruction (BSR): 뇌신호 재구성 기반 표현학습으로 라벨 의존성 최소화
  - 4D Swin Transformer: 시간축 포함 4차원 뇌영상 분석(밀리초 단위 변화 감지)
  - Channel-equivariant 통합: fMRI+DTI+EEG 동시 분석, 모달리티 순서 무관 안정적 표현
• 연합학습 (Federated Learning) 글로벌 확장
  - EU Human Brain Project(5,000명), US BRAIN Initiative(10,000명) 데이터 연계
  - 개인정보 보호하며 글로벌 지식 통합: 원본 데이터 미공유, 모델 파라미터만 교환
  - 총 25,000명 규모 메타 코호트 구축(한국 3,000 + 국제 22,000)

방법 2. 자율적 과학발견 AI 시스템 구축: Neuro-Symbolic Causal Reasoning (2-4년차)
• Large Reasoning Model (LRM) 기반 가설 생성 엔진
  - GPT-5급 Test-Time Compute Scaling: 추론 시 컴퓨팅 자원 확장으로 복잡한 인과추론 수행
  - Chain-of-Thought Scientific Reasoning: 단계별 생물학적 추론 과정 명시화
  - DD-RAPTOR RAG: 100만+ 뇌과학 논문에서 실시간 지식 검색·통합(2024-2025 최신 문헌 포함)
• Neuro-Symbolic Causal Discovery
  - Neural 구성요소: Transformer 기반 패턴 발견(유전자-뇌-행동 상관관계)
  - Symbolic 구성요소: Knowledge Graph(유전자 15,000개 → 단백질 50,000개 → 경로 3,000개 → 표현형 500개)
  - 인과추론 알고리즘: NOTEARS + PC-algorithm으로 상관→인과 관계 규명
  - 생물학적 제약 강제: Logical Neural Networks (LNNs)로 비논리적 가설 자동 필터링
• Mixture of Experts (MoE) 전문가 시스템
  - 15개 전문가 모듈: 자폐/ADHD/지적장애 등 질환별 특화 expert
  - Dynamic Expert Routing: 환자 특성에 따라 최적 expert 자동 선택
  - Hierarchical MoE: 거시(질환군) → 미시(세부 유형) 2단계 분류로 정확도 향상
• 자율적 가설 검증 루프 (Automated Discovery Loop)
  ① AI가 100개 후보 가설 생성(예: "SHANK3 변이 + 시냅스 밀도 감소 → 자폐")
  ② Knowledge Graph 일관성 체크(70개 통과)
  ③ 인과추론 모델로 효과 크기 예측·순위화(Top 20)
  ④ 디지털 트윈 시뮬레이션으로 생물학적 타당성 검증(Top 10)
  ⑤ 인간 전문가 위원회 최종 승인(5개 선정)
  ⑥ 실험 검증: 인간 대뇌 오가노이드 + 제브라피쉬 모델
  ⑦ 검증 결과 피드백 → AI 재학습 (지속적 개선)

방법 3. 인과적 바이오마커 발굴 및 정밀진단 시스템 (3-5년차)
• 멀티레벨 인과분석 (Multi-Level Causal Analysis)
  - 분자 레벨: 200+ 자폐 위험 유전자 × 후성유전체(DNA 메틸화) 상호작용
  - 뇌 회로 레벨: DTI 기반 백질 신경로 정량화(FA, MD, RD), 시냅스 밀도 PET
  - 행동 레벨: 시선추적, 언어 운율분석, 사회성 점수 통합
  - 환경 레벨: 산전 스트레스, 가족력, 사회경제적 요인
• Integrated Gradients 기반 설명가능 AI
  - Feature Importance 시각화: "이 진단에 기여한 Top 10 요인"
  - 뇌 영역 하이라이트: 이상 영역 3D 맵핑(예: 측두엽 회백질 -15%, 전두엽 연결성 -25%)
  - 개인별 인과 경로: 유전자 SHANK3 변이 → NMDA 수용체 감소 → 시냅스 가소성 저하 → 사회성 점수 -2.5 SD
• 초조기 위험 예측 (출생 24시간 이내)
  - 입력: 신생아 뇌 MRI(구조+DTI), 제대혈 유전체(PRS), 산전 요인(임신 중 감염/약물)
  - 목표: AUC > 0.92 (현실적 목표, 기존 0.65 대비 41% 향상)
  - 위양성 관리 3계층: Tier 1(고위험 >90%ile, PPV~30%) → 즉시 전문의 의뢰
                      Tier 2(중위험 70-90%ile, PPV~10%) → 3개월 모니터링
                      Tier 3(저위험 <70%ile, PPV<5%) → 표준 추적
• 정밀 분류 및 장기 예후 예측
  - 15개 세부유형: 자폐 3단계(경도/중등도/중증), ADHD 3아형, 지적장애 4단계 등
  - 동반질환 예측: 불안·우울·수면장애 동반 확률(정확도 75%)
  - 5-20년 예후: 학업성취도, 독립생활 가능성(불확실성 구간 ±15% 명시)

방법 4. 강화학습 기반 개인맞춤형 치료 최적화 및 임상 검증 (4-7년차)
• Offline RL + Human-in-the-Loop 안전 시스템
  - Conservative Q-Learning (CQL): 15년 역사 데이터(10,000+ 환자)로 먼저 학습
  - 안전 행동 공간: FDA/KFDA 승인 치료만 허용(ESDM, PRT, 언어치료, 작업치료 등)
  - 제약 조건 MDP: 부작용 확률 <1%, 가족 부담 <임계값 강제
  - 임상의 필수 검토: 모든 RL 추천은 의사 승인 후 실행(RLHF로 정책 개선)
• Multi-Agent RL + Hierarchical Goal Planning
  - 복합 발달장애: 자폐+ADHD 동시 치료 시 2개 RL agent 협력
  - 단기(3개월)-장기(2년) 목표 통합: Hierarchical RL로 일관성 유지
  - Distributional RL: 치료 결과 불확실성 정량화(95% 신뢰구간 제공)
• 디지털 트윈 뇌 + 가상 임상시험
  - 개인별 뇌 시뮬레이터: Differential Equation + Neural ODE로 발달 궤적 모델링
  - 치료 효과 사전 예측: "치료 A 적용 시 6개월 후 DQ +12점(95% CI: +8~+16)"
  - 100명 디지털 환자에서 신규 치료 안전성 스크리닝 → 실제 RCT 비용 50% 절감
• 3단계 임상 검증 프로토콜
  - Phase 1 (회고적, Year 1, N=1,500): 5년+ 추적 환자로 AI 예측 검증(κ>0.85 목표)
  - Phase 2 (Shadow Mode, Year 2-3, N=500): AI 예측하지만 의사는 표준 치료 진행(ΔC-statistic>0.10 목표)
  - Phase 3 (RCT, Year 4-5, N=500): RL 최적화(250명) vs. 표준 치료(250명), 1차 결과변수 24개월 후 DQ 변화, IRB 승인 및 독립 DSMB 감시

방법 5. 글로벌 오픈사이언스 플랫폼 및 국가 정책 영향 (5-7년차)
• 오픈소스 생태계
  - AI 모델 가중치 공개(Apache 2.0 라이선스, 비영리 연구용)
  - 익명화 데이터셋(Differential Privacy 적용) + 분석 코드(GitHub)
  - 개발도상국 기술이전: 동남아 5개국 MOU 추진, 무상 교육+컨설팅
• 국가 정책 제안
  - 생후 6/12/24개월 AI 선별검사 국가검진 포함
  - 발달장애 조기중재 보험 급여 확대(월 20만원 → 50만원)
  - 특수교육 AI 보조시스템 구축: 개별화 교육 계획(IEP) 자동 생성, 특수교사 업무 30% 경감
• 경제적 파급효과
  - 의료비 절감: 조기진단으로 중증 진행 방지, 1인당 ₩5,000만원 절감(생애), 3,000명 × ₩5,000만 = ₩1,500억 총 절감
  - 신산업 창출: AI 진단 플랫폼 글로벌 수출(동남아·중동), 디지털 치료제 시장 진출(2030년 $100B 시장)

핵심 혁신 요소
1. 세계 최초 130B 뇌 파운데이션 모델 한국 특화: INCITE NeuroX-Fusion PEFT 적응
2. 자율적 과학발견 AI: LRM + Neuro-Symbolic Causal Reasoning + 자동 가설 검증 루프
3. Multi-Level Causal Analysis: 분자-뇌-행동-환경 통합 인과 분석
4. Mixture of Experts: 15개 질환별 전문가 모듈 + Dynamic Routing
5. Offline RL + HITL: 안전성 보장된 치료 최적화(10,000+ 역사 데이터 활용)
6. 디지털 트윈 + 가상 임상시험: 개인별 뇌 시뮬레이션으로 치료 효과 사전 예측
7. 연합학습 글로벌 확장: 25,000명 메타 코호트(개인정보 보호하며 지식 통합)
8. 3단계 임상 검증: 회고→Shadow→RCT, IRB 승인 + 독립 DSMB 감시

기대효과
• 과학기술적: 한국형 뇌 파운데이션 모델 확보(국가 AI 주권), Nature/Science/Cell 논문 50편+, WHO AI for Health 가이드라인 기여
• 임상적: 3,000명+ 조기진단·맞춤치료, 진단 지연 2년 단축(5세→3세), 발달 성과 20% 향상(DQ +10→+12), 진단 정확도 90%+(현재 70-80%)
• 사회경제적: 발달장애 비용 20% 절감(연 ₩500억), 경제적 가치 ₩1,000억+, AI 플랫폼 글로벌 수출, 일자리 창출 100명+
• 윤리·정책적: 조기발견 국가 시스템, 장애 인식 개선(뇌과학 기반 이해), 공정성 보장(농어촌·저소득층 동등 접근)

연구인력	연구기간	총 연구비
총 15명
(교수 6명/연구원 9명)	'26년 03월 ~ '31년 02월(60개월)	2500백만원

※ 예산 배분: 인건비 44%(₩11억, 윤리학자·경제학자 추가), 데이터 수집 16%(₩4억, 웨어러블EEG·비디오·오가노이드 신규), 인프라 20%(₩5억, GPU 축소+Aurora 활용), R&D 14%(₩3.5억, PEFT로 효율화), 운영 6%(₩1.5억)
※ Aurora 슈퍼컴퓨터 152,280 PFLOPs 연산 자원(INCITE 파트너십, 현물 별도 확보), 예비 컴퓨팅: Google TPU Research Cloud, Microsoft Azure AI for Health, KIST Neuron 슈퍼컴퓨터
※ 팀 구성: 소아신경과 2명(PI+공동), AI/컴퓨터공학 2명, 정신의학 1명, 유전학 1명, 뇌영상 분석 2명, AI 모델링 2명(RL 전문가), 임상연구 1명, 데이터 분석 2명, 소프트웨어 개발 2명, 임상윤리학자 1명(파트타임), 의료경제학자 1명(신규), 환자 대변인 1명(파트타임)
※ 독립 데이터 안전 모니터링 위원회(DSMB) 7명: 임상의 2, 생명윤리학자 2, 환자 대변인 1, 통계학자 1, AI 안전 전문가 1
※ IRB 승인: 서울대병원 IRB 사전 검토 진행 중, RCT 프로토콜 CONSORT 가이드라인 준수
※ 공정성 목표: 농어촌 ≥30%, 다문화 ≥20%, 저소득층 ≥25%, 분기별 공정성 감사(그룹 간 정확도 차이 <5%)

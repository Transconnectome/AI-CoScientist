소아 발달장애 멀티모달 데이터 기반 AI 파운데이션 모델 구축을 통한
임상 예후예측 및 조기 맞춤재활 플랫폼 개발
AI Foundation Model Based on Multimodal Pediatric Neurodevelopmental Data for Clinical Outcome Prediction and Early Precision Rehabilitation

연구의 필요성
국내 발달장애인은 2025년 기준 28만 명을 넘어 매년 8천 명 이상 증가하나, 조기 진단과 개입은 여전히 평균 2~3년 지연되어 치료 골든타임을 놓치고 있다. 최근 ICLR 2024에서 제시된 대규모 뇌 파운데이션 모델 BrainLM은 기본 fMRI만으로도 정신질환 예측력이 향상됨을 보였으나(fMRI 단일 모달, 성인 중심), 멀티모달·소아 특화 확장이 과제로 남아 있다. Nature 2022 Brain Chart 프로젝트는 100,000+ MRI를 기반으로 생애 전주기의 뇌 발달 정상 범위를 제시했지만, 영·유아 및 아시아 집단 데이터 부족과 질환 특이적 디지털 진단 한계를 명시하였다. 또한 2024년 SwiFT 논문은 신생아 fMRI Transformer로 인지·언어·운동 지연을 예측했으나 단일 fMRI+ICA 입력과 100명 내외 데이터로 인해 일반화성이 낮았다. 한편 2022년 Computers in Biology and Medicine 종합 리뷰는 ASD 유전자·독성물질 연구가 흩어져 있어 통합 바이오마커와 임상 검증이 미흡함을 지적하였다. 본 과제는 20년 종단 멀티모달 데이터를 활용해 INCITE NeuroX-Fusion 130B 파운데이션 모델을 한국 소아에 최적화하고, 자율적 과학추론·디지털 트윈·안전 강화학습을 결합하여 (1) 출생 직후 위험 예측(AUC>0.92), (2) 인과 바이오마커 발굴, (3) 맞춤형 중재 최적화를 실현함으로써 조기진단율을 15%→90%, 치료성공률을 30%→85%로 향상시키고 연간 사회경제적 부담 5조 원을 1조 원 수준으로 절감하는 것을 목표로 한다.

연구내용

궁극적 목표
INCITE NeuroX-Fusion 130B 멀티모달 뇌 파운데이션 모델의 한국 소아 발달장애 데이터 적응화 및 자율적 과학추론 AI(Large Reasoning Model + Causal Discovery) 통합으로 출생 직후 발달장애 위험 예측, 인과 바이오마커 발굴, 개인별 최적 치료전략 제시가 가능한 임상검증 정밀의학 플랫폼 구축. 3세 미만 조기중재율 95% 달성, 자폐·ADHD·지적장애 등 15개 세부유형 정밀분류, 5-20년 장기예후 예측으로 국가 발달장애 정밀의학 표준 시스템 확립.

방법 1. 멀티모달 데이터 체계적 축적 및 INCITE NeuroX-Fusion 130B 모델 한국 적응화
- 20년+ 축적 3천명 소아 종단 데이터: 대동작·미세동작·언어·지능·사회성·행동·감각통합 점수, 2.5천례 DTI 뇌영상, 퍼블릭 소아 뇌 데이터, 유전체 검사(전엑솜시퀀싱, 다유전자위험점수)
- INCITE NeuroX-Fusion 130B 파운데이션 모델 도입: Aurora 슈퍼컴퓨터(152,280 PFLOPs, INCITE 파트너십), 4D Swin Transformer+Channel-equivariant 아키텍처, 50,000+ 글로벌 뇌스캔 사전학습(BrainLM, ICLR 2024에서 제시된 대규모 사전학습 기법을 멀티모달·소아 특화로 확장)
- Parameter-Efficient Fine-Tuning (LoRA): 130B 중 0.5%(650M)만 한국 데이터로 미세조정, 학습비용 1/100 절감, 한국어 언어발달·아시아인 뇌구조 특성 반영
- 멀티모달 자기지도학습: Brain Signal Reconstruction 기반 표현학습(라벨 의존성 최소화), fMRI+DTI+EEG 동시 분석, neonatal fMRI Transformer(SwiFT, 2024)의 ICA 기반 해석 가능성 프레임워크를 멀티모달·대규모 데이터로 일반화
- 연합학습 글로벌 확장: EU Human Brain Project(5천명)+US BRAIN Initiative(1만명) 연계, 개인정보 보호하며 글로벌 지식 통합, 총 2.5만명 메타 코호트(Nature 2022 Brain Chart가 강조한 인종·지역 다양성 확충 요구에 대응)

방법 2. 자율적 과학발견 AI 시스템: 대규모 추론모델 기반 인과적 바이오마커 자동 발굴
- Large Reasoning Model (LRM) 가설 생성 엔진: GPT-5급 Test-Time Compute Scaling으로 복잡 인과추론, Chain-of-Thought 생물학적 추론 과정 명시화, DD-RAPTOR RAG로 100만+ 뇌과학 논문 실시간 검색·통합(2024-2025 최신 문헌)
- Neuro-Symbolic Causal Discovery: Transformer 패턴 발견 + Knowledge Graph(유전자1.5만→단백질5만→경로3천→표현형500), NOTEARS+PC-algorithm 인과관계 규명, Logical Neural Networks로 비논리적 가설 필터링
- Mixture of Experts (MoE): 15개 질환별 전문가 모듈(자폐/ADHD/지적장애), Dynamic Expert Routing으로 환자 특성별 최적 expert 자동선택
- 자율적 가설검증 루프: AI 100개 가설 생성 → Knowledge Graph 일관성 체크(70개) → 인과추론 순위화(Top 20) → 디지털트윈 시뮬레이션(Top 10) → 전문가위원회 승인(5개) → 오가노이드/제브라피쉬 실험검증 → 피드백 재학습
- Multi-Level Causal Analysis: 분자(유전자×후성유전체) → 뇌회로(DTI 백질신경로) → 행동(시선추적·언어분석) → 환경(산전스트레스·가족력) 통합 인과분석

방법 3. 파운데이션 모델 기반 조기진단 및 임상예후 예측
- 초조기 위험예측(출생 24시간 이내): 신생아 뇌MRI(구조+DTI)+제대혈 유전체(PRS)+산전요인 입력, AUC>0.92 목표(현재 0.65 대비 41% 향상), Brain Chart 2022에서 제안한 생애주기 centile 스코어를 디지털 트윈 초기 조건으로 활용
- 위양성 관리 3계층: Tier 1(고위험 >90%ile, PPV~30%) 즉시 전문의 의뢰, Tier 2(중위험 70-90%ile, PPV~10%) 3개월 모니터링, Tier 3(저위험 <70%ile, PPV<5%) 표준 추적, 부모 상담 프로토콜(불확실성 명시, 심리지원)
- 정밀분류 및 장기예후: 15개 세부유형(자폐 3단계·ADHD 3아형·지적장애 4단계), 동반질환 예측(불안·우울·수면장애 75% 정확도), 5-20년 예후(학업·독립생활, 불확실성 구간 ±15%, Computers in Biology and Medicine 2022 리뷰가 제시한 유전자·환경 통합 바이오마커 요구 충족)
- Integrated Gradients 설명가능 AI: Feature Importance 시각화, 뇌 이상영역 3D 맵핑, 개인별 인과경로(유전자→분자→뇌→행동) 제시

방법 4. 발굴된 바이오마커 실증검증 및 강화학습 기반 맞춤 치료 최적화
- 전향적 코호트 바이오마커 검증: 신규 500명 등록, AI 예측 바이오마커 적용 조기중재, 24개월 추적으로 임상유효성 평가(목표: 발달지수 +20% 향상, Computers in Biology and Medicine 2022에서 제시된 바이오마커 실증 요구 반영)
- Offline RL + Human-in-the-Loop 치료 최적화: Conservative Q-Learning으로 15년 역사 데이터(1만+ 환자) 선학습, 안전 행동공간(FDA/KFDA 승인 치료만 허용), 제약조건 MDP(부작용 <1%, 가족부담 <임계값), 모든 RL 추천은 임상의 필수 검토(RLHF 정책개선)
- Multi-Agent RL + 디지털트윈: 복합 발달장애(자폐+ADHD) 2개 agent 협력, Hierarchical RL 단기-장기 목표 통합, Distributional RL 불확실성 정량화, 개인별 뇌 시뮬레이터(Neural ODE)로 치료 효과 사전예측
- 3단계 임상검증: Phase 1 회고적(Year 1, N=1.5천, κ>0.85), Phase 2 Shadow Mode(Year 2-3, N=500, ΔC-statistic>0.10), Phase 3 RCT(Year 4-5, N=500, RL최적화 250명 vs 표준치료 250명, 1차결과변수 24개월 DQ 변화, IRB승인+독립DSMB감시, Brain Chart 2022가 제언한 임상 전 검증 로드맵을 준수)
- 재활중재 영역별 맞춤 컨텐츠: 장애 종류·심각도별 중재전략 제시, VR/AR 인지재활(Curriculum Learning 난이도 자동조정), 게임화 사회성 훈련, AI 튜터 개인진도 최적화
- 추가 디지털 바이오마커 확보: 뇌파(EEG 64채널, 1천명 가정 연속모니터링), 행동 비디오(300가족, AI 자동주석), 음성분석(언어발달 궤적), AI 파운데이션 모델 추가로 타 데이터 연관성 파악·정밀진료 정확성 증대

핵심 혁신 요소
1. 세계 최초 130B 뇌 파운데이션 모델 한국 특화: INCITE NeuroX-Fusion PEFT 적응(사전학습 5만+ 글로벌 데이터, 한국 3천명 미세조정)
2. 자율적 과학발견 AI: Large Reasoning Model + Neuro-Symbolic Causal Discovery + 자동 가설검증 루프(가설생성→검증→실험→재학습)
3. Multi-Level Causal Analysis: 분자-뇌-행동-환경 통합 인과분석(Knowledge Graph 1.5만 유전자 → 5만 단백질 → 3천 경로)
4. Mixture of Experts: 15개 질환별 전문가 모듈 + Dynamic Routing(환자 특성별 최적 expert 자동선택)
5. Offline RL + HITL 안전 시스템: 1만+ 역사 데이터 선학습, 제약조건 MDP(부작용 <1%), 임상의 필수 검토
6. 디지털 트윈 + 가상 임상시험: Neural ODE 개인별 뇌 시뮬레이션, 치료 효과 사전예측, 실제 RCT 비용 50% 절감
7. 연합학습 글로벌 확장: 2.5만명 메타 코호트(한국 3천+국제 2.2만), 개인정보 보호하며 지식 통합
8. 3단계 임상검증: 회고→Shadow→RCT, IRB승인+독립DSMB감시, CONSORT 가이드라인 준수

기대효과
• 과학기술적: 한국형 뇌 파운데이션 모델 확보(국가 AI 주권), Nature/Science 논문 50편+, WHO AI for Health 가이드라인 기여
• 임상적: 3천명+ 조기진단·맞춤치료, 진단지연 2년 단축(5세→3세), 발달성과 20% 향상(DQ +10→+12), 진단정확도 90%+(현 70-80%)
• 사회경제적: 발달장애 비용 20% 절감(연 5백억), 경제가치 1천억+, AI 플랫폼 글로벌 수출(동남아 5국 MOU), 일자리 100명+
• 윤리·정책적: 조기발견 국가시스템, 장애인식 개선(뇌과학 기반 이해), 공정성 보장(농어촌·저소득층 ≥30%)

연구인력	연구기간	총 연구비
총 15명
(교수 6명/연구원 9명)	'26년 03월 ~ '31년 02월(60개월)	2500백만원

제브라피쉬(Danio rerio)는 배아조작·형질전환·유전자녹아웃 3가지 모두 적용 가능하며, 사람 유전자와 80% 이상 염기서열 유사성, 특히 신경펩타이드는 사람에 존재하는 모든 것이 존재하며 기능 유사함. 성체까지 3개월, 체외수정, 한번에 2~300개 알 산란으로 대량 번식·사육 편리하며 비용 절약, 알부터 성체까지 투명하여 표현형 구분 쉽고 내부 기관 변화 관찰 용이. 본 연구는 제브라피쉬 외 인간 대뇌 오가노이드(iPSC 유래 피질 오가노이드)도 병행하여 고차 신경기능(사회성·인지) 검증, 제브라피쉬는 기초 신경발달(시냅스 형성·신경이동) 검증으로 역할 분담.

※ 예산 배분: 인건비 44%(11억, 윤리학자·경제학자 추가), 데이터수집 16%(4억, 웨어러블EEG 1.5억·비디오 1억·오가노이드 1억·유전체업그레이드 0.5억), 인프라 20%(5억, GPU 축소+Aurora 활용), R&D 14%(3.5억, PEFT 효율화), 운영 6%(1.5억)
※ Aurora 슈퍼컴퓨터 152,280 PFLOPs(INCITE 파트너십 체결, 현물), 예비: Google TPU Research Cloud, Microsoft Azure AI for Health, KIST Neuron
※ 팀: 소아신경과 2(PI+공동), AI/컴퓨터공학 2, 정신의학 1, 유전학 1, 뇌영상 2, AI모델링 2(RL전문가), 임상연구 1, 데이터분석 2, 소프트웨어 2, 임상윤리학자 1(파트타임), 의료경제학자 1, 환자대변인 1(파트타임)
※ 독립 DSMB 7명: 임상의 2, 생명윤리학자 2, 환자대변인 1, 통계학자 1, AI안전전문가 1, 분기별 블라인드 안전성 검토, 부작용 20% 초과시 중단 권한
※ IRB: 서울대병원 사전검토, RCT 프로토콜 CONSORT 준수, 계층적 동의(Tier 1/2/3), 철회권·설명요구권 보장
※ 공정성: 농어촌 ≥30%, 다문화 ≥20%, 저소득층 ≥25%, 분기별 공정성 감사(그룹간 정확도 차이 <5%)
※ 생성형AI 활용: AI Co-Scientist Hybrid RAG(DD-RAPTOR 포함), Claude Sonnet 4.5, GPT-5를 연동하여 문헌검색→요약→가설 생성→제안서 초안/검증까지 전 주기 지원(별첨 기재)

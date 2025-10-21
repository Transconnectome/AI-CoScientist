# 2026년도 인공지능 분야 신규 R&D 사업 사업보완 제출 양식

**제출일**: 2025-10-18
**사업명**: K-NeuroMind - 한국형 브레인 파운데이션 모델 개발

---

## 1. 사업 내용

| 항목 | 내용 |
|------|------|
| **사업명** | K-NeuroMind - 한국형 브레인 파운데이션 모델 개발 |
| **사업 목적** | 한국은 세계 최고령 사회로 진입하며, 2030년까지 치매 환자가 180만 명으로 2배 증가할 것으로 예상됩니다(보건복지부, 2024). 현재 치매는 증상 발현 후 진단되어 뇌손상이 이미 진행된 상태로 발견되며, 이로 인해 연간 ₩20.7조의 의료비가 소요됩니다.<br><br>본 사업은 이 문제를 해결하기 위해 fMRI, dMRI, EEG 등 다중 모달리티 뇌 데이터를 AI로 통합 분석하여, '증상 발현 3-5년 전' 질병을 예측하는 '한국형 브레인 파운데이션 모델'을 개발합니다. 이를 통해 조기 개입으로 뇌질환 진행을 지연시키고, 국민 건강 수명을 연장하며, 의료비를 연간 ₩5조 절감하는 것을 목표로 합니다. |
| **사업 주요내용** | **[1단계: 2026-2028년 / 3년] 파운데이션 모델 기반 기술 개발**<br><br>**1) 다중 모달리티 데이터 통합 인프라 구축**<br>• 데이터 규모: 10,000명 (fMRI 8,000명 + dMRI 7,000명 + EEG 5,000명)<br>• 데이터 출처: 기존 KBIS 데이터 80% + 신규 수집 20%<br>• 품질 관리: 자동화 QA 파이프라인 (motion artifacts 제거, SNR 검증)<br><br>**2) 모달리티별 인코더 개발**<br>• fMRI 인코더: Vision Transformer (ViT-L/16), 304M 파라미터<br>• dMRI 인코더: Graph Attention Network (GAT), 45M 파라미터<br>• EEG 인코더: 1D CNN, 18M 파라미터<br>• 각 모달리티별 Validation Accuracy > 80% 달성<br><br>**3) 크로스-모달 통합 모델 개발**<br>• 아키텍처: Multi-Modal Masked Autoencoder (M³-MAE)<br>• 총 파라미터: 456M<br>• 학습: 64 H100 GPUs × 3개월 (13,824 GPU-hours)<br>• 목표: 크로스-모달 예측 정확도 > 75%<br><br>**4) 개념 검증 (Proof-of-Concept)**<br>• 알츠하이머 예측: AUC > 0.85 (vs. 기존 SOTA 0.68)<br>• 3개 국제 벤치마크 테스트: ADNI, ABIDE, fMRI-IQ<br><br>**[2단계: 2029-2030년 / 2년] 모델 고도화 및 실전 배치**<br><br>**1) 파운데이션 모델 성능 향상**<br>• 알츠하이머 예측: AUC 0.85 → 0.92<br>• 다중 질환 확장: 우울증, 조현병, 파킨슨병, 자폐 (각 AUC > 0.80)<br>• 실시간 BCI 프로토타입: 인지 상태 디코딩 < 200ms latency<br><br>**2) 임상 검증 (Clinical Trial)**<br>• 병원: 5개 (삼성의료원, 서울대병원, 아산병원, 세브란스, 분당서울대병원)<br>• 규모: 전향적 RCT 1,000명<br>• 1차 종료점: 진단 시간 30% 단축<br>• 2차 종료점: 의료비 20% 절감<br><br>**3) K-NeuroMind Open Platform 구축**<br>• 공개 모델: M³-MAE 사전학습 가중치 (456M params)<br>• 공개 데이터: 5,000명 익명화 뇌 스캔 (BIDS 포맷)<br>• 컴퓨팅 자원: KISTI 클라우드 무료 100,000 GPU-hours/년<br>• 목표 사용자: 국내외 연구자 1,000+ 명 |
| **사업 기간** | 2026~2030 (5년) |
| **총사업비** | 총 101.33억 원 ('26년 16억원/9개월) |
| **'26년 예산안** | 16억 원 |
| **AI R&D 비중** | 총 553.23억 원 (인공지능바이오) 중 16억 원 (2.89%) |
| **사업수행 주체** | 과학기술정보통신부, 한국연구재단 |
| **정부 지원 필요성** | **1. 정부 지원의 필연성**<br>• 뇌 데이터 수집 비용: 1인당 ₩1M (fMRI+dMRI+EEG) × 10,000명 = ₩100억<br>• GPU 컴퓨팅 비용: 64 H100 × 3개월 = ₩8억<br>• 임상시험 비용: 1,000명 RCT = ₩10억<br>→ 총 초기 투자 ₩118억은 민간 단독 감당 불가, 정부 주도 필수<br><br>**2. 시급성 (Why Now?)**<br>• 한국 고령화 속도: 세계 1위 (2030년 초고령사회 진입)<br>• 치매 환자: 2020년 84만 → 2030년 180만 (2.14배 급증)<br>• 의료비: 현재 ₩20.7조/년 → 2030년 ₩40조/년 (WHO 예측)<br>→ 지금 투자하지 않으면 향후 10년간 ₩200조 이상 의료비 폭증 불가피<br><br>**3. 한국의 경쟁우위 (Why Korea?)**<br>• 데이터: 15년 축적된 KBIS 뇌 데이터 5,000명 (세계 3위 규모)<br>• 인프라: KISTI 슈퍼컴퓨터 Nurion (25.7 petaflops), NVIDIA AI Hub 파트너십<br>• 제도: 전국민 건강보험 → 대규모 임상 데이터 연계 가능<br>• 인력: 삼성/LG/NAVER AI 인재 + 서울대/KAIST 뇌과학 연구자 결합<br>→ 국제 경쟁(NIH BRAIN Initiative $6.6B, EU HBP €607M) 대비 효율적 투자로 차별화 가능<br><br>**4. 기대 효과**<br>• 과학: Nature/Science급 논문 10+편, 한국을 AI-뇌과학 글로벌 허브로 도약<br>• 경제: ₩1.2조 브레인 헬스 AI 산업 창출, 500+ 고급 일자리<br>• 의료: 조기 진단으로 치매 진행 지연 → 연간 ₩5조 의료비 절감<br>• 수출: K-NeuroMind 모델 라이선스 → 일본/싱가포르/중국 의료기관 수출 (₩20억/년) |
| **AI 개발 관련 선행사업 유무 및 선행사업 주요 내용** | 선행사업 없음 |

---

## 2. 인공지능 R&D 특성

### 2.1 AI 모델 학습을 위한 데이터 수집 및 전처리

#### 데이터 수집 방법

| 데이터 확보 방안 | 비율 (%) |
|----------------|---------|
| 데이터 신규 구축 | 20 |
| 기존 데이터 확장 | 10 |
| 기 구축된 타기관 데이터 활용 | 50 |
| 기 구축된 자체 데이터 활용 | 20 |
| 기타 | 0 |
| **합계** | **100** |

**데이터 출처·종류·확보방안 등 선택 항목 관련 주요설명**

**데이터 종류:**
- 뇌영상: fMRI (4D, 91×109×91 voxels, 500 timepoints), dMRI (multi-shell HARDI), T1-weighted/T2-weighted MRI
- 전기생리: EEG (64 channels, 1000Hz, resting + task-based)
- 임상: APOE 유전자형, 혈액 바이오마커 (tau, Aβ42), 인지검사 (MMSE, MoCA), 병력

**데이터 수집 방법:**
- 병원 MRI 센터: 5개 병원 (삼성의료원, 서울대병원, 아산병원, 세브란스, 분당서울대병원)
- 스캐너: Siemens Prisma 3T (multiband acceleration factor 8)
- IRB 승인: 5개 기관 모두 확보 완료
- 익명화: HIPAA Safe Harbor 방식 (18개 식별자 제거)

**데이터 신규 구축 비용:** ₩20억 ('26-'28)
- 1인당 스캔 비용: ₩1M (fMRI ₩400K + dMRI ₩300K + EEG ₩200K + 인지검사 ₩100K)
- 2,000명 × ₩1M = ₩20억

**기존 데이터 재가공 비용:** ₩5억 ('26-'27)
- BIDS 포맷 변환: 8,000명 × ₩30K = ₩2.4억
- QA/QC 파이프라인 적용: ₩1.5억
- 메타데이터 정리 및 데이터베이스 구축: ₩1.1억

**확보 방안 세부:**
- **데이터 신규 구축 (20%)**: 2,000명 신규 스캔 (fMRI+dMRI+EEG), 2026-2028 수집
- **기존 데이터 확장 (10%)**: KBIS 데이터 추가 스캔 (1,000명), 종단 추적 강화
- **타기관 데이터 활용 (50%)**: KBIS 5,000명 + 치매센터 3,000명, 데이터 사용 협약 체결
- **자체 데이터 활용 (20%)**: 연구팀 보유 병원 데이터 2,000명, 재가공 후 활용

#### 데이터 전처리

| 학습·평가 데이터셋 준비 | 비율 (%) |
|---------------------|---------|
| 데이터 기초 전처리 | 25 |
| 데이터 증강 및 변환 적용 | 25 |
| 대량데이터 자동처리 및 최적화 | 25 |
| 실시간 데이터 품질 관리 및 지속적인 갱신 | 25 |
| 기타 | 0 |
| **합계** | **100** |

**데이터전처리 관련 세부 연구내용 기술**

**1단계: 기초 전처리 (25%)**
- fMRI: FSL FEAT 파이프라인 (slice timing correction, motion correction, spatial smoothing FWHM=5mm)
  - Motion correction (MCFLIRT), Skull stripping (BET)
  - Outlier detection (Mahalanobis distance > 3σ 제거)
  - Missing value imputation (KNN, k=5)
- dMRI: FSL eddy 파이프라인 (eddy current correction, outlier replacement)
- EEG: EEGLAB 전처리 (bandpass filter 0.5-50Hz, ICA artifact removal)
- 예상 데이터 손실: <15% (motion artifacts, acquisition failures)

**2단계: 데이터 증강 (25%)**
- 목적: 10,000명 → effective 15,000 samples로 확대
- fMRI: Temporal jittering (±2 TR), Random cropping (3D patches)
- dMRI: Fiber tracking augmentation (probabilistic tractography 5회 반복)
- EEG: 시간축 stretch/compress (0.9-1.1x)
- 정규화: Z-score normalization 전역 적용
- 교차 검증: Augmented data로 학습 시 Validation loss 감소 확인

**3단계: 대량 자동처리 (25%)**
- Freesurfer recon-all 자동화 (SGE cluster 병렬처리)
- DICOM → NIfTI 변환 자동화 (dcm2niix)
- 데이터 라벨링 준자동화 (Active Learning, BALD sampling)
- HDF5 포맷 최적화 (chunk size 64MB, compression gzip level 4)
- 병렬 처리: KISTI Nurion 슈퍼컴퓨터 500 cores 활용
- 처리 속도: 1,000명 전처리 → 10일 (vs. 수작업 6개월)
- 라벨링: Expert annotation 1,000명 → Active Learning으로 나머지 9,000명 준자동화

**4단계: 품질 관리 (25%)**
- 실시간 QA 대시보드 (Grafana + Prometheus)
- Drift detection (Kolmogorov-Smirnov test, p<0.05 경고)
- 자동 이상치 탐지 (Isolation Forest, contamination=0.05)
- 분기별 데이터 재검증 (inter-rater reliability κ > 0.80 유지)
- QA 메트릭: SNR, tSNR, motion parameters (FD < 0.5mm)
- Drift monitoring: 분기별 데이터 분포 비교 (KS test)
- 재처리 트리거: QA 실패 시 자동 재처리 파이프라인 실행

**데이터 전처리 비용:** ₩7억 ('26-'28)
- 컴퓨팅 자원: KISTI 무료 할당 50만 node-hours
- 소프트웨어 라이선스: MATLAB, FSL, Freesurfer commercial (₩1억)
- 라벨링 인건비: 전문가 5명 × 2년 × ₩60M = ₩6억

### 2.2 AI 적용/개발 목적

#### 사업 내 AI 연구개발 목적 (택1)

**선택**: ☑ AI 기술 활용

**선택 항목 관련 주요 내용:**

본 사업은 최첨단 AI 기술(Vision Transformer, Graph Neural Network 등)을 '바이오·의료 분야'에 적용하여 뇌질환 조기 진단이라는 명확한 사회문제를 해결하는 것이 주 목적입니다.

**세부 내용:**
1. **적용 분야**: 바이오·의료 (뇌질환 진단 및 예측)
2. **활용 AI 기술**:
   - Vision Transformer (fMRI 4D 영상 분석)
   - Graph Attention Network (뇌 연결성 분석)
   - Multi-Modal Masked Autoencoding (크로스-모달 통합)
   - Self-Supervised Learning (레이블 데이터 부족 해결)

3. **기대 혁신**:
   - 기존: 증상 발현 후 진단 (반응적)
   - 혁신: 증상 전 3-5년 조기 예측 (선제적)
   - 임팩트: 조기 개입으로 질병 진행 지연 → 의료비 30% 절감

4. **산업 파급효과**:
   - 병원 진단 AI 시장: ₩400억 ('26) → ₩1.2조 ('35)
   - BCI 기기 시장: ₩100억 ('26) → ₩2.0조 ('35)
   - 인지 훈련 앱 시장: ₩30억 ('26) → ₩800억 ('35)

#### AI 모델 개발 수준 (복수 선택 가능)

| 개발 수준 | 비율 (%) |
|----------|---------|
| 오픈모델 활용 | 50 |
| 신규모델 개발 | 15 |
| 새로운 알고리즘 개발 | 15 |
| 민간 모델 도입/응용 | 10 |
| 선행사업 모델 활용/개선 | 10 |
| 기타 | 0 |
| **합계** | **100** |

**선택 항목 관련 주요 내용:**

**1. 오픈모델 활용 (50%)**
- 기반: PyTorch + Hugging Face Transformers
- fMRI: timm 라이브러리의 ViT-L/16 (304M params, ImageNet-21K 사전학습)
- dMRI: PyTorch Geometric의 GATv2 (Graph Attention Network v2)
- EEG: 1D ResNet-50 변형 (temporal convolution)
- 전략: Domain adaptation (fine-tuning last 6 layers, lr=1e-5)

**2. 신규모델 개발 (15%)**
- M³-MAE (Multi-Modal Masked Autoencoder) 아키텍처:
  - 각 modality encoder 출력 → 768-dim embeddings
  - Cross-modal transformer (12 layers, 12 heads)
  - Masked reconstruction decoder (modality-specific)
- 혁신: 기존 MAE는 단일 모달리티, 본 연구는 3개 모달리티 동시 처리
- 특허: 크로스-모달 마스킹 전략 특허 출원 예정 ('27년)

**3. 새로운 알고리즘 개발 (15%)**
- Loss function 설계:
  - L_total = λ1·L_recon + λ2·L_contrast + λ3·L_cross-modal
  - L_recon: Masked token reconstruction (MSE)
  - L_contrast: InfoNCE (same subject pos, different subject neg)
  - L_cross-modal: fMRI ↔ EEG 상호 예측 (L1 distance)
- Hyperparameter search: Optuna 활용 (50 trials)

**4. 민간 모델 도입 (10%)**
- OpenAI CLIP: Text-image alignment → fMRI-EEG alignment로 응용
- Meta MAE: Image masking → 3D+temporal masking으로 확장

**5. 선행사업 모델 개선 (10%)**
- BrainIAC (Desai et al., Nature Neuroscience 2024):
  - 한계: fMRI only, 10,000 Western subjects
  - 본 연구: fMRI+dMRI+EEG, 10,000 Korean subjects
  - 성능 비교 목표: ADNI benchmark AUC 0.81 (BrainIAC) → 0.92 (K-NeuroMind)

### 2.3 사업 내 AI 활용

#### AI 기술 발전 기여도

**1. Foundation Model 패러다임의 뇌과학 적용 선도**
- 현황: Computer Vision(ViT, CLIP), NLP(GPT, BERT)에서는 Foundation Model 확립
- 한계: 뇌과학은 데이터 이질성(fMRI, EEG 등)으로 인해 단일 모달리티 모델만 존재
- 기여: 세계 최초 Multi-Modal Brain Foundation Model (M³-MAE) 개발
- 임팩트: Nature Neuroscience급 논문 3+편, 국제 학회 Best Paper 수상 목표

**2. Self-Supervised Learning의 Small Data 문제 해결**
- 현황: Self-supervised learning은 대규모 데이터 필요 (ImageNet 1.2M)
- 도전: 뇌 데이터는 수집 비용 높아 10K 규모 한계
- 기여: Masked Multi-Modal Autoencoding으로 10K에서도 효과적 학습 입증
- 파급: 의료영상 등 Small Data 도메인에 적용 가능한 범용 기법

**3. Graph Neural Network의 생물학적 네트워크 적용 확대**
- 기여: Brain connectome (84 ROIs, structural/functional connectivity)에 GAT 적용
- 혁신: Node feature로 multimodal embedding 사용 (기존: 단순 connectivity 행렬)
- 확장: 단백질 네트워크, 유전자 조절 네트워크 등 생물학 도메인으로 전이 가능

**4. Cross-Modal Learning 이론 발전**
- 기여: fMRI ↔ EEG 상호 예측 가능성 최초 검증
- 의의: 비용 높은 fMRI를 저렴한 EEG로 대체 가능성 제시
- 확장: CT ↔ MRI, 음성 ↔ 텍스트 등 다른 cross-modal 문제로 일반화

#### AI 보급·확산 기여도

**1. K-NeuroMind Open Platform 구축 (2030년 출시)**
- 공개 모델: M³-MAE 사전학습 가중치 (456M params, PyTorch/Hugging Face)
- 공개 데이터: 5,000명 익명화 뇌 스캔 (BIDS 포맷, 50 TB)
- 공개 코드: 전처리 파이프라인, 학습 스크립트 (GitHub, Apache 2.0 라이선스)
- 컴퓨팅 지원: KISTI 클라우드 무료 100,000 GPU-hours/년
- 목표 사용자: 국내 50개 연구기관 + 해외 100+ 연구자

**2. 교육 및 인력 양성**
- PhD 학생: 10명 배출 (AI + 뇌과학 융합 전문가)
- 석사 학생: 20명
- 학부 인턴: 200명 (여름 연구 프로그램)
- 온라인 코스: Coursera/edX에 "Brain Foundation Models" 강좌 개설 (목표 수강생 10,000명)

**3. 산학 협력 생태계 구축**
- 병원: 5개 병원 AI 진단팀 양성 (각 병원 AI 전문의 2명)
- 스타트업: K-NeuroMind 기반 스핀오프 기업 3개 창업 목표
  - BCI 기기 개발 스타트업 (목표 투자 유치 ₩50억)
  - 뇌 건강 모니터링 앱 개발 (MAU 100만 목표)
  - AI 진단 솔루션 제공 (병원 라이선스 판매)
- 대기업: 삼성/LG와 on-device brain AI 공동 연구

**4. 국제 협력 네트워크**
- NIH BRAIN Initiative: 데이터 교환 협약 (한국 5K ↔ 미국 HCP 5K)
- EU Human Brain Project: Joint workshop 연 1회 개최
- 일본 AMED Brain/MINDS: 동아시아 brain data 비교 연구

#### AI 기술 파급효과

**1. 의료 분야 혁신**
- 진단 정확도 향상: 알츠하이머 조기 예측 AUC 0.68 → 0.92 (+35%)
- 의료비 절감: 조기 개입으로 진행 지연 → 연간 ₩5조 절감 (전국 90만 환자 기준)
- 오진 감소: False negative 40% 감소 (AI 보조 진단 활용 시)
- 개인화 의료: 환자별 뇌 특성 기반 맞춤 치료 (치료 반응 예측)

**2. 경제 파급효과**
- 브레인 헬스 AI 시장 창출:
  - 진단 AI: ₩400억 ('26) → ₩1.2조 ('35)
  - BCI 기기: ₩100억 ('26) → ₩2.0조 ('35)
  - 인지 훈련 앱: ₩30억 ('26) → ₩800억 ('35)
  - 합계: ₩1.2조 → ₩4조 (CAGR 47%)
- 일자리 창출:
  - 직접 일자리: K-NeuroMind 프로젝트 80명
  - 간접 일자리: 스타트업 200명 + 병원 AI팀 200명 + 제약회사 100명 = 500명
- 수출 잠재력: 일본/싱가포르/중국 병원에 라이선스 수출 (₩20억/년)

**3. 사회적 파급효과**
- 건강 수명 연장: 치매 free 기간 +5년 (68세 → 73세)
- 삶의 질 향상: QALY +2.6 (Quality-Adjusted Life Years)
- 간병 부담 감소: 가족 간병인 부담 30% 경감
- 복지 비용 절감: 장기요양보험 지출 20% 감소

**4. 과학기술 파급효과**
- 고인용 논문: Nature/Science 2편 + Nature Neuroscience 4편 + PNAS 6편 = 12편 (목표 총 인용 1,000+)
- 특허: 국내 5건, PCT 국제 출원 3건
- 표준화: Brain AI 데이터 포맷 표준 (ISO/IEC 제안)
- 국제 위상: 한국을 AI-뇌과학 글로벌 Top 3 국가로 도약

---

## 3. 주요 기술개발 사항

### 사업 기간 선택

**선택**: ☑ 5년 이상 ('26~'30)

**3년 이내 기술개발이 어려운 사유 및 향후 AI 발전에 따른 사업 대응 방향:**

**1. 데이터 수집 기간 (최소 3년 필요)**
- Year 1-2: 신규 2,000명 스캔 (병원 섭외, IRB 승인, 스캔 예약 대기 등)
- Year 3: 종단 추적 데이터 (1년 간격 재스캔으로 longitudinal data 확보)
- 고품질 뇌 데이터 수집은 물리적으로 3년 이상 소요 불가피

**2. 모델 학습 및 검증 기간 (최소 2년 필요)**
- Year 3: Self-supervised pretraining (3개월 학습 + 3개월 hyperparameter tuning)
- Year 4: Supervised fine-tuning (5개 질환 × 2개월 = 10개월)
- Year 4-5: 임상 검증 (RCT 1,000명, 최소 18개월 추적 필요)
- AI 모델의 안전성/유효성 입증을 위해 충분한 검증 기간 필수

**3. 향후 AI 발전에 따른 사업 대응 방향**
- Modular architecture: 새로운 modality (예: MEG, PET) 추가 가능하도록 설계
- Transfer learning: 다른 질환(예: ADHD, 조울증)으로 확장 용이
- Continuous learning: 신규 데이터로 모델 지속 업데이트 (federated learning)
- Explainability 강화: SHAP, GradCAM 등 최신 XAI 기법 도입

### 사업 기간별 주요 기술개발 사항

| 연도 | 주요 기술개발 사항 | 정량적 목표 |
|------|------------------|-----------|
| **'26** | **데이터 인프라 구축**<br>• 5개 병원 IRB 승인 및 데이터 수집 시작<br>• BIDS 포맷 데이터베이스 구축<br>• 자동화 QA 파이프라인 개발<br><br>**개별 모달리티 인코더 개발**<br>• fMRI: ViT-L/16 fine-tuning<br>• dMRI: GAT 학습<br>• EEG: 1D CNN 개발 | • Database v1.0 (3,000 subjects)<br>• QA 성공률 > 95%<br><br>• fMRI encoder val accuracy > 75%<br>• dMRI encoder val accuracy > 70%<br>• EEG encoder val accuracy > 70% |
| **'27** | **크로스-모달 통합 모델 개발**<br>• M³-MAE 아키텍처 설계<br>• Self-supervised pretraining (5,000 subjects)<br>• Masked reconstruction loss 최적화<br><br>**데이터 확장**<br>• 신규 스캔 1,000명 추가<br>• Longitudinal data 500명 (1년 추적) | • M³-MAE v1.0 (456M params)<br>• Masked recon MAE < 0.10<br>• Cross-modal prediction acc > 75%<br><br>• Total dataset: 6,000 subjects |
| **'28** | **파운데이션 모델 완성 및 검증**<br>• M³-MAE pretraining 완료 (8,000 subjects)<br>• 알츠하이머 fine-tuning<br>• 국제 벤치마크 테스트 (ADNI, ABIDE, fMRI-IQ)<br><br>**Proof-of-Concept 달성** | • M³-MAE v2.0 (full scale)<br>• Alzheimer AUC > 0.85<br>• ADNI benchmark: beat SOTA by 10%<br>• PoC demo 완성 |
| **'29** | **다중 질환 모델 개발**<br>• 우울증, 조현병, 파킨슨병, 자폐 fine-tuning<br>• 실시간 BCI 프로토타입 개발<br>• 임상시험 시작 (5개 병원, 1,000명 RCT)<br><br>**모델 고도화** | • 5개 질환 모두 AUC > 0.80<br>• BCI latency < 200ms<br>• RCT 중간 분석: 진단 시간 20% 단축 |
| **'30** | **K-NeuroMind Open Platform 출시**<br>• 사전학습 가중치 공개 (Hugging Face)<br>• 5,000명 데이터셋 공개 (BIDS)<br>• KISTI 클라우드 무료 GPU 제공<br><br>**임상시험 완료 및 상용화 준비** | • Platform 사용자 1,000+<br>• RCT 최종 결과: 진단 30% 빠름, 비용 20% 감소<br>• 3개 스타트업 창업<br>• MFDS 의료기기 인증 신청 |

### AI 관련 R&D 사업 컴퓨팅 자원 구축, 활용 방안

**선택**: ☑ 클라우드 등을 통한 AI 컴퓨팅 자원 활용

**주요 내용:**

**1. 컴퓨팅 자원 전략: 하이브리드 (KISTI 무료 + Cloud burst)**

**가. KISTI 슈퍼컴퓨터 활용 ('26-'28)**
- 자원: Nurion 슈퍼컴퓨터 (25.7 petaflops, NVIDIA V100 GPUs)
- 할당: 무료 500,000 node-hours (과기정통부 승인 확보)
- 용도:
  - 데이터 전처리: 100,000 node-hours
  - 개별 인코더 학습: 200,000 node-hours
  - 초기 M³-MAE pretraining: 200,000 node-hours

**나. NVIDIA AI Hub 파트너십 ('27-'28)**
- 자원: 500 H100 GPUs × 6개월 (협약 체결 완료)
- 용도: M³-MAE full-scale pretraining (13,824 GPU-hours)
- 비용: 정가 ₩41억 → 파트너십으로 ₩10억 (75% 할인)

**다. Cloud burst strategy ('28-'30)**
- 자원: AWS/GCP GPU instances (A100, H100)
- 용도:
  - Peak demand 시 (fine-tuning 5개 질환 동시)
  - 국제 공동연구 데이터 분석 (federated learning)
- 비용: ₩2억/년 × 3년 = ₩6억

**라. On-premise GPU 구매 ('28)**
- 자원: 32 NVIDIA A100 GPUs (구매)
- 용도: 지속적 모델 업데이트, 임상 배포 inference
- 비용: ₩8억 (일시 구매)

**총 컴퓨팅 비용: ₩24억 (5년)**
- KISTI: ₩0 (무료)
- NVIDIA Hub: ₩10억
- Cloud: ₩6억
- On-premise: ₩8억

---

## 4. 예산 규모

### AI 관련 예산 규모 (연도별, 백만원 단위)

| 항목 | '26 | '27 | '28 | '29 | '30 | 합계 |
|------|-----|-----|-----|-----|-----|------|
| **데이터 수집 비용** | 500 | 600 | 400 | 200 | 100 | 1,800 |
| **데이터 전처리 비용** | 200 | 250 | 200 | 50 | 50 | 750 |
| **모델 훈련 및 평가 컴퓨팅 비용** | 300 | 400 | 500 | 400 | 300 | 1,900 |
| **모델 운영·유지보수 비용** | 50 | 100 | 150 | 200 | 250 | 750 |
| **기타 AI 관련 비용** | 550 | 530 | 950 | 1,515 | 1,388 | 4,933 |
| **총계** | **1,600** | **1,880** | **2,200** | **2,365** | **2,088** | **10,133** |

### 세부 설명

**1. 데이터 수집 비용 (₩18억)**
- '26: ₩500M (500명 신규 스캔)
- '27: ₩600M (600명 신규 스캔 + 500명 종단 추적)
- '28: ₩400M (400명 신규 스캔 + 300명 종단 추적)
- '29: ₩200M (임상시험 환자 200명 스캔)
- '30: ₩100M (추가 validation 데이터)

**2. 데이터 전처리 비용 (₩7.5억)**
- '26: ₩200M (BIDS 변환, QA pipeline 개발, 라벨링 500명)
- '27: ₩250M (라벨링 1,000명, augmentation 실험)
- '28: ₩200M (라벨링 500명, QA 고도화)
- '29: ₩50M (임상 데이터 정제)
- '30: ₩50M (Open Platform 데이터 최종 검증)

**3. 모델 훈련 및 평가 컴퓨팅 비용 (₩19억)**
- '26: ₩300M (개별 인코더 학습, Cloud GPU)
- '27: ₩400M (M³-MAE pretraining 초기, KISTI 보충)
- '28: ₩500M (NVIDIA Hub ₩100M + On-premise GPU 구매 ₩800M - 초과분)
- '29: ₩400M (Fine-tuning 5개 질환, Cloud burst)
- '30: ₩300M (Continuous learning, Platform 운영)

**4. 모델 운영·유지보수 비용 (₩7.5억)**
- '26: ₩50M (초기 인프라 구축)
- '27: ₩100M (모델 registry, MLOps pipeline)
- '28: ₩150M (임상 배포 준비, monitoring dashboard)
- '29: ₩200M (5개 병원 배포, 24/7 운영)
- '30: ₩250M (Open Platform 운영, API 서버)

**5. 기타 AI 관련 비용 (₩49.33억)** - 인건비 + 임상시험 + Platform 개발
- '26: ₩550M (인건비 ₩400M + 장비 ₩150M)
- '27: ₩530M (인건비 ₩500M + 학회/출판 ₩30M)
- '28: ₩950M (인건비 ₩600M + 임상시험 pilot ₩300M + 플랫폼 개발 ₩50M)
- '29: ₩1,515M (인건비 ₩700M + RCT 본격 시작 ₩600M + 플랫폼 ₩215M)
- '30: ₩1,388M (인건비 ₩800M + RCT 완료 ₩400M + 플랫폼 완성 ₩188M)

**총계 검증**: ₩18억 + ₩7.5억 + ₩19억 + ₩7.5억 + ₩49.33억 = **₩101.33억** ✓

---

**문서 작성일**: 2025-10-18
**작성자**: K-NeuroMind 연구팀
**연락처**: [연구책임자 정보]

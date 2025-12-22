# Block 3: 추진전략/방법 및 추진체계

> 권장 분량: 3-3.5페이지 | 평가 가중치: 30% (방법론 적합성)

## 핵심 체크리스트
- [x] 목표별 세부 방법이 구체적으로 기술됨
- [x] 각 실험/분석의 성공 기준(정량지표) 정의됨
- [x] 핵심 리스크 + 대안 프로토콜 명시됨
- [x] 추론 시스템 개발 방법론 포함 (목표 4)
- [ ] Fig 2 (모델 아키텍처) 포함
- [ ] Fig 3 (데이터 파이프라인) 포함
- [ ] Fig 4 (5년 Gantt + Go/No-Go) 포함
- [x] 추진체계(인력/장비/협력) 명시됨
- [x] 연구기간/연구비 적정성 근거 제시됨

---

## 1) 연구과제의 추진전략/방법

**[Fig 2 삽입 위치]**: LifeSpan-FM 모델 + 추론 시스템 아키텍처
- 4D Swin Transformer Encoder (시공간 자기지도학습)
- Cross-Modal Attention (뇌영상-유전체 융합)
- Age-Continuous Decoder (발달-노화 연속 예측)
- Inference Engine (ONNX/TensorRT 최적화)

---

### 목표 1: LifeSpan-FM 파운데이션 모델 개발 (1-2차년도)

#### 1.1 데이터 수집 및 전처리

| 데이터셋 | 규모 | 연령 범위 | 모달리티 | 전처리 |
|----------|------|----------|----------|--------|
| UK Biobank | 50,000 | 45-82세 | 구조, 기능적 자기공명영상, 유전체 | FreeSurfer + 기능적 자기공명영상 전처리 |
| ABCD Study | 12,000 | 9-14세 | 구조, 기능적 자기공명영상, 뇌파, 유전체 | ABCD-HCP 파이프라인 + MNE-Python |
| HCP-Development | 1,300 | 5-21세 | 구조, 기능적 자기공명영상, MEG | HCP Minimal Preprocessing |
| HCP-Aging | 1,200 | 36-100세 | 구조, 기능적 자기공명영상 | HCP Minimal Preprocessing |
| TUEG (뇌파 사전학습) | 30,000시간 | 전 연령 | 뇌파 (19-128채널) | DIVER 전처리 (200Hz) |
| FACED/SeedV | 2,000+ | 20-60세 | 뇌파 (62-64채널) | 감정/인지 벤치마크 |

**전처리 파이프라인**:
1. 구조 자기공명영상: FreeSurfer recon-all → Parcellation (Schaefer 400 ROI)
2. 기능 자기공명영상: 기능적 자기공명영상 전처리 → ICA-AROMA 잡음제거 → 연결성 행렬
3. 뇌파: MNE-Python → 200Hz resampling → Bad channel interpolation → ICA artifact rejection
4. 유전체: QC → Imputation → 다유전자위험점수 계산 (LDpred2)

**성공 기준**: QC 통과율 > 90%, 통합 데이터셋 N > 60,000 (자기공명영상), N > 30,000시간 (뇌파)

#### 1.2 4D Swin Transformer 구현

```
아키텍처 명세 (5B 파라미터 목표):
├── Input: 3D 자기공명영상 volume (182×218×182) + Time dimension
├── Patch Embedding: 4×4×4 voxel patches → 1024-dim tokens
├── 4D Swin Blocks (×6 stages):
│   ├── Window Multi-head Self-Attention (W-MSA)
│   ├── Shifted Window MSA (SW-MSA)
│   └── MLP + LayerNorm + Residual
├── Self-Supervised Objectives:
│   ├── Masked Autoencoder (평균절대오차): 75% masking ratio
│   ├── Contrastive Learning (SimCLR): multi-view augmentation
│   └── Age-Continuous Prediction (auxiliary)
└── Output: 2048-dim representation per subject
```

**스케일링 전략 (1-2차년도)**:
| 단계 | 파라미터 | 데이터 | 컴퓨팅 |
|------|----------|--------|--------|
| 1차년도 전반 | 500M | UK Biobank 20K | A100 × 8 |
| 1차년도 후반 | 1B | UK Biobank 50K | A100 × 16 |
| 2차년도 전반 | 3B | +ABCD +HCP | A100 × 32 |
| 2차년도 후반 | 5B | 전체 70K | A100 × 64 |

**핵심 리스크 및 대안**:
| 리스크 | 대안 |
|--------|------|
| GPU 메모리 부족 | Gradient Checkpointing + DeepSpeed ZeRO-3 |
| 수렴 실패 | Learning rate warmup + Cosine annealing |
| 발달-노화 분포 불균형 | Balanced sampling + Age-stratified batching |

---

### 목표 2: 멀티모달 융합 및 바이오마커 발견 (2-3차년도)

#### 2.1 DIVER 기반 뇌파 파운데이션 모델

**DIVER 뇌파 Encoder 아키텍처** (채널 순열 등변성):
```
아키텍처 명세 (DIVER-0 기반):
├── Input: 뇌파 (C channels × T samples, 200Hz)
├── Patch Embedding:
│   ├── 1초 패치 단위 (200 samples/patch)
│   ├── CNN: Conv1D(1→64→128) → temporal features
│   └── FFT: 주파수 대역 파워 (delta~gamma)
├── DIVER Transformer Blocks (×12):
│   ├── Unified Spatio-Temporal Attention
│   │   ├── RoPE (Rotary Position Embedding): 시간 관계
│   │   └── Binary Attention Bias: 채널 구분
│   ├── STCPE (Sliding Temporal Conditional Positional Encoding)
│   │   └── 채널 순열 등변성 보장
│   └── MLP + LayerNorm + Residual
├── Hidden dim: 200, Attention heads: 10
└── Output: 2048-dim 뇌파 representation

핵심 특징:
- 채널 순열 등변성: 전극 배치 변화에 강건
- TUEG 30,000시간 사전학습 활용
- Cross-dataset 일반화 (FACED: 59.2% balanced acc)
```

**뇌파 바이오마커 추출**:
| 바이오마커 | 발달 지표 | 노화 지표 |
|-----------|----------|----------|
| Alpha Peak Frequency | 증가 (아동→성인) | 감소 (노화) |
| Theta/Beta Ratio | 감소 (주의력 발달) | 증가 (인지저하) |
| P300 Latency | 감소 (처리속도 향상) | 증가 (인지저하) |
| N170 Amplitude | 증가 (안면인식 발달) | 감소 |
| PLV Connectivity | 증가 (네트워크 성숙) | 감소 (연결성 저하) |

#### 2.2 뇌영상-뇌파-유전체 3-Modal 융합

**Cross-Modal Contrastive Learning (확장)**:
```
손실 함수: L = L_img + L_eeg + L_gen + λ·L_cross + μ·L_align

L_cross = L_cross(img,eeg) + L_cross(img,gen) + L_cross(eeg,gen)
L_align = ||proj(z_img) - proj(z_eeg) - proj(z_gen)||²

z_img: 뇌영상 임베딩 (4D Swin output, 2048-dim)
z_eeg: 뇌파 임베딩 (DIVER output, 2048-dim)
z_gen: 유전체 임베딩 (다유전자위험점수 Transformer, 2048-dim)
τ: temperature parameter = 0.07
```

**Flexible 3-Modal Inference 설계**:
```
┌────────────────────────────────────────────────────────┐
│         3-Modal Fusion Module (LifeSpan-FM)            │
├────────────────────────────────────────────────────────┤
│  Modality Encoders:                                    │
│  ├── 자기공명영상 Encoder (4D Swin) → z_img (2048-dim)         │
│  ├── 뇌파 Encoder (DIVER) → z_eeg (2048-dim)           │
│  ├── Genomic Encoder (Transformer) → z_gen (2048-dim) │
│  └── Missing Modality Token → z_missing               │
│                                                        │
│  Cross-Modal Attention (Flexible):                     │
│  ├── Full: Attention(z_img, z_eeg, z_gen)             │
│  ├── 자기공명영상+뇌파: Attention(z_img, z_eeg, z_missing)      │
│  ├── 자기공명영상+Gen: Attention(z_img, z_missing, z_gen)      │
│  ├── 뇌파+Gen: Attention(z_missing, z_eeg, z_gen)      │
│  ├── 자기공명영상 only: Attention(z_img, z_missing, z_missing) │
│  └── 뇌파 only: Attention(z_missing, z_eeg, z_missing) │
│                                                        │
│  Output: Unified representation (2048-dim)             │
└────────────────────────────────────────────────────────┘
```

**성공 기준**:
- Cross-modal retrieval accuracy > 75%
- 발달-노화 상관 Pearson r > 0.3
- 뇌파-자기공명영상 구조-기능 상관 r > 0.25
- Missing modality 성능 저하 < 15%

---

### 목표 3: 임상 검증 및 한국인 적응 (3-4차년도)

#### 3.1 Brain Age Gap 예측 모델

**Fine-tuning 전략**:
- 기법: 저랭크적응 (Low-Rank Adaptation), rank=32
- 학습 데이터: UK Biobank 80% train / 20% test
- 손실 함수: L1 Loss + Age distribution regularization

#### 3.2 한국인 코호트 검증

| 단계 | 방법 | 데이터 |
|------|------|--------|
| 도메인 적응 | 저랭크적응 fine-tuning (서양→한국인) | KoGES 8,000명 |
| 성능 검증 | Age prediction + Gap validation | KoGES held-out 2,000명 |
| 임상 상관 | 인지검사, 치매 위험도 연관 분석 | KoGES 인지평가 데이터 |

**성공 기준**:
- 한국인 평균절대오차 < 3.0년 (서양인 대비 10% 이내 차이)
- 인지저하 예측 곡선하면적 > 0.75

---

### 목표 4: 추론 시스템 및 임상 배포 (4-5차년도)

#### 4.1 모델 최적화 및 경량화

**최적화 파이프라인**:
```
PyTorch Model (5B params)
    ↓ Knowledge Distillation
Student Model (500M params)
    ↓ ONNX Export
ONNX Model
    ↓ TensorRT Optimization
TensorRT Engine (INT8 quantization)
    ↓ Triton Inference Server
Production Deployment
```

| 최적화 기법 | 효과 | 정확도 손실 |
|------------|------|------------|
| Knowledge Distillation | 10× 파라미터 감소 | 평균절대오차 +0.2년 |
| ONNX Conversion | 플랫폼 호환성 | 없음 |
| TensorRT FP16 | 2× 속도 향상 | 없음 |
| INT8 Quantization | 4× 속도 향상 | 평균절대오차 +0.1년 |

**최종 목표**: 5B → 500M (distilled), 추론 시간 120초 → 25초

#### 4.2 추론 서비스 개발

**시스템 아키텍처**:
```yaml
# docker-compose.yml 구조
services:
  api-gateway:
    image: nginx:alpine
    ports: ["443:443"]

  inference-service:
    image: lifespan-fm:inference
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  preprocessing:
    image: lifespan-fm:preprocess
    volumes:
      - dicom-data:/data

  result-generator:
    image: lifespan-fm:report
    environment:
      - 의료정보교환표준_SERVER_URL=...
```

**서비스 명세**:
```
POST /api/v1/predict
Content-Type: multipart/form-data

Request:
  - dicom_file: 구조 자기공명영상 의료영상표준 (required if no eeg_file)
  - fmri_file: 기능적 자기공명영상 의료영상표준 (optional)
  - eeg_file: 뇌파 EDF/BDF file (optional, or required if no dicom)
  - prs_file: 다유전자위험점수 scores JSON (optional)
  - patient_id: string

Response:
  {
    "brain_age": 45.3,
    "chronological_age": 42.0,
    "brain_age_gap": 3.3,
    "risk_category": "elevated",
    "confidence": 0.92,
    "modalities_used": ["구조", "기능적 자기공명영상", "뇌파"],
    "eeg_biomarkers": {
      "alpha_peak_freq": 10.2,
      "theta_beta_ratio": 1.8,
      "p300_latency": 320,
      "plv_global": 0.45
    },
    "report_url": "/reports/xxx.pdf",
    "fhir_resource": {...}
  }
```

#### 4.3 임상의사결정지원시스템 통합

**전자의무기록/의료영상저장전송시스템 연동**:
| 표준 | 용도 | 구현 |
|------|------|------|
| 의료영상표준 | 영상 수신 | Orthanc 의료영상표준 server |
| HL7 의료정보교환표준 R4 | 결과 전송 | HAPI 의료정보교환표준 |
| IHE XDS | 문서 공유 | OpenXDS |

**임상 워크플로우**:
```
자기공명영상 워크플로우:
1. 자기공명영상 촬영 완료 → 의료영상저장전송시스템 저장
2. 의료영상표준 Listener가 새 영상 감지
3. 자동 전처리 + 추론 실행
4. Brain Age Report 생성

뇌파 워크플로우 (저비용 스크리닝):
1. 뇌파 측정 완료 → EDF 저장
2. 뇌파 Listener가 새 파일 감지
3. DIVER 전처리 + 추론 실행
4. 뇌파 Brain Age Report 생성

멀티모달 워크플로우 (종합):
1. 자기공명영상 + 뇌파 측정 완료
2. 모달리티별 전처리 병렬 실행
3. 3-Modal Fusion 추론
4. 통합 Brain Age Report 생성
5. 전자의무기록에 의료정보교환표준 Observation 전송
6. 담당의에게 알림 (Risk > threshold)
```

#### 4.4 파일럿 운영

| 단계 | 기간 | 대상 | 목표 |
|------|------|------|------|
| Alpha | 4차년도 Q3 | 내부 테스트 50건 | 시스템 안정성 |
| Beta | 4차년도 Q4 | 협력병원 200건 | 워크플로우 검증 |
| Pilot | 5차년도 | 1,000건 | 임상 유용성 평가 |

**성공 기준**:
- 추론 지연시간 < 30초/영상
- 시스템 가용성 > 99.5%
- 임상의 만족도 > 4.0/5.0

---

**[Fig 3 삽입 위치]**: End-to-End 파이프라인
- 좌: 데이터 (UK Biobank, ABCD, KoGES, 임상)
- 중: 모델 학습 + 최적화
- 우: 추론 시스템 + 임상 배포

---

## 2) 연구과제의 추진체계

### 연구 인력 구성 (5년)

| 역할 | 인원 | 담당 업무 | 참여 기간 |
|------|------|----------|----------|
| 연구책임자 | 1명 | 총괄, 목표 설계 | 전 기간 |
| 박사후연구원 | 2명 | 목표 1-2 모델, 목표 4 시스템 | 전 기간 |
| 박사과정생 | 3명 | 데이터, 모델, 검증 | 전 기간 |
| 석사과정생 | 2명 | 전처리, 서비스 개발 | 3-5차년도 |
| 소프트웨어 엔지니어 | 1명 | 추론 시스템 개발 | 4-5차년도 |

### 연구 장비 및 인프라

| 인프라 | 사양 | 용도 | 확보 시점 |
|--------|------|------|----------|
| GPU 클러스터 | A100 80GB × 8 | 사전학습 | 보유 |
| 클라우드 GPU | A100 × 64 (임대) | 스케일업 | 2차년도 |
| 추론 서버 | L40 × 4 | 배포 | 4차년도 |
| 데이터 서버 | 200TB Storage | 저장 | 보유 |

### 협력 네트워크

| 기관 | 협력 내용 | 역할 |
|------|----------|------|
| UK Biobank | 데이터 접근 | Approved Researcher |
| ABCD Study | 데이터 접근 | Data Use Agreement |
| [한국 협력병원] | 파일럿 운영 | 임상 검증 |
| [한국 협력기관] | KoGES 데이터 | 한국인 검증 |

---

## 3) 연구기간 및 연구비 적정성

**[Fig 4 삽입 위치]**: 5년 Gantt 차트 + 마일스톤

### 연차별 목표 및 마일스톤

| 년차 | 주요 목표 | 마일스톤 | Go/No-Go |
|------|----------|----------|----------|
| **1차** | 데이터 통합 + 10억 파라미터 모델 | M1: 데이터 60K (6개월) | N > 50,000 |
| | | M2: 10억 파라미터 모델 완료 (12개월) | 평균절대오차 < 3.5년 |
| **2차** | 50억 파라미터 스케일업 + 멀티모달 | M3: 50억 파라미터 사전학습 (6개월) | 평균절대오차 < 3.0년 |
| | | M4: 교차모달 융합 (12개월) | 검색 정확도 > 70% |
| **3차** | 바이오마커 + 한국인 적응 | M5: 바이오마커 (6개월) | r > 0.3 |
| | | M6: KoGES 적응 (12개월) | 평균절대오차 < 3.0년 |
| **4차** | 검증 완료 + 추론 시스템 | M7: 임상 검증 (6개월) | 곡선하면적 > 0.75 |
| | | M8: 서비스 개발 (12개월) | 지연시간 < 60초 |
| **5차** | 파일럿 + 배포 | M9: 최적화 완료 (6개월) | 지연시간 < 30초 |
| | | M10: 파일럿 1K (12개월) | 만족도 > 4.0 |

### 연구비 산정 근거 (연 1.5억원 × 5년 = 7.5억원)

| 항목 | 비율 | 금액/년 | 산정 근거 |
|------|------|---------|----------|
| 인건비 | 45% | 6,750만원 | 박사후연구원 2명 + 학생 5명 |
| 장비/재료비 | 25% | 3,750만원 | GPU 클라우드, 추론 서버 |
| 위탁/협력비 | 15% | 2,250만원 | 국제 협력, 데이터 접근 |
| 간접비 | 15% | 2,250만원 | 기관 간접비 |

**연차별 장비/재료비 상세**:
| 년차 | 주요 지출 | 금액 |
|------|----------|------|
| 1-2차 | GPU 클라우드 (A100 × 64, 6개월) | 6,000만원 |
| 3차 | 데이터 접근 비용, 협력 | 2,000만원 |
| 4-5차 | 추론 서버 (L40 × 4) 구매 | 5,000만원 |

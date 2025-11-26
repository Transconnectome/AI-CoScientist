# CENTaUR dgx-spark GPU 실험 계획서

## 환경 정보
- **서버**: dgx-spark (ARM64)
- **GPU**: NVIDIA GB10 (Blackwell, compute capability 12.1)
- **PyTorch**: 2.10.0.dev20251105+cu128
- **프로젝트 경로**: ~/git/CENTaUR
- **가상환경**: ~/git/CENTaUR/venv

## 실험 목표
Binz & Schulz (2023) 방법론을 한국어 LLM으로 재현하여 인지 모델링 성능 평가

### 평가 모델
1. **Qwen2.5-32B-Instruct** (base model)
2. **DeepSeek-R1-Distill-Qwen-32B** (base model)

### 성공 기준
- Random baseline (≈ 120K NLL) 대비 유의미한 성능 향상
- LLaMA-65B baseline (≈ 30K NLL) 수준 달성

---

## Phase 1: 환경 검증 및 벤치마킹 (1-2시간)

### 1.1 GPU 성능 기초 테스트 ✅ COMPLETED
```bash
# 이미 완료된 테스트
python /tmp/test_feature_extraction.py
```
**결과**:
- 10 samples @ 0.040 sec/sample
- GPU memory: 136 MB allocated, 162 MB reserved
- 정상 작동 확인 ✅

### 1.2 CENTaUR Feature Extraction 벤치마크
```bash
#!/bin/bash
# ~/git/CENTaUR/benchmark_gpu.sh

set -e
cd ~/git/CENTaUR
source venv/bin/activate

echo "=== CENTaUR GPU 벤치마크 시작 ==="
date

# Test 1: 소규모 샘플 (10개)
echo -e "\n[Test 1] 10 samples 처리 시간 측정"
time python scripts/extract_centaur_features.py \
    --model qwen25-base \
    --n_samples 10 \
    --output_path outputs/benchmark_10samples.npz

# Test 2: 중규모 샘플 (100개)
echo -e "\n[Test 2] 100 samples 처리 시간 측정"
time python scripts/extract_centaur_features.py \
    --model qwen25-base \
    --n_samples 100 \
    --output_path outputs/benchmark_100samples.npz

# Test 3: GPU 메모리 프로파일링
echo -e "\n[Test 3] GPU 메모리 사용량 확인"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 > gpu_profile.csv &
PROFILE_PID=$!

python scripts/extract_centaur_features.py \
    --model qwen25-base \
    --n_samples 100 \
    --output_path outputs/benchmark_memory.npz

kill $PROFILE_PID

echo -e "\n=== 벤치마크 완료 ==="
date
```

**실행**:
```bash
ssh dgx-spark 'bash ~/git/CENTaUR/benchmark_gpu.sh 2>&1 | tee ~/benchmark_results.log'
```

**예상 결과**:
- 10 samples: ~1-2분
- 100 samples: ~10-20분
- GPU memory usage: 2-4 GB (32B 모델 기준)

---

## Phase 2: Feature Extraction 본실험 (2-4시간)

### 2.1 Qwen2.5-32B-Instruct Base Model

```bash
#!/bin/bash
# ~/git/CENTaUR/run_qwen25_full.sh

set -e
cd ~/git/CENTaUR
source venv/bin/activate

echo "=== Qwen2.5-32B Feature Extraction 시작 ==="
date

# 전체 데이터셋 처리
python scripts/extract_centaur_features.py \
    --model qwen25-base \
    --output_path outputs/qwen25_base_features.npz \
    --use_quantization \
    2>&1 | tee logs/qwen25_base_extraction.log

echo "=== Feature extraction 완료 ==="
date

# 결과 확인
python -c "
import numpy as np
data = np.load('outputs/qwen25_base_features.npz')
print(f'Features shape: {data[\"features\"].shape}')
print(f'Labels shape: {data[\"labels\"].shape}')
print(f'Total samples: {len(data[\"labels\"])}')
"
```

### 2.2 DeepSeek-R1-Distill-Qwen-32B Base Model

```bash
#!/bin/bash
# ~/git/CENTaUR/run_deepseek_full.sh

set -e
cd ~/git/CENTaUR
source venv/bin/activate

echo "=== DeepSeek-R1 Feature Extraction 시작 ==="
date

python scripts/extract_centaur_features.py \
    --model deepseek-base \
    --output_path outputs/deepseek_base_features.npz \
    --use_quantization \
    2>&1 | tee logs/deepseek_base_extraction.log

echo "=== Feature extraction 완료 ==="
date

# 결과 확인
python -c "
import numpy as np
data = np.load('outputs/deepseek_base_features.npz')
print(f'Features shape: {data[\"features\"].shape}')
print(f'Labels shape: {data[\"labels\"].shape}')
print(f'Total samples: {len(data[\"labels\"])}')
"
```

**실행 (tmux 세션에서)**:
```bash
# tmux 세션 시작
ssh dgx-spark
tmux new -s centaur_exp

# Qwen 실험
cd ~/git/CENTaUR
bash run_qwen25_full.sh

# 완료 후 DeepSeek 실험
bash run_deepseek_full.sh

# 세션 detach: Ctrl+b, d
# 세션 재접속: tmux attach -t centaur_exp
```

---

## Phase 3: 100-Fold LOO Cross-Validation (4-8시간)

### 3.1 Qwen2.5 모델 평가

```bash
#!/bin/bash
# ~/git/CENTaUR/run_qwen25_loo_cv.sh

set -e
cd ~/git/CENTaUR
source venv/bin/activate

echo "=== Qwen2.5 LOO CV 시작 ==="
date

python scripts/fit_centaur_loo_cv.py \
    --features_path outputs/qwen25_base_features.npz \
    --output_path outputs/qwen25_base_nll_results.json \
    --n_folds 100 \
    2>&1 | tee logs/qwen25_loo_cv.log

echo "=== LOO CV 완료 ==="
date

# NLL 결과 확인
python -c "
import json
with open('outputs/qwen25_base_nll_results.json') as f:
    results = json.load(f)
print(f'Mean NLL: {results[\"mean_nll\"]:.2f}')
print(f'Std NLL: {results[\"std_nll\"]:.2f}')
print(f'Total samples: {results[\"total_samples\"]}')
"
```

### 3.2 DeepSeek 모델 평가

```bash
#!/bin/bash
# ~/git/CENTaUR/run_deepseek_loo_cv.sh

set -e
cd ~/git/CENTaUR
source venv/bin/activate

echo "=== DeepSeek LOO CV 시작 ==="
date

python scripts/fit_centaur_loo_cv.py \
    --features_path outputs/deepseek_base_features.npz \
    --output_path outputs/deepseek_base_nll_results.json \
    --n_folds 100 \
    2>&1 | tee logs/deepseek_loo_cv.log

echo "=== LOO CV 완료 ==="
date

# NLL 결과 확인
python -c "
import json
with open('outputs/deepseek_base_nll_results.json') as f:
    results = json.load(f)
print(f'Mean NLL: {results[\"mean_nll\"]:.2f}')
print(f'Std NLL: {results[\"std_nll\"]:.2f}')
print(f'Total samples: {results[\"total_samples\"]}')
"
```

**실행**:
```bash
ssh dgx-spark
tmux attach -t centaur_exp

# Qwen LOO CV
bash ~/git/CENTaUR/run_qwen25_loo_cv.sh

# DeepSeek LOO CV
bash ~/git/CENTaUR/run_deepseek_loo_cv.sh
```

---

## Phase 4: 결과 분석 및 비교

### 4.1 통합 결과 리포트 생성

```python
# ~/git/CENTaUR/generate_report.py

import json
import numpy as np
from pathlib import Path

def generate_report():
    print("=" * 60)
    print("CENTaUR dgx-spark 실험 결과 리포트")
    print("=" * 60)

    # Baseline NLL values
    random_nll = 120000
    llama65b_nll = 30000

    # Qwen2.5 results
    print("\n1. Qwen2.5-32B-Instruct Base Model")
    with open('outputs/qwen25_base_nll_results.json') as f:
        qwen_results = json.load(f)

    qwen_nll = qwen_results['mean_nll']
    qwen_std = qwen_results['std_nll']

    print(f"   Mean NLL: {qwen_nll:.2f} ± {qwen_std:.2f}")
    print(f"   vs Random: {((random_nll - qwen_nll) / random_nll * 100):.1f}% improvement")
    print(f"   vs LLaMA-65B: {((llama65b_nll - qwen_nll) / llama65b_nll * 100):.1f}% {'improvement' if qwen_nll < llama65b_nll else 'degradation'}")

    # DeepSeek results
    print("\n2. DeepSeek-R1-Distill-Qwen-32B Base Model")
    with open('outputs/deepseek_base_nll_results.json') as f:
        deepseek_results = json.load(f)

    deepseek_nll = deepseek_results['mean_nll']
    deepseek_std = deepseek_results['std_nll']

    print(f"   Mean NLL: {deepseek_nll:.2f} ± {deepseek_std:.2f}")
    print(f"   vs Random: {((random_nll - deepseek_nll) / random_nll * 100):.1f}% improvement")
    print(f"   vs LLaMA-65B: {((llama65b_nll - deepseek_nll) / llama65b_nll * 100):.1f}% {'improvement' if deepseek_nll < llama65b_nll else 'degradation'}")

    # Model comparison
    print("\n3. 모델 비교")
    better_model = "Qwen2.5" if qwen_nll < deepseek_nll else "DeepSeek"
    diff = abs(qwen_nll - deepseek_nll)
    print(f"   Best model: {better_model}")
    print(f"   NLL difference: {diff:.2f}")

    # GPU performance
    print("\n4. GPU 성능 메트릭")
    benchmark_log = Path('logs/qwen25_base_extraction.log').read_text()
    # Extract timing info from logs
    print("   Feature extraction 시간: [로그에서 추출]")
    print("   평균 처리 속도: [samples/sec]")
    print("   GPU 메모리 사용량: [GB]")

    print("\n" + "=" * 60)
    print("실험 완료")
    print("=" * 60)

if __name__ == "__main__":
    generate_report()
```

**실행**:
```bash
ssh dgx-spark 'cd ~/git/CENTaUR && python generate_report.py'
```

---

## 실험 체크리스트

### Phase 1: 환경 검증 ✅
- [x] GPU 기초 테스트 완료 (test_feature_extraction.py)
- [ ] CENTaUR 벤치마크 실행 (10, 100 samples)
- [ ] GPU 메모리 프로파일링

### Phase 2: Feature Extraction
- [ ] Qwen2.5-32B base model 전체 데이터셋 처리
- [ ] DeepSeek-R1 base model 전체 데이터셋 처리
- [ ] Features 파일 생성 확인 (.npz)

### Phase 3: LOO Cross-Validation
- [ ] Qwen2.5 100-fold LOO CV 완료
- [ ] DeepSeek 100-fold LOO CV 완료
- [ ] NLL 결과 JSON 파일 생성

### Phase 4: 결과 분석
- [ ] 통합 리포트 생성
- [ ] Baseline 비교 분석
- [ ] GPU 성능 메트릭 정리

---

## 예상 실험 소요 시간

| Phase | 작업 | 예상 시간 |
|-------|------|----------|
| 1 | 환경 검증 및 벤치마킹 | 1-2시간 |
| 2 | Qwen2.5 feature extraction | 2-3시간 |
| 2 | DeepSeek feature extraction | 2-3시간 |
| 3 | Qwen2.5 LOO CV | 3-4시간 |
| 3 | DeepSeek LOO CV | 3-4시간 |
| 4 | 결과 분석 | 0.5시간 |
| **Total** | | **12-17시간** |

**권장 실행 방식**: tmux 세션에서 순차 실행, 각 단계별 로그 모니터링

---

## 트러블슈팅

### GPU 메모리 부족
```bash
# Quantization 강제 활성화
python scripts/extract_centaur_features.py --use_quantization

# Batch size 조정 (스크립트 내부 수정 필요)
```

### 처리 속도 느림
```bash
# GPU 사용 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 세션 타임아웃
```bash
# tmux 세션 사용 (필수)
tmux new -s centaur_exp
# Ctrl+b, d로 detach
# tmux attach -t centaur_exp로 재접속
```

---

## 다음 단계 (실험 완료 후)

1. **Fine-tuned 모델 평가** (adapter 사용)
   - Qwen2.5 + LoRA adapter
   - DeepSeek + LoRA adapter

2. **하이퍼파라미터 최적화**
   - Quantization 방식 비교 (NF4 vs INT8)
   - Batch size 최적화
   - Temperature 조정

3. **결과 논문 작성**
   - Binz & Schulz (2023) 재현 성공 여부
   - 한국어 LLM의 인지 모델링 성능 분석
   - GPU 가속화 효과 정량 분석

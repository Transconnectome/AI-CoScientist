# DGX Spark 환경 설정 가이드

NVIDIA DGX Spark에서 AI-CoScientist를 실행하기 위한 종합 가이드입니다.

## 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [사전 준비](#사전-준비)
3. [설치 및 설정](#설치-및-설정)
4. [GPU 최적화 설정](#gpu-최적화-설정)
5. [Docker 기반 배포](#docker-기반-배포)
6. [성능 튜닝](#성능-튜닝)
7. [문제 해결](#문제-해결)

---

## 시스템 요구사항

### DGX Spark 하드웨어 사양

| 구성 요소 | 사양 |
|-----------|------|
| **프로세서** | NVIDIA GB10 Grace Blackwell Superchip |
| **GPU** | Blackwell GPU (1000 AI TOPS) |
| **메모리** | 128GB 통합 메모리 |
| **저장장치** | 최대 4TB NVMe SSD |
| **연결** | USB-C, DisplayPort, Ethernet |

### 소프트웨어 요구사항

- **운영체제**: Ubuntu 22.04 LTS (NVIDIA 권장)
- **NVIDIA Driver**: 570+ (DGX Spark 번들 포함)
- **CUDA Toolkit**: 12.6+
- **Docker**: 24.0+ with NVIDIA Container Toolkit
- **Python**: 3.11+
- **Poetry**: 1.7+

---

## 사전 준비

### 1. NVIDIA 드라이버 확인

```bash
# GPU 상태 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version

# 예상 출력:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 570.xx.xx    Driver Version: 570.xx.xx    CUDA Version: 12.6    |
# +-----------------------------------------------------------------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA GB10         On   | 00000000:01:00.0 Off |                    0 |
# | N/A   45C    P8    25W / 300W |    512MiB / 131072MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### 2. NVIDIA Container Toolkit 설치

```bash
# GPG 키 추가
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 저장소 추가
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 데몬 재시작
sudo systemctl restart docker

# 설치 확인
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### 3. Docker Compose V2 확인

```bash
# Docker Compose 버전 확인
docker compose version

# 출력 예: Docker Compose version v2.24.0
```

---

## 설치 및 설정

### 1. 프로젝트 클론

```bash
git clone https://github.com/Transconnectome/AI-CoScientist.git
cd AI-CoScientist
```

### 2. Python 환경 설정

```bash
# Poetry 설치 (미설치 시)
curl -sSL https://install.python-poetry.org | python3 -

# 가상환경 생성 및 의존성 설치
poetry install

# PyTorch CUDA 지원 버전 설치 (DGX Spark 최적화)
poetry run pip install torch==2.5.1+cu126 torchvision==0.20.1+cu126 torchaudio==2.5.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
```

### 3. 환경 변수 설정

```bash
# DGX Spark 전용 환경 설정 파일 복사
cp .env.dgx-spark.example .env

# 필수 API 키 설정
nano .env  # 또는 선호하는 편집기 사용
```

### 4. 데이터베이스 초기화

```bash
# Alembic 마이그레이션 실행
poetry run alembic upgrade head
```

---

## GPU 최적화 설정

### PyTorch GPU 설정

```python
# src/core/gpu_config.py에서 자동으로 로드됨
import torch

# DGX Spark GPU 설정
def configure_gpu():
    """DGX Spark에 최적화된 GPU 설정"""
    if torch.cuda.is_available():
        # 기본 GPU 설정
        torch.cuda.set_device(0)

        # 메모리 관리 최적화
        torch.cuda.empty_cache()

        # TF32 활성화 (Blackwell 아키텍처 최적화)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # cuDNN 벤치마크 활성화
        torch.backends.cudnn.benchmark = True

        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### 환경 변수 (GPU 관련)

```bash
# CUDA 메모리 할당 설정
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Transformers 캐시 디렉토리
export TRANSFORMERS_CACHE="/path/to/cache/transformers"
export HF_HOME="/path/to/cache/huggingface"

# GPU 메모리 분할 (다중 모델 로드 시)
export CUDA_VISIBLE_DEVICES=0
```

---

## Docker 기반 배포

### DGX Spark 전용 Docker Compose 실행

```bash
# GPU 지원 Docker Compose 실행
docker compose -f docker-compose.dgx-spark.yml up -d

# 로그 확인
docker compose -f docker-compose.dgx-spark.yml logs -f api

# 상태 확인
docker compose -f docker-compose.dgx-spark.yml ps
```

### GPU 컨테이너 검증

```bash
# API 컨테이너에서 GPU 확인
docker exec ai-coscientist-api-gpu python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# GPU 메모리 사용량 모니터링
watch -n 1 nvidia-smi
```

---

## 성능 튜닝

### 1. 배치 크기 최적화

DGX Spark의 128GB 통합 메모리를 활용하여 큰 배치 크기 사용 가능:

```python
# 권장 배치 크기 설정
BATCH_SIZES = {
    "scibert_scorer": 32,      # 기본: 8
    "hybrid_scorer": 16,       # 기본: 4
    "multitask_scorer": 24,    # 기본: 8
    "embedding_generation": 64, # 기본: 16
}
```

### 2. 모델 로딩 최적화

```python
# 혼합 정밀도 추론 활성화
from torch.cuda.amp import autocast

with autocast():
    outputs = model(inputs)

# 모델 컴파일 (PyTorch 2.0+)
model = torch.compile(model, mode="max-autotune")
```

### 3. 메모리 관리

```bash
# .env 파일에 추가
GPU_MEMORY_FRACTION=0.95  # GPU 메모리 95% 사용
MAX_WORKERS=8             # 병렬 워커 수
BATCH_SIZE_AUTO=true      # 자동 배치 크기 조정
```

### 4. Transformer 모델 최적화

```python
# Flash Attention 2 활성화 (지원되는 모델에서)
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,  # BF16 정밀도 사용
    device_map="auto"
)
```

---

## 서비스 접속

### 엔드포인트

| 서비스 | URL | 설명 |
|--------|-----|------|
| **API** | http://localhost:8000 | FastAPI 메인 서버 |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **Health** | http://localhost:8000/health | 상태 확인 |
| **Prometheus** | http://localhost:9090 | 메트릭 수집 |
| **Grafana** | http://localhost:3001 | 시각화 대시보드 |
| **RabbitMQ** | http://localhost:15672 | 메시지 큐 관리 |

### 빠른 테스트

```bash
# API 헬스체크
curl http://localhost:8000/health

# GPU 정보 확인 (API 엔드포인트)
curl http://localhost:8000/api/v1/system/gpu-info

# 논문 평가 테스트
curl -X POST http://localhost:8000/api/v1/papers/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your paper text here..."}'
```

---

## 문제 해결

### 일반적인 문제

#### 1. CUDA Out of Memory

```bash
# 원인: GPU 메모리 부족
# 해결: 배치 크기 감소 또는 메모리 정리

# 메모리 캐시 정리
docker exec ai-coscientist-api-gpu python -c "import torch; torch.cuda.empty_cache()"

# 배치 크기 감소 (.env 수정)
BATCH_SIZE=8
```

#### 2. Docker GPU 접근 불가

```bash
# NVIDIA Container Toolkit 재설치
sudo apt-get install -y nvidia-container-toolkit --reinstall

# Docker 데몬 설정 확인
sudo nano /etc/docker/daemon.json

# 내용:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}

# Docker 재시작
sudo systemctl restart docker
```

#### 3. PyTorch CUDA 버전 불일치

```bash
# 현재 설치된 PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 올바른 버전 재설치
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

#### 4. 통합 메모리 활용 문제

```bash
# 통합 메모리 활성화 확인
nvidia-smi -q | grep "Unified Memory"

# PyTorch 통합 메모리 설정
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
```

### 로그 확인

```bash
# API 서버 로그
docker compose -f docker-compose.dgx-spark.yml logs -f api

# Celery 워커 로그
docker compose -f docker-compose.dgx-spark.yml logs -f celery-worker

# 전체 시스템 로그
docker compose -f docker-compose.dgx-spark.yml logs -f
```

---

## 부록: 권장 시스템 모니터링

### GPU 모니터링 대시보드

```bash
# NVIDIA DCGM 설치 (선택사항)
sudo apt-get install -y datacenter-gpu-manager

# GPU 모니터링 시작
dcgmi discovery -l

# Prometheus 연동을 위한 DCGM Exporter
docker run -d --gpus all --rm -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:latest
```

### 시스템 리소스 모니터링

```bash
# htop으로 CPU/메모리 모니터링
htop

# GPU 실시간 모니터링
watch -n 0.5 nvidia-smi

# nvtop 설치 및 실행 (GPU 전용 htop)
sudo apt install nvtop
nvtop
```

---

## 다음 단계

1. **API 키 설정**: `.env` 파일에 OpenAI/Anthropic API 키 추가
2. **테스트 실행**: `poetry run pytest` 로 테스트 스위트 실행
3. **논문 평가 시작**: `python scripts/evaluate_docx.py your_paper.docx`
4. **성능 벤치마크**: GPU vs CPU 성능 비교 실행

---

## 참고 자료

- [NVIDIA DGX Spark 공식 문서](https://www.nvidia.com/dgx-spark/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch CUDA 설정 가이드](https://pytorch.org/docs/stable/notes/cuda.html)
- [AI-CoScientist API 문서](./API_REFERENCE.md)

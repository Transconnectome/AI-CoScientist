# 🔧 Docker GPU Runtime 수정 가이드

**문제**: Connectome 서버의 모든 노드에 nvidia-container-toolkit이 설치되어 있지 않아 Docker가 GPU에 접근할 수 없습니다.

**해결**: nvidia-container-toolkit 설치 (5-10분 소요)

---

## ⚡ 빠른 수정 (권장)

### 1단계: 서버 접속
```bash
ssh server
```

### 2단계: 설치 스크립트 실행
```bash
cd ~
./install_nvidia_docker.sh
```

**참고**: sudo 비밀번호를 입력하라는 메시지가 여러 번 나타납니다. 비밀번호를 입력하세요.

### 3단계: 배포 재시작
설치가 성공하면 (✅ 메시지 확인):
```bash
cd ~/AI-CoScientist
sbatch deploy_slurm.sh
```

---

## 📋 수동 설치 (스크립트가 작동하지 않는 경우)

### 1. NVIDIA Container Toolkit 저장소 추가
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 2. 패키지 설치
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 3. Docker 설정
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4. 테스트
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

**성공하면**: NVIDIA GPU 정보가 표시됩니다
**실패하면**: 오류 메시지를 확인하고 관리자에게 문의하세요

---

## ✅ 설치 확인

설치가 성공했는지 확인:
```bash
# GPU 접근 테스트
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi

# 다음과 같은 출력이 나타나야 합니다:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 550.90.07    Driver Version: 550.90.07    CUDA Version: 12.4     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# ...
```

---

## 🚀 배포 재개

nvidia-container-toolkit 설치 후:

### 방법 1: SLURM 배치 작업 (권장)
```bash
cd ~/AI-CoScientist
sbatch deploy_slurm.sh

# 작업 상태 확인
squeue -u connectome1

# 로그 확인 (작업 번호는 sbatch 출력 참조)
tail -f ~/AI-CoScientist/logs/deploy_XXXXX.out
```

### 방법 2: 직접 실행 (테스트용)
```bash
cd ~/AI-CoScientist
./scripts/deploy_to_connectome_hybrid.sh
```

---

## ⏱️ 예상 배포 시간

nvidia-container-toolkit 설치 후 배포 단계:

| 단계 | 시간 | 설명 |
|------|------|------|
| Prerequisites | 1분 | GPU, Docker 확인 |
| Image Pull | 10-12분 | NIM 컨테이너 다운로드 (~10GB) |
| Infrastructure | 1-2분 | PostgreSQL, Redis, ChromaDB 시작 |
| Nemotron GPU | 3-5분 | 모델을 GPU에 로드 |
| Application | 1-2분 | API, Celery 시작 |
| Monitoring | 1분 | Prometheus, Grafana 시작 |
| **전체** | **17-23분** | 완전한 배포 |

---

## 🐛 문제 해결

### "sudo: a password is required"
**원인**: sudo 비밀번호가 필요합니다
**해결**: 비밀번호를 입력하세요

### "Permission denied"
**원인**: sudo 권한이 없습니다
**해결**: Connectome 관리자에게 연락하여 nvidia-container-toolkit 설치를 요청하세요

### "E: Could not get lock"
**원인**: 다른 apt 프로세스가 실행 중입니다
**해결**:
```bash
# 다른 apt 프로세스 확인
ps aux | grep apt

# 완료될 때까지 기다리거나 관리자에게 문의
```

### 설치 후에도 Docker GPU가 작동하지 않음
**원인**: Docker 재시작이 제대로 되지 않았을 수 있습니다
**해결**:
```bash
sudo systemctl status docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

---

## 📞 추가 지원 필요 시

nvidia-container-toolkit 설치가 계속 실패하는 경우:

1. **Connectome 관리자에게 연락**:
   - nvidia-container-toolkit 설치 요청
   - 모든 compute 노드(node1, node2, node3)에 설치 필요
   - Docker GPU runtime 활성화 요청

2. **제공할 정보**:
   - 오류 메시지: `~/AI-CoScientist/logs/deploy_*.err`
   - 시스템 정보: `uname -a && nvidia-smi`
   - Docker 버전: `docker --version`

---

## 🎯 요약

**현재 상태**: 모든 준비 완료, Docker GPU runtime만 필요
**필요 작업**: nvidia-container-toolkit 설치 (sudo 권한 필요)
**소요 시간**: 5-10분 (설치) + 17-23분 (배포)
**다음 단계**:
1. `ssh server`
2. `./install_nvidia_docker.sh` 실행 및 비밀번호 입력
3. `cd ~/AI-CoScientist && sbatch deploy_slurm.sh`

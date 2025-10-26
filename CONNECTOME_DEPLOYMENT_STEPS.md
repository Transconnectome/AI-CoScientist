# 🚀 Connectome 서버 배포 - 실행 단계

**현재 상태**: 로컬 MacBook에서 배포 파일 준비 완료 ✅
**목표**: Connectome 서버에 Hybrid AI-CoScientist 배포
**예상 시간**: 15-20분

---

## 사전 준비 확인

배포 전에 다음 정보가 필요합니다:

- [ ] Connectome 서버 주소: `connectome.server.address`
- [ ] SSH 사용자명: `your_username`
- [ ] Connectome에서 프로젝트 위치: `/path/to/AI-CoScientist`
- [ ] SSH 접속 가능 여부 확인

---

## 📋 단계별 배포 가이드

### Step 1: Connectome 서버 접속 확인

**로컬 터미널에서 실행**:
```bash
# SSH 접속 테스트
ssh your_username@connectome.server.address

# 접속 성공하면 다음 명령어로 환경 확인
nvidia-smi  # GPU 8개 확인
docker --version  # Docker 설치 확인
docker ps  # Docker 실행 확인

# 확인 후 로그아웃
exit
```

**확인 사항**:
- ✅ SSH 접속 성공
- ✅ nvidia-smi에서 8x RTX 3090 보임
- ✅ Docker가 설치되어 있음

---

### Step 2: 프로젝트 파일 Connectome으로 전송

**방법 A: Git으로 전송 (권장)**

```bash
# 1. 로컬에서 최신 변경사항 커밋
cd /Users/jiookcha/Documents/git/AI-CoScientist
git add .
git commit -m "Ready for Connectome deployment"
git push origin feature/nemotron-hybrid-integration

# 2. Connectome 서버에서 pull
ssh your_username@connectome.server.address
cd /path/to/AI-CoScientist  # 또는 git clone if needed
git pull origin feature/nemotron-hybrid-integration
```

**방법 B: rsync로 직접 전송**

```bash
# 로컬에서 전체 프로젝트 전송
rsync -avz --exclude='.git' --exclude='chromadb_data' --exclude='__pycache__' \
  /Users/jiookcha/Documents/git/AI-CoScientist/ \
  your_username@connectome.server.address:/path/to/AI-CoScientist/
```

---

### Step 3: 환경 설정 파일 전송

**중요**: `.env.local` 파일을 Connectome 서버로 복사

```bash
# 로컬 터미널에서 실행
scp /Users/jiookcha/Documents/git/AI-CoScientist/.env.local \
    your_username@connectome.server.address:/path/to/AI-CoScientist/.env.production

# 전송 확인
ssh your_username@connectome.server.address "cat /path/to/AI-CoScientist/.env.production | head -20"
```

**확인 사항**:
- ✅ NGC_API_KEY가 포함되어 있음
- ✅ OPENAI_API_KEY가 포함되어 있음
- ✅ GPU 설정이 올바름 (NEMOTRON_GPU_ID=1, NEMO_EMBEDDER_GPU_ID=5, NEMO_RERANKER_GPU_ID=6)

---

### Step 4: Connectome 서버에서 배포 실행

**Connectome 서버 터미널에서**:

```bash
# 1. Connectome에 SSH 접속
ssh your_username@connectome.server.address

# 2. 프로젝트 디렉토리로 이동
cd /path/to/AI-CoScientist

# 3. 배포 스크립트 실행 권한 부여
chmod +x scripts/deploy_to_connectome_hybrid.sh

# 4. 배포 시작 (10-15분 소요)
./scripts/deploy_to_connectome_hybrid.sh
```

**예상 출력**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI-CoScientist - Connectome Hybrid Deployment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/9] Checking prerequisites...
✓ GPU prerequisites verified (8x RTX 3090 found)
✓ Docker GPU runtime configured
✓ NGC API key configured

[2/9] Generating secure passwords...
✓ PostgreSQL password: ******************
✓ Redis password: ******************
✓ Grafana password: ******************
✓ Secret key: ******************

[3/9] Pulling Docker images...
⏳ Downloading NIM containers (10-15 minutes)...
(이 부분이 가장 오래 걸립니다 - 5.2GB + 2.1GB + 2.1GB)

[4/9] Starting infrastructure services...
✓ postgres (port 5432)
✓ redis (port 6379)
✓ chromadb (port 8003)

[5/9] Starting Nemotron GPU services...
⏳ Loading models (3-5 minutes)...
✓ nemotron-llm on GPU 1 (18GB VRAM)
✓ nemo-embedder on GPU 5 (4GB VRAM)
✓ nemo-reranker on GPU 6 (4GB VRAM)

[6/9] Starting application services...
✓ api (port 8080)
✓ celery-worker
✓ celery-beat

[7/9] Starting monitoring services...
✓ prometheus (port 9090)
✓ grafana (port 3000)

[8/9] Running health checks...
✓ All 10 services healthy

[9/9] Deployment summary...

Deployment successful! 🎉
```

---

### Step 5: 배포 검증

**Connectome 서버에서 실행**:

```bash
# 1. 모든 서비스 실행 확인
docker-compose -f docker-compose.connectome.yml ps

# 예상 출력: 11개 서비스 모두 "Up" 상태
# NAME                          STATUS    PORTS
# ai-coscientist-postgres       Up        0.0.0.0:5432->5432/tcp
# ai-coscientist-redis          Up        0.0.0.0:6379->6379/tcp
# ai-coscientist-chromadb       Up        0.0.0.0:8003->8000/tcp
# ai-coscientist-nemotron-llm   Up        0.0.0.0:8000->8000/tcp
# ai-coscientist-nemo-embedder  Up        0.0.0.0:8001->8000/tcp
# ai-coscientist-nemo-reranker  Up        0.0.0.0:8002->8000/tcp
# ai-coscientist-api            Up        0.0.0.0:8080->8080/tcp
# ai-coscientist-celery-worker  Up
# ai-coscientist-celery-beat    Up
# ai-coscientist-prometheus     Up        0.0.0.0:9090->9090/tcp
# ai-coscientist-grafana        Up        0.0.0.0:3000->3000/tcp

# 2. GPU 사용 확인
nvidia-smi

# 예상 출력: GPU 1, 5, 6에서 메모리 사용 중
# GPU 1: ~18GB (nemotron-llm)
# GPU 5: ~4GB (nemo-embedder)
# GPU 6: ~4GB (nemo-reranker)

# 3. API 헬스 체크
curl http://localhost:8080/api/v1/health

# 예상 출력:
# {"status":"healthy","timestamp":"2025-10-25T..."}

# 4. Hybrid RAG 상태 확인
curl http://localhost:8080/api/v1/hybrid-rag/status

# 예상 출력:
# {
#   "hybrid_mode": true,
#   "enabled_providers": ["gpt4", "nemotron"],
#   "nemotron_services": {
#     "llm": "http://nemotron-llm:8000/v1",
#     "embedder": "http://nemo-embedder:8000/v1",
#     "reranker": "http://nemo-reranker:8000/v1"
#   }
# }

# 5. Nemotron 서비스 헬스 체크
curl http://localhost:8000/v1/health  # Nemotron LLM
curl http://localhost:8001/v1/health  # NeMo Embedder
curl http://localhost:8002/v1/health  # NeMo Reranker
```

---

### Step 6: 테스트 Evaluation 실행

**실제 논문 평가 테스트**:

```bash
# Connectome 서버에서 실행
curl -X POST http://localhost:8080/api/v1/hybrid-rag/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "paper_text": "Recent advances in deep learning have revolutionized natural language processing. Our novel transformer architecture achieves state-of-the-art results on multiple benchmarks, demonstrating significant improvements in both accuracy and efficiency. We introduce a multi-head attention mechanism with sparse gating that reduces computational complexity while maintaining high performance.",
    "section": "abstract",
    "use_ensemble": true
  }'
```

**예상 응답 (2-3초 후)**:
```json
{
  "overall_quality": 8.14,
  "novelty": 7.9,
  "methodology": 8.44,
  "clarity": 8.24,
  "significance": 8.04,
  "feedback": "[GPT-4] Strong methodological approach...\n[Nemotron] Novel transformer architecture...",
  "provider_scores": {
    "gpt4": {
      "overall_quality": 8.3,
      "confidence": 0.92,
      "latency_ms": 1504
    },
    "nemotron": {
      "overall_quality": 7.9,
      "confidence": 0.78,
      "latency_ms": 234
    }
  },
  "ensemble_confidence": 0.86,
  "total_latency_ms": 1738
}
```

---

## 🎯 배포 성공 확인 체크리스트

- [ ] 11개 서비스 모두 "Up" 상태
- [ ] GPU 1, 5, 6에서 메모리 사용 확인 (~18GB, ~4GB, ~4GB)
- [ ] API health check 통과 (status: healthy)
- [ ] Hybrid RAG status에서 2개 provider 확인 (gpt4, nemotron)
- [ ] Nemotron 서비스 3개 모두 health check 통과
- [ ] 테스트 evaluation 성공 (overall_quality 8.0-8.5 범위)
- [ ] 응답 시간 2-3초 이내
- [ ] Ensemble confidence 0.8 이상

---

## 🛠️ 문제 해결

### 문제 1: Docker 이미지 다운로드 실패

**증상**: "Error pulling image" 또는 "timeout"

**해결**:
```bash
# NGC API key 확인
echo $NGC_API_KEY

# Docker login to NGC
echo $NGC_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin

# 수동으로 이미지 pull
docker pull nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:latest
```

### 문제 2: GPU 사용 불가

**증상**: "could not select device driver with capabilities: [[gpu]]"

**해결**:
```bash
# GPU runtime 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# nvidia-container-toolkit 설치 (필요시)
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 문제 3: Nemotron 서비스 시작 실패

**증상**: "nemotron-llm exited with code 1"

**해결**:
```bash
# 로그 확인
docker-compose -f docker-compose.connectome.yml logs nemotron-llm

# GPU 메모리 확인 (GPU 1에 최소 20GB 필요)
nvidia-smi

# 5분 대기 후 재시도 (모델 로딩 시간)
docker-compose -f docker-compose.connectome.yml restart nemotron-llm
```

### 문제 4: 포트 충돌

**증상**: "port is already allocated"

**해결**:
```bash
# 사용 중인 포트 확인
sudo lsof -i :8080  # API
sudo lsof -i :8000  # Nemotron

# 충돌하는 서비스 중지 또는
# .env.production에서 포트 변경
```

---

## 📊 배포 후 모니터링

### 실시간 GPU 모니터링
```bash
# GPU 사용량 1초마다 업데이트
watch -n 1 nvidia-smi
```

### 서비스 로그 확인
```bash
# API 로그
docker-compose -f docker-compose.connectome.yml logs -f api

# Nemotron 로그
docker-compose -f docker-compose.connectome.yml logs -f nemotron-llm

# 모든 로그
docker-compose -f docker-compose.connectome.yml logs -f
```

### Grafana 대시보드 접속
```
1. 브라우저에서 http://connectome.server.address:3000 열기
2. 로그인: admin / <.env.production의 GRAFANA_PASSWORD>
3. Dashboard → AI-CoScientist 선택
```

### Prometheus 메트릭
```bash
# API 메트릭 확인
curl http://localhost:8080/metrics

# Prometheus UI
# http://connectome.server.address:9090
```

---

## 🔐 보안: API 키 교체

**배포 완료 후 즉시 수행**:

1. **OpenAI API Key 교체**:
   - https://platform.openai.com/api-keys
   - 현재 키 삭제: `sk-proj-PWozG2qtPpmc...yH3oUOaEsA`
   - 새 키 생성
   - `.env.production`에서 `OPENAI_API_KEY` 업데이트

2. **NGC API Key 교체**:
   - https://org.ngc.nvidia.com/setup/api-key
   - 현재 키 삭제: `nvapi-8wfJi1Tt8ZTw7-...eqHDTqnojD`
   - 새 키 생성
   - `.env.production`에서 `NGC_API_KEY` 업데이트

3. **서비스 재시작**:
```bash
# API key 변경 후
docker-compose -f docker-compose.connectome.yml restart api celery-worker nemotron-llm nemo-embedder nemo-reranker
```

---

## ✅ 다음 단계

배포 성공 후:

1. **성능 테스트**:
   - 다양한 논문으로 evaluation 테스트
   - 응답 시간 및 품질 확인
   - GPU 사용량 모니터링

2. **자동 백업 설정**:
```bash
# 매일 자동 백업 cron job 설정
crontab -e

# 추가:
0 2 * * * cd /path/to/AI-CoScientist && docker-compose -f docker-compose.connectome.yml exec postgres pg_dump -U postgres ai_coscientist > /backups/db_$(date +\%Y\%m\%d).sql
```

3. **Grafana 대시보드 설정**:
   - GPU 사용량 그래프
   - API 응답 시간
   - Evaluation 품질 추이

4. **팀원 교육**:
   - API 사용법 공유
   - 모니터링 방법 안내

---

**도움이 필요하면**: 각 단계의 에러 메시지를 공유해주세요!

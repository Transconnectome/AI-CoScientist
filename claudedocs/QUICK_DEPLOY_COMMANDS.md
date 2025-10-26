# 🚀 Quick Deployment Commands - Copy & Paste

**For Connectome Server Deployment**

---

## 🔐 Step 1: SSH to Connectome

```bash
# Replace with your actual username and server address
ssh your_username@connectome.server.address
```

---

## 📂 Step 2: Navigate to Project Directory

```bash
cd ~/AI-CoScientist  # Adjust path if different

# Verify you're in the right place
pwd
ls -la scripts/deploy_to_connectome_hybrid.sh
```

---

## 🔄 Step 3: Pull Latest Code (If Using Git)

```bash
git fetch origin
git checkout feature/nemotron-hybrid-integration
git pull origin feature/nemotron-hybrid-integration

# Verify deployment script exists and is executable
ls -lh scripts/deploy_to_connectome_hybrid.sh
```

---

## 🔑 Step 4: Create .env.production File

**Option A: Transfer from local machine** (Run on your LOCAL machine)
```bash
scp /Users/jiookcha/Documents/git/AI-CoScientist/.env.local \
    your_username@connectome:/path/to/AI-CoScientist/.env.production
```

**Option B: Create directly on server** (Run on CONNECTOME server)
```bash
cat > .env.production << 'EOF'
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRITICAL API KEYS - ⚠️ ROTATE AFTER DEPLOYMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NGC_API_KEY=YOUR_NGC_API_KEY_HERE
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEMOTRON_GPU_ID=1
NEMO_EMBEDDER_GPU_ID=5
NEMO_RERANKER_GPU_ID=6

ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

HYBRID_MODE=true
USE_GPT4_FOR_EVALUATION=true
USE_CLAUDE_FOR_EVALUATION=false
USE_NEMOTRON_FOR_SUMMARIZATION=true
USE_NEMOTRON_FOR_EXTRACTION=true

ENSEMBLE_WEIGHT_GPT4=0.60
ENSEMBLE_WEIGHT_CLAUDE=0.0
ENSEMBLE_WEIGHT_NEMOTRON=0.40

NEMOTRON_CONFIDENCE_THRESHOLD=0.75

# OpenAI Configuration
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=4096

# Nemotron Configuration
NIM_OPTIMIZATION_PROFILE=throughput
NEMOTRON_BASE_URL=http://nemotron-llm:8000/v1
NEMOTRON_MODEL=nvidia/nvidia-nemotron-nano-9b-v2
NEMOTRON_TEMPERATURE=0.7
NEMOTRON_MAX_TOKENS=2048

NEMO_EMBEDDER_URL=http://nemo-embedder:8000/v1
NEMO_EMBEDDER_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2
EMBEDDING_DIMENSION=1024

NEMO_RERANKER_URL=http://nemo-reranker:8000/v1
NEMO_RERANKER_MODEL=nvidia/llama-3.2-nv-rerankqa-1b-v2
RERANKER_TOP_K=5

# Database
POSTGRES_USER=postgres
POSTGRES_DB=ai_coscientist
POSTGRES_PORT=5432

CHROMADB_HOST=chromadb
CHROMADB_PORT=8000
CHROMA_TELEMETRY=FALSE

REDIS_HOST=redis
REDIS_PORT=6379

# Application
APP_NAME=AI-CoScientist
APP_VERSION=1.0.0
API_PORT=8080

# Performance
UVICORN_WORKERS=4
CELERY_CONCURRENCY=4

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_USER=admin
GRAFANA_PORT=3000

# Paths
PAPERS_COLLECTION_DIR=./papers_collection
LOGS_DIR=./logs

CORS_ORIGINS=http://localhost,http://127.0.0.1
EOF

# Secure the file
chmod 600 .env.production

# Verify it was created
ls -lh .env.production
```

---

## ✅ Step 5: Pre-Deployment Checks

```bash
# Check GPUs available
nvidia-smi

# Verify Docker and Docker Compose installed
docker --version
docker-compose --version

# Test Docker GPU runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check disk space (need ~50GB)
df -h

# Verify .env.production exists and has API keys
grep "NGC_API_KEY" .env.production
grep "OPENAI_API_KEY" .env.production
```

---

## 🚀 Step 6: RUN DEPLOYMENT

```bash
# Make script executable (should already be)
chmod +x scripts/deploy_to_connectome_hybrid.sh

# Run deployment (takes 10-15 minutes)
./scripts/deploy_to_connectome_hybrid.sh
```

**What happens:**
1. Checks GPU prerequisites (~30s)
2. Generates secure passwords (~5s)
3. Pulls Docker images (~10 min)
4. Starts infrastructure (~60s)
5. Starts Nemotron GPU services (~3-5 min)
6. Runs database migrations (~30s)
7. Sets up monitoring (~20s)
8. Starts application services (~60s)
9. Verifies deployment (~30s)

---

## ✅ Step 7: Verify Deployment

```bash
# Check all services running
docker-compose -f docker-compose.connectome.yml ps

# Should show 11 services: postgres, redis, chromadb,
# nemotron-llm, nemo-embedder, nemo-reranker,
# api, celery-worker, celery-beat, prometheus, grafana

# Test API health
curl http://localhost:8080/api/v1/health

# Test Hybrid RAG status
curl http://localhost:8080/api/v1/hybrid-rag/status

# Monitor GPU usage
nvidia-smi -l 1
# Ctrl+C to stop

# Should see:
# GPU 1: ~18GB VRAM (nemotron-llm)
# GPU 5: ~4GB VRAM (nemo-embedder)
# GPU 6: ~4GB VRAM (nemo-reranker)
```

---

## 🧪 Step 8: Test Hybrid Evaluation

```bash
# Test hybrid evaluation endpoint
curl -X POST http://localhost:8080/api/v1/hybrid-rag/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "paper_text": "Recent advances in deep learning have revolutionized natural language processing. Our novel transformer architecture achieves state-of-the-art results on multiple benchmarks, demonstrating significant improvements in both accuracy and efficiency.",
    "section": "abstract",
    "use_ensemble": true
  }'

# Expected: JSON response with ensemble scores from GPT-4 and Nemotron
# Should complete in 2-3 seconds
```

---

## 🔐 Step 9: ROTATE API KEYS (CRITICAL)

**After successful deployment, immediately rotate all API keys:**

### Rotate NGC API Key
1. Visit: https://org.ngc.nvidia.com/setup/api-key
2. Generate new key
3. Update .env.production:
```bash
nano .env.production
# Change: NGC_API_KEY=YOUR_NGC_API_KEY_HERE
```

### Rotate OpenAI API Key
1. Visit: https://platform.openai.com/api-keys
2. Create new key, delete old one
3. Update .env.production:
```bash
nano .env.production
# Change: OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
```

### Restart Services After Key Rotation
```bash
docker-compose -f docker-compose.connectome.yml restart api celery-worker
```

---

## 📊 Step 10: Access Monitoring Dashboards

### Grafana
```
URL: http://localhost:3000
Username: admin
Password: <check .env.production>

# To view Grafana password:
grep GRAFANA_PASSWORD .env.production
```

### Prometheus
```
URL: http://localhost:9090
```

### API Documentation
```
URL: http://localhost:8080/docs
```

---

## 🛠️ Useful Management Commands

### View Logs
```bash
# All services
docker-compose -f docker-compose.connectome.yml logs -f

# Specific service
docker-compose -f docker-compose.connectome.yml logs -f api
docker-compose -f docker-compose.connectome.yml logs -f nemotron-llm
docker-compose -f docker-compose.connectome.yml logs -f celery-worker

# Last 100 lines
docker-compose -f docker-compose.connectome.yml logs --tail=100
```

### Restart Services
```bash
# Restart all
docker-compose -f docker-compose.connectome.yml restart

# Restart specific service
docker-compose -f docker-compose.connectome.yml restart api
docker-compose -f docker-compose.connectome.yml restart nemotron-llm
```

### Stop Services
```bash
# Stop all
docker-compose -f docker-compose.connectome.yml stop

# Stop specific service
docker-compose -f docker-compose.connectome.yml stop api
```

### Start Services
```bash
# Start all
docker-compose -f docker-compose.connectome.yml up -d

# Start specific service
docker-compose -f docker-compose.connectome.yml up -d api
```

### Check Service Status
```bash
# Quick status
docker-compose -f docker-compose.connectome.yml ps

# Detailed status
docker-compose -f docker-compose.connectome.yml ps -a

# Service health
docker inspect ai-coscientist-api | grep -A 10 Health
```

### Monitor Resources
```bash
# GPU usage (refresh every 1 second)
nvidia-smi -l 1

# GPU memory usage
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv

# Docker container stats
docker stats

# Disk space
df -h

# Paper collection size
du -sh papers_collection/
```

### Database Operations
```bash
# Backup database
docker-compose -f docker-compose.connectome.yml exec postgres \
  pg_dump -U postgres ai_coscientist > backup_$(date +%Y%m%d).sql

# Restore database
cat backup_20251026.sql | docker-compose -f docker-compose.connectome.yml exec -T postgres \
  psql -U postgres ai_coscientist

# Connect to database
docker-compose -f docker-compose.connectome.yml exec postgres \
  psql -U postgres -d ai_coscientist
```

### Clean Up
```bash
# Remove stopped containers
docker-compose -f docker-compose.connectome.yml rm

# Prune unused volumes (⚠️ CAREFUL - will delete data)
docker volume prune

# Clean up Docker system (⚠️ CAREFUL)
docker system prune -a
```

---

## 🐛 Troubleshooting Commands

### GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check which containers are using GPUs
nvidia-smi | grep -A 5 "Processes"

# Restart nvidia-docker if needed
sudo systemctl restart docker
```

### Port Conflicts
```bash
# Check what's using ports
sudo lsof -i :8080  # API port
sudo lsof -i :8000  # Nemotron LLM
sudo lsof -i :8001  # Embedder
sudo lsof -i :8002  # Reranker
sudo lsof -i :5432  # PostgreSQL
sudo lsof -i :6379  # Redis
sudo lsof -i :3000  # Grafana
sudo lsof -i :9090  # Prometheus

# Kill process using port
sudo kill -9 <PID>
```

### Network Issues
```bash
# Check Docker network
docker network ls
docker network inspect ai-coscientist_coscientist-network

# Test connectivity between containers
docker-compose -f docker-compose.connectome.yml exec api ping postgres
docker-compose -f docker-compose.connectome.yml exec api curl http://nemotron-llm:8000/v1/health
```

### Service Won't Start
```bash
# Check logs for errors
docker-compose -f docker-compose.connectome.yml logs <service-name>

# Rebuild specific service
docker-compose -f docker-compose.connectome.yml build <service-name>
docker-compose -f docker-compose.connectome.yml up -d <service-name>

# Force recreate
docker-compose -f docker-compose.connectome.yml up -d --force-recreate <service-name>
```

---

## 📝 Complete Teardown (If Needed)

```bash
# Stop all services
docker-compose -f docker-compose.connectome.yml down

# Stop and remove volumes (⚠️ DELETES ALL DATA)
docker-compose -f docker-compose.connectome.yml down -v

# Full cleanup
docker-compose -f docker-compose.connectome.yml down -v --rmi all
docker system prune -a --volumes
```

---

## ✅ Success Checklist

- [ ] All 11 services running
- [ ] API health endpoint returns healthy
- [ ] Hybrid RAG status shows GPU assignments
- [ ] nvidia-smi shows GPU 1, 5, 6 in use
- [ ] Test evaluation completes successfully
- [ ] Grafana accessible
- [ ] API keys rotated
- [ ] .env.production secured (chmod 600)
- [ ] Monitoring dashboards configured

---

## 📚 Documentation References

- **This Quick Guide**: `claudedocs/QUICK_DEPLOY_COMMANDS.md`
- **Pre-Deployment Checklist**: `claudedocs/PRE_DEPLOYMENT_CHECKLIST.md`
- **Full Deployment Guide**: `DEPLOY_TO_CONNECTOME_NOW.md`
- **Technical Details**: `claudedocs/NEMOTRON_HYBRID_GUIDE.md`
- **Deployment Summary**: `DEPLOYMENT_COMPLETE_SUMMARY.md`

---

**🎉 You're ready to deploy! Start with Step 1 and work through each step.**

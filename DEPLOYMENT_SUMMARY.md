# AI-CoScientist Deployment Summary

**Quick reference for Connectome server deployment (Option A: Backend First)**

## 📦 Deployment Package

**Created Files**:
- ✅ `docker-compose.yml` - 5-container orchestration (updated)
- ✅ `Dockerfile` - Production Python 3.11 image
- ✅ `.env.production.template` - Production configuration template
- ✅ `claudedocs/DEPLOYMENT_GUIDE.md` - Complete 1000+ line guide
- ✅ `scripts/deploy_to_connectome.sh` - Automated deployment script
- ✅ `scripts/backup_system.sh` - Automated backup script
- ✅ `.gitignore` - Updated to protect production secrets

## 🏗️ Architecture

**5 Containers** (simplified from original 9):
1. **PostgreSQL 16** - Database with literature_sources + monitoring_alerts tables
2. **Redis 7** - Cache + Celery broker (combined, removed RabbitMQ)
3. **FastAPI** - REST API (4 workers)
4. **Celery Worker** - Async paper downloads (4 threads)
5. **Celery Beat** - Scheduler for daily/weekly syncs

**Removed** (not needed for Phase 1A):
- ChromaDB (vector DB - not used in Phase 1A)
- RabbitMQ (replaced by Redis as broker)
- Prometheus + Grafana (monitoring - can add later)

**What Runs 24/7**:
- ArXiv syncs: Daily (Core ML, Comp Neuro) + Weekly (Med Imaging, AI4Science)
- PubMed syncs: Weekly (4 domains)
- Alert matching: Real-time when papers downloaded
- API server: Always available for frontend (Phase 5)

## 🚀 Quick Deployment (3 Commands)

### On Connectome Server:

```bash
# 1. Clone repository
git clone https://github.com/your-org/AI-CoScientist.git
cd AI-CoScientist
git checkout feature/phase1a-literature-monitoring

# 2. Run automated deployment (handles everything)
./scripts/deploy_to_connectome.sh

# 3. Verify health
curl http://localhost:8000/api/v1/health
```

**What the script does** (automatically):
1. ✅ Checks prerequisites (Docker, Docker Compose)
2. ✅ Creates `.env.production` with secure passwords
3. ✅ Prompts for your OpenAI API key (REQUIRED)
4. ✅ Creates directory structure for papers
5. ✅ Builds Docker images (~5-10 min)
6. ✅ Starts PostgreSQL + Redis
7. ✅ Runs database migrations
8. ✅ Sets up 8 sources + 4 alerts
9. ✅ Starts all 5 containers
10. ✅ Verifies deployment health

## 🔑 Required: OpenAI API Key

**CRITICAL**: The deployment script will pause and ask you to add your OpenAI API key to `.env.production`:

```bash
# Edit after script creates the file
nano .env.production

# Set this line (REQUIRED):
OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY_HERE
```

Without this, the strategic monitoring setup will fail.

## 📊 What Gets Created

**8 Literature Sources**:
1. Core ML (cs.LG, cs.AI, stat.ML) - ArXiv - Daily
2. Computational Neuroscience (q-bio.NC, cs.NE, q-bio.QM) - ArXiv - Daily
3. Medical Imaging + AI (cs.CV, eess.IV, physics.med-ph) - ArXiv - Weekly
4. AI for Science (cs.AI, cs.CL, cs.HC) - ArXiv - Weekly
5. Neuroimaging + ML - PubMed - Weekly
6. Computational Psychiatry - PubMed - Weekly
7. AI for Biomedical Research - PubMed - Weekly
8. Cognitive Neuroscience + Foundation Models - PubMed - Weekly

**4 Monitoring Alerts**:
1. Brain Decoding + Foundation Models (12 keywords) - Daily
2. AI for Scientific Discovery (11 keywords) - Daily
3. Computational Psychiatry (11 keywords) - Weekly
4. Multimodal Neuroimaging + AI (14 keywords) - Daily

**Target**: NeurIPS, ICLR, ICML, MICCAI conference papers

## 📁 Data Storage

```
AI-CoScientist/
├── papers_collection/          # Downloaded PDFs (grows over time)
│   ├── arxiv/
│   │   ├── 2024/              # Organized by year
│   │   └── 2025/
│   ├── pubmed/
│   └── conferences/
│       ├── neurips_2024/
│       ├── iclr_2025/
│       ├── icml_2025/
│       └── miccai_2024/
├── logs/                       # Application logs
└── backups/                    # Database & paper backups
```

**Expected Growth**: ~50-100 papers/day = ~1-2 GB/week

## 🔍 Verification

```bash
# 1. Check all containers running
docker-compose ps
# Should show 5 containers "Up"

# 2. API health check
curl http://localhost:8000/api/v1/health
# Response: {"status":"healthy","database":"connected","redis":"connected"}

# 3. Check literature sources (should be 8)
curl http://localhost:8000/api/v1/monitoring/sources | jq length
# Response: 8

# 4. Check monitoring alerts (should be 4)
curl http://localhost:8000/api/v1/monitoring/alerts | jq length
# Response: 4

# 5. Check Celery worker logs
docker-compose logs celery-worker | grep "ready"
# Should see: "celery@worker ready"

# 6. Trigger manual sync test
SOURCE_ID=$(curl -s http://localhost:8000/api/v1/monitoring/sources | jq -r '.[0].id')
curl -X POST http://localhost:8000/api/v1/monitoring/sources/$SOURCE_ID/sync
# Check worker logs for sync activity

# 7. Wait and check for downloaded papers (takes a few minutes)
ls -lh papers_collection/arxiv/2025/
# Should see .pdf files after first sync
```

## 🛠️ Common Operations

### View Logs
```bash
docker-compose logs -f              # All services
docker-compose logs -f api          # API only
docker-compose logs -f celery-worker  # Worker only
```

### Restart Services
```bash
docker-compose restart              # All services
docker-compose restart api          # API only
```

### Stop System
```bash
docker-compose stop                 # Preserves data
docker-compose down                 # Removes containers (data in volumes safe)
```

### Backup
```bash
# Full backup (database + papers + config + logs)
./scripts/backup_system.sh

# Database only
./scripts/backup_system.sh db

# Papers only
./scripts/backup_system.sh papers
```

### Update Code
```bash
git pull origin feature/phase1a-literature-monitoring
docker-compose up -d --build
```

## 📈 Monitoring

**Daily Checks**:
```bash
# 1. Disk space (papers accumulate!)
df -h papers_collection/

# 2. Sync status
curl http://localhost:8000/api/v1/monitoring/sources | jq '.[] | {category, last_sync_time, status}'

# 3. Error logs
docker-compose logs --since 1h | grep ERROR
```

**Weekly Checks**:
```bash
# 1. Paper count
find papers_collection/ -name "*.pdf" | wc -l

# 2. Database size
docker-compose exec postgres psql -U postgres -d ai_coscientist -c "SELECT pg_size_pretty(pg_database_size('ai_coscientist'));"

# 3. Backup verification
ls -lh backups/
```

## 🔒 Security Notes

**Protected Files** (never commit to git):
- `.env.production` - Contains API keys and passwords
- `backups/` - Database dumps and paper archives
- `*.sql`, `*.sql.gz` - Database exports

**Passwords Generated Automatically**:
- PostgreSQL password (32-char random)
- Redis password (32-char random)
- Secret key (64-char random)

**Only YOU Provide**:
- OpenAI API key (required)
- ArXiv API key (optional, for higher rate limits)
- PubMed API key (optional, for higher rate limits)

## 🎯 Success Criteria

After deployment, you should see:

✅ All 5 containers running and healthy
✅ API responding at http://localhost:8000
✅ 8 literature sources created
✅ 4 monitoring alerts configured
✅ Celery worker processing tasks
✅ Celery beat scheduling syncs
✅ Papers appearing in `papers_collection/` after first sync
✅ No ERROR logs in `docker-compose logs`

## 📚 Next Steps After Deployment

### Week 1-2: Stabilization Period
1. **Monitor daily** - Check disk space, sync status, error logs
2. **Review collected papers** - Verify quality and relevance
3. **Tune if needed** - Adjust keywords, frequencies, categories
4. **Set up automated backups** - Add cron job for daily backups

### After Stabilization: Phase 5 (Frontend)
1. **Develop React UI** (3-4 weeks estimated)
2. **Integrate with backend** (already compatible)
3. **Deploy frontend** alongside backend
4. **User acceptance testing**

## 🆘 Troubleshooting Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| Container won't start | `docker-compose logs SERVICE_NAME` |
| Database connection error | `docker-compose restart postgres` → wait 30s |
| Redis connection error | `docker-compose restart redis` |
| Celery not processing | `docker-compose restart celery-worker` |
| API not responding | `docker-compose restart api` |
| Disk full | Run `./scripts/backup_system.sh papers` → archive/delete old |
| Migration failed | Check logs → `docker-compose run --rm api alembic current` |
| No papers downloading | Check OpenAI API key in `.env.production` |

## 📞 Support

- **Full Guide**: `claudedocs/DEPLOYMENT_GUIDE.md` (comprehensive 1000+ lines)
- **Deployment Script**: `scripts/deploy_to_connectome.sh` (automated)
- **Backup Script**: `scripts/backup_system.sh` (automated)
- **Git Issues**: https://github.com/your-org/AI-CoScientist/issues

## 📊 System Requirements

**Minimum** (for testing):
- CPU: 2 cores
- RAM: 4 GB
- Disk: 50 GB (papers grow quickly!)
- Network: Stable internet for API calls

**Recommended** (for production):
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 500 GB+ (for long-term paper collection)
- Network: High-speed internet

**Estimated Costs** (monthly):
- OpenAI API: ~$50-100 (depends on usage)
- Server: Connectome (free for researchers)
- Storage: ~10-20 GB/month (papers)

---

**Deployment Method**: Option A (Backend First)
**Status**: Ready for Production
**Estimated Setup Time**: 15-30 minutes
**System Uptime**: 24/7 continuous operation
**Data Collection**: Starts immediately after deployment

# AI-CoScientist Deployment Guide

Complete guide for deploying AI-CoScientist backend to Connectome server for 24/7 literature monitoring.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Initial Setup](#initial-setup)
4. [Deployment Steps](#deployment-steps)
5. [Verification](#verification)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Backup & Recovery](#backup--recovery)

---

## Prerequisites

### Required Software
- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- SSH access to Connectome server

### Required Credentials
- OpenAI API key (REQUIRED)
- Connectome server SSH credentials
- Database passwords (generate securely)

### Recommended Knowledge
- Basic Docker/Docker Compose concepts
- Linux command line
- PostgreSQL basics
- Redis basics

---

## Architecture Overview

### Container Architecture
```
┌─────────────────────────────────────────────────────┐
│               Connectome Server                      │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │PostgreSQL│  │  Redis   │  │   FastAPI (4x)   │ │
│  │  :5432   │  │  :6379   │  │     :8000        │ │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       │             │                  │           │
│       │             │                  │           │
│  ┌────┴─────────────┴──────────────────┴─────┐    │
│  │         Celery Worker (4 threads)          │    │
│  │        (Paper Downloads & Processing)      │    │
│  └────────────────────┬───────────────────────┘    │
│                       │                             │
│  ┌────────────────────┴───────────────────────┐    │
│  │         Celery Beat (Scheduler)            │    │
│  │    (Daily/Weekly Sync Orchestration)       │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
│  Volumes:                                           │
│  • postgres_data (Database persistence)             │
│  • redis_data (Cache & queue persistence)           │
│  • ./papers_collection (Downloaded PDFs)            │
│  • ./logs (Application logs)                        │
└─────────────────────────────────────────────────────┘
```

### Data Flow
```
Celery Beat (Scheduler)
    ↓
Celery Task Queue (Redis)
    ↓
Celery Worker → ArXiv/PubMed APIs → Download PDFs → papers_collection/
    ↓
PostgreSQL (Metadata storage)
    ↓
FastAPI REST API
```

### Strategic Monitoring System
- **8 Literature Sources**: ArXiv (Core ML, Comp Neuro, Med Imaging, AI4Science) + PubMed (4 domains)
- **4 Strategic Alerts**: Brain decoding, AI4Science, Comp Psychiatry, Multimodal Neuroimaging
- **Target Conferences**: NeurIPS, ICLR, ICML, MICCAI papers

---

## Initial Setup

### 1. Connect to Connectome Server

```bash
# SSH into Connectome server
ssh your_username@connectome.snu.ac.kr

# Navigate to deployment directory
cd /path/to/deployment  # Choose appropriate path
```

### 2. Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/AI-CoScientist.git
cd AI-CoScientist

# Checkout the specific branch/tag if needed
git checkout feature/phase1a-literature-monitoring
```

### 3. Create Production Environment File

```bash
# Copy template
cp .env.production.template .env.production

# Edit with secure values
nano .env.production
```

**CRITICAL**: Fill in these values in `.env.production`:

```bash
# Generate secure passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
SECRET_KEY=$(openssl rand -hex 32)

# Add your OpenAI API key (REQUIRED!)
OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY_HERE

# Add optional API keys for better rate limits
ARXIV_API_KEY=your_arxiv_key_if_you_have_one
PUBMED_API_KEY=your_pubmed_key_if_you_have_one

# Update CORS origins for your frontend domain
CORS_ORIGINS=http://localhost:3000,http://connectome.snu.ac.kr
```

**Security Note**: Never commit `.env.production` to git! It's already in `.gitignore`.

### 4. Create Required Directories

```bash
# Create directories for persistent data
mkdir -p papers_collection/{arxiv/{2024,2025},pubmed,conferences/{neurips_2024,iclr_2025,icml_2025,miccai_2024},manual}
mkdir -p logs

# Set proper permissions
chmod -R 755 papers_collection logs
```

---

## Deployment Steps

### Step 1: Build Docker Images

```bash
# Build all images (takes 5-10 minutes first time)
docker-compose build

# Verify images were created
docker images | grep ai-coscientist
```

Expected output:
```
ai-coscientist-api           latest    <IMAGE_ID>    2 minutes ago    500MB
ai-coscientist-celery-worker latest    <IMAGE_ID>    2 minutes ago    500MB
ai-coscientist-celery-beat   latest    <IMAGE_ID>    2 minutes ago    500MB
```

### Step 2: Start Infrastructure Services

```bash
# Start PostgreSQL and Redis first
docker-compose up -d postgres redis

# Wait for health checks to pass (30-60 seconds)
docker-compose ps
```

Verify both services show "(healthy)":
```
NAME                         STATUS              PORTS
ai-coscientist-postgres      Up (healthy)        0.0.0.0:5432->5432/tcp
ai-coscientist-redis         Up (healthy)        0.0.0.0:6379->6379/tcp
```

### Step 3: Initialize Database

```bash
# Run Alembic migrations inside API container
docker-compose run --rm api alembic upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Running upgrade  -> a32a81c0d290, add literature monitoring tables
```

### Step 4: Setup Strategic Monitoring Configuration

```bash
# Run setup script to create 8 sources + 4 alerts
docker-compose run --rm api python scripts/setup_strategic_monitoring.py
```

Expected output:
```
✅ Created 8 literature sources
✅ Created 4 monitoring alerts
✅ Strategic monitoring system configured
```

### Step 5: Start All Services

```bash
# Start all containers
docker-compose up -d

# Verify all 5 containers are running
docker-compose ps
```

Expected output (all healthy):
```
NAME                              STATUS              PORTS
ai-coscientist-api                Up (healthy)        0.0.0.0:8000->8000/tcp
ai-coscientist-celery-beat        Up
ai-coscientist-celery-worker      Up
ai-coscientist-postgres           Up (healthy)        0.0.0.0:5432->5432/tcp
ai-coscientist-redis              Up (healthy)        0.0.0.0:6379->6379/tcp
```

---

## Verification

### 1. Health Check API

```bash
# Check API health endpoint
curl http://localhost:8000/api/v1/health

# Expected response
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "redis": "connected"
}
```

### 2. Verify Literature Sources

```bash
# Check literature sources
curl http://localhost:8000/api/v1/monitoring/sources | jq

# Should return 8 sources (4 ArXiv + 4 PubMed)
```

### 3. Verify Monitoring Alerts

```bash
# Check monitoring alerts
curl http://localhost:8000/api/v1/monitoring/alerts | jq

# Should return 4 strategic alerts
```

### 4. Check Celery Workers

```bash
# Check Celery worker logs
docker-compose logs -f celery-worker

# Should see:
# [INFO] celery@worker ready
# [INFO] Registered tasks: [sync_literature_source, process_paper_alert, ...]
```

### 5. Check Celery Beat Scheduler

```bash
# Check Celery beat logs
docker-compose logs -f celery-beat

# Should see scheduled tasks:
# [INFO] Scheduler: Sending due task sync-arxiv-core-ml
# [INFO] Scheduler: Sending due task sync-pubmed-neuroimaging
```

### 6. Manual Test Sync

```bash
# Trigger a manual sync for one source
SOURCE_ID=$(curl -s http://localhost:8000/api/v1/monitoring/sources | jq -r '.[0].id')
curl -X POST http://localhost:8000/api/v1/monitoring/sources/$SOURCE_ID/sync

# Check worker logs
docker-compose logs -f celery-worker
# Should see: [INFO] Syncing ArXiv source: Core ML (cs.LG,cs.AI,stat.ML)
```

### 7. Verify Paper Downloads

```bash
# After first successful sync (may take a few minutes)
ls -lh papers_collection/arxiv/2025/

# Should see PDF files
-rw-r--r-- 1 coscientist coscientist 2.3M Jan 15 10:23 2501.12345.pdf
-rw-r--r-- 1 coscientist coscientist 1.8M Jan 15 10:24 2501.12346.pdf
```

---

## Monitoring & Maintenance

### Real-time Monitoring

```bash
# View all logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f celery-worker
docker-compose logs -f celery-beat

# View container resource usage
docker stats

# View container health status
watch -n 5 'docker-compose ps'
```

### Daily Monitoring Tasks

1. **Check disk space** (papers accumulate quickly!)
   ```bash
   df -h /path/to/AI-CoScientist/papers_collection
   ```

2. **Check sync status**
   ```bash
   curl http://localhost:8000/api/v1/monitoring/sources | jq '.[] | {category, last_sync_time, status}'
   ```

3. **Check error logs**
   ```bash
   docker-compose logs --tail=100 | grep ERROR
   ```

### Weekly Monitoring Tasks

1. **Review collected papers count**
   ```bash
   find papers_collection/ -name "*.pdf" | wc -l
   ```

2. **Check database size**
   ```bash
   docker-compose exec postgres psql -U postgres -d ai_coscientist -c "SELECT pg_size_pretty(pg_database_size('ai_coscientist'));"
   ```

3. **Review alerts triggered**
   ```bash
   curl http://localhost:8000/api/v1/monitoring/alerts | jq '.[] | {topic, last_alert_sent, keywords}'
   ```

### Updating Configuration

#### Add New Literature Source

```bash
# Use API to add new source
curl -X POST http://localhost:8000/api/v1/monitoring/sources \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "arxiv",
    "category": "cs.RO,cs.AI",
    "sync_frequency": "daily",
    "status": "active"
  }'
```

#### Add New Alert

```bash
# Use API to add new alert
curl -X POST http://localhost:8000/api/v1/monitoring/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Robotics + Foundation Models",
    "keywords": ["embodied ai", "robot learning", "foundation models", "manipulation"],
    "frequency": "daily",
    "active": true
  }'
```

#### Update Sync Frequency

```bash
# Patch source to change frequency
SOURCE_ID="your-source-uuid"
curl -X PATCH http://localhost:8000/api/v1/monitoring/sources/$SOURCE_ID \
  -H "Content-Type: application/json" \
  -d '{"sync_frequency": "weekly"}'
```

### Restarting Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart api
docker-compose restart celery-worker

# Restart with rebuild (after code changes)
docker-compose up -d --build
```

### Stopping Services

```bash
# Stop all services (preserves data)
docker-compose stop

# Stop and remove containers (preserves volumes)
docker-compose down

# Stop and remove everything including volumes (DANGER!)
docker-compose down -v  # Only use for complete reset!
```

---

## Troubleshooting

### Issue: API Container Won't Start

**Symptoms:**
```
ai-coscientist-api    Exit 1
```

**Solution:**
```bash
# Check logs for error details
docker-compose logs api

# Common issues:
# 1. Database not ready → Wait 30s, restart: docker-compose restart api
# 2. Missing env vars → Check .env.production has all required values
# 3. Port conflict → Change API_PORT in .env.production
```

### Issue: Celery Worker Can't Connect to Redis

**Symptoms:**
```
[ERROR] Cannot connect to redis://redis:6379/0
```

**Solution:**
```bash
# Check Redis is running and healthy
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli ping
# Should return: PONG

# Check CELERY_BROKER_URL in .env.production matches Redis config
```

### Issue: Database Migration Fails

**Symptoms:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solution:**
```bash
# Ensure PostgreSQL is fully started
docker-compose logs postgres | grep "ready to accept connections"

# Wait 30 seconds after seeing "ready", then retry
docker-compose run --rm api alembic upgrade head
```

### Issue: Paper Downloads Failing

**Symptoms:**
```
[ERROR] Failed to download paper from ArXiv: HTTP 403
```

**Solutions:**
1. **Rate limiting**: Add ARXIV_API_KEY to .env.production for higher limits
2. **Network issues**: Check Connectome server internet connectivity
3. **Disk space**: Check `df -h` - papers_collection may be full

### Issue: Sync Tasks Not Running

**Symptoms:**
- No PDFs appearing in papers_collection/
- Celery beat shows no scheduled tasks

**Solution:**
```bash
# Check Celery beat is running
docker-compose ps celery-beat

# Check beat logs for schedule
docker-compose logs celery-beat | grep "Scheduler: Sending"

# Manually trigger sync to test
SOURCE_ID=$(curl -s http://localhost:8000/api/v1/monitoring/sources | jq -r '.[0].id')
curl -X POST http://localhost:8000/api/v1/monitoring/sources/$SOURCE_ID/sync

# Watch worker logs
docker-compose logs -f celery-worker
```

### Issue: High Disk Usage

**Symptoms:**
```
df -h shows papers_collection at 80%+
```

**Solution:**
```bash
# Find old papers (older than 6 months)
find papers_collection/ -name "*.pdf" -mtime +180

# Archive old papers (optional)
tar -czf papers_archive_2024.tar.gz papers_collection/arxiv/2024/
mv papers_archive_2024.tar.gz /archive/location/

# Delete archived papers
rm papers_collection/arxiv/2024/*.pdf

# Or implement retention policy in monitoring config
```

### Issue: Container Memory Issues

**Symptoms:**
```
docker stats shows high memory usage (>80%)
```

**Solution:**
```bash
# Add memory limits to docker-compose.yml
services:
  celery-worker:
    deploy:
      resources:
        limits:
          memory: 2G

# Reduce Celery worker concurrency in .env.production
CELERY_WORKER_CONCURRENCY=2  # Reduced from 4

# Restart services
docker-compose up -d
```

---

## Backup & Recovery

### Database Backup

```bash
# Create backup directory
mkdir -p backups/

# Backup PostgreSQL database
docker-compose exec postgres pg_dump -U postgres ai_coscientist | gzip > backups/db_backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Automate daily backups (crontab)
crontab -e
# Add: 0 2 * * * /path/to/backup_script.sh
```

### Restore Database

```bash
# Stop API and workers (to prevent conflicts)
docker-compose stop api celery-worker celery-beat

# Restore from backup
gunzip < backups/db_backup_20250115_020000.sql.gz | docker-compose exec -T postgres psql -U postgres ai_coscientist

# Restart services
docker-compose start api celery-worker celery-beat
```

### Backup Papers Collection

```bash
# Incremental backup of PDFs (only new/changed files)
rsync -avz --progress papers_collection/ /backup/location/papers_collection/

# Or full tar backup
tar -czf backups/papers_$(date +%Y%m%d).tar.gz papers_collection/
```

### Complete System Backup

```bash
# Backup everything: database, papers, config
mkdir -p backups/full_$(date +%Y%m%d)/

# 1. Database
docker-compose exec postgres pg_dump -U postgres ai_coscientist | gzip > backups/full_$(date +%Y%m%d)/database.sql.gz

# 2. Papers
rsync -avz papers_collection/ backups/full_$(date +%Y%m%d)/papers_collection/

# 3. Configuration
cp .env.production backups/full_$(date +%Y%m%d)/.env.production
cp docker-compose.yml backups/full_$(date +%Y%m%d)/docker-compose.yml

# 4. Logs (last 7 days)
find logs/ -name "*.log" -mtime -7 -exec cp {} backups/full_$(date +%Y%m%d)/ \;
```

### Disaster Recovery

```bash
# 1. Clone repository on new server
git clone https://github.com/your-org/AI-CoScientist.git
cd AI-CoScientist

# 2. Restore configuration
cp /backup/location/.env.production .env.production

# 3. Start infrastructure
docker-compose up -d postgres redis

# 4. Restore database
gunzip < /backup/location/database.sql.gz | docker-compose exec -T postgres psql -U postgres ai_coscientist

# 5. Restore papers
rsync -avz /backup/location/papers_collection/ papers_collection/

# 6. Start all services
docker-compose up -d

# 7. Verify health
curl http://localhost:8000/api/v1/health
```

---

## Production Best Practices

### Security

1. **Change default passwords**: Never use default passwords in production
2. **Enable Redis authentication**: Always set REDIS_PASSWORD
3. **Use HTTPS**: Configure reverse proxy (nginx) with SSL certificates
4. **Restrict API access**: Use firewall rules to limit access to known IPs
5. **Regular security updates**: Keep Docker images and system packages updated

### Performance

1. **Monitor resource usage**: Set up alerts for high CPU/memory/disk
2. **Scale workers**: Increase CELERY_WORKER_CONCURRENCY for higher throughput
3. **Database optimization**: Regular VACUUM and index maintenance
4. **Redis memory**: Monitor Redis memory usage, configure maxmemory-policy

### Reliability

1. **Health checks**: Enable all Docker healthcheck configurations
2. **Log rotation**: Configure log rotation to prevent disk fill
3. **Restart policies**: Use `restart: unless-stopped` for all services
4. **Regular backups**: Automate daily database and weekly full backups

### Monitoring

1. **Set up alerting**: Email/Slack notifications for failures
2. **Track sync metrics**: Monitor sync success rates and paper counts
3. **Log aggregation**: Use centralized logging (ELK stack or similar)
4. **Uptime monitoring**: External monitoring service to detect downtime

---

## Next Steps

After successful deployment:

1. **Monitor for 1-2 weeks**: Ensure stable operation and paper collection
2. **Review collected papers**: Verify quality and relevance of collected papers
3. **Tune alert keywords**: Refine monitoring alert keywords based on results
4. **Develop frontend** (Phase 5): React UI for browsing and managing collected papers
5. **Add features**: Paper summarization, similarity search, hypothesis generation

---

## Support & Contact

For issues or questions:
- Check logs: `docker-compose logs`
- Review troubleshooting section above
- Check git issues: https://github.com/your-org/AI-CoScientist/issues
- Contact: your-email@snu.ac.kr

---

## Appendix

### Quick Reference Commands

```bash
# Start system
docker-compose up -d

# Stop system
docker-compose stop

# View logs
docker-compose logs -f

# Restart service
docker-compose restart api

# Check health
curl http://localhost:8000/api/v1/health

# Backup database
docker-compose exec postgres pg_dump -U postgres ai_coscientist | gzip > backup.sql.gz

# Check disk usage
df -h papers_collection/

# Count papers
find papers_collection/ -name "*.pdf" | wc -l

# View scheduled tasks
docker-compose logs celery-beat | grep "Scheduler:"

# Manual sync
curl -X POST http://localhost:8000/api/v1/monitoring/sources/SOURCE_ID/sync
```

### Environment Variables Reference

See `.env.production.template` for complete list of supported environment variables.

Key variables:
- `POSTGRES_PASSWORD`: Database password (REQUIRED)
- `REDIS_PASSWORD`: Redis password (recommended)
- `OPENAI_API_KEY`: OpenAI API key (REQUIRED)
- `API_PORT`: API server port (default: 8000)
- `CELERY_WORKER_CONCURRENCY`: Worker threads (default: 4)
- `LOG_LEVEL`: Logging level (INFO/DEBUG/WARNING/ERROR)

### File Structure

```
AI-CoScientist/
├── docker-compose.yml           # Container orchestration
├── Dockerfile                   # Application container image
├── .env.production             # Production secrets (not in git!)
├── .env.production.template    # Template for production config
├── alembic/                    # Database migrations
├── src/                        # Application source code
├── papers_collection/          # Downloaded PDFs
│   ├── arxiv/
│   │   ├── 2024/
│   │   └── 2025/
│   ├── pubmed/
│   └── conferences/
├── logs/                       # Application logs
└── backups/                    # Database and paper backups
```

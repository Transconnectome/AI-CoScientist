# 🚀 AI-CoScientist Quick Start Guide

Complete setup guide to get AI-CoScientist running in minutes.

---

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry (Python package manager)
- 10GB free disk space

---

## ⚡ Quick Setup (Recommended)

### Step 1: Install Dependencies
```bash
./scripts/setup.sh
```

This script will:
- ✅ Check Python and Poetry
- ✅ Install all Python dependencies
- ✅ Verify .env configuration
- ✅ Create necessary directories
- ✅ Start Docker services
- ✅ Run database migrations

### Step 2: Verify Configuration

The `.env` file has been created with your API keys:
```bash
cat .env | grep API_KEY
```

You should see:
- ✅ OPENAI_API_KEY configured
- ✅ ANTHROPIC_API_KEY configured
- ✅ GEMINI_API_KEY configured

### Step 3: Start Services

**Terminal 1 - API Server:**
```bash
./scripts/run-api.sh
```

**Terminal 2 - Celery Worker (Optional for background tasks):**
```bash
./scripts/run-worker.sh
```

### Step 4: Test the API

Open your browser: **http://localhost:8000/docs**

Or test with curl:
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Should return: {"status":"healthy"}
```

---

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

### 1. Install Dependencies
```bash
poetry install
```

### 2. Start Docker Services
```bash
docker-compose up -d postgres redis chromadb
```

Wait for services to start (check with `docker-compose ps`)

### 3. Run Database Migrations
```bash
poetry run alembic upgrade head
```

### 4. Start API Server
```bash
poetry run python -m src.main
```

### 5. Start Celery Worker (Optional)
```bash
poetry run celery -A src.core.celery_app worker --loglevel=info
```

---

## 🧪 Testing the System

### 1. Health Check
```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "database": "connected",
    "redis": "connected",
    "vector_store": "ready"
  }
}
```

### 2. Create a Research Project
```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Research Project",
    "description": "Testing AI-CoScientist",
    "domain": "Computer Science",
    "research_question": "How does AI improve research efficiency?"
  }'
```

### 3. Search Literature
```bash
curl -X POST http://localhost:8000/api/v1/literature/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning research automation",
    "search_type": "hybrid",
    "top_k": 5
  }'
```

### 4. Generate Hypotheses
```bash
# First, get your project_id from step 2
PROJECT_ID="your-project-id-here"

curl -X POST "http://localhost:8000/api/v1/projects/${PROJECT_ID}/hypotheses/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "research_question": "How does AI improve research efficiency?",
    "num_hypotheses": 3,
    "creativity_level": 0.7
  }'
```

---

## 📊 Access Points

Once running, you can access:

| Service | URL | Description |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **ReDoc** | http://localhost:8000/redoc | Alternative API docs |
| **Health** | http://localhost:8000/api/v1/health | Health check endpoint |
| **Root** | http://localhost:8000/ | API information |

---

## 🐳 Docker Services

The system uses these Docker services:

| Service | Port | Purpose |
|---------|------|---------|
| **PostgreSQL** | 5432 | Main database |
| **Redis** | 6379 | Cache & task queue |
| **ChromaDB** | 8001 | Vector database |

Check service status:
```bash
docker-compose ps
```

View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f postgres
```

---

## 🔑 API Keys Configuration

Your API keys are already configured in `.env`:

```bash
# OpenAI (Primary LLM)
OPENAI_API_KEY=sk-proj-YpURN...

# Anthropic Claude (Fallback LLM)
ANTHROPIC_API_KEY=sk-ant-api03-0Y1ST...

# Google Gemini (Optional)
GEMINI_API_KEY=AIzaSyBYq...
```

The system will:
1. Use OpenAI GPT-4 as primary LLM
2. Fall back to Anthropic Claude if OpenAI fails
3. Cache LLM responses in Redis for efficiency

---

## 🔍 Troubleshooting

### Issue: "Connection refused" errors

**Solution:**
```bash
# Check if Docker services are running
docker-compose ps

# Restart services
docker-compose restart

# Or stop and start fresh
docker-compose down
docker-compose up -d
```

### Issue: Database migration errors

**Solution:**
```bash
# Reset database
docker-compose down -v
docker-compose up -d postgres

# Wait 5 seconds, then migrate
sleep 5
poetry run alembic upgrade head
```

### Issue: Port already in use

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in .env
# API_PORT=8001
```

### Issue: Import errors

**Solution:**
```bash
# Reinstall dependencies
poetry install --no-cache

# Or update packages
poetry update
```

### Issue: ChromaDB connection failed

**Solution:**
```bash
# Restart ChromaDB
docker-compose restart chromadb

# Check logs
docker-compose logs chromadb

# Ensure port 8001 is free
lsof -i :8001
```

---

## 📚 Complete Research Workflow Example

Here's a complete example workflow:

```python
import httpx
import json

base_url = "http://localhost:8000/api/v1"

# 1. Create Project
response = httpx.post(f"{base_url}/projects", json={
    "name": "Enzyme Kinetics Study",
    "description": "Temperature effects on enzyme activity",
    "domain": "Biochemistry",
    "research_question": "How does temperature affect enzyme catalytic efficiency?"
})
project = response.json()
print(f"✅ Project created: {project['id']}")

# 2. Ingest Literature
response = httpx.post(f"{base_url}/literature/ingest", json={
    "source_type": "query",
    "source_value": "enzyme kinetics temperature",
    "max_results": 20
})
print(f"✅ Literature ingested: {response.json()['count']} papers")

# 3. Search Literature
response = httpx.post(f"{base_url}/literature/search", json={
    "query": "enzyme activity temperature dependence",
    "search_type": "hybrid",
    "top_k": 10
})
papers = response.json()
print(f"✅ Found {len(papers)} relevant papers")

# 4. Generate Hypotheses
response = httpx.post(
    f"{base_url}/projects/{project['id']}/hypotheses/generate",
    json={
        "research_question": "How does temperature affect enzyme activity?",
        "num_hypotheses": 5,
        "creativity_level": 0.7
    }
)
hypotheses = response.json()
hypothesis_id = hypotheses["hypothesis_ids"][0]
print(f"✅ Generated {hypotheses['hypotheses_generated']} hypotheses")

# 5. Design Experiment
response = httpx.post(
    f"{base_url}/hypotheses/{hypothesis_id}/experiments/design",
    json={
        "research_question": "How does temperature affect enzyme activity?",
        "hypothesis_content": "Enzyme activity peaks at 37°C",
        "desired_power": 0.8,
        "significance_level": 0.05,
        "expected_effect_size": 0.6
    }
)
experiment = response.json()
print(f"✅ Experiment designed: {experiment['title']}")
print(f"   Sample size: {experiment['sample_size']} per group")
print(f"   Statistical power: {experiment['power']}")

# 6. Analyze Data (simulated data)
response = httpx.post(
    f"{base_url}/experiments/{experiment['experiment_id']}/analyze",
    json={
        "data": {
            "records": [
                {"group": "25C", "activity": 45.2, "temperature": 25},
                {"group": "25C", "activity": 43.8, "temperature": 25},
                {"group": "37C", "activity": 68.5, "temperature": 37},
                {"group": "37C", "activity": 70.1, "temperature": 37},
                {"group": "45C", "activity": 52.1, "temperature": 45},
                {"group": "45C", "activity": 51.3, "temperature": 45},
            ]
        },
        "analysis_types": ["descriptive", "inferential", "effect_size"],
        "visualization_types": ["distribution", "comparison"]
    }
)
analysis = response.json()
print(f"✅ Data analyzed")
print(f"   Interpretation: {analysis['overall_interpretation'][:200]}...")
print(f"   Visualizations: {len(analysis['visualizations'])} generated")

print("\n🎉 Complete research workflow executed successfully!")
```

---

## 🎯 Next Steps

Now that your system is running:

1. **Explore the API**: http://localhost:8000/docs
2. **Read the documentation**: See `PHASE3_COMPLETE.md` for features
3. **Run the example**: Copy the workflow example above
4. **Check integration tests**: `poetry run pytest tests/`
5. **Monitor logs**: `docker-compose logs -f`

---

## 🛑 Stopping Services

When you're done:

```bash
# Stop API server: Ctrl+C in the terminal

# Stop Celery worker: Ctrl+C in the terminal

# Stop Docker services
docker-compose down

# Stop and remove all data (⚠️ WARNING: Deletes all data!)
docker-compose down -v
```

---

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. View logs: `docker-compose logs`
3. Check service status: `docker-compose ps`
4. Review `.env` configuration
5. Ensure all prerequisites are installed

---

**🚀 You're ready to start conducting AI-assisted research!**

For detailed API documentation and advanced features, see:
- [PHASE2_COMPLETE.md](./PHASE2_COMPLETE.md) - Research Engine
- [PHASE3_COMPLETE.md](./PHASE3_COMPLETE.md) - Experiment Engine
- [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) - Full system overview

#!/bin/bash
#
# AI-CoScientist - Connectome Server Hybrid Deployment Script
#
# This script automates hybrid deployment to Connectome server:
# 1. Verifies GPU availability (8x RTX 3090)
# 2. Creates .env.production from hybrid template
# 3. Builds Docker images
# 4. Starts all 11 services (7 traditional + 4 Nemotron)
# 5. Runs database migrations
# 6. Sets up strategic monitoring
# 7. Verifies hybrid deployment health
#
# Usage:
#   ./scripts/deploy_to_connectome_hybrid.sh
#

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="AI-CoScientist Hybrid"
DEPLOYMENT_DATE=$(date +"%Y-%m-%d %H:%M:%S")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
COMPOSE_FILE="docker-compose.connectome.yml"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_gpu() {
    echo -e "${BLUE}[GPU]${NC} $1"
}

# Check GPU prerequisites
check_gpu_prerequisites() {
    log_gpu "Checking GPU availability on Connectome server..."

    # Check nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. GPU support not available."
        exit 1
    fi

    # Get GPU count and models
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

    log_gpu "Detected: ${GPU_COUNT}x ${GPU_MODEL} (${GPU_MEMORY}MB each)"

    # Verify minimum requirements
    if [ "$GPU_COUNT" -lt 3 ]; then
        log_error "Insufficient GPUs. Need at least 3 GPUs for Nemotron deployment."
        exit 1
    fi

    if [ "$GPU_MEMORY" -lt 20000 ]; then
        log_error "Insufficient GPU memory. Need at least 20GB per GPU (found ${GPU_MEMORY}MB)."
        exit 1
    fi

    # Check GPU utilization
    log_gpu "Checking GPU utilization..."
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv

    # Recommend GPU assignments
    log_gpu "Recommended GPU assignments for Connectome (8x RTX 3090):"
    log_gpu "  - GPU 1: Nemotron LLM (9B model, ~18GB VRAM)"
    log_gpu "  - GPU 5: NeMo Embedder (1B model, ~4GB VRAM)"
    log_gpu "  - GPU 6: NeMo Reranker (1B model, ~4GB VRAM)"
    echo

    # Verify Docker GPU runtime
    log_gpu "Verifying Docker GPU runtime..."
    if ! docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_error "Docker GPU runtime test failed. Please install nvidia-docker2."
        exit 1
    fi

    log_gpu "✓ GPU prerequisites check passed"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if running on Connectome server (optional check)
    if [[ ! "$HOSTNAME" =~ "connectome" ]] && [[ ! "$HOSTNAME" =~ "snu.ac.kr" ]] && [[ ! "$HOSTNAME" =~ "node" ]]; then
        log_warn "Not running on Connectome server (hostname: $HOSTNAME)"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    elif [[ "$HOSTNAME" =~ "node" ]]; then
        log_warn "Running on SLURM compute node (hostname: $HOSTNAME) - auto-continuing"
    fi

    log_info "✓ Prerequisites check passed"
}

# Create production environment file
create_env_file() {
    log_info "Creating .env.production file from hybrid template..."

    if [ -f .env.production ]; then
        log_warn ".env.production already exists"
        # In non-interactive mode (SLURM), always use existing file
        if [[ -z "$PS1" ]] || [[ "$HOSTNAME" =~ "node" ]]; then
            log_info "Using existing .env.production (non-interactive mode)"
            return
        fi
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Using existing .env.production"
            return
        fi
    fi

    # Copy hybrid template
    cp .env.connectome.hybrid.template .env.production

    # Generate secure passwords
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    REDIS_PASSWORD=$(openssl rand -base64 32)
    SECRET_KEY=$(openssl rand -hex 32)
    GRAFANA_PASSWORD=$(openssl rand -base64 24)

    # Update .env.production with generated passwords
    sed -i.bak "s/YOUR_SECURE_PASSWORD_HERE/$POSTGRES_PASSWORD/" .env.production
    sed -i.bak "s/YOUR_REDIS_PASSWORD_HERE/$REDIS_PASSWORD/" .env.production
    sed -i.bak "s/YOUR_SECURE_SECRET_KEY_HERE/$SECRET_KEY/" .env.production
    sed -i.bak "s/change-this-password/$GRAFANA_PASSWORD/" .env.production
    sed -i.bak "s/DEPLOYED_AT=/DEPLOYED_AT=$DEPLOYMENT_DATE/" .env.production
    sed -i.bak "s/GIT_COMMIT=/GIT_COMMIT=$GIT_COMMIT/" .env.production
    rm -f .env.production.bak

    log_info "✓ Generated secure passwords for PostgreSQL, Redis, and Grafana"

    log_warn "⚠️  IMPORTANT: You must add your API keys to .env.production"
    log_warn "   Required:"
    log_warn "     - NGC_API_KEY=<your_ngc_key> (from https://org.ngc.nvidia.com/setup/api-key)"
    log_warn "     - OPENAI_API_KEY=sk-proj-<your_key>"
    log_warn "   Optional:"
    log_warn "     - ANTHROPIC_API_KEY=sk-ant-<your_key>"
    echo
    read -p "Press Enter after adding your API keys to continue..."

    # Verify API keys were added
    if grep -q "YOUR_NGC_API_KEY_HERE" .env.production || \
       grep -q "NGC_API_KEY=$" .env.production; then
        log_error "NGC API key not found in .env.production"
        log_error "NGC_API_KEY is required for Nemotron services"
        exit 1
    fi

    if grep -q "YOUR_OPENAI_API_KEY_HERE" .env.production || \
       grep -q "OPENAI_API_KEY=$" .env.production; then
        log_error "OpenAI API key not found in .env.production"
        log_error "Please edit .env.production and add your API key, then run this script again"
        exit 1
    fi

    log_info "✓ .env.production created successfully"
}

# Create required directories
create_directories() {
    log_info "Creating required directories..."

    mkdir -p papers_collection/{arxiv/{2024,2025},pubmed,conferences/{neurips_2024,iclr_2025,icml_2025,miccai_2024},manual}
    mkdir -p logs
    mkdir -p monitoring/grafana/{dashboards,datasources}

    chmod -R 755 papers_collection logs monitoring

    log_info "✓ Directories created"
}

# Build Docker images
build_images() {
    log_info "Building Docker images (this may take 5-10 minutes)..."

    docker-compose -f "$COMPOSE_FILE" build

    log_info "✓ Docker images built successfully"
}

# Start infrastructure services
start_infrastructure() {
    log_info "Starting PostgreSQL, Redis, and ChromaDB..."

    docker-compose -f "$COMPOSE_FILE" up -d postgres redis chromadb

    # Wait for health checks
    log_info "Waiting for infrastructure services to be healthy (30-60 seconds)..."

    for i in {1..30}; do
        if docker-compose -f "$COMPOSE_FILE" ps | grep -q "(healthy).*postgres" && \
           docker-compose -f "$COMPOSE_FILE" ps | grep -q "(healthy).*redis" && \
           docker-compose -f "$COMPOSE_FILE" ps | grep -q "(healthy).*chromadb"; then
            log_info "✓ Infrastructure services are healthy"
            return
        fi
        echo -n "."
        sleep 2
    done

    log_error "Timeout waiting for infrastructure services to be healthy"
    docker-compose -f "$COMPOSE_FILE" ps
    exit 1
}

# Start Nemotron GPU services
start_nemotron_services() {
    log_gpu "Starting Nemotron GPU services (3-5 minutes for model loading)..."

    docker-compose -f "$COMPOSE_FILE" up -d nemotron-llm nemo-embedder nemo-reranker

    log_gpu "Waiting for Nemotron services to be healthy (180-300 seconds)..."
    log_gpu "This is normal - NIM containers need time to download and load models"

    for i in {1..150}; do
        HEALTHY_COUNT=0

        if docker-compose -f "$COMPOSE_FILE" ps nemotron-llm | grep -q "(healthy)"; then
            ((HEALTHY_COUNT++))
        fi
        if docker-compose -f "$COMPOSE_FILE" ps nemo-embedder | grep -q "(healthy)"; then
            ((HEALTHY_COUNT++))
        fi
        if docker-compose -f "$COMPOSE_FILE" ps nemo-reranker | grep -q "(healthy)"; then
            ((HEALTHY_COUNT++))
        fi

        if [ "$HEALTHY_COUNT" -eq 3 ]; then
            log_gpu "✓ All Nemotron services are healthy"
            return
        fi

        if [ $((i % 10)) -eq 0 ]; then
            log_gpu "Still waiting... ($i/150 attempts, $HEALTHY_COUNT/3 healthy)"
        fi
        echo -n "."
        sleep 2
    done

    log_warn "Nemotron services health check timeout"
    log_warn "Services may still be loading models. Check logs with:"
    log_warn "  docker-compose -f $COMPOSE_FILE logs nemotron-llm"
    log_warn "Continuing deployment..."
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."

    docker-compose -f "$COMPOSE_FILE" run --rm api alembic upgrade head

    log_info "✓ Database migrations completed"
}

# Setup strategic monitoring
setup_monitoring() {
    log_info "Setting up strategic monitoring (8 sources + 4 alerts)..."

    docker-compose -f "$COMPOSE_FILE" run --rm api python scripts/setup_strategic_monitoring.py

    log_info "✓ Strategic monitoring configured"
}

# Start all services
start_all_services() {
    log_info "Starting all remaining services..."

    docker-compose -f "$COMPOSE_FILE" up -d

    # Wait for API to be healthy
    log_info "Waiting for API to be healthy..."

    for i in {1..30}; do
        if curl -s -f http://localhost:8080/api/v1/health > /dev/null 2>&1; then
            log_info "✓ All services started successfully"
            return
        fi
        echo -n "."
        sleep 2
    done

    log_error "Timeout waiting for API to be healthy"
    docker-compose -f "$COMPOSE_FILE" logs --tail=50 api
    exit 1
}

# Verify deployment
verify_deployment() {
    log_info "Verifying hybrid deployment..."

    # 1. Check all containers are running
    log_info "Checking container status..."
    EXPECTED_CONTAINERS=11  # postgres, redis, chromadb, nemotron-llm, nemo-embedder, nemo-reranker, api, celery-worker, celery-beat, prometheus, grafana
    RUNNING_CONTAINERS=$(docker-compose -f "$COMPOSE_FILE" ps | grep "Up" | wc -l | tr -d ' ')

    if [ "$RUNNING_CONTAINERS" -lt "$EXPECTED_CONTAINERS" ]; then
        log_error "Not all containers are running ($RUNNING_CONTAINERS/$EXPECTED_CONTAINERS)"
        docker-compose -f "$COMPOSE_FILE" ps
        exit 1
    fi

    log_info "✓ All $EXPECTED_CONTAINERS containers are running"

    # 2. Check API health
    log_info "Checking API health..."
    HEALTH_RESPONSE=$(curl -s http://localhost:8080/api/v1/health)
    if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
        log_info "✓ API health check passed"
    else
        log_error "API health check failed: $HEALTH_RESPONSE"
        exit 1
    fi

    # 3. Check Nemotron services
    log_gpu "Checking Nemotron services..."

    # Nemotron LLM
    if curl -s -f http://localhost:8000/v1/health > /dev/null 2>&1; then
        log_gpu "✓ Nemotron LLM is healthy (GPU 1)"
    else
        log_warn "Nemotron LLM health check failed"
    fi

    # NeMo Embedder
    if curl -s -f http://localhost:8001/v1/health > /dev/null 2>&1; then
        log_gpu "✓ NeMo Embedder is healthy (GPU 5)"
    else
        log_warn "NeMo Embedder health check failed"
    fi

    # NeMo Reranker
    if curl -s -f http://localhost:8002/v1/health > /dev/null 2>&1; then
        log_gpu "✓ NeMo Reranker is healthy (GPU 6)"
    else
        log_warn "NeMo Reranker health check failed"
    fi

    # 4. Check ChromaDB
    log_info "Checking ChromaDB..."
    if curl -s -f http://localhost:8003/api/v1/heartbeat > /dev/null 2>&1; then
        log_info "✓ ChromaDB is healthy"
    else
        log_warn "ChromaDB may not be accessible"
    fi

    # 5. Check literature sources
    log_info "Checking literature sources..."
    SOURCES_COUNT=$(curl -s http://localhost:8080/api/v1/monitoring/sources | grep -o '"id"' | wc -l | tr -d ' ')
    if [ "$SOURCES_COUNT" -eq 8 ]; then
        log_info "✓ All 8 literature sources created"
    else
        log_warn "Expected 8 sources, found $SOURCES_COUNT"
    fi

    # 6. Check Celery workers
    log_info "Checking Celery workers..."
    CELERY_LOGS=$(docker-compose -f "$COMPOSE_FILE" logs --tail=20 celery-worker 2>&1)
    if echo "$CELERY_LOGS" | grep -q "ready"; then
        log_info "✓ Celery worker is ready"
    else
        log_warn "Celery worker may not be ready"
    fi

    # 7. Check Prometheus
    log_info "Checking Prometheus..."
    if curl -s -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        log_info "✓ Prometheus is healthy"
    else
        log_warn "Prometheus may not be accessible"
    fi

    # 8. Check Grafana
    log_info "Checking Grafana..."
    if curl -s http://localhost:3000/api/health | grep -q "ok"; then
        log_info "✓ Grafana is healthy"
    else
        log_warn "Grafana may not be accessible"
    fi

    log_info "✓ Deployment verification completed"
}

# Display deployment summary
display_summary() {
    echo
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║           $PROJECT_NAME Deployment Complete!                            ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo
    log_info "Deployment Information:"
    echo "  • Deployment Date: $DEPLOYMENT_DATE"
    echo "  • Git Commit: $GIT_COMMIT"
    echo "  • Configuration: Hybrid Mode (GPT-4 + Claude + Nemotron)"
    echo
    log_info "Service URLs:"
    echo "  • API: http://localhost:8080"
    echo "  • API Docs: http://localhost:8080/docs"
    echo "  • Health Check: http://localhost:8080/api/v1/health"
    echo "  • Nemotron LLM: http://localhost:8000/v1"
    echo "  • NeMo Embedder: http://localhost:8001/v1"
    echo "  • NeMo Reranker: http://localhost:8002/v1"
    echo "  • ChromaDB: http://localhost:8003"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3000 (admin / check .env.production)"
    echo
    log_gpu "GPU Assignments (Connectome 8x RTX 3090):"
    echo "  • GPU 1: Nemotron LLM (9B model, ~18GB VRAM)"
    echo "  • GPU 5: NeMo Embedder (1B model, ~4GB VRAM)"
    echo "  • GPU 6: NeMo Reranker (1B model, ~4GB VRAM)"
    echo "  • GPUs 0, 2, 3, 4, 7: Available for other workloads"
    echo
    log_info "Running Services (11 total):"
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    log_info "Hybrid Mode Configuration:"
    echo "  • Evaluation: GPT-4 (40%) + Claude (30%) + Nemotron (30%)"
    echo "  • Summarization: Nemotron (8x faster than GPT-4)"
    echo "  • Extraction: Nemotron (cost-effective)"
    echo "  • Retrieval: NeMo EmbedQA + RerankQA (25-40% better relevance)"
    echo "  • Cost Savings: 59% reduction ($350 → $143/month)"
    echo
    log_info "Useful Commands:"
    echo "  • View all logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "  • View API logs: docker-compose -f $COMPOSE_FILE logs -f api"
    echo "  • View Nemotron logs: docker-compose -f $COMPOSE_FILE logs -f nemotron-llm"
    echo "  • Check GPU usage: nvidia-smi -l 1"
    echo "  • Stop services: docker-compose -f $COMPOSE_FILE stop"
    echo "  • Restart services: docker-compose -f $COMPOSE_FILE restart"
    echo "  • View papers: ls -lh papers_collection/arxiv/2025/"
    echo
    log_info "Next Steps:"
    echo "  1. Access Grafana at http://localhost:3000 and log in"
    echo "  2. Monitor GPU utilization: nvidia-smi -l 1"
    echo "  3. Check Nemotron logs for model loading status"
    echo "  4. Test hybrid evaluation endpoint:"
    echo "     curl -X POST http://localhost:8080/api/v1/hybrid-rag/evaluate \\"
    echo "       -H 'Content-Type: application/json' \\"
    echo "       -d '{\"paper_text\":\"Test paper\", \"use_ensemble\":true}'"
    echo "  5. View RAG Evaluation dashboard for performance metrics"
    echo "  6. Monitor papers_collection/ for downloaded PDFs"
    echo "  7. Review NEMOTRON_HYBRID_GUIDE.md for usage examples"
    echo
    log_warn "Important Monitoring:"
    log_warn "  • GPU Memory: nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1"
    log_warn "  • Disk Space: df -h papers_collection/"
    log_warn "  • Container Health: docker-compose -f $COMPOSE_FILE ps"
    echo
}

# Main deployment flow
main() {
    echo
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║      $PROJECT_NAME - Connectome Deployment Script                       ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo

    check_prerequisites
    check_gpu_prerequisites
    create_env_file
    create_directories
    build_images
    start_infrastructure
    start_nemotron_services
    run_migrations
    setup_monitoring
    start_all_services
    verify_deployment
    display_summary

    log_info "🎉 Hybrid deployment completed successfully!"
    log_gpu "📊 Monitor GPU usage: nvidia-smi -l 1"
}

# Run main function
main

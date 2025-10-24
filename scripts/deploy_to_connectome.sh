#!/bin/bash
#
# AI-CoScientist - Connectome Server Deployment Script
#
# This script automates deployment to Connectome server:
# 1. Creates .env.production from template
# 2. Builds Docker images
# 3. Starts all services
# 4. Runs database migrations
# 5. Sets up strategic monitoring
# 6. Verifies deployment health
#
# Usage:
#   ./scripts/deploy_to_connectome.sh
#

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="AI-CoScientist"
DEPLOYMENT_DATE=$(date +"%Y-%m-%d %H:%M:%S")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

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
    if [[ ! "$HOSTNAME" =~ "connectome" ]] && [[ ! "$HOSTNAME" =~ "snu.ac.kr" ]]; then
        log_warn "Not running on Connectome server (hostname: $HOSTNAME)"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    log_info "✓ Prerequisites check passed"
}

# Create production environment file
create_env_file() {
    log_info "Creating .env.production file..."

    if [ -f .env.production ]; then
        log_warn ".env.production already exists"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Using existing .env.production"
            return
        fi
    fi

    # Copy template
    cp .env.production.template .env.production

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

    log_warn "⚠️  IMPORTANT: You must add your OpenAI API key to .env.production"
    log_warn "   Edit .env.production and set: OPENAI_API_KEY=sk-proj-YOUR_KEY"
    echo
    read -p "Press Enter after adding your API key to continue..."

    # Verify API key was added
    if grep -q "sk-proj-YOUR_OPENAI_API_KEY_HERE" .env.production || \
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

    docker-compose build

    log_info "✓ Docker images built successfully"
}

# Start infrastructure services
start_infrastructure() {
    log_info "Starting PostgreSQL, Redis, Prometheus, and Grafana..."

    docker-compose up -d postgres redis prometheus grafana

    # Wait for health checks
    log_info "Waiting for services to be healthy (30-60 seconds)..."

    for i in {1..30}; do
        if docker-compose ps | grep -q "(healthy).*postgres" && \
           docker-compose ps | grep -q "(healthy).*redis"; then
            log_info "✓ Infrastructure services are healthy"
            return
        fi
        echo -n "."
        sleep 2
    done

    log_error "Timeout waiting for services to be healthy"
    docker-compose ps
    exit 1
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."

    docker-compose run --rm api alembic upgrade head

    log_info "✓ Database migrations completed"
}

# Setup strategic monitoring
setup_monitoring() {
    log_info "Setting up strategic monitoring (8 sources + 4 alerts)..."

    docker-compose run --rm api python scripts/setup_strategic_monitoring.py

    log_info "✓ Strategic monitoring configured"
}

# Start all services
start_all_services() {
    log_info "Starting all services..."

    docker-compose up -d

    # Wait for API to be healthy
    log_info "Waiting for API to be healthy..."

    for i in {1..30}; do
        if curl -s -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            log_info "✓ All services started successfully"
            return
        fi
        echo -n "."
        sleep 2
    done

    log_error "Timeout waiting for API to be healthy"
    docker-compose logs --tail=50 api
    exit 1
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."

    # 1. Check all containers are running
    log_info "Checking container status..."
    EXPECTED_CONTAINERS=7  # postgres, redis, api, celery-worker, celery-beat, prometheus, grafana
    RUNNING_CONTAINERS=$(docker-compose ps | grep "Up" | wc -l | tr -d ' ')

    if [ "$RUNNING_CONTAINERS" -lt "$EXPECTED_CONTAINERS" ]; then
        log_error "Not all containers are running ($RUNNING_CONTAINERS/$EXPECTED_CONTAINERS)"
        docker-compose ps
        exit 1
    fi

    # 2. Check API health
    log_info "Checking API health..."
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/v1/health)
    if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
        log_info "✓ API health check passed"
    else
        log_error "API health check failed: $HEALTH_RESPONSE"
        exit 1
    fi

    # 3. Check literature sources
    log_info "Checking literature sources..."
    SOURCES_COUNT=$(curl -s http://localhost:8000/api/v1/monitoring/sources | grep -o '"id"' | wc -l | tr -d ' ')
    if [ "$SOURCES_COUNT" -eq 8 ]; then
        log_info "✓ All 8 literature sources created"
    else
        log_warn "Expected 8 sources, found $SOURCES_COUNT"
    fi

    # 4. Check monitoring alerts
    log_info "Checking monitoring alerts..."
    ALERTS_COUNT=$(curl -s http://localhost:8000/api/v1/monitoring/alerts | grep -o '"id"' | wc -l | tr -d ' ')
    if [ "$ALERTS_COUNT" -eq 4 ]; then
        log_info "✓ All 4 monitoring alerts created"
    else
        log_warn "Expected 4 alerts, found $ALERTS_COUNT"
    fi

    # 5. Check Celery workers
    log_info "Checking Celery workers..."
    CELERY_LOGS=$(docker-compose logs --tail=20 celery-worker 2>&1)
    if echo "$CELERY_LOGS" | grep -q "ready"; then
        log_info "✓ Celery worker is ready"
    else
        log_warn "Celery worker may not be ready"
    fi

    # 6. Check Celery beat
    log_info "Checking Celery beat scheduler..."
    BEAT_LOGS=$(docker-compose logs --tail=20 celery-beat 2>&1)
    if echo "$BEAT_LOGS" | grep -q "beat"; then
        log_info "✓ Celery beat is running"
    else
        log_warn "Celery beat may not be running"
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
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           $PROJECT_NAME Deployment Complete!                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo
    log_info "Deployment Information:"
    echo "  • Deployment Date: $DEPLOYMENT_DATE"
    echo "  • Git Commit: $GIT_COMMIT"
    echo "  • API URL: http://localhost:8000"
    echo "  • API Docs: http://localhost:8000/docs"
    echo "  • Health Check: http://localhost:8000/api/v1/health"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3000 (admin / check .env.production)"
    echo
    log_info "Running Services:"
    docker-compose ps
    echo
    log_info "Strategic Monitoring:"
    echo "  • Literature Sources: 8 (4 ArXiv + 4 PubMed)"
    echo "  • Monitoring Alerts: 4"
    echo "  • Sync Schedule: Daily (ArXiv Core ML, Comp Neuro) + Weekly (Others)"
    echo "  • Target Conferences: NeurIPS, ICLR, ICML, MICCAI"
    echo
    log_info "Useful Commands:"
    echo "  • View logs: docker-compose logs -f"
    echo "  • View API logs: docker-compose logs -f api"
    echo "  • View worker logs: docker-compose logs -f celery-worker"
    echo "  • View monitoring: docker-compose logs -f prometheus grafana"
    echo "  • Stop services: docker-compose stop"
    echo "  • Restart services: docker-compose restart"
    echo "  • View papers: ls -lh papers_collection/arxiv/2025/"
    echo
    log_info "Next Steps:"
    echo "  1. Access Grafana at http://localhost:3000 and log in"
    echo "  2. View RAG Evaluation dashboard for performance metrics"
    echo "  3. Monitor logs for first paper sync (may take a few minutes)"
    echo "  4. Check papers_collection/ for downloaded PDFs"
    echo "  5. Review DEPLOYMENT_GUIDE.md for monitoring and maintenance"
    echo "  6. Set up automated backups (see guide)"
    echo "  7. Configure firewall rules (see guide)"
    echo
    log_warn "Important: This system will now run 24/7 collecting papers"
    log_warn "Monitor disk space regularly: df -h papers_collection/"
    echo
}

# Main deployment flow
main() {
    echo
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║      $PROJECT_NAME - Connectome Deployment Script           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo

    check_prerequisites
    create_env_file
    create_directories
    build_images
    start_infrastructure
    run_migrations
    setup_monitoring
    start_all_services
    verify_deployment
    display_summary

    log_info "🎉 Deployment completed successfully!"
}

# Run main function
main

#!/bin/bash

# AI-CoScientist Start Script
# Starts all necessary services

set -e

echo "🚀 Starting AI-CoScientist"
echo "========================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please run ./scripts/setup.sh first"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d postgres redis chromadb
echo "   ✅ Docker services started"

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service health
echo ""
echo "🏥 Checking service health..."

# PostgreSQL
if docker-compose exec -T postgres pg_isready -U postgres &> /dev/null; then
    echo "   ✅ PostgreSQL is ready"
else
    echo "   ⚠️  PostgreSQL not ready yet, waiting..."
    sleep 5
fi

# Redis
if docker-compose exec -T redis redis-cli ping &> /dev/null; then
    echo "   ✅ Redis is ready"
else
    echo "   ⚠️  Redis not ready"
fi

# Run migrations
echo ""
echo "🗄️  Running database migrations..."
poetry run alembic upgrade head
echo "   ✅ Migrations completed"

echo ""
echo "✅ All services are ready!"
echo ""
echo "Starting application servers..."
echo ""
echo "To start the services, run the following commands in separate terminals:"
echo ""
echo "  Terminal 1 (API Server):"
echo "    poetry run python -m src.main"
echo ""
echo "  Terminal 2 (Celery Worker):"
echo "    poetry run celery -A src.core.celery_app worker --loglevel=info"
echo ""
echo "Or use the quick start commands:"
echo "  ./scripts/run-api.sh    # Start API server"
echo "  ./scripts/run-worker.sh # Start Celery worker"
echo ""

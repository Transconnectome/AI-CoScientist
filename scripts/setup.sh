#!/bin/bash

# AI-CoScientist Setup Script
# This script sets up the development environment

set -e

echo "🚀 AI-CoScientist Setup"
echo "======================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.11+ is required but not found"
    exit 1
fi

# Check Poetry
echo ""
echo "📌 Checking Poetry..."
if ! command -v poetry &> /dev/null; then
    echo "   Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "   ✅ Poetry installed"
else
    poetry_version=$(poetry --version 2>&1 | awk '{print $3}')
    echo "   Poetry version: $poetry_version"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
poetry install --no-interaction
echo "   ✅ Dependencies installed"

# Check .env file
echo ""
echo "🔧 Checking environment configuration..."
if [ ! -f .env ]; then
    echo "   ❌ .env file not found!"
    echo "   Creating .env from .env.example..."
    cp .env.example .env
    echo "   ⚠️  Please edit .env with your API keys"
else
    echo "   ✅ .env file exists"
fi

# Check Docker services
echo ""
echo "🐳 Checking Docker services..."
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        echo "   ✅ Docker is running"

        # Check if services are running
        if docker-compose ps | grep -q "postgres"; then
            echo "   ✅ PostgreSQL is running"
        else
            echo "   ⚠️  PostgreSQL is not running"
            echo "   Starting services..."
            docker-compose up -d postgres redis chromadb
        fi
    else
        echo "   ❌ Docker daemon is not running"
        echo "   Please start Docker and run: docker-compose up -d"
    fi
else
    echo "   ❌ Docker not found"
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
fi

# Database migrations
echo ""
echo "🗄️  Running database migrations..."
if docker ps | grep -q postgres; then
    sleep 2  # Wait for postgres to be ready
    poetry run alembic upgrade head 2>/dev/null && echo "   ✅ Database migrations completed" || echo "   ⚠️  Migration failed - database may not be ready"
else
    echo "   ⚠️  Skipping migrations - PostgreSQL not running"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p uploads logs
echo "   ✅ Directories created"

# Summary
echo ""
echo "✅ Setup Complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your API keys (if not already done)"
echo "  2. Start Docker services: docker-compose up -d"
echo "  3. Run database migrations: poetry run alembic upgrade head"
echo "  4. Start the API server: poetry run python -m src.main"
echo "  5. Start Celery worker: poetry run celery -A src.core.celery_app worker -l info"
echo ""
echo "Access points:"
echo "  - API Documentation: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/api/v1/health"
echo ""

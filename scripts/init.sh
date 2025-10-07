#!/bin/bash

# AI CoScientist Initialization Script

set -e

echo "🚀 Initializing AI CoScientist..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env file"
    echo -e "${YELLOW}⚠️  Please edit .env file with your API keys before continuing${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗${NC} Docker is not running. Please start Docker first."
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker is running"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}⚠️  Poetry not found. Installing Poetry...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

echo -e "${GREEN}✓${NC} Poetry is installed"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install

echo -e "${GREEN}✓${NC} Dependencies installed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads logs data

echo -e "${GREEN}✓${NC} Directories created"

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d postgres redis rabbitmq chromadb

echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."

# Check PostgreSQL
until docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done
echo -e "${GREEN}✓${NC} PostgreSQL is ready"

# Check Redis
until docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do
    echo "Waiting for Redis..."
    sleep 2
done
echo -e "${GREEN}✓${NC} Redis is ready"

echo ""
echo -e "${GREEN}✅ AI CoScientist initialized successfully!${NC}"
echo ""
echo "📚 Next steps:"
echo "  1. Review and update .env file with your API keys"
echo "  2. Start the API server:"
echo "     ${YELLOW}poetry run uvicorn src.main:app --reload${NC}"
echo "  3. Or use Docker Compose:"
echo "     ${YELLOW}docker-compose up${NC}"
echo ""
echo "📖 Documentation:"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/api/v1/health"
echo "  - README: ./README.md"
echo ""

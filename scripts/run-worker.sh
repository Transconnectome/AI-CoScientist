#!/bin/bash

# Start Celery Worker

echo "⚡ Starting AI-CoScientist Celery Worker..."
echo ""

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Start Celery worker
poetry run celery -A src.core.celery_app worker --loglevel=info

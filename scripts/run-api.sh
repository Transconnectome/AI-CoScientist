#!/bin/bash

# Start API Server

echo "🚀 Starting AI-CoScientist API Server..."
echo ""

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Start API server
poetry run python -m src.main

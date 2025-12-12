#!/bin/bash

# RAG Enhancement Workflow Starter Script
# This script initializes the workflow environment and guides through the first steps

set -e

echo "🚀 AI-CoScientist RAG Enhancement Workflow"
echo "=========================================="
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.11+ required but not found"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Check Poetry
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry required but not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "✅ Poetry installed"
else
    poetry_version=$(poetry --version 2>&1 | awk '{print $3}')
    echo "✅ Poetry version: $poetry_version"
fi

# Check Git
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi
echo "✅ Git repository detected"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
poetry install --with dev
echo "✅ Dependencies installed"

# Setup pre-commit hooks
echo ""
echo "🔧 Setting up pre-commit hooks..."
poetry run pre-commit install
echo "✅ Pre-commit hooks installed"

# Install workflow automation dependencies
echo ""
echo "🛠️ Installing workflow automation dependencies..."
poetry add click rich pyyaml
echo "✅ Workflow dependencies installed"

# Make scripts executable
chmod +x scripts/workflow_automation.py

# Create workspace structure
echo ""
echo "📁 Creating workspace structure..."

# Create directories for Phase 1 tasks
mkdir -p src/services/rag/{unified,evaluation,adaptive}
mkdir -p tests/rag/{unit,integration,performance}
mkdir -p data/evaluation
mkdir -p scripts/benchmarking
mkdir -p src/monitoring

# Create basic files with TODOs
echo "✅ Workspace structure created"

# Initialize workflow tracking
echo ""
echo "⚙️ Initializing workflow tracking..."

# Create workflow progress file
cat > workflow_progress.json << 'EOF'
{
  "last_updated": null,
  "current_phase": "phase1",
  "current_sprint": "sprint1_1",
  "tasks": {}
}
EOF

echo "✅ Workflow tracking initialized"

# Display status
echo ""
echo "📊 Current workflow status:"
python3 scripts/workflow_automation.py status

echo ""
echo "🎯 Next steps:"
echo "  1. Review the workflow plan: cat RAG_ENHANCEMENT_WORKFLOW.md"
echo "  2. Check task details: cat workflow_config.yaml"
echo "  3. Start first task: python3 scripts/workflow_automation.py start ragas_integration"
echo "  4. Monitor progress: python3 scripts/workflow_automation.py status"
echo ""

# Offer to start first task
read -p "Would you like to start the first task (RAGAS integration)? (y/N): " start_task
if [[ $start_task =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting RAGAS integration task..."
    python3 scripts/workflow_automation.py start ragas_integration

    echo ""
    echo "📝 Task started! Key files created:"
    echo "  - src/services/rag/rag_evaluator.py (implementation file)"
    echo "  - Check acceptance criteria in the file comments"
    echo ""
    echo "💡 Tips:"
    echo "  - Follow TDD: write tests first in tests/rag/test_rag_evaluation.py"
    echo "  - Use existing ChromaDB integration patterns"
    echo "  - Reference RAGAS documentation: https://docs.ragas.io/"
    echo ""
    echo "✅ When complete, run: python3 scripts/workflow_automation.py complete ragas_integration"
fi

echo ""
echo "🔗 Useful commands:"
echo "  - View all tasks: python3 scripts/workflow_automation.py status"
echo "  - Start a task: python3 scripts/workflow_automation.py start <task_id>"
echo "  - Complete a task: python3 scripts/workflow_automation.py complete <task_id>"
echo "  - Validate sprint: python3 scripts/workflow_automation.py validate_sprint sprint1_1"
echo "  - Generate report: python3 scripts/workflow_automation.py generate_report"
echo ""

echo "🎉 Workflow environment ready! Happy coding!"
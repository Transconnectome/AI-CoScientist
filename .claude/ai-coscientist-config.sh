#!/bin/bash
# AI-CoScientist System Configuration
# Source this file to enable AI-CoScientist functionality in Claude Code

# Core System Paths
export AI_COSCIENTIST_HOME="/home/juke/git/AI-CoScientist"
export AI_COSCIENTIST_DATA="$AI_COSCIENTIST_HOME/data"
export AI_COSCIENTIST_SCRIPTS="$AI_COSCIENTIST_HOME/scripts"

# Python Environment
export PYTHONPATH="$AI_COSCIENTIST_HOME:$PYTHONPATH"

# API Configuration
export AI_COSCIENTIST_API_HOST="localhost"
export AI_COSCIENTIST_API_PORT="8000"
export AI_COSCIENTIST_API_BASE="http://$AI_COSCIENTIST_API_HOST:$AI_COSCIENTIST_API_PORT"

# Database Configuration
export AI_COSCIENTIST_DB_PATH="$AI_COSCIENTIST_HOME/ai_coscientist.db"
export AI_COSCIENTIST_CHROMADB="$AI_COSCIENTIST_HOME/chromadb_data"

# Specialized Data Paths
export DD_RAPTOR_DATA="$AI_COSCIENTIST_DATA/발달장애"
export QUANTERA_DATA="$AI_COSCIENTIST_DATA/QuantERA"
export RAG_EVALUATION_DATA="$AI_COSCIENTIST_DATA/validation"

# Agent System Aliases
alias acs-paper-review="python $AI_COSCIENTIST_SCRIPTS/chat_reviewer_enhanced.py"
alias acs-grant-analyze="python $AI_COSCIENTIST_SCRIPTS/analyze_grant_structure.py"
alias acs-rag-query="python $AI_COSCIENTIST_SCRIPTS/query_dd_rag.py"
alias acs-literature-review="python $AI_COSCIENTIST_SCRIPTS/analyze_dd_literature.py"
alias acs-api-start="cd $AI_COSCIENTIST_HOME && uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"

# Quick Access Functions
acs_status() {
    echo "🤖 AI-CoScientist System Status"
    echo "================================="
    echo "📁 Home: $AI_COSCIENTIST_HOME"
    echo "🔗 API: $AI_COSCIENTIST_API_BASE"
    echo "🗄️ Database: $AI_COSCIENTIST_DB_PATH"
    echo "📚 ChromaDB: $AI_COSCIENTIST_CHROMADB"
    echo ""
    echo "Available Commands:"
    echo "  acs-paper-review    - Interactive paper reviewer"
    echo "  acs-grant-analyze   - Grant structure analysis"
    echo "  acs-rag-query       - RAG system queries"
    echo "  acs-literature-review - Literature analysis"
    echo "  acs-api-start       - Start FastAPI server"
}

acs_agents() {
    echo "🤖 Available AI-CoScientist Agents:"
    echo "===================================="
    echo "📊 Statistical Analyst    - Methodology validation"
    echo "✍️ Grant Writer          - Proposal optimization"
    echo "🧠 Neuroscience Expert   - Domain expertise"
    echo "💡 Hypothesis Generator  - Research questions"
    echo "🏥 Clinical Validator    - Medical validation"
    echo "📚 Literature Analyst    - Systematic reviews"
    echo ""
    echo "Multi-agent coordination via Agent Pool system"
}

acs_rag() {
    echo "🔍 AI-CoScientist RAG Capabilities:"
    echo "===================================="
    echo "📖 DD-RAPTOR System     - 26 developmental disorder papers"
    echo "⚛️ QuantERA Database    - Quantum ML research corpus"
    echo "🧪 Hybrid RAG Service   - 6 specialized strategies"
    echo "📊 Performance Metrics  - RAGAS evaluation framework"
    echo ""
    echo "Query: acs-rag-query \"your research question\""
}

# Initialize message
echo "✅ AI-CoScientist System Configured"
echo "   Run 'acs_status' for system overview"
echo "   Run 'acs_agents' for available agents"
echo "   Run 'acs_rag' for RAG capabilities"
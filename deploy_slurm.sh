#!/bin/bash
#SBATCH --job-name=ai-coscientist-deploy
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:3
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/connectome/connectome1/AI-CoScientist/logs/deploy_%j.out
#SBATCH --error=/home/connectome/connectome1/AI-CoScientist/logs/deploy_%j.err

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI-CoScientist - SLURM Deployment Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# This script deploys AI-CoScientist using SLURM on Connectome server
# Target: node1 with 3x RTX A5000 GPUs (GPU 0, 2, 3)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║          AI-CoScientist SLURM Deployment Starting                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Change to project directory
cd /home/connectome/connectome1/AI-CoScientist

# Create logs directory if it doesn't exist
mkdir -p logs

# Display environment information
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Environment Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $SLURM_NODELIST"
echo "CPUs Allocated: $SLURM_CPUS_PER_TASK"
echo "Memory Allocated: ${SLURM_MEM_PER_NODE}MB"
echo ""

# Display GPU information
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GPU Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Check if .env.production exists
if [ ! -f .env.production ]; then
    echo "ERROR: .env.production not found!"
    exit 1
fi

echo "✓ .env.production found"
echo ""

# Verify deployment script exists
if [ ! -f scripts/deploy_to_connectome_hybrid.sh ]; then
    echo "ERROR: Deployment script not found!"
    exit 1
fi

echo "✓ Deployment script found"
echo ""

# Make deployment script executable
chmod +x scripts/deploy_to_connectome_hybrid.sh

# Display disk space
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Disk Space"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
df -h $HOME
echo ""

# Run deployment script
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

./scripts/deploy_to_connectome_hybrid.sh

DEPLOY_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Deployment Completed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Exit Code: $DEPLOY_EXIT_CODE"
echo "End Time: $(date)"
echo ""

# Display final GPU status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Final GPU Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi
echo ""

# Display running containers
if command -v docker &> /dev/null; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Docker Containers Status"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    docker-compose -f docker-compose.connectome.yml ps
    echo ""
fi

if [ $DEPLOY_EXIT_CODE -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║                    Deployment Successful! 🎉                             ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next Steps:"
    echo "1. Verify services: docker-compose -f docker-compose.connectome.yml ps"
    echo "2. Check API health: curl http://localhost:8080/api/v1/health"
    echo "3. Monitor GPUs: nvidia-smi -l 1"
    echo "4. ⚠️  IMPORTANT: Rotate all API keys immediately!"
    echo ""
else
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║                    Deployment Failed! ❌                                 ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Check logs at:"
    echo "  - SLURM output: logs/deploy_${SLURM_JOB_ID}.out"
    echo "  - SLURM errors: logs/deploy_${SLURM_JOB_ID}.err"
    echo "  - Docker logs: docker-compose -f docker-compose.connectome.yml logs"
    echo ""
fi

exit $DEPLOY_EXIT_CODE

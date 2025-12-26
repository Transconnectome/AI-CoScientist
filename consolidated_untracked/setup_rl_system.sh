#!/bin/bash

# AI-CoScientist RL System Setup Script
# Sets up the complete RL-enhanced agent selection system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RL_MODEL_DIR="${PROJECT_ROOT}/models/rl_agent_selection"
METRICS_DIR="${PROJECT_ROOT}/storage/metrics"
BACKUP_DIR="${PROJECT_ROOT}/backups"

echo -e "${BLUE}=== AI-CoScientist RL System Setup ===${NC}"
echo "Setting up RL-enhanced agent selection system..."
echo

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python package installation
check_python_package() {
    python -c "import $1" 2>/dev/null
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."

    # Check Python version
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python $PYTHON_VERSION found ✓"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found"
        exit 1
    fi

    # Check pip
    if command_exists pip || command_exists pip3; then
        print_status "pip found ✓"
    else
        print_error "pip not found"
        exit 1
    fi

    # Check git
    if command_exists git; then
        print_status "git found ✓"
    else
        print_warning "git not found (recommended for version control)"
    fi

    # Check available memory
    if command_exists free; then
        AVAILABLE_RAM=$(free -m | awk 'NR==2{printf "%.0f", $7*100/$2}')
        if [ "$AVAILABLE_RAM" -gt 1000 ]; then
            print_status "Sufficient RAM available ✓"
        else
            print_warning "Low available RAM (${AVAILABLE_RAM}MB). RL training may be slow."
        fi
    fi

    echo
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."

    # Core RL dependencies
    print_status "Installing RL frameworks..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install stable-baselines3[extra]
    pip install gymnasium[classic-control,box2d,atari,accept-rom-license]

    # Scientific computing
    print_status "Installing scientific computing packages..."
    pip install numpy scipy pandas scikit-learn matplotlib seaborn

    # Monitoring and metrics
    print_status "Installing monitoring packages..."
    pip install prometheus-client psutil

    # Web framework for dashboard
    print_status "Installing web framework..."
    pip install fastapi uvicorn websockets

    # Optional advanced packages
    print_status "Installing optional packages..."
    pip install wandb tensorboard optuna --quiet || print_warning "Some optional packages failed to install"

    print_status "Dependencies installed ✓"
    echo
}

# Create necessary directories
setup_directories() {
    print_status "Setting up directories..."

    # Create model storage directory
    mkdir -p "$RL_MODEL_DIR"
    mkdir -p "$RL_MODEL_DIR/checkpoints"
    mkdir -p "$RL_MODEL_DIR/versions"
    print_status "Created model directories ✓"

    # Create metrics storage directory
    mkdir -p "$METRICS_DIR"
    mkdir -p "$METRICS_DIR/performance"
    mkdir -p "$METRICS_DIR/safety"
    print_status "Created metrics directories ✓"

    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR/models"
    mkdir -p "$BACKUP_DIR/configs"
    print_status "Created backup directories ✓"

    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs/rl_system"
    print_status "Created logs directory ✓"

    echo
}

# Create configuration files
create_config_files() {
    print_status "Creating configuration files..."

    # Create main RL config
    cat > "${PROJECT_ROOT}/config/rl_system.yaml" << EOF
# AI-CoScientist RL System Configuration

# Core RL Settings
rl_enabled: true
rl_model_path: "${RL_MODEL_DIR}"
rl_training_enabled: true

# A/B Testing Configuration
ab_testing_enabled: true
initial_rl_traffic_pct: 10.0  # Start with 10% traffic to RL
max_rl_traffic_pct: 90.0      # Maximum 90% traffic to RL

# Performance Thresholds
performance_thresholds:
  success_rate_warning: 0.85
  success_rate_critical: 0.75
  latency_p95_warning_ms: 2000.0
  latency_p95_critical_ms: 5000.0

# Safety Settings
safety_enabled: true
circuit_breaker_enabled: true
rate_limiting_enabled: true

# Monitoring Configuration
monitoring_enabled: true
prometheus_enabled: true
dashboard_enabled: true
dashboard_port: 8001

# Continuous Learning
continuous_learning_enabled: true
learning_mode: "hybrid"  # online_only, periodic_retrain, hybrid
retrain_interval_hours: 24

# Storage Configuration
model_storage_path: "${RL_MODEL_DIR}"
metrics_storage_path: "${METRICS_DIR}"
backup_enabled: true

# Resource Limits
max_memory_usage_mb: 2048
max_cpu_usage_percent: 80.0
max_concurrent_requests: 100

# Integration Settings
agent_pool_integration: "enhanced"  # enhanced, replacement
existing_agent_pool_backup: true
EOF

    # Create environment configuration
    cat > "${PROJECT_ROOT}/.env.rl" << EOF
# RL System Environment Configuration
# Copy to .env or include in your existing .env file

# Core Settings
RL_ENABLED=true
RL_MODEL_PATH=${RL_MODEL_DIR}
RL_TRAINING_ENABLED=true

# A/B Testing
AB_TESTING_ENABLED=true
INITIAL_RL_TRAFFIC_PCT=10
MAX_RL_TRAFFIC_PCT=90

# Performance Monitoring
SUCCESS_RATE_WARNING=0.85
SUCCESS_RATE_CRITICAL=0.75
LATENCY_P95_WARNING_MS=2000
LATENCY_P95_CRITICAL_MS=5000

# Dashboard
DASHBOARD_ENABLED=true
DASHBOARD_PORT=8001

# Continuous Learning
CONTINUOUS_LEARNING_ENABLED=true
LEARNING_MODE=hybrid
RETRAIN_INTERVAL_HOURS=24

# Storage
MODEL_STORAGE_PATH=${RL_MODEL_DIR}
METRICS_STORAGE_PATH=${METRICS_DIR}

# Resource Limits
MAX_MEMORY_USAGE_MB=2048
MAX_CPU_USAGE_PERCENT=80
MAX_CONCURRENT_REQUESTS=100
EOF

    print_status "Created configuration files ✓"
    echo
}

# Set up logging configuration
setup_logging() {
    print_status "Setting up logging configuration..."

    cat > "${PROJECT_ROOT}/config/rl_logging.yaml" << EOF
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  detailed:
    format: '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: ${PROJECT_ROOT}/logs/rl_system/rl_system.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: ${PROJECT_ROOT}/logs/rl_system/rl_errors.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  src.agents.rl:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
EOF

    print_status "Logging configuration created ✓"
    echo
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."

    # Create RL system startup script
    cat > "${PROJECT_ROOT}/scripts/start_rl_system.sh" << 'EOF'
#!/bin/bash

# AI-CoScientist RL System Startup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting AI-CoScientist RL System..."

# Load environment variables
if [ -f "${PROJECT_ROOT}/.env.rl" ]; then
    export $(cat "${PROJECT_ROOT}/.env.rl" | grep -v '^#' | xargs)
fi

# Start the RL system
cd "$PROJECT_ROOT"
python -m src.agents.rl.deployment \
    --config config/rl_system.yaml \
    "$@"
EOF

    # Create health check script
    cat > "${PROJECT_ROOT}/scripts/check_rl_health.sh" << 'EOF'
#!/bin/bash

# RL System Health Check Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Checking RL System Health..."

# Load environment variables
if [ -f "${PROJECT_ROOT}/.env.rl" ]; then
    export $(cat "${PROJECT_ROOT}/.env.rl" | grep -v '^#' | xargs)
fi

cd "$PROJECT_ROOT"
python -m src.agents.rl.deployment --health --config config/rl_system.yaml
EOF

    # Create metrics collection script
    cat > "${PROJECT_ROOT}/scripts/collect_rl_metrics.sh" << 'EOF'
#!/bin/bash

# RL System Metrics Collection Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "${PROJECT_ROOT}/.env.rl" ]; then
    export $(cat "${PROJECT_ROOT}/.env.rl" | grep -v '^#' | xargs)
fi

cd "$PROJECT_ROOT"
python -m src.agents.rl.deployment --metrics --config config/rl_system.yaml
EOF

    # Make scripts executable
    chmod +x "${PROJECT_ROOT}/scripts/start_rl_system.sh"
    chmod +x "${PROJECT_ROOT}/scripts/check_rl_health.sh"
    chmod +x "${PROJECT_ROOT}/scripts/collect_rl_metrics.sh"

    print_status "Startup scripts created ✓"
    echo
}

# Create Docker configuration
create_docker_config() {
    print_status "Creating Docker configuration..."

    cat > "${PROJECT_ROOT}/Dockerfile.rl" << 'EOF'
# AI-CoScientist RL System Docker Configuration
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install RL-specific dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install stable-baselines3[extra] && \
    pip install gymnasium[classic-control] && \
    pip install fastapi uvicorn prometheus-client psutil

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models/rl_agent_selection && \
    mkdir -p /app/storage/metrics && \
    mkdir -p /app/logs/rl_system

# Expose ports
EXPOSE 8000 8001

# Set environment variables
ENV RL_ENABLED=true
ENV DASHBOARD_ENABLED=true
ENV DASHBOARD_PORT=8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m src.agents.rl.deployment --health || exit 1

# Start command
CMD ["python", "-m", "src.agents.rl.deployment", "--config", "config/rl_system.yaml"]
EOF

    cat > "${PROJECT_ROOT}/docker-compose.rl.yml" << 'EOF'
version: '3.8'

services:
  ai-coscientist-rl:
    build:
      context: .
      dockerfile: Dockerfile.rl
    container_name: ai-coscientist-rl
    restart: unless-stopped
    ports:
      - "8000:8000"  # API port
      - "8001:8001"  # Dashboard port
    volumes:
      - ./models:/app/models
      - ./storage:/app/storage
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - RL_ENABLED=true
      - AB_TESTING_ENABLED=true
      - DASHBOARD_ENABLED=true
      - CONTINUOUS_LEARNING_ENABLED=true
    healthcheck:
      test: ["CMD", "python", "-m", "src.agents.rl.deployment", "--health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  rl-prometheus:
    image: prom/prometheus:latest
    container_name: rl-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

networks:
  default:
    name: ai-coscientist-rl
EOF

    print_status "Docker configuration created ✓"
    echo
}

# Validate installation
validate_installation() {
    print_status "Validating RL system installation..."

    cd "$PROJECT_ROOT"

    # Check if we can import RL modules
    python -c "
import sys
sys.path.append('.')
try:
    from src.agents.rl import hybrid_agent_selector, performance_monitor
    print('✓ RL modules imported successfully')
except ImportError as e:
    print(f'✗ Failed to import RL modules: {e}')
    sys.exit(1)
"

    # Check configuration files
    if [ -f "${PROJECT_ROOT}/config/rl_system.yaml" ]; then
        print_status "Configuration files present ✓"
    else
        print_error "Configuration files missing"
        return 1
    fi

    # Check directories
    if [ -d "$RL_MODEL_DIR" ] && [ -d "$METRICS_DIR" ]; then
        print_status "Storage directories created ✓"
    else
        print_error "Storage directories missing"
        return 1
    fi

    # Run deployment validation
    python -m src.agents.rl.deployment --validate --config config/rl_system.yaml
    if [ $? -eq 0 ]; then
        print_status "Deployment validation passed ✓"
    else
        print_error "Deployment validation failed"
        return 1
    fi

    print_status "Installation validation completed ✓"
    echo
}

# Create usage documentation
create_documentation() {
    print_status "Creating usage documentation..."

    cat > "${PROJECT_ROOT}/docs/RL_SYSTEM_USAGE.md" << 'EOF'
# AI-CoScientist RL System Usage Guide

## Overview
The RL (Reinforcement Learning) system enhances AI-CoScientist's agent selection capabilities by learning from past performance to make better decisions.

## Quick Start

### 1. Start the RL System
```bash
./scripts/start_rl_system.sh
```

### 2. Access the Monitoring Dashboard
Open your browser to: http://localhost:8001

### 3. Check System Health
```bash
./scripts/check_rl_health.sh
```

### 4. View Metrics
```bash
./scripts/collect_rl_metrics.sh
```

## Configuration

### Environment Variables
Edit `.env.rl` to customize settings:
- `RL_ENABLED`: Enable/disable RL system
- `AB_TESTING_ENABLED`: Enable A/B testing
- `INITIAL_RL_TRAFFIC_PCT`: Starting percentage of traffic to RL
- `DASHBOARD_PORT`: Port for monitoring dashboard

### YAML Configuration
Edit `config/rl_system.yaml` for advanced settings.

## Using in Code

### Enhanced Agent Pool
```python
from src.agents.rl_integration import enhance_agent_pool_with_rl

# Enhance existing agent pool
enhanced_pool = enhance_agent_pool_with_rl(agent_pool)

# Use smart agent selection
agents, metadata = await enhanced_pool.select_agents_smart(task)
```

### Performance Monitoring
```python
from src.agents.rl.performance_monitor import create_performance_monitor

monitor = create_performance_monitor()
await monitor.start_background_monitoring()

# Record selection events
monitor.record_selection_event(
    strategy="rl_enabled",
    agent_ids=["agent1", "agent2"],
    task_type="complex",
    selection_time=1.2,
    confidence=0.8,
    success=True,
    quality_score=0.85
)
```

## Safety Features

### Circuit Breaker
Automatically disables RL if error rates spike.

### A/B Testing
Gradually increases RL traffic based on performance.

### Automatic Rollback
Reverts to traditional selection if RL performance degrades.

### Monitoring
Real-time dashboards track system health and performance.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Permission Errors**: Check directory permissions
   ```bash
   chmod 755 scripts/*.sh
   ```

3. **Memory Issues**: Adjust `MAX_MEMORY_USAGE_MB` in configuration

### Logs
Check logs in `logs/rl_system/` for detailed error information.

### Health Check
Run health check to diagnose issues:
```bash
python -m src.agents.rl.deployment --health
```

## Advanced Features

### Continuous Learning
The system learns from task outcomes to improve over time.

### Traffic Migration
Gradually migrate traffic from traditional to RL selection.

### Model Versioning
Track and rollback to previous model versions.

### Custom Metrics
Add custom performance metrics for domain-specific evaluation.
EOF

    print_status "Documentation created ✓"
    echo
}

# Main setup function
main() {
    echo -e "${BLUE}Starting RL system setup...${NC}"
    echo

    # Create config directory if it doesn't exist
    mkdir -p "${PROJECT_ROOT}/config"

    # Run setup steps
    check_requirements
    install_dependencies
    setup_directories
    create_config_files
    setup_logging
    create_startup_scripts
    create_docker_config
    validate_installation
    create_documentation

    echo -e "${GREEN}=== Setup Complete! ===${NC}"
    echo
    echo "Next steps:"
    echo "1. Review configuration: ${PROJECT_ROOT}/config/rl_system.yaml"
    echo "2. Start the system: ./scripts/start_rl_system.sh"
    echo "3. Access dashboard: http://localhost:8001"
    echo "4. Read documentation: docs/RL_SYSTEM_USAGE.md"
    echo
    echo -e "${YELLOW}Note: The system starts with 10% traffic to RL by default.${NC}"
    echo -e "${YELLOW}Monitor performance before increasing traffic percentage.${NC}"
    echo
}

# Run setup
main "$@"
# AI-CoScientist Deployment Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-05

Complete guide for deploying AI-CoScientist to production environments.

---

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Docker Deployment](#docker-deployment)
4. [Production Configuration](#production-configuration)
5. [Database Setup](#database-setup)
6. [Monitoring & Logging](#monitoring--logging)
7. [Security](#security)
8. [Scaling](#scaling)
9. [Maintenance](#maintenance)
10. [Troubleshooting](#troubleshooting)

---

## Deployment Overview

### Deployment Options

```
┌──────────────────────────────────────────────────────┐
│  Option 1: Docker Compose (Single Server)            │
│  - Best for: Small-medium deployments                │
│  - Complexity: Low                                    │
│  - Cost: $50-200/month                               │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  Option 2: Kubernetes (Multi-Server)                 │
│  - Best for: Large-scale deployments                 │
│  - Complexity: High                                   │
│  - Cost: $500+/month                                 │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  Option 3: Cloud Platform (Managed)                  │
│  - Best for: Quick deployment, low maintenance       │
│  - Complexity: Medium                                 │
│  - Cost: $100-500/month                              │
└──────────────────────────────────────────────────────┘
```

### Recommended Infrastructure

**Minimum Requirements**:
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 20GB SSD
- **Network**: 100Mbps

**Recommended for Production**:
- **CPU**: 4-8 cores
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **Network**: 1Gbps
- **Backup**: Daily automated backups

---

## Prerequisites

### Required Services

```bash
# Docker & Docker Compose
docker --version  # 24.0+
docker-compose --version  # 2.20+

# Or Kubernetes
kubectl version  # 1.28+
helm version  # 3.12+
```

### API Keys & Credentials

```bash
# Required
OPENAI_API_KEY=sk-...          # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-...   # Anthropic API key

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0

# Optional
SENTRY_DSN=https://...         # Error tracking
PROMETHEUS_URL=http://...      # Metrics
```

---

## Docker Deployment

### 1. Production Dockerfile

```dockerfile
# Dockerfile

FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. Docker Compose Production

```yaml
# docker-compose.prod.yml

version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    image: ai-coscientist:latest
    restart: always
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ENVIRONMENT=production
      - DEBUG=false
      - LOG_LEVEL=INFO
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    networks:
      - app-network
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  postgres:
    image: postgres:15-alpine
    restart: always
    environment:
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=${DB_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - app-network
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - app-network
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    networks:
      - app-network

  prometheus:
    image: prom/prometheus:latest
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - app-network

  grafana:
    image: grafana/grafana:latest
    restart: always
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - app-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  app-network:
    driver: bridge
```

### 3. Deploy with Docker Compose

```bash
# 1. Set environment variables
export DATABASE_URL="postgresql+asyncpg://..."
export REDIS_URL="redis://..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Build images
docker-compose -f docker-compose.prod.yml build

# 3. Run database migrations
docker-compose -f docker-compose.prod.yml run --rm app \
    alembic upgrade head

# 4. Start services
docker-compose -f docker-compose.prod.yml up -d

# 5. Check status
docker-compose -f docker-compose.prod.yml ps

# 6. View logs
docker-compose -f docker-compose.prod.yml logs -f app

# 7. Test health endpoint
curl http://localhost/api/v1/health
```

---

## Production Configuration

### 1. Environment Variables

```bash
# .env.production

# Application
APP_NAME=AI-CoScientist
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/ai_coscientist
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_ECHO=false

# Redis
REDIS_URL=redis://redis:6379/0

# LLM APIs
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Performance
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL=3600

# Security
ALLOWED_HOSTS=["yourdomain.com", "www.yourdomain.com"]
CORS_ORIGINS=["https://yourdomain.com"]

# Monitoring
SENTRY_DSN=https://...@sentry.io/...
PROMETHEUS_ENABLED=true

# Logging
LOG_FORMAT=json
LOG_FILE=/app/logs/app.log
```

### 2. Nginx Configuration

```nginx
# nginx.conf

events {
    worker_connections 1024;
}

http {
    upstream app {
        least_conn;
        server app:8000;
    }

    server {
        listen 80;
        server_name yourdomain.com www.yourdomain.com;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com www.yourdomain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security Headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000" always;

        # Gzip Compression
        gzip on;
        gzip_types text/plain text/css application/json application/javascript;

        # Rate Limiting
        limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

        location / {
            limit_req zone=api_limit burst=20 nodelay;

            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location /metrics {
            # Restrict access to metrics
            allow 10.0.0.0/8;
            deny all;
            proxy_pass http://app;
        }
    }
}
```

### 3. Prometheus Configuration

```yaml
# prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-coscientist'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres_exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis_exporter:9121']
```

---

## Database Setup

### 1. PostgreSQL Production Setup

```bash
# Create database
docker-compose exec postgres psql -U postgres -c \
    "CREATE DATABASE ai_coscientist;"

# Create user
docker-compose exec postgres psql -U postgres -c \
    "CREATE USER ai_user WITH PASSWORD 'secure_password';"

# Grant privileges
docker-compose exec postgres psql -U postgres -c \
    "GRANT ALL PRIVILEGES ON DATABASE ai_coscientist TO ai_user;"

# Run migrations
docker-compose exec app alembic upgrade head

# Verify
docker-compose exec postgres psql -U ai_user -d ai_coscientist -c \
    "SELECT tablename FROM pg_tables WHERE schemaname='public';"
```

### 2. Database Backups

```bash
# Create backup script
cat > scripts/backup_db.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/ai_coscientist_$TIMESTAMP.sql.gz"

# Create backup
docker-compose exec -T postgres pg_dump -U ai_user ai_coscientist | \
    gzip > "$BACKUP_FILE"

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "Backup created: $BACKUP_FILE"
EOF

chmod +x scripts/backup_db.sh

# Add to crontab
crontab -e
# Add: 0 2 * * * /path/to/scripts/backup_db.sh
```

### 3. Database Restore

```bash
# Restore from backup
gunzip -c /backups/ai_coscientist_20250105_020000.sql.gz | \
    docker-compose exec -T postgres psql -U ai_user ai_coscientist

# Verify restore
docker-compose exec postgres psql -U ai_user -d ai_coscientist -c \
    "SELECT COUNT(*) FROM projects;"
```

---

## Monitoring & Logging

### 1. Application Monitoring

```python
# Enable Prometheus metrics in main.py

from prometheus_client import make_asgi_app

# Add metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### 2. Logging Configuration

```python
# src/core/logging.py

import structlog
from pythonjsonlogger import jsonlogger

def configure_logging(environment: str):
    """Configure structured logging."""
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if environment == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(processors=processors)
```

### 3. Grafana Dashboards

```json
// grafana_dashboard.json

{
  "dashboard": {
    "title": "AI-CoScientist Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(api_requests_total[5m])"
        }]
      },
      {
        "title": "Response Time P95",
        "targets": [{
          "expr": "histogram_quantile(0.95, api_request_duration_seconds_bucket)"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(api_requests_total{status=\"error\"}[5m])"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))"
        }]
      }
    ]
  }
}
```

---

## Security

### 1. SSL/TLS Configuration

```bash
# Generate SSL certificate with Let's Encrypt
docker run -it --rm \
    -v /etc/letsencrypt:/etc/letsencrypt \
    -v /var/lib/letsencrypt:/var/lib/letsencrypt \
    certbot/certbot certonly \
    --standalone \
    -d yourdomain.com \
    -d www.yourdomain.com

# Copy certificates
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./ssl/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./ssl/key.pem

# Auto-renewal (crontab)
0 0 1 * * certbot renew --quiet
```

### 2. Secrets Management

```bash
# Use Docker secrets for sensitive data
echo "your-db-password" | docker secret create db_password -
echo "your-api-key" | docker secret create openai_key -

# Reference in docker-compose
services:
  app:
    secrets:
      - db_password
      - openai_key
    environment:
      - DATABASE_PASSWORD_FILE=/run/secrets/db_password
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key

secrets:
  db_password:
    external: true
  openai_key:
    external: true
```

### 3. Security Checklist

- [ ] HTTPS enabled with valid certificate
- [ ] Environment variables secured
- [ ] Database credentials rotated regularly
- [ ] API rate limiting configured
- [ ] CORS properly configured
- [ ] Security headers enabled
- [ ] Regular security updates
- [ ] Firewall configured
- [ ] SSH key-only access
- [ ] Regular backups automated

---

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml

services:
  app:
    deploy:
      replicas: 3  # Run 3 instances
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
```

```bash
# Scale up
docker-compose up -d --scale app=5

# Load balancing automatically handled by nginx
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-coscientist
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-coscientist
  template:
    metadata:
      labels:
        app: ai-coscientist
    spec:
      containers:
      - name: app
        image: ai-coscientist:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: ai-coscientist
spec:
  selector:
    app: ai-coscientist
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Maintenance

### Regular Tasks

```bash
# Daily: Check logs
docker-compose logs --tail=100 app

# Weekly: Database backup verification
scripts/verify_backups.sh

# Monthly: Update dependencies
poetry update
docker-compose build
docker-compose up -d

# Quarterly: Security audit
docker scan ai-coscientist:latest
```

### Updates & Rollbacks

```bash
# Update application
git pull origin main
docker-compose build
docker-compose up -d

# Rollback if needed
docker-compose down
git checkout previous-tag
docker-compose build
docker-compose up -d
```

---

## Troubleshooting

### Common Issues

**Issue**: Container won't start
```bash
# Check logs
docker-compose logs app

# Check resources
docker stats

# Restart container
docker-compose restart app
```

**Issue**: Database connection failed
```bash
# Test connection
docker-compose exec app python -c "from src.core.database import engine; print(engine)"

# Check PostgreSQL
docker-compose exec postgres psql -U ai_user -d ai_coscientist -c "SELECT 1;"
```

**Issue**: High memory usage
```bash
# Check memory
docker stats

# Limit memory in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

---

## Support & Resources

- **Documentation**: https://docs.ai-coscientist.com
- **Issues**: https://github.com/your-org/AI-CoScientist/issues
- **Email**: support@ai-coscientist.com

---

**Last Updated**: 2025-10-05
**Version**: 1.0.0

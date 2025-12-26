# RAG Enhancement System - Deployment & Operations Guide

## 🚀 Deployment Overview

This guide provides comprehensive instructions for deploying and operating the RAG Enhancement System in production environments. The system supports multiple deployment strategies from local development to enterprise Kubernetes clusters.

## 📋 Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB minimum (16GB+ recommended)
- **Storage**: 50GB+ for data and models
- **Network**: High-bandwidth connection for model downloads

### Software Dependencies
- Docker 24.0+
- Docker Compose 2.20+
- Kubernetes 1.28+ (for K8s deployment)
- Python 3.12+
- Poetry (for local development)

## 🏗️ Deployment Strategies

### 1. Local Development Deployment

#### Quick Start Setup
```bash
# Clone repository
git clone <repository-url>
cd AI-CoScientist

# Install dependencies
poetry install

# Setup environment
cp .env.example .env
# Edit .env with your configurations

# Start core services
docker-compose up -d postgres redis chromadb neo4j

# Run database migrations
poetry run alembic upgrade head

# Start the application
poetry run uvicorn src.main:app --reload
```

#### Environment Configuration (.env)
```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/ai_coscientist
REDIS_URL=redis://localhost:6379

# Vector Database
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Knowledge Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### 2. Docker Compose Deployment

#### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  rag-system:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://rag_user:${DB_PASSWORD}@postgres:5432/rag_db
      - REDIS_URL=redis://redis:6379
      - CHROMADB_HOST=chromadb
      - NEO4J_URI=bolt://neo4j:7687
    depends_on:
      - postgres
      - redis
      - chromadb
      - neo4j
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:latest
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
    volumes:
      - chromadb_data:/chroma/data
    ports:
      - "8001:8000"
    restart: unless-stopped

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_dbms_memory_heap_max__size: 2G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    ports:
      - "7474:7474"
      - "7687:7687"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  chromadb_data:
  neo4j_data:
  neo4j_logs:
  prometheus_data:
  grafana_data:
```

#### Deployment Commands
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f rag-system

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale rag-system=3
```

### 3. Kubernetes Deployment

#### Namespace Setup
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
  labels:
    name: rag-system
```

#### ConfigMap and Secrets
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  CHROMADB_HOST: "chromadb-service"
  NEO4J_URI: "bolt://neo4j-service:7687"
  PROMETHEUS_ENABLED: "true"
  LOG_LEVEL: "INFO"

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
data:
  DATABASE_URL: <base64-encoded-url>
  OPENAI_API_KEY: <base64-encoded-key>
  ANTHROPIC_API_KEY: <base64-encoded-key>
  NEO4J_PASSWORD: <base64-encoded-password>
```

#### Main Application Deployment
```yaml
# k8s/rag-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-system
  namespace: rag-system
  labels:
    app: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-system
  template:
    metadata:
      labels:
        app: rag-system
    spec:
      containers:
      - name: rag-system
        image: ai-coscientist/rag-system:latest
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: rag-config
        - secretRef:
            name: rag-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: logs
        emptyDir: {}
      imagePullSecrets:
      - name: registry-secret

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: rag-system-service
  namespace: rag-system
spec:
  selector:
    app: rag-system
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-system-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-system
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Supporting Services
```yaml
# k8s/supporting-services.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chromadb
  namespace: rag-system
spec:
  serviceName: chromadb-service
  replicas: 1
  selector:
    matchLabels:
      app: chromadb
  template:
    metadata:
      labels:
        app: chromadb
    spec:
      containers:
      - name: chromadb
        image: chromadb/chroma:latest
        ports:
        - containerPort: 8000
        env:
        - name: CHROMA_SERVER_HOST
          value: "0.0.0.0"
        - name: CHROMA_SERVER_HTTP_PORT
          value: "8000"
        volumeMounts:
        - name: chromadb-storage
          mountPath: /chroma/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
  volumeClaimTemplates:
  - metadata:
      name: chromadb-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi

---
apiVersion: v1
kind: Service
metadata:
  name: chromadb-service
  namespace: rag-system
spec:
  selector:
    app: chromadb
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP

---
# Neo4j StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
  namespace: rag-system
spec:
  serviceName: neo4j-service
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5-community
        ports:
        - containerPort: 7474
        - containerPort: 7687
        env:
        - name: NEO4J_AUTH
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: NEO4J_AUTH
        - name: NEO4J_dbms_memory_heap_max__size
          value: "2G"
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
  volumeClaimTemplates:
  - metadata:
      name: neo4j-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi

---
apiVersion: v1
kind: Service
metadata:
  name: neo4j-service
  namespace: rag-system
spec:
  selector:
    app: neo4j
  ports:
  - port: 7474
    targetPort: 7474
    name: http
  - port: 7687
    targetPort: 7687
    name: bolt
  type: ClusterIP
```

#### Ingress Configuration
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-system-ingress
  namespace: rag-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - rag-api.yourcompany.com
    secretName: rag-system-tls
  rules:
  - host: rag-api.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-system-service
            port:
              number: 80
```

#### Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Deploying RAG Enhancement System to Kubernetes..."

# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply ConfigMaps and Secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Deploy supporting services
kubectl apply -f k8s/supporting-services.yaml

# Wait for supporting services
echo "⏳ Waiting for supporting services..."
kubectl wait --for=condition=ready pod -l app=chromadb -n rag-system --timeout=300s
kubectl wait --for=condition=ready pod -l app=neo4j -n rag-system --timeout=300s

# Deploy main application
kubectl apply -f k8s/rag-deployment.yaml

# Deploy monitoring
kubectl apply -f k8s/monitoring.yaml

# Configure ingress
kubectl apply -f k8s/ingress.yaml

# Wait for deployment
echo "⏳ Waiting for RAG system deployment..."
kubectl wait --for=condition=available deployment/rag-system -n rag-system --timeout=600s

echo "✅ RAG Enhancement System deployed successfully!"
echo "🌐 API available at: https://rag-api.yourcompany.com"
echo "📊 Monitoring available at: https://monitoring.yourcompany.com"

# Show status
kubectl get pods -n rag-system
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Core Application
API_VERSION=v1
DEBUG=false
LOG_LEVEL=INFO
CORS_ORIGINS=["https://yourapp.com"]

# Database Settings
DATABASE_URL=postgresql://user:pass@host:5432/db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=0

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=100
CACHE_TTL=3600

# RAG System Configuration
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000
CHROMADB_COLLECTION_PREFIX=rag_prod

# Knowledge Graph
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password
NEO4J_MAX_CONNECTION_POOL_SIZE=100

# LLM Providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_MAX_RETRIES=3
ANTHROPIC_MAX_RETRIES=3

# Strategy Configuration
STRATEGY_WEIGHTS={"simple": 0.1, "hybrid": 0.3, "graph_rag": 0.4, "multimodal": 0.2}
QUALITY_THRESHOLD=0.8
RESPONSE_TIME_THRESHOLD=2.0

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# Security
JWT_SECRET_KEY=your_jwt_secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20
```

### Configuration Validation
```python
# config_validator.py
import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    REQUIRED_VARS = [
        'DATABASE_URL',
        'REDIS_URL',
        'CHROMADB_HOST',
        'OPENAI_API_KEY'
    ]

    OPTIONAL_VARS = {
        'ANTHROPIC_API_KEY': None,
        'NEO4J_URI': 'bolt://localhost:7687',
        'PROMETHEUS_ENABLED': 'true',
        'LOG_LEVEL': 'INFO'
    }

    @classmethod
    def validate_config(cls) -> bool:
        """Validate all required configuration variables."""
        missing_vars = []

        for var in cls.REQUIRED_VARS:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False

        # Set defaults for optional variables
        for var, default in cls.OPTIONAL_VARS.items():
            if not os.getenv(var) and default:
                os.environ[var] = default
                logger.info(f"Set default value for {var}: {default}")

        return True

    @classmethod
    def validate_connections(cls) -> Dict[str, bool]:
        """Validate connections to external services."""
        connections = {}

        # Test database connection
        try:
            from src.core.database import test_connection
            connections['database'] = test_connection()
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            connections['database'] = False

        # Test Redis connection
        try:
            from src.core.redis import test_redis_connection
            connections['redis'] = test_redis_connection()
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            connections['redis'] = False

        # Test ChromaDB connection
        try:
            from src.services.knowledge_base.vector_store import test_chromadb
            connections['chromadb'] = test_chromadb()
        except Exception as e:
            logger.error(f"ChromaDB connection failed: {e}")
            connections['chromadb'] = False

        return connections
```

## 📊 Monitoring & Observability

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rag_system_rules.yml"

scrape_configs:
  - job_name: 'rag-system'
    static_configs:
      - targets: ['rag-system:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'chromadb'
    static_configs:
      - targets: ['chromadb:8000']

  - job_name: 'neo4j'
    static_configs:
      - targets: ['neo4j:2004']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "RAG Enhancement System",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_request_total[5m])",
            "legendFormat": "{{strategy}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Quality Scores",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(rag_quality_score) by (strategy)",
            "legendFormat": "{{strategy}}"
          }
        ]
      }
    ]
  }
}
```

### Health Check Endpoints
```python
# src/api/health.py
from fastapi import APIRouter, HTTPException
from src.core.health import HealthChecker

router = APIRouter()
health_checker = HealthChecker()

@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/ready")
async def readiness_check():
    """Readiness check with dependency validation."""
    health_status = await health_checker.check_all_dependencies()

    if not health_status.healthy:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "status": "ready",
        "checks": health_status.checks,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from src.monitoring.metrics import generate_metrics
    return generate_metrics()
```

## 🔍 Troubleshooting

### Common Issues

#### 1. ChromaDB Connection Issues
```bash
# Check ChromaDB status
curl -f http://chromadb:8000/api/v1/heartbeat

# Debug connection
kubectl logs -n rag-system deployment/chromadb

# Reset ChromaDB data (if needed)
kubectl delete pvc -n rag-system chromadb-storage-chromadb-0
kubectl delete pod -n rag-system chromadb-0
```

#### 2. High Memory Usage
```bash
# Check memory usage
kubectl top pods -n rag-system

# Restart high-memory pods
kubectl rollout restart deployment/rag-system -n rag-system

# Check for memory leaks
kubectl exec -n rag-system deployment/rag-system -- python -m memory_profiler
```

#### 3. Slow Response Times
```bash
# Check strategy performance
curl http://rag-system:8000/api/v1/metrics/performance

# Analyze query patterns
kubectl logs -n rag-system deployment/rag-system | grep "SLOW_QUERY"

# Check database performance
kubectl exec -n rag-system postgresql-0 -- psql -c "SELECT * FROM pg_stat_activity;"
```

### Debug Commands
```bash
# Get system status
python scripts/validate_complete_system.py --verbose

# Check all services
docker-compose ps

# View specific service logs
docker-compose logs -f rag-system

# Interactive debugging
kubectl exec -n rag-system deployment/rag-system -it -- bash

# Database queries
kubectl exec -n rag-system postgresql-0 -- psql -c "SELECT COUNT(*) FROM papers;"

# Redis monitoring
kubectl exec -n rag-system redis-0 -- redis-cli monitor
```

## 🔐 Security Considerations

### Network Security
```yaml
# Network Policy Example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rag-system-netpol
  namespace: rag-system
spec:
  podSelector:
    matchLabels:
      app: rag-system
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: chromadb
    ports:
    - protocol: TCP
      port: 8000
```

### Secret Management
```bash
# Create secrets from files
kubectl create secret generic rag-secrets \
  --from-file=openai-key=openai.key \
  --from-file=anthropic-key=anthropic.key \
  -n rag-system

# Use external secret management
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: rag-system
spec:
  provider:
    vault:
      server: "https://vault.company.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "rag-system"
```

## 🚀 Performance Optimization

### Resource Optimization
```yaml
# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: rag-system-pdb
  namespace: rag-system
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: rag-system

# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: rag-system-vpa
  namespace: rag-system
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-system
  updatePolicy:
    updateMode: "Auto"
```

### Database Optimization
```sql
-- PostgreSQL optimization queries
-- Index creation for better query performance
CREATE INDEX CONCURRENTLY idx_papers_created_at ON papers(created_at);
CREATE INDEX CONCURRENTLY idx_papers_domain ON papers(domain);
CREATE INDEX CONCURRENTLY idx_improvements_paper_id ON improvements(paper_id);

-- Update table statistics
ANALYZE papers;
ANALYZE improvements;

-- Check slow queries
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

## 📈 Scaling Guidelines

### Horizontal Scaling
```bash
# Manual scaling
kubectl scale deployment/rag-system --replicas=5 -n rag-system

# Auto-scaling configuration
kubectl autoscale deployment/rag-system \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n rag-system
```

### Vertical Scaling
```yaml
# Increase resource limits
spec:
  containers:
  - name: rag-system
    resources:
      requests:
        memory: "4Gi"
        cpu: "1"
      limits:
        memory: "8Gi"
        cpu: "4"
```

## 🔄 Backup & Recovery

### Backup Strategy
```bash
#!/bin/bash
# backup.sh

set -e

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup PostgreSQL
kubectl exec -n rag-system postgresql-0 -- \
  pg_dump -U rag_user rag_db > "$BACKUP_DIR/postgres.sql"

# Backup ChromaDB
kubectl exec -n rag-system chromadb-0 -- \
  tar czf - /chroma/data > "$BACKUP_DIR/chromadb.tar.gz"

# Backup Neo4j
kubectl exec -n rag-system neo4j-0 -- \
  tar czf - /data > "$BACKUP_DIR/neo4j.tar.gz"

# Upload to cloud storage
aws s3 sync "$BACKUP_DIR" "s3://rag-system-backups/$(date +%Y%m%d)/"
```

### Recovery Procedure
```bash
#!/bin/bash
# restore.sh

BACKUP_DATE=$1
if [ -z "$BACKUP_DATE" ]; then
  echo "Usage: $0 <backup_date>"
  exit 1
fi

# Download from cloud storage
aws s3 sync "s3://rag-system-backups/$BACKUP_DATE/" "/restore/"

# Restore PostgreSQL
kubectl exec -i -n rag-system postgresql-0 -- \
  psql -U rag_user -d rag_db < "/restore/postgres.sql"

# Restore ChromaDB
kubectl exec -i -n rag-system chromadb-0 -- \
  tar xzf - -C / < "/restore/chromadb.tar.gz"

# Restart services
kubectl rollout restart deployment/rag-system -n rag-system
```

## 🎯 Success Metrics

### Key Performance Indicators (KPIs)
- **Availability**: 99.9% uptime target
- **Response Time**: P95 < 2 seconds
- **Quality Score**: Average > 0.8
- **Error Rate**: < 0.1%
- **Throughput**: 1000+ requests per minute

### Monitoring Dashboard URLs
- **Grafana**: https://monitoring.yourcompany.com/grafana
- **Prometheus**: https://monitoring.yourcompany.com/prometheus
- **AlertManager**: https://monitoring.yourcompany.com/alerts

---

## 🎉 Conclusion

This deployment guide provides comprehensive instructions for operating the RAG Enhancement System in production. The system is designed for high availability, scalability, and maintainability across multiple deployment environments.

**Deployment Status**: 🚀 Production Ready
- Multiple deployment strategies supported
- Comprehensive monitoring and observability
- Security best practices implemented
- Automated backup and recovery procedures
- Performance optimization guidelines included
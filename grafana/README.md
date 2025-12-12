# RAG Evaluation Grafana Dashboard

This directory contains Grafana dashboard configuration for visualizing RAG evaluation metrics.

## Overview

The dashboard visualizes RAGAS metrics exported via Prometheus:
- **Faithfulness**: Factual consistency with retrieved contexts
- **Answer Relevancy**: Relevance of answers to questions
- **Context Precision**: Precision of retrieved contexts
- **Context Recall**: Recall of retrieved contexts

## Setup

### Prerequisites

- Prometheus server running
- Grafana instance running
- Python application exporting metrics via `PrometheusMetricsExporter`

### 1. Configure Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'ai-coscientist-rag'
    static_configs:
      - targets: ['localhost:8000']  # Your FastAPI app
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### 2. Expose Metrics Endpoint

In your FastAPI application:

```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from src.services.rag.metrics_exporter import PrometheusMetricsExporter

app = FastAPI()

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Use exporter in your evaluation pipeline
exporter = PrometheusMetricsExporter()

@app.post("/evaluate")
async def evaluate_rag():
    # ... run evaluation
    exporter.export_metrics(results, labels={'env': 'production', 'model': 'gpt-4'})
    return results
```

### 3. Import Dashboard to Grafana

#### Option A: Via Grafana UI
1. Open Grafana (default: http://localhost:3000)
2. Go to Dashboards → Import
3. Upload `dashboards/rag_evaluation_dashboard.json`
4. Select your Prometheus data source
5. Click Import

#### Option B: Via Provisioning
1. Copy dashboard JSON to Grafana provisioning directory:
   ```bash
   cp dashboards/rag_evaluation_dashboard.json /etc/grafana/provisioning/dashboards/
   ```
2. Restart Grafana
3. Dashboard will auto-load

### 4. Configure Data Source

1. Go to Configuration → Data Sources
2. Add Prometheus data source
3. URL: `http://localhost:9090` (your Prometheus instance)
4. Save & Test

## Dashboard Panels

### Metrics Overview
- **Faithfulness Score**: Time series of faithfulness metric
- **Answer Relevancy Score**: Time series of answer relevancy
- **Context Precision Score**: Time series of context precision
- **Context Recall Score**: Time series of context recall

### Combined View
- **All RAG Metrics Combined**: Single graph with all metrics for comparison

### Statistics
- **Total Evaluations**: Counter showing total number of evaluations run
- **Faithfulness Distribution**: Heatmap showing score distribution over time

### Variables
- **env**: Filter by environment (production, staging, test)
- **model**: Filter by model (gpt-4, gpt-3.5, etc.)

## Usage Example

```python
from src.services.rag.baseline_evaluator import BaselineEvaluator
from src.services.rag.metrics_exporter import PrometheusMetricsExporter

# Run evaluation
evaluator = BaselineEvaluator()
results = evaluator.run_evaluation_pipeline(test_cases, rag_system)

# Export to Prometheus
exporter = PrometheusMetricsExporter()
exporter.export_metrics(
    results,
    labels={'env': 'production', 'model': 'gpt-4'}
)

# Metrics now available at /metrics endpoint
# Grafana will automatically scrape and visualize
```

## Metrics Reference

### Gauge Metrics
- `rag_faithfulness{env, model}`: Current faithfulness score (0-1)
- `rag_answer_relevancy{env, model}`: Current answer relevancy (0-1)
- `rag_context_precision{env, model}`: Current context precision (0-1)
- `rag_context_recall{env, model}`: Current context recall (0-1)

### Counter Metrics
- `rag_evaluations_total{env, model}`: Total evaluations run

### Histogram Metrics
- `rag_faithfulness_histogram{env, model}`: Distribution of faithfulness scores

## Alerting

You can configure alerts in Grafana for:

- **Low Faithfulness**: Alert when faithfulness < 0.7
- **Low Answer Relevancy**: Alert when answer_relevancy < 0.7
- **Evaluation Failures**: Alert when no new evaluations in X minutes

Example alert rule:
```yaml
alert: LowFaithfulness
expr: rag_faithfulness{env="production"} < 0.7
for: 5m
labels:
  severity: warning
annotations:
  summary: "RAG faithfulness score is low"
  description: "Faithfulness score {{ $value }} is below threshold (0.7)"
```

## Troubleshooting

### No data showing in Grafana
1. Check Prometheus is scraping: http://localhost:9090/targets
2. Verify metrics endpoint: http://localhost:8000/metrics
3. Check Prometheus query in Grafana Explore

### Metrics not updating
1. Verify evaluation pipeline is running
2. Check metric export is being called
3. Confirm Prometheus scrape interval

### Dashboard not loading
1. Verify Prometheus data source is configured
2. Check dashboard JSON is valid
3. Ensure Prometheus URL is correct

## Development

To test locally:
```bash
# Start Prometheus (with config pointing to your app)
prometheus --config.file=prometheus.yml

# Start Grafana
grafana-server

# Run your FastAPI app with metrics
uvicorn main:app --reload

# Access Grafana
open http://localhost:3000
```

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [RAGAS Metrics](https://docs.ragas.io/en/latest/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

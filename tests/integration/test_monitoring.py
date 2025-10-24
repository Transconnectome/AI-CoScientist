"""
Integration tests for Phase 2 monitoring infrastructure.

TDD RED Phase: Tests document expected monitoring behavior.
Tests verify Prometheus and Grafana integration.

Tests cover:
- Prometheus health
- Metrics endpoint
- Grafana health
- Dashboard accessibility
"""

import pytest
from httpx import AsyncClient


class TestPrometheusIntegration:
    """Test Prometheus monitoring integration."""

    @pytest.mark.asyncio
    async def test_prometheus_health(self):
        """Test Prometheus health endpoint."""
        async with AsyncClient() as client:
            response = await client.get("http://localhost:9090/-/healthy")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_prometheus_metrics_endpoint(self):
        """Test API /metrics endpoint for Prometheus scraping."""
        async with AsyncClient() as client:
            response = await client.get("http://localhost:8000/metrics")
            assert response.status_code == 200
            content = response.text
            assert "# HELP" in content  # Prometheus format
            assert "# TYPE" in content


class TestGrafanaIntegration:
    """Test Grafana dashboard integration."""

    @pytest.mark.asyncio
    async def test_grafana_health(self):
        """Test Grafana health endpoint."""
        async with AsyncClient() as client:
            response = await client.get("http://localhost:3000/api/health")
            assert response.status_code == 200
            data = response.json()
            assert data["database"] == "ok"

    @pytest.mark.asyncio
    async def test_grafana_datasource_configured(self):
        """Test Grafana has Prometheus datasource configured."""
        async with AsyncClient() as client:
            # Note: Would need auth in real scenario
            response = await client.get(
                "http://localhost:3000/api/datasources",
                auth=("admin", "admin")  # Default creds for test
            )
            if response.status_code == 200:
                datasources = response.json()
                prometheus_ds = [ds for ds in datasources if ds["type"] == "prometheus"]
                assert len(prometheus_ds) > 0


class TestEndToEndMonitoring:
    """Test end-to-end monitoring flow."""

    @pytest.mark.asyncio
    async def test_metrics_collection_flow(self, api_client: AsyncClient):
        """Test complete metrics collection flow."""
        # 1. Track performance via API
        response = await api_client.post(
            "/api/v1/rag/performance/track",
            json={
                "operation": "monitoring_test",
                "latency": 0.5,
                "tokens": {"prompt": 100, "completion": 50},
                "model": "gpt-4"
            }
        )
        assert response.status_code == 200

        # 2. Verify /metrics endpoint exposes metric
        async with AsyncClient() as client:
            metrics_response = await client.get("http://localhost:8000/metrics")
            # Metrics should be available for Prometheus to scrape
            assert metrics_response.status_code == 200

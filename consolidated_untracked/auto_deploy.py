"""
Automated Deployment System for DD-RAPTOR
Multi-environment deployment with blue-green, canary, and rollback capabilities
"""

import asyncio
import docker
import logging
import yaml
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import tempfile
import hashlib
import boto3
import kubernetes
from kubernetes import client, config
import redis
from concurrent.futures import ThreadPoolExecutor
import time
import requests
from contextlib import asynccontextmanager

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    IMMEDIATE = "immediate"

class DeploymentStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYING = "deploying"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: str  # dev, staging, production
    strategy: DeploymentStrategy
    image_tag: str
    replicas: int = 3
    health_check_path: str = "/health"
    health_check_timeout: int = 300
    canary_percentage: int = 10
    rollback_on_failure: bool = True
    notifications: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "2000m",
        "memory": "4Gi"
    })

@dataclass
class DeploymentResult:
    """Deployment result"""
    deployment_id: str
    status: DeploymentStatus
    environment: str
    strategy: DeploymentStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    image_tag: str = ""
    error_message: Optional[str] = None
    rollback_info: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

class ContainerBuilder:
    """Docker container builder with optimization"""

    def __init__(self, registry_url: str = "localhost:5000"):
        self.docker_client = docker.from_env()
        self.registry_url = registry_url
        self.logger = logging.getLogger(__name__)

    async def build_image(self,
                         dockerfile_path: str,
                         context_path: str,
                         image_name: str,
                         tag: str,
                         build_args: Optional[Dict[str, str]] = None) -> str:
        """Build Docker image with optimization"""
        full_image_name = f"{self.registry_url}/{image_name}:{tag}"

        try:
            # Create optimized Dockerfile
            optimized_dockerfile = await self._create_optimized_dockerfile(dockerfile_path)

            # Build image
            self.logger.info(f"Building image: {full_image_name}")

            # Use BuildKit for better performance
            build_args = build_args or {}
            build_args.update({
                "BUILDKIT_INLINE_CACHE": "1",
                "DOCKER_BUILDKIT": "1"
            })

            image, logs = self.docker_client.images.build(
                path=context_path,
                dockerfile=optimized_dockerfile,
                tag=full_image_name,
                buildargs=build_args,
                pull=True,
                rm=True,
                forcerm=True,
                nocache=False,
                use_config_proxy=True
            )

            # Log build output
            for log in logs:
                if 'stream' in log:
                    self.logger.info(log['stream'].strip())

            self.logger.info(f"Successfully built image: {full_image_name}")
            return full_image_name

        except Exception as e:
            self.logger.error(f"Failed to build image {full_image_name}: {e}")
            raise

    async def _create_optimized_dockerfile(self, original_dockerfile: str) -> str:
        """Create optimized Dockerfile with multi-stage build"""

        optimized_content = """
# Multi-stage build for DD-RAPTOR optimization
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libc6-dev \\
    libffi-dev \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Build stage
FROM base as builder

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY config/ ./config/

# Production stage
FROM base as production

WORKDIR /app

# Copy from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app .

# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Change to app user
USER appuser

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""

        # Write optimized Dockerfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.Dockerfile', delete=False)
        temp_file.write(optimized_content)
        temp_file.close()

        return temp_file.name

    async def push_image(self, image_name: str) -> bool:
        """Push image to registry"""
        try:
            self.logger.info(f"Pushing image: {image_name}")

            # Push to registry
            response = self.docker_client.images.push(image_name)
            self.logger.info(f"Push response: {response}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to push image {image_name}: {e}")
            return False

class KubernetesDeployer:
    """Kubernetes deployment manager"""

    def __init__(self, kubeconfig_path: Optional[str] = None):
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()

        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.logger = logging.getLogger(__name__)

    async def deploy_blue_green(self,
                               config: DeploymentConfig,
                               image_name: str) -> DeploymentResult:
        """Blue-green deployment strategy"""

        deployment_id = self._generate_deployment_id(config.environment)
        result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.DEPLOYING,
            environment=config.environment,
            strategy=DeploymentStrategy.BLUE_GREEN,
            start_time=datetime.now(),
            image_tag=config.image_tag
        )

        try:
            namespace = f"dd-raptor-{config.environment}"
            service_name = "dd-raptor-service"
            current_color = await self._get_current_color(namespace, service_name)
            new_color = "green" if current_color == "blue" else "blue"

            # Deploy new version
            deployment_name = f"dd-raptor-{new_color}"
            await self._deploy_k8s_deployment(
                namespace, deployment_name, image_name, config, new_color
            )

            # Wait for rollout
            await self._wait_for_rollout(namespace, deployment_name)

            # Health check
            if await self._health_check(namespace, deployment_name, config):
                # Switch traffic
                await self._switch_service_traffic(namespace, service_name, new_color)

                # Clean up old deployment
                old_deployment_name = f"dd-raptor-{current_color}"
                await self._cleanup_deployment(namespace, old_deployment_name)

                result.status = DeploymentStatus.SUCCESS
                result.end_time = datetime.now()

            else:
                # Rollback
                if config.rollback_on_failure:
                    await self._cleanup_deployment(namespace, deployment_name)
                    result.status = DeploymentStatus.FAILED
                    result.error_message = "Health check failed, rolled back"

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            self.logger.error(f"Blue-green deployment failed: {e}")

        return result

    async def deploy_canary(self,
                           config: DeploymentConfig,
                           image_name: str) -> DeploymentResult:
        """Canary deployment strategy"""

        deployment_id = self._generate_deployment_id(config.environment)
        result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.DEPLOYING,
            environment=config.environment,
            strategy=DeploymentStrategy.CANARY,
            start_time=datetime.now(),
            image_tag=config.image_tag
        )

        try:
            namespace = f"dd-raptor-{config.environment}"

            # Deploy canary version
            canary_deployment = "dd-raptor-canary"
            canary_replicas = max(1, (config.replicas * config.canary_percentage) // 100)

            canary_config = DeploymentConfig(
                environment=config.environment,
                strategy=config.strategy,
                image_tag=config.image_tag,
                replicas=canary_replicas,
                health_check_path=config.health_check_path,
                health_check_timeout=config.health_check_timeout,
                environment_vars=config.environment_vars,
                resource_limits=config.resource_limits
            )

            await self._deploy_k8s_deployment(
                namespace, canary_deployment, image_name, canary_config, "canary"
            )

            # Wait for canary rollout
            await self._wait_for_rollout(namespace, canary_deployment)

            # Monitor canary performance
            canary_metrics = await self._monitor_canary(namespace, canary_deployment, 300)  # 5 minutes

            if canary_metrics['success_rate'] > 0.95 and canary_metrics['error_rate'] < 0.05:
                # Promote canary to production
                main_deployment = "dd-raptor-main"
                await self._deploy_k8s_deployment(
                    namespace, main_deployment, image_name, config, "main"
                )

                await self._wait_for_rollout(namespace, main_deployment)

                # Clean up canary
                await self._cleanup_deployment(namespace, canary_deployment)

                result.status = DeploymentStatus.SUCCESS
                result.end_time = datetime.now()
                result.metrics = canary_metrics

            else:
                # Rollback canary
                await self._cleanup_deployment(namespace, canary_deployment)
                result.status = DeploymentStatus.FAILED
                result.error_message = f"Canary metrics failed: {canary_metrics}"

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            self.logger.error(f"Canary deployment failed: {e}")

        return result

    async def _deploy_k8s_deployment(self,
                                   namespace: str,
                                   deployment_name: str,
                                   image_name: str,
                                   config: DeploymentConfig,
                                   label_suffix: str):
        """Deploy to Kubernetes"""

        # Create namespace if it doesn't exist
        await self._ensure_namespace(namespace)

        # Deployment spec
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=deployment_name,
                namespace=namespace,
                labels={
                    "app": "dd-raptor",
                    "version": config.image_tag,
                    "environment": config.environment,
                    "color": label_suffix
                }
            ),
            spec=client.V1DeploymentSpec(
                replicas=config.replicas,
                selector=client.V1LabelSelector(
                    match_labels={
                        "app": "dd-raptor",
                        "color": label_suffix
                    }
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": "dd-raptor",
                            "version": config.image_tag,
                            "color": label_suffix
                        }
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="dd-raptor",
                                image=image_name,
                                ports=[client.V1ContainerPort(container_port=8000)],
                                env=[
                                    client.V1EnvVar(name=k, value=v)
                                    for k, v in config.environment_vars.items()
                                ],
                                resources=client.V1ResourceRequirements(
                                    limits=config.resource_limits,
                                    requests={
                                        "cpu": "500m",
                                        "memory": "1Gi"
                                    }
                                ),
                                liveness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path=config.health_check_path,
                                        port=8000
                                    ),
                                    initial_delay_seconds=30,
                                    period_seconds=10
                                ),
                                readiness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path=config.health_check_path,
                                        port=8000
                                    ),
                                    initial_delay_seconds=5,
                                    period_seconds=5
                                )
                            )
                        ]
                    )
                )
            )
        )

        # Apply deployment
        try:
            self.apps_v1.create_namespaced_deployment(namespace, deployment)
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self.apps_v1.replace_namespaced_deployment(
                    deployment_name, namespace, deployment
                )
            else:
                raise

    async def _ensure_namespace(self, namespace: str):
        """Ensure namespace exists"""
        try:
            self.v1.read_namespace(namespace)
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_obj = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                self.v1.create_namespace(namespace_obj)

    async def _wait_for_rollout(self, namespace: str, deployment_name: str, timeout: int = 600):
        """Wait for deployment rollout to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            deployment = self.apps_v1.read_namespaced_deployment(deployment_name, namespace)

            if (deployment.status.ready_replicas and
                deployment.status.ready_replicas == deployment.spec.replicas):
                self.logger.info(f"Deployment {deployment_name} rolled out successfully")
                return True

            await asyncio.sleep(10)

        raise TimeoutError(f"Deployment {deployment_name} rollout timed out")

    async def _health_check(self,
                           namespace: str,
                           deployment_name: str,
                           config: DeploymentConfig) -> bool:
        """Perform health check on deployment"""
        try:
            # Get service endpoint
            service_url = await self._get_service_url(namespace, deployment_name)
            health_url = f"{service_url}{config.health_check_path}"

            # Check health for specified timeout
            start_time = time.time()
            while time.time() - start_time < config.health_check_timeout:
                try:
                    response = requests.get(health_url, timeout=10)
                    if response.status_code == 200:
                        self.logger.info(f"Health check passed for {deployment_name}")
                        return True
                except requests.RequestException:
                    pass

                await asyncio.sleep(10)

            return False

        except Exception as e:
            self.logger.error(f"Health check failed for {deployment_name}: {e}")
            return False

    async def _get_service_url(self, namespace: str, deployment_name: str) -> str:
        """Get service URL for deployment"""
        # This is a simplified implementation
        # In production, you'd get the actual service endpoint
        return f"http://{deployment_name}.{namespace}.svc.cluster.local:8000"

    async def _get_current_color(self, namespace: str, service_name: str) -> str:
        """Get current color (blue/green) from service selector"""
        try:
            service = self.v1.read_namespaced_service(service_name, namespace)
            return service.spec.selector.get('color', 'blue')
        except client.exceptions.ApiException:
            return 'blue'  # Default

    async def _switch_service_traffic(self, namespace: str, service_name: str, new_color: str):
        """Switch service traffic to new color"""
        service = self.v1.read_namespaced_service(service_name, namespace)
        service.spec.selector['color'] = new_color
        self.v1.replace_namespaced_service(service_name, namespace, service)

    async def _cleanup_deployment(self, namespace: str, deployment_name: str):
        """Clean up deployment"""
        try:
            self.apps_v1.delete_namespaced_deployment(deployment_name, namespace)
        except client.exceptions.ApiException:
            pass  # Already deleted

    async def _monitor_canary(self, namespace: str, deployment_name: str, duration: int) -> Dict[str, float]:
        """Monitor canary deployment metrics"""
        # Simplified monitoring - in production, integrate with Prometheus
        start_time = time.time()
        success_count = 0
        error_count = 0
        total_requests = 0

        while time.time() - start_time < duration:
            # Simulate metric collection
            # In production, query Prometheus or monitoring system
            await asyncio.sleep(30)

            # Mock metrics
            success_count += 95
            error_count += 3
            total_requests += 98

        if total_requests == 0:
            return {'success_rate': 0.0, 'error_rate': 1.0}

        return {
            'success_rate': success_count / total_requests,
            'error_rate': error_count / total_requests,
            'total_requests': total_requests
        }

    def _generate_deployment_id(self, environment: str) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{environment}-{timestamp}"

class AutoDeploySystem:
    """Main automated deployment system"""

    def __init__(self,
                 registry_url: str = "localhost:5000",
                 redis_client: Optional[redis.Redis] = None):
        self.container_builder = ContainerBuilder(registry_url)
        self.k8s_deployer = KubernetesDeployer()
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)

    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Main deployment function"""

        deployment_id = self._generate_deployment_id(config.environment)

        # Store deployment status
        await self._update_deployment_status(deployment_id, DeploymentStatus.PENDING)

        try:
            # Build image
            await self._update_deployment_status(deployment_id, DeploymentStatus.BUILDING)
            image_name = await self._build_and_push_image(config)

            # Run tests
            await self._update_deployment_status(deployment_id, DeploymentStatus.TESTING)
            if not await self._run_tests(image_name):
                raise Exception("Tests failed")

            # Deploy based on strategy
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYING)

            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self.k8s_deployer.deploy_blue_green(config, image_name)
            elif config.strategy == DeploymentStrategy.CANARY:
                result = await self.k8s_deployer.deploy_canary(config, image_name)
            else:
                raise ValueError(f"Unsupported deployment strategy: {config.strategy}")

            # Update final status
            await self._update_deployment_status(deployment_id, result.status)

            # Send notifications
            if config.notifications:
                await self._send_notifications(config.notifications, result)

            return result

        except Exception as e:
            error_result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                environment=config.environment,
                strategy=config.strategy,
                start_time=datetime.now(),
                end_time=datetime.now(),
                image_tag=config.image_tag,
                error_message=str(e)
            )

            await self._update_deployment_status(deployment_id, DeploymentStatus.FAILED)

            if config.notifications:
                await self._send_notifications(config.notifications, error_result)

            return error_result

    async def _build_and_push_image(self, config: DeploymentConfig) -> str:
        """Build and push Docker image"""

        # Determine paths
        dockerfile_path = "Dockerfile"
        context_path = "."
        image_name = "dd-raptor"

        # Build image
        full_image_name = await self.container_builder.build_image(
            dockerfile_path=dockerfile_path,
            context_path=context_path,
            image_name=image_name,
            tag=config.image_tag,
            build_args={"ENVIRONMENT": config.environment}
        )

        # Push image
        if not await self.container_builder.push_image(full_image_name):
            raise Exception("Failed to push image")

        return full_image_name

    async def _run_tests(self, image_name: str) -> bool:
        """Run tests against the image"""
        try:
            # Run container with tests
            test_command = [
                "docker", "run", "--rm",
                image_name,
                "python", "-m", "pytest", "tests/", "-v"
            ]

            result = subprocess.run(test_command, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info("Tests passed")
                return True
            else:
                self.logger.error(f"Tests failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return False

    async def _update_deployment_status(self, deployment_id: str, status: DeploymentStatus):
        """Update deployment status in Redis"""
        try:
            status_data = {
                'deployment_id': deployment_id,
                'status': status.value,
                'timestamp': datetime.now().isoformat()
            }

            await self.redis_client.hset(
                f"deployment:{deployment_id}",
                mapping=status_data
            )

            # Also update in deployments list
            await self.redis_client.zadd(
                "deployments",
                {deployment_id: time.time()}
            )

        except Exception as e:
            self.logger.warning(f"Could not update deployment status: {e}")

    async def _send_notifications(self,
                                 notification_urls: List[str],
                                 result: DeploymentResult):
        """Send deployment notifications"""
        notification_data = {
            'deployment_id': result.deployment_id,
            'status': result.status.value,
            'environment': result.environment,
            'strategy': result.strategy.value,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'image_tag': result.image_tag,
            'error_message': result.error_message
        }

        for url in notification_urls:
            try:
                response = requests.post(url, json=notification_data, timeout=10)
                if response.status_code == 200:
                    self.logger.info(f"Notification sent to {url}")
                else:
                    self.logger.warning(f"Failed to send notification to {url}: {response.status_code}")
            except Exception as e:
                self.logger.warning(f"Error sending notification to {url}: {e}")

    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        try:
            status_data = await self.redis_client.hgetall(f"deployment:{deployment_id}")
            return {k.decode(): v.decode() for k, v in status_data.items()} if status_data else None
        except Exception as e:
            self.logger.error(f"Error getting deployment status: {e}")
            return None

    async def list_deployments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent deployments"""
        try:
            deployment_ids = await self.redis_client.zrevrange("deployments", 0, limit-1)
            deployments = []

            for deployment_id in deployment_ids:
                status = await self.get_deployment_status(deployment_id.decode())
                if status:
                    deployments.append(status)

            return deployments

        except Exception as e:
            self.logger.error(f"Error listing deployments: {e}")
            return []

    def _generate_deployment_id(self, environment: str) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{environment}-{timestamp}"

# CLI interface for deployment
async def main():
    import argparse

    parser = argparse.ArgumentParser(description='DD-RAPTOR Auto Deploy')
    parser.add_argument('--environment', required=True, choices=['dev', 'staging', 'production'])
    parser.add_argument('--strategy', required=True, choices=['blue_green', 'canary', 'rolling'])
    parser.add_argument('--image-tag', required=True)
    parser.add_argument('--replicas', type=int, default=3)
    parser.add_argument('--canary-percentage', type=int, default=10)

    args = parser.parse_args()

    # Create deployment config
    config = DeploymentConfig(
        environment=args.environment,
        strategy=DeploymentStrategy(args.strategy),
        image_tag=args.image_tag,
        replicas=args.replicas,
        canary_percentage=args.canary_percentage,
        health_check_path="/health",
        rollback_on_failure=True,
        environment_vars={
            "ENVIRONMENT": args.environment,
            "LOG_LEVEL": "INFO" if args.environment == "production" else "DEBUG"
        }
    )

    # Deploy
    deploy_system = AutoDeploySystem()
    result = await deploy_system.deploy(config)

    print(f"Deployment {result.deployment_id} completed with status: {result.status.value}")
    if result.error_message:
        print(f"Error: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
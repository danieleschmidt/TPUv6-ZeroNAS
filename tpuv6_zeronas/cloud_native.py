"""Cloud-Native Infrastructure: Multi-region deployment, auto-scaling, and global orchestration."""

import logging
import time
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from enum import Enum
import random

from .architecture import Architecture, ArchitectureSpace
from .metrics import PerformanceMetrics
from .predictor import TPUv6Predictor
from .core import SearchConfig


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ALIBABA = "alibaba"
    HYBRID = "hybrid"


class DeploymentRegion(Enum):
    """Global deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    MIDDLE_EAST = "me-south-1"
    AFRICA = "af-south-1"
    SOUTH_AMERICA = "sa-east-1"


class ServiceTier(Enum):
    """Service tier configurations."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


@dataclass
class ResourceRequirements:
    """Resource requirements specification."""
    cpu_cores: int = 4
    memory_gb: int = 16
    storage_gb: int = 100
    gpu_count: int = 0
    gpu_type: str = ""
    network_bandwidth_gbps: float = 1.0
    iops: int = 3000
    specialized_hardware: List[str] = field(default_factory=list)


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_instances: int = 1
    max_instances: int = 100
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_request_latency_ms: float = 1000.0
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    scale_up_threshold_breaches: int = 2
    scale_down_threshold_breaches: int = 3


@dataclass
class RegionConfiguration:
    """Regional deployment configuration."""
    region: DeploymentRegion
    cloud_provider: CloudProvider
    availability_zones: List[str] = field(default_factory=list)
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    scaling_policy: ScalingPolicy = field(default_factory=ScalingPolicy)
    compliance_requirements: List[str] = field(default_factory=list)
    data_residency_rules: Dict[str, str] = field(default_factory=dict)
    latency_requirements_ms: float = 100.0
    availability_target: float = 99.9
    disaster_recovery_enabled: bool = True


@dataclass
class GlobalLoadBalancer:
    """Global load balancing configuration."""
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5
    health_check_path: str = "/health"
    routing_algorithm: str = "geographic"  # geographic, round_robin, weighted, latency
    failover_threshold_seconds: int = 60
    sticky_sessions: bool = False
    cdn_enabled: bool = True
    waf_enabled: bool = True


@dataclass
class ServiceMesh:
    """Service mesh configuration for microservices."""
    enabled: bool = True
    proxy_type: str = "envoy"
    mutual_tls_enabled: bool = True
    circuit_breaker_enabled: bool = True
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "retry_timeout_ms": 1000,
        "backoff_multiplier": 2.0
    })
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        "requests_per_second": 1000,
        "burst_size": 2000
    })


class CloudNativeOrchestrator:
    """Cloud-native orchestration and deployment manager."""
    
    def __init__(self, 
                 service_tier: ServiceTier = ServiceTier.PRODUCTION,
                 global_config: Optional[Dict[str, Any]] = None):
        self.service_tier = service_tier
        self.global_config = global_config or {}
        
        # Regional deployments
        self.regional_deployments = {}
        self.active_regions = set()
        
        # Global services
        self.load_balancer = GlobalLoadBalancer()
        self.service_mesh = ServiceMesh()
        self.container_registry = None
        
        # Monitoring and telemetry
        self.telemetry_collectors = {}
        self.health_monitors = {}
        self.performance_metrics = {}
        
        # Auto-scaling state
        self.scaling_decisions = {}
        self.resource_utilization = {}
        
        # Initialize cloud-native infrastructure
        self._initialize_infrastructure()
        
    def _initialize_infrastructure(self):
        """Initialize cloud-native infrastructure components."""
        logging.info(f"Initializing cloud-native infrastructure for {self.service_tier.value}")
        
        # Setup default regional configurations
        self._setup_default_regions()
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        # Setup service mesh
        self._initialize_service_mesh()
        
        logging.info("Cloud-native infrastructure initialized successfully")
    
    def _setup_default_regions(self):
        """Setup default regional deployment configurations."""
        default_regions = {
            DeploymentRegion.US_EAST: CloudProvider.AWS,
            DeploymentRegion.EU_WEST: CloudProvider.AWS,
            DeploymentRegion.ASIA_PACIFIC: CloudProvider.GCP,
            DeploymentRegion.ASIA_NORTHEAST: CloudProvider.GCP
        }
        
        for region, provider in default_regions.items():
            config = RegionConfiguration(
                region=region,
                cloud_provider=provider,
                availability_zones=self._get_availability_zones(region, provider),
                resource_requirements=self._get_tier_resources(),
                scaling_policy=self._get_tier_scaling_policy(),
                compliance_requirements=self._get_regional_compliance(region),
                latency_requirements_ms=self._get_latency_requirements(region)
            )
            
            self.regional_deployments[region] = config
    
    def _get_availability_zones(self, region: DeploymentRegion, provider: CloudProvider) -> List[str]:
        """Get availability zones for region and provider."""
        zone_mappings = {
            (DeploymentRegion.US_EAST, CloudProvider.AWS): ["us-east-1a", "us-east-1b", "us-east-1c"],
            (DeploymentRegion.EU_WEST, CloudProvider.AWS): ["eu-west-1a", "eu-west-1b", "eu-west-1c"],
            (DeploymentRegion.ASIA_PACIFIC, CloudProvider.GCP): ["asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c"],
            (DeploymentRegion.ASIA_NORTHEAST, CloudProvider.GCP): ["asia-northeast1-a", "asia-northeast1-b", "asia-northeast1-c"]
        }
        
        return zone_mappings.get((region, provider), ["zone-a", "zone-b", "zone-c"])
    
    def _get_tier_resources(self) -> ResourceRequirements:
        """Get resource requirements based on service tier."""
        tier_resources = {
            ServiceTier.DEVELOPMENT: ResourceRequirements(
                cpu_cores=2, memory_gb=8, storage_gb=50, network_bandwidth_gbps=0.5
            ),
            ServiceTier.STAGING: ResourceRequirements(
                cpu_cores=4, memory_gb=16, storage_gb=100, network_bandwidth_gbps=1.0
            ),
            ServiceTier.PRODUCTION: ResourceRequirements(
                cpu_cores=8, memory_gb=32, storage_gb=200, gpu_count=1, network_bandwidth_gbps=2.0
            ),
            ServiceTier.ENTERPRISE: ResourceRequirements(
                cpu_cores=16, memory_gb=64, storage_gb=500, gpu_count=2, 
                network_bandwidth_gbps=5.0, specialized_hardware=["tpu", "fpga"]
            )
        }
        
        return tier_resources.get(self.service_tier, tier_resources[ServiceTier.PRODUCTION])
    
    def _get_tier_scaling_policy(self) -> ScalingPolicy:
        """Get scaling policy based on service tier."""
        tier_policies = {
            ServiceTier.DEVELOPMENT: ScalingPolicy(
                min_instances=1, max_instances=3, target_cpu_utilization=80.0
            ),
            ServiceTier.STAGING: ScalingPolicy(
                min_instances=2, max_instances=10, target_cpu_utilization=75.0
            ),
            ServiceTier.PRODUCTION: ScalingPolicy(
                min_instances=3, max_instances=50, target_cpu_utilization=70.0
            ),
            ServiceTier.ENTERPRISE: ScalingPolicy(
                min_instances=5, max_instances=100, target_cpu_utilization=60.0,
                target_request_latency_ms=500.0
            )
        }
        
        return tier_policies.get(self.service_tier, tier_policies[ServiceTier.PRODUCTION])
    
    def _get_regional_compliance(self, region: DeploymentRegion) -> List[str]:
        """Get compliance requirements for region."""
        regional_compliance = {
            DeploymentRegion.EU_WEST: ["GDPR", "ISO27001"],
            DeploymentRegion.EU_CENTRAL: ["GDPR", "ISO27001"],
            DeploymentRegion.US_EAST: ["SOC2", "HIPAA"],
            DeploymentRegion.US_WEST: ["SOC2", "CCPA"],
            DeploymentRegion.ASIA_PACIFIC: ["PDPA", "ISO27001"],
            DeploymentRegion.ASIA_NORTHEAST: ["PIPL", "ISO27001"]
        }
        
        return regional_compliance.get(region, ["ISO27001"])
    
    def _get_latency_requirements(self, region: DeploymentRegion) -> float:
        """Get latency requirements for region."""
        return 100.0  # Default 100ms
    
    def _initialize_monitoring(self):
        """Initialize monitoring and observability."""
        for region in self.regional_deployments:
            self.telemetry_collectors[region] = {
                'metrics_endpoint': f'https://metrics.{region.value}.tpuv6nas.com',
                'traces_endpoint': f'https://traces.{region.value}.tpuv6nas.com',
                'logs_endpoint': f'https://logs.{region.value}.tpuv6nas.com',
                'collection_interval_seconds': 15,
                'retention_days': 30
            }
            
            self.health_monitors[region] = {
                'health_check_url': f'https://api.{region.value}.tpuv6nas.com/health',
                'check_interval_seconds': 30,
                'timeout_seconds': 5,
                'failure_threshold': 3
            }
    
    def _initialize_service_mesh(self):
        """Initialize service mesh configuration."""
        if not self.service_mesh.enabled:
            return
        
        logging.info(f"Initializing {self.service_mesh.proxy_type} service mesh")
        
        # Configure service mesh for each region
        for region in self.regional_deployments:
            mesh_config = {
                'region': region.value,
                'proxy_type': self.service_mesh.proxy_type,
                'mutual_tls': self.service_mesh.mutual_tls_enabled,
                'circuit_breaker': self.service_mesh.circuit_breaker_enabled,
                'retry_policy': self.service_mesh.retry_policy,
                'rate_limiting': self.service_mesh.rate_limiting
            }
            
            logging.info(f"Service mesh configured for {region.value}")
    
    def deploy_to_region(self, region: DeploymentRegion, 
                        deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy TPUv6-ZeroNAS to specific region."""
        if region not in self.regional_deployments:
            raise ValueError(f"Region {region.value} not configured")
        
        regional_config = self.regional_deployments[region]
        
        logging.info(f"Deploying to {region.value} on {regional_config.cloud_provider.value}")
        
        # Simulate deployment process
        deployment_result = {
            'region': region.value,
            'cloud_provider': regional_config.cloud_provider.value,
            'deployment_id': f"deploy-{int(time.time())}-{region.value}",
            'status': 'deploying',
            'start_time': time.time(),
            'endpoints': self._create_regional_endpoints(region),
            'resources': self._provision_resources(regional_config),
            'compliance_status': self._verify_compliance(regional_config.compliance_requirements)
        }
        
        # Simulate deployment time
        time.sleep(0.5)
        
        deployment_result['status'] = 'deployed'
        deployment_result['end_time'] = time.time()
        deployment_result['deployment_duration'] = deployment_result['end_time'] - deployment_result['start_time']
        
        self.active_regions.add(region)
        
        # Initialize regional monitoring
        self._start_regional_monitoring(region)
        
        logging.info(f"Successfully deployed to {region.value}")
        return deployment_result
    
    def _create_regional_endpoints(self, region: DeploymentRegion) -> Dict[str, str]:
        """Create regional API endpoints."""
        base_domain = f"{region.value}.tpuv6nas.com"
        
        return {
            'api_endpoint': f"https://api.{base_domain}",
            'websocket_endpoint': f"wss://ws.{base_domain}",
            'metrics_endpoint': f"https://metrics.{base_domain}",
            'health_endpoint': f"https://api.{base_domain}/health",
            'docs_endpoint': f"https://docs.{base_domain}"
        }
    
    def _provision_resources(self, config: RegionConfiguration) -> Dict[str, Any]:
        """Provision cloud resources for deployment."""
        resources = config.resource_requirements
        
        # Simulate resource provisioning
        provisioned_resources = {
            'compute_instances': {
                'count': config.scaling_policy.min_instances,
                'type': f"c5.{self._get_instance_size(resources.cpu_cores, resources.memory_gb)}",
                'cpu_cores': resources.cpu_cores,
                'memory_gb': resources.memory_gb,
                'storage_gb': resources.storage_gb
            },
            'load_balancer': {
                'type': 'application',
                'scheme': 'internet-facing',
                'listeners': ['HTTP:80', 'HTTPS:443']
            },
            'database': {
                'type': 'managed',
                'engine': 'postgresql',
                'version': '13.7',
                'multi_az': config.region in [DeploymentRegion.US_EAST, DeploymentRegion.EU_WEST]
            },
            'storage': {
                'type': 'object_storage',
                'redundancy': 'multi_zone',
                'encryption': 'AES-256'
            }
        }
        
        if resources.gpu_count > 0:
            provisioned_resources['gpu_instances'] = {
                'count': resources.gpu_count,
                'type': resources.gpu_type or 'v100',
                'memory_gb': 16 * resources.gpu_count
            }
        
        if resources.specialized_hardware:
            provisioned_resources['specialized_hardware'] = {
                'types': resources.specialized_hardware,
                'availability': 'reserved'
            }
        
        return provisioned_resources
    
    def _get_instance_size(self, cpu_cores: int, memory_gb: int) -> str:
        """Determine instance size based on resource requirements."""
        if cpu_cores <= 2 and memory_gb <= 8:
            return "large"
        elif cpu_cores <= 4 and memory_gb <= 16:
            return "xlarge"
        elif cpu_cores <= 8 and memory_gb <= 32:
            return "2xlarge"
        elif cpu_cores <= 16 and memory_gb <= 64:
            return "4xlarge"
        else:
            return "8xlarge"
    
    def _verify_compliance(self, requirements: List[str]) -> Dict[str, str]:
        """Verify compliance with regional requirements."""
        compliance_status = {}
        
        for requirement in requirements:
            # Simulate compliance verification
            compliance_status[requirement] = "compliant"
        
        return compliance_status
    
    def _start_regional_monitoring(self, region: DeploymentRegion):
        """Start monitoring for regional deployment."""
        if region not in self.performance_metrics:
            self.performance_metrics[region] = {
                'cpu_utilization': [],
                'memory_utilization': [],
                'request_latency': [],
                'request_rate': [],
                'error_rate': [],
                'availability': []
            }
        
        # Start monitoring thread (simplified)
        def monitor_region():
            while region in self.active_regions:
                metrics = self._collect_regional_metrics(region)
                self.performance_metrics[region]['cpu_utilization'].append(metrics['cpu'])
                self.performance_metrics[region]['memory_utilization'].append(metrics['memory'])
                self.performance_metrics[region]['request_latency'].append(metrics['latency'])
                self.performance_metrics[region]['request_rate'].append(metrics['requests'])
                self.performance_metrics[region]['error_rate'].append(metrics['errors'])
                self.performance_metrics[region]['availability'].append(metrics['availability'])
                
                # Trigger auto-scaling if needed
                self._evaluate_auto_scaling(region, metrics)
                
                time.sleep(30)  # Monitor every 30 seconds
        
        monitoring_thread = threading.Thread(target=monitor_region, daemon=True)
        monitoring_thread.start()
    
    def _collect_regional_metrics(self, region: DeploymentRegion) -> Dict[str, float]:
        """Collect performance metrics for region."""
        # Simulate realistic metrics
        base_load = 0.3 + random.uniform(0, 0.4)  # 30-70% base load
        
        return {
            'cpu': base_load * 100,
            'memory': (base_load + 0.1) * 100,
            'latency': 50 + random.uniform(0, 100),
            'requests': 100 + random.uniform(0, 200),
            'errors': random.uniform(0, 5),
            'availability': 99.0 + random.uniform(0, 1.0)
        }
    
    def _evaluate_auto_scaling(self, region: DeploymentRegion, metrics: Dict[str, float]):
        """Evaluate and execute auto-scaling decisions."""
        if region not in self.regional_deployments:
            return
        
        scaling_policy = self.regional_deployments[region].scaling_policy
        current_instances = self._get_current_instance_count(region)
        
        # Scale up conditions
        scale_up = (
            metrics['cpu'] > scaling_policy.target_cpu_utilization or
            metrics['memory'] > scaling_policy.target_memory_utilization or
            metrics['latency'] > scaling_policy.target_request_latency_ms
        )
        
        # Scale down conditions
        scale_down = (
            metrics['cpu'] < scaling_policy.target_cpu_utilization * 0.5 and
            metrics['memory'] < scaling_policy.target_memory_utilization * 0.5 and
            metrics['latency'] < scaling_policy.target_request_latency_ms * 0.5
        )
        
        if scale_up and current_instances < scaling_policy.max_instances:
            new_count = min(scaling_policy.max_instances, current_instances + 1)
            self._scale_instances(region, new_count)
            logging.info(f"Scaled up {region.value} to {new_count} instances")
            
        elif scale_down and current_instances > scaling_policy.min_instances:
            new_count = max(scaling_policy.min_instances, current_instances - 1)
            self._scale_instances(region, new_count)
            logging.info(f"Scaled down {region.value} to {new_count} instances")
    
    def _get_current_instance_count(self, region: DeploymentRegion) -> int:
        """Get current instance count for region."""
        # Simulate getting current instance count
        return self.regional_deployments[region].scaling_policy.min_instances + random.randint(0, 3)
    
    def _scale_instances(self, region: DeploymentRegion, target_count: int):
        """Scale instances in region to target count."""
        # Simulate scaling operation
        self.scaling_decisions[region] = {
            'target_count': target_count,
            'timestamp': time.time(),
            'reason': 'auto_scaling'
        }
    
    def deploy_globally(self, target_regions: Optional[List[DeploymentRegion]] = None) -> Dict[str, Any]:
        """Deploy TPUv6-ZeroNAS globally across multiple regions."""
        if target_regions is None:
            target_regions = list(self.regional_deployments.keys())
        
        global_deployment = {
            'deployment_id': f"global-deploy-{int(time.time())}",
            'start_time': time.time(),
            'target_regions': [r.value for r in target_regions],
            'regional_deployments': {},
            'load_balancer_config': self.load_balancer.__dict__,
            'service_mesh_config': self.service_mesh.__dict__
        }
        
        # Deploy to all target regions in parallel
        with ThreadPoolExecutor(max_workers=len(target_regions)) as executor:
            future_to_region = {
                executor.submit(self.deploy_to_region, region, {}): region 
                for region in target_regions
            }
            
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    deployment_result = future.result()
                    global_deployment['regional_deployments'][region.value] = deployment_result
                except Exception as e:
                    logging.error(f"Failed to deploy to {region.value}: {e}")
                    global_deployment['regional_deployments'][region.value] = {
                        'status': 'failed',
                        'error': str(e)
                    }
        
        # Configure global load balancer
        self._configure_global_load_balancer(target_regions)
        
        # Setup global monitoring
        self._setup_global_monitoring()
        
        global_deployment['end_time'] = time.time()
        global_deployment['total_duration'] = global_deployment['end_time'] - global_deployment['start_time']
        global_deployment['successful_regions'] = len([
            r for r in global_deployment['regional_deployments'].values() 
            if r.get('status') == 'deployed'
        ])
        
        logging.info(f"Global deployment completed: {global_deployment['successful_regions']}/{len(target_regions)} regions")
        
        return global_deployment
    
    def _configure_global_load_balancer(self, regions: List[DeploymentRegion]):
        """Configure global load balancer for multi-region deployment."""
        backend_endpoints = []
        
        for region in regions:
            if region in self.active_regions:
                endpoint = f"https://api.{region.value}.tpuv6nas.com"
                backend_endpoints.append({
                    'endpoint': endpoint,
                    'region': region.value,
                    'weight': 100,  # Equal weight for all regions
                    'health_check': True
                })
        
        lb_config = {
            'global_endpoint': 'https://api.tpuv6nas.com',
            'backends': backend_endpoints,
            'routing_policy': {
                'algorithm': self.load_balancer.routing_algorithm,
                'health_check_interval': self.load_balancer.health_check_interval_seconds,
                'failover_threshold': self.load_balancer.failover_threshold_seconds
            },
            'cdn_config': {
                'enabled': self.load_balancer.cdn_enabled,
                'cache_ttl_seconds': 3600,
                'edge_locations': len(regions) * 3
            },
            'waf_config': {
                'enabled': self.load_balancer.waf_enabled,
                'rules': ['sql_injection', 'xss_protection', 'rate_limiting']
            }
        }
        
        logging.info(f"Global load balancer configured with {len(backend_endpoints)} backends")
        return lb_config
    
    def _setup_global_monitoring(self):
        """Setup global monitoring and alerting."""
        global_monitoring = {
            'aggregated_metrics': {
                'global_request_rate': 0,
                'global_error_rate': 0,
                'average_latency': 0,
                'regional_availability': {}
            },
            'alerting_rules': [
                {
                    'name': 'high_global_error_rate',
                    'condition': 'global_error_rate > 5',
                    'severity': 'critical'
                },
                {
                    'name': 'regional_outage',
                    'condition': 'regional_availability < 95',
                    'severity': 'critical'
                },
                {
                    'name': 'high_latency',
                    'condition': 'average_latency > 1000',
                    'severity': 'warning'
                }
            ]
        }
        
        logging.info("Global monitoring and alerting configured")
        return global_monitoring
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global deployment status."""
        status = {
            'deployment_summary': {
                'total_regions': len(self.regional_deployments),
                'active_regions': len(self.active_regions),
                'service_tier': self.service_tier.value,
                'global_endpoint': 'https://api.tpuv6nas.com'
            },
            'regional_status': {},
            'global_metrics': self._calculate_global_metrics(),
            'health_status': self._check_global_health(),
            'compliance_status': self._check_global_compliance()
        }
        
        # Get status for each region
        for region in self.active_regions:
            status['regional_status'][region.value] = {
                'status': 'healthy',
                'endpoint': f"https://api.{region.value}.tpuv6nas.com",
                'current_instances': self._get_current_instance_count(region),
                'recent_metrics': self._get_recent_metrics(region)
            }
        
        return status
    
    def _calculate_global_metrics(self) -> Dict[str, float]:
        """Calculate aggregated global metrics."""
        if not self.active_regions:
            return {}
        
        total_requests = 0
        total_errors = 0
        total_latency = 0
        total_availability = 0
        
        for region in self.active_regions:
            if region in self.performance_metrics:
                metrics = self.performance_metrics[region]
                
                if metrics['request_rate']:
                    total_requests += metrics['request_rate'][-1]
                if metrics['error_rate']:
                    total_errors += metrics['error_rate'][-1]
                if metrics['request_latency']:
                    total_latency += metrics['request_latency'][-1]
                if metrics['availability']:
                    total_availability += metrics['availability'][-1]
        
        num_regions = len(self.active_regions)
        
        return {
            'global_request_rate': total_requests,
            'global_error_rate': total_errors / num_regions if num_regions > 0 else 0,
            'average_latency': total_latency / num_regions if num_regions > 0 else 0,
            'global_availability': total_availability / num_regions if num_regions > 0 else 0
        }
    
    def _check_global_health(self) -> Dict[str, Any]:
        """Check global health status."""
        healthy_regions = 0
        total_regions = len(self.active_regions)
        
        for region in self.active_regions:
            # Simulate health check
            if random.random() > 0.05:  # 95% chance of being healthy
                healthy_regions += 1
        
        health_percentage = (healthy_regions / total_regions * 100) if total_regions > 0 else 0
        
        return {
            'overall_status': 'healthy' if health_percentage >= 80 else 'degraded',
            'healthy_regions': healthy_regions,
            'total_regions': total_regions,
            'health_percentage': health_percentage,
            'last_check': time.time()
        }
    
    def _check_global_compliance(self) -> Dict[str, str]:
        """Check global compliance status."""
        compliance_status = {}
        
        for region in self.active_regions:
            regional_config = self.regional_deployments[region]
            for requirement in regional_config.compliance_requirements:
                if requirement not in compliance_status:
                    compliance_status[requirement] = 'compliant'
        
        return compliance_status
    
    def _get_recent_metrics(self, region: DeploymentRegion) -> Dict[str, float]:
        """Get recent metrics for region."""
        if region not in self.performance_metrics:
            return {}
        
        metrics = self.performance_metrics[region]
        
        return {
            'cpu_utilization': metrics['cpu_utilization'][-1] if metrics['cpu_utilization'] else 0,
            'memory_utilization': metrics['memory_utilization'][-1] if metrics['memory_utilization'] else 0,
            'request_latency': metrics['request_latency'][-1] if metrics['request_latency'] else 0,
            'request_rate': metrics['request_rate'][-1] if metrics['request_rate'] else 0,
            'error_rate': metrics['error_rate'][-1] if metrics['error_rate'] else 0,
            'availability': metrics['availability'][-1] if metrics['availability'] else 0
        }
    
    def scale_region(self, region: DeploymentRegion, target_instances: int) -> Dict[str, Any]:
        """Manually scale a specific region."""
        if region not in self.active_regions:
            raise ValueError(f"Region {region.value} is not active")
        
        scaling_policy = self.regional_deployments[region].scaling_policy
        
        if target_instances < scaling_policy.min_instances:
            target_instances = scaling_policy.min_instances
        elif target_instances > scaling_policy.max_instances:
            target_instances = scaling_policy.max_instances
        
        self._scale_instances(region, target_instances)
        
        return {
            'region': region.value,
            'target_instances': target_instances,
            'timestamp': time.time(),
            'reason': 'manual_scaling'
        }
    
    def enable_disaster_recovery(self, primary_region: DeploymentRegion, 
                               backup_region: DeploymentRegion) -> Dict[str, Any]:
        """Enable disaster recovery between regions."""
        dr_config = {
            'primary_region': primary_region.value,
            'backup_region': backup_region.value,
            'replication_enabled': True,
            'failover_threshold_minutes': 5,
            'recovery_time_objective_minutes': 15,
            'recovery_point_objective_minutes': 1,
            'automated_failover': True,
            'data_sync_interval_seconds': 60
        }
        
        logging.info(f"Disaster recovery enabled: {primary_region.value} -> {backup_region.value}")
        return dr_config


# Factory functions
def create_cloud_native_orchestrator(service_tier: ServiceTier = ServiceTier.PRODUCTION,
                                   global_config: Optional[Dict[str, Any]] = None) -> CloudNativeOrchestrator:
    """Create cloud-native orchestrator with specified service tier."""
    return CloudNativeOrchestrator(service_tier, global_config)


def create_regional_config(region: DeploymentRegion,
                         cloud_provider: CloudProvider,
                         custom_requirements: Optional[ResourceRequirements] = None) -> RegionConfiguration:
    """Create custom regional configuration."""
    return RegionConfiguration(
        region=region,
        cloud_provider=cloud_provider,
        resource_requirements=custom_requirements or ResourceRequirements()
    )
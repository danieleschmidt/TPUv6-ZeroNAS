"""Auto-scaling and load balancing for TPUv6-ZeroNAS."""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

from .parallel import ParallelEvaluator, WorkerConfig
from .monitoring import SearchMonitor, get_profiler
from .advanced_monitoring import get_advanced_monitor, record_operation_performance
from .error_handling import safe_operation, robust_operation


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    CONSERVATIVE = "conservative"  # Scale slowly, prefer stability
    AGGRESSIVE = "aggressive"      # Scale quickly, prefer performance
    BALANCED = "balanced"          # Balance between stability and performance
    CUSTOM = "custom"              # Custom scaling rules


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    current_load: float          # Current system load (0.0 - 1.0)
    avg_response_time_ms: float  # Average response time
    queue_depth: int             # Number of pending tasks
    error_rate: float            # Error rate (0.0 - 1.0)
    resource_utilization: float  # Resource utilization (0.0 - 1.0)
    throughput_ops_per_sec: float # Current throughput


@dataclass
class ScalingDecision:
    """Decision made by auto-scaler."""
    action: str                  # 'scale_up', 'scale_down', 'no_change'
    target_workers: int          # Target number of workers
    reason: str                  # Reason for decision
    confidence: float            # Confidence in decision (0.0 - 1.0)


class AutoScaler:
    """Intelligent auto-scaling system for search operations."""
    
    def __init__(self, 
                 policy: ScalingPolicy = ScalingPolicy.BALANCED,
                 min_workers: int = 1,
                 max_workers: int = 8,
                 check_interval: float = 10.0):
        self.policy = policy
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_workers = min_workers
        self.scaling_history: List[ScalingDecision] = []
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scaling_time = 0
        
        # Scaling thresholds based on policy
        self._setup_thresholds()
        
        # Monitoring
        self.is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self.advanced_monitor = get_advanced_monitor()
    
    def _setup_thresholds(self):
        """Setup scaling thresholds based on policy."""
        if self.policy == ScalingPolicy.CONSERVATIVE:
            self.scale_up_thresholds = {
                'load_threshold': 0.8,
                'response_time_ms': 2000,
                'queue_depth': 20,
                'error_rate': 0.05,
                'cooldown_seconds': 60
            }
            self.scale_down_thresholds = {
                'load_threshold': 0.3,
                'response_time_ms': 500,
                'queue_depth': 5,
                'error_rate': 0.01,
                'cooldown_seconds': 120
            }
        elif self.policy == ScalingPolicy.AGGRESSIVE:
            self.scale_up_thresholds = {
                'load_threshold': 0.6,
                'response_time_ms': 1000,
                'queue_depth': 10,
                'error_rate': 0.03,
                'cooldown_seconds': 30
            }
            self.scale_down_thresholds = {
                'load_threshold': 0.4,
                'response_time_ms': 300,
                'queue_depth': 2,
                'error_rate': 0.005,
                'cooldown_seconds': 60
            }
        else:  # BALANCED
            self.scale_up_thresholds = {
                'load_threshold': 0.7,
                'response_time_ms': 1500,
                'queue_depth': 15,
                'error_rate': 0.04,
                'cooldown_seconds': 45
            }
            self.scale_down_thresholds = {
                'load_threshold': 0.35,
                'response_time_ms': 400,
                'queue_depth': 3,
                'error_rate': 0.01,
                'cooldown_seconds': 90
            }
    
    def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics for scaling decisions."""
        try:
            # Get performance summary from advanced monitor
            health_summary = self.advanced_monitor.get_health_summary()
            perf_summary = health_summary.get('performance_summary', {})
            
            # Calculate current load (approximation)
            current_load = 1.0 - perf_summary.get('success_rate', 1.0)
            if current_load < 0.1:
                current_load = 0.1  # Minimum baseline load
            
            # Get average response time
            avg_response_time = perf_summary.get('avg_duration_ms', 100)
            
            # Estimate queue depth (simplified)
            queue_depth = max(0, int(current_load * 20))
            
            # Calculate error rate
            error_rate = 1.0 - perf_summary.get('success_rate', 1.0)
            
            # Estimate resource utilization
            resource_util = min(current_load * 1.2, 1.0)
            
            # Calculate throughput
            total_ops = perf_summary.get('total_operations', 0)
            time_window = 60  # 1 minute window
            throughput = total_ops / time_window if time_window > 0 else 0
            
            metrics = ScalingMetrics(
                current_load=current_load,
                avg_response_time_ms=avg_response_time,
                queue_depth=queue_depth,
                error_rate=error_rate,
                resource_utilization=resource_util,
                throughput_ops_per_sec=throughput
            )
            
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to collect scaling metrics: {e}")
            # Return default metrics
            return ScalingMetrics(
                current_load=0.5,
                avg_response_time_ms=1000,
                queue_depth=5,
                error_rate=0.02,
                resource_utilization=0.5,
                throughput_ops_per_sec=1.0
            )
    
    def make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Make scaling decision based on current metrics."""
        current_time = time.time()
        
        # Check if we're in cooldown period
        time_since_last_scaling = current_time - self.last_scaling_time
        min_cooldown = min(self.scale_up_thresholds['cooldown_seconds'],
                          self.scale_down_thresholds['cooldown_seconds'])
        
        if time_since_last_scaling < min_cooldown:
            return ScalingDecision(
                action='no_change',
                target_workers=self.current_workers,
                reason=f'Cooldown period ({time_since_last_scaling:.1f}s < {min_cooldown}s)',
                confidence=1.0
            )
        
        # Evaluate scale up conditions
        scale_up_score = 0
        scale_up_reasons = []
        
        if metrics.current_load > self.scale_up_thresholds['load_threshold']:
            scale_up_score += 1
            scale_up_reasons.append(f"High load ({metrics.current_load:.2f})")
        
        if metrics.avg_response_time_ms > self.scale_up_thresholds['response_time_ms']:
            scale_up_score += 1
            scale_up_reasons.append(f"Slow response ({metrics.avg_response_time_ms:.1f}ms)")
        
        if metrics.queue_depth > self.scale_up_thresholds['queue_depth']:
            scale_up_score += 1
            scale_up_reasons.append(f"Queue backlog ({metrics.queue_depth})")
        
        if metrics.error_rate > self.scale_up_thresholds['error_rate']:
            scale_up_score += 0.5  # Errors might not always need more workers
            scale_up_reasons.append(f"High error rate ({metrics.error_rate:.3f})")
        
        # Evaluate scale down conditions
        scale_down_score = 0
        scale_down_reasons = []
        
        if metrics.current_load < self.scale_down_thresholds['load_threshold']:
            scale_down_score += 1
            scale_down_reasons.append(f"Low load ({metrics.current_load:.2f})")
        
        if metrics.avg_response_time_ms < self.scale_down_thresholds['response_time_ms']:
            scale_down_score += 1
            scale_down_reasons.append(f"Fast response ({metrics.avg_response_time_ms:.1f}ms)")
        
        if metrics.queue_depth < self.scale_down_thresholds['queue_depth']:
            scale_down_score += 1
            scale_down_reasons.append(f"Empty queue ({metrics.queue_depth})")
        
        # Make decision
        if scale_up_score >= 2 and self.current_workers < self.max_workers:
            target_workers = min(self.current_workers + 1, self.max_workers)
            return ScalingDecision(
                action='scale_up',
                target_workers=target_workers,
                reason='; '.join(scale_up_reasons),
                confidence=scale_up_score / 3.0
            )
        
        elif scale_down_score >= 2 and self.current_workers > self.min_workers:
            target_workers = max(self.current_workers - 1, self.min_workers)
            return ScalingDecision(
                action='scale_down',
                target_workers=target_workers,
                reason='; '.join(scale_down_reasons),
                confidence=scale_down_score / 3.0
            )
        
        else:
            return ScalingDecision(
                action='no_change',
                target_workers=self.current_workers,
                reason='Metrics within normal ranges',
                confidence=0.8
            )
    
    def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        if decision.action == 'no_change':
            return True
        
        try:
            old_workers = self.current_workers
            self.current_workers = decision.target_workers
            self.last_scaling_time = time.time()
            
            self.scaling_history.append(decision)
            if len(self.scaling_history) > 50:
                self.scaling_history.pop(0)
            
            self.logger.info(
                f"Scaling {decision.action}: {old_workers} -> {decision.target_workers} workers. "
                f"Reason: {decision.reason} (confidence: {decision.confidence:.2f})"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling decision: {e}")
            return False
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations based on historical data."""
        if not self.metrics_history:
            return {'status': 'insufficient_data'}
        
        recent_metrics = self.metrics_history[-10:]
        avg_load = sum(m.current_load for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.avg_response_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        recommendations = []
        
        if avg_load > 0.8:
            recommendations.append("Consider increasing max_workers limit")
        elif avg_load < 0.2:
            recommendations.append("Consider reducing min_workers to save resources")
        
        if avg_response_time > 2000:
            recommendations.append("Response times high - consider scaling policy tuning")
        
        if avg_error_rate > 0.05:
            recommendations.append("High error rate - investigate root cause before scaling")
        
        return {
            'avg_load': avg_load,
            'avg_response_time_ms': avg_response_time,
            'avg_error_rate': avg_error_rate,
            'current_workers': self.current_workers,
            'recommendations': recommendations,
            'recent_scaling_actions': [d.action for d in self.scaling_history[-5:]]
        }
    
    def start_monitoring(self):
        """Start continuous auto-scaling monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Auto-scaling monitoring stopped")
    
    def _scaling_loop(self):
        """Background auto-scaling loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                
                # Make scaling decision
                decision = self.make_scaling_decision(metrics)
                
                # Execute decision if needed
                if decision.action != 'no_change':
                    self.execute_scaling_decision(decision)
                
                # Wait before next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Auto-scaling loop error: {e}")
                time.sleep(self.check_interval * 2)  # Wait longer on error


class LoadBalancer:
    """Intelligent load balancer for distributing search tasks."""
    
    def __init__(self, balancing_strategy: str = "least_connections"):
        self.balancing_strategy = balancing_strategy
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
    @record_operation_performance("load_balancer", "distribute_tasks")
    def distribute_tasks(self, tasks: List[Any], workers: List[Any]) -> Dict[str, List[Any]]:
        """Distribute tasks among workers using load balancing."""
        if not tasks or not workers:
            return {}
        
        distribution = {str(i): [] for i in range(len(workers))}
        
        if self.balancing_strategy == "round_robin":
            for i, task in enumerate(tasks):
                worker_id = str(i % len(workers))
                distribution[worker_id].append(task)
                
        elif self.balancing_strategy == "least_connections":
            # Simplified: assume equal initial load
            for i, task in enumerate(tasks):
                # Find worker with least tasks assigned so far
                min_worker = min(distribution.keys(), 
                               key=lambda w: len(distribution[w]))
                distribution[min_worker].append(task)
                
        elif self.balancing_strategy == "weighted_round_robin":
            # Weighted based on worker performance (simplified)
            weights = [1.0] * len(workers)  # Equal weights for now
            total_weight = sum(weights)
            
            for task in tasks:
                # Choose worker based on weights
                import random
                rand_val = random.random() * total_weight
                cumulative = 0
                chosen_worker = 0
                
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if rand_val <= cumulative:
                        chosen_worker = i
                        break
                
                distribution[str(chosen_worker)].append(task)
        
        else:  # Default to round robin
            for i, task in enumerate(tasks):
                worker_id = str(i % len(workers))
                distribution[worker_id].append(task)
        
        self.logger.debug(f"Distributed {len(tasks)} tasks among {len(workers)} workers")
        return distribution


class AdaptiveSearchOptimizer:
    """Adaptive optimizer that learns from search patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.search_patterns: Dict[str, List[float]] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
    @safe_operation(component="adaptive_optimizer")
    def optimize_search_parameters(self, current_performance: Dict[str, float]) -> Dict[str, Any]:
        """Optimize search parameters based on current performance."""
        optimizations = {}
        
        # Analyze current performance
        latency = current_performance.get('avg_latency_ms', 1000)
        success_rate = current_performance.get('success_rate', 0.9)
        throughput = current_performance.get('throughput', 1.0)
        
        # Adaptive population size
        if success_rate > 0.95 and latency < 500:
            # Good performance, can increase population for better results
            optimizations['population_size_multiplier'] = 1.2
        elif success_rate < 0.8 or latency > 2000:
            # Poor performance, reduce population to stabilize
            optimizations['population_size_multiplier'] = 0.8
        else:
            optimizations['population_size_multiplier'] = 1.0
        
        # Adaptive iteration count
        if throughput > 5.0:
            # High throughput, can afford more iterations
            optimizations['max_iterations_multiplier'] = 1.1
        elif throughput < 1.0:
            # Low throughput, reduce iterations
            optimizations['max_iterations_multiplier'] = 0.9
        else:
            optimizations['max_iterations_multiplier'] = 1.0
        
        # Adaptive caching strategy
        if latency > 1000:
            optimizations['enable_aggressive_caching'] = True
        else:
            optimizations['enable_aggressive_caching'] = False
        
        self.optimization_history.append({
            'timestamp': time.time(),
            'performance': current_performance,
            'optimizations': optimizations
        })
        
        return optimizations


# Global instances
_auto_scaler = AutoScaler()
_load_balancer = LoadBalancer()
_adaptive_optimizer = AdaptiveSearchOptimizer()


def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    return _auto_scaler


def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance."""
    return _load_balancer


def get_adaptive_optimizer() -> AdaptiveSearchOptimizer:
    """Get global adaptive optimizer instance."""
    return _adaptive_optimizer


def enable_auto_scaling(policy: ScalingPolicy = ScalingPolicy.BALANCED,
                       min_workers: int = 1, 
                       max_workers: int = 8) -> AutoScaler:
    """Enable auto-scaling with specified configuration."""
    scaler = get_auto_scaler()
    scaler.policy = policy
    scaler.min_workers = min_workers
    scaler.max_workers = max_workers
    scaler._setup_thresholds()
    
    # Start monitoring if not already started
    if not scaler.is_monitoring:
        scaler.start_monitoring()
    
    return scaler
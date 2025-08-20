"""Adaptive scaling and performance optimization for TPUv6-ZeroNAS."""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import math

@dataclass
class ScalingMetrics:
    """Metrics for adaptive scaling decisions."""
    throughput: float = 0.0
    latency_p95: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ScalingConfig:
    """Configuration for adaptive scaling."""
    target_throughput: float = 100.0
    max_latency_p95: float = 1.0
    memory_threshold: float = 0.8
    cpu_threshold: float = 0.8
    scale_up_threshold: float = 0.9
    scale_down_threshold: float = 0.3
    min_replicas: int = 1
    max_replicas: int = 8
    cooldown_period: float = 30.0

class AdaptiveScaler:
    """Adaptive scaling system for TPUv6-ZeroNAS components."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history: deque = deque(maxlen=100)
        self.scaling_history: List[Tuple[float, int]] = []
        self.current_replicas = config.min_replicas
        self.last_scale_time = 0.0
        self.lock = threading.Lock()
        
        # Scaling decisions cache
        self.scaling_decisions: deque = deque(maxlen=10)
        
        # Performance predictors
        self.throughput_predictor = ThroughputPredictor()
        self.resource_predictor = ResourcePredictor()
    
    def update_metrics(self, metrics: ScalingMetrics) -> None:
        """Update scaling metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
            
            # Trigger scaling decision if needed
            self._evaluate_scaling_decision()
    
    def _evaluate_scaling_decision(self) -> None:
        """Evaluate whether scaling is needed."""
        if len(self.metrics_history) < 5:
            return
        
        current_time = time.time()
        if current_time - self.last_scale_time < self.config.cooldown_period:
            return
        
        recent_metrics = list(self.metrics_history)[-5:]
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_latency = sum(m.latency_p95 for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        
        # Scale up conditions
        if (avg_cpu > self.config.scale_up_threshold or
            avg_memory > self.config.scale_up_threshold or
            avg_latency > self.config.max_latency_p95 or
            avg_throughput < self.config.target_throughput * 0.8):
            
            if self.current_replicas < self.config.max_replicas:
                self._scale_up()
        
        # Scale down conditions
        elif (avg_cpu < self.config.scale_down_threshold and
              avg_memory < self.config.scale_down_threshold and
              avg_latency < self.config.max_latency_p95 * 0.5 and
              avg_throughput > self.config.target_throughput):
            
            if self.current_replicas > self.config.min_replicas:
                self._scale_down()
    
    def _scale_up(self) -> None:
        """Scale up resources."""
        new_replicas = min(self.current_replicas + 1, self.config.max_replicas)
        if new_replicas != self.current_replicas:
            self.current_replicas = new_replicas
            self.last_scale_time = time.time()
            self.scaling_history.append((time.time(), new_replicas))
            self.logger.info(f"Scaled up to {new_replicas} replicas")
    
    def _scale_down(self) -> None:
        """Scale down resources."""
        new_replicas = max(self.current_replicas - 1, self.config.min_replicas)
        if new_replicas != self.current_replicas:
            self.current_replicas = new_replicas
            self.last_scale_time = time.time()
            self.scaling_history.append((time.time(), new_replicas))
            self.logger.info(f"Scaled down to {new_replicas} replicas")
    
    def get_current_scale(self) -> int:
        """Get current replica count."""
        with self.lock:
            return self.current_replicas
    
    def predict_optimal_scale(self, target_metrics: Dict[str, float]) -> int:
        """Predict optimal scaling for target metrics."""
        if not self.metrics_history:
            return self.config.min_replicas
        
        # Use throughput predictor
        predicted_throughput = self.throughput_predictor.predict(
            target_metrics.get('workload', 100.0),
            self.current_replicas
        )
        
        # Use resource predictor
        predicted_resources = self.resource_predictor.predict(
            target_metrics.get('workload', 100.0),
            self.current_replicas
        )
        
        # Calculate optimal replicas based on predictions
        throughput_replicas = math.ceil(
            target_metrics.get('target_throughput', 100.0) / 
            max(predicted_throughput, 1.0)
        )
        
        resource_replicas = math.ceil(
            predicted_resources.get('cpu_usage', 0.5) / 
            self.config.cpu_threshold
        )
        
        optimal_replicas = max(throughput_replicas, resource_replicas)
        return max(self.config.min_replicas, min(optimal_replicas, self.config.max_replicas))

class ThroughputPredictor:
    """Predict throughput based on historical data."""
    
    def __init__(self):
        self.history: Dict[int, List[float]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def update(self, replicas: int, throughput: float) -> None:
        """Update throughput history."""
        self.history[replicas].append(throughput)
        # Keep only recent data
        if len(self.history[replicas]) > 50:
            self.history[replicas] = self.history[replicas][-50:]
    
    def predict(self, workload: float, replicas: int) -> float:
        """Predict throughput for given workload and replicas."""
        if replicas not in self.history or not self.history[replicas]:
            # Fallback prediction: assume linear scaling
            return workload * replicas * 0.8  # 80% efficiency
        
        recent_throughput = self.history[replicas][-10:]  # Last 10 measurements
        avg_throughput = sum(recent_throughput) / len(recent_throughput)
        
        # Scale prediction based on workload
        baseline_workload = 100.0  # Assumption
        scaling_factor = workload / baseline_workload
        
        return avg_throughput * scaling_factor

class ResourcePredictor:
    """Predict resource usage based on historical data."""
    
    def __init__(self):
        self.cpu_history: Dict[int, List[float]] = defaultdict(list)
        self.memory_history: Dict[int, List[float]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def update(self, replicas: int, cpu_usage: float, memory_usage: float) -> None:
        """Update resource usage history."""
        self.cpu_history[replicas].append(cpu_usage)
        self.memory_history[replicas].append(memory_usage)
        
        # Keep only recent data
        for history in [self.cpu_history, self.memory_history]:
            if len(history[replicas]) > 50:
                history[replicas] = history[replicas][-50:]
    
    def predict(self, workload: float, replicas: int) -> Dict[str, float]:
        """Predict resource usage for given workload and replicas."""
        # CPU prediction
        if replicas in self.cpu_history and self.cpu_history[replicas]:
            recent_cpu = self.cpu_history[replicas][-10:]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
        else:
            avg_cpu = 0.5  # Fallback
        
        # Memory prediction
        if replicas in self.memory_history and self.memory_history[replicas]:
            recent_memory = self.memory_history[replicas][-10:]
            avg_memory = sum(recent_memory) / len(recent_memory)
        else:
            avg_memory = 0.3  # Fallback
        
        # Scale based on workload
        baseline_workload = 100.0
        scaling_factor = workload / baseline_workload
        
        return {
            'cpu_usage': min(avg_cpu * scaling_factor, 1.0),
            'memory_usage': min(avg_memory * scaling_factor, 1.0)
        }

class LoadBalancer:
    """Intelligent load balancing for scaled components."""
    
    def __init__(self, replicas: List[Any]):
        self.replicas = replicas
        self.replica_metrics: Dict[int, ScalingMetrics] = {}
        self.request_counts: Dict[int, int] = defaultdict(int)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_replica(self, replica: Any) -> None:
        """Add a new replica to the pool."""
        with self.lock:
            self.replicas.append(replica)
            replica_id = len(self.replicas) - 1
            self.replica_metrics[replica_id] = ScalingMetrics()
            self.logger.info(f"Added replica {replica_id}")
    
    def remove_replica(self, replica_id: int) -> None:
        """Remove a replica from the pool."""
        with self.lock:
            if 0 <= replica_id < len(self.replicas):
                self.replicas.pop(replica_id)
                if replica_id in self.replica_metrics:
                    del self.replica_metrics[replica_id]
                if replica_id in self.request_counts:
                    del self.request_counts[replica_id]
                self.logger.info(f"Removed replica {replica_id}")
    
    def get_best_replica(self) -> Tuple[int, Any]:
        """Get the best replica for next request using weighted round-robin."""
        with self.lock:
            if not self.replicas:
                raise RuntimeError("No replicas available")
            
            # Find replica with lowest load
            best_replica_id = 0
            best_score = float('inf')
            
            for i, replica in enumerate(self.replicas):
                if i in self.replica_metrics:
                    metrics = self.replica_metrics[i]
                    # Score combines CPU, memory, and request count
                    score = (metrics.cpu_usage * 0.4 + 
                            metrics.memory_usage * 0.3 + 
                            self.request_counts[i] * 0.3)
                else:
                    score = 0.0  # New replica gets preference
                
                if score < best_score:
                    best_score = score
                    best_replica_id = i
            
            # Update request count
            self.request_counts[best_replica_id] += 1
            
            return best_replica_id, self.replicas[best_replica_id]
    
    def update_replica_metrics(self, replica_id: int, metrics: ScalingMetrics) -> None:
        """Update metrics for a specific replica."""
        with self.lock:
            if replica_id in self.replica_metrics:
                self.replica_metrics[replica_id] = metrics

class AutoScalingManager:
    """Central manager for auto-scaling functionality."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.scaler = AdaptiveScaler(config)
        self.load_balancer = None
        self.logger = logging.getLogger(__name__)
        self.monitoring_thread = None
        self.running = False
    
    def start(self, replicas: List[Any]) -> None:
        """Start auto-scaling with initial replicas."""
        self.load_balancer = LoadBalancer(replicas)
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Auto-scaling manager started")
    
    def stop(self) -> None:
        """Stop auto-scaling."""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Auto-scaling manager stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring and scaling loop."""
        while self.running:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Update scaler
                self.scaler.update_metrics(metrics)
                
                # Check if scaling is needed
                current_replicas = len(self.load_balancer.replicas)
                target_replicas = self.scaler.get_current_scale()
                
                if target_replicas > current_replicas:
                    self._scale_up(target_replicas - current_replicas)
                elif target_replicas < current_replicas:
                    self._scale_down(current_replicas - target_replicas)
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaling monitoring error: {e}")
                time.sleep(5.0)
    
    def _collect_system_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            process = psutil.Process()
            cpu_usage = process.cpu_percent() / 100.0
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / (1024 * 1024 * 1024)  # GB
        except ImportError:
            cpu_usage = 0.5  # Mock values
            memory_usage = 0.3
        
        return ScalingMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            throughput=0.0,  # Would be updated by actual workload
            latency_p95=0.0,
            error_rate=0.0,
            queue_depth=0
        )
    
    def _scale_up(self, count: int) -> None:
        """Scale up by adding replicas."""
        self.logger.info(f"Scaling up by {count} replicas")
        # In a real implementation, this would create new replica instances
        # For now, we'll just log the intent
        
    def _scale_down(self, count: int) -> None:
        """Scale down by removing replicas."""
        self.logger.info(f"Scaling down by {count} replicas")
        # In a real implementation, this would gracefully remove replicas
        # For now, we'll just log the intent

# Global scaling manager instance
_scaling_manager = None

def get_scaling_manager(config: Optional[ScalingConfig] = None) -> AutoScalingManager:
    """Get global scaling manager instance."""
    global _scaling_manager
    if _scaling_manager is None:
        _scaling_manager = AutoScalingManager(config or ScalingConfig())
    return _scaling_manager

def create_scaling_config(**kwargs) -> ScalingConfig:
    """Create scaling configuration with custom parameters."""
    return ScalingConfig(**kwargs)
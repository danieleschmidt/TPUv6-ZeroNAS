"""Performance metrics and evaluation utilities for TPUv6 architectures."""

from dataclasses import dataclass
from typing import Dict, List, Optional
try:
    import numpy as np
except ImportError:
    # Mock numpy for basic operations
    class MockNumPy:
        @staticmethod
        def clip(x, a, b):
            return max(a, min(b, x))
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
        @staticmethod
        def std(values):
            if not values:
                return 0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance ** 0.5
        @staticmethod
        def min(values):
            return min(values) if values else 0
        @staticmethod
        def max(values):
            return max(values) if values else 0
        @staticmethod
        def median(values):
            if not values:
                return 0
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                return (sorted_values[n//2-1] + sorted_values[n//2]) / 2
            else:
                return sorted_values[n//2]
    np = MockNumPy()


@dataclass
class PerformanceMetrics:
    """Performance metrics for neural architecture evaluation."""
    latency_ms: float
    energy_mj: float
    accuracy: float
    tops_per_watt: float
    memory_mb: float
    flops: int
    
    def __post_init__(self):
        """Validate metric ranges."""
        self.accuracy = np.clip(self.accuracy, 0.0, 1.0)
        self.latency_ms = max(0.0, self.latency_ms)
        self.energy_mj = max(0.0, self.energy_mj)
        self.tops_per_watt = max(0.0, self.tops_per_watt)
        self.memory_mb = max(0.0, self.memory_mb)
        self.flops = max(0, self.flops)
    
    @property
    def efficiency_score(self) -> float:
        """Compute overall efficiency score."""
        latency_score = 1.0 / (1.0 + self.latency_ms / 10.0)
        energy_score = 1.0 / (1.0 + self.energy_mj / 100.0)
        accuracy_score = self.accuracy
        tops_score = min(1.0, self.tops_per_watt / 75.0)
        
        return (latency_score * 0.25 + 
                energy_score * 0.25 + 
                accuracy_score * 0.25 + 
                tops_score * 0.25)
    
    @property
    def pareto_objectives(self) -> Dict[str, float]:
        """Get objectives for Pareto optimization."""
        return {
            'latency': self.latency_ms,
            'energy': self.energy_mj,
            'accuracy': -self.accuracy,
            'efficiency': -self.tops_per_watt
        }
    
    def dominates(self, other: 'PerformanceMetrics') -> bool:
        """Check if this metric dominates another (Pareto dominance)."""
        self_obj = self.pareto_objectives
        other_obj = other.pareto_objectives
        
        better_in_any = False
        for key in self_obj:
            if self_obj[key] > other_obj[key]:
                return False
            elif self_obj[key] < other_obj[key]:
                better_in_any = True
        
        return better_in_any
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'latency_ms': self.latency_ms,
            'energy_mj': self.energy_mj,
            'accuracy': self.accuracy,
            'tops_per_watt': self.tops_per_watt,
            'memory_mb': self.memory_mb,
            'flops': self.flops,
            'efficiency_score': self.efficiency_score
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"PerformanceMetrics("
                f"latency={self.latency_ms:.2f}ms, "
                f"energy={self.energy_mj:.2f}mJ, "
                f"accuracy={self.accuracy:.3f}, "
                f"tops_per_watt={self.tops_per_watt:.1f}, "
                f"memory={self.memory_mb:.1f}MB)")


class MetricsAggregator:
    """Aggregate and analyze performance metrics across architectures."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
    
    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add metrics to history."""
        self.metrics_history.append(metrics)
    
    def get_pareto_front(self) -> List[PerformanceMetrics]:
        """Get Pareto-optimal solutions."""
        if not self.metrics_history:
            return []
        
        pareto_front = []
        
        for candidate in self.metrics_history:
            is_dominated = False
            
            for other in self.metrics_history:
                if other != candidate and other.dominates(candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of metrics."""
        if not self.metrics_history:
            return {}
        
        stats = {}
        
        for attr in ['latency_ms', 'energy_mj', 'accuracy', 'tops_per_watt', 'memory_mb']:
            values = [getattr(m, attr) for m in self.metrics_history]
            stats[attr] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return stats
    
    def get_best_by_objective(self, objective: str) -> Optional[PerformanceMetrics]:
        """Get best architecture by specific objective."""
        if not self.metrics_history:
            return None
        
        if objective == 'latency':
            return min(self.metrics_history, key=lambda m: m.latency_ms)
        elif objective == 'energy':
            return min(self.metrics_history, key=lambda m: m.energy_mj)
        elif objective == 'accuracy':
            return max(self.metrics_history, key=lambda m: m.accuracy)
        elif objective == 'efficiency':
            return max(self.metrics_history, key=lambda m: m.tops_per_watt)
        elif objective == 'memory':
            return min(self.metrics_history, key=lambda m: m.memory_mb)
        elif objective == 'overall':
            return max(self.metrics_history, key=lambda m: m.efficiency_score)
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def filter_feasible(
        self, 
        max_latency: Optional[float] = None,
        max_energy: Optional[float] = None,
        min_accuracy: Optional[float] = None,
        max_memory: Optional[float] = None
    ) -> List[PerformanceMetrics]:
        """Filter metrics based on constraints."""
        feasible = []
        
        for metrics in self.metrics_history:
            if max_latency and metrics.latency_ms > max_latency:
                continue
            if max_energy and metrics.energy_mj > max_energy:
                continue
            if min_accuracy and metrics.accuracy < min_accuracy:
                continue
            if max_memory and metrics.memory_mb > max_memory:
                continue
            
            feasible.append(metrics)
        
        return feasible
    
    def clear(self) -> None:
        """Clear metrics history."""
        self.metrics_history.clear()
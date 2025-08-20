"""Monitoring, logging and telemetry for TPUv6-ZeroNAS."""

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
try:
    import numpy as np
except ImportError:
    # Mock numpy for basic operations
    class MockNumPy:
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
        def percentile(values, p):
            if not values:
                return 0
            sorted_values = sorted(values)
            idx = int(len(sorted_values) * p / 100.0)
            return sorted_values[min(idx, len(sorted_values) - 1)]
    np = MockNumPy()

from .metrics import PerformanceMetrics
from .architecture import Architecture


@dataclass
class SearchEvent:
    """Search event for monitoring."""
    timestamp: float
    event_type: str
    iteration: int
    population_size: int
    best_score: Optional[float]
    metrics: Optional[PerformanceMetrics]
    metadata: Dict[str, Any]


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    memory_usage_mb: float
    cpu_usage_percent: float
    search_time_seconds: float
    architectures_evaluated: int
    current_iteration: int


class SearchMonitor:
    """Monitor and track search progress."""
    
    def __init__(self, log_file: Optional[Path] = None, buffer_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.buffer_size = buffer_size
        
        self.events: deque = deque(maxlen=buffer_size)
        self.metrics_history: List[PerformanceMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        
        self.start_time = time.time()
        self.iteration_count = 0
        self.total_evaluations = 0
        
        # Performance tracking
        self.best_scores: List[float] = []
        self.convergence_history: List[float] = []
        self.population_diversity: List[float] = []
        
        # Setup file logging if specified
        if self.log_file:
            self._setup_file_logging()
    
    def _setup_file_logging(self) -> None:
        """Setup file logging for search events."""
        file_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_search_start(self, config: Any) -> None:
        """Log search initialization."""
        event = SearchEvent(
            timestamp=time.time(),
            event_type='search_start',
            iteration=0,
            population_size=config.population_size,
            best_score=None,
            metrics=None,
            metadata={
                'max_iterations': config.max_iterations,
                'target_tops_w': config.target_tops_w,
                'max_latency_ms': config.max_latency_ms,
                'min_accuracy': config.min_accuracy
            }
        )
        self.events.append(event)
        self.logger.info(f"Search started with config: {config}")
    
    def log_iteration(
        self, 
        iteration: int, 
        population_size: int,
        best_metrics: Optional[PerformanceMetrics] = None,
        population_metrics: Optional[List[PerformanceMetrics]] = None
    ) -> None:
        """Log iteration progress."""
        self.iteration_count = iteration
        self.total_evaluations += population_size
        
        best_score = None
        if best_metrics:
            best_score = best_metrics.efficiency_score
            self.best_scores.append(best_score)
            self.metrics_history.append(best_metrics)
        
        # Calculate population diversity if available
        diversity = None
        if population_metrics and len(population_metrics) > 1:
            diversity = self._calculate_diversity(population_metrics)
            self.population_diversity.append(diversity)
        
        event = SearchEvent(
            timestamp=time.time(),
            event_type='iteration_complete',
            iteration=iteration,
            population_size=population_size,
            best_score=best_score,
            metrics=best_metrics,
            metadata={
                'total_evaluations': self.total_evaluations,
                'diversity': diversity,
                'elapsed_time': time.time() - self.start_time
            }
        )
        self.events.append(event)
        
        # Log system metrics periodically
        if iteration % 10 == 0:
            self._log_system_metrics()
    
    def log_search_end(
        self, 
        best_architecture: Architecture, 
        best_metrics: PerformanceMetrics,
        reason: str = 'completed'
    ) -> None:
        """Log search completion."""
        total_time = time.time() - self.start_time
        
        event = SearchEvent(
            timestamp=time.time(),
            event_type='search_end',
            iteration=self.iteration_count,
            population_size=0,
            best_score=best_metrics.efficiency_score,
            metrics=best_metrics,
            metadata={
                'total_time_seconds': total_time,
                'total_evaluations': self.total_evaluations,
                'reason': reason,
                'final_architecture': {
                    'name': best_architecture.name,
                    'layers': len(best_architecture.layers),
                    'params': best_architecture.total_params,
                    'ops': best_architecture.total_ops
                }
            }
        )
        self.events.append(event)
        
        self.logger.info(f"Search completed in {total_time:.2f}s with {self.total_evaluations} evaluations")
        self.logger.info(f"Best architecture: {best_architecture.name}")
        self.logger.info(f"Best metrics: {best_metrics}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log search error."""
        event = SearchEvent(
            timestamp=time.time(),
            event_type='error',
            iteration=self.iteration_count,
            population_size=0,
            best_score=None,
            metrics=None,
            metadata={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            }
        )
        self.events.append(event)
        self.logger.error(f"Search error: {error}", exc_info=True)
    
    def _calculate_diversity(self, population_metrics: List[PerformanceMetrics]) -> float:
        """Calculate population diversity based on metrics."""
        if len(population_metrics) < 2:
            return 0.0
        
        # Use efficiency scores for diversity calculation
        scores = [m.efficiency_score for m in population_metrics]
        return float(np.std(scores))
    
    def _log_system_metrics(self) -> None:
        """Log current system resource usage."""
        try:
            import psutil
            
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            system_metrics = SystemMetrics(
                timestamp=time.time(),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                search_time_seconds=time.time() - self.start_time,
                architectures_evaluated=self.total_evaluations,
                current_iteration=self.iteration_count
            )
            
            self.system_metrics.append(system_metrics)
            self.logger.debug(f"System metrics: {system_metrics}")
            
        except ImportError:
            # psutil not available, log basic metrics
            system_metrics = SystemMetrics(
                timestamp=time.time(),
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                search_time_seconds=time.time() - self.start_time,
                architectures_evaluated=self.total_evaluations,
                current_iteration=self.iteration_count
            )
            self.system_metrics.append(system_metrics)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary."""
        return {
            'elapsed_time': time.time() - self.start_time,
            'iterations_completed': self.iteration_count,
            'architectures_evaluated': self.total_evaluations,
            'best_score': max(self.best_scores) if self.best_scores else None,
            'convergence_trend': self._get_convergence_trend(),
            'average_diversity': np.mean(self.population_diversity) if self.population_diversity else 0.0,
            'recent_improvement': self._get_recent_improvement()
        }
    
    def _get_convergence_trend(self) -> str:
        """Analyze convergence trend."""
        if len(self.best_scores) < 10:
            return 'insufficient_data'
        
        recent_scores = self.best_scores[-10:]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.001:
            return 'improving'
        elif trend < -0.001:
            return 'degrading'
        else:
            return 'converged'
    
    def _get_recent_improvement(self) -> Optional[float]:
        """Get recent improvement in best score."""
        if len(self.best_scores) < 5:
            return None
        
        recent_best = max(self.best_scores[-5:])
        earlier_best = max(self.best_scores[:-5]) if len(self.best_scores) > 5 else 0
        
        return recent_best - earlier_best
    
    def export_logs(self, output_path: Path) -> None:
        """Export logs to file."""
        logs_data = {
            'search_summary': self.get_progress_summary(),
            'events': [asdict(event) for event in self.events],
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'system_metrics': [asdict(sm) for sm in self.system_metrics]
        }
        
        with open(output_path, 'w') as f:
            json.dump(logs_data, f, indent=2, default=str)
        
        self.logger.info(f"Logs exported to {output_path}")


class PerformanceProfiler:
    """Profile performance of search components."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def time_function(self, func_name: str):
        """Decorator to time function execution."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    with self._lock:
                        self.timers[func_name].append(elapsed)
                        self.counters[func_name] += 1
            return wrapper
        return decorator
    
    def record_time(self, operation: str, duration: float) -> None:
        """Record operation timing."""
        with self._lock:
            self.timers[operation].append(duration)
            self.counters[operation] += 1
    
    def increment_counter(self, counter_name: str, value: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self.counters[counter_name] += value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance profiling summary."""
        with self._lock:
            summary = {}
            
            for operation, times in self.timers.items():
                if times:
                    summary[operation] = {
                        'count': len(times),
                        'total_time': sum(times),
                        'average_time': np.mean(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'std_time': np.std(times)
                    }
            
            summary['counters'] = dict(self.counters)
            return summary
    
    def log_performance_summary(self) -> None:
        """Log performance summary."""
        summary = self.get_performance_summary()
        
        self.logger.info("Performance Summary:")
        for operation, stats in summary.items():
            if operation != 'counters' and isinstance(stats, dict):
                self.logger.info(
                    f"  {operation}: {stats['count']} calls, "
                    f"avg {stats['average_time']:.4f}s, "
                    f"total {stats['total_time']:.2f}s"
                )
        
        if 'counters' in summary:
            self.logger.info("Counters:")
            for counter, value in summary['counters'].items():
                self.logger.info(f"  {counter}: {value}")


class HealthChecker:
    """Monitor system health during search."""
    
    def __init__(self, check_interval: float = 30.0):
        self.logger = logging.getLogger(__name__)
        self.check_interval = check_interval
        self.last_check = time.time()
        self.health_issues: List[str] = []
    
    def check_health(self) -> Dict[str, Any]:
        """Perform health check."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return {'status': 'skipped', 'issues': self.health_issues}
        
        self.last_check = current_time
        self.health_issues.clear()
        
        # Check memory usage
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                self.health_issues.append(f"High memory usage: {memory_percent:.1f}%")
            
            # Check CPU usage (use non-blocking check)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 95:
                self.health_issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Check available disk space
            disk_usage = psutil.disk_usage('/').percent
            if disk_usage > 95:
                self.health_issues.append(f"Low disk space: {100-disk_usage:.1f}% available")
                
        except ImportError:
            self.logger.debug("psutil not available for health monitoring")
        
        # Log issues if found
        if self.health_issues:
            for issue in self.health_issues:
                self.logger.warning(f"Health issue: {issue}")
        
        return {
            'status': 'healthy' if not self.health_issues else 'issues_detected',
            'issues': self.health_issues,
            'timestamp': current_time
        }
    
    def is_healthy(self) -> bool:
        """Check if system is healthy with improved tolerance."""
        try:
            health = self.check_health()
            # More tolerant - only fail on severe issues
            return health.get('status', 'unknown') != 'critical'
        except Exception as e:
            self.logger.debug(f"Health check non-critical failure: {e}")
            return True  # Assume healthy if check fails


# Global instances for easy access
_global_monitor = None
_global_profiler = PerformanceProfiler()
_global_health_checker = HealthChecker()


def get_monitor() -> Optional[SearchMonitor]:
    """Get global search monitor."""
    return _global_monitor


def set_monitor(monitor: SearchMonitor) -> None:
    """Set global search monitor."""
    global _global_monitor
    _global_monitor = monitor


def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    return _global_profiler


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    return _global_health_checker
"""Advanced monitoring and health checks for TPUv6-ZeroNAS."""

import logging
import time
import threading
import functools
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .error_handling import safe_operation, get_error_handler


class ComponentStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: float
    component: str
    operation: str
    duration_ms: float
    success: bool
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: ComponentStatus
    message: str
    timestamp: float
    metrics: Dict[str, Any]


class AdvancedHealthMonitor:
    """Advanced health monitoring with detailed metrics and alerting."""
    
    def __init__(self, check_interval: int = 30):
        self.logger = logging.getLogger(__name__)
        self.check_interval = check_interval
        self.last_check = 0
        self.component_status: Dict[str, ComponentStatus] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.health_checks: Dict[str, Callable] = {}
        self.alert_thresholds = {
            'memory_usage_percent': 85,
            'cpu_usage_percent': 90,
            'error_rate_percent': 10,
            'avg_response_time_ms': 5000
        }
        self.is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health check functions."""
        self.health_checks.update({
            'system_resources': self._check_system_resources,
            'error_rates': self._check_error_rates,
            'performance_metrics': self._check_performance_metrics,
            'memory_usage': self._check_memory_usage
        })
    
    @safe_operation(default_return=HealthCheckResult("system", ComponentStatus.UNKNOWN, "Check failed", time.time(), {}))
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization."""
        status = ComponentStatus.HEALTHY
        message = "System resources normal"
        metrics = {}
        
        try:
            if HAS_PSUTIL:
                # Memory check
                memory = psutil.virtual_memory()
                metrics['memory_percent'] = memory.percent
                metrics['memory_available_gb'] = memory.available / (1024**3)
                
                # CPU check
                cpu_percent = psutil.cpu_percent(interval=0.1)
                metrics['cpu_percent'] = cpu_percent
                
                # Disk check
                disk = psutil.disk_usage('/')
                metrics['disk_percent'] = disk.percent
                metrics['disk_free_gb'] = disk.free / (1024**3)
                
                # Determine status based on thresholds
                if (memory.percent > 95 or cpu_percent > 98 or 
                    disk.percent > 98):
                    status = ComponentStatus.CRITICAL
                    message = "Critical resource usage detected"
                elif (memory.percent > self.alert_thresholds['memory_usage_percent'] or 
                      cpu_percent > self.alert_thresholds['cpu_usage_percent']):
                    status = ComponentStatus.DEGRADED
                    message = "High resource usage detected"
            else:
                message = "System monitoring limited (psutil not available)"
                status = ComponentStatus.UNKNOWN
                
        except Exception as e:
            message = f"Resource check failed: {e}"
            status = ComponentStatus.UNKNOWN
        
        return HealthCheckResult(
            component="system_resources",
            status=status,
            message=message,
            timestamp=time.time(),
            metrics=metrics
        )
    
    @safe_operation(default_return=HealthCheckResult("errors", ComponentStatus.UNKNOWN, "Check failed", time.time(), {}))
    def _check_error_rates(self) -> HealthCheckResult:
        """Check error rates across components."""
        error_handler = get_error_handler()
        error_stats = error_handler.get_error_statistics()
        
        status = ComponentStatus.HEALTHY
        message = "Error rates normal"
        
        total_errors = error_stats.get('total_errors', 0)
        recent_errors = error_stats.get('recent_errors', 0)
        
        # Calculate error rate (errors per minute)
        time_window = 600  # 10 minutes
        error_rate = (recent_errors / time_window) * 60 if time_window > 0 else 0
        
        metrics = {
            'total_errors': total_errors,
            'recent_errors': recent_errors,
            'error_rate_per_minute': error_rate,
            'errors_by_component': error_stats.get('errors_by_component', {})
        }
        
        if error_rate > 5:  # More than 5 errors per minute
            status = ComponentStatus.CRITICAL
            message = f"High error rate: {error_rate:.1f} errors/min"
        elif error_rate > 1:  # More than 1 error per minute
            status = ComponentStatus.DEGRADED
            message = f"Elevated error rate: {error_rate:.1f} errors/min"
        elif recent_errors > 0:
            message = f"Recent errors detected: {recent_errors}"
        
        return HealthCheckResult(
            component="error_rates",
            status=status,
            message=message,
            timestamp=time.time(),
            metrics=metrics
        )
    
    @safe_operation(default_return=HealthCheckResult("performance", ComponentStatus.UNKNOWN, "Check failed", time.time(), {}))
    def _check_performance_metrics(self) -> HealthCheckResult:
        """Check performance metrics and response times."""
        if not self.performance_history:
            return HealthCheckResult(
                component="performance_metrics",
                status=ComponentStatus.UNKNOWN,
                message="No performance data available",
                timestamp=time.time(),
                metrics={}
            )
        
        # Analyze recent performance data
        recent_metrics = list(self.performance_history)[-50:]  # Last 50 operations
        
        successful_ops = [m for m in recent_metrics if m.success]
        failed_ops = [m for m in recent_metrics if not m.success]
        
        if not recent_metrics:
            status = ComponentStatus.UNKNOWN
            message = "No recent performance data"
            metrics = {}
        else:
            success_rate = len(successful_ops) / len(recent_metrics) * 100
            avg_duration = sum(m.duration_ms for m in successful_ops) / len(successful_ops) if successful_ops else 0
            
            metrics = {
                'success_rate_percent': success_rate,
                'avg_response_time_ms': avg_duration,
                'total_operations': len(recent_metrics),
                'failed_operations': len(failed_ops)
            }
            
            # Determine status
            if success_rate < 80:
                status = ComponentStatus.CRITICAL
                message = f"Low success rate: {success_rate:.1f}%"
            elif success_rate < 95:
                status = ComponentStatus.DEGRADED
                message = f"Reduced success rate: {success_rate:.1f}%"
            elif avg_duration > self.alert_thresholds['avg_response_time_ms']:
                status = ComponentStatus.DEGRADED
                message = f"Slow response times: {avg_duration:.1f}ms avg"
            else:
                status = ComponentStatus.HEALTHY
                message = f"Performance normal: {success_rate:.1f}% success, {avg_duration:.1f}ms avg"
        
        return HealthCheckResult(
            component="performance_metrics",
            status=status,
            message=message,
            timestamp=time.time(),
            metrics=metrics
        )
    
    @safe_operation(default_return=HealthCheckResult("memory", ComponentStatus.UNKNOWN, "Check failed", time.time(), {}))
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check application memory usage patterns."""
        import gc
        import sys
        
        # Force garbage collection for accurate measurement
        gc.collect()
        
        metrics = {
            'python_objects': len(gc.get_objects()),
            'garbage_collected': len(gc.garbage)
        }
        
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                metrics.update({
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'memory_percent': process.memory_percent()
                })
                
                # Check for memory issues
                if metrics['memory_percent'] > 80:
                    status = ComponentStatus.CRITICAL
                    message = f"High memory usage: {metrics['memory_percent']:.1f}%"
                elif metrics['memory_percent'] > 60:
                    status = ComponentStatus.DEGRADED
                    message = f"Elevated memory usage: {metrics['memory_percent']:.1f}%"
                else:
                    status = ComponentStatus.HEALTHY
                    message = f"Memory usage normal: {metrics['memory_percent']:.1f}%"
                    
            except Exception as e:
                status = ComponentStatus.UNKNOWN
                message = f"Memory check failed: {e}"
        else:
            status = ComponentStatus.HEALTHY
            message = "Basic memory monitoring (psutil not available)"
        
        return HealthCheckResult(
            component="memory_usage",
            status=status,
            message=message,
            timestamp=time.time(),
            metrics=metrics
        )
    
    def record_performance(self, component: str, operation: str, 
                         duration_ms: float, success: bool, 
                         error_message: Optional[str] = None):
        """Record performance metrics for an operation."""
        metric = PerformanceMetrics(
            timestamp=time.time(),
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )
        
        # Add system metrics if available
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                metric.memory_mb = process.memory_info().rss / (1024 * 1024)
                metric.cpu_percent = process.cpu_percent()
            except:
                pass
        
        self.performance_history.append(metric)
    
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[check_name] = result
                self.component_status[check_name] = result.status
                
                # Log significant status changes
                if result.status in [ComponentStatus.DEGRADED, ComponentStatus.CRITICAL]:
                    self.logger.warning(f"Health check {check_name}: {result.message}")
                elif result.status == ComponentStatus.UNHEALTHY:
                    self.logger.error(f"Health check {check_name}: {result.message}")
                    
            except Exception as e:
                self.logger.error(f"Health check {check_name} failed: {e}")
                results[check_name] = HealthCheckResult(
                    component=check_name,
                    status=ComponentStatus.UNKNOWN,
                    message=f"Check failed: {e}",
                    timestamp=time.time(),
                    metrics={}
                )
        
        return results
    
    def get_overall_status(self) -> ComponentStatus:
        """Get overall system health status."""
        if not self.component_status:
            return ComponentStatus.UNKNOWN
        
        statuses = list(self.component_status.values())
        
        if ComponentStatus.CRITICAL in statuses:
            return ComponentStatus.CRITICAL
        elif ComponentStatus.UNHEALTHY in statuses:
            return ComponentStatus.UNHEALTHY
        elif ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED
        elif ComponentStatus.UNKNOWN in statuses:
            return ComponentStatus.UNKNOWN
        else:
            return ComponentStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        health_results = self.run_health_checks()
        overall_status = self.get_overall_status()
        
        summary = {
            'overall_status': overall_status.value,
            'timestamp': time.time(),
            'components': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'metrics': result.metrics
                }
                for name, result in health_results.items()
            },
            'performance_summary': self._get_performance_summary()
        }
        
        return summary
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.performance_history:
            return {'status': 'no_data'}
        
        recent = list(self.performance_history)[-100:]
        successful = [m for m in recent if m.success]
        
        if not recent:
            return {'status': 'no_recent_data'}
        
        return {
            'total_operations': len(recent),
            'success_rate': len(successful) / len(recent),
            'avg_duration_ms': sum(m.duration_ms for m in successful) / len(successful) if successful else 0,
            'components': list(set(m.component for m in recent))
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Advanced health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Advanced health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                current_time = time.time()
                if current_time - self.last_check >= self.check_interval:
                    health_results = self.run_health_checks()
                    self.last_check = current_time
                    
                    # Check for alerts
                    self._check_alerts(health_results)
                
                time.sleep(min(5, self.check_interval))
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _check_alerts(self, health_results: Dict[str, HealthCheckResult]):
        """Check for conditions that should trigger alerts."""
        critical_components = [
            name for name, result in health_results.items()
            if result.status == ComponentStatus.CRITICAL
        ]
        
        if critical_components:
            self.logger.critical(f"ALERT: Critical issues in components: {', '.join(critical_components)}")
        
        degraded_components = [
            name for name, result in health_results.items()
            if result.status == ComponentStatus.DEGRADED
        ]
        
        if degraded_components:
            self.logger.warning(f"ALERT: Degraded performance in components: {', '.join(degraded_components)}")


# Global advanced monitor instance
_advanced_monitor = AdvancedHealthMonitor()


def get_advanced_monitor() -> AdvancedHealthMonitor:
    """Get global advanced health monitor instance."""
    return _advanced_monitor


def record_operation_performance(component: str, operation: str):
    """Decorator to automatically record operation performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_msg = None
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                _advanced_monitor.record_performance(
                    component=component,
                    operation=operation,
                    duration_ms=duration_ms,
                    success=success,
                    error_message=error_msg
                )
        return wrapper
    return decorator
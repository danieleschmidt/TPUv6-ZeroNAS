"""Enhanced robustness and reliability improvements for TPUv6-ZeroNAS."""

import logging
import time
import functools
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import threading
from contextlib import contextmanager

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            current_time = time.time()
            
            if self.state == "OPEN":
                if current_time - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise RuntimeError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "HALF_OPEN":
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self.state = "CLOSED"
                        self.failure_count = 0
                        self.logger.info("Circuit breaker recovered to CLOSED")
                elif self.state == "CLOSED":
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = current_time
                
                if self.failure_count >= self.config.failure_threshold:
                    self.state = "OPEN"
                    self.logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")
                
                raise e

class RetryHandler:
    """Advanced retry handling with exponential backoff."""
    
    def __init__(
        self, 
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    if self.jitter:
                        delay *= (0.5 + hash(time.time()) % 500 / 1000.0)
                    
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.max_retries + 1} attempts failed for {func.__name__}: {e}"
                    )
        
        raise last_exception

class GracefulShutdownHandler:
    """Handle graceful shutdown of system components."""
    
    def __init__(self):
        self.shutdown_hooks: List[Callable] = []
        self.is_shutting_down = False
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def register_shutdown_hook(self, hook: Callable) -> None:
        """Register a function to be called during shutdown."""
        with self.lock:
            self.shutdown_hooks.append(hook)
    
    def shutdown(self) -> None:
        """Execute graceful shutdown."""
        with self.lock:
            if self.is_shutting_down:
                return
            
            self.is_shutting_down = True
            self.logger.info("Starting graceful shutdown...")
            
            for i, hook in enumerate(self.shutdown_hooks):
                try:
                    self.logger.debug(f"Executing shutdown hook {i + 1}/{len(self.shutdown_hooks)}")
                    hook()
                except Exception as e:
                    self.logger.error(f"Shutdown hook {i + 1} failed: {e}")
            
            self.logger.info("Graceful shutdown completed")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        with self.lock:
            return self.is_shutting_down

@contextmanager
def resource_cleanup(*resources):
    """Context manager for automatic resource cleanup."""
    try:
        yield resources
    finally:
        for resource in resources:
            try:
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'shutdown'):
                    resource.shutdown()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Resource cleanup failed: {e}")

class HealthMonitor:
    """Enhanced health monitoring with recovery actions."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.recovery_actions: Dict[str, Callable] = {}
        self.check_interval = 60.0
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._thread = None
    
    def register_health_check(self, name: str, check: Callable, recovery: Optional[Callable] = None) -> None:
        """Register a health check with optional recovery action."""
        self.health_checks[name] = check
        if recovery:
            self.recovery_actions[name] = recovery
    
    def start_monitoring(self) -> None:
        """Start health monitoring in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                for name, check in self.health_checks.items():
                    try:
                        if not check():
                            self.logger.warning(f"Health check '{name}' failed")
                            
                            # Execute recovery action if available
                            if name in self.recovery_actions:
                                try:
                                    self.logger.info(f"Executing recovery action for '{name}'")
                                    self.recovery_actions[name]()
                                except Exception as e:
                                    self.logger.error(f"Recovery action for '{name}' failed: {e}")
                    
                    except Exception as e:
                        self.logger.error(f"Health check '{name}' threw exception: {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(min(self.check_interval, 10.0))

# Global instances
_circuit_breaker_config = CircuitBreakerConfig()
_retry_handler = RetryHandler()
_shutdown_handler = GracefulShutdownHandler()
_health_monitor = HealthMonitor()

def get_circuit_breaker(config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get circuit breaker instance."""
    return CircuitBreaker(config or _circuit_breaker_config)

def get_retry_handler(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> RetryHandler:
    """Get retry handler instance."""
    return RetryHandler(max_retries, base_delay, max_delay)

def get_shutdown_handler() -> GracefulShutdownHandler:
    """Get global shutdown handler."""
    return _shutdown_handler

def get_health_monitor() -> HealthMonitor:
    """Get global health monitor."""
    return _health_monitor
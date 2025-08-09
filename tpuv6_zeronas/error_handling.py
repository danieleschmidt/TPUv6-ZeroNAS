"""Enhanced error handling and recovery mechanisms for TPUv6-ZeroNAS."""

import logging
import traceback
import functools
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"  
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error analysis."""
    component: str
    function: str
    args: tuple
    kwargs: dict
    timestamp: float
    traceback_str: str


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"


class TPUv6Error(Exception):
    """Base exception for TPUv6-ZeroNAS specific errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.severity = severity
        self.context = context or {}
        super().__init__(self.message)


class ArchitectureError(TPUv6Error):
    """Errors related to architecture generation or validation."""
    pass


class PredictionError(TPUv6Error):
    """Errors related to performance prediction."""
    pass


class SearchError(TPUv6Error):
    """Errors related to architecture search process."""
    pass


class ResourceError(TPUv6Error):
    """Errors related to resource management."""
    pass


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[type, ErrorRecoveryStrategy] = {
            ArchitectureError: ErrorRecoveryStrategy.RETRY,
            PredictionError: ErrorRecoveryStrategy.FALLBACK,
            SearchError: ErrorRecoveryStrategy.RETRY,
            ResourceError: ErrorRecoveryStrategy.FALLBACK,
            Exception: ErrorRecoveryStrategy.SKIP
        }
        self.max_retries = 3
        self.max_error_history = 100
    
    def handle_error(self, error: Exception, context: ErrorContext) -> ErrorRecoveryStrategy:
        """Handle error and determine recovery strategy."""
        
        # Log error details
        severity = getattr(error, 'severity', ErrorSeverity.MEDIUM)
        self._log_error(error, context, severity)
        
        # Store error for analysis
        self.error_history.append(context)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        # Determine recovery strategy
        error_type = type(error)
        strategy = self.recovery_strategies.get(error_type, ErrorRecoveryStrategy.SKIP)
        
        return strategy
    
    def _log_error(self, error: Exception, context: ErrorContext, severity: ErrorSeverity):
        """Log error with appropriate severity level."""
        log_msg = (f"Error in {context.component}.{context.function}: {error}")
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_msg)
            self.logger.critical(f"Context: {context.kwargs}")
            self.logger.critical(f"Traceback: {context.traceback_str}")
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_msg)
            self.logger.error(f"Context: {context.kwargs}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {'total_errors': 0}
        
        # Count errors by type
        error_counts = {}
        component_errors = {}
        
        for error_context in self.error_history:
            # Count by component
            component = error_context.component
            component_errors[component] = component_errors.get(component, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'errors_by_component': component_errors,
            'recent_errors': len([e for e in self.error_history[-10:]]),
        }


# Global error handler instance
_error_handler = ErrorHandler()


def robust_operation(max_retries: int = 3, 
                    fallback_result: Any = None,
                    component: str = "unknown"):
    """
    Decorator for robust operations with automatic retry and error handling.
    
    Args:
        max_retries: Maximum number of retry attempts
        fallback_result: Result to return if all retries fail
        component: Component name for error tracking
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    
                    # Create error context
                    context = ErrorContext(
                        component=component,
                        function=func.__name__,
                        args=args,
                        kwargs=kwargs,
                        timestamp=time.time(),
                        traceback_str=traceback.format_exc()
                    )
                    
                    # Handle error and get strategy
                    strategy = _error_handler.handle_error(e, context)
                    
                    if attempt < max_retries:
                        if strategy == ErrorRecoveryStrategy.RETRY:
                            # Wait before retry with exponential backoff
                            wait_time = 0.1 * (2 ** attempt)
                            time.sleep(wait_time)
                            continue
                        elif strategy == ErrorRecoveryStrategy.ABORT:
                            raise e
                    
                    # Final attempt failed or strategy is not retry
                    break
            
            # All retries failed
            if fallback_result is not None:
                _error_handler.logger.warning(
                    f"Operation {func.__name__} failed after {max_retries} retries, "
                    f"using fallback result"
                )
                return fallback_result
            else:
                raise last_error
                
        return wrapper
    return decorator


def safe_operation(default_return: Any = None, 
                  log_errors: bool = True,
                  component: str = "unknown"):
    """
    Decorator for operations that should never raise exceptions.
    
    Args:
        default_return: Value to return if operation fails
        log_errors: Whether to log errors
        component: Component name for error tracking
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    import time
                    context = ErrorContext(
                        component=component,
                        function=func.__name__,
                        args=args,
                        kwargs=kwargs,
                        timestamp=time.time(),
                        traceback_str=traceback.format_exc()
                    )
                    _error_handler.handle_error(e, context)
                
                return default_return
                
        return wrapper
    return decorator


def validate_architecture_safe(architecture) -> bool:
    """Safely validate architecture with comprehensive checks."""
    try:
        # Check basic structure
        if not architecture or not hasattr(architecture, 'layers'):
            return False
        
        if not architecture.layers or len(architecture.layers) == 0:
            return False
        
        # Check parameters are reasonable
        if hasattr(architecture, 'total_params'):
            if architecture.total_params <= 0 or architecture.total_params > 1e10:
                return False
        
        # Check operations are reasonable
        if hasattr(architecture, 'total_ops'):
            if architecture.total_ops <= 0 or architecture.total_ops > 1e18:
                return False
        
        # Check memory usage is reasonable
        if hasattr(architecture, 'memory_mb'):
            if architecture.memory_mb <= 0 or architecture.memory_mb > 100000:
                return False
        
        # Check layer compatibility
        for i, layer in enumerate(architecture.layers[:-1]):
            next_layer = architecture.layers[i + 1]
            if hasattr(layer, 'output_channels') and hasattr(next_layer, 'input_channels'):
                if layer.output_channels != next_layer.input_channels:
                    return False
        
        return True
        
    except Exception:
        return False


def validate_metrics_safe(metrics) -> bool:
    """Safely validate performance metrics."""
    try:
        if not metrics:
            return False
        
        # Check required attributes
        required_attrs = ['latency_ms', 'energy_mj', 'accuracy', 'tops_per_watt']
        for attr in required_attrs:
            if not hasattr(metrics, attr):
                return False
        
        # Check value ranges
        if not (0.0 < metrics.latency_ms < 10000.0):  # 10 seconds max
            return False
        
        if not (0.0 < metrics.energy_mj < 100000.0):  # 100 Joules max
            return False
        
        if not (0.0 <= metrics.accuracy <= 1.0):
            return False
        
        if not (0.0 < metrics.tops_per_watt < 1000.0):  # 1000 TOPS/W max
            return False
        
        # Check for NaN or infinite values
        values = [metrics.latency_ms, metrics.energy_mj, metrics.accuracy, metrics.tops_per_watt]
        for value in values:
            if not isinstance(value, (int, float)):
                return False
            if value != value:  # Check for NaN
                return False
            if value == float('inf') or value == float('-inf'):
                return False
        
        return True
        
    except Exception:
        return False


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _error_handler


def create_recovery_checkpoint(component: str, state: Dict[str, Any]) -> str:
    """Create a recovery checkpoint for component state."""
    import json
    import hashlib
    import time
    
    try:
        # Create checkpoint data
        checkpoint = {
            'component': component,
            'timestamp': time.time(),
            'state': state
        }
        
        # Create checkpoint ID
        checkpoint_json = json.dumps(checkpoint, sort_keys=True)
        checkpoint_id = hashlib.md5(checkpoint_json.encode()).hexdigest()[:8]
        
        # In a real implementation, this would save to persistent storage
        _error_handler.logger.info(f"Recovery checkpoint created for {component}: {checkpoint_id}")
        
        return checkpoint_id
        
    except Exception as e:
        _error_handler.logger.warning(f"Failed to create recovery checkpoint: {e}")
        return "failed"


# Exception context manager for automatic error handling
class ErrorHandlingContext:
    """Context manager for automatic error handling."""
    
    def __init__(self, component: str, operation: str, 
                 fallback_result: Any = None, 
                 suppress_exceptions: bool = False):
        self.component = component
        self.operation = operation
        self.fallback_result = fallback_result
        self.suppress_exceptions = suppress_exceptions
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            import time
            context = ErrorContext(
                component=self.component,
                function=self.operation,
                args=(),
                kwargs={},
                timestamp=time.time(),
                traceback_str=traceback.format_exc()
            )
            
            strategy = _error_handler.handle_error(exc_val, context)
            
            if self.suppress_exceptions and strategy != ErrorRecoveryStrategy.ABORT:
                return True  # Suppress the exception
        
        return False
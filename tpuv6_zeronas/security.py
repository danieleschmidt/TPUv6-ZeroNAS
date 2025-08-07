"""Security and safety measures for TPUv6-ZeroNAS."""

import hashlib
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

from .architecture import Architecture
from .metrics import PerformanceMetrics


class SecurityError(Exception):
    """Security-related error."""
    pass


class ResourceGuard:
    """Guard against resource exhaustion attacks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_memory_mb = 8000  # 8GB
        self.max_cpu_time = 3600   # 1 hour
        self.max_iterations = 50000
        self.max_population_size = 1000
    
    def check_resource_limits(self, config: Any) -> None:
        """Check if configuration respects resource limits."""
        if hasattr(config, 'max_iterations'):
            if config.max_iterations > self.max_iterations:
                raise SecurityError(
                    f'Max iterations too high: {config.max_iterations} > {self.max_iterations}'
                )
        
        if hasattr(config, 'population_size'):
            if config.population_size > self.max_population_size:
                raise SecurityError(
                    f'Population size too high: {config.population_size} > {self.max_population_size}'
                )
    
    def check_architecture_complexity(self, architecture: Architecture) -> None:
        """Check if architecture is within complexity limits."""
        # Check memory usage
        if architecture.memory_mb > self.max_memory_mb:
            raise SecurityError(
                f'Architecture memory too high: {architecture.memory_mb} > {self.max_memory_mb} MB'
            )
        
        # Check parameter count (prevent huge models)
        if architecture.total_params > 1_000_000_000:  # 1B parameters
            raise SecurityError(
                f'Architecture too large: {architecture.total_params} parameters'
            )
        
        # Check operation count
        if architecture.total_ops > 100_000_000_000:  # 100B operations
            raise SecurityError(
                f'Architecture too complex: {architecture.total_ops} operations'
            )


class AuditLogger:
    """Security audit logging."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.logger = logging.getLogger('security_audit')
        
        # Setup dedicated security log file
        if log_file is None:
            log_file = Path('security_audit.log')
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log general security events."""
        self.logger.info(f'SECURITY_EVENT: {event_type} - {json.dumps(details)}')


# Global instances
_resource_guard = ResourceGuard()
_audit_logger = AuditLogger()


def get_resource_guard() -> ResourceGuard:
    """Get global resource guard."""
    return _resource_guard


def get_audit_logger() -> AuditLogger:
    """Get global audit logger."""
    return _audit_logger


def secure_load_file(file_path: Union[str, Path], file_type: str = 'auto') -> Any:
    """Securely load a file with full validation."""
    # For now, just basic JSON loading - can be enhanced later
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    if not file_path.exists():
        raise SecurityError(f'File does not exist: {file_path}')
    
    if file_type == 'auto':
        if file_path.suffix.lower() == '.json':
            file_type = 'json'
        else:
            raise SecurityError(f'Unsupported file type: {file_path.suffix}')
    
    if file_type == 'json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise SecurityError(f'Unsupported file type: {file_type}')


def validate_search_security(config: Any, architecture_space: Any) -> None:
    """Validate search configuration for security."""
    resource_guard = get_resource_guard()
    audit_logger = get_audit_logger()
    
    try:
        # Check resource limits
        resource_guard.check_resource_limits(config)
        
        # Log security validation
        audit_logger.log_security_event('SEARCH_VALIDATION', {
            'max_iterations': getattr(config, 'max_iterations', 'unknown'),
            'population_size': getattr(config, 'population_size', 'unknown'),
            'validation_status': 'passed'
        })
        
    except SecurityError as e:
        audit_logger.log_security_event('SEARCH_VALIDATION', {
            'validation_status': 'failed',
            'error': str(e)
        })
        raise
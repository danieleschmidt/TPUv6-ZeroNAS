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
    """Enhanced guard against resource exhaustion attacks and malicious inputs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_memory_mb = 8000  # 8GB
        self.max_cpu_time = 3600   # 1 hour  
        self.max_iterations = 10000  # Reduced for safety
        self.max_population_size = 500  # Reduced for safety
        self.max_architecture_depth = 50
        self.max_architecture_params = 1e9  # 1B parameters max
        self.max_string_length = 1000
        self.blocked_file_extensions = {'.exe', '.bat', '.sh', '.dll', '.so', '.dylib'}
        self.suspicious_patterns = ['eval', 'exec', 'import', '__', 'subprocess', 'system']
    
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
        
        # Check for negative values that could cause issues
        if hasattr(config, 'max_iterations') and config.max_iterations < 0:
            raise SecurityError('Max iterations cannot be negative')
        
        if hasattr(config, 'population_size') and config.population_size < 0:
            raise SecurityError('Population size cannot be negative')
    
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
    
    def sanitize_string_input(self, input_str: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not isinstance(input_str, str):
            raise SecurityError(f"Expected string input, got {type(input_str)}")
        
        max_len = max_length or self.max_string_length
        if len(input_str) > max_len:
            raise SecurityError(f"String too long: {len(input_str)} > {max_len}")
        
        # Check for suspicious patterns
        input_lower = input_str.lower()
        for pattern in self.suspicious_patterns:
            if pattern in input_lower:
                raise SecurityError(f"Suspicious pattern detected: {pattern}")
        
        # Remove/escape dangerous characters
        sanitized = input_str.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
        
        # Limit to printable ASCII + common unicode
        sanitized = ''.join(char for char in sanitized 
                           if char.isprintable() or char.isspace())
        
        return sanitized
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and sanitize file path."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Check for path traversal attempts
        str_path = str(file_path)
        if '..' in str_path or str_path.startswith('/'):
            raise SecurityError(f"Potentially unsafe path: {file_path}")
        
        # Check file extension
        if file_path.suffix.lower() in self.blocked_file_extensions:
            raise SecurityError(f"Blocked file extension: {file_path.suffix}")
        
        # Resolve to absolute path and check it's within safe directory
        try:
            resolved = file_path.resolve()
            # Ensure it's within current working directory or temp directory
            cwd = Path.cwd()
            temp_dir = Path(tempfile.gettempdir())
            
            if not (str(resolved).startswith(str(cwd)) or 
                   str(resolved).startswith(str(temp_dir))):
                raise SecurityError(f"Path outside safe directory: {resolved}")
                
        except (OSError, ValueError) as e:
            raise SecurityError(f"Invalid path: {e}")
        
        return resolved
    
    def validate_numeric_input(self, value: Union[int, float], 
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None,
                             allow_negative: bool = True) -> Union[int, float]:
        """Validate numeric input."""
        if not isinstance(value, (int, float)):
            raise SecurityError(f"Expected numeric input, got {type(value)}")
        
        # Check for special float values
        if isinstance(value, float):
            if not (-1e308 <= value <= 1e308):  # Within float64 range
                raise SecurityError(f"Numeric value out of range: {value}")
            if value != value:  # NaN check
                raise SecurityError("NaN values not allowed")
            if value == float('inf') or value == float('-inf'):
                raise SecurityError("Infinite values not allowed")
        
        # Check sign
        if not allow_negative and value < 0:
            raise SecurityError(f"Negative values not allowed: {value}")
        
        # Check bounds
        if min_val is not None and value < min_val:
            raise SecurityError(f"Value too small: {value} < {min_val}")
        
        if max_val is not None and value > max_val:
            raise SecurityError(f"Value too large: {value} > {max_val}")
        
        return value


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
"""TPUv6-ZeroNAS: Neural Architecture Search for TPUv6 Hardware Optimization."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core import ZeroNASSearcher, SearchConfig
from .predictor import TPUv6Predictor
from .architecture import ArchitectureSpace
from .metrics import PerformanceMetrics
from .optimizations import TPUv6Optimizer, TPUv6Config
from .validation import validate_input
from .monitoring import SearchMonitor
from .security import secure_load_file

__all__ = [
    "ZeroNASSearcher",
    "SearchConfig",
    "TPUv6Predictor", 
    "ArchitectureSpace",
    "PerformanceMetrics",
    "TPUv6Optimizer",
    "TPUv6Config",
    "validate_input",
    "SearchMonitor",
    "secure_load_file",
]
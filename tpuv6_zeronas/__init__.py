"""TPUv6-ZeroNAS: Neural Architecture Search for TPUv6 Hardware Optimization."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core import ZeroNASSearcher
from .predictor import TPUv6Predictor
from .architecture import ArchitectureSpace
from .metrics import PerformanceMetrics

# Optional advanced modules (may have additional dependencies)
try:
    from .parallel import DistributedSearcher, ParallelSearchConfig
    from .optimization import ProgressiveSearchOptimizer, MultiObjectiveOptimizer, OptimizationConfig
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

__all__ = [
    "ZeroNASSearcher",
    "TPUv6Predictor", 
    "ArchitectureSpace",
    "PerformanceMetrics",
]

if ADVANCED_AVAILABLE:
    __all__.extend([
        "DistributedSearcher",
        "ParallelSearchConfig",
        "ProgressiveSearchOptimizer",
        "MultiObjectiveOptimizer",
        "OptimizationConfig",
    ])
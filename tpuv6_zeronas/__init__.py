```python
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

# Optional advanced modules (may have additional dependencies)
try:
    from .parallel import DistributedSearcher, ParallelSearchConfig
    from .optimization import ProgressiveSearchOptimizer, MultiObjectiveOptimizer, OptimizationConfig
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

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

if ADVANCED_AVAILABLE:
    __all__.extend([
        "DistributedSearcher",
        "ParallelSearchConfig",
        "ProgressiveSearchOptimizer",
        "MultiObjectiveOptimizer",
        "OptimizationConfig",
    ])
```

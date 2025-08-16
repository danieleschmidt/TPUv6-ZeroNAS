"""TPUv6-ZeroNAS: Neural Architecture Search for TPUv6 Hardware Optimization."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core import ZeroNASSearcher, SearchConfig
from .predictor import TPUv6Predictor
from .architecture import ArchitectureSpace
from .metrics import PerformanceMetrics, MetricsAggregator
from .optimizations import TPUv6Optimizer, TPUv6Config
from .validation import validate_input
from .monitoring import SearchMonitor
from .security import secure_load_file

# Optional advanced modules (may have additional dependencies)
try:
    from .parallel import DistributedSearcher, ParallelSearchConfig
    from .optimization import ProgressiveSearchOptimizer, MultiObjectiveOptimizer, OptimizationConfig
    from .quantum_nas import QuantumInspiredNAS, create_quantum_nas_searcher
    from .federated_nas import FederatedNAS, create_federated_nas_searcher
    from .neuromorphic_nas import NeuromorphicNAS, create_neuromorphic_nas_searcher
    ADVANCED_AVAILABLE = True
    QUANTUM_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    QUANTUM_AVAILABLE = False

# Revolutionary research modules (cutting-edge capabilities)
try:
    from .universal_hardware_transfer import (
        UniversalHardwareTransferEngine, create_universal_transfer_engine,
        HardwarePlatform, validate_transfer_accuracy
    )
    from .autonomous_hypothesis_engine import (
        AutonomousHypothesisEngine, create_autonomous_hypothesis_engine,
        HypothesisType, validate_autonomous_discoveries
    )
    from .ai_research_assistant import (
        AIResearchAssistant, create_ai_research_assistant, 
        ResearchTaskType, create_integrated_research_platform
    )
    REVOLUTIONARY_RESEARCH_AVAILABLE = True
except ImportError:
    REVOLUTIONARY_RESEARCH_AVAILABLE = False

__all__ = [
    "ZeroNASSearcher",
    "SearchConfig",
    "TPUv6Predictor", 
    "ArchitectureSpace",
    "PerformanceMetrics",
    "MetricsAggregator",
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

if QUANTUM_AVAILABLE:
    __all__.extend([
        "QuantumInspiredNAS",
        "create_quantum_nas_searcher",
        "FederatedNAS",
        "create_federated_nas_searcher",
        "NeuromorphicNAS",
        "create_neuromorphic_nas_searcher",
    ])

if REVOLUTIONARY_RESEARCH_AVAILABLE:
    __all__.extend([
        "UniversalHardwareTransferEngine",
        "create_universal_transfer_engine",
        "HardwarePlatform",
        "validate_transfer_accuracy",
        "AutonomousHypothesisEngine", 
        "create_autonomous_hypothesis_engine",
        "HypothesisType",
        "validate_autonomous_discoveries",
        "AIResearchAssistant",
        "create_ai_research_assistant",
        "ResearchTaskType",
        "create_integrated_research_platform",
    ])

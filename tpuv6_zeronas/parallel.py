"""Parallel evaluation and distributed computing for TPUv6-ZeroNAS."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from queue import Queue, Empty

from .architecture import Architecture
from .metrics import PerformanceMetrics
from .predictor import TPUv6Predictor


@dataclass
class WorkerConfig:
    """Configuration for parallel workers."""
    num_workers: Optional[int] = None  # Auto-detect if None
    worker_type: str = 'thread'  # 'thread' or 'process'
    batch_size: int = 5
    timeout_seconds: float = 30.0
    max_retries: int = 3


class ParallelEvaluator:
    """Parallel evaluation of neural architectures."""
    
    def __init__(self, predictor: TPUv6Predictor, config: WorkerConfig):
        self.predictor = predictor
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect worker count
        if self.config.num_workers is None:
            try:
                import multiprocessing
                self.config.num_workers = min(4, max(1, multiprocessing.cpu_count() // 2))
            except:
                self.config.num_workers = 2
        
        self.executor = None
        self._initialize_executor()
    
    def _initialize_executor(self):
        """Initialize the appropriate executor."""
        try:
            if self.config.worker_type == 'process':
                self.executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
                
            self.logger.info(f"Initialized {self.config.worker_type} executor with {self.config.num_workers} workers")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize executor: {e}")
            # Fallback to sequential execution
            self.executor = None
    
    def evaluate_batch(self, architectures: List[Architecture]) -> List[Optional[PerformanceMetrics]]:
        """Evaluate a batch of architectures in parallel."""
        if not architectures:
            return []
        
        if self.executor is None:
            # Sequential fallback
            return [self._evaluate_single_safe(arch) for arch in architectures]
        
        try:
            # Split into smaller batches for better load balancing
            if len(architectures) <= self.config.num_workers:
                # Small population: one architecture per batch
                batch_size = 1
            else:
                batch_size = min(self.config.batch_size, max(1, len(architectures) // self.config.num_workers))
            
            # Ensure batch_size is never 0
            batch_size = max(1, batch_size)
            batches = [architectures[i:i + batch_size] for i in range(0, len(architectures), batch_size)]
            
            futures = []
            for batch in batches:
                future = self.executor.submit(self._evaluate_batch_worker, batch)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    self.logger.warning(f"Batch evaluation failed: {e}")
                    # Add None results for failed batch
                    all_results.extend([None] * len(batches[futures.index(future)]))
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Parallel evaluation failed: {e}")
            # Fallback to sequential
            return [self._evaluate_single_safe(arch) for arch in architectures]
    
    def _evaluate_batch_worker(self, architectures: List[Architecture]) -> List[Optional[PerformanceMetrics]]:
        """Worker function to evaluate a batch of architectures."""
        results = []
        
        for arch in architectures:
            result = self._evaluate_single_safe(arch)
            results.append(result)
        
        return results
    
    def _evaluate_single_safe(self, architecture: Architecture) -> Optional[PerformanceMetrics]:
        """Safely evaluate a single architecture with retries."""
        for attempt in range(self.config.max_retries):
            try:
                return self.predictor.predict(architecture)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    self.logger.error(f"Failed to evaluate {architecture.name} after {self.config.max_retries} attempts: {e}")
                    return None
                else:
                    self.logger.warning(f"Evaluation attempt {attempt + 1} failed for {architecture.name}: {e}")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
        
        return None
    
    def shutdown(self):
        """Shutdown the executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None


class PerformanceOptimizer:
    """Optimizes performance of parallel operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evaluations_optimized': 0
        }
    
    def optimize_population_evaluation(
        self, 
        population: List[Architecture], 
        evaluator: ParallelEvaluator
    ) -> List[tuple]:
        """Optimize evaluation of a population."""
        try:
            # Remove duplicates to avoid redundant computation
            unique_archs = self._deduplicate_architectures(population)
            
            self.logger.info(f"Optimized population from {len(population)} to {len(unique_archs)} unique architectures")
            
            # Evaluate unique architectures
            unique_metrics = evaluator.evaluate_batch(unique_archs)
            
            # Create mapping back to original population
            results = []
            arch_to_metrics = {
                arch.name: metrics for arch, metrics in zip(unique_archs, unique_metrics)
                if metrics is not None
            }
            
            for arch in population:
                if arch.name in arch_to_metrics:
                    results.append((arch, arch_to_metrics[arch.name]))
                    self.cache_stats['hits'] += 1
                else:
                    self.cache_stats['misses'] += 1
            
            self.cache_stats['evaluations_optimized'] += len(population) - len(unique_archs)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Population optimization failed: {e}")
            # Fallback to direct evaluation
            metrics = evaluator.evaluate_batch(population)
            return [(arch, metric) for arch, metric in zip(population, metrics) if metric is not None]
    
    def _deduplicate_architectures(self, architectures: List[Architecture]) -> List[Architecture]:
        """Remove duplicate architectures based on their structure."""
        seen_signatures = set()
        unique_archs = []
        
        for arch in architectures:
            try:
                # Create a signature based on architecture structure
                signature = self._create_architecture_signature(arch)
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_archs.append(arch)
                    
            except Exception as e:
                self.logger.warning(f"Failed to create signature for {arch.name}: {e}")
                # Include architecture if signature creation fails
                unique_archs.append(arch)
        
        return unique_archs
    
    def _create_architecture_signature(self, arch: Architecture) -> str:
        """Create a unique signature for an architecture."""
        try:
            # Create signature based on layer structure
            layer_sigs = []
            for layer in arch.layers:
                layer_sig = (
                    layer.layer_type.value,
                    layer.input_channels,
                    layer.output_channels,
                    str(layer.kernel_size) if layer.kernel_size else 'None',
                    str(layer.stride) if layer.stride else 'None',
                    layer.activation.value if layer.activation else 'None'
                )
                layer_sigs.append(str(layer_sig))
            
            signature = f"{arch.input_shape}_{arch.num_classes}_{'_'.join(layer_sigs)}"
            return signature
            
        except Exception as e:
            self.logger.warning(f"Signature creation failed: {e}")
            # Fallback to basic signature
            return f"{arch.name}_{len(arch.layers)}_{arch.total_params}"
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return self.cache_stats.copy()


class ResourceManager:
    """Manages computational resources during search."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.resource_usage = {
            'peak_memory_mb': 0,
            'total_evaluations': 0,
            'parallel_efficiency': 0.0
        }
    
    def monitor_resource_usage(self):
        """Monitor current resource usage."""
        try:
            # Try to get memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.resource_usage['peak_memory_mb'] = max(
                    self.resource_usage['peak_memory_mb'], 
                    memory_mb
                )
            except ImportError:
                pass  # psutil not available
                
        except Exception as e:
            self.logger.debug(f"Resource monitoring failed: {e}")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        return self.resource_usage.copy()


class DistributedSearcher:
    """Distributed architecture search across multiple nodes."""
    
    def __init__(self, node_configs: List[Dict[str, Any]]):
        self.node_configs = node_configs
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Note: Distributed search is a placeholder for future implementation")
        self.logger.info(f"Configured for {len(node_configs)} nodes")
    
    def search_distributed(self, *args, **kwargs):
        """Placeholder for distributed search."""
        self.logger.warning("Distributed search not yet implemented - falling back to local search")
        # Would implement actual distributed coordination here
        return None


@dataclass 
class ParallelSearchConfig:
    """Configuration for parallel search operations."""
    enable_parallel_evaluation: bool = True
    max_parallel_workers: int = 4
    batch_evaluation_size: int = 10
    use_process_pool: bool = False  # Thread pool is usually sufficient
    timeout_per_batch: float = 60.0
    

# Factory functions for dependency injection
def get_performance_optimizer() -> PerformanceOptimizer:
    """Get performance optimizer instance."""
    return PerformanceOptimizer()


def get_resource_manager() -> ResourceManager:
    """Get resource manager instance.""" 
    return ResourceManager()


def create_parallel_evaluator(predictor: TPUv6Predictor, config: Optional[WorkerConfig] = None) -> ParallelEvaluator:
    """Create parallel evaluator with given configuration."""
    if config is None:
        config = WorkerConfig()
    
    return ParallelEvaluator(predictor, config)

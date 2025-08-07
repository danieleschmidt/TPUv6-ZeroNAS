"""Parallel and concurrent processing for TPUv6-ZeroNAS."""

import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass
import queue
import numpy as np

from .architecture import Architecture, ArchitectureSpace
from .predictor import TPUv6Predictor
from .metrics import PerformanceMetrics


@dataclass
class WorkerConfig:
    """Configuration for parallel workers."""
    num_workers: int = None  # Auto-detect if None
    worker_type: str = 'thread'  # 'thread' or 'process'
    batch_size: int = 10
    timeout_seconds: float = 60.0
    max_queue_size: int = 1000


class ParallelEvaluator:
    """Parallel evaluation of architecture populations."""
    
    def __init__(self, predictor: TPUv6Predictor, config: Optional[WorkerConfig] = None):
        self.predictor = predictor
        self.config = config or WorkerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect number of workers if not specified
        if self.config.num_workers is None:
            self.config.num_workers = min(mp.cpu_count(), 8)
        
        self.executor = None
        self._initialize_executor()
    
    def _initialize_executor(self) -> None:
        """Initialize the appropriate executor."""
        if self.config.worker_type == 'thread':
            self.executor = ThreadPoolExecutor(
                max_workers=self.config.num_workers,
                thread_name_prefix='arch_eval'
            )
        elif self.config.worker_type == 'process':
            self.executor = ProcessPoolExecutor(
                max_workers=self.config.num_workers
            )
        else:
            raise ValueError(f"Unknown worker type: {self.config.worker_type}")
        
        self.logger.info(f"Initialized {self.config.worker_type} executor with {self.config.num_workers} workers")
    
    def evaluate_population_parallel(
        self, 
        population: List[Architecture]
    ) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Evaluate population in parallel."""
        if not population:
            return []
        
        start_time = time.time()
        results = []
        
        # Submit all evaluation tasks
        future_to_arch = {}
        for arch in population:
            future = self.executor.submit(self._evaluate_single, arch)
            future_to_arch[future] = arch
        
        # Collect results as they complete
        for future in as_completed(future_to_arch, timeout=self.config.timeout_seconds):
            arch = future_to_arch[future]
            try:
                metrics = future.result()
                if metrics is not None:
                    results.append((arch, metrics))
            except Exception as e:
                self.logger.warning(f"Evaluation failed for architecture: {e}")
                continue
        
        eval_time = time.time() - start_time
        self.logger.info(f"Parallel evaluation completed: {len(results)}/{len(population)} in {eval_time:.2f}s")
        
        return results
    
    def _evaluate_single(self, architecture: Architecture) -> Optional[PerformanceMetrics]:
        """Evaluate single architecture."""
        try:
            return self.predictor.predict(architecture)
        except Exception as e:
            self.logger.warning(f"Single evaluation failed: {e}")
            return None
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
    
    def __del__(self):
        """Cleanup on destruction."""
        self.shutdown()


class DistributedSearchCoordinator:
    """Coordinate distributed search across multiple workers."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        self.work_queue = queue.Queue(maxsize=1000)
        self.result_queue = queue.Queue()
        self.workers = []
        self.coordinator_thread = None
        self.running = False
    
    def start_workers(self, architecture_space: ArchitectureSpace, predictor: TPUv6Predictor) -> None:
        """Start distributed worker processes."""
        self.running = True
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i, architecture_space, predictor),
                name=f'SearchWorker-{i}'
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.num_workers} distributed search workers")
    
    def _worker_loop(
        self, 
        worker_id: int, 
        architecture_space: ArchitectureSpace, 
        predictor: TPUv6Predictor
    ) -> None:
        """Main worker loop."""
        while self.running:
            try:
                # Get work from queue
                task = self.work_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                task_type, data = task
                
                if task_type == 'evaluate':
                    architecture = data
                    metrics = predictor.predict(architecture)
                    self.result_queue.put(('evaluated', (architecture, metrics)))
                
                elif task_type == 'evolve':
                    population_slice = data
                    # Simple local evolution
                    evolved = []
                    for arch in population_slice:
                        if np.random.random() < 0.5:
                            mutated = architecture_space.mutate(arch)
                            evolved.append(mutated)
                        else:
                            evolved.append(arch)
                    
                    self.result_queue.put(('evolved', evolved))
                
                self.work_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                continue
    
    def submit_evaluation_batch(self, architectures: List[Architecture]) -> None:
        """Submit architectures for evaluation."""
        for arch in architectures:
            self.work_queue.put(('evaluate', arch))
    
    def submit_evolution_batch(self, population_slices: List[List[Architecture]]) -> None:
        """Submit population slices for evolution."""
        for slice_data in population_slices:
            self.work_queue.put(('evolve', slice_data))
    
    def collect_results(self, expected_count: int, timeout: float = 60.0) -> List[Any]:
        """Collect results from workers."""
        results = []
        end_time = time.time() + timeout
        
        while len(results) < expected_count and time.time() < end_time:
            try:
                result_type, data = self.result_queue.get(timeout=1.0)
                results.append((result_type, data))
            except queue.Empty:
                continue
        
        return results
    
    def stop_workers(self) -> None:
        """Stop all workers."""
        self.running = False
        
        # Send shutdown signals
        for _ in range(self.num_workers):
            self.work_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()
        
        self.workers.clear()
        self.logger.info("All distributed workers stopped")


class PerformanceOptimizer:
    """Optimize search performance through various techniques."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def optimize_population_evaluation(
        self, 
        population: List[Architecture], 
        evaluator: ParallelEvaluator
    ) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Optimize population evaluation with caching and batching."""
        
        # Check cache first
        uncached_population = []
        cached_results = []
        
        for arch in population:
            cache_key = self._get_cache_key(arch)
            if cache_key in self.evaluation_cache:
                cached_results.append((arch, self.evaluation_cache[cache_key]))
                self.cache_hits += 1
            else:
                uncached_population.append(arch)
                self.cache_misses += 1
        
        # Evaluate uncached architectures
        if uncached_population:
            new_results = evaluator.evaluate_population_parallel(uncached_population)
            
            # Update cache
            for arch, metrics in new_results:
                cache_key = self._get_cache_key(arch)
                self.evaluation_cache[cache_key] = metrics
            
            cached_results.extend(new_results)
        
        self.logger.debug(f"Cache stats - Hits: {self.cache_hits}, Misses: {self.cache_misses}")
        
        return cached_results
    
    def _get_cache_key(self, architecture: Architecture) -> str:
        """Generate cache key for architecture."""
        # Simple hash based on architecture properties
        key_parts = [
            str(architecture.total_params),
            str(architecture.total_ops),
            str(len(architecture.layers)),
            str(architecture.input_shape),
            str(architecture.num_classes)
        ]
        
        # Add layer signatures
        for layer in architecture.layers:
            key_parts.extend([
                layer.layer_type.value,
                str(layer.input_channels),
                str(layer.output_channels),
                str(layer.kernel_size) if layer.kernel_size else 'None'
            ])
        
        return hash(tuple(key_parts))
    
    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self.evaluation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Evaluation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        return {
            'cache_size': len(self.evaluation_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class AdaptiveSearchScheduler:
    """Dynamically adjust search parameters based on progress."""
    
    def __init__(self, initial_config: Any):
        self.initial_config = initial_config
        self.current_config = initial_config
        self.logger = logging.getLogger(__name__)
        
        # Adaptation parameters
        self.stagnation_threshold = 10
        self.improvement_threshold = 0.01
        self.max_population_multiplier = 4.0
        self.min_population_multiplier = 0.5
        
        # Progress tracking
        self.best_scores = []
        self.stagnation_counter = 0
        
    def update_config(self, current_best_score: float, iteration: int) -> Any:
        """Update search configuration based on progress."""
        self.best_scores.append(current_best_score)
        
        # Check for improvement
        if len(self.best_scores) >= 5:
            recent_improvement = max(self.best_scores[-5:]) - max(self.best_scores[:-5])
            
            if recent_improvement < self.improvement_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # Adapt parameters based on stagnation
        if self.stagnation_counter >= self.stagnation_threshold:
            self._increase_exploration()
        elif self.stagnation_counter == 0:
            self._increase_exploitation()
        
        return self.current_config
    
    def _increase_exploration(self) -> None:
        """Increase exploration (larger population, higher mutation rate)."""
        old_pop_size = self.current_config.population_size
        new_pop_size = min(
            int(old_pop_size * 1.5),
            int(self.initial_config.population_size * self.max_population_multiplier)
        )
        
        self.current_config.population_size = new_pop_size
        self.current_config.mutation_rate = min(0.3, self.current_config.mutation_rate * 1.2)
        
        self.logger.info(f"Increased exploration - Population: {old_pop_size} -> {new_pop_size}")
        self.stagnation_counter = 0
    
    def _increase_exploitation(self) -> None:
        """Increase exploitation (smaller population, lower mutation rate)."""
        old_pop_size = self.current_config.population_size
        new_pop_size = max(
            int(old_pop_size * 0.9),
            int(self.initial_config.population_size * self.min_population_multiplier)
        )
        
        self.current_config.population_size = new_pop_size
        self.current_config.mutation_rate = max(0.05, self.current_config.mutation_rate * 0.9)
        
        self.logger.debug(f"Increased exploitation - Population: {old_pop_size} -> {new_pop_size}")


class ResourceManager:
    """Manage computational resources during search."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.evaluation_times = []
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor current resource usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            self.cpu_usage_history.append(cpu_percent)
            self.memory_usage_history.append(memory_percent)
            
            # Keep only recent history
            if len(self.cpu_usage_history) > 100:
                self.cpu_usage_history = self.cpu_usage_history[-50:]
                self.memory_usage_history = self.memory_usage_history[-50:]
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'avg_cpu': np.mean(self.cpu_usage_history),
                'avg_memory': np.mean(self.memory_usage_history)
            }
            
        except ImportError:
            return {'cpu_percent': 0, 'memory_percent': 0, 'avg_cpu': 0, 'avg_memory': 0}
    
    def should_scale_up(self) -> bool:
        """Check if we should scale up resources."""
        if len(self.cpu_usage_history) < 10:
            return False
        
        avg_cpu = np.mean(self.cpu_usage_history[-10:])
        avg_memory = np.mean(self.memory_usage_history[-10:])
        
        # Scale up if both CPU and memory are underutilized
        return avg_cpu < 50 and avg_memory < 60
    
    def should_scale_down(self) -> bool:
        """Check if we should scale down resources."""
        if len(self.cpu_usage_history) < 10:
            return False
        
        avg_cpu = np.mean(self.cpu_usage_history[-10:])
        avg_memory = np.mean(self.memory_usage_history[-10:])
        
        # Scale down if resources are heavily utilized
        return avg_cpu > 85 or avg_memory > 80
    
    def get_optimal_batch_size(self, base_batch_size: int = 10) -> int:
        """Calculate optimal batch size based on resource usage."""
        resources = self.monitor_resources()
        
        # Adjust batch size based on available resources
        if resources['memory_percent'] > 80:
            return max(1, base_batch_size // 2)
        elif resources['memory_percent'] < 40:
            return min(50, base_batch_size * 2)
        else:
            return base_batch_size


# Global instances for easy access
_performance_optimizer = PerformanceOptimizer()
_resource_manager = ResourceManager()


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    return _performance_optimizer


def get_resource_manager() -> ResourceManager:
    """Get global resource manager."""
    return _resource_manager
"""Parallel and distributed search capabilities for TPUv6-ZeroNAS."""

import logging
import concurrent.futures
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
from dataclasses import dataclass

from .architecture import Architecture, ArchitectureSpace
from .predictor import TPUv6Predictor
from .metrics import PerformanceMetrics
from .core import ZeroNASSearcher, SearchConfig


@dataclass
class ParallelSearchConfig:
    """Configuration for parallel search."""
    num_workers: int = mp.cpu_count()
    batch_size: int = 32
    enable_gpu_parallel: bool = False
    memory_limit_gb: float = 8.0
    timeout_seconds: int = 3600
    checkpoint_interval: int = 100


class ParallelEvaluator:
    """Parallel architecture evaluation engine."""
    
    def __init__(self, predictor: TPUv6Predictor, config: ParallelSearchConfig):
        self.predictor = predictor
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def evaluate_batch(
        self, 
        architectures: List[Architecture]
    ) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Evaluate a batch of architectures in parallel."""
        if not architectures:
            return []
        
        self.logger.debug(f"Evaluating batch of {len(architectures)} architectures")
        
        results = []
        
        if self.config.num_workers <= 1:
            # Sequential evaluation
            for arch in architectures:
                try:
                    metrics = self.predictor.predict(arch)
                    results.append((arch, metrics))
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for {arch.name}: {e}")
                    continue
        else:
            # Parallel evaluation
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.num_workers
            ) as executor:
                future_to_arch = {
                    executor.submit(self._safe_evaluate, arch): arch 
                    for arch in architectures
                }
                
                for future in concurrent.futures.as_completed(
                    future_to_arch, 
                    timeout=self.config.timeout_seconds
                ):
                    arch = future_to_arch[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Future failed for {arch.name}: {e}")
                        continue
        
        self.logger.debug(f"Successfully evaluated {len(results)}/{len(architectures)} architectures")
        return results
    
    def _safe_evaluate(
        self, 
        architecture: Architecture
    ) -> Optional[Tuple[Architecture, PerformanceMetrics]]:
        """Thread-safe architecture evaluation."""
        try:
            metrics = self.predictor.predict(architecture)
            return (architecture, metrics)
        except Exception as e:
            self.logger.debug(f"Safe evaluation failed for {architecture.name}: {e}")
            return None


class DistributedSearcher:
    """Distributed neural architecture search coordinator."""
    
    def __init__(
        self,
        architecture_space: ArchitectureSpace,
        predictor: TPUv6Predictor,
        search_config: SearchConfig,
        parallel_config: Optional[ParallelSearchConfig] = None
    ):
        self.architecture_space = architecture_space
        self.predictor = predictor
        self.search_config = search_config
        self.parallel_config = parallel_config or ParallelSearchConfig()
        
        self.logger = logging.getLogger(__name__)
        self.evaluator = ParallelEvaluator(predictor, self.parallel_config)
        
        self.best_architecture: Optional[Architecture] = None
        self.best_metrics: Optional[PerformanceMetrics] = None
        self.search_history: List[Tuple[Architecture, PerformanceMetrics]] = []
        
        self.checkpoints: List[Dict[str, Any]] = []
    
    def search(self) -> Tuple[Architecture, PerformanceMetrics]:
        """Run distributed neural architecture search."""
        self.logger.info("Starting distributed TPUv6-ZeroNAS search...")
        self.logger.info(f"Workers: {self.parallel_config.num_workers}")
        self.logger.info(f"Batch size: {self.parallel_config.batch_size}")
        
        start_time = time.time()
        
        try:
            # Initialize island populations
            num_islands = max(1, self.parallel_config.num_workers // 2)
            island_populations = self._initialize_island_populations(num_islands)
            
            for iteration in range(self.search_config.max_iterations):
                self.logger.info(f"Distributed iteration {iteration + 1}/{self.search_config.max_iterations}")
                
                # Evolve each island in parallel
                evolved_islands = self._evolve_islands_parallel(island_populations)
                
                # Evaluate all architectures from all islands
                all_architectures = []
                for island in evolved_islands:
                    all_architectures.extend(island)
                
                # Batch evaluation for efficiency
                batch_results = self._evaluate_in_batches(all_architectures)
                
                # Update global best
                self._update_global_best(batch_results)
                
                # Migration between islands
                if iteration % 10 == 0 and num_islands > 1:
                    island_populations = self._migrate_between_islands(
                        evolved_islands, batch_results, num_islands
                    )
                else:
                    island_populations = self._update_island_populations(
                        evolved_islands, batch_results, num_islands
                    )
                
                # Checkpointing
                if iteration % self.parallel_config.checkpoint_interval == 0:
                    self._save_checkpoint(iteration)
                
                # Early stopping check
                if self._should_early_stop():
                    self.logger.info(f"Early stopping at iteration {iteration + 1}")
                    break
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Distributed search completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Total evaluations: {len(self.search_history)}")
            
            if self.best_architecture is None or self.best_metrics is None:
                raise RuntimeError("Distributed search failed to find any valid architecture")
            
            return self.best_architecture, self.best_metrics
            
        except Exception as e:
            self.logger.error(f"Distributed search failed: {e}")
            raise
    
    def _initialize_island_populations(
        self, 
        num_islands: int
    ) -> List[List[Architecture]]:
        """Initialize multiple island populations."""
        island_populations = []
        
        pop_per_island = self.search_config.population_size // num_islands
        remainder = self.search_config.population_size % num_islands
        
        for i in range(num_islands):
            island_size = pop_per_island + (1 if i < remainder else 0)
            
            island_pop = []
            attempts = 0
            max_attempts = island_size * 3
            
            while len(island_pop) < island_size and attempts < max_attempts:
                try:
                    arch = self.architecture_space.sample_random()
                    island_pop.append(arch)
                except Exception as e:
                    self.logger.debug(f"Failed to create architecture for island {i}: {e}")
                attempts += 1
            
            if len(island_pop) == 0:
                # Create minimal architecture as fallback
                island_pop = [self.architecture_space._create_minimal_architecture()]
            
            island_populations.append(island_pop)
            
        self.logger.info(f"Initialized {num_islands} islands with sizes: {[len(pop) for pop in island_populations]}")
        return island_populations
    
    def _evolve_islands_parallel(
        self, 
        island_populations: List[List[Architecture]]
    ) -> List[List[Architecture]]:
        """Evolve each island population in parallel."""
        if self.parallel_config.num_workers <= 1:
            # Sequential evolution
            return [self._evolve_island(pop) for pop in island_populations]
        
        # Parallel evolution
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(island_populations), self.parallel_config.num_workers)
        ) as executor:
            futures = [
                executor.submit(self._evolve_island, pop) 
                for pop in island_populations
            ]
            
            evolved_islands = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    evolved_pop = future.result(timeout=60)
                    evolved_islands.append(evolved_pop)
                except Exception as e:
                    self.logger.warning(f"Island {i} evolution failed: {e}")
                    # Use original population as fallback
                    evolved_islands.append(island_populations[i])
            
        return evolved_islands
    
    def _evolve_island(self, population: List[Architecture]) -> List[Architecture]:
        """Evolve a single island population."""
        evolved_pop = []
        
        # Keep best individuals (elitism)
        elite_count = max(1, len(population) // 4)
        evolved_pop.extend(population[:elite_count])
        
        # Generate offspring
        while len(evolved_pop) < len(population):
            try:
                if len(population) >= 2 and hash(time.time()) % 2 == 0:
                    # Crossover
                    parent1 = population[hash(time.time()) % len(population)]
                    parent2 = population[hash(time.time() + 1) % len(population)]
                    child = self.architecture_space.crossover(parent1, parent2)
                else:
                    # Mutation
                    parent = population[hash(time.time()) % len(population)]
                    child = self.architecture_space.mutate(parent)
                
                evolved_pop.append(child)
                
            except Exception as e:
                # Fallback to random architecture
                try:
                    child = self.architecture_space.sample_random()
                    evolved_pop.append(child)
                except:
                    break  # Give up if we can't create any architecture
        
        return evolved_pop
    
    def _evaluate_in_batches(
        self, 
        architectures: List[Architecture]
    ) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Evaluate architectures in parallel batches."""
        all_results = []
        
        batch_size = self.parallel_config.batch_size
        
        for i in range(0, len(architectures), batch_size):
            batch = architectures[i:i + batch_size]
            batch_results = self.evaluator.evaluate_batch(batch)
            all_results.extend(batch_results)
        
        return all_results
    
    def _update_global_best(
        self, 
        batch_results: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> None:
        """Update global best architecture."""
        for arch, metrics in batch_results:
            if self._is_better_architecture(metrics):
                self.best_architecture = arch
                self.best_metrics = metrics
                self.logger.debug(f"New best architecture: {arch.name} with score {self._compute_score(metrics):.4f}")
        
        self.search_history.extend(batch_results)
    
    def _is_better_architecture(self, metrics: PerformanceMetrics) -> bool:
        """Check if architecture is better than current best."""
        if metrics.latency_ms > self.search_config.max_latency_ms:
            return False
        if metrics.accuracy < self.search_config.min_accuracy:
            return False
        
        if self.best_metrics is None:
            return True
        
        current_score = self._compute_score(metrics)
        best_score = self._compute_score(self.best_metrics)
        
        return current_score > best_score
    
    def _compute_score(self, metrics: PerformanceMetrics) -> float:
        """Compute multi-objective score."""
        efficiency_score = metrics.tops_per_watt / self.search_config.target_tops_w
        accuracy_score = metrics.accuracy
        latency_penalty = max(0, metrics.latency_ms - self.search_config.max_latency_ms)
        
        score = (0.4 * efficiency_score + 
                0.4 * accuracy_score - 
                0.2 * latency_penalty)
        
        return score
    
    def _migrate_between_islands(
        self,
        evolved_islands: List[List[Architecture]],
        batch_results: List[Tuple[Architecture, PerformanceMetrics]],
        num_islands: int
    ) -> List[List[Architecture]]:
        """Perform migration between islands."""
        # Sort results by score
        sorted_results = sorted(
            batch_results, 
            key=lambda x: self._compute_score(x[1]), 
            reverse=True
        )
        
        # Take top architectures for migration
        migrants_per_island = max(1, len(sorted_results) // (num_islands * 4))
        top_migrants = [arch for arch, _ in sorted_results[:migrants_per_island * num_islands]]
        
        # Redistribute migrants to islands
        new_islands = []
        
        for i, island in enumerate(evolved_islands):
            new_island = island.copy()
            
            # Replace worst individuals with migrants
            replace_count = min(len(new_island) // 4, len(top_migrants) // num_islands)
            
            if replace_count > 0:
                new_island = new_island[:-replace_count]  # Remove worst
                
                # Add migrants from other islands
                migrant_start = (i * replace_count) % len(top_migrants)
                migrants = top_migrants[migrant_start:migrant_start + replace_count]
                new_island.extend(migrants)
            
            new_islands.append(new_island)
        
        self.logger.debug(f"Migration completed: {len(top_migrants)} migrants distributed")
        return new_islands
    
    def _update_island_populations(
        self,
        evolved_islands: List[List[Architecture]],
        batch_results: List[Tuple[Architecture, PerformanceMetrics]],
        num_islands: int
    ) -> List[List[Architecture]]:
        """Update island populations with evaluated results."""
        # For simplicity, just return evolved islands
        # In a more sophisticated implementation, we would
        # reorganize populations based on evaluation results
        return evolved_islands
    
    def _should_early_stop(self) -> bool:
        """Check early stopping criteria."""
        if len(self.search_history) < 200:
            return False
        
        # Check improvement in last 100 evaluations
        recent_scores = [
            self._compute_score(metrics) 
            for _, metrics in self.search_history[-100:]
        ]
        
        if len(recent_scores) < 50:
            return False
        
        improvement = max(recent_scores) - min(recent_scores)
        return improvement < self.search_config.early_stop_threshold
    
    def _save_checkpoint(self, iteration: int) -> None:
        """Save search checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'best_architecture': self.best_architecture.name if self.best_architecture else None,
            'best_metrics': self.best_metrics.to_dict() if self.best_metrics else None,
            'search_history_length': len(self.search_history),
            'timestamp': time.time()
        }
        
        self.checkpoints.append(checkpoint)
        self.logger.debug(f"Checkpoint saved at iteration {iteration}")


class AdaptiveSearchScheduler:
    """Adaptive scheduler for dynamic resource allocation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_history = []
    
    def adjust_search_parameters(
        self,
        current_iteration: int,
        search_config: SearchConfig,
        parallel_config: ParallelSearchConfig,
        recent_performance: List[float]
    ) -> Tuple[SearchConfig, ParallelSearchConfig]:
        """Dynamically adjust search parameters based on performance."""
        self.performance_history.extend(recent_performance)
        
        # Adaptive population size
        if len(recent_performance) >= 10:
            recent_avg = sum(recent_performance[-10:]) / 10
            overall_avg = sum(self.performance_history) / len(self.performance_history)
            
            if recent_avg < overall_avg * 0.9:  # Performance declining
                # Increase diversity
                search_config.mutation_rate = min(0.3, search_config.mutation_rate * 1.1)
                search_config.population_size = min(200, int(search_config.population_size * 1.1))
                self.logger.info("Increasing diversity due to performance decline")
            elif recent_avg > overall_avg * 1.1:  # Performance improving
                # Focus search
                search_config.mutation_rate = max(0.05, search_config.mutation_rate * 0.9)
                parallel_config.batch_size = min(64, parallel_config.batch_size + 4)
                self.logger.info("Focusing search due to performance improvement")
        
        return search_config, parallel_config
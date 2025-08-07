"""Core ZeroNAS search functionality for TPUv6 optimization."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from dataclasses import dataclass

from .architecture import ArchitectureSpace, Architecture
from .predictor import TPUv6Predictor
from .metrics import PerformanceMetrics
from .validation import validate_input, ArchitectureValidator
from .monitoring import SearchMonitor, get_profiler, get_health_checker
from .security import get_resource_guard, SecurityError
from .parallel import ParallelEvaluator, WorkerConfig, get_performance_optimizer, get_resource_manager
from .caching import create_cached_predictor


@dataclass
class SearchConfig:
    """Configuration for ZeroNAS search."""
    max_iterations: int = 1000
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    early_stop_threshold: float = 1e-6
    target_tops_w: float = 75.0
    max_latency_ms: float = 10.0
    min_accuracy: float = 0.95
    # Optimization features
    enable_parallel: bool = True
    enable_caching: bool = True
    enable_adaptive: bool = True
    parallel_workers: int = None  # Auto-detect if None


class ZeroNASSearcher:
    """Main ZeroNAS searcher for TPUv6 architecture optimization."""
    
    def __init__(
        self,
        architecture_space: ArchitectureSpace,
        predictor: TPUv6Predictor,
        config: Optional[SearchConfig] = None
    ):
        self.architecture_space = architecture_space
        self.predictor = predictor
        self.config = config or SearchConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_initialization()
        
        self.best_architecture: Optional[Architecture] = None
        self.best_metrics: Optional[PerformanceMetrics] = None
        self.search_history: List[Tuple[Architecture, PerformanceMetrics]] = []
        
        # Initialize monitoring and profiling
        self.monitor = SearchMonitor()
        self.profiler = get_profiler()
        self.health_checker = get_health_checker()
        self.validator = ArchitectureValidator()
        
        # Initialize optimization features
        self.performance_optimizer = get_performance_optimizer()
        self.resource_manager = get_resource_manager()
        
        # Setup caching if enabled
        if self.config.enable_caching:
            self.predictor = create_cached_predictor(self.predictor)
        
        # Setup parallel evaluation if enabled
        self.parallel_evaluator = None
        if self.config.enable_parallel:
            worker_config = WorkerConfig(
                num_workers=self.config.parallel_workers,
                worker_type='thread',  # More stable for our use case
                batch_size=min(10, self.config.population_size // 4)
            )
            self.parallel_evaluator = ParallelEvaluator(self.predictor, worker_config)
    
    def _validate_initialization(self) -> None:
        """Validate searcher initialization parameters."""
        try:
            # Validate search config
            config_validation = validate_input(self.config, 'search_config')
            if not config_validation['is_valid']:
                raise ValueError(f"Invalid search config: {config_validation['errors']}")
            
            # Check resource limits
            resource_guard = get_resource_guard()
            resource_guard.check_resource_limits(self.config)
            
        except Exception as e:
            self.logger.error(f"Initialization validation failed: {e}")
            raise
        
    def search(self) -> Tuple[Architecture, PerformanceMetrics]:
        """Run neural architecture search to find optimal TPUv6 architecture."""
        try:
            self.monitor.log_search_start(self.config)
            self.logger.info("Starting TPUv6-ZeroNAS search...")
            self.logger.info(f"Target: {self.config.target_tops_w} TOPS/W")
            
            start_time = time.time()
            population = self._initialize_population()
            
            for iteration in range(self.config.max_iterations):
                # Check system health
                if not self.health_checker.is_healthy():
                    self.logger.warning("System health issues detected, continuing with caution")
                
                self.logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
                
                iteration_start = time.time()
                # Use optimized evaluation if available
                if self.config.enable_parallel and self.parallel_evaluator:
                    evaluated_pop = self.performance_optimizer.optimize_population_evaluation(
                        population, self.parallel_evaluator
                    )
                else:
                    evaluated_pop = self._evaluate_population_safe(population)
                
                self._update_best(evaluated_pop)
                
                # Log iteration progress
                self.monitor.log_iteration(
                    iteration + 1, 
                    len(population), 
                    self.best_metrics,
                    [metrics for _, metrics in evaluated_pop]
                )
                
                self.profiler.record_time('iteration', time.time() - iteration_start)
                
                if self._should_early_stop():
                    self.logger.info(f"Early stopping at iteration {iteration + 1}")
                    break
                    
                population = self._evolve_population(evaluated_pop)
            
            total_time = time.time() - start_time
            self.profiler.record_time('total_search', total_time)
            
            if self.best_architecture is None:
                # Fallback: return the best from the final population
                final_pop = self._initialize_population()
                evaluated = self._evaluate_population_safe(final_pop)
                if evaluated:
                    best_pair = max(evaluated, key=lambda x: self._compute_score(x[1]))
                    self.best_architecture, self.best_metrics = best_pair
            
            self.logger.info("Search completed!")
            self.logger.info(f"Best architecture: {self.best_architecture}")
            self.logger.info(f"Best metrics: {self.best_metrics}")
            
            # Final validation of result
            if self.best_architecture and self.best_metrics:
                self._validate_final_result(self.best_architecture, self.best_metrics)
                self.monitor.log_search_end(self.best_architecture, self.best_metrics)
            
            return self.best_architecture, self.best_metrics
            
        except Exception as e:
            self.monitor.log_error(e, {'iteration': iteration if 'iteration' in locals() else 0})
            self.logger.error(f"Search failed: {e}", exc_info=True)
            raise
    
    def _initialize_population(self) -> List[Architecture]:
        """Initialize random population of architectures."""
        population = []
        for _ in range(self.config.population_size):
            arch = self.architecture_space.sample_random()
            population.append(arch)
        return population
    
    def _evaluate_population(
        self, 
        population: List[Architecture]
    ) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Evaluate population using TPUv6 predictor."""
        evaluated = []
        
        for arch in population:
            metrics = self.predictor.predict(arch)
            evaluated.append((arch, metrics))
            
        return evaluated
    
    def _update_best(
        self, 
        evaluated_pop: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> None:
        """Update best architecture based on multi-objective optimization."""
        for arch, metrics in evaluated_pop:
            if self._is_better_architecture(metrics):
                self.best_architecture = arch
                self.best_metrics = metrics
                
        self.search_history.extend(evaluated_pop)
    
    def _is_better_architecture(self, metrics: PerformanceMetrics) -> bool:
        """Check if architecture is better based on constraints and objectives."""
        if metrics.latency_ms > self.config.max_latency_ms:
            return False
        if metrics.accuracy < self.config.min_accuracy:
            return False
            
        if self.best_metrics is None:
            return True
            
        current_score = self._compute_score(metrics)
        best_score = self._compute_score(self.best_metrics)
        
        return current_score > best_score
    
    def _compute_score(self, metrics: PerformanceMetrics) -> float:
        """Compute multi-objective score for architecture."""
        efficiency_score = metrics.tops_per_watt / self.config.target_tops_w
        accuracy_score = metrics.accuracy
        latency_penalty = max(0, metrics.latency_ms - self.config.max_latency_ms)
        
        score = (0.4 * efficiency_score + 
                0.4 * accuracy_score - 
                0.2 * latency_penalty)
        
        return score
    
    def _should_early_stop(self) -> bool:
        """Check if search should stop early."""
        if len(self.search_history) < 100:
            return False
            
        recent_scores = [
            self._compute_score(metrics) 
            for _, metrics in self.search_history[-50:]
        ]
        
        improvement = max(recent_scores) - min(recent_scores)
        return bool(improvement < self.config.early_stop_threshold)
    
    def _evolve_population(
        self, 
        evaluated_pop: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> List[Architecture]:
        """Evolve population using genetic algorithm."""
        evaluated_pop.sort(key=lambda x: self._compute_score(x[1]), reverse=True)
        
        elite_size = self.config.population_size // 4
        elite = [arch for arch, _ in evaluated_pop[:elite_size]]
        
        new_population = elite.copy()
        
        while len(new_population) < self.config.population_size:
            parent_pool = evaluated_pop[:elite_size * 2] if evaluated_pop else evaluated_pop
            if len(parent_pool) < 1:
                # Emergency fallback: generate new random architecture
                child = self.architecture_space.sample_random()
            elif len(parent_pool) == 1:
                # Only one parent available
                child = parent_pool[0][0]
            elif np.random.random() < self.config.crossover_rate:
                parent1, parent2 = self._select_parents(parent_pool)
                child = self._crossover(parent1, parent2)
            else:
                parent = self._select_parents(parent_pool, n=1)[0]
                child = parent
                
            if np.random.random() < self.config.mutation_rate:
                child = self._mutate(child)
                
            new_population.append(child)
            
        return new_population
    
    def _select_parents(
        self, 
        population: List[Tuple[Architecture, PerformanceMetrics]], 
        n: int = 2
    ) -> List[Architecture]:
        """Tournament selection for parent architectures."""
        if not population:
            return []
        
        parents = []
        for _ in range(n):
            tournament_size = min(3, len(population))
            if tournament_size == 0:
                continue
                
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = max(tournament, key=lambda i: self._compute_score(population[i][1]))
            parents.append(population[best_idx][0])
        return parents
    
    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover two parent architectures."""
        return self.architecture_space.crossover(parent1, parent2)
    
    def _mutate(self, architecture: Architecture) -> Architecture:
        """Mutate architecture."""
        return self.architecture_space.mutate(architecture)
    
    def _evaluate_population_safe(
        self, 
        population: List[Architecture]
    ) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Safely evaluate population with validation."""
        evaluated = []
        
        for arch in population:
            try:
                # Validate architecture before evaluation
                validation = self.validator.validate_architecture(arch)
                if not validation['is_valid']:
                    self.logger.warning(f"Skipping invalid architecture: {validation['errors']}")
                    continue
                
                # Check resource constraints
                resource_guard = get_resource_guard()
                resource_guard.check_architecture_complexity(arch)
                
                metrics = self.predictor.predict(arch)
                
                # Validate metrics
                metrics_validation = validate_input(metrics, 'metrics')
                if not metrics_validation['is_valid']:
                    self.logger.warning(f"Invalid metrics generated: {metrics_validation['errors']}")
                    continue
                
                evaluated.append((arch, metrics))
                
            except SecurityError as e:
                self.logger.error(f"Security violation during evaluation: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"Evaluation failed for architecture: {e}")
                continue
        
        return evaluated
    
    def _validate_final_result(self, architecture: Architecture, metrics: PerformanceMetrics) -> None:
        """Validate final search result."""
        # Validate architecture (warnings only)
        arch_validation = self.validator.validate_architecture(architecture)
        if not arch_validation['is_valid']:
            self.logger.warning(f"Final architecture has issues: {arch_validation['errors']}")
            # Don't raise error, just log
        
        # Validate metrics (warnings only)
        metrics_validation = validate_input(metrics, 'metrics')
        if not metrics_validation['is_valid']:
            self.logger.warning(f"Final metrics have issues: {metrics_validation['errors']}")
            # Don't raise error, just log
        
        # Check constraints satisfaction
        if metrics.latency_ms > self.config.max_latency_ms:
            self.logger.warning(f"Final result exceeds latency constraint: {metrics.latency_ms} > {self.config.max_latency_ms}")
        
        if metrics.accuracy < self.config.min_accuracy:
            self.logger.warning(f"Final result below accuracy constraint: {metrics.accuracy} < {self.config.min_accuracy}")
        
        self.logger.info("Final result validation completed")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.parallel_evaluator:
            self.parallel_evaluator.shutdown()
        
        # Log final performance stats
        if hasattr(self.predictor, 'get_performance_stats'):
            stats = self.predictor.get_performance_stats()
            self.logger.info(f"Final cache stats: {stats}")
        
        cache_stats = self.performance_optimizer.get_cache_stats()
        self.logger.info(f"Performance optimizer stats: {cache_stats}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()
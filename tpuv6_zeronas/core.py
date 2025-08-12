"""Core ZeroNAS search functionality for TPUv6 optimization."""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union

try:
    import numpy as np
except ImportError:
    np = None

from dataclasses import dataclass, asdict
import json

from .architecture import ArchitectureSpace, Architecture
from .predictor import TPUv6Predictor
from .metrics import PerformanceMetrics
from .validation import validate_input, ArchitectureValidator
from .monitoring import SearchMonitor, get_profiler, get_health_checker
from .security import get_resource_guard, SecurityError
from .parallel import ParallelEvaluator, WorkerConfig, get_performance_optimizer, get_resource_manager
from .caching import create_cached_predictor
from .error_handling import (
    robust_operation, safe_operation, validate_architecture_safe, 
    validate_metrics_safe, ErrorHandlingContext, get_error_handler
)


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
            if not self._validate_search_setup():
                raise ValueError("Search setup validation failed")
            
            config_validation = validate_input(self.config, 'search_config')
            if not config_validation['is_valid']:
                raise ValueError(f"Invalid search config: {config_validation['errors']}")
            
            # Check resource limits
            resource_guard = get_resource_guard()
            resource_guard.check_resource_limits(self.config)
            
        except Exception as e:
            self.logger.error(f"Initialization validation failed: {e}")
            raise
    
    def _validate_search_setup(self) -> bool:
        """Validate search configuration and dependencies."""
        try:
            if self.config.max_iterations <= 0:
                self.logger.error("max_iterations must be positive")
                return False
            
            if self.config.population_size <= 0:
                self.logger.error("population_size must be positive")
                return False
            
            if not (0.0 < self.config.mutation_rate < 1.0):
                self.logger.error("mutation_rate must be between 0 and 1")
                return False
            
            if not (0.0 < self.config.crossover_rate < 1.0):
                self.logger.error("crossover_rate must be between 0 and 1")
                return False
            
            test_arch = self.architecture_space.sample_random()
            test_metrics = self.predictor.predict(test_arch)
            
            if not isinstance(test_metrics, PerformanceMetrics):
                self.logger.error("Predictor does not return valid PerformanceMetrics")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Search setup validation failed: {e}")
            return False
        
    def search(self) -> Tuple[Architecture, PerformanceMetrics]:
        """Run neural architecture search to find optimal TPUv6 architecture."""
        try:
            self.monitor.log_search_start(self.config)
            self.logger.info("Starting TPUv6-ZeroNAS search...")
            self.logger.info(f"Target: {self.config.target_tops_w} TOPS/W")
            self.logger.info(f"Max iterations: {self.config.max_iterations}")
            self.logger.info(f"Population size: {self.config.population_size}")
            
            start_time = time.time()
            population = self._initialize_population()
            consecutive_failures = 0
            max_consecutive_failures = 5
            
            for iteration in range(self.config.max_iterations):
                # Enhanced system health monitoring
                if not self.health_checker.is_healthy():
                    self.logger.warning("System health issues detected, applying auto-recovery measures")
                    self._apply_auto_recovery_measures()
                
                self.logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
                
                try:
                    iteration_start = time.time()
                    
                    # Use optimized evaluation if available
                    if self.config.enable_parallel and self.parallel_evaluator:
                        evaluated_pop = self.performance_optimizer.optimize_population_evaluation(
                            population, self.parallel_evaluator
                        )
                    else:
                        evaluated_pop = self._evaluate_population_safe(population)
                    
                    if not evaluated_pop:
                        consecutive_failures += 1
                        self.logger.warning(f"No valid architectures in iteration {iteration + 1}")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            self.logger.error("Too many consecutive failures, terminating search")
                            break
                        
                        population = self._initialize_population()  # Reset population
                        continue
                    
                    consecutive_failures = 0
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
                    
                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration + 1}: {e}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error("Too many consecutive iteration failures")
                        break
                    
                    continue
            
            total_time = time.time() - start_time
            self.profiler.record_time('total_search', total_time)
            
            if self.best_architecture is None or self.best_metrics is None:
                # Fallback: try one more time with a fresh population
                self.logger.warning("No best architecture found, attempting fallback")
                final_pop = self._initialize_population()
                evaluated = self._evaluate_population_safe(final_pop)
                if evaluated:
                    best_pair = max(evaluated, key=lambda x: self._compute_score(x[1]))
                    self.best_architecture, self.best_metrics = best_pair
                else:
                    raise RuntimeError("Search failed to find any valid architecture")
            
            self.logger.info("Search completed successfully!")
            self.logger.info(f"Best architecture: {self.best_architecture.name}")
            self.logger.info(f"Best metrics: {self.best_metrics}")
            self.logger.info(f"Total evaluations: {len(self.search_history)}")
            
            # Final validation of result
            if self.best_architecture and self.best_metrics:
                self._validate_final_result(self.best_architecture, self.best_metrics)
                self.monitor.log_search_end(self.best_architecture, self.best_metrics)
            
            return self.best_architecture, self.best_metrics
            
        except Exception as e:
            self.monitor.log_error(e, {'iteration': iteration if 'iteration' in locals() else 0})
            self.logger.error(f"Search failed with exception: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @safe_operation(default_return=False, component="core")
    def _validate_architecture(self, arch: Architecture) -> bool:
        """Validate architecture before evaluation with enhanced safety checks."""
        return validate_architecture_safe(arch)
    
    @safe_operation(default_return=False, component="core")
    def _validate_metrics(self, metrics: PerformanceMetrics) -> bool:
        """Validate performance metrics with enhanced safety checks."""
        return validate_metrics_safe(metrics)
    
    @robust_operation(max_retries=3, component="core")
    def _initialize_population(self) -> List[Architecture]:
        """Initialize random population of architectures with robust error handling."""
        population = []
        attempts = 0
        max_attempts = self.config.population_size * 5  # Increased for safety
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        with ErrorHandlingContext("core", "population_initialization"):
            while len(population) < self.config.population_size and attempts < max_attempts:
                try:
                    arch = self.architecture_space.sample_random()
                    
                    if self._validate_architecture(arch):
                        population.append(arch)
                        consecutive_failures = 0  # Reset counter on success
                        self.logger.debug(f"Valid architecture added: {arch.name}")
                    else:
                        consecutive_failures += 1
                        self.logger.debug(f"Invalid architecture rejected: {getattr(arch, 'name', 'unnamed')}")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            self.logger.warning("Too many consecutive invalid architectures, adjusting sampling")
                            # Reset counter and continue
                            consecutive_failures = 0
                        
                except Exception as e:
                    consecutive_failures += 1
                    self.logger.warning(f"Failed to create architecture (attempt {attempts}): {e}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error("Too many consecutive failures in architecture generation")
                        break
                    
                attempts += 1
            
            # Ensure we have at least some population
            if len(population) == 0:
                self.logger.error("Failed to initialize any valid architectures, creating minimal fallback")
                # Create a minimal fallback architecture
                try:
                    fallback_arch = self.architecture_space._create_minimal_architecture()
                    if self._validate_architecture(fallback_arch):
                        population.append(fallback_arch)
                except Exception as e:
                    raise RuntimeError(f"Failed to create fallback architecture: {e}")
            
            if len(population) < self.config.population_size:
                self.logger.warning(f"Initialized {len(population)} architectures (target: {self.config.population_size})")
            else:
                self.logger.info(f"Successfully initialized {len(population)} architectures")
            
            return population
    
    def _evaluate_population(
        self, 
        population: List[Architecture]
    ) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Evaluate population using TPUv6 predictor."""
        evaluated = []
        failed_evaluations = 0
        
        for i, arch in enumerate(population):
            try:
                if not self._validate_architecture(arch):
                    self.logger.debug(f"Skipping invalid architecture {i}")
                    failed_evaluations += 1
                    continue
                    
                metrics = self.predictor.predict(arch)
                
                if not self._validate_metrics(metrics):
                    self.logger.debug(f"Invalid metrics for architecture {arch.name}")
                    failed_evaluations += 1
                    continue
                    
                evaluated.append((arch, metrics))
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate architecture {arch.name}: {e}")
                failed_evaluations += 1
                continue
        
        if failed_evaluations > 0:
            self.logger.info(f"Failed to evaluate {failed_evaluations}/{len(population)} architectures")
        
        return evaluated
    
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
                
                # Additional validation
                if not self._validate_architecture(arch):
                    self.logger.debug(f"Skipping invalid architecture: {arch.name}")
                    continue
                
                # Check resource constraints
                resource_guard = get_resource_guard()
                resource_guard.check_architecture_complexity(arch)
                
                metrics = self.predictor.predict(arch)
                
                # Validate metrics
                if not self._validate_metrics(metrics):
                    self.logger.debug(f"Invalid metrics for architecture {arch.name}")
                    continue
                
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
            elif np and np.random.random() < self.config.crossover_rate:
                parent1, parent2 = self._select_parents(parent_pool)
                child = self._crossover(parent1, parent2)
            else:
                # Use hash-based randomness as fallback if numpy not available
                crossover_chance = hash(time.time()) % 100 / 100.0
                if crossover_chance < self.config.crossover_rate:
                    parent1, parent2 = self._select_parents(parent_pool)
                    child = self._crossover(parent1, parent2)
                else:
                    parent = self._select_parents(parent_pool, n=1)[0]
                    child = parent
            
            # Apply mutation
            if np and np.random.random() < self.config.mutation_rate:
                child = self._mutate(child)
            else:
                mutation_chance = hash(time.time() + 1) % 100 / 100.0
                if mutation_chance < self.config.mutation_rate:
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
            
            if np:
                # Use numpy for better randomness if available
                tournament = np.random.choice(len(population), tournament_size, replace=False)
                best_idx = max(tournament, key=lambda i: self._compute_score(population[i][1]))
            else:
                # Fallback to hash-based randomness
                tournament = [hash(time.time() + i * 37) % len(population) for i in range(tournament_size)]
                best_idx = tournament[0]
                best_score = self._compute_score(population[best_idx][1])
                for idx in tournament[1:]:
                    score = self._compute_score(population[idx][1])
                    if score > best_score:
                        best_idx = idx
                        best_score = score
            
            parents.append(population[best_idx][0])
        
        return parents
    
    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover two parent architectures."""
        return self.architecture_space.crossover(parent1, parent2)
    
    def _mutate(self, architecture: Architecture) -> Architecture:
        """Mutate architecture."""
        return self.architecture_space.mutate(architecture)
    
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
    
    def _apply_auto_recovery_measures(self) -> None:
        """Apply automatic recovery measures when system health is compromised."""
        try:
            # Reduce resource consumption temporarily
            if self.config.enable_parallel and self.parallel_evaluator:
                self.logger.info("Reducing resource usage for recovery")
                if hasattr(self.parallel_evaluator, 'reduce_workers'):
                    self.parallel_evaluator.reduce_workers()
                else:
                    self.logger.debug("Parallel evaluator does not support worker reduction")
            
            # Clear caches to free memory
            if hasattr(self.predictor, 'clear_cache'):
                self.logger.info("Clearing predictor cache for recovery")
                self.predictor.clear_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Auto-recovery measures applied successfully")
            
        except Exception as e:
            self.logger.error(f"Auto-recovery failed: {e}")
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics for analysis."""
        stats = {
            'search_completed': self.best_architecture is not None,
            'total_evaluations': len(self.search_history),
            'best_metrics': asdict(self.best_metrics) if self.best_metrics else None,
            'convergence_metrics': self._analyze_convergence(),
            'resource_usage': self._get_resource_usage(),
            'error_summary': self._get_error_summary()
        }
        return stats
    
    def _analyze_convergence(self) -> Dict[str, float]:
        """Analyze search convergence patterns."""
        if len(self.search_history) < 10:
            return {'convergence_rate': 0.0, 'improvement_trend': 0.0}
        
        scores = [self._compute_score(metrics) for _, metrics in self.search_history[-50:]]
        
        if len(scores) < 2:
            return {'convergence_rate': 0.0, 'improvement_trend': 0.0}
        
        # Calculate improvement trend
        recent_scores = scores[-10:]
        early_scores = scores[:10] if len(scores) >= 20 else scores[:len(scores)//2]
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        early_avg = sum(early_scores) / len(early_scores)
        improvement_trend = (recent_avg - early_avg) / max(abs(early_avg), 1e-6)
        
        # Calculate convergence rate (variance reduction)
        if len(scores) >= 20:
            early_var = sum((s - early_avg) ** 2 for s in early_scores) / len(early_scores)
            recent_var = sum((s - recent_avg) ** 2 for s in recent_scores) / len(recent_scores)
            convergence_rate = max(0, (early_var - recent_var) / max(early_var, 1e-6))
        else:
            convergence_rate = 0.0
        
        return {
            'convergence_rate': convergence_rate,
            'improvement_trend': improvement_trend
        }
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            return {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
        except ImportError:
            self.logger.debug("psutil not available, using basic resource monitoring")
            return {'memory_mb': 0, 'cpu_percent': 0, 'num_threads': 1}
        except Exception as e:
            self.logger.debug(f"Resource usage monitoring failed: {e}")
            return {'memory_mb': 0, 'cpu_percent': 0, 'num_threads': 1}
    
    def _get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors encountered during search."""
        # This would be enhanced with actual error tracking
        return {
            'prediction_errors': 0,
            'evaluation_errors': 0,
            'validation_errors': 0,
            'recovery_actions': 0
        }
    
    def save_search_state(self, filepath: str) -> bool:
        """Save search state for resumption capability."""
        try:
            state = {
                'config': asdict(self.config),
                'search_history': [(arch.name, asdict(metrics)) for arch, metrics in self.search_history[-100:]],  # Last 100
                'best_architecture': self.best_architecture.name if self.best_architecture else None,
                'best_metrics': asdict(self.best_metrics) if self.best_metrics else None,
                'statistics': self.get_search_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Search state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save search state: {e}")
            return False
    
    def cleanup(self) -> None:
        """Enhanced cleanup with comprehensive resource management."""
        try:
            # Shutdown parallel processing
            if self.parallel_evaluator:
                self.logger.info("Shutting down parallel evaluator")
                self.parallel_evaluator.shutdown()
            
            # Log comprehensive performance stats
            if hasattr(self.predictor, 'get_performance_stats'):
                stats = self.predictor.get_performance_stats()
                self.logger.info(f"Predictor final stats: {stats}")
            
            # Performance optimizer stats
            if hasattr(self.performance_optimizer, 'get_cache_stats'):
                cache_stats = self.performance_optimizer.get_cache_stats()
                self.logger.info(f"Performance optimizer stats: {cache_stats}")
            
            # Monitor final resource usage
            resource_usage = self._get_resource_usage()
            self.logger.info(f"Final resource usage: {resource_usage}")
            
            # Save final search state for analysis
            try:
                self.save_search_state('.search_state_final.json')
            except Exception as e:
                self.logger.debug(f"Could not save final search state: {e}")
            
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Safe destruction with error handling."""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid exceptions during destruction

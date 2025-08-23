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
from .advanced_optimization import AdvancedSearchOptimizer
from .hyperparameter_optimization import AdaptiveHyperparameterManager


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
    enable_research: bool = False  # Advanced research analysis
    enable_adaptive_scaling: bool = True  # Generation 3: Adaptive scaling
    enable_advanced_optimization: bool = True  # Generation 3: Advanced optimization 
    enable_hyperparameter_optimization: bool = True  # Generation 3: Hyperparameter optimization
    parallel_workers: int = None  # Auto-detect if None
    adaptive_scaling_factor: float = 1.5  # Dynamic scaling multiplier


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
        
        # Enhanced search state tracking
        self.constraint_violation_history = {
            'accuracy': [], 'latency': [], 'energy': []
        }
        self.search_adaptation_enabled = True
        
        # Generation 3: Advanced optimization
        self.advanced_optimizer = None
        if getattr(self.config, 'enable_advanced_optimization', False):
            try:
                self.advanced_optimizer = AdvancedSearchOptimizer(self.predictor)
                self.logger.info("🚀 Advanced optimization enabled")
            except Exception as e:
                self.logger.warning(f"Advanced optimization initialization failed: {e}")
        
        # Generation 3: Hyperparameter optimization
        self.hp_manager = None
        if getattr(self.config, 'enable_hyperparameter_optimization', False):
            try:
                self.hp_manager = AdaptiveHyperparameterManager()
                self.logger.info("🎯 Hyperparameter optimization enabled")
            except Exception as e:
                self.logger.warning(f"Hyperparameter optimization initialization failed: {e}")
        
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
        
        # Setup adaptive scaling if enabled (Generation 3)
        self.adaptive_scaler = None
        if getattr(self.config, 'enable_adaptive_scaling', False):
            try:
                from .adaptive_scaling import AdaptiveScaler, ScalingConfig
                scaling_config = ScalingConfig(
                    target_throughput=self.config.population_size * 2,
                    max_replicas=min(8, self.config.population_size // 2),
                    min_replicas=1
                )
                self.adaptive_scaler = AdaptiveScaler(scaling_config)
                self.logger.info("🚀 Adaptive scaling system enabled")
            except ImportError:
                self.logger.debug("Adaptive scaling not available, continuing without")
    
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
                    
                    # Use advanced optimization if available
                    if self.advanced_optimizer:
                        evaluated_pop = self.advanced_optimizer.optimize_population_evaluation(population)
                    elif self.config.enable_parallel and self.parallel_evaluator:
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
                    
                    # Adaptive hyperparameter optimization
                    if self.hp_manager and iteration > 0:
                        current_score = self._compute_score(self.best_metrics) if self.best_metrics else 0.0
                        if self.hp_manager.should_adapt_hyperparameters(iteration):
                            adapted_params = self.hp_manager.adapt_hyperparameters(self.config, current_score)
                            if adapted_params:
                                self.logger.info(f"🔄 Hyperparameters adapted at iteration {iteration + 1}")
                    
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
            
            # Advanced research analysis (if enabled)
            if hasattr(self.config, 'enable_research') and self.config.enable_research:
                self._run_advanced_research_analysis(population)
            
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
        """Update best architecture with constraint violation tracking."""
        for arch, metrics in evaluated_pop:
            # Track constraint violations for adaptive behavior
            self._track_constraint_violations(metrics)
            
            if self._is_better_architecture(metrics):
                self.best_architecture = arch
                self.best_metrics = metrics
                
        self.search_history.extend(evaluated_pop)
        
        # Adapt search strategy based on constraint violation patterns
        if self.search_adaptation_enabled and len(self.search_history) > 20:
            self._adapt_search_strategy()
    
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
        """Compute enhanced multi-objective score with adaptive weighting."""
        # Progressive constraint handling with graduated penalties
        constraint_penalty = 0.0
        constraint_satisfaction = 1.0
        
        # Accuracy constraint with graduated penalty
        if metrics.accuracy < self.config.min_accuracy:
            accuracy_deficit = self.config.min_accuracy - metrics.accuracy
            if accuracy_deficit > 0.1:  # Severe violation
                constraint_penalty += accuracy_deficit * 20
                constraint_satisfaction *= 0.3
            else:  # Minor violation
                constraint_penalty += accuracy_deficit * 5
                constraint_satisfaction *= 0.7
        else:
            # Bonus for exceeding accuracy requirements
            accuracy_bonus = min(0.05, (metrics.accuracy - self.config.min_accuracy) * 0.5)
            constraint_penalty -= accuracy_bonus
        
        # Latency constraint with graduated penalty  
        if metrics.latency_ms > self.config.max_latency_ms:
            latency_excess = metrics.latency_ms - self.config.max_latency_ms
            if latency_excess > self.config.max_latency_ms * 0.5:  # Severe violation
                constraint_penalty += latency_excess * 0.5
                constraint_satisfaction *= 0.5
            else:  # Minor violation
                constraint_penalty += latency_excess * 0.1
                constraint_satisfaction *= 0.8
        
        # Enhanced scoring components with better balance
        efficiency_score = min(metrics.tops_per_watt / self.config.target_tops_w, 1.5)
        accuracy_score = metrics.accuracy
        latency_score = max(0.1, 1.0 / max(metrics.latency_ms, 0.1))  # Favor lower latency
        energy_score = max(0.1, 1.0 / max(metrics.energy_mj, 0.1))  # Favor lower energy
        
        # Adaptive weighting based on constraint satisfaction
        if constraint_satisfaction > 0.9:  # Well-constrained solution
            # Focus on optimization objectives
            score = (0.35 * efficiency_score + 
                    0.35 * accuracy_score + 
                    0.15 * latency_score + 
                    0.15 * energy_score)
        else:  # Constraint-violating solution
            # Heavily weight constraint satisfaction
            score = (0.2 * efficiency_score + 
                    0.6 * accuracy_score + 
                    0.1 * latency_score + 
                    0.1 * energy_score)
        
        # Apply constraint penalty and satisfaction multiplier
        final_score = (score * constraint_satisfaction) - constraint_penalty
        
        # Ensure reasonable bounds with better dynamic range
        return max(final_score, -20.0)
    
    def _track_constraint_violations(self, metrics: PerformanceMetrics) -> None:
        """Track constraint violations to adapt search strategy."""
        try:
            # Track accuracy violations
            if metrics.accuracy < self.config.min_accuracy:
                self.constraint_violation_history['accuracy'].append(
                    self.config.min_accuracy - metrics.accuracy
                )
            
            # Track latency violations
            if metrics.latency_ms > self.config.max_latency_ms:
                self.constraint_violation_history['latency'].append(
                    metrics.latency_ms - self.config.max_latency_ms
                )
            
            # Track energy efficiency (if significantly poor)
            if metrics.tops_per_watt < self.config.target_tops_w * 0.3:
                self.constraint_violation_history['energy'].append(
                    self.config.target_tops_w * 0.3 - metrics.tops_per_watt
                )
            
            # Keep only recent history (last 50 violations per type)
            for constraint_type in self.constraint_violation_history:
                if len(self.constraint_violation_history[constraint_type]) > 50:
                    self.constraint_violation_history[constraint_type] = \
                        self.constraint_violation_history[constraint_type][-50:]
                        
        except Exception as e:
            self.logger.debug(f"Constraint tracking failed: {e}")
    
    def _adapt_search_strategy(self) -> None:
        """Adapt search strategy based on constraint violation patterns."""
        try:
            total_evaluations = len(self.search_history)
            
            # Calculate violation rates
            accuracy_violations = len(self.constraint_violation_history['accuracy'])
            latency_violations = len(self.constraint_violation_history['latency'])
            energy_violations = len(self.constraint_violation_history['energy'])
            
            accuracy_violation_rate = accuracy_violations / max(total_evaluations, 1)
            latency_violation_rate = latency_violations / max(total_evaluations, 1)
            
            # Adapt mutation rate based on constraint satisfaction
            if accuracy_violation_rate > 0.7:  # High accuracy violations
                # Encourage more conservative mutations to improve accuracy
                self.config.mutation_rate = max(0.05, self.config.mutation_rate * 0.8)
                self.logger.debug("Reducing mutation rate due to accuracy violations")
            elif accuracy_violation_rate < 0.2 and latency_violation_rate < 0.2:
                # Good constraint satisfaction, can be more aggressive
                self.config.mutation_rate = min(0.3, self.config.mutation_rate * 1.1)
                self.logger.debug("Increasing mutation rate due to good constraint satisfaction")
            
            # Adapt population diversity based on convergence
            if hasattr(self, 'parallel_evaluator') and self.parallel_evaluator:
                if accuracy_violation_rate > 0.5:
                    # Need more diversity to find feasible solutions
                    self.config.population_size = min(100, int(self.config.population_size * 1.2))
                    self.logger.debug("Increasing population size for better exploration")
            
        except Exception as e:
            self.logger.debug(f"Search adaptation failed: {e}")
    
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
    
    def _run_advanced_research_analysis(self, population: List[Architecture]) -> None:
        """Run advanced research analysis on the population."""
        try:
            from .advanced_research_engine import AdvancedResearchEngine
            
            self.logger.info("🔬 Running advanced research analysis...")
            research_engine = AdvancedResearchEngine(self.predictor, self.config)
            
            research_results = research_engine.run_comprehensive_research_experiment(
                population[:50],  # Limit for efficiency
                ['pareto_optimization', 'scaling_law_discovery', 'pattern_discovery']
            )
            
            # Save research results
            from pathlib import Path
            research_path = Path('advanced_research_results.json')
            with open(research_path, 'w') as f:
                import json
                json.dump(research_results, f, indent=2, default=str)
            
            self.logger.info(f"💾 Advanced research results saved to {research_path}")
            
        except ImportError as e:
            self.logger.warning("Advanced research engine not available")
        except Exception as e:
            self.logger.error(f"Advanced research analysis failed: {e}")
    
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
            
            # Generate advanced optimization report
            if self.advanced_optimizer:
                try:
                    opt_report = self.advanced_optimizer.get_optimization_report()
                    self.logger.info(f"📊 Advanced optimization report: {opt_report}")
                    self.advanced_optimizer.cleanup()
                except Exception as e:
                    self.logger.debug(f"Advanced optimizer cleanup failed: {e}")
            
            # Generate hyperparameter optimization summary
            if self.hp_manager:
                try:
                    hp_summary = self.hp_manager.get_optimization_summary()
                    self.logger.info(f"🎯 Hyperparameter optimization summary: {hp_summary}")
                except Exception as e:
                    self.logger.debug(f"Hyperparameter optimization summary failed: {e}")
            
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Safe destruction with error handling."""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid exceptions during destruction

"""Core ZeroNAS search functionality for TPUv6 optimization."""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union

try:
    import numpy as np
except ImportError:
    np = None

from dataclasses import dataclass

from .architecture import ArchitectureSpace, Architecture
from .predictor import TPUv6Predictor
from .metrics import PerformanceMetrics


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
        
        self.best_architecture: Optional[Architecture] = None
        self.best_metrics: Optional[PerformanceMetrics] = None
        self.search_history: List[Tuple[Architecture, PerformanceMetrics]] = []
        
    def search(self) -> Tuple[Architecture, PerformanceMetrics]:
        """Run neural architecture search to find optimal TPUv6 architecture."""
        if not self._validate_search_setup():
            raise ValueError("Search setup validation failed")
            
        self.logger.info("Starting TPUv6-ZeroNAS search...")
        self.logger.info(f"Target: {self.config.target_tops_w} TOPS/W")
        self.logger.info(f"Max iterations: {self.config.max_iterations}")
        self.logger.info(f"Population size: {self.config.population_size}")
        
        try:
            population = self._initialize_population()
            consecutive_failures = 0
            max_consecutive_failures = 5
            
            for iteration in range(self.config.max_iterations):
                self.logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
                
                try:
                    evaluated_pop = self._evaluate_population(population)
                    
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
            
            if self.best_architecture is None or self.best_metrics is None:
                raise RuntimeError("Search failed to find any valid architecture")
            
            self.logger.info("Search completed successfully!")
            self.logger.info(f"Best architecture: {self.best_architecture.name}")
            self.logger.info(f"Best metrics: {self.best_metrics}")
            self.logger.info(f"Total evaluations: {len(self.search_history)}")
            
            return self.best_architecture, self.best_metrics
            
        except Exception as e:
            self.logger.error(f"Search failed with exception: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
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
    
    def _validate_architecture(self, arch: Architecture) -> bool:
        """Validate architecture before evaluation."""
        try:
            if not arch or not arch.layers:
                return False
            
            if arch.total_params <= 0:
                return False
            
            if arch.total_ops <= 0:
                return False
            
            if arch.memory_mb <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Architecture validation failed: {e}")
            return False
    
    def _validate_metrics(self, metrics: PerformanceMetrics) -> bool:
        """Validate performance metrics."""
        try:
            if not isinstance(metrics, PerformanceMetrics):
                return False
            
            if not (0.0 < metrics.latency_ms < 1000.0):  # Reasonable latency bounds
                return False
            
            if not (0.0 < metrics.energy_mj < 10000.0):  # Reasonable energy bounds
                return False
            
            if not (0.0 <= metrics.accuracy <= 1.0):
                return False
            
            if not (0.0 < metrics.tops_per_watt < 1000.0):  # Reasonable efficiency bounds
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Metrics validation failed: {e}")
            return False
    
    def _initialize_population(self) -> List[Architecture]:
        """Initialize random population of architectures."""
        population = []
        attempts = 0
        max_attempts = self.config.population_size * 3
        
        while len(population) < self.config.population_size and attempts < max_attempts:
            try:
                arch = self.architecture_space.sample_random()
                
                if self._validate_architecture(arch):
                    population.append(arch)
                else:
                    self.logger.debug(f"Invalid architecture rejected: {arch.name}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to create architecture: {e}")
                
            attempts += 1
        
        if len(population) == 0:
            raise RuntimeError("Failed to initialize any valid architectures")
        
        if len(population) < self.config.population_size:
            self.logger.warning(f"Only initialized {len(population)} architectures (target: {self.config.population_size})")
        
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
        return improvement < self.config.early_stop_threshold
    
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
            crossover_chance = hash(time.time()) % 100 / 100.0
            if crossover_chance < self.config.crossover_rate:
                parent1, parent2 = self._select_parents(evaluated_pop[:elite_size * 2])
                child = self._crossover(parent1, parent2)
            else:
                parent = self._select_parents(evaluated_pop[:elite_size * 2], n=1)[0]
                child = parent
                
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
        parents = []
        for _ in range(n):
            tournament_size = min(3, len(population))
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
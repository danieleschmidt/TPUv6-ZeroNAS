"""Core ZeroNAS search functionality for TPUv6 optimization."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
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
        self.logger.info("Starting TPUv6-ZeroNAS search...")
        self.logger.info(f"Target: {self.config.target_tops_w} TOPS/W")
        
        population = self._initialize_population()
        
        for iteration in range(self.config.max_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
            
            evaluated_pop = self._evaluate_population(population)
            
            self._update_best(evaluated_pop)
            
            if self._should_early_stop():
                self.logger.info(f"Early stopping at iteration {iteration + 1}")
                break
                
            population = self._evolve_population(evaluated_pop)
            
        self.logger.info("Search completed!")
        self.logger.info(f"Best architecture: {self.best_architecture}")
        self.logger.info(f"Best metrics: {self.best_metrics}")
        
        return self.best_architecture, self.best_metrics
    
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
            if np.random.random() < self.config.crossover_rate:
                parent1, parent2 = self._select_parents(evaluated_pop[:elite_size * 2])
                child = self._crossover(parent1, parent2)
            else:
                parent = self._select_parents(evaluated_pop[:elite_size * 2], n=1)[0]
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
        parents = []
        for _ in range(n):
            tournament_size = min(3, len(population))
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = min(tournament, key=lambda i: self._compute_score(population[i][1]))
            parents.append(population[best_idx][0])
        return parents
    
    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover two parent architectures."""
        return self.architecture_space.crossover(parent1, parent2)
    
    def _mutate(self, architecture: Architecture) -> Architecture:
        """Mutate architecture."""
        return self.architecture_space.mutate(architecture)
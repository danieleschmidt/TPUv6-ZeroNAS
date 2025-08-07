"""Advanced optimization techniques for TPUv6-ZeroNAS."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math

from .architecture import Architecture, ArchitectureSpace
from .predictor import TPUv6Predictor
from .metrics import PerformanceMetrics, MetricsAggregator


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    MULTI_OBJECTIVE = "multi_objective"
    PROGRESSIVE = "progressive"


@dataclass
class OptimizationConfig:
    """Configuration for advanced optimization."""
    strategy: OptimizationStrategy = OptimizationStrategy.EVOLUTIONARY
    max_evaluations: int = 5000
    convergence_threshold: float = 1e-6
    exploration_factor: float = 0.3
    exploitation_factor: float = 0.7
    population_diversity_threshold: float = 0.1
    adaptive_parameters: bool = True
    use_surrogate_model: bool = True
    warm_start: bool = True


class SurrogateModel:
    """Lightweight surrogate model for fast architecture evaluation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.training_data = []
        self.is_trained = False
        
        # Simple analytical model coefficients
        self.latency_coeff = {
            'ops': 1e-9,
            'params': 5e-8,
            'depth': 0.1,
            'width': 0.001,
            'base': 0.5
        }
        
        self.energy_coeff = {
            'ops': 5e-6,
            'params': 1e-7,
            'memory': 0.1,
            'base': 1.0
        }
        
        self.accuracy_coeff = {
            'params': 1e-7,
            'depth': 0.02,
            'width': 0.001,
            'base': 0.88
        }
    
    def predict_fast(self, architecture: Architecture) -> PerformanceMetrics:
        """Fast surrogate prediction without complex models."""
        try:
            # Analytical latency prediction
            latency = (
                self.latency_coeff['ops'] * architecture.total_ops +
                self.latency_coeff['params'] * architecture.total_params +
                self.latency_coeff['depth'] * architecture.depth +
                self.latency_coeff['width'] * architecture.avg_width +
                self.latency_coeff['base']
            )
            
            # Analytical energy prediction
            energy = (
                self.energy_coeff['ops'] * architecture.total_ops +
                self.energy_coeff['params'] * architecture.total_params +
                self.energy_coeff['memory'] * architecture.memory_mb +
                self.energy_coeff['base']
            )
            
            # Analytical accuracy prediction
            accuracy = (
                self.accuracy_coeff['base'] +
                min(0.1, self.accuracy_coeff['params'] * architecture.total_params) +
                min(0.05, self.accuracy_coeff['depth'] * architecture.depth) +
                min(0.03, self.accuracy_coeff['width'] * architecture.avg_width / 1000)
            )
            
            # Apply diminishing returns
            accuracy = min(0.99, accuracy)
            
            # Calculate TOPS/W
            tops_per_watt = (architecture.total_ops / 1e12) / max(energy / 1000, 1e-6)
            
            return PerformanceMetrics(
                latency_ms=max(0.1, latency),
                energy_mj=max(0.1, energy),
                accuracy=max(0.5, accuracy),
                tops_per_watt=max(1.0, tops_per_watt),
                memory_mb=architecture.memory_mb,
                flops=architecture.total_ops
            )
            
        except Exception as e:
            self.logger.warning(f"Surrogate prediction failed: {e}")
            # Return conservative fallback
            return PerformanceMetrics(
                latency_ms=10.0,
                energy_mj=100.0,
                accuracy=0.85,
                tops_per_watt=30.0,
                memory_mb=architecture.memory_mb,
                flops=architecture.total_ops
            )
    
    def update_coefficients(self, training_data: List[Tuple[Architecture, PerformanceMetrics]]):
        """Update model coefficients based on real evaluations."""
        if len(training_data) < 10:
            return
        
        try:
            # Simple coefficient adjustment based on errors
            for arch, true_metrics in training_data[-20:]:  # Use recent data
                pred_metrics = self.predict_fast(arch)
                
                # Adjust latency coefficients
                latency_error = true_metrics.latency_ms - pred_metrics.latency_ms
                if abs(latency_error) > 0.5:
                    adjust_factor = 1.0 + (latency_error / pred_metrics.latency_ms) * 0.1
                    self.latency_coeff['ops'] *= adjust_factor
                
                # Adjust energy coefficients
                energy_error = true_metrics.energy_mj - pred_metrics.energy_mj
                if abs(energy_error) > 5.0:
                    adjust_factor = 1.0 + (energy_error / pred_metrics.energy_mj) * 0.1
                    self.energy_coeff['ops'] *= adjust_factor
                
                # Adjust accuracy coefficients
                accuracy_error = true_metrics.accuracy - pred_metrics.accuracy
                if abs(accuracy_error) > 0.02:
                    adjust_factor = 1.0 + accuracy_error * 0.1
                    self.accuracy_coeff['params'] *= adjust_factor
            
            self.logger.debug("Surrogate model coefficients updated")
            
        except Exception as e:
            self.logger.warning(f"Coefficient update failed: {e}")


class ProgressiveSearchOptimizer:
    """Progressive optimization that starts coarse and refines."""
    
    def __init__(
        self,
        architecture_space: ArchitectureSpace,
        predictor: TPUv6Predictor,
        config: OptimizationConfig
    ):
        self.architecture_space = architecture_space
        self.predictor = predictor
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.surrogate_model = SurrogateModel()
        self.metrics_aggregator = MetricsAggregator()
        
        # Progressive search phases
        self.phases = [
            {"name": "exploration", "evaluations": config.max_evaluations // 3, "population": 100, "diversity": 0.8},
            {"name": "exploitation", "evaluations": config.max_evaluations // 3, "population": 50, "diversity": 0.4},
            {"name": "refinement", "evaluations": config.max_evaluations // 3, "population": 20, "diversity": 0.1}
        ]
        
        self.current_phase = 0
        self.phase_evaluations = 0
        
    def optimize(self) -> Tuple[Architecture, PerformanceMetrics]:
        """Run progressive optimization."""
        self.logger.info("Starting progressive optimization...")
        
        best_architecture = None
        best_metrics = None
        total_evaluations = 0
        
        for phase_idx, phase in enumerate(self.phases):
            self.current_phase = phase_idx
            self.phase_evaluations = 0
            
            self.logger.info(f"Phase {phase_idx + 1}: {phase['name']} ({phase['evaluations']} evaluations)")
            
            # Adjust search parameters for this phase
            phase_arch_space = self._adapt_architecture_space_for_phase(phase)
            
            # Run phase optimization
            phase_best_arch, phase_best_metrics = self._run_phase_optimization(
                phase_arch_space, phase
            )
            
            total_evaluations += self.phase_evaluations
            
            # Update global best
            if best_metrics is None or self._is_better_metrics(phase_best_metrics, best_metrics):
                best_architecture = phase_best_arch
                best_metrics = phase_best_metrics
            
            # Update surrogate model with real evaluations
            if self.config.use_surrogate_model:
                recent_data = self.metrics_aggregator.metrics_history[-100:]
                if len(recent_data) >= 10:
                    training_data = [(arch, metrics) for arch, metrics in recent_data]
                    self.surrogate_model.update_coefficients(training_data)
            
            self.logger.info(f"Phase {phase_idx + 1} completed. Best: {phase_best_metrics}")
        
        self.logger.info(f"Progressive optimization completed. Total evaluations: {total_evaluations}")
        
        if best_architecture is None or best_metrics is None:
            raise RuntimeError("Progressive optimization failed to find any valid architecture")
        
        return best_architecture, best_metrics
    
    def _adapt_architecture_space_for_phase(self, phase: Dict[str, Any]) -> ArchitectureSpace:
        """Adapt architecture space for current phase."""
        adapted_space = ArchitectureSpace(
            input_shape=self.architecture_space.input_shape,
            num_classes=self.architecture_space.num_classes,
            max_depth=self.architecture_space.max_depth,
            channel_choices=self.architecture_space.channel_choices.copy(),
            kernel_choices=self.architecture_space.kernel_choices.copy()
        )
        
        if phase['name'] == 'exploration':
            # Wide search space
            adapted_space.max_depth = max(15, self.architecture_space.max_depth)
            adapted_space.channel_choices.extend([32, 48, 96, 192, 384, 768])
            
        elif phase['name'] == 'exploitation':
            # Focus on promising regions
            if hasattr(self, 'promising_ranges'):
                adapted_space.max_depth = min(12, self.architecture_space.max_depth)
                adapted_space.channel_choices = [c for c in adapted_space.channel_choices if c <= 512]
            
        elif phase['name'] == 'refinement':
            # Fine-grained search around best solutions
            if hasattr(self, 'best_patterns'):
                adapted_space.max_depth = min(10, self.architecture_space.max_depth)
                adapted_space.channel_choices = [c for c in adapted_space.channel_choices if c <= 256]
        
        return adapted_space
    
    def _run_phase_optimization(
        self,
        phase_arch_space: ArchitectureSpace,
        phase: Dict[str, Any]
    ) -> Tuple[Architecture, PerformanceMetrics]:
        """Run optimization for a single phase."""
        population_size = phase['population']
        target_evaluations = phase['evaluations']
        diversity_target = phase['diversity']
        
        # Initialize population
        population = self._initialize_diverse_population(phase_arch_space, population_size)
        
        best_arch = None
        best_metrics = None
        
        evaluations_this_phase = 0
        generation = 0
        
        while evaluations_this_phase < target_evaluations:
            generation += 1
            
            # Evaluate population
            evaluated_population = []
            
            for arch in population:
                # Use surrogate model for pre-screening
                if self.config.use_surrogate_model and evaluations_this_phase > 50:
                    surrogate_metrics = self.surrogate_model.predict_fast(arch)
                    
                    # Only evaluate promising architectures with real predictor
                    if self._is_promising_architecture(surrogate_metrics):
                        real_metrics = self.predictor.predict(arch)
                        evaluated_population.append((arch, real_metrics))
                        self.metrics_aggregator.add_metrics(real_metrics)
                        evaluations_this_phase += 1
                    else:
                        # Use surrogate prediction for non-promising architectures
                        evaluated_population.append((arch, surrogate_metrics))
                else:
                    # Direct evaluation
                    real_metrics = self.predictor.predict(arch)
                    evaluated_population.append((arch, real_metrics))
                    self.metrics_aggregator.add_metrics(real_metrics)
                    evaluations_this_phase += 1
                
                if evaluations_this_phase >= target_evaluations:
                    break
            
            # Update phase best
            for arch, metrics in evaluated_population:
                if best_metrics is None or self._is_better_metrics(metrics, best_metrics):
                    best_arch = arch
                    best_metrics = metrics
            
            # Evolve population
            if evaluations_this_phase < target_evaluations:
                population = self._evolve_population_with_diversity(
                    evaluated_population, phase_arch_space, diversity_target
                )
            
            # Adaptive parameter adjustment
            if self.config.adaptive_parameters and generation % 10 == 0:
                self._adapt_parameters_based_on_progress(evaluated_population)
            
            self.logger.debug(f"Phase {self.current_phase + 1}, Gen {generation}: {evaluations_this_phase}/{target_evaluations} evaluations")
        
        self.phase_evaluations = evaluations_this_phase
        return best_arch, best_metrics
    
    def _initialize_diverse_population(
        self,
        arch_space: ArchitectureSpace,
        population_size: int
    ) -> List[Architecture]:
        """Initialize diverse population with different architectural patterns."""
        population = []
        
        # Different architectural patterns
        patterns = [
            {'depth_range': (3, 8), 'channel_scale': 0.5, 'name': 'shallow_narrow'},
            {'depth_range': (8, 15), 'channel_scale': 1.0, 'name': 'medium_standard'},
            {'depth_range': (15, 25), 'channel_scale': 1.5, 'name': 'deep_wide'},
            {'depth_range': (5, 12), 'channel_scale': 0.75, 'name': 'efficient'},
        ]
        
        per_pattern = population_size // len(patterns)
        remainder = population_size % len(patterns)
        
        for i, pattern in enumerate(patterns):
            pattern_count = per_pattern + (1 if i < remainder else 0)
            
            for _ in range(pattern_count):
                try:
                    # Temporarily adjust architecture space
                    original_max_depth = arch_space.max_depth
                    original_channels = arch_space.channel_choices.copy()
                    
                    arch_space.max_depth = pattern['depth_range'][1]
                    scaled_channels = [int(c * pattern['channel_scale']) for c in original_channels]
                    arch_space.channel_choices = [c for c in scaled_channels if 8 <= c <= 2048]
                    
                    arch = arch_space.sample_random()
                    arch.name = f"{pattern['name']}_{arch.name}"
                    population.append(arch)
                    
                    # Restore original space
                    arch_space.max_depth = original_max_depth
                    arch_space.channel_choices = original_channels
                    
                except Exception as e:
                    self.logger.debug(f"Failed to create {pattern['name']} architecture: {e}")
                    continue
        
        # Fill remaining spots with random architectures
        while len(population) < population_size:
            try:
                arch = arch_space.sample_random()
                population.append(arch)
            except:
                break
        
        return population
    
    def _is_promising_architecture(self, surrogate_metrics: PerformanceMetrics) -> bool:
        """Check if architecture is promising based on surrogate prediction."""
        # Simple heuristics for promising architectures
        if surrogate_metrics.latency_ms > 20.0:  # Too slow
            return False
        
        if surrogate_metrics.accuracy < 0.80:  # Too inaccurate
            return False
        
        if surrogate_metrics.tops_per_watt < 20.0:  # Too inefficient
            return False
        
        return True
    
    def _evolve_population_with_diversity(
        self,
        evaluated_population: List[Tuple[Architecture, PerformanceMetrics]],
        arch_space: ArchitectureSpace,
        diversity_target: float
    ) -> List[Architecture]:
        """Evolve population while maintaining diversity."""
        # Sort by performance
        sorted_population = sorted(
            evaluated_population,
            key=lambda x: self._compute_fitness(x[1]),
            reverse=True
        )
        
        new_population = []
        
        # Elite selection (top performers)
        elite_count = max(1, len(sorted_population) // 4)
        for arch, _ in sorted_population[:elite_count]:
            new_population.append(arch)
        
        # Diversity-aware selection
        while len(new_population) < len(sorted_population):
            # Tournament selection with diversity consideration
            tournament_size = min(5, len(sorted_population))
            tournament = sorted_population[:tournament_size]
            
            # Select based on performance + diversity
            selected_arch = self._select_with_diversity(tournament, new_population, diversity_target)
            
            # Generate offspring
            if len(new_population) > 0 and hash(time.time()) % 2 == 0:
                # Crossover with random parent
                parent2 = new_population[hash(time.time()) % len(new_population)]
                try:
                    offspring = arch_space.crossover(selected_arch, parent2)
                except:
                    offspring = selected_arch
            else:
                # Mutation
                try:
                    offspring = arch_space.mutate(selected_arch)
                except:
                    offspring = selected_arch
            
            new_population.append(offspring)
        
        return new_population
    
    def _select_with_diversity(
        self,
        tournament: List[Tuple[Architecture, PerformanceMetrics]],
        current_population: List[Architecture],
        diversity_target: float
    ) -> Architecture:
        """Select architecture considering both performance and diversity."""
        best_arch = tournament[0][0]  # Default to best performer
        
        if len(current_population) == 0:
            return best_arch
        
        # Simple diversity measure: architecture depth and width
        for arch, metrics in tournament:
            diversity_score = self._compute_diversity_score(arch, current_population)
            performance_score = self._compute_fitness(metrics)
            
            combined_score = (1 - diversity_target) * performance_score + diversity_target * diversity_score
            
            if combined_score > self._compute_fitness(tournament[0][1]):
                best_arch = arch
        
        return best_arch
    
    def _compute_diversity_score(self, arch: Architecture, population: List[Architecture]) -> float:
        """Compute diversity score of architecture relative to population."""
        if not population:
            return 1.0
        
        # Simple diversity based on structural differences
        total_distance = 0.0
        
        for other_arch in population:
            distance = (
                abs(arch.depth - other_arch.depth) / max(arch.depth, other_arch.depth) +
                abs(arch.avg_width - other_arch.avg_width) / max(arch.avg_width, other_arch.avg_width) +
                abs(arch.total_params - other_arch.total_params) / max(arch.total_params, other_arch.total_params)
            ) / 3.0
            
            total_distance += distance
        
        return total_distance / len(population)
    
    def _compute_fitness(self, metrics: PerformanceMetrics) -> float:
        """Compute fitness score for architecture."""
        # Multi-objective fitness function
        latency_score = max(0, 1 - metrics.latency_ms / 20.0)  # Prefer < 20ms
        energy_score = max(0, 1 - metrics.energy_mj / 200.0)   # Prefer < 200mJ
        accuracy_score = metrics.accuracy
        efficiency_score = min(1.0, metrics.tops_per_watt / 100.0)  # Normalize to 100 TOPS/W
        
        return (0.3 * accuracy_score + 
                0.25 * latency_score + 
                0.25 * energy_score + 
                0.2 * efficiency_score)
    
    def _is_better_metrics(self, metrics1: PerformanceMetrics, metrics2: PerformanceMetrics) -> bool:
        """Compare two metrics to determine which is better."""
        return self._compute_fitness(metrics1) > self._compute_fitness(metrics2)
    
    def _adapt_parameters_based_on_progress(
        self, 
        evaluated_population: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> None:
        """Adapt optimization parameters based on current progress."""
        if len(evaluated_population) < 10:
            return
        
        # Analyze recent performance
        recent_scores = [self._compute_fitness(metrics) for _, metrics in evaluated_population]
        
        score_variance = sum((s - sum(recent_scores)/len(recent_scores))**2 for s in recent_scores) / len(recent_scores)
        
        # If variance is low, increase exploration
        if score_variance < 0.01:
            self.config.exploration_factor = min(0.8, self.config.exploration_factor * 1.1)
            self.logger.debug("Increasing exploration due to low variance")
        
        # If variance is high, increase exploitation  
        elif score_variance > 0.1:
            self.config.exploitation_factor = min(0.9, self.config.exploitation_factor * 1.1)
            self.logger.debug("Increasing exploitation due to high variance")


class MultiObjectiveOptimizer:
    """Multi-objective optimizer using NSGA-II inspired approach."""
    
    def __init__(
        self,
        architecture_space: ArchitectureSpace,
        predictor: TPUv6Predictor,
        objectives: List[str] = None
    ):
        self.architecture_space = architecture_space
        self.predictor = predictor
        self.objectives = objectives or ['accuracy', 'latency', 'energy', 'efficiency']
        self.logger = logging.getLogger(__name__)
    
    def optimize_pareto_front(
        self, 
        population_size: int = 100, 
        generations: int = 50
    ) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Find Pareto-optimal solutions."""
        self.logger.info(f"Starting multi-objective optimization for {len(self.objectives)} objectives")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            try:
                arch = self.architecture_space.sample_random()
                metrics = self.predictor.predict(arch)
                population.append((arch, metrics))
            except Exception as e:
                self.logger.debug(f"Failed to create initial architecture: {e}")
                continue
        
        for generation in range(generations):
            self.logger.debug(f"Multi-objective generation {generation + 1}/{generations}")
            
            # Non-dominated sorting
            fronts = self._non_dominated_sort(population)
            
            # Select next generation
            new_population = []
            
            for front in fronts:
                if len(new_population) + len(front) <= population_size:
                    new_population.extend(front)
                else:
                    # Crowding distance selection
                    remaining = population_size - len(new_population)
                    crowding_distances = self._calculate_crowding_distance(front)
                    
                    # Sort by crowding distance and select best
                    front_with_distance = list(zip(front, crowding_distances))
                    front_with_distance.sort(key=lambda x: x[1], reverse=True)
                    
                    for i in range(remaining):
                        new_population.append(front_with_distance[i][0])
                    break
            
            # Generate offspring
            offspring = self._generate_offspring(new_population)
            
            # Combine parent and offspring populations
            combined_population = new_population + offspring
            
            # Evaluate new architectures
            for i, (arch, metrics) in enumerate(combined_population):
                if metrics is None:  # New offspring
                    try:
                        metrics = self.predictor.predict(arch)
                        combined_population[i] = (arch, metrics)
                    except Exception as e:
                        self.logger.debug(f"Evaluation failed: {e}")
                        # Remove failed architecture
                        combined_population[i] = None
            
            # Filter out failed evaluations
            population = [item for item in combined_population if item is not None]
        
        # Return final Pareto front
        fronts = self._non_dominated_sort(population)
        return fronts[0] if fronts else []
    
    def _non_dominated_sort(
        self, 
        population: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> List[List[Tuple[Architecture, PerformanceMetrics]]]:
        """Perform non-dominated sorting."""
        fronts = [[]]
        
        domination_count = {}  # Number of solutions that dominate this solution
        dominated_solutions = {}  # Solutions that this solution dominates
        
        # Initialize
        for i, (arch1, metrics1) in enumerate(population):
            domination_count[i] = 0
            dominated_solutions[i] = []
            
            for j, (arch2, metrics2) in enumerate(population):
                if i != j:
                    if self._dominates(metrics1, metrics2):
                        dominated_solutions[i].append(j)
                    elif self._dominates(metrics2, metrics1):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(population[i])
        
        # Build subsequent fronts
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []
            
            for arch_idx in [i for i, (arch, _) in enumerate(population) if (arch, _) in fronts[front_index]]:
                for dominated_idx in dominated_solutions[arch_idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        next_front.append(population[dominated_idx])
            
            if next_front:
                fronts.append(next_front)
                front_index += 1
            else:
                break
        
        return fronts
    
    def _dominates(self, metrics1: PerformanceMetrics, metrics2: PerformanceMetrics) -> bool:
        """Check if metrics1 dominates metrics2 in Pareto sense."""
        objectives1 = self._get_objective_values(metrics1)
        objectives2 = self._get_objective_values(metrics2)
        
        at_least_one_better = False
        
        for obj in self.objectives:
            val1 = objectives1[obj]
            val2 = objectives2[obj]
            
            # For minimization objectives (latency, energy)
            if obj in ['latency', 'energy']:
                if val1 > val2:  # metrics1 is worse
                    return False
                elif val1 < val2:  # metrics1 is better
                    at_least_one_better = True
            
            # For maximization objectives (accuracy, efficiency)
            else:
                if val1 < val2:  # metrics1 is worse
                    return False
                elif val1 > val2:  # metrics1 is better
                    at_least_one_better = True
        
        return at_least_one_better
    
    def _get_objective_values(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """Extract objective values from metrics."""
        return {
            'accuracy': metrics.accuracy,
            'latency': metrics.latency_ms,
            'energy': metrics.energy_mj,
            'efficiency': metrics.tops_per_watt
        }
    
    def _calculate_crowding_distance(
        self, 
        front: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> List[float]:
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        distances = [0.0] * len(front)
        
        for obj in self.objectives:
            # Sort by objective
            sorted_indices = sorted(range(len(front)), key=lambda i: self._get_objective_values(front[i][1])[obj])
            
            # Boundary points have infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate distances for intermediate points
            obj_values = [self._get_objective_values(front[i][1])[obj] for i in sorted_indices]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    if distances[sorted_indices[i]] != float('inf'):
                        distance = (obj_values[i + 1] - obj_values[i - 1]) / obj_range
                        distances[sorted_indices[i]] += distance
        
        return distances
    
    def _generate_offspring(
        self, 
        population: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> List[Tuple[Architecture, Optional[PerformanceMetrics]]]:
        """Generate offspring through crossover and mutation."""
        offspring = []
        
        for _ in range(len(population)):
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            try:
                # Crossover
                if hash(time.time()) % 2 == 0:
                    child = self.architecture_space.crossover(parent1[0], parent2[0])
                else:
                    child = self.architecture_space.mutate(parent1[0])
                
                offspring.append((child, None))  # Metrics will be calculated later
                
            except Exception as e:
                self.logger.debug(f"Offspring generation failed: {e}")
                continue
        
        return offspring
    
    def _tournament_selection(
        self, 
        population: List[Tuple[Architecture, PerformanceMetrics]]
    ) -> Tuple[Architecture, PerformanceMetrics]:
        """Tournament selection for parent selection."""
        tournament_size = min(3, len(population))
        tournament = [population[hash(time.time() + i) % len(population)] for i in range(tournament_size)]
        
        # Select best (first front, then crowding distance)
        # For simplicity, just select the first one (could be improved)
        return tournament[0]
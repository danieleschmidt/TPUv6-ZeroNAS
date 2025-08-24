"""Advanced optimization techniques for Generation 3 TPUv6-ZeroNAS."""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import random

from .architecture import Architecture
from .metrics import PerformanceMetrics
from .predictor import TPUv6Predictor

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class OptimizationHint:
    """Hint for guiding optimization decisions."""
    architecture_pattern: str
    expected_improvement: float
    confidence: float
    optimization_type: str  # 'accuracy', 'latency', 'energy', 'tops_w'


class ParetoParetoFrontTracker:
    """Track and maintain Pareto-optimal solutions."""
    
    def __init__(self):
        self.pareto_front: List[Tuple[Architecture, PerformanceMetrics]] = []
        self.dominated_solutions: List[Tuple[Architecture, PerformanceMetrics]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_solution(self, architecture: Architecture, metrics: PerformanceMetrics) -> bool:
        """Add solution and return True if it's Pareto-optimal."""
        try:
            # Check if this solution dominates any existing solutions
            new_dominates = []
            dominated_by = []
            
            for i, (arch, existing_metrics) in enumerate(self.pareto_front):
                dominance = self._check_dominance(metrics, existing_metrics)
                if dominance == 1:  # New solution dominates existing
                    new_dominates.append(i)
                elif dominance == -1:  # Existing dominates new
                    dominated_by.append(i)
            
            # If not dominated by any existing solution
            if not dominated_by:
                # Remove dominated solutions from front
                for i in sorted(new_dominates, reverse=True):
                    dominated = self.pareto_front.pop(i)
                    self.dominated_solutions.append(dominated)
                
                # Add new solution to front
                self.pareto_front.append((architecture, metrics))
                return True
            else:
                # Add to dominated solutions
                self.dominated_solutions.append((architecture, metrics))
                return False
                
        except Exception as e:
            self.logger.warning(f"Pareto front update failed: {e}")
            return False
    
    def _check_dominance(self, metrics1: PerformanceMetrics, metrics2: PerformanceMetrics) -> int:
        """Check dominance relationship. Returns 1 if metrics1 dominates, -1 if dominated, 0 if non-dominated."""
        try:
            # Objectives: maximize accuracy, minimize latency, minimize energy, maximize TOPS/W
            better_count = 0
            worse_count = 0
            
            # Accuracy (maximize)
            if metrics1.accuracy > metrics2.accuracy:
                better_count += 1
            elif metrics1.accuracy < metrics2.accuracy:
                worse_count += 1
            
            # Latency (minimize)
            if metrics1.latency_ms < metrics2.latency_ms:
                better_count += 1
            elif metrics1.latency_ms > metrics2.latency_ms:
                worse_count += 1
            
            # Energy (minimize)  
            if metrics1.energy_mj < metrics2.energy_mj:
                better_count += 1
            elif metrics1.energy_mj > metrics2.energy_mj:
                worse_count += 1
            
            # TOPS/W (maximize)
            if metrics1.tops_per_watt > metrics2.tops_per_watt:
                better_count += 1
            elif metrics1.tops_per_watt < metrics2.tops_per_watt:
                worse_count += 1
            
            # Dominance logic
            if better_count > 0 and worse_count == 0:
                return 1  # metrics1 dominates
            elif worse_count > 0 and better_count == 0:
                return -1  # metrics2 dominates
            else:
                return 0  # Non-dominated
                
        except:
            return 0
    
    def get_pareto_front(self) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Get current Pareto front."""
        return self.pareto_front.copy()
    
    def get_best_for_objective(self, objective: str) -> Optional[Tuple[Architecture, PerformanceMetrics]]:
        """Get best solution for specific objective from Pareto front."""
        try:
            if not self.pareto_front:
                return None
            
            if objective == 'accuracy':
                return max(self.pareto_front, key=lambda x: x[1].accuracy)
            elif objective == 'latency':
                return min(self.pareto_front, key=lambda x: x[1].latency_ms)
            elif objective == 'energy':
                return min(self.pareto_front, key=lambda x: x[1].energy_mj)
            elif objective == 'tops_w':
                return max(self.pareto_front, key=lambda x: x[1].tops_per_watt)
            else:
                return self.pareto_front[0]
                
        except:
            return None


class IntelligentEvaluationScheduler:
    """Schedule evaluations intelligently based on predicted value."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.evaluation_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_evaluations: Dict[str, Future] = {}
        self.completion_callbacks: List[callable] = []
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.evaluation_times: List[float] = []
        self.success_rates: Dict[str, float] = {}
    
    def schedule_evaluation(
        self, 
        architecture: Architecture, 
        predictor: TPUv6Predictor,
        priority: float = 1.0,
        callback: Optional[callable] = None
    ) -> str:
        """Schedule architecture evaluation with priority."""
        try:
            eval_id = f"eval_{time.time()}_{random.randint(1000, 9999)}"
            
            # Create evaluation task
            future = self.executor.submit(self._evaluate_architecture, architecture, predictor, eval_id)
            
            if callback:
                future.add_done_callback(lambda f: callback(eval_id, f.result()))
            
            self.active_evaluations[eval_id] = future
            self.logger.debug(f"Scheduled evaluation {eval_id} with priority {priority}")
            
            return eval_id
            
        except Exception as e:
            self.logger.error(f"Failed to schedule evaluation: {e}")
            return ""
    
    def _evaluate_architecture(
        self, 
        architecture: Architecture, 
        predictor: TPUv6Predictor, 
        eval_id: str
    ) -> Tuple[Architecture, PerformanceMetrics]:
        """Perform architecture evaluation."""
        start_time = time.time()
        try:
            metrics = predictor.predict(architecture)
            
            evaluation_time = time.time() - start_time
            self.evaluation_times.append(evaluation_time)
            
            # Keep only recent evaluation times (last 100)
            if len(self.evaluation_times) > 100:
                self.evaluation_times = self.evaluation_times[-100:]
            
            return (architecture, metrics)
            
        except Exception as e:
            self.logger.warning(f"Evaluation {eval_id} failed: {e}")
            # Return fallback metrics
            fallback_metrics = PerformanceMetrics(
                latency_ms=10.0, energy_mj=100.0, accuracy=0.5, 
                tops_per_watt=30.0, memory_mb=1.0, flops=1000000
            )
            return (architecture, fallback_metrics)
        finally:
            # Clean up
            if eval_id in self.active_evaluations:
                del self.active_evaluations[eval_id]
    
    def get_average_evaluation_time(self) -> float:
        """Get average evaluation time."""
        try:
            if not self.evaluation_times:
                return 1.0
            return sum(self.evaluation_times) / len(self.evaluation_times)
        except:
            return 1.0
    
    def shutdown(self):
        """Shutdown the scheduler."""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("Evaluation scheduler shutdown complete")
        except Exception as e:
            self.logger.error(f"Scheduler shutdown error: {e}")


class AdvancedSearchOptimizer:
    """Advanced search optimization with Generation 3 techniques."""
    
    def __init__(self, predictor: TPUv6Predictor):
        self.predictor = predictor
        self.logger = logging.getLogger(__name__)
        
        # Advanced tracking
        self.pareto_tracker = ParetoParetoFrontTracker()
        self.scheduler = IntelligentEvaluationScheduler()
        self.optimization_hints: List[OptimizationHint] = []
        
        # Pattern recognition for architecture optimization
        self.successful_patterns: Dict[str, float] = {}
        self.failed_patterns: Dict[str, float] = {}
        
        # Adaptive parameters
        self.exploration_rate = 0.3
        self.exploitation_rate = 0.7
    
    def optimize_population_evaluation(
        self, 
        population: List[Architecture], 
        constraint_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[Architecture, PerformanceMetrics]]:
        """Optimize population evaluation using advanced scheduling."""
        try:
            self.logger.info(f"🚀 Advanced optimization: evaluating {len(population)} architectures")
            
            # Prioritize architectures based on predicted value
            prioritized_pop = self._prioritize_architectures(population)
            
            # Schedule evaluations
            evaluation_futures = []
            for priority, arch in prioritized_pop:
                future_id = self.scheduler.schedule_evaluation(arch, self.predictor, priority)
                if future_id:
                    evaluation_futures.append((arch, future_id))
            
            # Collect results
            results = []
            for arch, eval_id in evaluation_futures:
                try:
                    if eval_id in self.scheduler.active_evaluations:
                        future = self.scheduler.active_evaluations[eval_id]
                        arch_result, metrics = future.result(timeout=30.0)  # 30 second timeout
                        results.append((arch_result, metrics))
                        
                        # Update Pareto front
                        self.pareto_tracker.add_solution(arch_result, metrics)
                        
                        # Learn from result
                        self._learn_from_evaluation(arch_result, metrics)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get evaluation result: {e}")
                    continue
            
            self.logger.info(f"✅ Completed {len(results)} advanced evaluations")
            return results
            
        except Exception as e:
            self.logger.error(f"Advanced optimization failed: {e}")
            # Fallback to simple evaluation
            return [(arch, self.predictor.predict(arch)) for arch in population]
    
    def _prioritize_architectures(
        self, 
        population: List[Architecture]
    ) -> List[Tuple[float, Architecture]]:
        """Prioritize architectures based on predicted value and patterns."""
        try:
            prioritized = []
            
            for arch in population:
                # Base priority
                priority = 1.0
                
                # Pattern-based priority adjustment
                arch_pattern = self._get_architecture_pattern(arch)
                if arch_pattern in self.successful_patterns:
                    priority *= (1.0 + self.successful_patterns[arch_pattern])
                elif arch_pattern in self.failed_patterns:
                    priority *= (1.0 - self.failed_patterns[arch_pattern])
                
                # Diversity bonus (prefer architectures different from Pareto front)
                diversity_bonus = self._calculate_diversity_bonus(arch)
                priority *= (1.0 + diversity_bonus)
                
                # Size-based priority (prefer reasonable-sized architectures)
                if 10000 <= arch.total_params <= 10000000:  # Sweet spot
                    priority *= 1.2
                
                prioritized.append((priority, arch))
            
            # Sort by priority (higher first)
            prioritized.sort(key=lambda x: x[0], reverse=True)
            
            return prioritized
            
        except Exception as e:
            self.logger.warning(f"Architecture prioritization failed: {e}")
            return [(1.0, arch) for arch in population]
    
    def _get_architecture_pattern(self, architecture: Architecture) -> str:
        """Extract architectural pattern signature."""
        try:
            # Create a pattern signature based on key characteristics
            depth = len(architecture.layers)
            
            # Layer type distribution
            layer_types = [layer.layer_type.value for layer in architecture.layers]
            layer_type_counts = {}
            for lt in layer_types:
                layer_type_counts[lt] = layer_type_counts.get(lt, 0) + 1
            
            # Create pattern string
            pattern_parts = [f"d{depth}"]
            for lt, count in sorted(layer_type_counts.items()):
                pattern_parts.append(f"{lt[:3]}{count}")
            
            # Parameter size category
            if architecture.total_params < 100000:
                pattern_parts.append("small")
            elif architecture.total_params < 10000000:
                pattern_parts.append("medium")
            else:
                pattern_parts.append("large")
            
            return "_".join(pattern_parts)
            
        except Exception as e:
            return "unknown_pattern"
    
    def _calculate_diversity_bonus(self, architecture: Architecture) -> float:
        """Calculate diversity bonus compared to Pareto front."""
        try:
            if not self.pareto_tracker.pareto_front:
                return 0.0
            
            # Calculate average distance to Pareto front architectures
            distances = []
            
            for pareto_arch, _ in self.pareto_tracker.pareto_front:
                # Simple distance metric based on parameter count and layer count
                param_dist = abs(architecture.total_params - pareto_arch.total_params) / max(architecture.total_params, 1)
                layer_dist = abs(len(architecture.layers) - len(pareto_arch.layers)) / max(len(architecture.layers), 1)
                
                distances.append(param_dist + layer_dist)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                return min(0.3, avg_distance)  # Cap at 30% bonus
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def _learn_from_evaluation(self, architecture: Architecture, metrics: PerformanceMetrics):
        """Learn from evaluation results to improve future prioritization."""
        try:
            pattern = self._get_architecture_pattern(architecture)
            
            # Calculate success score based on constraint satisfaction and performance
            success_score = 0.0
            
            # Accuracy component
            if metrics.accuracy >= 0.85:  # Good accuracy
                success_score += 0.4
            
            # Efficiency component
            if metrics.tops_per_watt >= 30.0:  # Reasonable efficiency
                success_score += 0.3
            
            # Latency component
            if metrics.latency_ms <= 5.0:  # Good latency
                success_score += 0.3
            
            # Update pattern tracking
            if success_score >= 0.6:  # Success threshold
                self.successful_patterns[pattern] = self.successful_patterns.get(pattern, 0.0) + 0.1
                # Cap at 0.5 (50% bonus)
                self.successful_patterns[pattern] = min(0.5, self.successful_patterns[pattern])
            else:
                self.failed_patterns[pattern] = self.failed_patterns.get(pattern, 0.0) + 0.05
                # Cap at 0.3 (30% penalty)
                self.failed_patterns[pattern] = min(0.3, self.failed_patterns[pattern])
            
            # Decay old patterns (keep learning fresh)
            if len(self.successful_patterns) > 100:
                # Remove least successful patterns
                sorted_patterns = sorted(self.successful_patterns.items(), key=lambda x: x[1])
                patterns_to_remove = [p[0] for p in sorted_patterns[:10]]
                for p in patterns_to_remove:
                    del self.successful_patterns[p]
            
        except Exception as e:
            self.logger.debug(f"Learning from evaluation failed: {e}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        try:
            pareto_front = self.pareto_tracker.get_pareto_front()
            
            report = {
                'pareto_front_size': len(pareto_front),
                'successful_patterns': len(self.successful_patterns),
                'failed_patterns': len(self.failed_patterns),
                'avg_evaluation_time': self.scheduler.get_average_evaluation_time(),
                'top_patterns': sorted(self.successful_patterns.items(), 
                                     key=lambda x: x[1], reverse=True)[:5],
                'pareto_objectives': {}
            }
            
            # Pareto front analysis
            if pareto_front:
                accuracies = [m.accuracy for _, m in pareto_front]
                latencies = [m.latency_ms for _, m in pareto_front]
                tops_w = [m.tops_per_watt for _, m in pareto_front]
                
                report['pareto_objectives'] = {
                    'accuracy_range': [min(accuracies), max(accuracies)],
                    'latency_range': [min(latencies), max(latencies)],
                    'tops_w_range': [min(tops_w), max(tops_w)],
                    'best_accuracy': max(accuracies),
                    'best_latency': min(latencies),
                    'best_tops_w': max(tops_w)
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Optimization report generation failed: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.scheduler.shutdown()
            self.logger.info("Advanced search optimizer cleanup complete")
        except Exception as e:
            self.logger.error(f"Optimizer cleanup error: {e}")
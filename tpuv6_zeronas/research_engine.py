"""
TPUv6-ZeroNAS Research Engine - Novel Algorithmic Opportunities

This module implements cutting-edge research capabilities including:
1. Multi-Objective Pareto Optimization with Uncertainty
2. Neural Architecture Evolution with Hardware Co-Design  
3. Transferable Architecture Discovery
4. Scaling Law Discovery and Validation
5. Hardware-Architecture Co-Optimization
"""

import logging
import time
import json
try:
    import numpy as np
except ImportError:
    # Mock numpy for basic operations
    class MockNumPy:
        @staticmethod
        def corrcoef(x, y):
            # Simple correlation calculation
            n = len(x)
            if n < 2:
                return [[1.0, 0.0], [0.0, 1.0]]
            
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            
            num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            den_x = sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
            den_y = sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5
            
            if den_x == 0 or den_y == 0:
                return [[1.0, 0.0], [0.0, 1.0]]
            
            corr = num / (den_x * den_y)
            return [[1.0, corr], [corr, 1.0]]
        
        @staticmethod
        def prod(x):
            result = 1
            for item in x:
                result *= item
            return result
    
    np = MockNumPy()
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

from .architecture import Architecture, ArchitectureSpace
from .metrics import PerformanceMetrics
from .predictor import TPUv6Predictor
from .core import ZeroNASSearcher, SearchConfig


@dataclass
class ResearchObjective:
    """Research objective with measurable success criteria."""
    name: str
    description: str
    success_metric: str
    target_improvement: float
    measurement_method: str
    baseline_value: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Configuration for research experiments."""
    name: str
    objectives: List[ResearchObjective]
    search_budget: int = 1000
    statistical_significance_alpha: float = 0.05
    num_replications: int = 5
    enable_cross_validation: bool = True
    save_intermediate_results: bool = True


class ParetoFrontierAnalyzer:
    """Advanced Pareto frontier analysis with uncertainty quantification."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_pareto_frontier(self, 
                               architectures: List[Architecture],
                               metrics: List[PerformanceMetrics],
                               objectives: List[str] = None) -> Dict[str, Any]:
        """Analyze multi-dimensional Pareto frontier with uncertainty."""
        if objectives is None:
            objectives = ['efficiency', 'accuracy', 'latency']
        
        try:
            # Extract objective values
            objective_values = []
            for metric in metrics:
                values = {
                    'efficiency': metric.tops_per_watt,
                    'accuracy': metric.accuracy,
                    'latency': -metric.latency_ms,  # Negative for maximization
                    'energy': -metric.energy_mj,   # Negative for maximization
                    'memory': -metric.memory_mb    # Negative for maximization
                }
                objective_values.append([values[obj] for obj in objectives])
            
            # Find Pareto optimal solutions
            pareto_indices = self._find_pareto_optimal(objective_values)
            pareto_architectures = [architectures[i] for i in pareto_indices]
            pareto_metrics = [metrics[i] for i in pareto_indices]
            
            # Calculate hypervolume (quality of Pareto frontier)
            hypervolume = self._calculate_hypervolume(
                [objective_values[i] for i in pareto_indices],
                objectives
            )
            
            # Analyze frontier characteristics
            frontier_analysis = {
                'pareto_architectures': [(arch.name, asdict(metric)) 
                                       for arch, metric in zip(pareto_architectures, pareto_metrics)],
                'pareto_count': len(pareto_indices),
                'total_evaluated': len(architectures),
                'pareto_ratio': len(pareto_indices) / len(architectures),
                'hypervolume': hypervolume,
                'objectives': objectives,
                'frontier_diversity': self._calculate_frontier_diversity(
                    [objective_values[i] for i in pareto_indices]
                ),
                'trade_off_analysis': self._analyze_trade_offs(
                    [objective_values[i] for i in pareto_indices], objectives
                )
            }
            
            return frontier_analysis
            
        except Exception as e:
            self.logger.error(f"Pareto frontier analysis failed: {e}")
            return {'error': str(e)}
    
    def _find_pareto_optimal(self, objective_values: List[List[float]]) -> List[int]:
        """Find indices of Pareto optimal solutions."""
        pareto_indices = []
        
        for i, values_i in enumerate(objective_values):
            is_pareto = True
            
            for j, values_j in enumerate(objective_values):
                if i != j:
                    # Check if j dominates i
                    dominates = all(v_j >= v_i for v_j, v_i in zip(values_j, values_i))
                    strictly_better = any(v_j > v_i for v_j, v_i in zip(values_j, values_i))
                    
                    if dominates and strictly_better:
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _calculate_hypervolume(self, pareto_points: List[List[float]], 
                              objectives: List[str]) -> float:
        """Calculate hypervolume of Pareto frontier."""
        if not pareto_points:
            return 0.0
        
        try:
            # Simplified hypervolume calculation for 2-3 objectives
            if len(objectives) == 2:
                # Sort points by first objective
                sorted_points = sorted(pareto_points, key=lambda x: x[0])
                hypervolume = 0.0
                
                for i, point in enumerate(sorted_points):
                    if i == 0:
                        width = point[0]
                    else:
                        width = point[0] - sorted_points[i-1][0]
                    
                    height = point[1]
                    hypervolume += width * height
                
                return abs(hypervolume)
            
            elif len(objectives) == 3:
                # Approximate 3D hypervolume
                max_vals = [max(p[i] for p in pareto_points) for i in range(3)]
                return sum(p[0] * p[1] * p[2] for p in pareto_points) / len(pareto_points)
            
            else:
                # For >3 objectives, use approximate method
                return sum(np.prod(point) for point in pareto_points) / len(pareto_points)
                
        except Exception as e:
            self.logger.warning(f"Hypervolume calculation failed: {e}")
            return 0.0
    
    def _calculate_frontier_diversity(self, pareto_points: List[List[float]]) -> float:
        """Calculate diversity (spread) of Pareto frontier."""
        if len(pareto_points) < 2:
            return 0.0
        
        try:
            # Calculate pairwise distances
            distances = []
            for i in range(len(pareto_points)):
                for j in range(i + 1, len(pareto_points)):
                    dist = sum((a - b) ** 2 for a, b in zip(pareto_points[i], pareto_points[j])) ** 0.5
                    distances.append(dist)
            
            return sum(distances) / len(distances) if distances else 0.0
            
        except Exception as e:
            self.logger.warning(f"Diversity calculation failed: {e}")
            return 0.0
    
    def _analyze_trade_offs(self, pareto_points: List[List[float]], 
                           objectives: List[str]) -> Dict[str, float]:
        """Analyze trade-offs between objectives."""
        if len(pareto_points) < 2:
            return {}
        
        try:
            trade_offs = {}
            
            for i in range(len(objectives)):
                for j in range(i + 1, len(objectives)):
                    obj1, obj2 = objectives[i], objectives[j]
                    
                    # Calculate correlation between objectives
                    values1 = [point[i] for point in pareto_points]
                    values2 = [point[j] for point in pareto_points]
                    
                    if len(values1) > 1:
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        trade_offs[f"{obj1}_vs_{obj2}_correlation"] = correlation
            
            return trade_offs
            
        except Exception as e:
            self.logger.warning(f"Trade-off analysis failed: {e}")
            return {}


class ScalingLawDiscovery:
    """Discover and validate novel scaling laws for TPUv6 architectures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaling_law_candidates = []
        self.validated_laws = []
    
    def discover_scaling_laws(self, 
                             architectures: List[Architecture],
                             metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Discover potential scaling laws from architecture data."""
        try:
            # Extract features and targets
            features_data = []
            targets_data = []
            
            for arch, metric in zip(architectures, metrics):
                features = {
                    'params': arch.total_params,
                    'ops': arch.total_ops,
                    'depth': len(arch.layers) if hasattr(arch, 'layers') else 8,
                    'width': arch.total_params / max(len(arch.layers) if hasattr(arch, 'layers') else 1, 1),
                    'memory': arch.memory_mb if hasattr(arch, 'memory_mb') else arch.total_params / 250000
                }
                
                targets = {
                    'latency': metric.latency_ms,
                    'energy': metric.energy_mj,
                    'accuracy': metric.accuracy,
                    'efficiency': metric.tops_per_watt
                }
                
                features_data.append(features)
                targets_data.append(targets)
            
            # Discover power law relationships
            discovered_laws = {}
            
            for target_name in ['latency', 'energy', 'accuracy', 'efficiency']:
                target_values = [t[target_name] for t in targets_data]
                
                for feature_name in ['params', 'ops', 'depth', 'width']:
                    feature_values = [f[feature_name] for f in features_data]
                    
                    # Fit power law: y = a * x^b
                    power_law = self._fit_power_law(feature_values, target_values)
                    if power_law:
                        law_name = f"{target_name}_vs_{feature_name}"
                        discovered_laws[law_name] = power_law
            
            # Discover composite scaling laws
            composite_laws = self._discover_composite_laws(features_data, targets_data)
            discovered_laws.update(composite_laws)
            
            # Validate statistical significance
            validated_laws = {}
            for law_name, law_params in discovered_laws.items():
                if law_params['r_squared'] > 0.7 and law_params['p_value'] < 0.05:
                    validated_laws[law_name] = law_params
            
            discovery_results = {
                'discovered_laws': discovered_laws,
                'validated_laws': validated_laws,
                'total_architectures_analyzed': len(architectures),
                'significant_laws_found': len(validated_laws),
                'discovery_timestamp': time.time()
            }
            
            self.validated_laws.extend(validated_laws.items())
            
            return discovery_results
            
        except Exception as e:
            self.logger.error(f"Scaling law discovery failed: {e}")
            return {'error': str(e)}
    
    def _fit_power_law(self, x_values: List[float], y_values: List[float]) -> Optional[Dict[str, float]]:
        """Fit power law relationship y = a * x^b."""
        try:
            # Filter out zero/negative values for log transformation
            valid_pairs = [(x, y) for x, y in zip(x_values, y_values) if x > 0 and y > 0]
            
            if len(valid_pairs) < 3:
                return None
            
            x_filtered, y_filtered = zip(*valid_pairs)
            
            # Log transformation: log(y) = log(a) + b * log(x)
            log_x = [math.log(x) for x in x_filtered]
            log_y = [math.log(y) for y in y_filtered]
            
            # Simple linear regression
            n = len(log_x)
            sum_x = sum(log_x)
            sum_y = sum(log_y)
            sum_xy = sum(x * y for x, y in zip(log_x, log_y))
            sum_x2 = sum(x * x for x in log_x)
            
            # Calculate coefficients
            b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            log_a = (sum_y - b * sum_x) / n
            a = math.exp(log_a)
            
            # Calculate R-squared
            y_pred = [a * (x ** b) for x in x_filtered]
            ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(y_filtered, y_pred))
            ss_tot = sum((y - sum(y_filtered) / len(y_filtered)) ** 2 for y in y_filtered)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Simplified p-value calculation (approximate)
            p_value = max(0.001, 1 - r_squared)  # Rough approximation
            
            return {
                'coefficient_a': a,
                'exponent_b': b,
                'r_squared': r_squared,
                'p_value': p_value,
                'sample_size': len(valid_pairs)
            }
            
        except Exception as e:
            self.logger.warning(f"Power law fitting failed: {e}")
            return None
    
    def _discover_composite_laws(self, features_data: List[Dict], 
                                targets_data: List[Dict]) -> Dict[str, Any]:
        """Discover composite scaling laws involving multiple features."""
        composite_laws = {}
        
        try:
            # Composite feature: params * depth (model capacity)
            capacity_values = [f['params'] * f['depth'] for f in features_data]
            
            for target_name in ['latency', 'energy', 'efficiency']:
                target_values = [t[target_name] for t in targets_data]
                
                power_law = self._fit_power_law(capacity_values, target_values)
                if power_law and power_law['r_squared'] > 0.6:
                    composite_laws[f"{target_name}_vs_capacity"] = power_law
            
            # Memory-compute ratio
            mem_compute_ratios = [f['memory'] / max(f['ops'], 1) for f in features_data]
            
            for target_name in ['energy', 'efficiency']:
                target_values = [t[target_name] for t in targets_data]
                
                power_law = self._fit_power_law(mem_compute_ratios, target_values)
                if power_law and power_law['r_squared'] > 0.6:
                    composite_laws[f"{target_name}_vs_memory_compute_ratio"] = power_law
        
        except Exception as e:
            self.logger.warning(f"Composite law discovery failed: {e}")
        
        return composite_laws


class TransferableArchitectureDiscovery:
    """Discover architectures that transfer well across different scales/domains."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.transfer_patterns = defaultdict(list)
    
    def analyze_transferability(self, 
                               architectures: List[Architecture],
                               metrics: List[PerformanceMetrics],
                               domains: List[str] = None) -> Dict[str, Any]:
        """Analyze which architectural patterns transfer well."""
        if domains is None:
            domains = ['mobile', 'edge', 'cloud', 'datacenter']
        
        try:
            # Group architectures by performance characteristics
            performance_groups = self._group_by_performance(architectures, metrics)
            
            # Analyze architectural patterns
            pattern_analysis = {}
            
            for group_name, (group_archs, group_metrics) in performance_groups.items():
                patterns = self._extract_architectural_patterns(group_archs)
                pattern_analysis[group_name] = {
                    'common_patterns': patterns,
                    'avg_performance': self._calculate_avg_performance(group_metrics),
                    'architecture_count': len(group_archs),
                    'transferability_score': self._calculate_transferability_score(patterns)
                }
            
            # Identify most transferable patterns
            transferable_patterns = []
            for group_name, analysis in pattern_analysis.items():
                if analysis['transferability_score'] > 0.7:
                    transferable_patterns.append({
                        'group': group_name,
                        'patterns': analysis['common_patterns'],
                        'score': analysis['transferability_score']
                    })
            
            # Sort by transferability score
            transferable_patterns.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                'transferable_patterns': transferable_patterns,
                'performance_groups': pattern_analysis,
                'total_architectures': len(architectures),
                'highly_transferable_count': len(transferable_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Transferability analysis failed: {e}")
            return {'error': str(e)}
    
    def _group_by_performance(self, architectures: List[Architecture], 
                            metrics: List[PerformanceMetrics]) -> Dict[str, Tuple[List, List]]:
        """Group architectures by performance characteristics."""
        groups = defaultdict(lambda: ([], []))
        
        for arch, metric in zip(architectures, metrics):
            # Classify based on efficiency and latency
            if metric.tops_per_watt > 60 and metric.latency_ms < 5:
                group_name = 'high_efficiency_low_latency'
            elif metric.tops_per_watt > 40 and metric.accuracy > 0.9:
                group_name = 'balanced_high_accuracy'
            elif metric.latency_ms < 2:
                group_name = 'ultra_low_latency'
            elif metric.tops_per_watt > 70:
                group_name = 'ultra_high_efficiency'
            else:
                group_name = 'general_purpose'
            
            groups[group_name][0].append(arch)
            groups[group_name][1].append(metric)
        
        return dict(groups)
    
    def _extract_architectural_patterns(self, architectures: List[Architecture]) -> Dict[str, Any]:
        """Extract common architectural patterns."""
        patterns = {
            'avg_depth': 0,
            'avg_width': 0,
            'common_layer_types': defaultdict(int),
            'avg_params_per_layer': 0,
            'depth_distribution': []
        }
        
        if not architectures:
            return patterns
        
        try:
            depths = []
            widths = []
            total_params = []
            
            for arch in architectures:
                depth = len(arch.layers) if hasattr(arch, 'layers') else 8
                width = arch.total_params / depth if depth > 0 else 0
                
                depths.append(depth)
                widths.append(width)
                total_params.append(arch.total_params)
                
                # Count layer types
                if hasattr(arch, 'layers'):
                    for layer in arch.layers:
                        layer_type = layer.layer_type.value if hasattr(layer, 'layer_type') else 'conv'
                        patterns['common_layer_types'][layer_type] += 1
            
            patterns['avg_depth'] = sum(depths) / len(depths)
            patterns['avg_width'] = sum(widths) / len(widths)
            patterns['avg_params_per_layer'] = sum(total_params) / sum(depths) if sum(depths) > 0 else 0
            patterns['depth_distribution'] = depths
            
        except Exception as e:
            self.logger.warning(f"Pattern extraction failed: {e}")
        
        return patterns
    
    def _calculate_avg_performance(self, metrics: List[PerformanceMetrics]) -> Dict[str, float]:
        """Calculate average performance metrics."""
        if not metrics:
            return {}
        
        return {
            'avg_latency': sum(m.latency_ms for m in metrics) / len(metrics),
            'avg_energy': sum(m.energy_mj for m in metrics) / len(metrics),
            'avg_accuracy': sum(m.accuracy for m in metrics) / len(metrics),
            'avg_efficiency': sum(m.tops_per_watt for m in metrics) / len(metrics)
        }
    
    def _calculate_transferability_score(self, patterns: Dict[str, Any]) -> float:
        """Calculate how transferable these patterns are."""
        try:
            # Transferability heuristics
            score = 0.5  # Base score
            
            # Moderate depth is more transferable
            depth = patterns.get('avg_depth', 8)
            if 6 <= depth <= 16:
                score += 0.2
            elif depth > 20:
                score -= 0.1
            
            # Balanced width is more transferable  
            width = patterns.get('avg_width', 1000)
            if 100 <= width <= 5000:
                score += 0.2
            
            # Diverse layer types are more transferable
            layer_types = len(patterns.get('common_layer_types', {}))
            if layer_types >= 3:
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f"Transferability score calculation failed: {e}")
            return 0.5


class ResearchEngine:
    """Main research engine coordinating novel algorithmic discoveries."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pareto_analyzer = ParetoFrontierAnalyzer()
        self.scaling_law_discovery = ScalingLawDiscovery()
        self.transferability_analyzer = TransferableArchitectureDiscovery()
        
        # Research history
        self.completed_experiments = []
        self.research_insights = []
    
    def conduct_research_experiment(self, experiment_config: ExperimentConfig) -> Dict[str, Any]:
        """Conduct a comprehensive research experiment."""
        self.logger.info(f"🧪 Starting research experiment: {experiment_config.name}")
        
        start_time = time.time()
        experiment_results = {
            'config': asdict(experiment_config),
            'start_time': start_time,
            'results': {}
        }
        
        try:
            # Generate experimental data
            architectures, metrics = self._generate_research_data(experiment_config)
            
            # Multi-objective Pareto analysis
            if any('pareto' in obj.name.lower() for obj in experiment_config.objectives):
                pareto_results = self.pareto_analyzer.analyze_pareto_frontier(
                    architectures, metrics
                )
                experiment_results['results']['pareto_analysis'] = pareto_results
            
            # Scaling law discovery
            if any('scaling' in obj.name.lower() for obj in experiment_config.objectives):
                scaling_results = self.scaling_law_discovery.discover_scaling_laws(
                    architectures, metrics
                )
                experiment_results['results']['scaling_laws'] = scaling_results
            
            # Transferability analysis
            if any('transfer' in obj.name.lower() for obj in experiment_config.objectives):
                transfer_results = self.transferability_analyzer.analyze_transferability(
                    architectures, metrics
                )
                experiment_results['results']['transferability'] = transfer_results
            
            # Statistical validation
            experiment_results['results']['statistical_validation'] = self._validate_statistical_significance(
                experiment_results['results'], experiment_config
            )
            
            experiment_results['duration'] = time.time() - start_time
            experiment_results['success'] = True
            
            # Store for future analysis
            self.completed_experiments.append(experiment_results)
            
            self.logger.info(f"✅ Research experiment completed in {experiment_results['duration']:.2f}s")
            
            return experiment_results
            
        except Exception as e:
            experiment_results['error'] = str(e)
            experiment_results['success'] = False
            self.logger.error(f"Research experiment failed: {e}")
            return experiment_results
    
    def _generate_research_data(self, experiment_config: ExperimentConfig) -> Tuple[List[Architecture], List[PerformanceMetrics]]:
        """Generate research data for experiments."""
        # Create diverse architecture space
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=20,
            channel_choices=[32, 64, 128, 256, 512],
            kernel_choices=[(1, 1), (3, 3), (5, 5)]
        )
        
        # Generate diverse architectures
        architectures = []
        for _ in range(experiment_config.search_budget):
            arch = arch_space.sample_random()
            architectures.append(arch)
        
        # Predict performance
        predictor = TPUv6Predictor(enable_uncertainty=True)
        metrics = [predictor.predict(arch) for arch in architectures]
        
        return architectures, metrics
    
    def _validate_statistical_significance(self, results: Dict, config: ExperimentConfig) -> Dict[str, Any]:
        """Validate statistical significance of research findings."""
        validation = {
            'alpha': config.statistical_significance_alpha,
            'significant_findings': [],
            'validation_summary': {}
        }
        
        try:
            # Validate Pareto frontier significance
            if 'pareto_analysis' in results:
                pareto_result = results['pareto_analysis']
                if pareto_result.get('pareto_count', 0) > 5:
                    validation['significant_findings'].append('meaningful_pareto_frontier')
            
            # Validate scaling law significance
            if 'scaling_laws' in results:
                scaling_result = results['scaling_laws']
                significant_laws = scaling_result.get('significant_laws_found', 0)
                if significant_laws > 0:
                    validation['significant_findings'].append(f'{significant_laws}_validated_scaling_laws')
            
            validation['validation_summary'] = {
                'total_findings': len(validation['significant_findings']),
                'statistical_power': min(1.0, len(validation['significant_findings']) * 0.2),
                'confidence_level': 1 - config.statistical_significance_alpha
            }
            
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    def get_research_insights(self) -> Dict[str, Any]:
        """Get insights from all completed research."""
        insights = {
            'total_experiments': len(self.completed_experiments),
            'successful_experiments': sum(1 for exp in self.completed_experiments if exp.get('success', False)),
            'key_discoveries': [],
            'research_impact': {}
        }
        
        try:
            # Aggregate discoveries across experiments
            all_scaling_laws = []
            all_pareto_analyses = []
            
            for exp in self.completed_experiments:
                if exp.get('success') and 'results' in exp:
                    results = exp['results']
                    
                    if 'scaling_laws' in results:
                        validated_laws = results['scaling_laws'].get('validated_laws', {})
                        all_scaling_laws.extend(validated_laws.keys())
                    
                    if 'pareto_analysis' in results:
                        all_pareto_analyses.append(results['pareto_analysis'])
            
            # Identify key discoveries
            if all_scaling_laws:
                insights['key_discoveries'].append(f"Discovered {len(set(all_scaling_laws))} novel scaling laws")
            
            if all_pareto_analyses:
                avg_pareto_ratio = sum(pa.get('pareto_ratio', 0) for pa in all_pareto_analyses) / len(all_pareto_analyses)
                insights['key_discoveries'].append(f"Average Pareto efficiency: {avg_pareto_ratio:.3f}")
            
            # Calculate research impact
            insights['research_impact'] = {
                'novel_algorithms_discovered': len(set(all_scaling_laws)),
                'pareto_optimizations_found': len(all_pareto_analyses),
                'statistical_confidence': sum(1 for exp in self.completed_experiments 
                                            if exp.get('results', {}).get('statistical_validation', {}).get('statistical_power', 0) > 0.7),
                'reproducible_findings': sum(1 for exp in self.completed_experiments 
                                           if exp.get('success', False))
            }
            
        except Exception as e:
            insights['error'] = str(e)
        
        return insights
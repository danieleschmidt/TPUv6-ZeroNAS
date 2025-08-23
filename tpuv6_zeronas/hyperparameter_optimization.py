"""Intelligent hyperparameter optimization for Generation 3."""

import logging
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import random


class HyperparameterType(Enum):
    """Types of hyperparameters."""
    CONTINUOUS = "continuous"
    INTEGER = "integer" 
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    name: str
    param_type: HyperparameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    default_value: Any = None


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration with performance tracking."""
    params: Dict[str, Any]
    performance_score: float = 0.0
    evaluations: int = 0
    timestamp: float = 0.0


class BayesianHyperparameterOptimizer:
    """Simplified Bayesian optimization for hyperparameters."""
    
    def __init__(self, search_spaces: List[HyperparameterSpace]):
        self.search_spaces = {space.name: space for space in search_spaces}
        self.logger = logging.getLogger(__name__)
        
        # Tracking
        self.evaluated_configs: List[HyperparameterConfig] = []
        self.best_config: Optional[HyperparameterConfig] = None
        
        # Simple acquisition function parameters
        self.exploration_weight = 0.1
        self.exploitation_weight = 0.9
    
    def suggest_next_config(self) -> Dict[str, Any]:
        """Suggest next hyperparameter configuration to evaluate."""
        try:
            # If we don't have many evaluations, use random exploration
            if len(self.evaluated_configs) < 10:
                return self._random_config()
            
            # Use simple acquisition function
            best_config = None
            best_score = float('-inf')
            
            # Generate candidates and score them
            for _ in range(50):  # Sample 50 candidates
                candidate = self._random_config()
                score = self._acquisition_function(candidate)
                
                if score > best_score:
                    best_score = score
                    best_config = candidate
            
            return best_config or self._random_config()
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter suggestion failed: {e}")
            return self._random_config()
    
    def update_with_result(self, config: Dict[str, Any], performance_score: float):
        """Update optimizer with evaluation result."""
        try:
            hp_config = HyperparameterConfig(
                params=config.copy(),
                performance_score=performance_score,
                evaluations=1,
                timestamp=time.time()
            )
            
            self.evaluated_configs.append(hp_config)
            
            # Update best config
            if self.best_config is None or performance_score > self.best_config.performance_score:
                self.best_config = hp_config
                self.logger.info(f"🎯 New best hyperparameter config: score={performance_score:.4f}")
            
            # Keep only recent configs (last 100)
            if len(self.evaluated_configs) > 100:
                self.evaluated_configs = self.evaluated_configs[-100:]
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter result update failed: {e}")
    
    def _random_config(self) -> Dict[str, Any]:
        """Generate random configuration."""
        config = {}
        
        for name, space in self.search_spaces.items():
            try:
                if space.param_type == HyperparameterType.CONTINUOUS:
                    config[name] = random.uniform(space.min_value, space.max_value)
                elif space.param_type == HyperparameterType.INTEGER:
                    config[name] = random.randint(int(space.min_value), int(space.max_value))
                elif space.param_type == HyperparameterType.CATEGORICAL:
                    config[name] = random.choice(space.choices)
                elif space.param_type == HyperparameterType.BOOLEAN:
                    config[name] = random.choice([True, False])
                else:
                    config[name] = space.default_value
            except:
                config[name] = space.default_value
        
        return config
    
    def _acquisition_function(self, candidate: Dict[str, Any]) -> float:
        """Simple acquisition function combining exploration and exploitation."""
        try:
            # Exploitation: similarity to best performing configs
            exploitation_score = 0.0
            if self.best_config:
                similarity = self._config_similarity(candidate, self.best_config.params)
                exploitation_score = similarity * self.best_config.performance_score
            
            # Exploration: novelty relative to evaluated configs
            exploration_score = 1.0  # Base exploration
            if self.evaluated_configs:
                min_distance = float('inf')
                for config in self.evaluated_configs[-20:]:  # Compare to recent configs
                    distance = self._config_distance(candidate, config.params)
                    min_distance = min(min_distance, distance)
                exploration_score = min_distance
            
            # Combine scores
            total_score = (self.exploitation_weight * exploitation_score + 
                          self.exploration_weight * exploration_score)
            
            return total_score
            
        except Exception as e:
            return random.random()  # Fallback to random
    
    def _config_similarity(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> float:
        """Calculate similarity between two configurations (0-1)."""
        try:
            similarities = []
            
            for name in config1.keys():
                if name in config2:
                    space = self.search_spaces.get(name)
                    if not space:
                        continue
                    
                    if space.param_type == HyperparameterType.CONTINUOUS:
                        # Normalized absolute difference
                        range_val = space.max_value - space.min_value
                        if range_val > 0:
                            diff = abs(config1[name] - config2[name]) / range_val
                            similarities.append(1.0 - diff)
                    elif space.param_type == HyperparameterType.INTEGER:
                        range_val = space.max_value - space.min_value
                        if range_val > 0:
                            diff = abs(config1[name] - config2[name]) / range_val
                            similarities.append(1.0 - diff)
                    elif space.param_type in [HyperparameterType.CATEGORICAL, HyperparameterType.BOOLEAN]:
                        similarities.append(1.0 if config1[name] == config2[name] else 0.0)
            
            return sum(similarities) / max(len(similarities), 1)
            
        except:
            return 0.5
    
    def _config_distance(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> float:
        """Calculate distance between two configurations."""
        return 1.0 - self._config_similarity(config1, config2)
    
    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get the best configuration found so far."""
        if self.best_config:
            return self.best_config.params.copy()
        return None
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return [
            {
                'params': config.params,
                'score': config.performance_score,
                'timestamp': config.timestamp
            }
            for config in self.evaluated_configs
        ]


class AdaptiveHyperparameterManager:
    """Manages hyperparameter optimization during search."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define search space for ZeroNAS hyperparameters
        self.search_spaces = [
            HyperparameterSpace(
                name="mutation_rate",
                param_type=HyperparameterType.CONTINUOUS,
                min_value=0.01,
                max_value=0.5,
                default_value=0.1
            ),
            HyperparameterSpace(
                name="crossover_rate", 
                param_type=HyperparameterType.CONTINUOUS,
                min_value=0.3,
                max_value=0.9,
                default_value=0.7
            ),
            HyperparameterSpace(
                name="population_size",
                param_type=HyperparameterType.INTEGER,
                min_value=10,
                max_value=100,
                default_value=50
            ),
            HyperparameterSpace(
                name="early_stop_threshold",
                param_type=HyperparameterType.CONTINUOUS,
                min_value=1e-8,
                max_value=1e-3,
                default_value=1e-6
            ),
            HyperparameterSpace(
                name="target_tops_w",
                param_type=HyperparameterType.CONTINUOUS,
                min_value=30.0,
                max_value=100.0,
                default_value=75.0
            )
        ]
        
        self.optimizer = BayesianHyperparameterOptimizer(self.search_spaces)
        self.adaptation_interval = 50  # Adapt every 50 evaluations
        self.last_adaptation = 0
    
    def should_adapt_hyperparameters(self, total_evaluations: int) -> bool:
        """Check if hyperparameters should be adapted."""
        return (total_evaluations - self.last_adaptation) >= self.adaptation_interval
    
    def adapt_hyperparameters(self, search_config, search_performance: float) -> Dict[str, Any]:
        """Adapt hyperparameters based on search performance."""
        try:
            # Current configuration
            current_config = {
                "mutation_rate": search_config.mutation_rate,
                "crossover_rate": search_config.crossover_rate,
                "population_size": search_config.population_size,
                "early_stop_threshold": search_config.early_stop_threshold,
                "target_tops_w": search_config.target_tops_w
            }
            
            # Update optimizer with current performance
            self.optimizer.update_with_result(current_config, search_performance)
            
            # Get new suggestion
            new_config = self.optimizer.suggest_next_config()
            
            # Apply new configuration
            search_config.mutation_rate = new_config["mutation_rate"]
            search_config.crossover_rate = new_config["crossover_rate"]
            search_config.population_size = int(new_config["population_size"])
            search_config.early_stop_threshold = new_config["early_stop_threshold"]
            search_config.target_tops_w = new_config["target_tops_w"]
            
            self.last_adaptation = search_performance  # Use performance as proxy for evaluations
            
            self.logger.info(f"🔧 Adapted hyperparameters: {new_config}")
            
            return new_config
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter adaptation failed: {e}")
            return {}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get hyperparameter optimization summary."""
        try:
            best_config = self.optimizer.get_best_config()
            history = self.optimizer.get_optimization_history()
            
            return {
                "best_config": best_config,
                "num_evaluations": len(history),
                "best_score": max([h["score"] for h in history]) if history else 0.0,
                "score_improvement": (max([h["score"] for h in history]) - 
                                     min([h["score"] for h in history])) if len(history) > 1 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Optimization summary generation failed: {e}")
            return {"error": str(e)}
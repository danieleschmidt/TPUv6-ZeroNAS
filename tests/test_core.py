"""Tests for core ZeroNAS search functionality."""

import pytest
import numpy as np

from tpuv6_zeronas.core import ZeroNASSearcher, SearchConfig
from tpuv6_zeronas.architecture import ArchitectureSpace, Architecture, Layer, LayerType
from tpuv6_zeronas.predictor import TPUv6Predictor
from tpuv6_zeronas.metrics import PerformanceMetrics


class TestSearchConfig:
    """Test SearchConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SearchConfig()
        
        assert config.max_iterations == 1000
        assert config.population_size == 50
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.7
        assert config.target_tops_w == 75.0
        assert config.max_latency_ms == 10.0
        assert config.min_accuracy == 0.95
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SearchConfig(
            max_iterations=500,
            population_size=30,
            target_tops_w=80.0
        )
        
        assert config.max_iterations == 500
        assert config.population_size == 30
        assert config.target_tops_w == 80.0


class TestZeroNASSearcher:
    """Test ZeroNAS searcher functionality."""
    
    def create_test_components(self):
        """Create test components for searcher."""
        arch_space = ArchitectureSpace(
            input_shape=(224, 224, 3),
            num_classes=1000,
            max_depth=8
        )
        
        predictor = TPUv6Predictor()
        
        config = SearchConfig(
            max_iterations=20,
            population_size=10,
            target_tops_w=75.0,
            max_latency_ms=15.0,
            min_accuracy=0.92
        )
        
        return arch_space, predictor, config
    
    def test_searcher_creation(self):
        """Test searcher creation."""
        arch_space, predictor, config = self.create_test_components()
        
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        assert searcher.architecture_space == arch_space
        # Predictor might be wrapped in caching, so check the original
        if hasattr(searcher.predictor, 'predictor'):
            assert searcher.predictor.predictor == predictor
        else:
            assert searcher.predictor == predictor
        assert searcher.config == config
        assert searcher.best_architecture is None
        assert searcher.best_metrics is None
        assert len(searcher.search_history) == 0
        
    def test_population_initialization(self):
        """Test population initialization."""
        arch_space, predictor, config = self.create_test_components()
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        population = searcher._initialize_population()
        
        assert len(population) == config.population_size
        assert all(isinstance(arch, Architecture) for arch in population)
        
    def test_population_evaluation(self):
        """Test population evaluation."""
        arch_space, predictor, config = self.create_test_components()
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        population = searcher._initialize_population()
        evaluated = searcher._evaluate_population(population)
        
        assert len(evaluated) == len(population)
        assert all(isinstance(item, tuple) for item in evaluated)
        assert all(isinstance(item[0], Architecture) for item in evaluated)
        assert all(isinstance(item[1], PerformanceMetrics) for item in evaluated)
        
    def test_score_computation(self):
        """Test multi-objective score computation."""
        arch_space, predictor, config = self.create_test_components()
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        metrics = PerformanceMetrics(
            latency_ms=5.0,
            energy_mj=50.0,
            accuracy=0.96,
            tops_per_watt=70.0,
            memory_mb=100.0,
            flops=1000000
        )
        
        score = searcher._compute_score(metrics)
        
        assert isinstance(score, float)
        assert score > 0
        
    def test_constraint_checking(self):
        """Test constraint checking for architectures."""
        arch_space, predictor, config = self.create_test_components()
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        valid_metrics = PerformanceMetrics(
            latency_ms=8.0,
            energy_mj=50.0,
            accuracy=0.96,
            tops_per_watt=70.0,
            memory_mb=100.0,
            flops=1000000
        )
        
        invalid_latency = PerformanceMetrics(
            latency_ms=20.0,  # Exceeds max_latency_ms=15.0
            energy_mj=50.0,
            accuracy=0.96,
            tops_per_watt=70.0,
            memory_mb=100.0,
            flops=1000000
        )
        
        invalid_accuracy = PerformanceMetrics(
            latency_ms=8.0,
            energy_mj=50.0,
            accuracy=0.90,  
            tops_per_watt=70.0,
            memory_mb=100.0,
            flops=1000000
        )
        
        assert searcher._is_better_architecture(valid_metrics)
        assert not searcher._is_better_architecture(invalid_latency)
        assert not searcher._is_better_architecture(invalid_accuracy)
        
    def test_parent_selection(self):
        """Test parent selection for genetic algorithm."""
        arch_space, predictor, config = self.create_test_components()
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        population = []
        for i in range(5):
            arch = arch_space.sample_random()
            metrics = PerformanceMetrics(
                latency_ms=5.0 + i,
                energy_mj=50.0,
                accuracy=0.95,
                tops_per_watt=70.0,
                memory_mb=100.0,
                flops=1000000
            )
            population.append((arch, metrics))
        
        parents = searcher._select_parents(population, n=2)
        
        assert len(parents) == 2
        assert all(isinstance(parent, Architecture) for parent in parents)
        
    def test_short_search(self):
        """Test short search run."""
        arch_space, predictor, config = self.create_test_components()
        config.max_iterations = 5
        config.population_size = 8
        
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        best_arch, best_metrics = searcher.search()
        
        assert isinstance(best_arch, Architecture)
        assert isinstance(best_metrics, PerformanceMetrics)
        assert len(searcher.search_history) > 0
        assert searcher.best_architecture is not None
        assert searcher.best_metrics is not None
        
    def test_early_stopping_check(self):
        """Test early stopping logic."""
        arch_space, predictor, config = self.create_test_components()
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        for i in range(100):
            arch = arch_space.sample_random()
            metrics = PerformanceMetrics(
                latency_ms=5.0,
                energy_mj=50.0,
                accuracy=0.95,
                tops_per_watt=70.0,
                memory_mb=100.0,
                flops=1000000
            )
            searcher.search_history.append((arch, metrics))
        
        should_stop = searcher._should_early_stop()
        assert isinstance(should_stop, bool)
        
    def test_evolution_operations(self):
        """Test genetic algorithm evolution operations."""
        arch_space, predictor, config = self.create_test_components()
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        evaluated_pop = []
        for i in range(10):
            arch = arch_space.sample_random()
            metrics = PerformanceMetrics(
                latency_ms=5.0 + np.random.random(),
                energy_mj=50.0,
                accuracy=0.95 + np.random.random() * 0.04,
                tops_per_watt=70.0 + np.random.random() * 10,
                memory_mb=100.0,
                flops=1000000
            )
            evaluated_pop.append((arch, metrics))
        
        new_population = searcher._evolve_population(evaluated_pop)
        
        assert len(new_population) == config.population_size
        assert all(isinstance(arch, Architecture) for arch in new_population)
        
    def test_crossover_operation(self):
        """Test crossover operation."""
        arch_space, predictor, config = self.create_test_components()
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        parent1 = arch_space.sample_random()
        parent2 = arch_space.sample_random()
        
        child = searcher._crossover(parent1, parent2)
        
        assert isinstance(child, Architecture)
        assert len(child.layers) > 0
        
    def test_mutation_operation(self):
        """Test mutation operation."""
        arch_space, predictor, config = self.create_test_components()
        searcher = ZeroNASSearcher(arch_space, predictor, config)
        
        original = arch_space.sample_random()
        mutated = searcher._mutate(original)
        
        assert isinstance(mutated, Architecture)
        assert len(mutated.layers) > 0


if __name__ == '__main__':
    pytest.main([__file__])
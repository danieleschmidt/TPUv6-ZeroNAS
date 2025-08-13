"""Neuromorphic Neural Architecture Search: Brain-Inspired Computing Paradigms."""

import logging
import time
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from pathlib import Path

from .architecture import Architecture, ArchitectureSpace, Layer
from .metrics import PerformanceMetrics
from .predictor import TPUv6Predictor
from .core import SearchConfig


@dataclass
class SpikingNeuron:
    """Spiking neuron model for neuromorphic computation."""
    membrane_potential: float = 0.0
    threshold: float = 1.0
    leak_rate: float = 0.1
    refractory_period: int = 0
    spike_count: int = 0
    last_spike_time: float = 0.0
    synaptic_weights: List[float] = field(default_factory=list)
    
    def update(self, inputs: List[float], dt: float = 1.0) -> bool:
        """Update neuron state and return True if spike occurs."""
        if self.refractory_period > 0:
            self.refractory_period -= 1
            return False
        
        # Integrate inputs
        input_current = sum(w * i for w, i in zip(self.synaptic_weights, inputs))
        
        # Update membrane potential with leak
        self.membrane_potential += input_current * dt
        self.membrane_potential *= (1.0 - self.leak_rate * dt)
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0
            self.refractory_period = 2
            self.spike_count += 1
            self.last_spike_time = time.time()
            return True
        
        return False
    
    def reset(self):
        """Reset neuron state."""
        self.membrane_potential = 0.0
        self.refractory_period = 0
        self.spike_count = 0


@dataclass
class SynapticPlasticity:
    """Spike-timing dependent plasticity for learning."""
    pre_spike_times: List[float] = field(default_factory=list)
    post_spike_times: List[float] = field(default_factory=list)
    weight: float = 1.0
    learning_rate: float = 0.01
    tau_pre: float = 20.0  # Pre-synaptic time constant
    tau_post: float = 20.0  # Post-synaptic time constant
    
    def update_weight(self, pre_spike: bool, post_spike: bool, current_time: float):
        """Update synaptic weight based on STDP rule."""
        if pre_spike:
            self.pre_spike_times.append(current_time)
            # LTD: Post-synaptic spikes before pre-synaptic
            for post_time in self.post_spike_times:
                if current_time - post_time > 0:
                    delta_t = current_time - post_time
                    weight_change = -self.learning_rate * math.exp(-delta_t / self.tau_pre)
                    self.weight = max(0.0, min(2.0, self.weight + weight_change))
        
        if post_spike:
            self.post_spike_times.append(current_time)
            # LTP: Pre-synaptic spikes before post-synaptic
            for pre_time in self.pre_spike_times:
                if current_time - pre_time > 0:
                    delta_t = current_time - pre_time
                    weight_change = self.learning_rate * math.exp(-delta_t / self.tau_post)
                    self.weight = max(0.0, min(2.0, self.weight + weight_change))
        
        # Cleanup old spike times
        cutoff_time = current_time - 5 * max(self.tau_pre, self.tau_post)
        self.pre_spike_times = [t for t in self.pre_spike_times if t > cutoff_time]
        self.post_spike_times = [t for t in self.post_spike_times if t > cutoff_time]


@dataclass
class NeuromorphicArchitecture:
    """Neuromorphic architecture representation."""
    spiking_layers: List[Dict[str, Any]] = field(default_factory=list)
    synaptic_connections: List[Tuple[int, int, float]] = field(default_factory=list)
    plasticity_rules: Dict[str, SynapticPlasticity] = field(default_factory=dict)
    temporal_dynamics: Dict[str, float] = field(default_factory=dict)
    energy_efficiency: float = 0.0
    spike_sparsity: float = 0.0
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not self.temporal_dynamics:
            self.temporal_dynamics = {
                'time_constant': random.uniform(1.0, 50.0),
                'refractory_period': random.uniform(1.0, 10.0),
                'spike_threshold': random.uniform(0.5, 2.0)
            }


@dataclass
class MemristiveDevice:
    """Memristive device for synaptic weight storage."""
    resistance: float = 1000.0  # Ohms
    resistance_min: float = 100.0
    resistance_max: float = 10000.0
    programming_voltage: float = 1.0
    retention_time: float = 86400.0  # seconds
    endurance_cycles: int = 1000000
    current_cycles: int = 0
    
    def program_resistance(self, target_resistance: float, voltage: float):
        """Program memristor to target resistance."""
        if abs(voltage) > self.programming_voltage:
            # Nonlinear programming model
            resistance_change = (target_resistance - self.resistance) * min(1.0, abs(voltage) / self.programming_voltage)
            self.resistance = max(self.resistance_min, 
                                min(self.resistance_max, 
                                    self.resistance + resistance_change))
            self.current_cycles += 1
    
    def get_conductance(self) -> float:
        """Get conductance (1/resistance) with degradation."""
        degradation_factor = 1.0 - (self.current_cycles / self.endurance_cycles) * 0.1
        effective_resistance = self.resistance / degradation_factor
        return 1.0 / effective_resistance
    
    def temporal_drift(self, time_elapsed: float):
        """Model resistance drift over time."""
        drift_rate = 0.01 * time_elapsed / self.retention_time
        self.resistance *= (1.0 + random.gauss(0, drift_rate))
        self.resistance = max(self.resistance_min, min(self.resistance_max, self.resistance))


class NeuromorphicNAS:
    """Neuromorphic Neural Architecture Search."""
    
    def __init__(self,
                 architecture_space: ArchitectureSpace,
                 predictor: TPUv6Predictor,
                 config: SearchConfig):
        self.architecture_space = architecture_space
        self.predictor = predictor
        self.config = config
        
        # Neuromorphic components
        self.spiking_networks = []
        self.memristive_arrays = []
        self.plasticity_rules = {}
        
        # Search state
        self.population = []
        self.generation = 0
        self.search_history = []
        
        # Energy and timing models
        self.energy_model = self._initialize_energy_model()
        self.timing_model = self._initialize_timing_model()
        
        logging.info("Neuromorphic NAS initialized")
    
    def _initialize_energy_model(self) -> Dict[str, float]:
        """Initialize neuromorphic energy consumption model."""
        return {
            'spike_energy_pj': 1.0,  # Energy per spike in picojoules
            'synapse_energy_pj': 0.1,  # Energy per synaptic operation
            'leak_power_nw': 0.01,  # Leak power in nanowatts
            'memristor_write_pj': 10.0,  # Memristor write energy
            'memristor_read_pj': 0.1,  # Memristor read energy
        }
    
    def _initialize_timing_model(self) -> Dict[str, float]:
        """Initialize neuromorphic timing model."""
        return {
            'spike_delay_ns': 1.0,  # Spike propagation delay
            'synapse_delay_ns': 0.1,  # Synaptic delay
            'memristor_write_ns': 10.0,  # Memristor write time
            'memristor_read_ns': 1.0,  # Memristor read time
            'refractory_period_ns': 1000.0,  # Neuron refractory period
        }
    
    def search(self) -> Tuple[Architecture, PerformanceMetrics]:
        """Execute neuromorphic-inspired architecture search."""
        logging.info("Starting Neuromorphic NAS search...")
        
        # Initialize neuromorphic population
        self.population = self._initialize_neuromorphic_population()
        
        best_arch = None
        best_metrics = None
        best_fitness = float('-inf')
        
        for generation in range(self.config.max_iterations):
            generation_start = time.time()
            
            logging.info(f"Neuromorphic Generation {generation + 1}/{self.config.max_iterations}")
            
            # Evaluate population with neuromorphic metrics
            population_metrics = self._evaluate_neuromorphic_population()
            
            # Apply spike-timing dependent plasticity learning
            self._apply_stdp_learning(population_metrics)
            
            # Select best architectures
            fitness_scores = [self._calculate_neuromorphic_fitness(m) for m in population_metrics]
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            
            if fitness_scores[best_idx] > best_fitness:
                best_arch = self.population[best_idx]
                best_metrics = population_metrics[best_idx]
                best_fitness = fitness_scores[best_idx]
                
                logging.info(f"Generation {generation}: New best neuromorphic architecture!")
                logging.info(f"  Architecture: {best_arch.name}")
                logging.info(f"  Neuromorphic fitness: {best_fitness:.4f}")
                logging.info(f"  Energy efficiency: {best_metrics.energy_mj:.4f}mJ")
                logging.info(f"  Spike sparsity: {getattr(best_metrics, 'spike_sparsity', 0.0):.3f}")
            
            # Evolve population using neuromorphic principles
            self.population = self._neuromorphic_evolution()
            
            # Track generation statistics
            self.search_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': sum(fitness_scores) / len(fitness_scores),
                'population_diversity': self._calculate_neuromorphic_diversity(),
                'avg_spike_sparsity': sum(getattr(m, 'spike_sparsity', 0.0) for m in population_metrics) / len(population_metrics),
                'avg_energy_efficiency': sum(m.energy_mj for m in population_metrics) / len(population_metrics),
                'generation_time': time.time() - generation_start
            })
            
            # Check convergence
            if self._check_neuromorphic_convergence():
                logging.info(f"Neuromorphic search converged after {generation + 1} generations")
                break
        
        logging.info("Neuromorphic NAS search completed!")
        return best_arch, best_metrics
    
    def _initialize_neuromorphic_population(self) -> List[Architecture]:
        """Initialize population with neuromorphic-inspired architectures."""
        population = []
        
        for i in range(self.config.population_size):
            # Start with standard architecture
            arch = self.architecture_space.sample_random()
            
            # Convert to neuromorphic architecture
            neuromorphic_arch = self._convert_to_neuromorphic(arch)
            neuromorphic_arch.name = f"neuro_arch_{i}_{int(time.time() * 1000) % 10000}"
            
            population.append(neuromorphic_arch)
        
        logging.info(f"Initialized neuromorphic population of {len(population)} architectures")
        return population
    
    def _convert_to_neuromorphic(self, architecture: Architecture) -> Architecture:
        """Convert standard architecture to neuromorphic version."""
        neuromorphic_arch = architecture.copy()
        
        # Add neuromorphic-specific layers
        neuromorphic_layers = []
        
        for layer in architecture.layers:
            if layer.layer_type in ['conv2d', 'dense']:
                # Convert to spiking version
                spiking_layer = Layer(
                    layer_type=f'spiking_{layer.layer_type}',
                    input_channels=getattr(layer, 'input_channels', 3),
                    output_channels=getattr(layer, 'output_channels', 64),
                    kernel_size=getattr(layer, 'kernel_size', (3, 3)),
                    stride=getattr(layer, 'stride', (1, 1)),
                    activation='spiking_relu'
                )
                spiking_layer.neuromorphic_params = {
                    'spike_threshold': random.uniform(0.5, 2.0),
                    'membrane_decay': random.uniform(0.1, 0.9),
                    'refractory_period': random.randint(1, 5),
                    'spike_rate': random.uniform(0.01, 0.1)
                }
                neuromorphic_layers.append(spiking_layer)
            else:
                neuromorphic_layers.append(layer)
        
        neuromorphic_arch.layers = neuromorphic_layers
        return neuromorphic_arch
    
    def _evaluate_neuromorphic_population(self) -> List[PerformanceMetrics]:
        """Evaluate population with neuromorphic-specific metrics."""
        metrics = []
        
        for arch in self.population:
            try:
                # Standard performance prediction
                base_metrics = self.predictor.predict(arch)
                
                # Add neuromorphic-specific metrics
                neuromorphic_metrics = self._calculate_neuromorphic_metrics(arch)
                
                # Combine metrics
                enhanced_metrics = PerformanceMetrics(
                    latency_ms=base_metrics.latency_ms * neuromorphic_metrics['timing_factor'],
                    energy_mj=base_metrics.energy_mj * neuromorphic_metrics['energy_factor'],
                    accuracy=base_metrics.accuracy * neuromorphic_metrics['accuracy_factor'],
                    tops_per_watt=base_metrics.tops_per_watt * neuromorphic_metrics['efficiency_factor'],
                    memory_mb=base_metrics.memory_mb * neuromorphic_metrics['memory_factor'],
                    flops=base_metrics.flops
                )
                
                # Add neuromorphic-specific attributes
                enhanced_metrics.spike_sparsity = neuromorphic_metrics['spike_sparsity']
                enhanced_metrics.temporal_dynamics = neuromorphic_metrics['temporal_dynamics']
                enhanced_metrics.synaptic_operations = neuromorphic_metrics['synaptic_operations']
                
                metrics.append(enhanced_metrics)
                
            except Exception as e:
                logging.warning(f"Failed to evaluate neuromorphic architecture {arch.name}: {e}")
                # Fallback metrics
                fallback_metrics = PerformanceMetrics(
                    latency_ms=999.0,
                    energy_mj=999.0,
                    accuracy=0.001,
                    tops_per_watt=0.001,
                    memory_mb=999.0,
                    flops=1000000
                )
                fallback_metrics.spike_sparsity = 0.0
                metrics.append(fallback_metrics)
        
        return metrics
    
    def _calculate_neuromorphic_metrics(self, architecture: Architecture) -> Dict[str, float]:
        """Calculate neuromorphic-specific performance metrics."""
        total_neurons = 0
        total_synapses = 0
        total_spikes = 0
        total_energy = 0.0
        total_operations = 0
        
        # Simulate neuromorphic execution
        simulation_time = 1000.0  # ms
        dt = 1.0  # ms timestep
        
        for layer in architecture.layers:
            if hasattr(layer, 'neuromorphic_params'):
                params = layer.neuromorphic_params
                layer_neurons = getattr(layer, 'channels', 64)
                layer_synapses = layer_neurons * 100  # Approximate fan-in
                
                # Simulate spike activity
                spike_rate = params.get('spike_rate', 0.05)
                layer_spikes = int(layer_neurons * spike_rate * simulation_time / dt)
                
                # Calculate energy consumption
                spike_energy = layer_spikes * self.energy_model['spike_energy_pj']
                synapse_energy = layer_synapses * self.energy_model['synapse_energy_pj'] * simulation_time
                leak_energy = layer_neurons * self.energy_model['leak_power_nw'] * simulation_time
                
                total_neurons += layer_neurons
                total_synapses += layer_synapses
                total_spikes += layer_spikes
                total_energy += spike_energy + synapse_energy + leak_energy
                total_operations += layer_spikes + layer_synapses
        
        # Calculate factors relative to conventional architectures
        spike_sparsity = 1.0 - (total_spikes / max(1, total_neurons * simulation_time / dt))
        
        # Neuromorphic advantages
        energy_factor = 0.1 + 0.9 * spike_sparsity  # Lower energy with higher sparsity
        timing_factor = 0.5 + 0.5 * spike_sparsity  # Better timing with sparsity
        efficiency_factor = 1.0 + spike_sparsity  # Higher efficiency
        accuracy_factor = 0.8 + 0.2 * (1.0 - spike_sparsity)  # Slight accuracy trade-off
        memory_factor = 0.7 + 0.3 * spike_sparsity  # Memory efficiency from sparsity
        
        return {
            'spike_sparsity': spike_sparsity,
            'energy_factor': energy_factor,
            'timing_factor': timing_factor,
            'efficiency_factor': efficiency_factor,
            'accuracy_factor': accuracy_factor,
            'memory_factor': memory_factor,
            'temporal_dynamics': total_spikes / max(1, simulation_time),
            'synaptic_operations': total_operations,
            'total_energy_pj': total_energy
        }
    
    def _calculate_neuromorphic_fitness(self, metrics: PerformanceMetrics) -> float:
        """Calculate fitness score emphasizing neuromorphic advantages."""
        # Standard fitness components
        latency_score = max(0, 1.0 - metrics.latency_ms / self.config.max_latency_ms)
        accuracy_score = metrics.accuracy
        efficiency_score = metrics.tops_per_watt / self.config.target_tops_w
        energy_score = max(0, 1.0 - metrics.energy_mj / 10.0)
        
        # Neuromorphic-specific components
        sparsity_score = getattr(metrics, 'spike_sparsity', 0.0)
        temporal_score = min(1.0, getattr(metrics, 'temporal_dynamics', 0.0) / 100.0)
        
        # Weighted combination emphasizing neuromorphic benefits
        fitness = (0.2 * accuracy_score + 
                  0.2 * latency_score + 
                  0.2 * efficiency_score + 
                  0.15 * energy_score +
                  0.15 * sparsity_score +
                  0.1 * temporal_score)
        
        return fitness
    
    def _apply_stdp_learning(self, population_metrics: List[PerformanceMetrics]):
        """Apply spike-timing dependent plasticity learning."""
        for i, (arch, metrics) in enumerate(zip(self.population, population_metrics)):
            fitness = self._calculate_neuromorphic_fitness(metrics)
            
            # Update synaptic weights based on performance
            for layer in arch.layers:
                if hasattr(layer, 'neuromorphic_params'):
                    params = layer.neuromorphic_params
                    
                    # Strengthen connections for high-performing architectures
                    if fitness > 0.7:
                        # Potentiation
                        params['spike_threshold'] *= 0.98  # Easier to spike
                        params['spike_rate'] = min(0.2, params['spike_rate'] * 1.02)
                    elif fitness < 0.3:
                        # Depression
                        params['spike_threshold'] *= 1.02  # Harder to spike
                        params['spike_rate'] = max(0.01, params['spike_rate'] * 0.98)
                    
                    # Homeostatic plasticity
                    target_sparsity = 0.9
                    current_sparsity = getattr(metrics, 'spike_sparsity', 0.5)
                    if current_sparsity < target_sparsity:
                        params['spike_threshold'] *= 1.01  # Increase sparsity
                    else:
                        params['spike_threshold'] *= 0.99  # Decrease sparsity
    
    def _neuromorphic_evolution(self) -> List[Architecture]:
        """Evolve population using neuromorphic-inspired operators."""
        new_population = []
        
        # Calculate fitness for selection
        current_metrics = self._evaluate_neuromorphic_population()
        fitness_scores = [self._calculate_neuromorphic_fitness(m) for m in current_metrics]
        
        # Elite selection
        elite_size = max(1, self.config.population_size // 4)
        sorted_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        
        for i in range(elite_size):
            new_population.append(self.population[sorted_indices[i]].copy())
        
        # Neuromorphic mutations and crossovers
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Synaptic crossover
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                child = self._synaptic_crossover(parent1, parent2)
            else:
                # Spike-based mutation
                parent = self._tournament_selection(fitness_scores)
                child = self._spike_mutation(parent)
            
            new_population.append(child)
        
        return new_population[:self.config.population_size]
    
    def _tournament_selection(self, fitness_scores: List[float]) -> Architecture:
        """Tournament selection for parent choice."""
        tournament_size = 3
        tournament_indices = random.sample(range(len(fitness_scores)), 
                                         min(tournament_size, len(fitness_scores)))
        
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return self.population[best_idx]
    
    def _synaptic_crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover based on synaptic connectivity patterns."""
        child = parent1.copy()
        child.name = f"synaptic_cross_{int(time.time() * 1000) % 10000}"
        
        # Exchange neuromorphic parameters
        for i in range(min(len(parent1.layers), len(parent2.layers))):
            if (hasattr(parent1.layers[i], 'neuromorphic_params') and
                hasattr(parent2.layers[i], 'neuromorphic_params')):
                
                if random.random() < 0.5:
                    # Exchange parameters
                    p1_params = parent1.layers[i].neuromorphic_params
                    p2_params = parent2.layers[i].neuromorphic_params
                    
                    child.layers[i].neuromorphic_params = {
                        'spike_threshold': (p1_params['spike_threshold'] + p2_params['spike_threshold']) / 2,
                        'membrane_decay': random.choice([p1_params['membrane_decay'], p2_params['membrane_decay']]),
                        'refractory_period': random.choice([p1_params['refractory_period'], p2_params['refractory_period']]),
                        'spike_rate': (p1_params['spike_rate'] + p2_params['spike_rate']) / 2
                    }
        
        return child
    
    def _spike_mutation(self, parent: Architecture) -> Architecture:
        """Mutation based on spike timing and patterns."""
        child = parent.copy()
        child.name = f"spike_mut_{parent.name}_{int(time.time() * 1000) % 10000}"
        
        for layer in child.layers:
            if hasattr(layer, 'neuromorphic_params'):
                params = layer.neuromorphic_params
                
                if random.random() < self.config.mutation_rate:
                    # Mutate spike threshold
                    noise = random.gauss(0, 0.1)
                    params['spike_threshold'] = max(0.1, params['spike_threshold'] + noise)
                
                if random.random() < self.config.mutation_rate:
                    # Mutate membrane decay
                    noise = random.gauss(0, 0.05)
                    params['membrane_decay'] = max(0.01, min(0.99, params['membrane_decay'] + noise))
                
                if random.random() < self.config.mutation_rate:
                    # Mutate spike rate
                    noise = random.gauss(0, 0.01)
                    params['spike_rate'] = max(0.001, min(0.2, params['spike_rate'] + noise))
                
                if random.random() < self.config.mutation_rate:
                    # Mutate refractory period
                    params['refractory_period'] = max(1, params['refractory_period'] + random.randint(-1, 1))
        
        return child
    
    def _calculate_neuromorphic_diversity(self) -> float:
        """Calculate diversity of neuromorphic population."""
        if not self.population:
            return 0.0
        
        # Collect neuromorphic parameters
        spike_thresholds = []
        spike_rates = []
        membrane_decays = []
        
        for arch in self.population:
            for layer in arch.layers:
                if hasattr(layer, 'neuromorphic_params'):
                    params = layer.neuromorphic_params
                    spike_thresholds.append(params['spike_threshold'])
                    spike_rates.append(params['spike_rate'])
                    membrane_decays.append(params['membrane_decay'])
        
        # Calculate standard deviations
        def std_dev(values):
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return math.sqrt(variance)
        
        threshold_diversity = std_dev(spike_thresholds)
        rate_diversity = std_dev(spike_rates)
        decay_diversity = std_dev(membrane_decays)
        
        return (threshold_diversity + rate_diversity + decay_diversity) / 3.0
    
    def _check_neuromorphic_convergence(self) -> bool:
        """Check convergence based on neuromorphic metrics."""
        if len(self.search_history) < 10:
            return False
        
        # Check fitness improvement
        recent_fitness = [h['best_fitness'] for h in self.search_history[-10:]]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        # Check diversity
        recent_diversity = [h['population_diversity'] for h in self.search_history[-5:]]
        avg_diversity = sum(recent_diversity) / len(recent_diversity)
        
        # Converged if little improvement and low diversity
        return (fitness_improvement < self.config.early_stop_threshold and avg_diversity < 0.01)
    
    def get_neuromorphic_statistics(self) -> Dict[str, Any]:
        """Get comprehensive neuromorphic search statistics."""
        if not self.search_history:
            return {}
        
        latest_stats = self.search_history[-1] if self.search_history else {}
        
        return {
            'search_progress': {
                'generations_completed': len(self.search_history),
                'best_fitness': latest_stats.get('best_fitness', 0.0),
                'avg_fitness': latest_stats.get('avg_fitness', 0.0),
                'population_diversity': latest_stats.get('population_diversity', 0.0)
            },
            'neuromorphic_metrics': {
                'avg_spike_sparsity': latest_stats.get('avg_spike_sparsity', 0.0),
                'avg_energy_efficiency': latest_stats.get('avg_energy_efficiency', 0.0),
                'total_neuromorphic_layers': sum(
                    1 for arch in self.population 
                    for layer in arch.layers 
                    if hasattr(layer, 'neuromorphic_params')
                )
            },
            'energy_model': self.energy_model,
            'timing_model': self.timing_model,
            'search_history': self.search_history
        }


# Integration with main search system
def create_neuromorphic_nas_searcher(architecture_space: ArchitectureSpace,
                                    predictor: TPUv6Predictor,
                                    config: SearchConfig) -> NeuromorphicNAS:
    """Factory function to create neuromorphic NAS searcher."""
    return NeuromorphicNAS(architecture_space, predictor, config)
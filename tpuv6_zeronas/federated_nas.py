"""Federated Neural Architecture Search: Distributed Privacy-Preserving NAS."""

import logging
import time
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from pathlib import Path

from .architecture import Architecture, ArchitectureSpace
from .metrics import PerformanceMetrics
from .predictor import TPUv6Predictor
from .core import SearchConfig
from .quantum_nas import QuantumInspiredNAS


@dataclass
class FederatedNode:
    """Represents a federated learning participant."""
    node_id: str
    hardware_profile: Dict[str, Any] = field(default_factory=dict)
    data_characteristics: Dict[str, Any] = field(default_factory=dict)
    privacy_level: float = 1.0  # 0-1, higher = more private
    compute_capacity: float = 1.0  # Relative compute capacity
    reliability_score: float = 1.0  # Historical reliability
    architectures_contributed: int = 0
    last_active: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize derived attributes."""
        if not self.hardware_profile:
            self.hardware_profile = {
                'memory_gb': random.uniform(4, 64),
                'compute_units': random.randint(1, 8),
                'bandwidth_mbps': random.uniform(10, 1000),
                'specialized_hardware': random.choice(['cpu', 'gpu', 'tpu', 'edge'])
            }
        
        if not self.data_characteristics:
            self.data_characteristics = {
                'domain': random.choice(['vision', 'nlp', 'audio', 'multimodal']),
                'dataset_size': random.randint(1000, 1000000),
                'data_quality': random.uniform(0.5, 1.0),
                'label_noise': random.uniform(0.0, 0.2)
            }


@dataclass
class FederatedSearchState:
    """Global state of federated NAS search."""
    global_round: int = 0
    participating_nodes: Set[str] = field(default_factory=set)
    global_pareto_front: List[Tuple[Architecture, PerformanceMetrics]] = field(default_factory=list)
    consensus_architectures: List[Architecture] = field(default_factory=list)
    node_contributions: Dict[str, int] = field(default_factory=dict)
    privacy_budget: float = 1.0
    convergence_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DifferentialPrivacyMechanism:
    """Differential privacy for architecture sharing."""
    epsilon: float = 1.0  # Privacy budget per round
    delta: float = 1e-5   # Privacy parameter
    sensitivity: float = 1.0  # Global sensitivity
    noise_scale: float = field(init=False)
    
    def __post_init__(self):
        """Calculate noise scale."""
        self.noise_scale = self.sensitivity / self.epsilon
    
    def add_noise(self, value: float) -> float:
        """Add Gaussian noise for differential privacy."""
        noise = random.gauss(0, self.noise_scale)
        return value + noise
    
    def privatize_metrics(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """Apply differential privacy to performance metrics."""
        return PerformanceMetrics(
            latency_ms=max(0.1, self.add_noise(metrics.latency_ms)),
            energy_mj=max(0.01, self.add_noise(metrics.energy_mj)),
            accuracy=min(1.0, max(0.0, self.add_noise(metrics.accuracy))),
            tops_per_watt=max(0.1, self.add_noise(metrics.tops_per_watt)),
            memory_mb=max(1.0, self.add_noise(metrics.memory_mb))
        )


@dataclass
class SecureAggregation:
    """Secure aggregation protocol for federated NAS."""
    threshold: int = 3  # Minimum nodes for aggregation
    max_nodes: int = 100
    aggregation_rounds: int = 0
    
    def aggregate_architectures(self, 
                              node_architectures: Dict[str, List[Architecture]],
                              node_weights: Dict[str, float]) -> List[Architecture]:
        """Securely aggregate architectures from multiple nodes."""
        if len(node_architectures) < self.threshold:
            logging.warning(f"Insufficient nodes for secure aggregation: {len(node_architectures)}")
            return []
        
        # Extract all unique architectures
        all_architectures = []
        architecture_votes = {}
        
        for node_id, architectures in node_architectures.items():
            weight = node_weights.get(node_id, 1.0)
            
            for arch in architectures:
                arch_hash = self._hash_architecture(arch)
                
                if arch_hash not in architecture_votes:
                    architecture_votes[arch_hash] = {
                        'architecture': arch,
                        'votes': 0,
                        'weighted_votes': 0.0,
                        'contributing_nodes': set()
                    }
                
                architecture_votes[arch_hash]['votes'] += 1
                architecture_votes[arch_hash]['weighted_votes'] += weight
                architecture_votes[arch_hash]['contributing_nodes'].add(node_id)
        
        # Select architectures with sufficient consensus
        consensus_threshold = max(2, len(node_architectures) // 2)
        consensus_architectures = []
        
        for arch_hash, vote_data in architecture_votes.items():
            if vote_data['votes'] >= consensus_threshold:
                consensus_architectures.append(vote_data['architecture'])
        
        logging.info(f"Secure aggregation: {len(consensus_architectures)} architectures "
                    f"from {len(node_architectures)} nodes")
        
        self.aggregation_rounds += 1
        return consensus_architectures
    
    def aggregate_metrics(self,
                         node_metrics: Dict[str, List[PerformanceMetrics]],
                         node_weights: Dict[str, float]) -> List[PerformanceMetrics]:
        """Securely aggregate performance metrics."""
        if len(node_metrics) < self.threshold:
            return []
        
        # Align metrics by architecture (simplified - assumes same order)
        aggregated_metrics = []
        max_metrics = max(len(metrics) for metrics in node_metrics.values())
        
        for i in range(max_metrics):
            latency_values = []
            energy_values = []
            accuracy_values = []
            tops_values = []
            memory_values = []
            total_weight = 0.0
            
            for node_id, metrics_list in node_metrics.items():
                if i < len(metrics_list):
                    weight = node_weights.get(node_id, 1.0)
                    metrics = metrics_list[i]
                    
                    latency_values.append(metrics.latency_ms * weight)
                    energy_values.append(metrics.energy_mj * weight)
                    accuracy_values.append(metrics.accuracy * weight)
                    tops_values.append(metrics.tops_per_watt * weight)
                    memory_values.append(metrics.memory_mb * weight)
                    total_weight += weight
            
            if total_weight > 0:
                aggregated = PerformanceMetrics(
                    latency_ms=sum(latency_values) / total_weight,
                    energy_mj=sum(energy_values) / total_weight,
                    accuracy=sum(accuracy_values) / total_weight,
                    tops_per_watt=sum(tops_values) / total_weight,
                    memory_mb=sum(memory_values) / total_weight
                )
                aggregated_metrics.append(aggregated)
        
        return aggregated_metrics
    
    def _hash_architecture(self, architecture: Architecture) -> str:
        """Create hash of architecture for consensus voting."""
        arch_string = f"{len(architecture.layers)}"
        for layer in architecture.layers:
            arch_string += f"_{layer.layer_type}_{getattr(layer, 'channels', 0)}"
        
        return hashlib.md5(arch_string.encode()).hexdigest()


class FederatedNAS:
    """Federated Neural Architecture Search coordinator."""
    
    def __init__(self,
                 architecture_space: ArchitectureSpace,
                 predictor: TPUv6Predictor,
                 config: SearchConfig,
                 num_nodes: int = 10,
                 privacy_epsilon: float = 1.0):
        self.architecture_space = architecture_space
        self.predictor = predictor
        self.config = config
        self.num_nodes = num_nodes
        
        # Federated components
        self.nodes = self._initialize_nodes()
        self.federated_state = FederatedSearchState()
        self.privacy_mechanism = DifferentialPrivacyMechanism(epsilon=privacy_epsilon)
        self.secure_aggregator = SecureAggregation()
        
        # Node-specific searchers
        self.node_searchers = {}
        self._initialize_node_searchers()
        
        # Global coordination
        self.communication_rounds = 0
        self.total_communication_cost = 0.0
        self.convergence_history = []
        
        logging.info(f"Federated NAS initialized with {num_nodes} nodes")
    
    def _initialize_nodes(self) -> List[FederatedNode]:
        """Initialize federated learning nodes."""
        nodes = []
        
        for i in range(self.num_nodes):
            node = FederatedNode(
                node_id=f"node_{i:03d}",
                privacy_level=random.uniform(0.5, 1.0),
                compute_capacity=random.uniform(0.5, 2.0),
                reliability_score=random.uniform(0.7, 1.0)
            )
            nodes.append(node)
        
        logging.info(f"Initialized {len(nodes)} federated nodes")
        return nodes
    
    def _initialize_node_searchers(self):
        """Initialize local searchers for each node."""
        for node in self.nodes:
            # Create node-specific configuration
            node_config = SearchConfig(
                max_iterations=max(5, self.config.max_iterations // 4),
                population_size=max(10, self.config.population_size // 2),
                mutation_rate=self.config.mutation_rate,
                crossover_rate=self.config.crossover_rate,
                target_tops_w=self.config.target_tops_w,
                max_latency_ms=self.config.max_latency_ms,
                min_accuracy=self.config.min_accuracy,
                enable_parallel=True,
                enable_caching=True,
                enable_research=True
            )
            
            # Create quantum-inspired searcher for each node
            searcher = QuantumInspiredNAS(
                self.architecture_space,
                self.predictor,
                node_config
            )
            
            self.node_searchers[node.node_id] = searcher
    
    def federated_search(self) -> Tuple[Architecture, PerformanceMetrics]:
        """Execute federated neural architecture search."""
        logging.info("Starting Federated NAS...")
        
        global_best_arch = None
        global_best_metrics = None
        global_best_fitness = float('-inf')
        
        for federated_round in range(self.config.max_iterations):
            round_start_time = time.time()
            
            logging.info(f"Federated Round {federated_round + 1}/{self.config.max_iterations}")
            
            # Node selection for this round
            selected_nodes = self._select_participating_nodes()
            
            # Parallel local search on selected nodes
            node_results = self._execute_parallel_local_search(selected_nodes)
            
            # Secure aggregation of results
            aggregated_architectures = self._secure_aggregate_results(node_results)
            
            # Update global state
            self._update_global_state(aggregated_architectures)
            
            # Evaluate global best
            round_best = self._evaluate_global_candidates()
            
            if round_best:
                best_arch, best_metrics = round_best
                fitness = self._calculate_fitness(best_metrics)
                
                if fitness > global_best_fitness:
                    global_best_arch = best_arch
                    global_best_metrics = best_metrics
                    global_best_fitness = fitness
                    
                    logging.info(f"Federated Round {federated_round}: New global best!")
                    logging.info(f"  Architecture: {best_arch.name}")
                    logging.info(f"  Fitness: {fitness:.4f}")
                    logging.info(f"  Metrics: {best_metrics.latency_ms:.2f}ms, "
                               f"{best_metrics.accuracy:.3f} acc, "
                               f"{best_metrics.tops_per_watt:.1f} TOPS/W")
            
            # Communication and privacy accounting
            round_time = time.time() - round_start_time
            communication_cost = self._calculate_communication_cost(len(selected_nodes))
            self.total_communication_cost += communication_cost
            
            # Update privacy budget
            self._update_privacy_budget()
            
            # Track convergence
            convergence_metrics = self._calculate_convergence_metrics()
            self.convergence_history.append({
                'round': federated_round,
                'global_best_fitness': global_best_fitness,
                'participating_nodes': len(selected_nodes),
                'communication_cost': communication_cost,
                'round_time': round_time,
                'privacy_budget': self.federated_state.privacy_budget,
                'convergence_score': convergence_metrics.get('consensus_score', 0.0)
            })
            
            # Check for early stopping
            if self._check_federated_convergence():
                logging.info(f"Federated search converged after {federated_round + 1} rounds")
                break
        
        logging.info("Federated NAS completed!")
        logging.info(f"Total communication cost: {self.total_communication_cost:.2f} MB")
        logging.info(f"Final privacy budget: {self.federated_state.privacy_budget:.4f}")
        
        return global_best_arch, global_best_metrics
    
    def _select_participating_nodes(self) -> List[FederatedNode]:
        """Select nodes for current federated round."""
        # Filter available nodes
        available_nodes = [
            node for node in self.nodes
            if (time.time() - node.last_active) < 3600  # Active within last hour
            and node.reliability_score > 0.5
        ]
        
        if len(available_nodes) < 3:
            available_nodes = self.nodes  # Fallback to all nodes
        
        # Select based on compute capacity and diversity
        participation_rate = min(1.0, max(0.3, 1.0 - self.communication_rounds * 0.1))
        num_selected = max(3, int(len(available_nodes) * participation_rate))
        
        # Weighted selection based on compute capacity and reliability
        weights = [
            node.compute_capacity * node.reliability_score * (1.0 + random.uniform(-0.1, 0.1))
            for node in available_nodes
        ]
        
        # Select top weighted nodes with some randomness
        node_weight_pairs = list(zip(available_nodes, weights))
        node_weight_pairs.sort(key=lambda x: x[1], reverse=True)
        
        selected = [pair[0] for pair in node_weight_pairs[:num_selected]]
        
        # Update participation tracking
        for node in selected:
            self.federated_state.participating_nodes.add(node.node_id)
            node.last_active = time.time()
        
        logging.info(f"Selected {len(selected)} nodes for federated round")
        return selected
    
    def _execute_parallel_local_search(self, 
                                     selected_nodes: List[FederatedNode]) -> Dict[str, Tuple[List[Architecture], List[PerformanceMetrics]]]:
        """Execute local search on selected nodes in parallel."""
        node_results = {}
        
        with ThreadPoolExecutor(max_workers=min(8, len(selected_nodes))) as executor:
            # Submit local search tasks
            future_to_node = {}
            
            for node in selected_nodes:
                future = executor.submit(self._local_node_search, node)
                future_to_node[future] = node
            
            # Collect results
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                
                try:
                    architectures, metrics = future.result(timeout=300)  # 5-minute timeout
                    node_results[node.node_id] = (architectures, metrics)
                    node.architectures_contributed += len(architectures)
                    
                    logging.info(f"Node {node.node_id} completed local search: "
                               f"{len(architectures)} architectures")
                    
                except Exception as e:
                    logging.warning(f"Node {node.node_id} failed local search: {e}")
                    # Penalize unreliable nodes
                    node.reliability_score *= 0.9
        
        return node_results
    
    def _local_node_search(self, node: FederatedNode) -> Tuple[List[Architecture], List[PerformanceMetrics]]:
        """Execute local search on a single node."""
        searcher = self.node_searchers[node.node_id]
        
        # Adjust search based on node characteristics
        if node.compute_capacity < 1.0:
            # Reduce search intensity for low-capacity nodes
            searcher.config.max_iterations = max(3, int(searcher.config.max_iterations * node.compute_capacity))
            searcher.config.population_size = max(5, int(searcher.config.population_size * node.compute_capacity))
        
        # Execute local search
        best_arch, best_metrics = searcher.search()
        
        # Collect diverse architectures from search
        local_architectures = []
        local_metrics = []
        
        if best_arch and best_metrics:
            local_architectures.append(best_arch)
            local_metrics.append(best_metrics)
        
        # Add additional diverse candidates from search state
        if hasattr(searcher, 'generation_stats') and searcher.generation_stats:
            # Extract diverse architectures from search history
            for stats in searcher.generation_stats[-3:]:  # Last 3 generations
                if 'diverse_candidates' in stats:
                    local_architectures.extend(stats['diverse_candidates'][:2])
        
        # Apply differential privacy to metrics
        private_metrics = []
        for metrics in local_metrics:
            private_metrics.append(self.privacy_mechanism.privatize_metrics(metrics))
        
        logging.info(f"Node {node.node_id} local search completed: "
                    f"{len(local_architectures)} architectures")
        
        return local_architectures, private_metrics
    
    def _secure_aggregate_results(self, 
                                node_results: Dict[str, Tuple[List[Architecture], List[PerformanceMetrics]]]) -> List[Architecture]:
        """Securely aggregate results from multiple nodes."""
        # Extract architectures and metrics
        node_architectures = {}
        node_metrics = {}
        node_weights = {}
        
        for node_id, (architectures, metrics) in node_results.items():
            node_architectures[node_id] = architectures
            node_metrics[node_id] = metrics
            
            # Weight based on node reliability and contribution
            node = next(n for n in self.nodes if n.node_id == node_id)
            node_weights[node_id] = node.reliability_score * node.compute_capacity
        
        # Secure aggregation
        consensus_architectures = self.secure_aggregator.aggregate_architectures(
            node_architectures, node_weights
        )
        
        aggregated_metrics = self.secure_aggregator.aggregate_metrics(
            node_metrics, node_weights
        )
        
        logging.info(f"Secure aggregation: {len(consensus_architectures)} consensus architectures")
        
        return consensus_architectures
    
    def _update_global_state(self, aggregated_architectures: List[Architecture]):
        """Update global federated state."""
        self.federated_state.global_round += 1
        self.federated_state.consensus_architectures = aggregated_architectures
        
        # Update node contributions
        for node_id in self.federated_state.participating_nodes:
            if node_id not in self.federated_state.node_contributions:
                self.federated_state.node_contributions[node_id] = 0
            self.federated_state.node_contributions[node_id] += 1
        
        self.communication_rounds += 1
    
    def _evaluate_global_candidates(self) -> Optional[Tuple[Architecture, PerformanceMetrics]]:
        """Evaluate global candidate architectures."""
        if not self.federated_state.consensus_architectures:
            return None
        
        best_arch = None
        best_metrics = None
        best_fitness = float('-inf')
        
        for arch in self.federated_state.consensus_architectures:
            try:
                metrics = self.predictor.predict(arch)
                fitness = self._calculate_fitness(metrics)
                
                if fitness > best_fitness:
                    best_arch = arch
                    best_metrics = metrics
                    best_fitness = fitness
                    
            except Exception as e:
                logging.warning(f"Failed to evaluate global candidate {arch.name}: {e}")
        
        # Update global Pareto front
        if best_arch and best_metrics:
            self._update_pareto_front(best_arch, best_metrics)
        
        return (best_arch, best_metrics) if best_arch else None
    
    def _update_pareto_front(self, architecture: Architecture, metrics: PerformanceMetrics):
        """Update global Pareto front."""
        # Simple Pareto dominance check
        is_dominated = False
        dominated_indices = []
        
        for i, (existing_arch, existing_metrics) in enumerate(self.federated_state.global_pareto_front):
            if self._dominates(existing_metrics, metrics):
                is_dominated = True
                break
            elif self._dominates(metrics, existing_metrics):
                dominated_indices.append(i)
        
        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(dominated_indices):
                self.federated_state.global_pareto_front.pop(i)
            
            # Add new solution
            self.federated_state.global_pareto_front.append((architecture, metrics))
            
            logging.info(f"Updated global Pareto front: {len(self.federated_state.global_pareto_front)} solutions")
    
    def _dominates(self, metrics1: PerformanceMetrics, metrics2: PerformanceMetrics) -> bool:
        """Check if metrics1 Pareto dominates metrics2."""
        # Minimize latency, energy, memory; maximize accuracy, TOPS/W
        better_in_all = (
            metrics1.latency_ms <= metrics2.latency_ms and
            metrics1.energy_mj <= metrics2.energy_mj and
            metrics1.memory_mb <= metrics2.memory_mb and
            metrics1.accuracy >= metrics2.accuracy and
            metrics1.tops_per_watt >= metrics2.tops_per_watt
        )
        
        better_in_some = (
            metrics1.latency_ms < metrics2.latency_ms or
            metrics1.energy_mj < metrics2.energy_mj or
            metrics1.memory_mb < metrics2.memory_mb or
            metrics1.accuracy > metrics2.accuracy or
            metrics1.tops_per_watt > metrics2.tops_per_watt
        )
        
        return better_in_all and better_in_some
    
    def _calculate_fitness(self, metrics: PerformanceMetrics) -> float:
        """Calculate fitness score for architecture."""
        # Multi-objective fitness (same as quantum NAS)
        latency_score = max(0, 1.0 - metrics.latency_ms / self.config.max_latency_ms)
        accuracy_score = metrics.accuracy
        efficiency_score = metrics.tops_per_watt / self.config.target_tops_w
        energy_score = max(0, 1.0 - metrics.energy_mj / 10.0)
        
        return (0.3 * accuracy_score + 
                0.25 * latency_score + 
                0.25 * efficiency_score + 
                0.2 * energy_score)
    
    def _calculate_communication_cost(self, num_nodes: int) -> float:
        """Calculate communication cost in MB."""
        # Estimate based on architecture and metrics size
        avg_arch_size = 0.1  # MB per architecture
        avg_metrics_size = 0.001  # MB per metrics
        overhead = 0.01  # Protocol overhead
        
        return num_nodes * (avg_arch_size + avg_metrics_size) + overhead
    
    def _update_privacy_budget(self):
        """Update privacy budget after communication round."""
        budget_consumed = self.privacy_mechanism.epsilon / self.config.max_iterations
        self.federated_state.privacy_budget = max(0, self.federated_state.privacy_budget - budget_consumed)
    
    def _calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate federated convergence metrics."""
        if len(self.convergence_history) < 3:
            return {'consensus_score': 0.0}
        
        # Check fitness improvement stability
        recent_fitness = [h['global_best_fitness'] for h in self.convergence_history[-5:]]
        fitness_variance = sum((f - sum(recent_fitness)/len(recent_fitness))**2 for f in recent_fitness) / len(recent_fitness)
        
        # Node participation consistency
        recent_participation = [h['participating_nodes'] for h in self.convergence_history[-5:]]
        participation_consistency = 1.0 - (max(recent_participation) - min(recent_participation)) / max(recent_participation)
        
        consensus_score = (1.0 / (1.0 + fitness_variance)) * participation_consistency
        
        return {
            'consensus_score': consensus_score,
            'fitness_variance': fitness_variance,
            'participation_consistency': participation_consistency
        }
    
    def _check_federated_convergence(self) -> bool:
        """Check if federated search has converged."""
        if len(self.convergence_history) < 10:
            return False
        
        # Check if global best hasn't improved in recent rounds
        recent_fitness = [h['global_best_fitness'] for h in self.convergence_history[-10:]]
        improvement = max(recent_fitness) - min(recent_fitness)
        
        # Check privacy budget
        privacy_exhausted = self.federated_state.privacy_budget < 0.1
        
        # Check consensus stability
        convergence_metrics = self._calculate_convergence_metrics()
        consensus_stable = convergence_metrics['consensus_score'] > 0.8
        
        return improvement < self.config.early_stop_threshold or privacy_exhausted or consensus_stable
    
    def get_federated_statistics(self) -> Dict[str, Any]:
        """Get comprehensive federated search statistics."""
        return {
            'federated_state': {
                'global_round': self.federated_state.global_round,
                'participating_nodes': len(self.federated_state.participating_nodes),
                'consensus_architectures': len(self.federated_state.consensus_architectures),
                'pareto_front_size': len(self.federated_state.global_pareto_front),
                'privacy_budget': self.federated_state.privacy_budget,
                'node_contributions': dict(list(self.federated_state.node_contributions.items())[:5])
            },
            'communication': {
                'total_rounds': self.communication_rounds,
                'total_cost_mb': self.total_communication_cost,
                'avg_cost_per_round': self.total_communication_cost / max(1, self.communication_rounds)
            },
            'privacy': {
                'epsilon': self.privacy_mechanism.epsilon,
                'remaining_budget': self.federated_state.privacy_budget,
                'budget_consumed': 1.0 - self.federated_state.privacy_budget
            },
            'nodes': {
                'total_nodes': len(self.nodes),
                'avg_reliability': sum(node.reliability_score for node in self.nodes) / len(self.nodes),
                'total_contributions': sum(node.architectures_contributed for node in self.nodes)
            },
            'convergence_history': self.convergence_history
        }


# Integration with main search system
def create_federated_nas_searcher(architecture_space: ArchitectureSpace,
                                 predictor: TPUv6Predictor,
                                 config: SearchConfig,
                                 num_nodes: int = 10,
                                 privacy_epsilon: float = 1.0) -> FederatedNAS:
    """Factory function to create federated NAS searcher."""
    return FederatedNAS(architecture_space, predictor, config, num_nodes, privacy_epsilon)
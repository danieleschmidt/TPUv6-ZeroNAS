"""Advanced caching optimizations for TPUv6-ZeroNAS Generation 3."""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import pickle

from .architecture import Architecture
from .metrics import PerformanceMetrics


@dataclass
class CacheConfig:
    """Configuration for advanced caching."""
    max_memory_cache_size: int = 1000
    max_disk_cache_mb: float = 100.0
    memory_cleanup_threshold: float = 0.8
    enable_compression: bool = True
    enable_predictive_loading: bool = True
    cache_hit_reward: float = 1.0
    cache_miss_penalty: float = 0.1


class AdaptiveLRUCache:
    """Adaptive Least Recently Used cache with intelligent eviction."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = OrderedDict()
        self.access_frequency = defaultdict(int)
        self.access_times = {}
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[PerformanceMetrics]:
        """Get value with adaptive LRU tracking."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recent)
                value = self.cache.pop(key)
                self.cache[key] = value
                
                # Update access patterns
                self.access_frequency[key] += 1
                self.access_times[key] = time.time()
                
                self.hits += 1
                return value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: PerformanceMetrics) -> None:
        """Store value with intelligent eviction policy."""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Check if eviction needed
            if len(self.cache) >= self.config.max_memory_cache_size:
                self._intelligent_eviction()
            
            # Add new entry
            self.cache[key] = value
            self.access_frequency[key] = 1
            self.access_times[key] = time.time()
    
    def _intelligent_eviction(self) -> None:
        """Intelligent eviction based on access patterns and age."""
        try:
            current_time = time.time()
            
            # Score each entry for eviction (higher score = more likely to evict)
            eviction_scores = []
            
            for key in list(self.cache.keys()):
                # Age penalty (older entries more likely to evict)
                age = current_time - self.access_times.get(key, current_time)
                age_score = min(age / 3600.0, 1.0)  # Normalize by 1 hour
                
                # Frequency bonus (frequently accessed entries less likely to evict)
                freq = self.access_frequency.get(key, 1)
                freq_score = 1.0 / (1.0 + freq)
                
                # Combined score
                eviction_score = age_score + freq_score
                eviction_scores.append((eviction_score, key))
            
            # Sort by eviction score (highest first)
            eviction_scores.sort(reverse=True)
            
            # Evict entries until we're under threshold
            target_size = int(self.config.max_memory_cache_size * self.config.memory_cleanup_threshold)
            entries_to_evict = len(self.cache) - target_size
            
            for i in range(min(entries_to_evict, len(eviction_scores))):
                _, key_to_evict = eviction_scores[i]
                if key_to_evict in self.cache:
                    del self.cache[key_to_evict]
                    if key_to_evict in self.access_frequency:
                        del self.access_frequency[key_to_evict]
                    if key_to_evict in self.access_times:
                        del self.access_times[key_to_evict]
                    self.evictions += 1
            
            self.logger.debug(f"Evicted {entries_to_evict} entries from cache")
            
        except Exception as e:
            self.logger.error(f"Intelligent eviction failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                'size': len(self.cache),
                'max_size': self.config.max_memory_cache_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_frequency.clear()
            self.access_times.clear()


class PredictiveCache:
    """Predictive caching system that pre-loads likely future requests."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.prediction_patterns = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self.request_history = []
        self.max_history_size = 1000
        
    def record_access(self, key: str) -> None:
        """Record access pattern for prediction."""
        self.request_history.append((key, time.time()))
        
        # Trim history if too large
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size//2:]
    
    def predict_next_requests(self, current_key: str, num_predictions: int = 5) -> List[str]:
        """Predict likely next cache requests based on patterns."""
        try:
            predictions = []
            
            # Find patterns: what typically follows current_key
            for i, (key, _) in enumerate(self.request_history[:-1]):
                if key == current_key:
                    next_key = self.request_history[i + 1][0]
                    self.prediction_patterns[current_key].append(next_key)
            
            # Get most common following keys
            if current_key in self.prediction_patterns:
                following_keys = self.prediction_patterns[current_key]
                key_counts = defaultdict(int)
                
                for key in following_keys:
                    key_counts[key] += 1
                
                # Sort by frequency and return top predictions
                sorted_predictions = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
                predictions = [key for key, count in sorted_predictions[:num_predictions]]
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return []


class HierarchicalCache:
    """Multi-level hierarchical cache system."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.l1_cache = AdaptiveLRUCache(config)  # Fast in-memory cache
        self.l2_cache = {}  # Compressed in-memory cache
        self.predictive_cache = PredictiveCache(config)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.l1_hits = 0
        self.l2_hits = 0
        self.total_requests = 0
        
    def get(self, key: str) -> Optional[PerformanceMetrics]:
        """Get value from hierarchical cache."""
        self.total_requests += 1
        self.predictive_cache.record_access(key)
        
        # Try L1 cache first (fastest)
        value = self.l1_cache.get(key)
        if value is not None:
            self.l1_hits += 1
            
            # Predictively load related entries
            if self.config.enable_predictive_loading:
                self._predictive_load(key)
            
            return value
        
        # Try L2 cache (compressed)
        if key in self.l2_cache:
            try:
                compressed_data = self.l2_cache[key]
                value = pickle.loads(compressed_data)
                
                # Promote to L1 cache
                self.l1_cache.put(key, value)
                
                self.l2_hits += 1
                return value
                
            except Exception as e:
                self.logger.warning(f"L2 cache deserialization failed: {e}")
                del self.l2_cache[key]
        
        return None
    
    def put(self, key: str, value: PerformanceMetrics) -> None:
        """Store value in hierarchical cache."""
        # Always put in L1 cache
        self.l1_cache.put(key, value)
        
        # Optionally compress and store in L2 cache
        if self.config.enable_compression:
            try:
                compressed_data = pickle.dumps(value)
                self.l2_cache[key] = compressed_data
                
            except Exception as e:
                self.logger.warning(f"L2 cache serialization failed: {e}")
    
    def _predictive_load(self, current_key: str) -> None:
        """Predictively load entries that are likely to be requested next."""
        try:
            predictions = self.predictive_cache.predict_next_requests(current_key, num_predictions=3)
            
            for predicted_key in predictions:
                # If prediction is in L2 but not L1, promote it
                if predicted_key in self.l2_cache and self.l1_cache.get(predicted_key) is None:
                    compressed_data = self.l2_cache[predicted_key]
                    value = pickle.loads(compressed_data)
                    self.l1_cache.put(predicted_key, value)
                    
                    self.logger.debug(f"Predictively loaded {predicted_key}")
                    
        except Exception as e:
            self.logger.debug(f"Predictive loading failed: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        
        l1_hit_rate = self.l1_hits / max(self.total_requests, 1)
        l2_hit_rate = self.l2_hits / max(self.total_requests, 1)
        overall_hit_rate = (self.l1_hits + self.l2_hits) / max(self.total_requests, 1)
        
        return {
            'l1_cache': l1_stats,
            'l2_cache_size': len(self.l2_cache),
            'l1_hit_rate': l1_hit_rate,
            'l2_hit_rate': l2_hit_rate,
            'overall_hit_rate': overall_hit_rate,
            'total_requests': self.total_requests,
            'predictive_patterns': len(self.predictive_cache.prediction_patterns)
        }
    
    def clear_all(self) -> None:
        """Clear all cache levels."""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.predictive_cache.request_history.clear()
        self.predictive_cache.prediction_patterns.clear()


class CacheOptimizer:
    """Main cache optimization coordinator."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.hierarchical_cache = HierarchicalCache(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Performance analytics
        self.optimization_events = []
        
    def get_optimized(self, key: str, compute_func: callable) -> PerformanceMetrics:
        """Get value with optimized caching strategy."""
        # Try cache first
        cached_value = self.hierarchical_cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Cache miss - compute and store
        start_time = time.time()
        computed_value = compute_func()
        computation_time = time.time() - start_time
        
        # Store in cache
        self.hierarchical_cache.put(key, computed_value)
        
        # Record optimization event
        self.optimization_events.append({
            'timestamp': time.time(),
            'key': key,
            'computation_time': computation_time,
            'cache_miss': True
        })
        
        return computed_value
    
    def analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance and suggest optimizations."""
        stats = self.hierarchical_cache.get_comprehensive_stats()
        
        analysis = {
            'performance_stats': stats,
            'optimization_suggestions': []
        }
        
        # Generate optimization suggestions
        if stats['overall_hit_rate'] < 0.5:
            analysis['optimization_suggestions'].append("Low hit rate - consider increasing cache size")
        
        if stats['l2_cache_size'] > stats['l1_cache']['max_size'] * 2:
            analysis['optimization_suggestions'].append("L2 cache growing large - consider cleanup")
        
        if len(self.optimization_events) > 100:
            avg_computation_time = sum(e['computation_time'] for e in self.optimization_events[-100:]) / 100
            if avg_computation_time > 0.1:
                analysis['optimization_suggestions'].append("High computation times - benefit from caching")
        
        return analysis
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.hierarchical_cache.clear_all()
        self.optimization_events.clear()
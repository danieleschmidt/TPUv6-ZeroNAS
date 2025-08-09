"""Advanced caching and memoization for TPUv6-ZeroNAS."""

import logging
import pickle
import hashlib
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import json

from .architecture import Architecture
from .metrics import PerformanceMetrics
from .predictor import TPUv6Predictor


@dataclass 
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    last_accessed: float
    size_bytes: int


class LRUCache:
    """Thread-safe LRU cache with size limits."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_memory = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.hits += 1
                return entry.value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            # Estimate size
            size_bytes = self._estimate_size(value)
            
            # Check if we need to evict
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + size_bytes > self.max_memory_bytes):
                if not self._evict_lru():
                    break
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                last_accessed=time.time(),
                size_bytes=size_bytes
            )
            
            # Remove old entry if exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory -= old_entry.size_bytes
                if key in self.access_order:
                    self.access_order.remove(key)
            
            self.cache[key] = entry
            self.access_order.append(key)
            self.current_memory += size_bytes
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self.access_order:
            return False
        
        lru_key = self.access_order.pop(0)
        entry = self.cache.pop(lru_key)
        self.current_memory -= entry.size_bytes
        self.evictions += 1
        
        return True
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, PerformanceMetrics):
                return 200  # Approximate size
            elif isinstance(value, dict):
                return len(json.dumps(value, default=str).encode())
            else:
                return 100  # Default estimate
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class PredictionCache:
    """Specialized cache for architecture predictions."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path('.cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        self.memory_cache = LRUCache(max_size=500, max_memory_mb=50)
        self.disk_cache_index = self._load_disk_index()
        self.logger = logging.getLogger(__name__)
    
    def get_prediction(self, architecture: Architecture) -> Optional[PerformanceMetrics]:
        """Get cached prediction for architecture."""
        cache_key = self._generate_architecture_key(architecture)
        
        # Try memory cache first
        result = self.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        # Try disk cache
        result = self._get_from_disk(cache_key)
        if result is not None:
            # Promote to memory cache
            self.memory_cache.put(cache_key, result)
            return result
        
        return None
    
    def cache_prediction(self, architecture: Architecture, metrics: PerformanceMetrics) -> None:
        """Cache prediction for architecture."""
        cache_key = self._generate_architecture_key(architecture)
        
        # Cache in memory
        self.memory_cache.put(cache_key, metrics)
        
        # Cache on disk for persistence
        self._save_to_disk(cache_key, metrics, architecture)
    
    def _generate_architecture_key(self, architecture: Architecture) -> str:
        """Generate unique key for architecture."""
        # Create a deterministic hash of architecture properties
        arch_dict = {
            'input_shape': architecture.input_shape,
            'num_classes': architecture.num_classes,
            'layers': []
        }
        
        for layer in architecture.layers:
            layer_dict = {
                'type': layer.layer_type.value,
                'input_channels': layer.input_channels,
                'output_channels': layer.output_channels,
                'kernel_size': layer.kernel_size,
                'stride': layer.stride,
                'activation': layer.activation.value if layer.activation else None
            }
            arch_dict['layers'].append(layer_dict)
        
        # Generate hash
        arch_str = json.dumps(arch_dict, sort_keys=True)
        return hashlib.md5(arch_str.encode()).hexdigest()
    
    def _get_from_disk(self, cache_key: str) -> Optional[PerformanceMetrics]:
        """Get prediction from disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                return data['metrics']
            except Exception as e:
                self.logger.warning(f"Failed to load from disk cache: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def _save_to_disk(self, cache_key: str, metrics: PerformanceMetrics, architecture: Architecture) -> None:
        """Save prediction to disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            data = {
                'key': cache_key,
                'metrics': metrics,
                'timestamp': time.time(),
                'architecture_summary': {
                    'total_params': architecture.total_params,
                    'total_ops': architecture.total_ops,
                    'depth': architecture.depth,
                    'memory_mb': architecture.memory_mb
                }
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
            # Update disk index
            self.disk_cache_index[cache_key] = {
                'file': str(cache_file),
                'timestamp': data['timestamp']
            }
            self._save_disk_index()
            
        except Exception as e:
            self.logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_disk_index(self) -> Dict[str, Any]:
        """Load disk cache index."""
        index_file = self.cache_dir / 'cache_index.json'
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {e}")
        
        return {}
    
    def _save_disk_index(self) -> None:
        """Save disk cache index."""
        index_file = self.cache_dir / 'cache_index.json'
        
        try:
            with open(index_file, 'w') as f:
                json.dump(self.disk_cache_index, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache index: {e}")
    
    def cleanup_disk_cache(self, max_age_days: int = 7) -> None:
        """Clean up old disk cache entries."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        removed_count = 0
        
        for cache_key, info in list(self.disk_cache_index.items()):
            if info['timestamp'] < cutoff_time:
                cache_file = Path(info['file'])
                if cache_file.exists():
                    cache_file.unlink()
                
                del self.disk_cache_index[cache_key]
                removed_count += 1
        
        if removed_count > 0:
            self._save_disk_index()
            self.logger.info(f"Cleaned up {removed_count} old cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        try:
            disk_size = sum(
                Path(info['file']).stat().st_size 
                for info in self.disk_cache_index.values() 
                if Path(info['file']).exists()
            )
        except (OSError, KeyError, TypeError):
            disk_size = 0
        
        return {
            'memory_cache': memory_stats,
            'disk_cache': {
                'entries': len(self.disk_cache_index),
                'size_mb': disk_size / (1024 * 1024),
                'directory': str(self.cache_dir)
            }
        }
    
    def clear_all_caches(self) -> None:
        """Clear both memory and disk caches."""
        self.memory_cache.clear()
        
        # Clear disk cache
        for info in self.disk_cache_index.values():
            cache_file = Path(info['file'])
            if cache_file.exists():
                cache_file.unlink()
        
        self.disk_cache_index.clear()
        self._save_disk_index()
        
        self.logger.info("All caches cleared")


class CachedPredictor:
    """Predictor wrapper with intelligent caching."""
    
    def __init__(self, predictor: TPUv6Predictor, cache_dir: Optional[Path] = None):
        self.predictor = predictor
        self.cache = PredictionCache(cache_dir)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.prediction_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def predict(self, architecture: Architecture) -> PerformanceMetrics:
        """Predict with caching."""
        start_time = time.time()
        
        # Try cache first
        cached_result = self.cache.get_prediction(architecture)
        if cached_result is not None:
            self.cache_hits += 1
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            return cached_result
        
        # Cache miss - compute prediction
        self.cache_misses += 1
        metrics = self.predictor.predict(architecture)
        
        # Cache the result
        self.cache.cache_prediction(architecture, metrics)
        
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        return metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get prediction performance statistics."""
        cache_stats = self.cache.get_cache_stats()
        
        avg_prediction_time = (
            sum(self.prediction_times) / max(len(self.prediction_times), 1)
        )
        
        return {
            'cache_stats': cache_stats,
            'predictions_made': len(self.prediction_times),
            'avg_prediction_time': avg_prediction_time,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        }
    
    def periodic_cleanup(self) -> None:
        """Perform periodic cache cleanup."""
        self.cache.cleanup_disk_cache(max_age_days=7)
    
    def warm_cache(self, architectures: List[Architecture]) -> None:
        """Warm up the cache with predictions for given architectures."""
        self.logger.info(f"Warming cache with {len(architectures)} architectures")
        
        for arch in architectures:
            if self.cache.get_prediction(arch) is None:
                self.predict(arch)
        
        self.logger.info("Cache warming completed")


def create_cached_predictor(predictor: TPUv6Predictor, cache_dir: Optional[Path] = None) -> CachedPredictor:
    """Factory function to create a cached predictor."""
    return CachedPredictor(predictor, cache_dir)
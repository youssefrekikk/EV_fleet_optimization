import os
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from functools import wraps
import threading
import gc

class CacheManager:
    """High-performance cache manager with memory optimization."""
    
    def __init__(self, cache_dir: str = "data/cache", max_memory_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_mb = max_memory_mb
        self.memory_cache = {}
        self.cache_metadata = {}
        self._lock = threading.Lock()
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        metadata_file = self.cache_dir / "metadata.pkl"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    self.cache_metadata = pickle.load(f)
            except:
                self.cache_metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        metadata_file = self.cache_dir / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.cache_metadata, f)
    
    def _get_cache_key(self, key: str, version: str = "v1") -> str:
        """Generate cache key with versioning."""
        return f"{key}_{version}"
    
    def _get_file_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        total_memory = 0.0
        for obj in self.memory_cache.values():
            try:
                if hasattr(obj, 'memory_usage'):
                    memory = obj.memory_usage(deep=True)
                    if hasattr(memory, 'sum'):  # pandas Series
                        total_memory += float(memory.sum()) / 1024 / 1024
                    else:
                        total_memory += float(memory) / 1024 / 1024
                elif isinstance(obj, pd.DataFrame):
                    memory = obj.memory_usage(deep=True)
                    total_memory += float(memory.sum()) / 1024 / 1024
                elif isinstance(obj, np.ndarray):
                    total_memory += float(obj.nbytes) / 1024 / 1024
                else:
                    # Rough estimate
                    total_memory += len(str(obj)) / 1024 / 1024
            except Exception:
                # Fallback estimate
                total_memory += len(str(obj)) / 1024 / 1024
        return float(total_memory)
    
    def _evict_if_needed(self):
        """Evict least recently used items if memory limit exceeded."""
        current_memory = self._get_memory_usage_mb()
        if current_memory > self.max_memory_mb:
            # Sort by last access time
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: self.cache_metadata.get(x[0], {}).get('last_access', 0)
            )
            
            # Evict oldest items until under limit
            for key, _ in sorted_items:
                if self._get_memory_usage_mb() <= self.max_memory_mb * 0.8:
                    break
                self._evict_from_memory(key)
    
    def _evict_from_memory(self, key: str):
        """Evict item from memory cache."""
        if key in self.memory_cache:
            del self.memory_cache[key]
        if key in self.cache_metadata:
            del self.cache_metadata[key]
        gc.collect()
    
    def get(self, key: str, version: str = "v1", max_age_seconds: int = 3600) -> Optional[Any]:
        """Get item from cache."""
        cache_key = self._get_cache_key(key, version)
        
        with self._lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.cache_metadata[cache_key]['last_access'] = time.time()
                return self.memory_cache[cache_key]
            
            # Check disk cache
            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                try:
                    # Check if cache is still valid
                    if cache_key in self.cache_metadata:
                        metadata = self.cache_metadata[cache_key]
                        if time.time() - metadata.get('created_at', 0) > max_age_seconds:
                            # Cache expired
                            file_path.unlink()
                            del self.cache_metadata[cache_key]
                            return None
                    
                    # Load from disk
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Store in memory cache
                    self.memory_cache[cache_key] = data
                    self.cache_metadata[cache_key] = {
                        'created_at': time.time(),
                        'last_access': time.time(),
                        'size_mb': self._estimate_size_mb(data)
                    }
                    
                    self._evict_if_needed()
                    return data
                    
                except Exception:
                    # Corrupted cache file
                    if file_path.exists():
                        file_path.unlink()
                    if cache_key in self.cache_metadata:
                        del self.cache_metadata[cache_key]
        
        return None
    
    def set(self, key: str, value: Any, version: str = "v1", persist: bool = True):
        """Set item in cache."""
        cache_key = self._get_cache_key(key, version)
        
        with self._lock:
            # Store in memory
            self.memory_cache[cache_key] = value
            self.cache_metadata[cache_key] = {
                'created_at': time.time(),
                'last_access': time.time(),
                'size_mb': self._estimate_size_mb(value)
            }
            
            # Persist to disk if requested
            if persist:
                file_path = self._get_file_path(cache_key)
                try:
                    with open(file_path, 'wb') as f:
                        pickle.dump(value, f)
                except Exception as e:
                    print(f"Failed to persist cache {cache_key}: {e}")
            
            self._evict_if_needed()
            self._save_metadata()
    
    def _estimate_size_mb(self, obj: Any) -> float:
        """Estimate object size in MB."""
        try:
            if hasattr(obj, 'memory_usage'):
                memory = obj.memory_usage(deep=True)
                if hasattr(memory, 'sum'):  # pandas Series
                    return float(memory.sum()) / 1024 / 1024
                else:
                    return float(memory) / 1024 / 1024
            elif isinstance(obj, pd.DataFrame):
                memory = obj.memory_usage(deep=True)
                return float(memory.sum()) / 1024 / 1024
            elif isinstance(obj, np.ndarray):
                return float(obj.nbytes) / 1024 / 1024
            else:
                return len(str(obj)) / 1024 / 1024
        except Exception:
            return 1.0  # Default estimate
    
    def invalidate(self, key: str, version: str = "v1"):
        """Invalidate cache item."""
        cache_key = self._get_cache_key(key, version)
        
        with self._lock:
            self._evict_from_memory(cache_key)
            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                file_path.unlink()
            self._save_metadata()
    
    def clear(self):
        """Clear all cache."""
        with self._lock:
            self.memory_cache.clear()
            self.cache_metadata.clear()
            
            # Clear disk cache
            for file_path in self.cache_dir.glob("*.pkl"):
                if file_path.name != "metadata.pkl":
                    file_path.unlink()
            
            self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_items = len(self.memory_cache)
            memory_usage = self._get_memory_usage_mb()
            disk_files = len(list(self.cache_dir.glob("*.pkl"))) - 1  # Exclude metadata
            
            return {
                "total_items": total_items,
                "memory_usage_mb": memory_usage,
                "memory_limit_mb": self.max_memory_mb,
                "disk_files": disk_files,
                "cache_dir": str(self.cache_dir)
            }

# Global cache manager instance
cache_manager = CacheManager()

def cached(ttl_seconds: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)    
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            
            # Filter out self parameter and object references
            filtered_args = []
            for arg in args:
                if hasattr(arg, '__class__') and hasattr(arg, '__module__'):
                    # For objects, use class name instead of object reference
                    if arg.__class__.__name__ in ['DataService', 'CacheManager', 'AsyncTaskService']:
                        filtered_args.append(f"{arg.__class__.__name__}")
                    else:
                        filtered_args.append(str(arg))
                else:
                    filtered_args.append(str(arg))
            
            key_parts.extend(filtered_args)
            
            # Filter kwargs to avoid object references
            filtered_kwargs = {}
            for k, v in kwargs.items():
                if hasattr(v, '__class__') and hasattr(v, '__module__'):
                    if v.__class__.__name__ in ['DataService', 'CacheManager', 'AsyncTaskService']:
                        filtered_kwargs[k] = f"{v.__class__.__name__}"
                    else:
                        filtered_kwargs[k] = str(v)
                else:
                    filtered_kwargs[k] = str(v)
            
            key_parts.extend([f"{k}={v}" for k, v in sorted(filtered_kwargs.items())])
            cache_key = "_".join(key_parts)
            
            # Clean cache key to be filesystem safe
            cache_key = "".join(c for c in cache_key if c.isalnum() or c in ('_', '-')).rstrip()
            
            # Try to get from cache
            result = cache_manager.get(cache_key, max_age_seconds=ttl_seconds)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, persist=True)
            return result
        return wrapper
    return decorator

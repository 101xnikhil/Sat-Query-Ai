import os
import time
import threading
from typing import Dict, Any, Optional, Tuple
import numpy as np

class RasterAndEmbeddingCache:
    """
    Thread-safe in-memory cache for satellite tiles, preprocessed rasters,
    and feature embeddings to accelerate repeated queries and multi-step workflows.
    """
    def __init__(self, max_items: int = 128, default_ttl_sec: float = 3600.0):
        self.max_items = max_items
        self.default_ttl_sec = default_ttl_sec
        self._lock = threading.Lock()
        self._store: Dict[str, Dict[str, Any]] = {}

    def _get_key(self, image_path: str, extra: str = "") -> str:
        try:
            mtime = os.path.getmtime(image_path)
        except (OSError, FileNotFoundError):
            mtime = 0.0
        return f"{image_path}:{mtime}:{extra}"

    def get(self, image_path: str, extra: str = "") -> Optional[Any]:
        key = self._get_key(image_path, extra)
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            if time.time() > item["expires_at"]:
                del self._store[key]
                return None
            item["last_accessed"] = time.time()
            return item["data"]

    def put(self, image_path: str, data: Any, extra: str = "", ttl_sec: Optional[float] = None):
        key = self._get_key(image_path, extra)
        ttl = ttl_sec if ttl_sec is not None else self.default_ttl_sec
        with self._lock:
            if len(self._store) >= self.max_items:
                # Evict least recently used item
                oldest_key = min(self._store.keys(), key=lambda k: self._store[k]["last_accessed"])
                del self._store[oldest_key]

            self._store[key] = {
                "data": data,
                "expires_at": time.time() + ttl,
                "last_accessed": time.time()
            }

    def clear(self):
        with self._lock:
            self._store.clear()

_global_cache: Optional[RasterAndEmbeddingCache] = None

def get_raster_cache() -> RasterAndEmbeddingCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = RasterAndEmbeddingCache()
    return _global_cache

"""Disk-based cache for API responses to avoid rate limits."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FileCache:
    """
    Simple file-based cache using pickle for arbitrary Python objects.

    Cache entries expire based on configured TTL (in minutes).
    """

    def __init__(self, cache_dir: str = ".cache") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        hashed = hashlib.md5(key.encode()).hexdigest()
        return self._dir / f"{hashed}.pkl"

    def get(self, key: str, ttl_minutes: float = 60.0) -> Optional[Any]:
        """Return cached value if it exists and hasn't expired, else None."""
        path = self._key_path(key)
        if not path.exists():
            return None
        age_minutes = (time.time() - path.stat().st_mtime) / 60.0
        if age_minutes > ttl_minutes:
            path.unlink(missing_ok=True)
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Cache read error for %s: %s", key, exc)
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        path = self._key_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except Exception as exc:
            logger.warning("Cache write error for %s: %s", key, exc)

    def invalidate(self, key: str) -> None:
        """Remove a specific cache entry."""
        self._key_path(key).unlink(missing_ok=True)

    def clear(self) -> int:
        """Remove all cache entries. Returns count removed."""
        count = 0
        for p in self._dir.glob("*.pkl"):
            p.unlink()
            count += 1
        return count


# Module-level singleton
_cache: Optional[FileCache] = None


def get_cache(cache_dir: str = ".cache") -> FileCache:
    """Return module-level cache singleton."""
    global _cache
    if _cache is None:
        _cache = FileCache(cache_dir)
    return _cache

"""
TTL and LRU Memoization with Advanced Patterns
================================================
Production-grade caching infrastructure:
  - TTL-based memoization with stale-while-revalidate (background refresh)
  - LRU memoization with configurable max size
  - In-flight request deduplication for concurrent calls
  - Thread-safe cache with clear/invalidate API
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')


# 鈹€鈹€ TTL Cache 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class _TTLCacheEntry:
    __slots__ = ('value', 'timestamp', 'refreshing')

    def __init__(self, value: Any, timestamp: float):
        self.value = value
        self.timestamp = timestamp
        self.refreshing = False


def memoize_with_ttl(
    ttl_seconds: float = 300.0,
    key_fn: Optional[Callable[..., str]] = None,
    stale_while_revalidate: bool = True,
) -> Callable:
    """
    TTL-based memoization decorator with optional background refresh.

    When stale_while_revalidate=True (default), returns stale value immediately
    while refreshing in a background thread. This prevents cache stampedes.

    Migrated from: upstream/src/utils/memoize.ts :: memoizeWithTTL

    Usage:
        @memoize_with_ttl(ttl_seconds=60)
        def get_config():
            return load_config_from_disk()

        # First call: computes and caches
        config = get_config()

        # Within 60s: returns cached value
        config = get_config()

        # After 60s with stale_while_revalidate:
        # Returns old value immediately, refreshes in background
        config = get_config()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[str, _TTLCacheEntry] = {}
        lock = threading.Lock()

        def _make_key(*args, **kwargs) -> str:
            if key_fn:
                return key_fn(*args, **kwargs)
            return repr((args, sorted(kwargs.items())))

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            key = _make_key(*args, **kwargs)
            now = time.time()

            with lock:
                entry = cache.get(key)

            if entry is None:
                # Cache miss 鈥?compute synchronously
                value = func(*args, **kwargs)
                with lock:
                    cache[key] = _TTLCacheEntry(value, now)
                return value

            age = now - entry.timestamp

            if age <= ttl_seconds:
                # Fresh 鈥?return cached
                return entry.value

            if stale_while_revalidate and not entry.refreshing:
                # Stale but usable 鈥?return old value, refresh in background
                entry.refreshing = True

                def _refresh():
                    try:
                        new_value = func(*args, **kwargs)
                        with lock:
                            cache[key] = _TTLCacheEntry(new_value, time.time())
                    except Exception as e:
                        logger.debug("Background refresh of %s failed: %s",
                                     func.__name__, e)
                        with lock:
                            # Remove stale entry on refresh failure
                            if cache.get(key) is entry:
                                del cache[key]

                t = threading.Thread(target=_refresh, daemon=True)
                t.start()
                return entry.value

            # Stale, no revalidate or already refreshing 鈥?recompute
            value = func(*args, **kwargs)
            with lock:
                cache[key] = _TTLCacheEntry(value, now)
            return value

        # Cache management API
        wrapper.cache_clear = lambda: cache.clear()  # type: ignore[attr-defined]
        wrapper.cache_size = lambda: len(cache)  # type: ignore[attr-defined]
        wrapper.cache_invalidate = lambda *a, **kw: cache.pop(  # type: ignore[attr-defined]
            _make_key(*a, **kw), None)

        return wrapper
    return decorator


# 鈹€鈹€ LRU Cache with Configurable Size 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class LRUCache(Generic[K, T]):
    """
    Thread-safe LRU cache with configurable max size.

    Uses OrderedDict for O(1) access and eviction.

    Migrated from: upstream/src/utils/memoize.ts :: memoizeWithLRU
    """

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[K, T] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: K, default: Optional[T] = None) -> Optional[T]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return default

    def peek(self, key: K, default: Optional[T] = None) -> Optional[T]:
        """Get without updating recency (for inspection)."""
        with self._lock:
            return self._cache.get(key, default)

    def put(self, key: K, value: T) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __contains__(self, key: K) -> bool:
        return key in self._cache


def memoize_with_lru(
    max_size: int = 100,
    key_fn: Optional[Callable[..., str]] = None,
) -> Callable:
    """
    LRU memoization decorator with configurable max cache size.

    Prevents unbounded memory growth unlike functools.lru_cache(maxsize=None).

    Migrated from: upstream/src/utils/memoize.ts :: memoizeWithLRU

    Usage:
        @memoize_with_lru(max_size=50)
        def parse_config(path):
            return json.load(open(path))
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = LRUCache[str, T](max_size=max_size)

        def _make_key(*args, **kwargs) -> str:
            if key_fn:
                return key_fn(*args, **kwargs)
            return repr((args, sorted(kwargs.items())))

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            key = _make_key(*args, **kwargs)
            cached = cache.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper
    return decorator


# 鈹€鈹€ In-Flight Deduplication 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class InflightDedup(Generic[T]):
    """
    Deduplicate concurrent calls to the same function with the same arguments.

    If multiple threads call the same function with the same key simultaneously,
    only one actual execution occurs. Other threads wait for and share the result.

    Migrated from: upstream/src/utils/memoize.ts :: memoizeWithTTLAsync (inFlight map)

    Usage:
        dedup = InflightDedup()

        def fetch(url):
            # Even if 10 threads call this simultaneously with same URL,
            # only 1 actual fetch occurs
            return dedup.call(url, lambda: urllib.request.urlopen(url).read())
    """

    def __init__(self):
        self._inflight: Dict[str, threading.Event] = {}
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, Exception] = {}
        self._lock = threading.Lock()

    def call(self, key: str, fn: Callable[[], T]) -> T:
        """
        Execute fn() if no in-flight call for this key exists.
        Otherwise, wait for and return the in-flight result.
        """
        with self._lock:
            if key in self._inflight:
                event = self._inflight[key]
            else:
                event = threading.Event()
                self._inflight[key] = event
                # We are the executor
                try:
                    result = fn()
                    self._results[key] = result
                except Exception as e:
                    self._errors[key] = e
                finally:
                    event.set()
                    with self._lock:
                        self._inflight.pop(key, None)

                if key in self._errors:
                    err = self._errors.pop(key)
                    raise err
                return self._results.pop(key)

        # We are a waiter
        event.wait(timeout=300)  # 5 min max wait

        if key in self._errors:
            raise self._errors.pop(key)
        if key in self._results:
            return self._results.pop(key)

        raise TimeoutError(f"In-flight dedup timeout for key: {key}")


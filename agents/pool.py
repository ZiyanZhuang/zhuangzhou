"""
API Key Pool — Multi-Key Round-Robin Manager
============================================
Rationale from research:
  - DeepSeek rate limits are ACCOUNT-level (not key-level): multiple keys
    from the same account share the same 200-500 RPM bucket.
  - Multiple keys are still useful for: organization (per-agent tracking),
    graceful degradation when a key is revoked, and mixing keys across
    different accounts.
  - This pool implements round-robin with per-key error tracking so a
    bad/expired key is automatically sidelined.
"""

import logging
import os
import hashlib
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class KeySlot:
    def __init__(self, key: str, label: str = "", index: int = 0):
        self.key = key
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:6]
        self.label = label or f"key#{index + 1}-{digest}"
        self.error_count = 0
        self.last_error_at: float = 0.0
        self.calls_this_minute: int = 0
        self._minute_start: float = time.time()
        self._lock = threading.Lock()

    def record_call(self):
        with self._lock:
            now = time.time()
            if now - self._minute_start >= 60:
                self.calls_this_minute = 0
                self._minute_start = now
            self.calls_this_minute += 1

    def record_error(self):
        with self._lock:
            self.error_count += 1
            self.last_error_at = time.time()

    def is_healthy(self, max_errors: int = 3, cooldown_s: float = 300) -> bool:
        if self.error_count < max_errors:
            return True
        return (time.time() - self.last_error_at) > cooldown_s

    def reset_errors(self):
        with self._lock:
            self.error_count = 0
            self.last_error_at = 0.0

    def __repr__(self):
        return f"KeySlot({self.label}, errors={self.error_count}, calls/min={self.calls_this_minute})"


class APIKeyPool:
    """Round-robin API key pool with per-key health tracking.

    Configuration:
      Pass a list of keys directly, OR set environment variables:
        DEEPSEEK_API_KEYS=key1,key2,key3
        OPENAI_API_KEYS=key1,key2
        ANTHROPIC_API_KEYS=key1

    Usage:
        pool = APIKeyPool.from_env("deepseek")
        key = pool.acquire()           # get next healthy key
        try:
            result = call_api(key)
            pool.release_ok(key)
        except RateLimitError:
            pool.release_error(key)
    """

    def __init__(self, keys: List[str], labels: Optional[List[str]] = None):
        if not keys:
            raise ValueError("APIKeyPool requires at least one key")
        labels = labels or [""] * len(keys)
        self._slots: List[KeySlot] = [
            KeySlot(k, l, i) for i, (k, l) in enumerate(zip(keys, labels))
        ]
        self._index = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, backend: str = "deepseek") -> "APIKeyPool":
        """Load keys from environment variables.

        Checks:
          DEEPSEEK_API_KEYS, OPENAI_API_KEYS, ANTHROPIC_API_KEYS (comma-separated)
          Falls back to single key: DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
        """
        env_map = {
            "deepseek": ("DEEPSEEK_API_KEYS", "DEEPSEEK_API_KEY"),
            "openai":   ("OPENAI_API_KEYS",   "OPENAI_API_KEY"),
            "claude":   ("ANTHROPIC_API_KEYS", "ANTHROPIC_API_KEY"),
            "qwen":     ("QWEN_API_KEYS",      "QWEN_API_KEY"),
        }
        multi_var, single_var = env_map.get(backend, (None, None))

        keys = []
        if multi_var:
            raw = os.environ.get(multi_var, "")
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys and single_var:
            single = os.environ.get(single_var, "").strip()
            if single:
                keys = [single]
        if not keys:
            raise RuntimeError(
                f"No API keys found for '{backend}'. "
                f"Set {multi_var} or {single_var} in environment."
            )
        return cls(keys)

    @classmethod
    def from_keys(cls, keys: List[str]) -> "APIKeyPool":
        return cls(keys)

    def acquire(self) -> str:
        """Get next healthy key (round-robin). Raises if all unhealthy."""
        with self._lock:
            n = len(self._slots)
            for _ in range(n):
                slot = self._slots[self._index % n]
                self._index += 1
                if slot.is_healthy():
                    slot.record_call()
                    return slot.key
            # All keys unhealthy — reset and try again
            logger.warning("All API keys unhealthy, resetting error counts")
            for s in self._slots:
                s.reset_errors()
            slot = self._slots[0]
            slot.record_call()
            return slot.key

    def release_ok(self, key: str):
        """Call after a successful API call."""
        pass  # no-op for now; could track success metrics

    def release_error(self, key: str):
        """Call after a failed API call (rate limit, auth error, etc.)."""
        for slot in self._slots:
            if slot.key == key:
                slot.record_error()
                logger.debug("Key %s error count now %d", slot.label, slot.error_count)
                return

    def stats(self) -> List[Dict]:
        return [
            {
                "label": s.label,
                "healthy": s.is_healthy(),
                "errors": s.error_count,
                "calls_this_minute": s.calls_this_minute,
            }
            for s in self._slots
        ]

    def __len__(self):
        return len(self._slots)

    def __repr__(self):
        healthy = sum(1 for s in self._slots if s.is_healthy())
        return f"APIKeyPool({healthy}/{len(self._slots)} healthy)"

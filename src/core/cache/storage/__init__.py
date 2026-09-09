"""组装 SafeFullCache，保持向后兼容。"""

import time
from typing import Any, Callable, Dict, Optional

from .backend import CacheBackend
from .eviction import MinExpiryPolicy
from .stats import CacheStats


class SafeFullCache:
    """安全的全量缓存。

    用于缓存 LLM 查询结果与检索策略生成的查询（MQE 扩展 / HyDE 假设文档），
    相同输入直接命中缓存，避免重复调用 LLM。
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600,
                 cleanup_interval: int = 100):
        self._backend = CacheBackend(max_size)
        self._eviction = MinExpiryPolicy()
        self._stats = CacheStats()
        self._expiry: Dict[str, float] = {}
        self._default_ttl = max(1, default_ttl)
        self._cleanup_interval = max(1, cleanup_interval)
        self._ops_since_cleanup = 0

    def get(self, key: str) -> Optional[Any]:
        value = self._backend.get(key)
        if value is not None:
            if time.time() < self._expiry.get(key, 0):
                self._stats.record_hit()
                return value
            self._delete(key)
        self._stats.record_miss()
        return None

    def set(self, key: str, value: Any, ttl: int = None):
        if key not in self._backend.keys and self._backend.is_full:
            self._evict_one()
        self._backend.set(key, value)
        self._expiry[key] = time.time() + (ttl or self._default_ttl)
        self._ops_since_cleanup += 1
        if self._ops_since_cleanup >= self._cleanup_interval:
            self._cleanup()

    def get_or_set(self, key: str, compute: Callable, ttl: int = None) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.set(key, value, ttl=ttl)
        return value

    def update(self, key: str, value: Any):
        if key in self._backend.keys:
            self._backend.set(key, value)
            self._expiry[key] = time.time() + self._default_ttl

    def clear(self):
        self._backend.clear()
        self._expiry.clear()
        self._ops_since_cleanup = 0
        self._stats.reset()

    def _delete(self, key: str):
        self._backend.delete(key)
        self._expiry.pop(key, None)

    def _evict_one(self):
        key = self._eviction.select(list(self._backend.keys), self._expiry)
        if key:
            self._delete(key)

    def _cleanup(self):
        now = time.time()
        expired = [k for k, v in self._expiry.items() if v < now]
        for k in expired:
            self._delete(k)
        self._ops_since_cleanup = 0

    def stats(self) -> Dict[str, Any]:
        expired_count = len([k for k, v in self._expiry.items() if time.time() > v])
        return self._stats.snapshot(self._backend.size, self._backend.max_size, expired_count)
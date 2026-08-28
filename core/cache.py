"""安全的全量缓存（SafeFullCache）。

通用缓存层，支持 TTL 淘汰、容量限制、一致性更新。
原位于 tools/rag/rag_tool.py，因 core/ 层也需要（practiceLLM 缓存）
提升到 core/ 避免循环依赖。
"""

import time
from typing import Any, Dict, Optional, Callable


class SafeFullCache:
    """安全的全量缓存。

    特点：
      1. 自动淘汰冷数据（TTL）
      2. 限制最大容量（内存保护）
      3. 过期数据自动清理
      4. 支持数据更新（一致性）

    用于缓存 LLM 查询结果与检索策略生成的查询（MQE 扩展 / HyDE 假设文档），
    相同输入直接命中缓存，避免重复调用 LLM。
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600,
                 cleanup_interval: int = 100):
        self.cache: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
        self.max_size = max(1, max_size)
        self.default_ttl = max(1, default_ttl)
        self.cleanup_interval = max(1, cleanup_interval)
        self.ops_since_cleanup = 0
        self.hits = 0
        self.misses = 0

    # --- 读写 ---

    def get(self, key: str) -> Optional[Any]:
        """读取缓存；未命中或已过期返回 None。"""
        if key in self.cache:
            if time.time() < self.expiry[key]:
                self.hits += 1
                return self.cache[key]
            self._delete(key)
            self.misses += 1
            return None
        self.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int = None):
        """写入缓存；容量满时淘汰最旧条目。"""
        if key not in self.cache and len(self.cache) >= self.max_size:
            self._evict_one()
        self.cache[key] = value
        self.expiry[key] = time.time() + (ttl or self.default_ttl)
        self.ops_since_cleanup += 1
        if self.ops_since_cleanup >= self.cleanup_interval:
            self._cleanup()

    def get_or_set(self, key: str, compute: Callable, ttl: int = None) -> Any:
        """读取缓存，未命中时用 compute() 计算并写入。"""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.set(key, value, ttl=ttl)
        return value

    def update(self, key: str, value: Any):
        """更新缓存（保持一致性），刷新 TTL。"""
        if key in self.cache:
            self.cache[key] = value
            self.expiry[key] = time.time() + self.default_ttl

    def clear(self):
        """清空全部缓存（知识库变更时调用，保证一致性）。"""
        self.cache.clear()
        self.expiry.clear()
        self.ops_since_cleanup = 0

    # --- 内部 ---

    def _delete(self, key: str):
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]

    def _evict_one(self):
        """淘汰一个条目（简单策略：删除最早过期者）。"""
        if self.cache:
            oldest = min(self.expiry, key=self.expiry.get)
            self._delete(oldest)

    def _cleanup(self):
        """清理所有过期条目。"""
        now = time.time()
        expired = [k for k, v in self.expiry.items() if v < now]
        for k in expired:
            self._delete(k)
        self.ops_since_cleanup = 0

    def stats(self) -> Dict[str, Any]:
        """缓存统计。"""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "hits": self.hits,
            "misses": self.misses,
            "expired_count": len([k for k, v in self.expiry.items() if time.time() > v]),
        }
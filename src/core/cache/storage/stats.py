"""缓存命中/未命中统计。"""


class CacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1

    def reset(self):
        self.hits = 0
        self.misses = 0

    def snapshot(self, size: int, max_size: int, expired_count: int) -> dict:
        total = self.hits + self.misses
        return {
            "size": size,
            "max_size": max_size,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "hits": self.hits,
            "misses": self.misses,
            "expired_count": expired_count,
        }
"""基于字典的键值存储后端，不包含策略逻辑。"""

from typing import Any, Dict, Optional


class CacheBackend:
    """纯字典 CRUD，不感知 TTL 或淘汰策略。"""

    def __init__(self, max_size: int):
        self._data: Dict[str, Any] = {}
        self._max_size = max(1, max_size)

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def delete(self, key: str):
        self._data.pop(key, None)

    def clear(self):
        self._data.clear()

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def is_full(self) -> bool:
        return len(self._data) >= self._max_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def keys(self):
        return self._data.keys()
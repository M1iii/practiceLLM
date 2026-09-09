"""可插拔的淘汰策略。"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class EvictionPolicy(ABC):
    """淘汰策略接口。"""

    @abstractmethod
    def select(self, keys: List[str], expiry: Dict[str, float]) -> Optional[str]:
        ...


class MinExpiryPolicy(EvictionPolicy):
    """淘汰最早过期的条目。"""

    def select(self, keys: List[str], expiry: Dict[str, float]) -> Optional[str]:
        if not keys:
            return None
        return min(keys, key=lambda k: expiry.get(k, float("inf")))
"""缓存模块。提供 SafeFullCache —— 基于字典的 TTL 缓存后端。"""

from .storage import SafeFullCache

__all__ = ["SafeFullCache"]
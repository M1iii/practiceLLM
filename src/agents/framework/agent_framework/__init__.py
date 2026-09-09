"""Agent 统一执行框架：BaseAgent + AgentRegistry + AgentFactory + AgentAdapter。"""

from .models import Message, AgentMeta, AgentResult
from .base import BaseAgent
from .adapter import AgentAdapter
from .registry import AgentRegistry, AgentFactory

__all__ = [
    "Message", "AgentMeta", "AgentResult",
    "BaseAgent",
    "AgentAdapter",
    "AgentRegistry", "AgentFactory",
]
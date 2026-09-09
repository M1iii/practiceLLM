"""数据模型：Message, AgentMeta, AgentResult。"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class Message:
    """对话消息类，封装角色和内容（统一消息模型）。"""
    content: str
    role: str


@dataclass
class AgentMeta:
    """Agent 元信息。"""
    name: str
    description: str
    version: str
    agent_type: str
    model: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class AgentResult:
    """统一执行结果。"""
    output: str
    metadata: AgentMeta
    status: str = "success"
    iterations: int = 0
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
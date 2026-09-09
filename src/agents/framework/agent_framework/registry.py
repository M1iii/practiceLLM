"""AgentRegistry（实例注册表）+ AgentFactory（类型工厂）。"""

from typing import Dict, List, Optional, Any, Type

from .base import BaseAgent
from .models import AgentMeta, AgentResult


class AgentRegistry:
    """Agent 实例注册表：按名称注册/查询/执行 Agent 实例。"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent, name: Optional[str] = None) -> str:
        key = name or agent.name
        self._agents[key] = agent
        return key

    def unregister(self, name: str) -> bool:
        return self._agents.pop(name, None) is not None

    def get(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def list_metadata(self) -> List[AgentMeta]:
        return [a.metadata() for a in self._agents.values()]

    def execute(self, name: str, input_text: str, **kwargs) -> AgentResult:
        agent = self._agents.get(name)
        if agent is None:
            return AgentResult(
                output="",
                metadata=AgentMeta(name=name, description="未注册的 Agent",
                                   version="-", agent_type="unknown"),
                status="error",
                error=f"未注册的 Agent: {name}（可用: {', '.join(self._agents)}）",
            )
        return agent.execute(input_text, **kwargs)

    def count(self) -> int:
        return len(self._agents)

    def clear(self):
        self._agents.clear()


class AgentFactory:
    """Agent 工厂：根据类型创建不同类型的 Agent。"""

    def __init__(self):
        self._types: Dict[str, Type[BaseAgent]] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, agent_type: str, agent_class: Type[BaseAgent],
                 description: str = "") -> None:
        self._types[agent_type] = agent_class
        self._descriptions[agent_type] = description or getattr(
            agent_class, "DESCRIPTION", "")

    def unregister(self, agent_type: str) -> bool:
        self._descriptions.pop(agent_type, None)
        return self._types.pop(agent_type, None) is not None

    def get_class(self, agent_type: str) -> Optional[Type[BaseAgent]]:
        return self._types.get(agent_type)

    def available_types(self) -> Dict[str, str]:
        return dict(self._descriptions)

    def create(self, agent_type: str, **kwargs) -> BaseAgent:
        agent_class = self._types.get(agent_type)
        if agent_class is None:
            raise KeyError(
                f"未知的 Agent 类型 '{agent_type}'，可用: {', '.join(self._types) or '无'}")
        return agent_class(**kwargs)

    def create_all(self, configs: List[Dict[str, Any]]) -> List[BaseAgent]:
        return [self.create(c.pop("agent_type"), **c) for c in configs]
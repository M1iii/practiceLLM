"""AgentAdapter：适配器，将已有 Agent（run/chat 接口）包装为 BaseAgent。"""

from typing import Any, Optional, List

from .base import BaseAgent
from .models import Message, AgentMeta


class AgentAdapter(BaseAgent):
    """适配器：包装具有 run()/chat() 接口的已有 Agent。"""

    AGENT_TYPE = "adapter"

    def __init__(self, agent: Any, name: Optional[str] = None,
                 agent_type: str = "adapter",
                 description: str = "适配器包装的 Agent",
                 version: str = "1.0.0",
                 session_store: Optional[Any] = None,
                 session_id: Optional[str] = None):
        super().__init__(name=name, session_store=session_store,
                         session_id=session_id)
        self._agent = agent
        self._agent_type = agent_type
        self._description = description
        self._version = version
        if session_store is not None and session_id is not None:
            self._restore_wrapped_history()

    def _restore_wrapped_history(self):
        try:
            history = self.restore_history()
            if not history or not hasattr(self._agent, "_history"):
                return
            simplified = [{"role": m["role"], "content": m["content"]}
                          for m in history if m["role"] in ("user", "assistant")]
            if not simplified:
                return

            history_list = self._agent._history
            if not hasattr(history_list, "append"):
                return

            if history_list and hasattr(history_list[0], "role"):
                restored = [Message(content=m["content"], role=m["role"])
                            for m in simplified]
            else:
                restored = simplified
            history_list.clear()
            history_list.extend(restored)
        except Exception:
            pass

    def _execute(self, input_text: str, **kwargs) -> str:
        if hasattr(self._agent, "run"):
            return self._agent.run(input_text, **kwargs)
        if hasattr(self._agent, "chat"):
            return self._agent.chat(input_text, **kwargs)
        raise AttributeError("被包装的 Agent 需要实现 run() 或 chat() 方法")

    def capabilities(self) -> List[str]:
        caps = ["执行"]
        if hasattr(self._agent, "run") and hasattr(self._agent, "tool_registry"):
            caps.append("工具调用")
        if hasattr(self._agent, "_history"):
            caps.append("多轮对话")
        return caps

    def metadata(self) -> AgentMeta:
        meta = super().metadata()
        meta.agent_type = self._agent_type
        meta.description = self._description
        meta.version = self._version
        return meta
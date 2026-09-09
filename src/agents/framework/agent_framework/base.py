"""BaseAgent：所有 Agent 的统一执行接口。"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .models import AgentMeta, AgentResult


class BaseAgent(ABC):
    """所有 Agent 的统一执行接口。
    子类需实现 _execute()（主逻辑）并可覆盖 metadata()/capabilities()。
    execute() 负责异常包装，保证主逻辑不崩溃并返回统一结果。
    """

    AGENT_TYPE = "base"
    DESCRIPTION = "通用 Agent 基类"
    VERSION = "1.0.0"

    def __init__(self, name: Optional[str] = None, tracer: Optional[Any] = None,
                 session_store: Optional[Any] = None,
                 session_id: Optional[str] = None, **kwargs):
        self.name = name or self.__class__.__name__
        self._extra = dict(kwargs)
        self._tracer = tracer
        self.session_store = session_store
        self.session_id = session_id
        self._auto_memory = kwargs.get("auto_memory", True)
        self._memory_top_k = kwargs.get("memory_top_k", 3)
        self._memory_threshold = kwargs.get("memory_threshold", 0.5)
        self.memory_engine = None
        if self._auto_memory:
            self._init_memory_engine()

    def _init_memory_engine(self):
        """惰性初始化记忆引擎。"""
        try:
            from src.agents.framework.memory_engine import MemoryEngine
            self.memory_engine = MemoryEngine(
                enable=True,
                top_k=self._memory_top_k,
                similarity_threshold=self._memory_threshold,
            )
        except Exception as e:
            logger.warning("记忆引擎初始化失败（静默降级）: %s", e)
            self.memory_engine = None

    @abstractmethod
    def _execute(self, input_text: str, **kwargs) -> str:
        ...

    def execute(self, input_text: str, **kwargs) -> AgentResult:
        """执行 Agent 主逻辑，返回统一 AgentResult（容错包装 + 自动轨迹 + 自动记忆）。"""
        logger = self._tracer or self._get_default_tracer()
        trace_id = None
        if logger is not None:
            trace_id = logger.start(self.name, self.AGENT_TYPE, input_text)

        enhanced_input = input_text
        if self._auto_memory and self.memory_engine is not None:
            try:
                memory_context = self.memory_engine.retrieve(
                    input_text, top_k=self._memory_top_k)
                if memory_context:
                    enhanced_input = f"{memory_context}\n\n{input_text}"
                    logger.info("MemoryEngine: 注入 %d 条记忆到上下文",
                                len(memory_context.split("\n")))
            except Exception:
                pass

        status, error, output = "success", None, None
        try:
            if logger is not None and trace_id is not None:
                with logger.active(trace_id):
                    output = self._execute(enhanced_input, **kwargs)
            else:
                output = self._execute(enhanced_input, **kwargs)
        except Exception as e:
            status, error, output = "error", f"{type(e).__name__}: {e}", ""
        finally:
            if logger is not None and trace_id is not None:
                logger.finish(trace_id, status=status, error=error)

        self._persist_exchange(input_text, output or "", status, error)
        self._auto_store_memory(input_text, output or "", status)

        return AgentResult(
            output=output or "",
            metadata=self.metadata(),
            status=status,
            iterations=int(kwargs.get("iterations", 0)),
            error=error,
        )

    def _auto_store_memory(self, question: str, answer: str, status: str = "success"):
        if not self._auto_memory or self.memory_engine is None:
            return
        if status != "success":
            return
        try:
            depth = 0
            store = getattr(self, "session_store", None)
            sid = getattr(self, "session_id", None)
            if store and sid:
                msgs = store.get_messages(sid, limit=10)
                depth = sum(1 for m in msgs[-6:] if m.get("role") == "user")
            self.memory_engine.store(question, answer, depth=depth)
        except Exception:
            pass

    def _persist_exchange(self, user_text: str, assistant_text: str,
                          status: str, error: Optional[str]):
        store = getattr(self, "session_store", None)
        sid = getattr(self, "session_id", None)
        if store is None or not sid:
            return
        try:
            if not store.session_exists(sid):
                store.create_session(sid, agent_type=self.AGENT_TYPE,
                                     meta={"name": self.name})
            store.append_message(sid, "user", user_text)
            extra = {}
            if status != "success":
                extra = {"status": status, "error": error}
            store.append_message(sid, "assistant", assistant_text or
                                 f"[{status}] {error or '无输出'}", extra=extra)
        except Exception:
            pass

    def restore_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        store = getattr(self, "session_store", None)
        sid = getattr(self, "session_id", None)
        if store is None or not sid:
            return []
        try:
            return store.get_messages(sid, limit=limit)
        except Exception:
            return []

    @staticmethod
    def _get_default_tracer():
        try:
            from src.agents.framework.agent_trace import TraceLogger
            return TraceLogger.get_default()
        except ImportError:
            return None

    def stream(self, input_text: str, **kwargs):
        from src.core.streaming import stream_agent
        yield from stream_agent(self, input_text, **kwargs)

    def capabilities(self) -> List[str]:
        return []

    def metadata(self) -> AgentMeta:
        return AgentMeta(
            name=self.name,
            description=self.DESCRIPTION,
            version=self.VERSION,
            agent_type=self.AGENT_TYPE,
            model=getattr(self, "model", None) or self._extra.get("model"),
            tools=self.list_tools() if hasattr(self, "list_tools") else [],
            capabilities=self.capabilities(),
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} type={self.AGENT_TYPE}>"
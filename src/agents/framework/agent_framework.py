"""Agent 统一执行框架：BaseAgent（统一执行接口）+ AgentRegistry + AgentFactory。

设计：
  1. BaseAgent(ABC)：所有 Agent 的统一执行接口
     - execute()：执行 Agent 主逻辑，返回统一 AgentResult（含元信息）
     - metadata()：返回 Agent 元信息（名称/描述/版本/类型/模型/工具/能力）
     - _execute()：抽象方法，由具体 Agent 实现主逻辑
  2. AgentRegistry：Agent 实例注册表
     - 按名称注册/查询/执行 Agent 实例，批量获取元信息
  3. AgentFactory：Agent 类型工厂
     - 类型 → 类映射，创建不同类型 Agent；可注册自定义类型
  4. AgentAdapter：适配器，将已有 Agent（run/chat 接口）包装为 BaseAgent，
     无需修改原 Agent 即可接入统一框架

使用方式:
    factory = AgentFactory()
    agent = factory.create("simple", name="助手A")     # 工厂创建
    registry = AgentRegistry()
    registry.register(agent, "main")                   # 注册实例
    result = registry.execute("main", "你好")          # 统一执行
    print(agent.metadata())
"""

import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Type



# ============================================================
# 元信息与结果数据结构
# ============================================================

@dataclass
class Message:
    """对话消息类，封装角色和内容（统一消息模型）。

    供 Agent 内部维护对话历史使用（如 SimpleAgent._history）；
    与 OpenAI API 的 {"role", "content"} dict 格式可互转。
    """
    content: str
    role: str


@dataclass
class AgentMeta:
    """Agent 元信息。"""
    name: str                                   # 实例名称
    description: str                            # 功能描述
    version: str                                # 版本
    agent_type: str                             # 类型标识（simple/react/...）
    model: Optional[str] = None                 # 使用的模型
    tools: List[str] = field(default_factory=list)     # 可用工具
    capabilities: List[str] = field(default_factory=list)  # 能力标签
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class AgentResult:
    """统一执行结果。"""
    output: str                                 # 主输出
    metadata: AgentMeta                         # Agent 元信息
    status: str = "success"                     # success / error
    iterations: int = 0                         # 迭代次数
    error: Optional[str] = None                 # 错误信息（status=error 时）
    extra: Dict[str, Any] = field(default_factory=dict)  # 附加信息


# ============================================================
# 1) BaseAgent：统一执行接口
# ============================================================

class BaseAgent(ABC):
    """所有 Agent 的统一执行接口。

    子类需实现 _execute()（主逻辑）并可覆盖 metadata()/capabilities()。
    execute() 负责异常包装，保证主逻辑不崩溃并返回统一结果。
    """

    AGENT_TYPE = "base"        # 类型标识（子类覆盖）
    DESCRIPTION = "通用 Agent 基类"
    VERSION = "1.0.0"

    def __init__(self, name: Optional[str] = None, tracer: Optional[Any] = None,
                 session_store: Optional[Any] = None,
                 session_id: Optional[str] = None, **kwargs):
        self.name = name or self.__class__.__name__
        self._extra = dict(kwargs)
        # 轨迹记录器（可观测性）：优先实例级，否则全局默认
        self._tracer = tracer
        # 会话持久化：配置后 execute() 自动保存对话轮次
        self.session_store = session_store
        self.session_id = session_id
        # 自动记忆引擎（检索 + 存储）
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

    # --- 子类实现的主逻辑 ---

    @abstractmethod
    def _execute(self, input_text: str, **kwargs) -> str:
        """Agent 主逻辑（子类实现），返回输出文本。"""
        ...

    # --- 统一执行入口 ---

    def execute(self, input_text: str, **kwargs) -> AgentResult:
        """执行 Agent 主逻辑，返回统一 AgentResult（容错包装 + 自动轨迹 + 自动记忆）。"""
        logger = self._tracer or self._get_default_tracer()
        trace_id = None
        if logger is not None:
            trace_id = logger.start(self.name, self.AGENT_TYPE, input_text)

        # 自动记忆检索：注入相关记忆到上下文
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
                pass  # 记忆检索失败不影响对话

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

        # 会话持久化：自动保存 user/assistant 一轮对话
        self._persist_exchange(input_text, output or "", status, error)

        # 自动记忆存储：将本轮对话存入记忆库
        self._auto_store_memory(input_text, output or "", status)

        return AgentResult(
            output=output or "",
            metadata=self.metadata(),
            status=status,
            iterations=int(kwargs.get("iterations", 0)),
            error=error,
        )

    def _auto_store_memory(self, question: str, answer: str, status: str = "success"):
        """自动将本轮对话存入记忆库。"""
        if not self._auto_memory or self.memory_engine is None:
            return
        if status != "success":
            return  # 执行失败的对话不存储
        try:
            # 估算对话深度：从历史中数出连续的追问次数
            depth = 0
            store = getattr(self, "session_store", None)
            sid = getattr(self, "session_id", None)
            if store and sid:
                msgs = store.get_messages(sid, limit=10)
                depth = sum(
                    1 for m in msgs[-6:] if m.get("role") == "user"
                )
            self.memory_engine.store(question, answer, depth=depth)
        except Exception:
            pass  # 存储失败不影响对话

    # --- 会话持久化 ---

    def _persist_exchange(self, user_text: str, assistant_text: str,
                          status: str, error: Optional[str]):
        """将一轮对话保存到会话存储（需配置 session_store + session_id）。"""
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
            pass   # 持久化失败不影响对话

    def restore_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """从会话存储恢复消息历史（role/content/timestamp）。

        供 Agent 注入上下文实现断点续聊（子类或适配器使用）。
        """
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
        """获取全局默认 TraceLogger（未安装时为 None）。"""
        try:
            from src.agents.framework.agent_trace import TraceLogger
            return TraceLogger.get_default()
        except ImportError:
            return None

    # --- 流式输出（SSE） ---

    def stream(self, input_text: str, **kwargs):
        """SSE 事件流：meta → delta/result → done/error。

        子类实现 _stream(input_text, **kwargs)（token 级生成器）时逐块
        输出 delta；否则回退为单条 result（非流式兼容）。
        """
        from src.core.streaming import stream_agent
        yield from stream_agent(self, input_text, **kwargs)

    # --- 元信息 ---

    def capabilities(self) -> List[str]:
        """能力标签（子类可覆盖）。"""
        return []

    def metadata(self) -> AgentMeta:
        """返回 Agent 元信息。"""
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


# ============================================================
# 适配器：将已有 Agent 包装为 BaseAgent（无需修改原 Agent）
# ============================================================

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
        # 断点续聊：将持久化历史恢复到被包装 Agent 的内部历史（如有 _history）
        if session_store is not None and session_id is not None:
            self._restore_wrapped_history()

    def _restore_wrapped_history(self):
        """尽力而为：把会话历史注入被包装 Agent 的 _history 列表。

        兼容两种 _history 格式：
          - SimpleAgent：Message 对象列表（有 .role / .content 属性）
          - FunctionCallAgent：dict 列表（{"role": ..., "content": ...}）
        """
        try:
            history = self.restore_history()
            if not history or not hasattr(self._agent, "_history"):
                return
            # 只保留 user/assistant 消息
            simplified = [{"role": m["role"], "content": m["content"]}
                          for m in history if m["role"] in ("user", "assistant")]
            if not simplified:
                return

            history_list = self._agent._history
            if not hasattr(history_list, "append"):
                return

            # 判断当前 _history 的元素类型
            if history_list and hasattr(history_list[0], "role"):
                # Message 对象模式（SimpleAgent）
                restored = [Message(content=m["content"], role=m["role"])
                            for m in simplified]
            else:
                # Dict 模式（FunctionCallAgent）
                restored = simplified
            history_list.clear()
            history_list.extend(restored)
        except Exception:
            pass   # 恢复失败不阻断启动

    def _execute(self, input_text: str, **kwargs) -> str:
        """优先调用 run()，否则调用 chat()。"""
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


# ============================================================
# 2) AgentRegistry：Agent 实例注册表
# ============================================================

class AgentRegistry:
    """Agent 实例注册表：按名称注册/查询/执行 Agent 实例。"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent, name: Optional[str] = None) -> str:
        """注册 Agent 实例，返回注册名。"""
        key = name or agent.name
        self._agents[key] = agent
        return key

    def unregister(self, name: str) -> bool:
        """注销 Agent 实例。"""
        return self._agents.pop(name, None) is not None

    def get(self, name: str) -> Optional[BaseAgent]:
        """按名称获取 Agent 实例。"""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """列出所有已注册 Agent 名称。"""
        return list(self._agents.keys())

    def list_metadata(self) -> List[AgentMeta]:
        """列出所有已注册 Agent 的元信息。"""
        return [a.metadata() for a in self._agents.values()]

    def execute(self, name: str, input_text: str, **kwargs) -> AgentResult:
        """按名称执行 Agent，返回统一结果。"""
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


# ============================================================
# 3) AgentFactory：Agent 类型工厂
# ============================================================

class AgentFactory:
    """Agent 工厂：根据类型创建不同类型的 Agent。"""

    def __init__(self):
        self._types: Dict[str, Type[BaseAgent]] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, agent_type: str, agent_class: Type[BaseAgent],
                 description: str = "") -> None:
        """注册 Agent 类型 → 类。"""
        self._types[agent_type] = agent_class
        self._descriptions[agent_type] = description or getattr(
            agent_class, "DESCRIPTION", "")

    def unregister(self, agent_type: str) -> bool:
        """注销 Agent 类型。"""
        self._descriptions.pop(agent_type, None)
        return self._types.pop(agent_type, None) is not None

    def get_class(self, agent_type: str) -> Optional[Type[BaseAgent]]:
        """获取类型对应的类。"""
        return self._types.get(agent_type)

    def available_types(self) -> Dict[str, str]:
        """列出可用类型（类型 → 描述）。"""
        return dict(self._descriptions)

    def create(self, agent_type: str, **kwargs) -> BaseAgent:
        """创建指定类型的 Agent 实例。

        Raises:
            KeyError: 类型未注册
        """
        agent_class = self._types.get(agent_type)
        if agent_class is None:
            raise KeyError(
                f"未知的 Agent 类型 '{agent_type}'，可用: {', '.join(self._types) or '无'}")
        return agent_class(**kwargs)

    def create_all(self, configs: List[Dict[str, Any]]) -> List[BaseAgent]:
        """批量创建：configs = [{"agent_type": ..., **kwargs}, ...]。"""
        return [self.create(c.pop("agent_type"), **c) for c in configs]


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Agent 统一框架演示（BaseAgent + AgentRegistry + AgentFactory）")
    print("=" * 60)

    # --- 演示用 EchoAgent（不依赖 LLM）---
    class EchoAgent(BaseAgent):
        """回声 Agent：原样返回输入（演示用）。"""
        AGENT_TYPE = "echo"
        DESCRIPTION = "回声 Agent：原样返回输入文本"
        VERSION = "1.0.0"

        def _execute(self, input_text: str, **kwargs) -> str:
            prefix = kwargs.get("prefix", "")
            return f"{prefix}{input_text}"

        def capabilities(self) -> List[str]:
            return ["回声", "无 LLM"]

    class ReverseAgent(BaseAgent):
        """反转 Agent：将输入反转（演示用）。"""
        AGENT_TYPE = "reverse"
        DESCRIPTION = "反转 Agent：将输入文本反转"
        VERSION = "1.0.0"

        def _execute(self, input_text: str, **kwargs) -> str:
            return input_text[::-1]

    # 1. AgentFactory 创建不同类型 Agent
    print("--- 1) AgentFactory 创建不同类型 Agent ---")
    factory = AgentFactory()
    factory.register("echo", EchoAgent, "回声输出")
    factory.register("reverse", ReverseAgent, "反转输出")
    print(f"   可用类型: {factory.available_types()}")

    echo = factory.create("echo", name="回声助手")
    reverse = factory.create("reverse", name="反转助手")
    print(f"   创建: {echo} / {reverse}")

    # 2. BaseAgent 统一执行接口（执行主逻辑 + 返回元信息）
    print("\n--- 2) BaseAgent.execute 统一执行（主逻辑 + 元信息）---")
    result = echo.execute("你好，RAG 系统", prefix="[回声] ")
    print(f"   输出: {result.output}")
    print(f"   状态: {result.status} | 元信息:")
    m = result.metadata
    print(f"     name={m.name} | type={m.agent_type} | version={m.version}")
    print(f"     description={m.description} | capabilities={m.capabilities}")

    # 3. AgentRegistry 实例注册表
    print("\n--- 3) AgentRegistry 注册/查询/执行 ---")
    registry = AgentRegistry()
    registry.register(echo, "main")
    registry.register(reverse, "backup")
    print(f"   已注册: {registry.list_agents()}（共 {registry.count()} 个）")
    r2 = registry.execute("main", "统一接口测试")
    print(f"   registry.execute('main') → {r2.output}")
    print(f"   未注册执行: {registry.execute('nobody', 'x').status}（容错 ✅）")
    for meta in registry.list_metadata():
        print(f"   [元信息] {meta.name} | {meta.agent_type} | {meta.description}")

    # 4. 异常容错：execute 不崩溃
    print("\n--- 4) 异常容错（主逻辑抛错 → 返回 error 结果）---")
    class BrokenAgent(BaseAgent):
        AGENT_TYPE = "broken"
        DESCRIPTION = "会抛异常的 Agent（演示）"

        def _execute(self, input_text, **kwargs):
            raise RuntimeError("演示异常")

    broken = factory.create("broken") if factory.get_class("broken") \
        else BrokenAgent(name="坏Agent")
    res = broken.execute("测试")
    print(f"   状态: {res.status} | error: {res.error}（智能体不崩溃 ✅）")

    # 5. AgentAdapter 包装已有 Agent（无需修改原 Agent）
    print("\n--- 5) AgentAdapter 包装已有风格 Agent ---")
    class LegacyAgent:            # 模拟旧版 Agent（只有 chat 方法）
        def chat(self, user_input: str, **kwargs) -> str:
            return f"旧版 Agent 回应: {user_input}"

    legacy = LegacyAgent()
    adapter = AgentAdapter(legacy, name="旧版包装", agent_type="legacy")
    registry.register(adapter, "legacy")
    print(f"   执行: {registry.execute('legacy', '你好').output}")
    print(f"   元信息: type={adapter.metadata().agent_type} "
          f"| capabilities={adapter.metadata().capabilities}")
    print(f"   注册表共 {registry.count()} 个 Agent: {registry.list_agents()}")
    print()
    print("✅ Agent 统一框架演示完成")
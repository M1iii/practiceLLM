"""AgentTrace / TraceLogger：Agent 级执行可观测性。

记录一次 Agent 执行的完整轨迹：
  - 生命周期：开始/结束/耗时/状态
  - LLM 调用事件：模型、输入输出摘要、耗时、状态
  - 工具调用事件：工具名、参数、结果摘要、耗时、状态
  - 失败链：失败事件的顺序序列（便于定位问题根因）

特性：
  - JSONL 落盘（logs/agent_trace.jsonl）+ 内存环形缓冲
  - LLM Hook：安装后自动记录 practiceLLM 调用事件
  - Tool Observer：挂到 ToolRegistry 后自动记录工具调用事件
  - BaseAgent 集成：execute() 自动开启/结束 trace（通过 TraceLogger.get_default()）

使用方式:
    logger = TraceLogger()
    TraceLogger.set_default(logger)        # 全局默认（BaseAgent 自动使用）
    logger.install_llm_hook()              # 记录 LLM 调用
    registry = ToolRegistry(observer=logger.tool_observer())  # 记录工具调用
    result = agent.execute("问题")         # 自动产生 trace
    trace = logger.get_recent(1)[0]
    print(trace.failure_chain())
"""

import sys
import os
import time
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TraceEvent:
    """一次执行事件（LLM 调用 / 工具调用 / 自定义步骤）。"""
    event_type: str                                # llm / tool / step
    name: str                                      # 模型名 / 工具名 / 步骤名
    input: str = ""                                # 输入摘要
    output: str = ""                               # 输出摘要
    status: str = "success"                        # success / error
    duration_ms: float = 0.0                       # 耗时
    error: Optional[str] = None                    # 错误信息
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_error(self) -> bool:
        return self.status != "success"


@dataclass
class AgentTrace:
    """一次 Agent 执行完整轨迹。"""
    trace_id: str
    agent_name: str
    agent_type: str
    session_id: str
    query: str
    started_at: str
    finished_at: Optional[str] = None
    duration_ms: float = 0.0
    status: str = "success"                        # success / error
    error: Optional[str] = None                    # 顶层错误
    events: List[TraceEvent] = field(default_factory=list)

    # --- 便捷统计 ---

    @property
    def llm_calls(self) -> int:
        return sum(1 for e in self.events if e.event_type == "llm")

    @property
    def tool_calls(self) -> int:
        return sum(1 for e in self.events if e.event_type == "tool")

    def failure_chain(self) -> List[Dict[str, Any]]:
        """失败事件链（按时间顺序）。"""
        return [
            {"event": e.event_type, "name": e.name, "error": e.error}
            for e in self.events if e.is_error()
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "session_id": self.session_id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "failed_events": len(self.failure_chain()),
            "events": [asdict(e) for e in self.events],
        }


class TraceLogger:
    """Agent 执行轨迹记录器。"""

    def __init__(self, log_path: Optional[str] = None,
                 max_entries: int = 200,
                 max_event_detail: int = 200):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_path = log_path or os.path.join(root, "logs", "agent_trace.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.max_entries = max(1, int(max_entries))
        self.max_event_detail = max(20, int(max_event_detail))
        self._traces: Dict[str, AgentTrace] = {}
        self._events: Dict[str, List[TraceEvent]] = {}
        self._order: List[str] = []                # trace_id 顺序（环形裁剪）
        self._active: List[str] = []               # 活动 trace 栈
        self._lock = threading.Lock()
        self._hook_installed = False
        self._write_failures = 0

    # ------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------

    def start(self, agent_name: str, agent_type: str, query: str,
              session_id: str = "default") -> str:
        """开启一条 trace，返回 trace_id。"""
        trace_id = f"trace_{int(time.time() * 1000)}_{len(self._order)}"
        with self._lock:
            self._traces[trace_id] = AgentTrace(
                trace_id=trace_id, agent_name=agent_name,
                agent_type=agent_type, session_id=session_id, query=query,
                started_at=datetime.now().isoformat(timespec="seconds"))
            self._events[trace_id] = []
            self._order.append(trace_id)
            # 环形裁剪
            while len(self._order) > self.max_entries:
                old = self._order.pop(0)
                self._traces.pop(old, None)
                self._events.pop(old, None)
        return trace_id

    def finish(self, trace_id: str, status: str = "success",
               error: Optional[str] = None) -> Optional[AgentTrace]:
        """结束一条 trace，返回完整 AgentTrace（并落盘）。"""
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return None
            trace.finished_at = datetime.now().isoformat(timespec="seconds")
            trace.status = status
            trace.error = error
            trace.events = list(self._events.get(trace_id, []))
            if trace.started_at:
                try:
                    started = datetime.fromisoformat(trace.started_at).timestamp()
                    trace.duration_ms = round(
                        (time.time() - started) * 1000, 1)
                except (ValueError, TypeError):
                    trace.duration_ms = 0.0
        self._append_jsonl(trace)
        return trace

    # ------------------------------------------------------------
    # 事件记录
    # ------------------------------------------------------------

    def log_event(self, trace_id: str, event_type: str, name: str,
                  input: str = "", output: str = "",
                  status: str = "success", duration_ms: float = 0.0,
                  error: Optional[str] = None, **extra) -> Optional[TraceEvent]:
        """向指定 trace 追加事件。"""
        event = TraceEvent(
            event_type=event_type, name=name,
            input=self._clip(input), output=self._clip(output),
            status=status, duration_ms=round(duration_ms, 1),
            error=error, extra=extra)
        with self._lock:
            if trace_id not in self._events:
                return None
            self._events[trace_id].append(event)
        return event

    def log_current(self, event_type: str, name: str, **kwargs) -> Optional[TraceEvent]:
        """向当前活动 trace 追加事件（供 Hook/Observer 使用）。"""
        trace_id = self.current_trace_id
        if trace_id is None:
            return None
        return self.log_event(trace_id, event_type, name, **kwargs)

    @contextmanager
    def active(self, trace_id: str):
        """上下文：将 trace 设为活动（Hook/Observer 记录目标）。"""
        self._active.append(trace_id)
        try:
            yield
        finally:
            if self._active and self._active[-1] == trace_id:
                self._active.pop()

    @property
    def current_trace_id(self) -> Optional[str]:
        return self._active[-1] if self._active else None

    def _clip(self, text: str) -> str:
        text = text or ""
        if len(text) > self.max_event_detail:
            return text[:self.max_event_detail] + "…"
        return text

    # ------------------------------------------------------------
    # 查询与统计
    # ------------------------------------------------------------

    def get_trace(self, trace_id: str) -> Optional[AgentTrace]:
        return self._traces.get(trace_id)

    def get_recent(self, n: int = 10) -> List[AgentTrace]:
        """最近 n 条 trace（最新在前）。"""
        with self._lock:
            ids = list(self._order)
        return [self._traces[i] for i in reversed(ids[-n:]) if i in self._traces]

    def get_aggregate(self) -> Dict[str, Any]:
        """聚合统计。"""
        traces = self.get_recent(self.max_entries)
        if not traces:
            return {"total_traces": 0}
        successes = sum(1 for t in traces if t.status == "success")
        return {
            "total_traces": len(traces),
            "success_rate": round(successes / len(traces), 3),
            "avg_duration_ms": round(
                sum(t.duration_ms for t in traces) / len(traces), 1),
            "total_llm_calls": sum(t.llm_calls for t in traces),
            "total_tool_calls": sum(t.tool_calls for t in traces),
            "total_failed_events": sum(len(t.failure_chain()) for t in traces),
            "recent_failures": [
                {"trace_id": t.trace_id, "agent": t.agent_name,
                 "error": t.error, "chain": t.failure_chain()}
                for t in traces if t.status == "error"][:5],
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "log_path": self.log_path,
            "max_entries": self.max_entries,
            "traces_in_memory": len(self._traces),
            "active_traces": len(self._active),
            "llm_hook_installed": self._hook_installed,
            "write_failures": self._write_failures,
        }

    def _append_jsonl(self, trace: AgentTrace):
        """trace 完成后落盘（JSONL 追加，default=str 兜底非标准类型）。"""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False,
                                   default=str) + "\n")
        except OSError:
            self._write_failures += 1

    # ------------------------------------------------------------
    # 集成 Hook
    # ------------------------------------------------------------

    def install_llm_hook(self) -> bool:
        """安装 LLM Hook：包装 practiceLLM.invoke/think 自动记录事件。

        幂等：重复调用不重复包装。
        """
        if self._hook_installed:
            return False
        try:
            from core.llm import practiceLLM
        except ImportError:
            return False

        _orig_invoke = practiceLLM.invoke
        _orig_think = practiceLLM.think
        logger = self

        def wrapped_invoke(self_, messages, temperature=0, stream=False, **kw):
            start = time.perf_counter()
            status, error = "success", None
            try:
                result = _orig_invoke(self_, messages, temperature=temperature,
                                      stream=stream, **kw)
            except Exception as e:
                result, status = None, "error"
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                first = ""
                if messages:
                    m = messages[0]
                    first = m.get("content", "") if isinstance(m, dict) else str(m)
                logger.log_current(
                    "llm", getattr(self_, "model", "llm") or "llm",
                    input=first, output=result,
                    status=status, duration_ms=(time.perf_counter() - start) * 1000,
                    error=error)
            return result

        def wrapped_think(self_, messages, temperature=0):
            start = time.perf_counter()
            status, error = "success", None
            try:
                result = _orig_think(self_, messages, temperature=temperature)
            except Exception as e:
                result, status = None, "error"
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                first = ""
                if messages:
                    m = messages[0]
                    first = m.get("content", "") if isinstance(m, dict) else str(m)
                logger.log_current(
                    "llm", getattr(self_, "model", "llm") or "llm",
                    input=first, output=result,
                    status=status, duration_ms=(time.perf_counter() - start) * 1000,
                    error=error)
            return result

        practiceLLM.invoke = wrapped_invoke
        practiceLLM.think = wrapped_think
        self._hook_installed = True
        return True

    def tool_observer(self) -> Callable[[str, Dict[str, Any], Any], None]:
        """生成 ToolRegistry observer：自动记录工具调用事件。

        用法: ToolRegistry(..., observer=logger.tool_observer())
        """
        def observer(name: str, args: dict, resp) -> None:
            self.log_current(
                "tool", name,
                input=json.dumps(args, ensure_ascii=False) if args else "",
                output=getattr(resp, "output", str(resp)),
                status="success" if not getattr(resp, "is_error", False) else "error",
                duration_ms=getattr(resp, "duration_ms", 0.0),
                error=getattr(resp, "error", None))
        return observer

    # ------------------------------------------------------------
    # 全局默认（BaseAgent 集成）
    # ------------------------------------------------------------

    _default: Optional["TraceLogger"] = None

    @classmethod
    def set_default(cls, logger: "TraceLogger") -> None:
        cls._default = logger

    @classmethod
    def get_default(cls) -> Optional["TraceLogger"]:
        return cls._default


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 TraceLogger Agent 级执行轨迹演示")
    print("=" * 60)

    import tempfile
    from tools.framework.tool_system import ToolRegistry, ToolParameter, Tool

    demo_dir = os.path.join(tempfile.gettempdir(), "trace_demo")
    os.makedirs(demo_dir, exist_ok=True)
    logger = TraceLogger(log_path=os.path.join(demo_dir, "agent_trace.jsonl"))

    # 1. 安装 LLM Hook（记录 practiceLLM 调用）
    print("--- 1) 安装 LLM Hook ---")
    installed = logger.install_llm_hook()
    print(f"   安装成功: {installed}")

    # 2. 工具注册表 + observer
    print("--- 2) ToolRegistry + observer ---")
    registry = ToolRegistry(observer=logger.tool_observer())

    class DoublerTool(Tool):
        @property
        def name(self) -> str:
            return "Doubler"

        @property
        def description(self) -> str:
            return "将输入数字翻倍"

        def get_parameters(self) -> List[ToolParameter]:
            return [ToolParameter(name="value", type="number",
                                  description="输入数字")]

        def run(self, args: dict) -> str:
            return str(int(args.get("value", 0)) * 2)

    def fail_tool(msg: str) -> str:
        raise ValueError(f"工具失败: {msg}")

    registry.register(DoublerTool())
    registry.register(fail_tool, name="Failer", description="必然失败的工具",
                      parameters=[ToolParameter(name="msg", type="string",
                                                description="错误消息")])

    # 3. 演示 Agent（含 LLM 调用 + 工具调用 + 失败）
    print("--- 3) 执行带轨迹的 Agent ---")
    from core.llm import practiceLLM
    from agents.framework.agent_framework import BaseAgent

    class TracerDemoAgent(BaseAgent):
        AGENT_TYPE = "trace_demo"
        DESCRIPTION = "TraceLogger 演示 Agent"

        def _execute(self, input_text: str, **kwargs) -> str:
            # 工具调用（成功 + 失败）
            r1 = registry.execute_structured("Doubler", {"value": 21})
            r2 = registry.execute_structured("Failer", {"msg": "演示异常"})
            # LLM 调用
            llm = practiceLLM()
            answer = llm.invoke([{"role": "user", "content": input_text}],
                                temperature=0)
            return f"工具结果: {r1.output} | LLM: {answer}"

    agent = TracerDemoAgent(name="演示Agent", tracer=logger)
    result = agent.execute("用一句话介绍 RAG")
    print(f"   Agent 执行完成: status={result.status}")

    # 4. 轨迹检查
    print("--- 4) 轨迹内容 ---")
    trace = logger.get_recent(1)[0]
    print(f"   trace_id: {trace.trace_id} | agent: {trace.agent_name}")
    print(f"   状态: {trace.status} | 耗时: {trace.duration_ms}ms")
    print(f"   LLM 调用: {trace.llm_calls} 次 | 工具调用: {trace.tool_calls} 次")
    for e in trace.events:
        print(f"   · [{e.event_type}] {e.name} → {e.status} "
              f"({e.duration_ms}ms){' | ' + e.error if e.error else ''}")

    print("\n--- 5) 失败链 ---")
    for f in trace.failure_chain():
        print(f"   ✗ {f['event']}:{f['name']} → {f['error']}")

    # 6. 聚合统计 + JSONL
    print("--- 6) 聚合统计与落盘 ---")
    print(f"   聚合: {logger.get_aggregate()}")
    print(f"   JSONL: {logger.stats()['log_path']} 已写入")
    with open(logger.log_path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    print(f"   文件行数: {len(lines)} | 含事件数: {lines[0].count(chr(34) + 'event_type' + chr(34))}")

    print("\n✅ TraceLogger 演示完成")

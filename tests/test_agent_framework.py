"""Agent 统一框架测试（无 LLM，使用演示 Agent）。"""
import pytest
from agents.framework.agent_framework import (
    BaseAgent, AgentFactory, AgentRegistry, AgentAdapter, AgentResult)


class EchoAgent(BaseAgent):
    AGENT_TYPE = "echo"
    DESCRIPTION = "回声测试 Agent"

    def _execute(self, input_text: str, **kwargs) -> str:
        return f"[echo] {input_text}"


class FailAgent(BaseAgent):
    AGENT_TYPE = "fail"

    def _execute(self, input_text: str, **kwargs) -> str:
        raise ValueError("主逻辑崩溃")


class ChatLike:
    """LegacyAgent：只有 chat() 接口（适配器测试）。"""

    def chat(self, text: str) -> str:
        return f"legacy: {text}"


def test_execute_success():
    agent = EchoAgent(name="e1")
    r = agent.execute("hello")
    assert isinstance(r, AgentResult)
    assert r.status == "success"
    assert r.output == "[echo] hello"
    assert r.metadata.name == "e1"


def test_execute_error_captured():
    agent = FailAgent()
    r = agent.execute("x")
    assert r.status == "error"
    assert "ValueError" in r.error


def test_factory_and_registry():
    factory = AgentFactory()
    factory.register("echo", EchoAgent, "回声")
    registry = AgentRegistry()
    agent = factory.create("echo", name="e")
    key = registry.register(agent, "echo")
    assert key == "echo"
    assert registry.get("echo").name == "e"
    assert "echo" in factory.available_types()
    r = registry.execute("echo", "hi")
    assert r.output == "[echo] hi"
    # 未注册执行容错
    assert registry.execute("nobody", "x").status == "error"


def test_adapter_wraps_chat_agent():
    class LegacyAdapter(AgentAdapter):
        def __init__(self, name=None, **kwargs):
            super().__init__(ChatLike(), name=name, agent_type="legacy",
                             description="旧接口 Agent")

    factory = AgentFactory()
    factory.register("legacy", LegacyAdapter, "旧接口 Agent")
    agent = factory.create("legacy")
    r = agent.execute("你好")
    assert r.status == "success"
    assert r.output == "legacy: 你好"
    assert agent.metadata().agent_type == "legacy"


def test_session_persistence_via_execute(tmp_path):
    """BaseAgent.execute 自动持久化（配置 session_store）。"""
    from agents.framework.session_store import SessionStore
    store = SessionStore(db_path=str(tmp_path / "t.db"))
    agent = EchoAgent(name="e", session_store=store, session_id="s1")
    agent.execute("第一问")
    msgs = store.get_messages("s1")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "第一问"
    assert msgs[1]["role"] == "assistant"
    # restore_history
    history = agent.restore_history()
    assert len(history) == 2


def test_stream_fallback_result():
    """无 _stream 的 Agent：stream() 回退单条 result。"""
    agent = EchoAgent(name="e")
    events = list(agent.stream("hello"))
    types = [ev.event for ev in events]
    assert "meta" in types and "result" in types and "done" in types
    result_ev = next(ev for ev in events if ev.event == "result")
    assert result_ev.data == "[echo] hello"

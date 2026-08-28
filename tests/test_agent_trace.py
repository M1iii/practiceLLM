"""Agent 级 TraceLogger 测试（不触发真实 LLM）。"""
import json
import tempfile
import os
from agents.framework.agent_trace import TraceLogger, TraceEvent, AgentTrace
from agents.framework.agent_framework import BaseAgent


class EchoAgent(BaseAgent):
    AGENT_TYPE = "echo_trace"

    def _execute(self, input_text: str, **kwargs) -> str:
        return f"[echo] {input_text}"


class FailAgent(BaseAgent):
    AGENT_TYPE = "fail_trace"

    def _execute(self, input_text: str, **kwargs) -> str:
        raise ValueError("trace 测试失败")


def _logger(tmp_path):
    return TraceLogger(log_path=str(tmp_path / "trace.jsonl"))


def test_lifecycle(tmp_path):
    logger = _logger(tmp_path)
    tid = logger.start("agent", "echo_trace", "你好")
    assert logger.current_trace_id is None          # 未进入 active 上下文
    with logger.active(tid):
        assert logger.current_trace_id == tid
        logger.log_current("tool", "Calculator", input="{}", output="3")
    assert logger.current_trace_id is None          # 退出上下文后清空
    logger.finish(tid)
    trace = logger.get_trace(tid)
    assert trace.status == "success"
    assert trace.tool_calls == 1
    assert len(trace.failure_chain()) == 0


def test_failure_chain(tmp_path):
    logger = _logger(tmp_path)
    tid = logger.start("a", "t", "q")
    logger.log_event(tid, "tool", "Failer", status="error",
                     error="ValueError: boom")
    logger.log_event(tid, "llm", "deepseek", status="success", output="ok")
    logger.finish(tid)
    trace = logger.get_trace(tid)
    chain = trace.failure_chain()
    assert len(chain) == 1
    assert chain[0]["name"] == "Failer"
    assert "ValueError" in chain[0]["error"]


def test_jsonl_written(tmp_path):
    logger = _logger(tmp_path)
    tid = logger.start("a", "t", "q")
    logger.finish(tid)
    with open(str(tmp_path / "trace.jsonl"), encoding="utf-8") as f:
        line = f.readline()
    data = json.loads(line)
    assert data["status"] == "success"
    assert data["trace_id"] == tid


def test_baseagent_auto_trace(tmp_path):
    logger = _logger(tmp_path)
    agent = EchoAgent(name="e", tracer=logger)
    agent.execute("hello")
    trace = logger.get_recent(1)[0]
    assert trace.agent_name == "e"
    assert trace.status == "success"
    assert trace.query == "hello"


def test_baseagent_error_trace(tmp_path):
    logger = _logger(tmp_path)
    agent = FailAgent(name="f", tracer=logger)
    r = agent.execute("x")
    assert r.status == "error"
    trace = logger.get_recent(1)[0]
    assert trace.status == "error"
    assert "ValueError" in trace.error


def test_aggregate(tmp_path):
    logger = _logger(tmp_path)
    for _ in range(3):
        tid = logger.start("a", "t", "q")
        logger.finish(tid)
    agg = logger.get_aggregate()
    assert agg["total_traces"] == 3
    assert agg["success_rate"] == 1.0


def test_ring_buffer_cap(tmp_path):
    logger = TraceLogger(log_path=str(tmp_path / "t.jsonl"), max_entries=5)
    for i in range(10):
        tid = logger.start("a", "t", f"q{i}")
        logger.finish(tid)
    assert len(logger.get_recent(100)) == 5     # 环形裁剪

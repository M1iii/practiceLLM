"""示例 03：会话持久化 + 执行轨迹（可观测性）。

运行: python examples/03_session_and_trace.py
（会调用真实 LLM 1 次）
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.framework.agent_framework import BaseAgent
from agents.framework.session_store import SessionStore
from agents.framework.agent_trace import TraceLogger


class SimpleEcho(BaseAgent):
    AGENT_TYPE = "echo"
    DESCRIPTION = "回声 Agent"

    def _execute(self, input_text: str, **kwargs) -> str:
        return f"[echo] {input_text}"


def main():
    db = os.path.join(tempfile.gettempdir(), "example_session.db")
    store = SessionStore(db_path=db)
    logger = TraceLogger(log_path=os.path.join(tempfile.gettempdir(),
                                               "example_trace.jsonl"))

    # 1. 会话持久化：同一会话连续执行（自动保存每轮）
    agent = SimpleEcho(name="e1", session_store=store, session_id="demo_s1")
    agent.execute("第一问")
    agent.execute("第二问")
    history = store.get_messages("demo_s1")
    print(f"会话 demo_s1 已保存 {len(history)} 条消息")
    print(f"  [第1条] {history[0]['role']}: {history[0]['content']}")

    # 2. 执行轨迹
    agent2 = SimpleEcho(name="e2", tracer=logger)
    agent2.execute("带轨迹执行")
    trace = logger.get_recent(1)[0]
    print(f"轨迹: {trace.agent_name} | {trace.status} | "
          f"{trace.duration_ms}ms | 事件 {len(trace.events)} 条")


if __name__ == "__main__":
    main()

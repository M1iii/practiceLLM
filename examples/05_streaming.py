"""示例 05：SSE 流式输出。

演示 BaseAgent.stream() 事件流 + HTTP text/event-stream 消费。
运行: python examples/05_streaming.py
（会调用真实 LLM 1 次）
"""
import sys
import os
import urllib.request
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from src.agents.framework.agent_framework import BaseAgent
from src.core.llm import practiceLLM
from src.core.streaming import SSEClientStream


class StreamAgent(BaseAgent):
    """token 级流式 Agent。"""
    AGENT_TYPE = "stream"
    DESCRIPTION = "SSE 流式示例"

    def _execute(self, input_text: str, **kwargs) -> str:
        return "".join(self._stream(input_text, **kwargs))

    def _stream(self, input_text: str, **kwargs):
        llm = practiceLLM()
        yield from llm.stream_chunks(
            [{"role": "user", "content": input_text}], temperature=0)


def main():
    agent = StreamAgent(name="流式示例")
    question = "用一句话介绍流式输出"

    # 1. 直接消费事件流（前 5 帧）
    print("--- 事件流前 5 帧 ---")
    for ev in agent.stream(question):
        if ev.event in ("meta", "done", "error"):
            print(f"  [{ev.event}] {str(ev.data)[:50]}")
        else:
            print(f"  [delta] {ev.data}")

    # 2. HTTP SSE
    print("\n--- HTTP SSE 冒烟测试 ---")
    frames = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in SSEClientStream(agent)(question):
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
            frames.append(1)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/stream",
            timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    server.shutdown()
    print(f"  HTTP 200 | 事件帧数: {raw.count('event:')} | "
          f"含 done: {'done' in raw}")


if __name__ == "__main__":
    main()

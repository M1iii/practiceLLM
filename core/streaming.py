"""core/streaming.py — SSE（Server-Sent Events）流式输出。

为 Agent 提供标准的 SSE 事件流，可对接 HTTP（text/event-stream）、CLI、WebSocket。

事件协议（event → data）:
  meta    → Agent 元信息
  delta   → LLM 增量文本块（流式）
  result  → 非流式 Agent 的完整结果（fallback）
  tool    → 工具调用（可选）
  done    → 结束（含累计耗时/状态统计）
  error   → 执行异常

使用方式:
    # 生成器事件流
    for event in stream_agent(agent, "你好"):
        print(event.format())

    # HTTP SSE 响应（可接 FastAPI/Starlette 或标准 http.server）
    from core.streaming import SSEClientStream
    return StreamingResponse(SSEClientStream(agent)("你好"),
                             media_type="text/event-stream")
"""

import sys
import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# SSE 事件与格式化
# ============================================================

@dataclass
class SSEEvent:
    """一个 SSE 事件。"""
    event: str                       # 事件类型
    data: Any = None                 # 数据（JSON 序列化）
    id: Optional[str] = None         # 事件 id

    def payload(self) -> str:
        """data 的 JSON 字符串。"""
        return json.dumps(self.data, ensure_ascii=False, default=str)

    def format(self) -> str:
        """SSE 线路格式：event: x\ndata: y\n\n"""
        lines = []
        if self.id is not None:
            lines.append(f"id: {self.id}")
        lines.append(f"event: {self.event}")
        for line in self.payload().splitlines() or [""]:
            lines.append(f"data: {line}")
        return "\n".join(lines) + "\n\n"


def format_sse(event: str, data: Any = None, id: Optional[str] = None) -> str:
    """便捷函数：构造并格式化一个 SSE 事件。"""
    return SSEEvent(event=event, data=data, id=id).format()


def _meta_dict(agent) -> Dict[str, Any]:
    """Agent 元信息 → 可 JSON 序列化的 dict。"""
    try:
        from dataclasses import asdict
        return asdict(agent.metadata())
    except Exception:
        return {"name": getattr(agent, "name", ""),
                "agent_type": getattr(agent, "AGENT_TYPE", "")}


# ============================================================
# 事件流生成器
# ============================================================

def stream_agent(agent, input_text: str, **kwargs) -> Iterator[SSEEvent]:
    """将 Agent 执行包装为 SSE 事件流。

    流式策略：
      - Agent 实现了 _stream()（token 级生成器）→ 逐个 yield delta
      - 否则调用 execute() → 单条 result 事件
    始终以 done（或 error）事件结束。
    """
    started = time.perf_counter()
    try:
        yield SSEEvent("meta", _meta_dict(agent))

        stream_fn = getattr(agent, "_stream", None)
        if callable(stream_fn):
            # token 级流式
            for chunk in stream_fn(input_text, **kwargs):
                yield SSEEvent("delta", chunk)
        else:
            # 非流式 fallback：单条结果（静默 Agent 自身的控制台打印）
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                result = agent.execute(input_text, **kwargs)
            if result.status == "success":
                yield SSEEvent("result", result.output)
            else:
                yield SSEEvent("error", {"message": result.error})

        yield SSEEvent("done", {
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "input": input_text[:200],
        })
    except Exception as e:
        yield SSEEvent("error", {
            "message": f"{type(e).__name__}: {e}",
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        })


class SSEClientStream:
    """SSE 客户端流：将 Agent 包装为可直接作 HTTP body 的迭代器。

    用法（FastAPI）:
        @app.get("/stream")
        async def stream(q: str):
            return StreamingResponse(
                SSEClientStream(agent)(q),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"})
    """

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, input_text: str, **kwargs) -> Iterator[str]:
        """yield SSE 线路格式字符串（HTTP body 直接用）。"""
        for event in stream_agent(self.agent, input_text, **kwargs):
            yield event.format()


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from agents.framework.agent_framework import BaseAgent
    from core.llm import practiceLLM

    print("=" * 60)
    print("📡 SSE 流式输出演示")
    print("=" * 60)

    # --- 演示用流式 Agent（token 级生成）---
    class StreamingAgent(BaseAgent):
        AGENT_TYPE = "stream"
        DESCRIPTION = "SSE 流式演示 Agent（token 级）"
        VERSION = "1.0.0"

        def _execute(self, input_text: str, **kwargs) -> str:
            return "".join(self._stream(input_text, **kwargs))

        def _stream(self, input_text: str, **kwargs):
            llm = practiceLLM()
            for chunk in llm.stream_chunks(
                    [{"role": "user", "content": input_text}], temperature=0):
                yield chunk

    agent = StreamingAgent(name="流式Agent")

    # 1. 事件流格式演示（前 5 帧）
    print("--- 1) SSE 事件流（前 5 帧）---")
    count = 0
    for ev in stream_agent(agent, "用一句话介绍 RAG"):
        if count < 5:
            print(ev.format(), end="")
        count += 1
    print(f"   （共 {count} 帧：meta + N×delta + done）\n")

    # 2. HTTP SSE 服务器冒烟测试
    print("--- 2) HTTP SSE 服务器（text/event-stream）---")
    query = "用一句话介绍向量检索"
    received = []
    keep = {"stop": False}

    class SSEHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for frame in SSEClientStream(agent)(query):
                    self.wfile.write(frame.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                keep["stop"] = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SSEHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"   服务器已启动: http://127.0.0.1:{port}/stream")

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/stream", timeout=60) as resp:
        print(f"   HTTP 状态: {resp.status} | Content-Type: "
              f"{resp.headers.get('Content-Type')}")
        raw = ""
        for line in resp:
            raw += line.decode("utf-8")
    server.shutdown()
    print(f"   收到 {raw.count('event:')} 个事件帧")
    # 重组 delta 文本验证流式完整性
    deltas = []
    event = None
    for line in raw.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event == "delta":
            deltas.append(line[6:])
    print(f"   delta 块数: {len(deltas)} | 重组文本: "
          f"{''.join(deltas)[:60]}…")
    print(f"   done 事件: {'done' in raw}")

    print("\n✅ SSE 流式输出演示完成")

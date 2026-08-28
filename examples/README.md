# examples 示例

| 示例 | 内容 | 需要 LLM |
|---|---|---|
| [01_agent_framework.py](01_agent_framework.py) | Agent 统一框架（BaseAgent/Factory/Registry） | 否 |
| [02_tools.py](02_tools.py) | 工具系统（ToolResponse/熔断/过滤/子代理） | 否 |
| [03_session_and_trace.py](03_session_and_trace.py) | 会话持久化 + 执行轨迹 | 否（内部演示 Agent） |
| [04_skills.py](04_skills.py) | Skills 知识外化（Loader + Tool） | 否 |
| [05_streaming.py](05_streaming.py) | SSE 流式输出（事件流 + HTTP） | 是 |

运行方式：

```bash
python examples/01_agent_framework.py
python examples/02_tools.py
# ...
```

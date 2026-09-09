"""演示代码：EchoAgent / ReverseAgent / BrokenAgent + AgentAdapter 包装示例。"""

from typing import List

from .base import BaseAgent
from .registry import AgentRegistry, AgentFactory
from .models import AgentMeta, AgentResult


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


class BrokenAgent(BaseAgent):
    """会抛异常的 Agent（演示）。"""
    AGENT_TYPE = "broken"
    DESCRIPTION = "会抛异常的 Agent（演示）"

    def _execute(self, input_text, **kwargs):
        raise RuntimeError("演示异常")


class LegacyAgent:
    """模拟旧版 Agent（只有 chat 方法）。"""
    def chat(self, user_input: str, **kwargs) -> str:
        return f"旧版 Agent 回应: {user_input}"


if __name__ == "__main__":
    print("=" * 60)
    print("Agent 统一框架演示（BaseAgent + AgentRegistry + AgentFactory）")
    print("=" * 60)

    # 1. AgentFactory 创建不同类型 Agent
    print("--- 1) AgentFactory 创建不同类型 Agent ---")
    factory = AgentFactory()
    factory.register("echo", EchoAgent, "回声输出")
    factory.register("reverse", ReverseAgent, "反转输出")
    print(f"   可用类型: {factory.available_types()}")

    echo = factory.create("echo", name="回声助手")
    reverse = factory.create("reverse", name="反转助手")
    print(f"   创建: {echo} / {reverse}")

    # 2. BaseAgent 统一执行接口
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
    print(f"   registry.execute('main') -> {r2.output}")
    print(f"   未注册执行: {registry.execute('nobody', 'x').status}（容错通过）")
    for meta in registry.list_metadata():
        print(f"   [元信息] {meta.name} | {meta.agent_type} | {meta.description}")

    # 4. 异常容错
    print("\n--- 4) 异常容错（主逻辑抛错 -> 返回 error 结果）---")
    broken = BrokenAgent(name="坏Agent")
    res = broken.execute("测试")
    print(f"   状态: {res.status} | error: {res.error}（智能体不崩溃）")

    # 5. AgentAdapter 包装已有 Agent
    print("\n--- 5) AgentAdapter 包装已有风格 Agent ---")
    from .adapter import AgentAdapter
    legacy = LegacyAgent()
    adapter = AgentAdapter(legacy, name="旧版包装", agent_type="legacy")
    registry.register(adapter, "legacy")
    print(f"   执行: {registry.execute('legacy', '你好').output}")
    print(f"   元信息: type={adapter.metadata().agent_type} "
          f"| capabilities={adapter.metadata().capabilities}")
    print(f"   注册表共 {registry.count()} 个 Agent: {registry.list_agents()}")
    print()
    print("Agent 统一框架演示完成")
"""示例 01：Agent 统一框架基础。

演示 BaseAgent / AgentFactory / AgentRegistry / AgentAdapter 的用法。
运行: python examples/01_agent_framework.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.framework.agent_framework import BaseAgent, AgentFactory, AgentRegistry


class GreetAgent(BaseAgent):
    """问候 Agent（无需 LLM 的演示实现）。"""
    AGENT_TYPE = "greet"
    DESCRIPTION = "问候 Agent：回复问候语"
    VERSION = "1.0.0"

    def _execute(self, input_text: str, **kwargs) -> str:
        return f"你好！你说的是：{input_text}"


def main():
    # 1. 注册
    factory = AgentFactory()
    factory.register("greet", GreetAgent, "问候 Agent")
    registry = AgentRegistry()

    # 2. 创建并激活
    agent = factory.create("greet", name="greet_1")
    registry.register(agent, "greet")

    # 3. 执行
    result = registry.execute("greet", "今天天气不错")
    print(f"状态: {result.status}")
    print(f"输出: {result.output}")
    print(f"元信息: {agent.metadata().description}")

    # 4. 列表
    print(f"可用类型: {factory.available_types()}")


if __name__ == "__main__":
    main()

"""示例 02：工具系统（ToolResponse + 熔断器 + 过滤 + 子代理）。

运行: python examples/02_tools.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.framework.tool_system import ToolRegistry, ToolParameter
from tools.framework.tool_filter import ToolFilter
from tools.framework.sub_agent_tool import SubAgentTool
from agents.framework.agent_framework import BaseAgent, AgentFactory


def add(a, b):
    """加法。"""
    return a + b


def flaky(a):
    """不稳定工具（模拟上游故障）。"""
    raise ConnectionError("上游服务不可用")


def main():
    # 1. 注册工具 + 熔断器
    registry = ToolRegistry(enable_circuit_breaker=True,
                            breaker_failure_threshold=2)
    registry.register(add, name="Calculator",
                      parameters=[ToolParameter(name="a", type="number",
                                                description="a"),
                                  ToolParameter(name="b", type="number",
                                                description="b")])
    registry.register(flaky, name="Flaky",
                      parameters=[ToolParameter(name="a", type="number",
                                                description="a")])

    # 2. ToolResponse 协议
    resp = registry.execute_structured("Calculator", {"a": 2, "b": 3})
    print(f"结构化执行: [{resp.status}] {resp.output} data={resp.data}")

    # 3. 熔断器：连续失败后快速失败
    for i in range(1, 4):
        r = registry.execute_structured("Flaky", {"a": 1})
        print(f"Flaky 调用{i}: [{r.status}] {r.error}")

    # 4. 工具过滤
    tf = ToolFilter()
    tf.set_policy("writer", allow=["Calculator"])
    reg2 = ToolRegistry(tool_filter=tf, owner="writer",
                        enable_circuit_breaker=False)
    reg2.register(add, name="Calculator",
                  parameters=[ToolParameter(name="a", type="number",
                                            description="a"),
                              ToolParameter(name="b", type="number",
                                            description="b")])
    print(f"writer 可见工具: {reg2.list_tools()}")

    # 5. 子代理委派（独立注册表，演示委派能力本身）
    factory = AgentFactory()

    class EchoSub(BaseAgent):
        AGENT_TYPE = "echo_sub"
        DESCRIPTION = "回声子代理"

        def _execute(self, input_text: str, **kwargs) -> str:
            return f"子代理处理: {input_text}"

    factory.register("echo_sub", EchoSub, "回声子代理")
    reg3 = ToolRegistry(enable_circuit_breaker=False)
    reg3.register(SubAgentTool(factory))
    resp = reg3.execute_structured("SubAgent",
                                   {"task": "整理要点", "agent_type": "echo_sub"})
    print(f"子代理委派: [{resp.status}] {resp.output}")
    # 组合演示：writer 角色无权调用 SubAgent（被过滤拦截）
    reg2.register(SubAgentTool(factory))
    resp_forbidden = reg2.execute_structured("SubAgent",
                                             {"task": "x", "agent_type": "echo_sub"})
    print(f"writer 调用 SubAgent: [{resp_forbidden.status}] {resp_forbidden.error}")


if __name__ == "__main__":
    main()

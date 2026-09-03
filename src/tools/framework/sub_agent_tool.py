"""SubAgentTool：子代理委派工具（Agent 可把子任务交给其他 Agent 执行）。

设计：
  - 包装 AgentFactory：以工具形式暴露"委派子任务"能力
  - 参数：task（任务描述）、agent_type（子代理类型，默认 simple）
  - 深度保护：threading.local 嵌套深度计数，超过 MAX_DEPTH 拒绝
    （防止 Agent 无限互相委派导致死循环）
  - 返回 ToolResponse：子代理执行结果或错误

使用方式:
    factory = AgentFactory()
    factory.register("simple", SimpleAgentWrapper, "简单聊天")
    registry = ToolRegistry()
    registry.register(SubAgentTool(factory))     # 名为 SubAgent
    resp = registry.execute_structured("SubAgent", {
        "task": "帮我计算 1+1", "agent_type": "simple"})
"""

import sys
import os
import threading
from typing import Any, Callable, Dict, List, Optional


from src.tools.framework.tool_system import Tool, ToolParameter
from src.tools.framework.tool_response import ToolResponse

MAX_DEPTH = 3   # 子代理最大嵌套深度


class SubAgentTool(Tool):
    """子代理委派工具。"""

    def __init__(self, agent_factory: Any,
                 name: str = "SubAgent",
                 max_depth: int = MAX_DEPTH):
        """
        Args:
            agent_factory: AgentFactory（用于创建子代理）
            name: 工具名
            max_depth: 最大委派嵌套深度
        """
        self._factory = agent_factory
        self._tool_name = name
        self._max_depth = max(1, int(max_depth))
        self._local = threading.local()

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str:
        return ("将子任务委派给另一个 Agent 独立执行，支持指定 agent_type 与嵌套深度保护。"
                "适用：把复杂任务拆分为可独立完成的子任务、需要不同 Agent 类型分工。"
                "不适用：简单可直接完成的任务、需要共享上下文或记忆的子任务。"
                "注意：嵌套深度默认限制 3 层，子代理不共享当前对话历史与记忆。")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="task", type="string", required=True,
                          description="子任务描述"),
            ToolParameter(name="agent_type", type="string", required=False,
                          default="simple",
                          description="子代理类型（simple/react/...）"),
        ]

    # ------------------------------------------------------------
    # 深度保护
    # ------------------------------------------------------------

    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def _enter(self) -> bool:
        """进入委派（返回是否允许）。"""
        depth = self._depth()
        if depth >= self._max_depth:
            return False
        self._local.depth = depth + 1
        return True

    def _exit(self):
        self._local.depth = max(0, self._depth() - 1)

    # ------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------

    def run(self, args: dict) -> str:
        task = str(args.get("task", "")).strip()
        agent_type = str(args.get("agent_type", "simple")).strip() or "simple"
        if not task:
            return "错误：缺少必填参数 'task'"

        if not self._enter():
            return (f"错误：子代理嵌套超过最大深度 {self._max_depth}，"
                    f"已拒绝委派（防止无限递归）")

        try:
            available = self._factory.available_types()
            if agent_type not in available:
                return (f"错误：未知子代理类型 '{agent_type}'，"
                        f"可用: {', '.join(available) or '无'}")
            # 创建子代理并执行
            sub = self._factory.create(agent_type,
                                       name=f"sub_{agent_type}_{self._depth()}")
            result = sub.execute(task)
            if result.status == "success":
                return f"[子代理 {agent_type}] {result.output}"
            return f"[子代理 {agent_type}] 执行失败: {result.error}"
        except Exception as e:
            return f"子代理执行异常: {type(e).__name__}: {e}"
        finally:
            self._exit()

    def execute(self, args: dict) -> ToolResponse:
        """结构化执行：识别委派失败为 error 响应。"""
        import time
        start = time.perf_counter()
        output = self.run(args)
        failed = (output.startswith("错误")
                  or "执行失败" in output[:40]
                  or "执行异常" in output[:40])
        resp = (ToolResponse.error(output=output, error="sub_agent_failed")
                if failed else ToolResponse.success(output=output))
        resp.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        return resp


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    from src.agents.framework.agent_framework import BaseAgent, AgentFactory

    print("=" * 60)
    print("🧩 SubAgentTool 子代理委派演示")
    print("=" * 60)

    # 演示工厂：Echo 子代理（无 LLM）
    factory = AgentFactory()

    class EchoSub(BaseAgent):
        AGENT_TYPE = "echo_sub"
        DESCRIPTION = "回声子代理"

        def _execute(self, input_text: str, **kwargs) -> str:
            return f"子代理收到: {input_text}"

    factory.register("echo_sub", EchoSub, "回声子代理")

    from src.tools.framework.tool_system import ToolRegistry

    # 1. 注册 SubAgentTool
    print("--- 1) 注册 SubAgentTool ---")
    registry = ToolRegistry()
    sub_tool = SubAgentTool(factory)
    registry.register(sub_tool)
    print(f"   已注册: {sub_tool.name} | 最大深度: {sub_tool._max_depth}")

    # 2. 委派执行
    print("\n--- 2) 委派子任务 ---")
    resp = registry.execute_structured("SubAgent",
                                       {"task": "整理会议要点",
                                        "agent_type": "echo_sub"})
    print(f"   状态: {resp.status} | 输出: {resp.output}")
    resp2 = registry.execute_structured("SubAgent",
                                        {"task": "翻译一句话",
                                         "agent_type": "不存在的类型"})
    print(f"   未知类型: {resp2.status} | {resp2.output}")

    # 3. 深度保护（嵌套委派）
    print("\n--- 3) 深度保护（防止无限递归）---")

    class NestedAgent(BaseAgent):
        AGENT_TYPE = "nested"
        DESCRIPTION = "会自我委派的 Agent"

        def _execute(self, input_text: str, **kwargs) -> str:
            # 自己调用 SubAgentTool 再委派给自己 → 递归
            return sub_tool.run({"task": input_text, "agent_type": "nested"})

    factory.register("nested", NestedAgent, "嵌套委派")
    result = factory.create("nested").execute("开始递归")
    print(f"   深度 {sub_tool._max_depth} 内: {'(被深度保护拦截)' if '最大深度' in result.output else result.output[:80]}")
    print(f"   输出: {result.output[:120]}")

    print("\n✅ SubAgentTool 演示完成")
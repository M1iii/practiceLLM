"""
工具链管理器
============
支持将多个工具按顺序编排成链式工作流，前一步的输出可作为后续步骤的输入。
适配项目现有的 ToolRegistry（execute(name, args_dict) 接口）。

使用方式:
    from src.tools.framework.tool_chain_manager import ToolChain, ToolChainManager
    from src.tools.search.search_registry import create_search_registry

    registry = create_search_registry()
    manager = ToolChainManager(registry)

    chain = ToolChain("research", "搜索后计算")
    chain.add_step("AdvancedSearch", "{input}", output_key="search_result", input_key="query")
    chain.add_step("Calculator", "2 + 3", output_key="calc_result", input_key="expression")

    manager.register_chain(chain)
    result = manager.execute_chain("research", "AI Agent 发展趋势")
"""

import sys
import os


from typing import List, Dict, Any, Optional
from src.tools.framework.tool_system import ToolRegistry


class ToolChain:
    """
    工具链 - 支持多个工具的顺序执行。
    每个步骤的输出存入上下文，后续步骤可通过 {output_key} 引用。
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(
        self,
        tool_name: str,
        input_template: str,
        output_key: str = None,
        input_key: str = "input",
    ):
        """
        添加工具执行步骤。

        Args:
            tool_name: 工具名称（需与注册表中的工具名一致）
            input_template: 输入模板字符串，支持 {variable} 变量替换。
                            变量来自上下文，如 {input}、{search_result} 等。
            output_key: 本步骤输出结果存入上下文的键名，供后续步骤引用。
                        默认为 step_{序号}_result。
            input_key: 格式化后的字符串映射到工具的哪个参数。
                       如 AdvancedSearch 的 "query"、Calculator 的 "expression"。
        """
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result",
            "input_key": input_key,
        })
        return self

    def execute(self, registry: ToolRegistry, initial_input: str, context: Dict[str, Any] = None) -> str:
        """
        执行工具链。

        Args:
            registry: 工具注册表
            initial_input: 初始输入，存入上下文的 "input" 键
            context: 额外上下文变量
        Returns:
            最后一步的输出结果
        """
        context = context or {}
        context["input"] = initial_input

        print(f"🔗 开始执行工具链: {self.name}")
        print(f"   描述: {self.description}")
        print(f"   步骤数: {len(self.steps)}")

        for i, step in enumerate(self.steps, 1):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]
            input_key = step["input_key"]

            # 模板变量替换
            try:
                tool_input = input_template.format(**context)
            except KeyError as e:
                return f"❌ 工具链执行失败: 模板变量 {e} 未找到"

            preview = tool_input[:80] + "..." if len(tool_input) > 80 else tool_input
            print(f"\n  📌 步骤 {i}/{len(self.steps)}: {tool_name}")
            print(f"     输入参数 [{input_key}]: {preview}")

            # 调用注册表执行工具
            result = registry.execute(tool_name, {input_key: tool_input})
            context[output_key] = result

            result_preview = result[:100] + "..." if len(result) > 100 else result
            print(f"     ✅ 完成，输出键: {output_key}")
            print(f"     结果预览: {result_preview}")

        # 返回最后一步的结果
        final_result = context[self.steps[-1]["output_key"]]
        print(f"\n🎉 工具链 '{self.name}' 执行完成")
        return final_result


class ToolChainManager:
    """
    工具链管理器 - 管理多个工具链的注册与执行。
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        """注册工具链。"""
        self.chains[chain.name] = chain
        print(f"✅ 工具链 '{chain.name}' 已注册（{len(chain.steps)} 个步骤）")

    def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
        """
        执行指定的工具链。

        Args:
            chain_name: 工具链名称
            input_data: 初始输入数据
            context: 额外上下文变量
        Returns:
            工具链最后一步的输出结果
        """
        if chain_name not in self.chains:
            available = ", ".join(self.chains.keys()) or "无"
            return f"❌ 工具链 '{chain_name}' 不存在，可用: {available}"

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        """列出所有已注册的工具链名称。"""
        return list(self.chains.keys())

    def get_chain(self, chain_name: str) -> Optional[ToolChain]:
        """获取指定工具链。"""
        return self.chains.get(chain_name)


# ============================================================
# 使用示例
# ============================================================

def create_search_chain() -> ToolChain:
    """
    创建搜索工具链：搜索信息。
    单步骤示例，展示基本用法。
    """
    chain = ToolChain(
        name="quick_search",
        description="快速搜索信息并返回结果",
    )
    chain.add_step(
        tool_name="AdvancedSearch",
        input_template="{input}",
        output_key="search_result",
        input_key="query",
    )
    return chain


def create_research_chain() -> ToolChain:
    """
    创建研究工具链：搜索 → 计算验证。
    多步骤示例，展示工具间的数据传递。
    """
    chain = ToolChain(
        name="research_and_calculate",
        description="搜索信息后执行数学计算验证",
    )

    # 步骤1: 搜索信息
    chain.add_step(
        tool_name="AdvancedSearch",
        input_template="{input}",
        output_key="search_result",
        input_key="query",
    )

    # 步骤2: 执行计算（独立于搜索结果，演示多步骤）
    chain.add_step(
        tool_name="Calculator",
        input_template="{calc_expression}",
        output_key="calc_result",
        input_key="expression",
    )

    return chain


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    from src.tools.search.search_registry import create_search_registry

    # 创建注册表和管理器
    registry = create_search_registry()
    manager = ToolChainManager(registry)

    # 注册工具链
    manager.register_chain(create_search_chain())
    manager.register_chain(create_research_chain())

    print(f"\n📋 已注册工具链: {manager.list_chains()}")
    print()

    # 执行搜索工具链
    print("=" * 60)
    print("测试1: 快速搜索工具链")
    print("=" * 60)
    result = manager.execute_chain("quick_search", "什么是 AI Agent")
    print(f"\n最终结果:\n{result[:500]}...")
    print()

    # 执行研究工具链（搜索 + 计算）
    print("=" * 60)
    print("测试2: 搜索+计算工具链")
    print("=" * 60)
    result = manager.execute_chain(
        "research_and_calculate",
        "DeepSeek V4 最新消息",
        context={"calc_expression": "(100 + 200) * 3"}
    )
    print(f"\n最终结果:\n{result}")
"""
高级搜索工具注册表
==================
预配置的 ToolRegistry，包含多源搜索工具及常用辅助工具，
可直接供各类 Agent 使用。

使用方式:
    from tools.search.search_registry import create_search_registry
    registry = create_search_registry()
    result = registry.execute("AdvancedSearch", {"query": "AI Agent"})
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.framework.tool_system import (
    ToolRegistry,
    CalculatorTool,
    TimeTool,
)
from tools.search.advanced_search_tool import AdvancedSearchTool


def create_search_registry() -> ToolRegistry:
    """
    创建并返回一个包含高级搜索工具的注册表。

    包含工具:
      - AdvancedSearch: 多源搜索（博查搜索优先，Tavily 回退）
      - Calculator: 数学计算
      - Time: 时间查询
    """
    registry = ToolRegistry()

    # 注册多源搜索工具
    registry.register(AdvancedSearchTool())

    # 注册辅助工具
    registry.register(CalculatorTool())
    registry.register(TimeTool())

    return registry


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import json

    registry = create_search_registry()

    # 1. 工具列表
    print("=" * 60)
    print("📋 高级搜索工具注册表")
    print("=" * 60)
    print(registry.get_tools_description())
    print()

    # 2. OpenAI function calling 格式
    print("--- OpenAI Function Calling 格式 ---")
    print(json.dumps(registry.to_openai_format(), ensure_ascii=False, indent=2))
    print()

    # 3. 执行搜索（需要配置有效的 API Key）
    print("--- 执行搜索 ---")
    result = registry.execute("AdvancedSearch", {"query": "Python AI Agent 开发", "count": 3})
    print(result)
    print()

    # 4. 执行计算器（验证注册表中其他工具正常）
    print("--- 执行 Calculator ---")
    result = registry.execute("Calculator", {"expression": "2 ** 10"})
    print(f"结果: {result}")

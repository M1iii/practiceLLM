"""预配置的搜索工具注册表，包含多源搜索及辅助工具。"""

from src.tools.framework.tool_system import ToolRegistry, CalculatorTool, TimeTool
from src.tools.search.advanced_search_tool import AdvancedSearchTool


def create_search_registry() -> ToolRegistry:
    """创建含 AdvancedSearch / Calculator / Time 的注册表。"""
    registry = ToolRegistry()
    registry.register(AdvancedSearchTool())
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    return registry


if __name__ == "__main__":
    import json

    registry = create_search_registry()
    print("Tools:", registry.get_tools_description())
    print("OpenAI format:", json.dumps(registry.to_openai_format(), indent=2))
    print("Search:", registry.execute("AdvancedSearch", {"query": "AI Agent", "count": 3}))
    print("Calculator:", registry.execute("Calculator", {"expression": "2 ** 10"}))
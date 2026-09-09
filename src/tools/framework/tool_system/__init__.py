"""Tool 工具系统：抽象基类、函数包装器、注册表与示例工具。"""

from src.tools.framework.tool_system.base import ToolParameter, Tool, dual_protocol_execute
from src.tools.framework.tool_system.function_tool import FunctionTool
from src.tools.framework.tool_system.registry import ToolRegistry
from src.tools.framework.tool_system.examples import CalculatorTool, TimeTool, SearchTool, get_weekday

__all__ = [
    "ToolParameter", "Tool", "dual_protocol_execute",
    "FunctionTool", "ToolRegistry",
    "CalculatorTool", "TimeTool", "SearchTool", "get_weekday",
]
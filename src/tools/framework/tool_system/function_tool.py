"""FunctionTool：函数包装器，将普通 Python 函数快速封装为 Tool 对象。"""

import time
from typing import Callable, List

from src.tools.framework.tool_system.base import Tool, ToolParameter
from src.tools.framework.tool_response import ToolResponse


class FunctionTool(Tool):
    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        parameters: List[ToolParameter] = None,
    ):
        self._func = func
        self._name = name or func.__name__
        self._description = description or (func.__doc__ or "").strip() or f"函数 {self._name}"
        self._parameters = parameters or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def get_parameters(self) -> List[ToolParameter]:
        return self._parameters

    def run(self, args: dict) -> str:
        try:
            result = self._func(**args)
            return str(result)
        except Exception as e:
            return f"执行失败: {e}"

    def execute(self, args: dict) -> ToolResponse:
        start = time.perf_counter()
        try:
            result = self._func(**args)
            resp = ToolResponse.success(output=str(result), data=result)
        except Exception as e:
            resp = ToolResponse.error(
                output=f"执行失败: {e}",
                error=f"{type(e).__name__}: {e}",
            )
        resp.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        return resp
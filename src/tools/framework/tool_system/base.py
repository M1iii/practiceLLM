"""Tool 工具抽象基类与参数定义。"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable, Union

from src.tools.framework.tool_response import ToolResponse


@dataclass
class ToolParameter:
    """
    工具参数定义，描述单个参数的元数据。
    每个工具通过 get_parameters() 返回 List[ToolParameter]，
    告诉调用者（Agent 或 LLM）自己需要什么参数。
    """
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    items: Optional["ToolParameter"] = None


class Tool(ABC):
    """
    所有工具的抽象基类。
    元数据：name, description
    统一接口：run(args: dict) -> str，接受字典参数并返回字符串结果
    参数声明：get_parameters() -> List[ToolParameter]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        ...

    @abstractmethod
    def run(self, args: dict) -> str:
        ...

    @staticmethod
    def _param_to_schema(param: "ToolParameter") -> dict:
        schema: dict = {
            "type": param.type,
            "description": param.description,
        }
        if param.enum is not None:
            schema["enum"] = param.enum
        if param.type == "array" and param.items is not None:
            schema["items"] = Tool._param_to_schema(param.items)
        if param.default is not None:
            schema["default"] = param.default
        return schema

    def to_json_schema(self) -> dict:
        properties = {}
        required = []
        for param in self.get_parameters():
            properties[param.name] = self._param_to_schema(param)
            if param.required:
                required.append(param.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    to_openai_schema = to_json_schema

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            }
        }

    def run_with_args(self, args: dict) -> str:
        return str(self.run(args or {}))

    def __str__(self):
        return f"- {self.name}: {self.description}"

    def execute(self, args: dict) -> ToolResponse:
        start = time.perf_counter()
        try:
            output = self.run(args)
            resp = ToolResponse.success(output=output)
        except Exception as e:
            resp = ToolResponse.error(
                output=f"执行失败: {e}",
                error=f"{type(e).__name__}: {e}",
            )
        resp.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        return resp


def dual_protocol_execute(tool: "Tool", action, **kwargs) -> Any:
    if isinstance(action, dict):
        return Tool.execute(tool, action)
    return tool.run({"action": action, **kwargs})
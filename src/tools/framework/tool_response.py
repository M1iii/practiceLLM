"""ToolResponse：工具执行的结构化返回协议。

统一返回格式 status/output/data/error/duration_ms/extra，
与旧版字符串返回兼容（str(resp) == output）。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResponse:
    status: str = "success"
    output: str = ""
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, output: str = "", data: Any = None, **extra) -> "ToolResponse":
        return cls(status="success", output=output, data=data, **extra)

    @classmethod
    def warning(cls, output: str = "", error: Optional[str] = None,
                data: Any = None, **extra) -> "ToolResponse":
        return cls(status="warning", output=output, error=error, data=data, **extra)

    @classmethod
    def error(cls, output: str = "", error: Optional[str] = None,
              **extra) -> "ToolResponse":
        return cls(status="error", output=output, error=error, **extra)

    @property
    def ok(self) -> bool:
        return self.status != "error"

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    def __post_init__(self):
        if not isinstance(self.error, (str, type(None))):
            self.error = None
        if not isinstance(self.output, str):
            self.output = str(self.output)

    def __str__(self) -> str:
        return self.output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "extra": self.extra,
        }


if __name__ == "__main__":
    r1 = ToolResponse.success(output="ok", data={"v": 42})
    r2 = ToolResponse.error(output="fail", error="divide by zero")
    r3 = ToolResponse.warning(output="warn", error="stale data")
    print(f"success: {r1.status} | str={r1} | ok={r1.ok} | is_error={r1.is_error}")
    print(f"error:   {r2.status} | error={r2.error} | ok={r2.ok} | is_error={r2.is_error}")
    print(f"warning: {r3.status} | error={r3.error} | ok={r3.ok}")

    from src.tools.framework.tool_system import FunctionTool

    def add(a: int, b: int) -> int:
        return a + b

    def divide(a: int, b: int) -> float:
        return a / b

    tool = FunctionTool(add, "add", "", [])
    resp = tool.execute({"a": 2, "b": 3})
    print(f"tool(add): {resp.status} | {resp.output} | {resp.duration_ms}ms")

    bad = FunctionTool(divide, "divide", "", [])
    resp2 = bad.execute({"a": 1, "b": 0})
    print(f"tool(divide): {resp2.status} | {resp2.error}")
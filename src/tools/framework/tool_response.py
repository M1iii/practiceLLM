"""ToolResponse：工具执行的结构化返回协议。

统一工具返回格式（status / output / data / error / duration_ms / extra），
与旧版字符串返回完全兼容（str(ToolResponse) == output）。

设计：
  - status: "success" / "error" / "warning"（warning 表示执行完成但有告警）
  - output: 人类可读输出（保持与 Tool.run() 字符串返回兼容）
  - data: 结构化数据（LLM/上层可解析）
  - error: 错误信息（status=error 时）
  - duration_ms: 执行耗时
  - ok / is_error 便捷属性

使用方式:
    resp = ToolResponse.success(output="结果", data={"n": 1})
    resp = ToolResponse.error(output="失败", error="ValueError: x")
    str(resp)   # "结果"（兼容旧接口）
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional



@dataclass
class ToolResponse:
    """工具执行的结构化返回。"""
    status: str = "success"                       # success / error / warning
    output: str = ""                              # 人类可读输出（兼容 str 返回）
    data: Any = None                              # 结构化数据
    error: Optional[str] = None                   # 错误信息
    duration_ms: float = 0.0                      # 执行耗时
    extra: Dict[str, Any] = field(default_factory=dict)  # 附加信息

    # --- 便捷构造 ---

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

    # --- 便捷属性 ---

    @property
    def ok(self) -> bool:
        """是否成功（success 或 warning 视为可接受）。"""
        return self.status != "error"

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    # --- 兼容旧接口 ---

    def __post_init__(self):
        """字段归一化：error 字段与同名 error() 工厂方法冲突，
        dataclass 可能把类方法当作字段默认值，这里强制归一化为 None/str。"""
        if not isinstance(self.error, (str, type(None))):
            self.error = None
        if not isinstance(self.output, str):
            self.output = str(self.output)

    def __str__(self) -> str:
        """与旧版字符串返回兼容：str(response) == output。"""
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


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📦 ToolResponse 协议演示")
    print("=" * 60)

    # 成功响应
    r1 = ToolResponse.success(output="计算结果: 42", data={"value": 42})
    print(f"  成功: status={r1.status} | str={str(r1)} | data={r1.data}")
    print(f"    ok={r1.ok} | duration={r1.duration_ms}ms")

    # 错误响应
    r2 = ToolResponse.error(output="计算失败", error="ZeroDivisionError: division by zero")
    print(f"  错误: status={r2.status} | str={str(r2)} | error={r2.error}")
    print(f"    is_error={r2.is_error}")

    # 警告响应
    r3 = ToolResponse.warning(output="执行完成但结果可能不准确", error="数据过期")
    print(f"  警告: status={r3.status} | ok={r3.ok}")

    # 与工具集成（FunctionTool 包装）
    from src.tools.framework.tool_system import FunctionTool

    def add(a: int, b: int) -> int:
        return a + b

    def divide(a: int, b: int) -> float:
        return a / b   # b=0 时抛异常

    tool = FunctionTool(add, "add", "加法", [])
    resp = tool.execute({"a": 2, "b": 3})
    print(f"\n  FunctionTool(add).execute → {resp.status} | {resp.output} "
          f"| {resp.duration_ms}ms")

    bad = FunctionTool(divide, "divide", "除法", [])
    resp2 = bad.execute({"a": 1, "b": 0})
    print(f"  FunctionTool(divide).execute → {resp2.status} | {resp2.error}")

    print("\n✅ ToolResponse 演示完成")
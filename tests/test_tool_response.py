"""ToolResponse 协议测试。"""
import pytest
from src.tools.framework.tool_response import ToolResponse
from src.tools.framework.tool_system import FunctionTool, ToolRegistry, ToolParameter


def test_success_response():
    r = ToolResponse.success(output="42", data={"value": 42})
    assert r.status == "success"
    assert r.ok is True and r.is_error is False
    assert r.data == {"value": 42}
    assert str(r) == "42"          # 兼容旧字符串接口


def test_error_response():
    r = ToolResponse.error(output="失败", error="ZeroDivisionError: x")
    assert r.status == "error"
    assert r.is_error is True and r.ok is False
    assert r.error == "ZeroDivisionError: x"


def test_warning_response():
    r = ToolResponse.warning(output="可能不准确", error="数据过期")
    assert r.status == "warning"
    assert r.ok is True              # warning 视为可接受


def test_error_field_not_conflicting_with_classmethod():
    """字段 error 与同名工厂方法冲突的回归测试。"""
    r = ToolResponse.success(output="ok")
    assert isinstance(r.error, (str, type(None)))   # 不能是绑定方法
    r2 = ToolResponse()
    assert r2.error is None
    # 显式传入字符串错误
    r3 = ToolResponse.success(output="x", error="显式错误")
    assert r3.error == "显式错误"


def test_to_dict_serializable():
    import json
    r = ToolResponse.success(output="hello", data=[1, 2])
    json.dumps(r.to_dict(), ensure_ascii=False)     # 不应抛异常


def test_function_tool_structured_execute():
    def add(a, b):
        return a + b
    tool = FunctionTool(add, "add", "加法", [])
    r = tool.execute({"a": 2, "b": 3})
    assert r.status == "success"
    assert r.data == 5


def test_function_tool_error_capture():
    def divide(a, b):
        return a / b
    tool = FunctionTool(divide, "divide", "除法", [])
    r = tool.execute({"a": 1, "b": 0})
    assert r.status == "error"
    assert "ZeroDivisionError" in r.error


def test_registry_execute_structured(tmp_path):
    def add(a, b):
        return a + b
    registry = ToolRegistry(enable_circuit_breaker=False)
    registry.register(add, name="Add",
                      parameters=[ToolParameter(name="a", type="number",
                                                description="a"),
                                  ToolParameter(name="b", type="number",
                                                description="b")])
    r = registry.execute_structured("Add", {"a": 1, "b": 2})
    assert r.status == "success" and r.output == "3"
    # 旧接口兼容
    assert registry.execute("Add", {"a": 1, "b": 2}) == "3"
    # 未知工具 / 缺参
    assert registry.execute_structured("Nope", {}).status == "error"
    assert registry.execute_structured("Add", {}).status == "error"

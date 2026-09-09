"""示例工具（CalculatorTool / TimeTool / SearchTool）与演示代码。"""

import ast
import operator
import json
from datetime import datetime
from typing import List

from src.tools.framework.tool_system.base import Tool, ToolParameter
from src.tools.framework.tool_system.registry import ToolRegistry


class CalculatorTool(Tool):
    """安全计算器工具，支持基本数学运算。"""

    @property
    def name(self) -> str:
        return "Calculator"

    @property
    def description(self) -> str:
        return ("进行数学计算，支持加减乘除、幂运算、取余等。"
                "适用：数值表达式计算、数学问题求解。"
                "不适用：需要外部实时数据的计算（如汇率/股票价格，请用 AdvancedSearch）、"
                "日期时间查询（请用 Time）。"
                "注意：幂运算请用 **（如 2**10），不要用 ^（^ 是位异或）")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="数学表达式，如 2+3*4、(15+27)*3",
                required=True,
            )
        ]

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def run(self, args: dict) -> str:
        expression = args.get("expression", "")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = self._eval_node(tree.body)
            return str(result)
        except Exception as e:
            return f"计算失败: {e}"

    def _eval_node(self, node) -> float:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_func = self._OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_func = self._OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
            return op_func(operand)
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


class TimeTool(Tool):
    """时间查询工具，返回当前日期和时间。"""

    @property
    def name(self) -> str:
        return "Time"

    @property
    def description(self) -> str:
        return ("获取当前日期和时间，支持自定义 strftime 格式。"
                "适用：查询当前时间/日期。"
                "不适用：未来或历史日期、时区换算、星期查询（请用 Weekday）。"
                "注意：返回系统本地时间，不支持时区参数。")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="format",
                type="string",
                description="时间格式字符串，如 %Y-%m-%d %H:%M:%S，为空则使用默认格式",
                required=False,
                default=None,
            )
        ]

    def run(self, args: dict) -> str:
        fmt = (args.get("format") or "").strip()
        if fmt:
            try:
                return datetime.now().strftime(fmt)
            except Exception:
                return f"格式无效，当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class SearchTool(Tool):
    """演示工具：展示数组类型和枚举值参数。"""

    @property
    def name(self) -> str:
        return "Search"

    @property
    def description(self) -> str:
        return ("演示级搜索工具（不执行真实联网），展示数组与枚举参数用法。"
                "适用：框架演示、参数 schema 示例。"
                "不适用：真实联网搜索（请用 AdvancedSearch）。")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="keywords",
                type="array",
                description="搜索关键词列表",
                required=True,
                items=ToolParameter(
                    name="keyword",
                    type="string",
                    description="单个关键词",
                    required=True,
                ),
            ),
            ToolParameter(
                name="sort",
                type="string",
                description="排序方式",
                required=False,
                default="relevance",
                enum=["relevance", "date", "popularity"],
            ),
            ToolParameter(
                name="limit",
                type="number",
                description="返回结果数量上限",
                required=False,
                default=10,
            ),
        ]

    def run(self, args: dict) -> str:
        keywords = args.get("keywords", [])
        sort = args.get("sort", "relevance")
        limit = args.get("limit", 10)
        return f"搜索关键词: {keywords}, 排序: {sort}, 数量上限: {limit}"


def get_weekday() -> str:
    """获取今天是星期几。"""
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return weekdays[datetime.now().weekday()]


if __name__ == "__main__":
    registry = ToolRegistry()

    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(SearchTool())
    registry.register(
        get_weekday,
        name="Weekday",
        description="获取今天是星期几",
        parameters=[],
    )

    print("=" * 60)
    print("工具详细描述:")
    print("=" * 60)
    print(registry.get_tools_description())
    print()
    print(f"工具列表: {registry.list_tools()}")
    print()

    print("--- 执行 Calculator ---")
    result = registry.execute("Calculator", {"expression": "(15 + 27) * 3"})
    print(f"结果: {result}")
    print()

    print("--- 执行 Time ---")
    result = registry.execute("Time", {})
    print(f"结果: {result}")
    print()

    print("--- 执行 Search ---")
    result = registry.execute("Search", {
        "keywords": ["Python", "AI Agent"],
        "sort": "date",
        "limit": 5,
    })
    print(f"结果: {result}")
    print()

    print("--- 执行 Weekday ---")
    result = registry.execute("Weekday", {})
    print(f"结果: {result}")
    print()

    print("--- 参数验证（缺少必填参数）---")
    result = registry.execute("Calculator", {})
    print(f"结果: {result}")
    print()

    print("=" * 60)
    print("OpenAI Function Calling 格式（含数组与枚举）:")
    print("=" * 60)
    print(json.dumps(registry.to_openai_format(), ensure_ascii=False, indent=2))

    print()
    print("=" * 60)
    print("ToolResponse 协议 + 熔断器集成演示")
    print("=" * 60)

    print("--- execute_structured（ToolResponse 协议）---")
    resp = registry.execute_structured("Calculator", {"expression": "6 * 7"})
    print(f"   Calculator -> {resp.status} | {resp.output} | {resp.duration_ms}ms")
    resp_err = registry.execute_structured("Calculator", {})
    print(f"   缺参 -> {resp_err.status} | {resp_err.error}")
    resp_unk = registry.execute_structured("Unknown", {})
    print(f"   未知工具 -> {resp_unk.status} | {resp_unk.error}")

    print("--- 熔断器（连续失败 -> 打开 -> 快速失败）---")

    def flaky(a: int, b: int) -> int:
        raise ConnectionError("上游服务不稳定")

    registry.register(flaky, name="Flaky", description="不稳定工具",
                      parameters=[ToolParameter(name="a", type="number",
                                                description="参数 a"),
                                  ToolParameter(name="b", type="number",
                                                description="参数 b")])
    for i in range(1, 5):
        r = registry.execute_structured("Flaky", {"a": 1, "b": 2})
        print(f"   调用{i}: {r.status} | {r.error}")
        if r.status == "error" and r.error == "circuit_open":
            print("     熔断已打开（快速失败，拒绝上游调用）")
            break
    print(f"   熔断器状态: {registry.circuit_status()['Flaky']['state']}")
    print(f"   旧接口兼容: registry.execute('Flaky', ...) = "
          f"{registry.execute('Flaky', {'a': 1, 'b': 2})[:30]}...")
    registry.circuit_reset("Flaky")
    print(f"   circuit_reset 后: {registry.circuit_status()['Flaky']['state']}")
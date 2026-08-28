import ast
import operator
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.framework.tool_response import ToolResponse
from tools.framework.circuit_breaker import CircuitBreaker, CircuitOpenError


# ============================================================
# 1. ToolParameter：工具参数定义类
# ============================================================

@dataclass
class ToolParameter :
    """
    工具参数定义，描述单个参数的元数据。
    每个工具通过 get_parameters() 返回 List[ToolParameter]，
    告诉调用者（Agent 或 LLM）自己需要什么参数。
    """
    name: str
    type: str                        # "string", "number", "boolean", "array", "object" 等
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None       # 枚举可选值，如 ["asc", "desc"]
    items: Optional["ToolParameter"] = None  # 数组元素的参数定义（当 type="array" 时使用）


# ============================================================
# 2. Tool：工具基类
# ============================================================

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
        """工具名称，唯一标识。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，帮助 LLM 理解何时使用该工具。"""
        ...

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """声明工具所需的参数列表。"""
        ...

    @abstractmethod
    def run(self, args: dict) -> str:
        """执行工具逻辑，接受字典参数，返回字符串结果。"""
        ...

    # --- 便捷方法：格式转换 ---

    @staticmethod
    def _param_to_schema(param: "ToolParameter") -> dict:
        """将单个 ToolParameter 转换为 JSON Schema 片段。"""
        schema: dict = {
            "type": param.type,
            "description": param.description,
        }

        # 枚举值
        if param.enum is not None:
            schema["enum"] = param.enum

        # 数组类型：递归处理 items
        if param.type == "array" and param.items is not None:
            schema["items"] = Tool._param_to_schema(param.items)

        # 默认值（区分"未设置"和"显式设置为 None/False/0"）
        if param.default is not None:
            schema["default"] = param.default

        return schema

    def to_json_schema(self) -> dict:
        """将参数列表转换为 JSON Schema 格式。"""
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

    # to_openai_schema 是 to_json_schema 的语义别名
    to_openai_schema = to_json_schema

    def to_openai_format(self) -> dict:
        """转换为 OpenAI function calling 的完整工具 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            }
        }

    # --- 旧框架兼容桥 ---

    def run_with_args(self, args: dict) -> str:
        """兼容旧框架执行入口：OpenAI function calling 参数字典 → run(args)。

        旧框架（tools.py）的 Tool 以 run_with_args(arguments) 执行，
        新框架以 run(args: dict) 执行；此桥让新旧调用方都能驱动新工具。
        """
        return str(self.run(args or {}))

    def __str__(self):
        return f"- {self.name}: {self.description}"

    # --- 统一结构化执行（ToolResponse 协议） ---

    def execute(self, args: dict) -> ToolResponse:
        """统一执行入口：包装 run()，计时 + 异常归一化为 ToolResponse。

        兼容旧接口：str(返回值) == run() 的输出文本。
        """
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
    """旧式 execute(action, **kwargs) 的双协议适配器。

    - action 为 dict（新 ToolResponse 协议）→ 走 Tool.execute，返回 ToolResponse
    - action 为 str（旧协议）→ 组装 {"action": action, **kwargs} 委托 run()

    用途：TerminalTool/NoteTool/RagTool/MemoryTool 的旧 execute 方法
    会遮蔽基类 execute(args)；本适配器让两种调用方式都可用。
    """
    if isinstance(action, dict):
        return Tool.execute(tool, action)   # 新协议：结构化执行
    return tool.run({"action": action, **kwargs})   # 旧协议：组装后委托 run


# ============================================================
# FunctionTool：函数包装器（用于函数直接注册）
# ============================================================

class FunctionTool(Tool):
    """
    函数包装器，将普通 Python 函数快速封装为 Tool 对象。
    适合简单工具的快速集成，无需继承 Tool 类。
    """

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
        """结构化执行：直接调用函数，返回带 data 的 ToolResponse。"""
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


# ============================================================
# 3. ToolRegistry：工具注册表
# ============================================================

class ToolRegistry:
    """
    工具注册表，统一管理工具的注册、发现和执行。

    支持两种注册方式：
    - Tool 对象注册：适合复杂工具，支持完整的参数定义和验证
    - 函数直接注册：适合简单工具，快速集成现有函数
    """

    def __init__(self, tools: Optional[List[Tool]] = None,
                 enable_circuit_breaker: bool = True,
                 breaker_failure_threshold: int = 3,
                 breaker_recovery_timeout: float = 30.0,
                 observer: Optional[Callable[[str, dict, ToolResponse], None]] = None,
                 owner: Optional[str] = None,
                 tool_filter: Optional[Any] = None):
        """
        Args:
            tools: 初始工具列表
            enable_circuit_breaker: 是否启用熔断器（按工具名独立熔断）
            breaker_failure_threshold: 连续失败多少次后熔断
            breaker_recovery_timeout: 熔断后多少秒进入半开试探
            observer: 工具执行观察者回调（name, args, ToolResponse），
                      用于 TraceLogger 等可观测性接入
            owner: 注册表所属 Agent 角色（配合 tool_filter 做权限校验）
            tool_filter: ToolFilter 实例（按 owner 裁剪可见工具）
        """
        self._tools: Dict[str, Tool] = {}
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.enable_circuit_breaker = enable_circuit_breaker
        self._breaker_failure_threshold = max(1, breaker_failure_threshold)
        self._breaker_recovery_timeout = max(0.1, float(breaker_recovery_timeout))
        self._observer = observer
        self.owner = owner
        self.tool_filter = tool_filter
        if tools:
            for tool in tools:
                self.register(tool)

    # --- 注册 ---

    def register(
        self,
        tool: Union[Tool, Callable],
        name: str = None,
        description: str = None,
        parameters: List[ToolParameter] = None,
    ) -> Tool:
        """
        注册工具，自动判断类型：
        - 传入 Tool 对象 → 直接注册
        - 传入函数 → 自动包装为 FunctionTool 后注册
        """
        if isinstance(tool, Tool):
            self._tools[tool.name] = tool
            return tool
        elif callable(tool):
            wrapped = FunctionTool(tool, name, description, parameters)
            self._tools[wrapped.name] = wrapped
            return wrapped
        else:
            raise TypeError(f"不支持的工具类型: {type(tool)}，期望 Tool 对象或可调用函数")

    # --- 执行 ---

    def execute(self, name: str, args: dict) -> str:
        """执行指定工具（旧接口），返回字符串（兼容历史调用方）。"""
        return self.execute_structured(name, args).output

    def execute_structured(self, name: str, args: dict,
                           owner: Optional[str] = None) -> ToolResponse:
        """执行指定工具（含熔断器保护 + 工具过滤 + 观察者通知）。"""
        resp = self._execute_structured_inner(name, args, owner)
        if self._observer is not None:
            try:
                self._observer(name, args, resp)
            except Exception:
                pass   # 观察者异常不影响工具执行
        return resp

    def _execute_structured_inner(self, name: str, args: dict,
                                  owner: Optional[str] = None) -> ToolResponse:
        """执行核心逻辑（供 execute_structured 包装）。"""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys())
            return ToolResponse.error(
                output=f"错误：未找到工具 '{name}'，可用工具：{available}",
                error=f"unknown_tool: {name}")

        # 工具过滤（按角色权限）
        role = owner if owner is not None else self.owner
        if role and self.tool_filter is not None \
                and not self.tool_filter.is_allowed(role, name):
            return ToolResponse.error(
                output=f"错误：Agent 角色 '{role}' 无权使用工具 '{name}'",
                error=f"agent_forbidden: {role}.{name}")

        # 参数验证：检查必填参数是否提供
        for param in tool.get_parameters():
            if param.required and param.name not in args:
                if param.default is not None:
                    args[param.name] = param.default
                else:
                    return ToolResponse.error(
                        output=f"错误：缺少必填参数 '{param.name}'",
                        error=f"missing_param: {param.name}")

        # 熔断器保护（显式门控：check + record_success/record_failure）
        if self.enable_circuit_breaker:
            breaker = self._breaker_for(name)
            try:
                breaker.check()
                resp = tool.execute(args)
            except CircuitOpenError as e:
                return ToolResponse.error(
                    output=str(e), error="circuit_open",
                    extra={"circuit": breaker.status()})
            # 错误响应计为失败，成功计为成功
            if resp.is_error:
                breaker.record_failure(resp.error)
            else:
                breaker.record_success()
            return resp

        return tool.execute(args)

    # --- 熔断器 ---

    def _breaker_for(self, name: str) -> CircuitBreaker:
        """获取（或创建）工具对应的熔断器。"""
        breaker = self._breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                name=name,
                failure_threshold=self._breaker_failure_threshold,
                recovery_timeout=self._breaker_recovery_timeout,
            )
            self._breakers[name] = breaker
        return breaker

    def circuit_status(self) -> Dict[str, Dict[str, Any]]:
        """所有工具熔断器状态。"""
        return {name: b.status() for name, b in self._breakers.items()}

    def circuit_reset(self, name: Optional[str] = None) -> None:
        """重置熔断器（指定工具或全部）。"""
        if name is not None:
            breaker = self._breakers.get(name)
            if breaker:
                breaker.reset()
        else:
            for breaker in self._breakers.values():
                breaker.reset()

    # --- 发现 ---

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具对象。"""
        return self._tools.get(name)

    def find_tool(self, name: str, case_insensitive: bool = True) -> Optional[Tool]:
        """按名称查找工具；case_insensitive=True 时兼容大小写差异（LLM 输出不可控）。

        供 Agent 统一使用：LLM 可能返回 Calculator/calculator/Calculator 等变体。
        """
        tool = self._tools.get(name)
        if tool or not case_insensitive:
            return tool
        for n in self._tools:
            if n.lower() == name.lower():
                return self._tools[n]
        return None

    def get_tools(self, owner: Optional[str] = None) -> List[Tool]:
        """获取工具对象列表（可指定 owner 走工具过滤视图）。"""
        names = self.list_tools(owner=owner)
        return [self._tools[n] for n in names if n in self._tools]

    def list_tools(self, owner: Optional[str] = None) -> List[str]:
        """列出已注册工具名（可指定 owner 走工具过滤视图）。"""
        names = list(self._tools.keys())
        role = owner if owner is not None else self.owner
        if role and self.tool_filter is not None:
            return self.tool_filter.get_visible(role, names)
        return names

    def has_tools(self) -> bool:
        """检查是否有已注册的工具。"""
        return len(self._tools) > 0

    def remove(self, name: str) -> bool:
        """移除一个工具，返回是否成功。"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tools_description(self) -> str:
        """
        返回所有工具的详细描述文本，包含参数信息，供 Agent 系统提示词使用。
        格式示例:
            Calculator: 进行数学计算
              参数:
                - expression (string, 必填): 数学表达式，如 2+3*4
            Time: 获取当前日期和时间
              参数:
                - format (string, 可选): 时间格式字符串
        """
        if not self._tools:
            return "暂无可用工具"

        blocks = []
        for tool in self._tools.values():
            lines = [f"{tool.name}: {tool.description}"]
            params = tool.get_parameters()
            if params:
                lines.append("  参数:")
                for p in params:
                    tag = "必填" if p.required else "可选"
                    line = f"    - {p.name} ({p.type}, {tag}): {p.description}"
                    if p.enum is not None:
                        line += f"  可选值: {p.enum}"
                    if p.default is not None:
                        line += f"  默认: {p.default}"
                    lines.append(line)
            else:
                lines.append("  参数: 无")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def to_openai_format(self) -> List[dict]:
        """将所有工具转换为 OpenAI function calling 格式。"""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def to_openai_schema(self) -> List[dict]:
        """to_openai_format 的语义别名。"""
        return self.to_openai_format()

    def clear(self):
        """清空所有注册的工具。"""
        self._tools.clear()


# ============================================================
# 示例工具：使用新 Tool 基类实现
# ============================================================

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


# ============================================================
# 示例：函数直接注册
# ============================================================

def get_weekday() -> str:
    """获取今天是星期几。"""
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return weekdays[datetime.now().weekday()]


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import json

    registry = ToolRegistry()

    # --- 方式一：Tool 对象注册 ---
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(SearchTool())

    # --- 方式二：函数直接注册 ---
    registry.register(
        get_weekday,
        name="Weekday",
        description="获取今天是星期几",
        parameters=[],  # 无参数
    )

    # 1. 工具详细描述（增强版 get_tools_description）
    print("=" * 60)
    print("📋 工具详细描述:")
    print("=" * 60)
    print(registry.get_tools_description())
    print()
    print(f"工具列表: {registry.list_tools()}")
    print()

    # 2. 执行 Calculator
    print("--- 执行 Calculator ---")
    result = registry.execute("Calculator", {"expression": "(15 + 27) * 3"})
    print(f"结果: {result}")
    print()

    # 3. 执行 Time
    print("--- 执行 Time ---")
    result = registry.execute("Time", {})
    print(f"结果: {result}")
    print()

    # 4. 执行 Search（数组 + 枚举参数）
    print("--- 执行 Search ---")
    result = registry.execute("Search", {
        "keywords": ["Python", "AI Agent"],
        "sort": "date",
        "limit": 5,
    })
    print(f"结果: {result}")
    print()

    # 5. 执行 Weekday（函数注册的工具）
    print("--- 执行 Weekday ---")
    result = registry.execute("Weekday", {})
    print(f"结果: {result}")
    print()

    # 6. 参数验证演示
    print("--- 参数验证（缺少必填参数）---")
    result = registry.execute("Calculator", {})
    print(f"结果: {result}")
    print()

    # 7. 转换为 OpenAI function calling 格式（增强版 schema）
    print("=" * 60)
    print("📋 OpenAI Function Calling 格式（含数组与枚举）:")
    print("=" * 60)
    print(json.dumps(registry.to_openai_format(), ensure_ascii=False, indent=2))

    # 8. ToolResponse 协议 + 熔断器集成
    print()
    print("=" * 60)
    print("🛡️ ToolResponse 协议 + 熔断器集成演示")
    print("=" * 60)

    # 8a. 结构化执行（ToolResponse）
    print("--- 8a) execute_structured（ToolResponse 协议）---")
    resp = registry.execute_structured("Calculator", {"expression": "6 * 7"})
    print(f"   Calculator → {resp.status} | {resp.output} | {resp.duration_ms}ms")
    resp_err = registry.execute_structured("Calculator", {})
    print(f"   缺参 → {resp_err.status} | {resp_err.error}")
    resp_unk = registry.execute_structured("Unknown", {})
    print(f"   未知工具 → {resp_unk.status} | {resp_unk.error}")

    # 8b. 熔断器：连续失败 → 打开 → 快速失败
    print("--- 8b) 熔断器（连续失败 → 打开 → 快速失败）---")

    def flaky(a: int, b: int) -> int:
        raise ConnectionError("上游服务不稳定")   # 每次必失败

    registry.register(flaky, name="Flaky", description="不稳定工具",
                      parameters=[ToolParameter(name="a", type="number",
                                                description="参数 a"),
                                  ToolParameter(name="b", type="number",
                                                description="参数 b")])
    for i in range(1, 5):
        r = registry.execute_structured("Flaky", {"a": 1, "b": 2})
        print(f"   调用{i}: {r.status} | {r.error}")
        if r.status == "error" and r.error == "circuit_open":
            print(f"     🛑 熔断已打开（快速失败，拒绝上游调用）✅")
            break
    print(f"   熔断器状态: {registry.circuit_status()['Flaky']['state']}")
    print(f"   旧接口兼容: registry.execute('Flaky', ...) = "
          f"{registry.execute('Flaky', {'a': 1, 'b': 2})[:30]}...")
    registry.circuit_reset("Flaky")
    print(f"   circuit_reset 后: {registry.circuit_status()['Flaky']['state']} ✅")

"""ToolRegistry：工具注册表，统一管理工具的注册、发现和执行。"""

from typing import Dict, List, Optional, Any, Callable, Union

from src.tools.framework.tool_system.base import Tool, ToolParameter
from src.tools.framework.tool_system.function_tool import FunctionTool
from src.tools.framework.tool_response import ToolResponse
from src.tools.framework.circuit_breaker import CircuitBreaker, CircuitOpenError


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
        return self.execute_structured(name, args).output

    def execute_structured(self, name: str, args: dict,
                           owner: Optional[str] = None) -> ToolResponse:
        resp = self._execute_structured_inner(name, args, owner)
        if self._observer is not None:
            try:
                self._observer(name, args, resp)
            except Exception:
                pass
        return resp

    def _execute_structured_inner(self, name: str, args: dict,
                                  owner: Optional[str] = None) -> ToolResponse:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys())
            return ToolResponse.error(
                output=f"错误：未找到工具 '{name}'，可用工具：{available}",
                error=f"unknown_tool: {name}")

        role = owner if owner is not None else self.owner
        if role and self.tool_filter is not None \
                and not self.tool_filter.is_allowed(role, name):
            return ToolResponse.error(
                output=f"错误：Agent 角色 '{role}' 无权使用工具 '{name}'",
                error=f"agent_forbidden: {role}.{name}")

        for param in tool.get_parameters():
            if param.required and param.name not in args:
                if param.default is not None:
                    args[param.name] = param.default
                else:
                    return ToolResponse.error(
                        output=f"错误：缺少必填参数 '{param.name}'",
                        error=f"missing_param: {param.name}")

        if self.enable_circuit_breaker:
            breaker = self._breaker_for(name)
            try:
                breaker.check()
                resp = tool.execute(args)
            except CircuitOpenError as e:
                return ToolResponse.error(
                    output=str(e), error="circuit_open",
                    extra={"circuit": breaker.status()})
            if resp.is_error:
                breaker.record_failure(resp.error)
            else:
                breaker.record_success()
            return resp

        return tool.execute(args)

    # --- 熔断器 ---

    def _breaker_for(self, name: str) -> CircuitBreaker:
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
        return {name: b.status() for name, b in self._breakers.items()}

    def circuit_reset(self, name: Optional[str] = None) -> None:
        if name is not None:
            breaker = self._breakers.get(name)
            if breaker:
                breaker.reset()
        else:
            for breaker in self._breakers.values():
                breaker.reset()

    # --- 发现 ---

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def find_tool(self, name: str, case_insensitive: bool = True) -> Optional[Tool]:
        tool = self._tools.get(name)
        if tool or not case_insensitive:
            return tool
        for n in self._tools:
            if n.lower() == name.lower():
                return self._tools[n]
        return None

    def get_tools(self, owner: Optional[str] = None) -> List[Tool]:
        names = self.list_tools(owner=owner)
        return [self._tools[n] for n in names if n in self._tools]

    def list_tools(self, owner: Optional[str] = None) -> List[str]:
        names = list(self._tools.keys())
        role = owner if owner is not None else self.owner
        if role and self.tool_filter is not None:
            return self.tool_filter.get_visible(role, names)
        return names

    def has_tools(self) -> bool:
        return len(self._tools) > 0

    def remove(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tools_description(self) -> str:
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
        return [tool.to_openai_format() for tool in self._tools.values()]

    to_openai_schema = to_openai_format

    def clear(self):
        self._tools.clear()
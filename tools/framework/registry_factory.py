"""registry_factory.py — build_all_tools_registry 工厂。

一键将项目全部工具装配进一个 ToolRegistry，并按角色（ToolFilter）裁剪。

工具分组（可按 include 选择）:
  basic     Calculator / Time / Search / Weekday
  terminal  TerminalTool（只读终端）
  search    AdvancedSearch（多源联网搜索）
  memory    MemoryTool（四类记忆）
  note      NoteTool（长时程笔记）
  rag       RagTool（RAG 全流程）
  skill     Skill（Skills 知识外化）
  subagent  SubAgent（子代理委派，需 agent_factory）

使用方式:
    from tools.framework.registry_factory import (
        build_all_tools_registry, build_role_tool_filter)

    factory = AgentFactory()  # 供 SubAgent 委派
    tf = build_role_tool_filter()              # 角色策略（analyst/writer/readonly）
    bundle = build_all_tools_registry(agent_factory=factory, tool_filter=tf)
    registry = bundle.registry                  # 全量装配
    registry.list_tools(owner="writer")        # 按角色裁剪视图
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.framework.tool_system import ToolRegistry
from tools.framework.tool_filter import ToolFilter


# 工具分组：分组名 → 注册函数（返回 (tool, 是否需要 factory)）
def _register_basic(registry: ToolRegistry):
    from tools.framework.tool_system import CalculatorTool, TimeTool, SearchTool
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(SearchTool())
    registry.register(_weekday, name="Weekday",
                      description="获取今天是星期几（中文，如'星期一'）。"
                                  "适用：查询当前星期。"
                                  "不适用：具体日期时间（请用 Time）。",
                      parameters=[])


def _weekday() -> str:
    from datetime import datetime
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return weekdays[datetime.now().weekday()]


def _register_terminal(registry: ToolRegistry):
    from tools.terminal.terminal_tool import TerminalTool
    registry.register(TerminalTool())


def _register_search(registry: ToolRegistry):
    from tools.search.advanced_search_tool import AdvancedSearchTool
    registry.register(AdvancedSearchTool())


def _register_memory(registry: ToolRegistry):
    from tools.memory.memory_tool import MemoryTool
    registry.register(MemoryTool())


def _register_note(registry: ToolRegistry):
    from tools.memory.note_tool import NoteTool
    registry.register(NoteTool())


def _register_rag(registry: ToolRegistry):
    from tools.rag.rag_tool import RagTool
    registry.register(RagTool())


def _register_skill(registry: ToolRegistry):
    from skills.skill_tool import SkillTool
    registry.register(SkillTool())


def _register_subagent(registry: ToolRegistry, agent_factory):
    from tools.framework.sub_agent_tool import SubAgentTool
    registry.register(SubAgentTool(agent_factory))


# 分组 → 注册函数
_TOOL_GROUPS: Dict[str, Callable] = {
    "basic": _register_basic,
    "terminal": _register_terminal,
    "search": _register_search,
    "memory": _register_memory,
    "note": _register_note,
    "rag": _register_rag,
    "skill": _register_skill,
    "subagent": _register_subagent,
}


@dataclass
class ToolRegistryBundle:
    """装配结果：注册表 + 工具名 + 失败记录。"""
    registry: ToolRegistry
    names: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)


def build_all_tools_registry(
        agent_factory: Optional[Any] = None,
        tool_filter: Optional[ToolFilter] = None,
        owner: Optional[str] = None,
        enable_circuit_breaker: bool = True,
        include: Optional[List[str]] = None) -> ToolRegistryBundle:
    """一键装配全部（或指定分组）工具。

    Args:
        agent_factory: AgentFactory（装配 subagent 分组时需要）
        tool_filter: ToolFilter 实例（按 owner 角色裁剪）
        owner: 注册表默认所属角色
        enable_circuit_breaker: 是否启用熔断器
        include: 分组白名单，None = 全部（subagent 无 factory 时自动跳过）
    """
    groups = include or list(_TOOL_GROUPS.keys())
    registry = ToolRegistry(enable_circuit_breaker=enable_circuit_breaker,
                            tool_filter=tool_filter, owner=owner)
    bundle = ToolRegistryBundle(registry=registry)

    for group in groups:
        register_fn = _TOOL_GROUPS.get(group)
        if register_fn is None:
            bundle.failures.append(f"未知分组: {group}")
            continue
        if group == "subagent" and agent_factory is None:
            bundle.failures.append("subagent 分组需要 agent_factory（已跳过）")
            continue
        try:
            if group == "subagent":
                register_fn(registry, agent_factory)
            else:
                register_fn(registry)
        except Exception as e:      # 单个分组失败不影响整体
            bundle.failures.append(f"{group}: {type(e).__name__}: {e}")

    bundle.names = registry.list_tools()
    return bundle


def build_role_tool_filter() -> ToolFilter:
    """默认角色策略：
      - analyst：全部工具
      - writer：写作/知识类（计算/时间/搜索/笔记/技能/记忆）
      - readonly：禁用终端与子代理
    """
    tf = ToolFilter()
    tf.set_policy("analyst", allow=[ToolFilter.WILDCARD])
    tf.set_policy("writer", allow=[
        "Calculator", "Time", "Search", "Weekday", "AdvancedSearch",
        "NoteTool", "MemoryTool", "Skill",
    ])
    tf.set_policy("readonly", deny=["TerminalTool", "SubAgent"])
    return tf


def describe_bundle(bundle: ToolRegistryBundle) -> str:
    """人类可读的工具装配清单。"""
    lines = [f"已装配 {len(bundle.names)} 个工具: {', '.join(bundle.names)}"]
    if bundle.failures:
        lines.append(f"⚠️ 失败 {len(bundle.failures)} 项:")
        lines += [f"  - {f}" for f in bundle.failures]
    return "\n".join(lines)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    from agents.framework.agent_framework import AgentFactory, BaseAgent

    print("=" * 60)
    print("🧰 build_all_tools_registry 一键装配演示")
    print("=" * 60)

    # 演示子代理工厂
    factory = AgentFactory()

    class EchoSub(BaseAgent):
        AGENT_TYPE = "echo_sub"
        DESCRIPTION = "回声子代理"

        def _execute(self, input_text: str, **kwargs) -> str:
            return f"子代理处理: {input_text}"

    factory.register("echo_sub", EchoSub, "回声子代理")

    # 1. 全量装配
    print("\n--- 1) 全量装配（默认分组）---")
    tf = build_role_tool_filter()
    bundle = build_all_tools_registry(agent_factory=factory, tool_filter=tf)
    print(describe_bundle(bundle))
    if bundle.failures:
        print("   说明: 失败的通常是需要外部密钥的分组，不影响其他工具。")

    # 2. 角色裁剪视图
    print("\n--- 2) 按角色裁剪 ---")
    registry = bundle.registry
    for role in ("analyst", "writer", "readonly", "no_policy"):
        visible = registry.list_tools(owner=role)
        print(f"   {role:<10} → {len(visible)} 个: {visible[:6]}{'…' if len(visible) > 6 else ''}")

    # 3. 实际执行（无 LLM 演示）
    print("\n--- 3) 实际执行 ---")
    r = registry.execute_structured("Calculator", {"expression": "(15 + 27) * 3"})
    print(f"   Calculator → [{r.status}] {r.output}")
    r2 = registry.execute_structured("Weekday", {})
    print(f"   Weekday → [{r2.status}] {r2.output}")
    r3 = registry.execute_structured("SubAgent",
                                     {"task": "整理要点", "agent_type": "echo_sub"})
    print(f"   SubAgent → [{r3.status}] {r3.output}")
    r4 = registry.execute_structured("Skill", {"action": "list"})
    print(f"   Skill.list → [{r4.status}] 前 40 字: {r4.output[:40].replace(chr(10), ' ')}")

    print("\n✅ build_all_tools_registry 演示完成")

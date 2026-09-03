"""ToolFilter：工具过滤（按 Agent 角色裁剪可见工具）。

设计：
  - ToolPolicy：单个 Agent 角色的工具策略（allow 白名单 / deny 黑名单）
  - ToolFilter：策略集合，按 agent_type 查询是否允许某个工具
  - 语义：
      - 无策略 → 全部允许（默认）
      - allow=None → 除 deny 外全部允许
      - allow=[...] → 仅允许列表内（再减去 deny）
      - allow 支持通配符 "*"（允许全部）
      - deny 优先于 allow
  - 用途：子代理机制中，不同角色的 Agent 只看到自己该用的工具，
    防止越权调用（如只读 Agent 不能调用写工具）

使用方式:
    tf = ToolFilter()
    tf.allow("writer", "NoteTool", "TerminalTool")
    tf.deny("readonly", "TerminalTool", "SubAgent")
    visible = tf.get_visible("writer", registry.list_tools())
    ok = tf.is_allowed("readonly", "Calculator")
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ToolPolicy:
    """一个 Agent 角色的工具策略。"""
    allow: Optional[List[str]] = None        # None=全部允许；列表=白名单
    deny: List[str] = field(default_factory=list)  # 黑名单（优先于 allow）


class ToolFilter:
    """工具过滤：按 agent_type 管理工具可见性。"""

    WILDCARD = "*"

    def __init__(self):
        self._policies: Dict[str, ToolPolicy] = {}

    # ------------------------------------------------------------
    # 策略配置
    # ------------------------------------------------------------

    def set_policy(self, agent_type: str,
                   allow: Optional[List[str]] = None,
                   deny: Optional[List[str]] = None) -> None:
        """设置角色完整策略（覆盖旧策略）。"""
        self._policies[agent_type] = ToolPolicy(
            allow=list(allow) if allow else None,
            deny=list(deny) if deny else [])

    def allow(self, agent_type: str, *tools: str,
              replace: bool = False) -> None:
        """追加白名单（replace=True 时替换）。"""
        policy = self._policies.setdefault(agent_type, ToolPolicy())
        if replace or policy.allow is None:
            policy.allow = list(tools)
        else:
            policy.allow.extend(tools)

    def deny(self, agent_type: str, *tools: str) -> None:
        """追加黑名单。"""
        policy = self._policies.setdefault(agent_type, ToolPolicy())
        policy.deny.extend(tools)

    def allow_all(self, agent_type: str) -> None:
        """允许全部工具（显式白名单 '*'）。"""
        policy = self._policies.setdefault(agent_type, ToolPolicy())
        policy.allow = [self.WILDCARD]

    def remove_policy(self, agent_type: str) -> bool:
        return self._policies.pop(agent_type, None) is not None

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------

    def get_policy(self, agent_type: str) -> Optional[ToolPolicy]:
        return self._policies.get(agent_type)

    def is_allowed(self, agent_type: str, tool_name: str) -> bool:
        """判断指定角色是否允许使用某工具。"""
        policy = self._policies.get(agent_type)
        if policy is None:
            return True                       # 无策略 → 默认允许
        if tool_name in policy.deny:
            return False                      # 黑名单优先
        if policy.allow is None:
            return True                       # 无白名单 → 仅受黑名单限制
        return (self.WILDCARD in policy.allow
                or tool_name in policy.allow)

    def get_visible(self, agent_type: str,
                    tool_names: List[str]) -> List[str]:
        """返回该角色可见的工具名列表。"""
        return [t for t in tool_names
                if self.is_allowed(agent_type, t)]

    def describe(self, agent_type: str,
                 tool_names: List[str]) -> List[Dict[str, str]]:
        """带状态的可见工具描述（便于调试/展示）。"""
        result = []
        for t in tool_names:
            allowed = self.is_allowed(agent_type, t)
            result.append({"tool": t, "allowed": allowed,
                           "reason": "ok" if allowed else
                           self._deny_reason(agent_type, t)})
        return result

    def _deny_reason(self, agent_type: str, tool_name: str) -> str:
        policy = self._policies.get(agent_type)
        if policy is None:
            return "ok"
        if tool_name in policy.deny:
            return f"deny 黑名单"
        if policy.allow is not None and self.WILDCARD not in policy.allow \
                and tool_name not in policy.allow:
            return "不在 allow 白名单"
        return "ok"

    def stats(self) -> Dict[str, Any]:
        return {"policies": {k: {"allow": v.allow, "deny": v.deny}
                             for k, v in self._policies.items()}}


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 ToolFilter 工具过滤演示")
    print("=" * 60)

    all_tools = ["Calculator", "Time", "Search", "NoteTool",
                 "TerminalTool", "RAGTool", "SubAgent"]

    tf = ToolFilter()
    # writer 角色：只允许笔记/搜索/计算
    tf.set_policy("writer", allow=["NoteTool", "Search", "Calculator"])
    # readonly 角色：默认全允许，但禁用 Terminal/SubAgent
    tf.set_policy("readonly", deny=["TerminalTool", "SubAgent"])
    # analyst 角色：允许全部
    tf.allow_all("analyst")

    print("\n--- 1) 各角色可见工具 ---")
    for role in ("writer", "readonly", "analyst", "no_policy"):
        visible = tf.get_visible(role, all_tools)
        print(f"   {role:<10} → {visible}")

    print("\n--- 2) 明细（describe）---")
    for d in tf.describe("writer", all_tools):
        if not d["allowed"]:
            print(f"   ✗ {d['tool']:<12} {d['reason']}")

    print("\n--- 3) 关键判定 ---")
    cases = [("writer", "Calculator"), ("writer", "TerminalTool"),
             ("readonly", "Calculator"), ("readonly", "SubAgent"),
             ("analyst", "SubAgent")]
    for role, tool in cases:
        print(f"   {role}.{tool} → {'✅ 允许' if tf.is_allowed(role, tool) else '❌ 拒绝'}")

    print(f"\n   stats: {tf.stats()}")
    print("\n✅ ToolFilter 演示完成")

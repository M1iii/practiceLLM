"""ToolFilter：按 Agent 角色裁剪可见工具。

语义：无策略→全部允许，allow=None→除 deny 外全部允许，deny 优先于 allow。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ToolPolicy:
    allow: Optional[List[str]] = None
    deny: List[str] = field(default_factory=list)


class ToolFilter:
    """工具过滤：按 agent_type 管理工具可见性。"""

    WILDCARD = "*"

    def __init__(self):
        self._policies: Dict[str, ToolPolicy] = {}

    def set_policy(self, agent_type: str,
                   allow: Optional[List[str]] = None,
                   deny: Optional[List[str]] = None) -> None:
        self._policies[agent_type] = ToolPolicy(
            allow=list(allow) if allow else None,
            deny=list(deny) if deny else [])

    def allow(self, agent_type: str, *tools: str,
              replace: bool = False) -> None:
        policy = self._policies.setdefault(agent_type, ToolPolicy())
        if replace or policy.allow is None:
            policy.allow = list(tools)
        else:
            policy.allow.extend(tools)

    def deny(self, agent_type: str, *tools: str) -> None:
        policy = self._policies.setdefault(agent_type, ToolPolicy())
        policy.deny.extend(tools)

    def allow_all(self, agent_type: str) -> None:
        policy = self._policies.setdefault(agent_type, ToolPolicy())
        policy.allow = [self.WILDCARD]

    def remove_policy(self, agent_type: str) -> bool:
        return self._policies.pop(agent_type, None) is not None

    def get_policy(self, agent_type: str) -> Optional[ToolPolicy]:
        return self._policies.get(agent_type)

    def is_allowed(self, agent_type: str, tool_name: str) -> bool:
        policy = self._policies.get(agent_type)
        if policy is None:
            return True
        if tool_name in policy.deny:
            return False
        if policy.allow is None:
            return True
        return (self.WILDCARD in policy.allow
                or tool_name in policy.allow)

    def get_visible(self, agent_type: str,
                    tool_names: List[str]) -> List[str]:
        return [t for t in tool_names
                if self.is_allowed(agent_type, t)]

    def describe(self, agent_type: str,
                 tool_names: List[str]) -> List[Dict[str, str]]:
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
            return "黑名单拒绝"
        if policy.allow is not None and self.WILDCARD not in policy.allow \
                and tool_name not in policy.allow:
            return "不在白名单"
        return "ok"

    def stats(self) -> Dict[str, Any]:
        return {"policies": {k: {"allow": v.allow, "deny": v.deny}
                             for k, v in self._policies.items()}}


if __name__ == "__main__":
    all_tools = ["Calculator", "Time", "Search", "NoteTool",
                 "TerminalTool", "RAGTool", "SubAgent"]

    tf = ToolFilter()
    tf.set_policy("writer", allow=["NoteTool", "Search", "Calculator"])
    tf.set_policy("readonly", deny=["TerminalTool", "SubAgent"])
    tf.allow_all("analyst")

    for role in ("writer", "readonly", "analyst", "no_policy"):
        visible = tf.get_visible(role, all_tools)
        print(f"  {role:<10} -> {visible}")

    cases = [("writer", "Calculator"), ("writer", "TerminalTool"),
             ("readonly", "Calculator"), ("readonly", "SubAgent"),
             ("analyst", "SubAgent")]
    for role, tool in cases:
        ok = tf.is_allowed(role, tool)
        print(f"  {role}.{tool} -> {'允许' if ok else '拒绝'}")
    print(f"  stats: {tf.stats()}")

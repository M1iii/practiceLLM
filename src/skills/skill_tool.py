"""SkillTool：Agent 运行时调用 Skills 的工具。

让 Agent 在对话过程中按需加载/查看技能，实现"知识外化、运行时取用"。

动作（参数 action）:
  list                  → 列出可用技能
  use   <skill_name>    → 返回技能正文（注入当前上下文）
  context [names]       → 拼接多个技能上下文块（默认全部已激活）
  activate <name>       → 启用技能
  deactivate <name>     → 停用技能
"""

import sys
import os
import time
from typing import List, Optional


from src.tools.framework.tool_system import Tool, ToolParameter
from src.tools.framework.tool_response import ToolResponse
from src.skills.loader import SkillLoader


class SkillTool(Tool):
    """技能调用工具。"""

    def __init__(self, loader: Optional[SkillLoader] = None,
                 name: str = "Skill"):
        self._loader = loader or SkillLoader()
        self._tool_name = name

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str:
        return ("技能管理工具（知识外化）：list 列出技能，"
                "use <名称> 获取技能正文，context 拼接上下文，"
                "activate/deactivate 激活或停用。"
                "适用：加载领域专业知识（如代码审查规范、实践指南）、"
                "将技能内容注入 Agent 上下文。"
                "不适用：实时网络搜索（请用 AdvancedSearch）、"
                "记忆管理（请用 MemoryTool）。")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="action", type="string", required=True,
                          description="list / use / context / activate / deactivate"),
            ToolParameter(name="skill_name", type="string", required=False,
                          default="", description="技能名（use/activate 时必填）"),
            ToolParameter(name="names", type="string", required=False,
                          default="", description="技能名列表（逗号分隔，context 用）"),
        ]

    def run(self, args: dict) -> str:
        action = str(args.get("action", "")).strip().lower()
        skill_name = str(args.get("skill_name", "")).strip()
        names = [n.strip() for n in str(args.get("names", "")).split(",")
                 if n.strip()]

        if action == "list":
            infos = self._loader.list_skills()
            if not infos:
                return "（无可用技能）"
            lines = [f"- {i['name']} (v{i['version']}): {i['description']}"
                     + ("" if i["activated"] else " [已停用]")
                     for i in infos]
            return "可用技能:\n" + "\n".join(lines)

        if action == "use":
            skill = self._loader.load(skill_name)
            if skill is None:
                available = ", ".join(i["name"]
                                      for i in self._loader.list_skills()) or "无"
                return f"错误：未找到技能 '{skill_name}'，可用: {available}"
            if not skill.activated:
                return f"错误：技能 '{skill_name}' 已停用"
            return (f"# 技能: {skill.name}（v{skill.version}）\n"
                    f"{skill.description}\n\n{skill.content}")

        if action == "context":
            context = self._loader.build_context(names or None)
            if not context:
                return "（无可用技能上下文）"
            return context

        if action == "activate":
            ok = self._loader.activate(skill_name)
            return f"✅ 已启用技能: {skill_name}" if ok \
                else f"错误：未找到技能 '{skill_name}'"

        if action == "deactivate":
            ok = self._loader.deactivate(skill_name)
            return f"⏸️ 已停用技能: {skill_name}" if ok \
                else f"错误：未找到技能 '{skill_name}'"

        return f"错误：未知动作 '{action}'（支持 list/use/context/activate/deactivate）"

    def execute(self, args: dict) -> ToolResponse:
        start = time.perf_counter()
        output = self.run(args)
        failed = output.startswith("错误")
        resp = (ToolResponse.error(output=output, error="skill_error")
                if failed else ToolResponse.success(output=output))
        resp.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        return resp


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    from src.tools.framework.tool_system import ToolRegistry

    print("=" * 60)
    print("🔗 SkillTool 技能调用演示")
    print("=" * 60)

    registry = ToolRegistry()
    tool = SkillTool()
    registry.register(tool)

    # 1. list
    print("\n--- 1) list 技能 ---")
    resp = registry.execute_structured("Skill", {"action": "list"})
    print(f"   [{resp.status}]\n{resp.output}")

    # 2. use
    print("--- 2) use code_review ---")
    resp = registry.execute_structured("Skill",
                                       {"action": "use", "skill_name": "code_review"})
    print(f"   [{resp.status}] 返回 {len(resp.output)} 字符，"
          f"开头: {resp.output[:40].replace(chr(10), ' ')}…")

    # 3. context（多技能拼接）
    print("--- 3) context 拼接 ---")
    resp = registry.execute_structured("Skill",
                                       {"action": "context",
                                        "names": "code_review,rag_practice"})
    print(f"   [{resp.status}] 上下文 {len(resp.output)} 字符，"
          f"包含 2 个技能标题: "
          f"{resp.output.count('## 技能:') == 2}")

    # 4. 错误处理
    print("--- 4) 错误处理 ---")
    resp = registry.execute_structured("Skill",
                                       {"action": "use", "skill_name": "不存在"})
    print(f"   [{resp.status}] {resp.output[:60]}")
    resp = registry.execute_structured("Skill", {"action": "未知动作"})
    print(f"   [{resp.status}] {resp.output[:50]}")

    print("\n✅ SkillTool 演示完成")
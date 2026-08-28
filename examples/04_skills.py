"""示例 04：Skills 知识外化。

演示 SkillLoader 发现/加载/注入技能，以及 SkillTool 运行时调用。
运行: python examples/04_skills.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.loader import SkillLoader
from tools.framework.tool_system import ToolRegistry


def main():
    # 1. 发现技能
    loader = SkillLoader()
    print("可用技能:")
    for info in loader.list_skills():
        print(f"  - {info['name']} (v{info['version']}): {info['description']}")

    # 2. 注入系统提示词
    prompt = loader.inject("你是一位专业助手。", ["code_review"])
    print(f"\n注入后提示词长度: {len(prompt)}（原 9 字符）")

    # 3. 运行时调用（SkillTool）
    registry = ToolRegistry()
    registry.register(loader_tool())
    resp = registry.execute_structured("Skill", {"action": "use",
                                                 "skill_name": "rag_practice"})
    print(f"\nSkillTool.use → [{resp.status}] {len(resp.output)} 字符")
    resp2 = registry.execute_structured("Skill", {"action": "list"})
    print(f"SkillTool.list → [{resp2.status}]\n{resp2.output}")


def loader_tool():
    from skills.skill_tool import SkillTool
    return SkillTool()


if __name__ == "__main__":
    main()

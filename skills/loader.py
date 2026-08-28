"""SkillLoader：Skills 知识外化系统。

将可复用的知识/方法论/指令封装为独立技能包（SKILL.md），
Agent 可按需加载注入上下文，实现"知识外化、按需取用"。

技能包格式（skills/builtin/<name>/SKILL.md）:
    ---
    name: code_review            # 技能名（目录名）
    description: 代码审查方法论   # 一句话描述
    version: 1.0.0               # 版本
    tags: [code, review]         # 标签（可选）
    ---
    # 技能正文（Markdown 指令，注入 Agent 上下文）

功能：
  - discover()：扫描技能目录
  - load(name)/get_skill(name)：加载技能（解析 front-matter）
  - list_skills()/activate()/deactivate()：管理技能
  - build_context(names)：拼接技能正文为可注入提示词
  - inject 示例：Agent 系统提示词中追加 build_context 结果

使用方式:
    loader = SkillLoader()                  # 默认扫描 skills/builtin
    loader.list_skills()                    # 发现技能
    context = loader.build_context(["code_review"])
    prompt = f"{system_prompt}\n\n{context}"   # 注入 Agent
"""

import sys
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKILL_FILE = "SKILL.md"


@dataclass
class Skill:
    """一个技能包。"""
    name: str
    description: str
    version: str
    path: str                              # SKILL.md 绝对路径
    content: str                           # 正文（Markdown 指令）
    metadata: Dict[str, Any] = field(default_factory=dict)   # front-matter 其他字段
    tags: List[str] = field(default_factory=list)
    activated: bool = True                 # 是否启用（默认启用）

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "version": self.version, "tags": self.tags,
                "activated": self.activated, "content_length": len(self.content)}


class SkillLoader:
    """技能加载器：发现 / 加载 / 管理技能包。"""

    def __init__(self, skills_dir: Optional[str] = None):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.skills_dir = skills_dir or os.path.join(root, "skills", "builtin")
        self._skills: Dict[str, Skill] = {}

    # ------------------------------------------------------------
    # 发现与加载
    # ------------------------------------------------------------

    def discover(self) -> List[Skill]:
        """扫描技能目录，加载全部技能（幂等刷新）。"""
        if not os.path.isdir(self.skills_dir):
            return []
        found = {}
        for entry in sorted(os.listdir(self.skills_dir)):
            skill_path = os.path.join(self.skills_dir, entry, SKILL_FILE)
            if os.path.isfile(skill_path):
                try:
                    skill = self._parse_skill(entry, skill_path)
                    if skill is not None:
                        found[skill.name] = skill
                except Exception:
                    continue
        self._skills = found
        return list(found.values())

    def _parse_skill(self, dir_name: str, path: str) -> Optional[Skill]:
        """解析 SKILL.md：front-matter（--- 包裹的键值对）+ 正文。"""
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        metadata, content = self._parse_front_matter(raw)
        name = metadata.pop("name", None) or dir_name
        return Skill(
            name=name,
            description=metadata.pop("description", "") or "",
            version=str(metadata.pop("version", "1.0.0")),
            path=path,
            content=content,
            metadata=metadata,
            tags=metadata.pop("tags", []) or [],
        )

    @staticmethod
    def _parse_front_matter(raw: str):
        """解析 --- 包裹的 front-matter（简单键值对 + 行内列表）。"""
        if not raw.startswith("---"):
            return {}, raw
        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw
        header, body = parts[1], parts[2]
        metadata: Dict[str, Any] = {}
        for line in header.strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if value.startswith("[") and value.endswith("]"):
                value = [v.strip().strip("'\"")
                         for v in value[1:-1].split(",") if v.strip()]
            elif value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            metadata[key] = value
        return metadata, body.strip()

    # ------------------------------------------------------------
    # 查询与管理
    # ------------------------------------------------------------

    def ensure_loaded(self) -> None:
        if not self._skills:
            self.discover()

    def load(self, name: str) -> Optional[Skill]:
        """按名称加载技能（未发现则先扫描）。"""
        self.ensure_loaded()
        return self._skills.get(name)

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.load(name)

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出全部技能（元信息）。"""
        self.ensure_loaded()
        return [s.to_dict() for s in self._skills.values()]

    def activate(self, name: str) -> bool:
        skill = self.load(name)
        if skill is None:
            return False
        skill.activated = True
        return True

    def deactivate(self, name: str) -> bool:
        skill = self.load(name)
        if skill is None:
            return False
        skill.activated = False
        return True

    # ------------------------------------------------------------
    # 上下文注入
    # ------------------------------------------------------------

    def build_context(self, names: Optional[List[str]] = None) -> str:
        """将指定技能（默认全部已激活）拼接为可注入提示词文本。"""
        self.ensure_loaded()
        if names is None:
            selected = [s for s in self._skills.values() if s.activated]
        else:
            selected = []
            for n in names:
                s = self._skills.get(n)
                if s is not None and s.activated:
                    selected.append(s)
        blocks = []
        for s in selected:
            blocks.append(
                f"## 技能: {s.name}（v{s.version}）\n"
                f"{s.description}\n\n{s.content}")
        if not blocks:
            return ""
        return "\n\n".join(blocks)

    def inject(self, system_prompt: str,
               names: Optional[List[str]] = None) -> str:
        """将技能上下文注入系统提示词末尾（无技能时原样返回）。"""
        context = self.build_context(names)
        if not context:
            return system_prompt
        return f"{system_prompt}\n\n# 可用技能\n\n{context}"

    def stats(self) -> Dict[str, Any]:
        self.ensure_loaded()
        return {"skills_dir": self.skills_dir,
                "skill_count": len(self._skills),
                "active": sum(1 for s in self._skills.values()
                              if s.activated)}


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 SkillLoader Skills 知识外化演示")
    print("=" * 60)

    loader = SkillLoader()
    print(f"技能目录: {loader.skills_dir}")

    # 1. 发现技能
    print("\n--- 1) 发现技能 ---")
    for info in loader.list_skills():
        print(f"   {info['name']:<16} v{info['version']:<6} "
              f"{info['description']}（正文 {info['content_length']} 字）")

    # 2. 加载单个技能（front-matter 解析）
    print("\n--- 2) 加载技能详情 ---")
    skill = loader.load("code_review")
    if skill:
        print(f"   名称: {skill.name} | 版本: {skill.version}")
        print(f"   标签: {skill.tags}")
        print(f"   正文前 60 字: {skill.content[:60].replace(chr(10), ' ')}…")

    # 3. 构建注入上下文
    print("\n--- 3) build_context（拼接注入文本）---")
    context = loader.build_context(["code_review", "rag_practice"])
    print(f"   上下文块长度: {len(context)} 字符")
    print(f"   开头: {context[:60].replace(chr(10), ' ')}…")

    # 4. 注入系统提示词
    print("\n--- 4) inject 系统提示词 ---")
    prompt = loader.inject("你是一个专业助手。", ["code_review"])
    print(f"   注入后提示词长度: {len(prompt)}（原 {len('你是一个专业助手。')}）")
    print(f"   包含技能标题: {'技能: code_review' in prompt}")

    # 5. 启停管理
    print("\n--- 5) activate/deactivate ---")
    loader.deactivate("rag_practice")
    print(f"   停用 rag_practice 后活跃技能: "
          f"{sum(1 for s in loader._skills.values() if s.activated)}")
    ctx_all = loader.build_context()
    print(f"   build_context(全部) 长度: {len(ctx_all)}"
          f"{'（已排除停用技能）' if 'rag_practice' not in ctx_all else ''}")

    print(f"\n   stats: {loader.stats()}")
    print("\n✅ SkillLoader 演示完成")

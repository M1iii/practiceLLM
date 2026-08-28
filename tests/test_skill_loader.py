"""SkillLoader 测试（无需 LLM）。"""
from skills.loader import SkillLoader


def test_discover_builtin_skills():
    loader = SkillLoader()
    infos = loader.list_skills()
    names = [i["name"] for i in infos]
    assert "code_review" in names
    assert "rag_practice" in names


def test_load_parses_front_matter():
    loader = SkillLoader()
    skill = loader.load("code_review")
    assert skill is not None
    assert skill.version == "1.0.0"
    assert "代码审查" in skill.description
    assert "正确性" in skill.content       # 正文解析
    assert any("review" in t for t in skill.tags)


def test_build_context_joins_skills():
    loader = SkillLoader()
    ctx = loader.build_context(["code_review", "rag_practice"])
    assert "## 技能: code_review" in ctx
    assert "## 技能: rag_practice" in ctx


def test_inject_appends_to_prompt():
    loader = SkillLoader()
    prompt = loader.inject("你是助手。", ["code_review"])
    assert prompt.startswith("你是助手。")
    assert "技能: code_review" in prompt
    # 无可用技能时原样返回
    empty = SkillLoader()
    for info in empty.list_skills():
        empty.deactivate(info["name"])
    assert empty.inject("原始") == "原始"


def test_activate_deactivate():
    loader = SkillLoader()
    loader.activate("rag_practice")
    assert loader.load("rag_practice").activated is True
    loader.deactivate("rag_practice")
    assert loader.load("rag_practice").activated is False
    # 停用后 build_context 排除
    assert "rag_practice" not in loader.build_context()
    loader.activate("rag_practice")


def test_unknown_skill():
    loader = SkillLoader()
    assert loader.load("不存在") is None
    assert loader.deactivate("不存在") is False


def test_parse_front_matter_types():
    from skills.loader import SkillLoader as SL
    meta, body = SL._parse_front_matter(
        "---\nname: x\ntags: [a, b]\ncount: 5\nenabled: true\n---\n正文")
    assert meta["tags"] == ["a", "b"]
    assert meta["count"] == 5
    assert meta["enabled"] is True
    assert body == "正文"

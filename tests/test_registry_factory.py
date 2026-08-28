"""registry_factory 一键装配工厂测试（无网络、无外部密钥）。"""
from agents.framework.agent_framework import AgentFactory, BaseAgent
from tools.framework.registry_factory import (
    build_all_tools_registry, build_role_tool_filter, describe_bundle)


class EchoSub(BaseAgent):
    AGENT_TYPE = "echo_sub"
    DESCRIPTION = "回声子代理"

    def _execute(self, input_text: str, **kwargs) -> str:
        return f"子代理处理: {input_text}"


def _factory():
    factory = AgentFactory()
    factory.register("echo_sub", EchoSub, "回声子代理")
    return factory


def test_full_build_no_failures():
    """默认分组全量装配，无失败。"""
    bundle = build_all_tools_registry(agent_factory=_factory())
    assert bundle.failures == []
    expected = {"Calculator", "Time", "Search", "Weekday", "TerminalTool",
                "AdvancedSearch", "MemoryTool", "NoteTool", "RagTool",
                "Skill", "SubAgent"}
    assert set(bundle.names) == expected
    assert len(bundle.names) == 11


def test_build_subset():
    bundle = build_all_tools_registry(agent_factory=_factory(),
                                      include=["basic", "skill", "subagent"])
    assert set(bundle.names) == {"Calculator", "Time", "Search", "Weekday",
                                 "Skill", "SubAgent"}


def test_subagent_without_factory_skipped():
    bundle = build_all_tools_registry(agent_factory=None)
    assert "SubAgent" not in bundle.names
    assert any("subagent" in f for f in bundle.failures)
    # 其他分组不受影响
    assert "Calculator" in bundle.names


def test_role_filter_views():
    tf = build_role_tool_filter()
    bundle = build_all_tools_registry(agent_factory=_factory(), tool_filter=tf)
    registry = bundle.registry
    assert len(registry.list_tools(owner="analyst")) == 11
    # writer 无终端/子代理
    writer = registry.list_tools(owner="writer")
    assert "TerminalTool" not in writer and "SubAgent" not in writer
    assert "NoteTool" in writer and "Skill" in writer
    # readonly 禁用终端/子代理
    readonly = registry.list_tools(owner="readonly")
    assert "TerminalTool" not in readonly and "SubAgent" not in readonly
    assert "Calculator" in readonly


def test_execute_after_build():
    bundle = build_all_tools_registry(agent_factory=_factory())
    registry = bundle.registry
    r = registry.execute_structured("Calculator", {"expression": "2 ** 10"})
    assert r.status == "success" and r.output == "1024"
    r2 = registry.execute_structured("Weekday", {})
    assert r2.status == "success"


def test_describe_bundle():
    bundle = build_all_tools_registry(agent_factory=None,
                                      include=["basic"])
    text = describe_bundle(bundle)
    assert "已装配 4 个工具" in text

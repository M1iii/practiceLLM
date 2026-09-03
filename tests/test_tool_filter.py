"""工具过滤测试。"""
from src.tools.framework.tool_filter import ToolFilter
from src.tools.framework.tool_system import ToolRegistry, FunctionTool, ToolParameter


def test_default_allow_all():
    tf = ToolFilter()
    assert tf.is_allowed("any_role", "Anything") is True


def test_allow_whitelist():
    tf = ToolFilter()
    tf.set_policy("writer", allow=["Note", "Search"])
    assert tf.is_allowed("writer", "Note") is True
    assert tf.is_allowed("writer", "Search") is True
    assert tf.is_allowed("writer", "Terminal") is False


def test_deny_blacklist():
    tf = ToolFilter()
    tf.set_policy("readonly", deny=["Terminal", "SubAgent"])
    assert tf.is_allowed("readonly", "Calculator") is True   # 默认允许
    assert tf.is_allowed("readonly", "Terminal") is False
    assert tf.is_allowed("readonly", "SubAgent") is False


def test_deny_overrides_allow():
    tf = ToolFilter()
    tf.set_policy("x", allow=["A", "B"], deny=["A"])
    assert tf.is_allowed("x", "A") is False
    assert tf.is_allowed("x", "B") is True


def test_wildcard_allow_all():
    tf = ToolFilter()
    tf.allow_all("admin")
    assert tf.is_allowed("admin", "anything") is True


def test_get_visible():
    tf = ToolFilter()
    tf.set_policy("writer", allow=["A", "C"])
    tools = ["A", "B", "C", "D"]
    assert tf.get_visible("writer", tools) == ["A", "C"]


def test_append_allow_and_deny():
    tf = ToolFilter()
    tf.allow("w", "A")
    tf.allow("w", "B")            # 追加
    assert tf.get_visible("w", ["A", "B", "C"]) == ["A", "B"]
    tf.deny("w", "A")             # 追加黑名单
    assert tf.get_visible("w", ["A", "B"]) == ["B"]


def test_registry_integration():
    def add(a, b):
        return a + b

    registry = ToolRegistry(tool_filter=ToolFilter(), owner="writer",
                            enable_circuit_breaker=False)
    registry.register(add, name="Add",
                      parameters=[ToolParameter(name="a", type="number",
                                                description="a"),
                                  ToolParameter(name="b", type="number",
                                                description="b")])
    registry.register(add, name="Hidden")

    registry.tool_filter.set_policy("writer", allow=["Add"])
    assert registry.list_tools() == ["Add"]                 # 过滤视图
    assert registry.execute_structured("Add", {"a": 1, "b": 2}).status == "success"
    r = registry.execute_structured("Hidden", {"a": 1, "b": 2})
    assert r.status == "error"
    assert "agent_forbidden" in r.error

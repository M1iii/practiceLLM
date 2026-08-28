"""防重试节流机制测试。

测试项：
1. AdvancedSearchTool 连续失败计数 + 终端错误信号
2. AdvancedSearchTool 成功重置计数器
3. AdvancedSearchTool reset_interval 自动恢复
4. FunctionCallAgent 工具禁用（连续失败超限）
5. FunctionCallAgent 终端错误立即禁用
6. FunctionCallAgent 禁用工具从 schema 中移除
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.search.advanced_search_tool import AdvancedSearchTool
from agents.function_call_agent import FunctionCallAgent


def test_search_tool_throttle():
    """测试1: AdvancedSearchTool 连续失败 → 终端错误信号。"""
    tool = AdvancedSearchTool(max_consecutive_failures=2, reset_interval=60)

    # 模拟连续失败（不依赖 API Key 是否配置）
    tool._record_failure()
    assert tool._consecutive_failures == 1, f"失败计数应为1, 实际 {tool._consecutive_failures}"

    tool._record_failure()
    assert tool._consecutive_failures == 2, f"失败计数应为2, 实际 {tool._consecutive_failures}"

    # 第3次通过 run() 调用应触发节流
    r3 = tool.run({"query": "test"})
    assert "【搜索不可用】" in r3, f"节流应返回终端错误: {r3}"
    assert "请勿重试" in r3, f"应提示请勿重试: {r3}"

    print("✅ test_search_tool_throttle: 连续失败→终端错误信号 OK")


def test_search_tool_success_reset():
    """测试2: 成功调用重置计数器。"""
    tool = AdvancedSearchTool(max_consecutive_failures=2, reset_interval=60)

    # 先失败
    tool._record_failure()
    assert tool._consecutive_failures == 1

    # 成功后重置
    tool._record_success()
    assert tool._consecutive_failures == 0, f"成功应重置计数器, 实际 {tool._consecutive_failures}"

    print("✅ test_search_tool_success_reset: 成功重置计数器 OK")


def test_search_tool_reset_interval():
    """测试3: reset_interval 后自动恢复。"""
    tool = AdvancedSearchTool(max_consecutive_failures=2, reset_interval=0.1)

    # 触发节流
    tool._record_failure()
    tool._record_failure()

    # 节流中
    msg = tool._check_throttle()
    assert msg is not None, "节流中应返回错误"

    # 等待 reset_interval
    time.sleep(0.15)

    # 应恢复
    msg = tool._check_throttle()
    assert msg is None, f"超过 reset_interval 应恢复, 实际返回: {msg}"
    assert tool._consecutive_failures == 0, "计数器应重置为0"

    print("✅ test_search_tool_reset_interval: reset_interval 自动恢复 OK")


def test_agent_tool_disabling():
    """测试4: FunctionCallAgent 连续失败超限 → 禁用工具。"""
    agent = FunctionCallAgent(
        llm=None,  # 不需要 LLM
        name="test_agent",
        tools=[],
        max_tool_failures=2,
        tool_reset_interval=60,
    )

    # 模拟工具失败
    agent._track_tool_result("TestTool", "错误：搜索失败")
    assert agent._tool_failures.get("TestTool") == 1
    assert "TestTool" not in agent._disabled_tools

    agent._track_tool_result("TestTool", "错误：搜索失败")
    assert agent._tool_failures.get("TestTool") == 2
    assert "TestTool" in agent._disabled_tools, "连续失败2次应禁用"

    print("✅ test_agent_tool_disabling: 连续失败超限禁用 OK")


def test_agent_terminal_error_immediate():
    """测试5: 终端错误信号立即禁用工具。"""
    agent = FunctionCallAgent(
        llm=None, name="test_agent", tools=[],
        max_tool_failures=5,  # 设高阈值，验证终端错误不受阈值限制
        tool_reset_interval=60,
    )

    # 终端错误信号 → 立即禁用
    agent._track_tool_result("SearchTool", "【搜索不可用】所有搜索后端均不可用")
    assert "SearchTool" in agent._disabled_tools, "终端错误应立即禁用"

    print("✅ test_agent_terminal_error_immediate: 终端错误立即禁用 OK")


def test_agent_success_resets_failure():
    """测试6: 成功结果重置连续失败计数。"""
    agent = FunctionCallAgent(
        llm=None, name="test_agent", tools=[],
        max_tool_failures=3, tool_reset_interval=60,
    )

    agent._track_tool_result("TestTool", "错误：失败")
    assert agent._tool_failures.get("TestTool") == 1

    # 成功 → 重置
    agent._track_tool_result("TestTool", "成功返回结果")
    assert "TestTool" not in agent._tool_failures, "成功应重置失败计数"

    print("✅ test_agent_success_resets_failure: 成功重置失败计数 OK")


def test_agent_disabled_tool_excluded():
    """测试7: 禁用工具从 OpenAI schema 中排除。"""
    from unittest.mock import MagicMock

    # 创建一个模拟的 LLM
    mock_llm = MagicMock()
    mock_llm.model = "gpt-4"
    mock_llm.client = MagicMock()

    # 创建工具
    from tools.framework.tool_system import CalculatorTool
    calculator = CalculatorTool()

    agent = FunctionCallAgent(
        llm=mock_llm, name="test_agent", tools=[calculator],
        max_tool_failures=1, tool_reset_interval=60,
    )

    # 初始：Calculator 在 schema 中
    schema = agent._convert_tools_to_openai_format()
    assert len(schema) == 1, f"初始应有1个工具, 实际 {len(schema)}"
    assert schema[0]["function"]["name"] == "Calculator"

    # 模拟失败 → 禁用 Calculator
    agent._track_tool_result("Calculator", "错误：计算失败")

    # 禁用后从 schema 中排除
    schema = agent._convert_tools_to_openai_format()
    assert len(schema) == 0, f"禁用后应有0个工具, 实际 {len(schema)}"

    print("✅ test_agent_disabled_tool_excluded: 禁用工具从 schema 排除 OK")


def test_agent_tool_recovery():
    """测试8: 工具禁用后自动恢复。"""
    agent = FunctionCallAgent(
        llm=None, name="test_agent", tools=[],
        max_tool_failures=1, tool_reset_interval=0.1,
    )

    agent._track_tool_result("TestTool", "错误：失败")
    assert "TestTool" in agent._disabled_tools

    # 等待恢复
    time.sleep(0.15)

    # 调用 _recover_disabled_tools 应恢复
    agent._recover_disabled_tools()
    assert "TestTool" not in agent._disabled_tools, "恢复后应不在禁用列表"
    assert "TestTool" not in agent._tool_failures, "恢复后失败计数应清除"

    print("✅ test_agent_tool_recovery: 禁用工具自动恢复 OK")


def test_search_tool_description_updated():
    """测试9: 工具描述包含防重试指引。"""
    tool = AdvancedSearchTool()
    desc = tool.description
    assert "【搜索不可用】" in desc, "描述应包含终端错误信号指引"
    assert "请勿重试" in desc, "描述应提示 LLM 不要重试"

    print("✅ test_search_tool_description_updated: 工具描述包含防重试指引 OK")


if __name__ == "__main__":
    failures = 0
    tests = [
        test_search_tool_throttle,
        test_search_tool_success_reset,
        test_search_tool_reset_interval,
        test_agent_tool_disabling,
        test_agent_terminal_error_immediate,
        test_agent_success_resets_failure,
        test_agent_disabled_tool_excluded,
        test_agent_tool_recovery,
        test_search_tool_description_updated,
    ]

    print("=" * 60)
    print("🧪 防重试节流机制测试")
    print("=" * 60)
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}")
            failures += 1

    print(f"\n{'=' * 60}")
    if failures:
        print(f"❌ {failures}/{len(tests)} 个测试失败")
    else:
        print(f"✅ 全部 {len(tests)} 个测试通过")
"""
MemoryEngine 单元测试。

测试内容：
  1. MemoryEngine 基本构造和属性
  2. retrieve 在记忆库为空时返回空字符串
  3. store 的基本调用
  4. set_top_k / set_threshold 边界
  5. 与 BaseAgent 的集成（auto_memory 配置）
"""

import sys
from pathlib import Path

# 确保项目根在 sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from unittest.mock import patch, MagicMock


class TestMemoryEngine:
    """MemoryEngine 单元测试（模拟 MemoryTool 内部）。"""

    def test_default_construction(self):
        """默认构造，属性正确。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        assert engine.enabled is True
        assert engine._top_k == 3
        assert engine._threshold == 0.5
        assert engine._memory_type == "episodic"
        assert engine._tool is None  # 惰性初始化未触发

    def test_custom_construction(self):
        """自定义参数构造。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine(enable=False, top_k=5, similarity_threshold=0.7)
        assert engine.enabled is False
        assert engine._top_k == 5
        assert engine._threshold == 0.7

    def test_retrieve_empty_memory(self):
        """记忆库为空时 retrieve 返回空字符串。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        # 直接模拟 _tool 为 None（未初始化）
        assert engine.retrieve("测试查询") == ""

    def test_retrieve_disabled(self):
        """关闭记忆时 retrieve 返回空字符串。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine(enable=False)
        assert engine.retrieve("测试查询") == ""

    @patch("src.agents.framework.memory_engine.MemoryEngine._lazy_init")
    def test_store_disabled(self, mock_init):
        """关闭记忆时 store 返回 False。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine(enable=False)
        assert engine.store("问题", "回答") is False

    def test_set_top_k_boundary(self):
        """set_top_k 边界处理。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        engine.set_top_k(0)
        assert engine._top_k == 1  # 下限 1
        engine.set_top_k(50)
        assert engine._top_k == 20  # 上限 20
        engine.set_top_k(5)
        assert engine._top_k == 5  # 正常值

    def test_set_threshold_boundary(self):
        """set_threshold 边界处理。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        engine.set_threshold(-0.5)
        assert engine._threshold == 0.0  # 下限 0
        engine.set_threshold(1.5)
        assert engine._threshold == 1.0  # 上限 1
        engine.set_threshold(0.6)
        assert engine._threshold == 0.6  # 正常值

    @patch("src.agents.framework.memory_engine.MemoryEngine._lazy_init")
    def test_retrieve_with_tool_search(self, mock_init):
        """模拟 MemoryTool.search 返回记忆内容。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        # 模拟 _tool 已初始化
        engine._tool = MagicMock()
        engine._tool.run.return_value = (
            "📌 相关记忆（情景记忆，混合检索）:\n"
            "  - 用户问了什么是 RAG\n"
            "  - 助手回答了 RAG 的定义\n"
        )

        result = engine.retrieve("什么是 RAG？")
        assert "【相关记忆】" in result
        assert "RAG" in result
        assert "【注意事项】" in result

    @patch("src.agents.framework.memory_engine.MemoryEngine._lazy_init")
    def test_retrieve_with_empty_tool(self, mock_init):
        """MemoryTool 返回空结果时 retrieve 返回空字符串。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        engine._tool = MagicMock()
        engine._tool.run.return_value = "📭 记忆库为空，无法搜索。"
        assert engine.retrieve("测试") == ""

    @patch("src.agents.framework.memory_engine.MemoryEngine._lazy_init")
    def test_store_success(self, mock_init):
        """store 成功返回 True。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        engine._tool = MagicMock()
        engine._tool.run.return_value = "✅ 记忆已添加"
        assert engine.store("问题", "回答") is True
        assert engine._tool.run.call_count == 2  # 问题和回答各一次

    @patch("src.agents.framework.memory_engine.MemoryEngine._lazy_init")
    def test_auto_forget_triggered(self, mock_init):
        """store 达到 FORGET_EVERY 次时触发自动遗忘。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        engine._tool = MagicMock()
        engine._tool.run.return_value = "✅ 记忆已添加"
        # 模拟 store 达到遗忘阈值
        engine._store_count = engine.FORGET_EVERY - 1  # 第 9 次
        assert engine.store("问题", "回答") is True
        # 第 10 次 store 应触发 _auto_forget，调用 forget action
        forget_calls = [
            call for call in engine._tool.run.call_args_list
            if call[0][0].get("action") == "forget"
        ]
        assert len(forget_calls) == 1
        assert forget_calls[0][0][0]["strategy"] == "mixed"
        assert forget_calls[0][0][0]["threshold"] == 0.3
        assert forget_calls[0][0][0]["max_age_days"] == 7

    @patch("src.agents.framework.memory_engine.MemoryEngine._lazy_init")
    def test_auto_forget_not_triggered_too_early(self, mock_init):
        """store 未达到 FORGET_EVERY 次时不触发自动遗忘。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        engine._tool = MagicMock()
        engine._tool.run.return_value = "✅ 记忆已添加"
        # 前几次 store 不应触发遗忘
        for i in range(engine.FORGET_EVERY - 1):
            engine.store(f"问题{i}", f"回答{i}")
        forget_calls = [
            call for call in engine._tool.run.call_args_list
            if call[0][0].get("action") == "forget"
        ]
        assert len(forget_calls) == 0

    @patch("src.agents.framework.memory_engine.MemoryEngine._lazy_init")
    def test_auto_forget_tool_exception(self, mock_init):
        """遗忘时 MemoryTool 异常不影响 store 返回值。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        engine._tool = MagicMock()
        # 让 forget 动作抛出异常
        def side_effect(args):
            if args.get("action") == "forget":
                raise RuntimeError("遗忘失败")
            return "✅ 记忆已添加"
        engine._tool.run.side_effect = side_effect
        # 设置 store_count 到遗忘阈值
        engine._store_count = engine.FORGET_EVERY
        # store 仍然返回 True（异常被静默捕获）
        assert engine.store("问题", "回答") is True

    @patch("src.agents.framework.memory_engine.MemoryEngine._lazy_init")
    def test_store_tool_exception(self, mock_init):
        """MemoryTool 异常时 store 静默返回 False。"""
        from src.agents.framework.memory_engine import MemoryEngine
        engine = MemoryEngine()
        engine._tool = MagicMock()
        engine._tool.run.side_effect = RuntimeError("模拟异常")
        assert engine.store("问题", "回答") is False


class TestBaseAgentIntegration:
    """BaseAgent 自动记忆集成测试。"""

    @pytest.fixture
    def concrete_agent(self):
        """创建一个具体 Agent 子类，避免抽象类实例化失败。"""
        from src.agents.framework.agent_framework import BaseAgent

        class ConcreteAgent(BaseAgent):
            AGENT_TYPE = "test"
            def _execute(self, input_text: str, **kwargs) -> str:
                return f"echo: {input_text}"

        return ConcreteAgent

    def test_memory_engine_init_auto(self, concrete_agent):
        """auto_memory=True 时自动初始化 memory_engine。"""
        agent = concrete_agent(auto_memory=True)
        assert agent.memory_engine is not None
        assert agent._auto_memory is True

    def test_memory_engine_init_disabled(self, concrete_agent):
        """auto_memory=False 时 memory_engine 为 None。"""
        agent = concrete_agent(auto_memory=False)
        assert agent.memory_engine is None

    def test_memory_engine_init_default(self, concrete_agent):
        """默认 auto_memory=True。"""
        agent = concrete_agent()
        assert agent._auto_memory is True

    def test_auto_store_memory_disabled(self, concrete_agent):
        """auto_memory=False 时 _auto_store_memory 不执行。"""
        agent = concrete_agent(auto_memory=False)
        assert agent.memory_engine is None
        agent._auto_store_memory("问题", "回答")

    def test_auto_store_memory_failed_status(self, concrete_agent):
        """执行失败的状态不存储记忆。"""
        agent = concrete_agent(auto_memory=True)
        agent._auto_store_memory("问题", "回答", status="error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
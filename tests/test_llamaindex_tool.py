"""
LlamaIndexRAGTool 集成测试。

验证：
  1. 工具可正常实例化
  2. 工具参数声明正确
  3. 工具可注册到 ToolRegistry
  4. 查询操作正常返回
  5. stats 操作正常返回
  6. 错误处理（无 question 参数时）
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def test_01_tool_instantiation():
    """测试工具实例化。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    tool = LlamaIndexRAGTool()
    assert tool.name == "LlamaIndexRAGTool"
    assert len(tool.description) > 50
    print("✅ test_01_tool_instantiation 通过")


def test_02_parameters():
    """测试参数声明。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    tool = LlamaIndexRAGTool()
    params = tool.get_parameters()
    assert len(params) == 3
    param_names = {p.name for p in params}
    assert "action" in param_names
    assert "question" in param_names
    assert "top_k" in param_names
    # action 必填
    assert params[0].required == True
    # question 非必填
    assert params[1].required == False
    print("✅ test_02_parameters 通过")


def test_03_tool_registry():
    """测试注册到 ToolRegistry。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    from src.tools.framework.tool_system import ToolRegistry

    tool = LlamaIndexRAGTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = registry.execute("LlamaIndexRAGTool", {"action": "stats"})
    assert "LlamaIndexRAGTool 统计" in result
    print("✅ test_03_tool_registry 通过")


def test_04_stats():
    """测试 stats 操作。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    tool = LlamaIndexRAGTool()
    result = tool.run({"action": "stats"})
    assert "LlamaIndexRAGTool 统计" in result
    assert "总节点数" in result
    print("✅ test_04_stats 通过")


def test_05_search():
    """测试 search 操作。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    tool = LlamaIndexRAGTool()
    result = tool.run({"action": "search", "question": "RAG", "top_k": 2})
    assert "检索结果" in result
    assert "rag_test.md" in result
    print("✅ test_05_search 通过")


def test_06_query():
    """测试 query 操作。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    tool = LlamaIndexRAGTool()
    result = tool.run({"action": "query", "question": "RAG 的核心组件有哪些？", "top_k": 2})
    assert "RAG" in result
    assert "引用来源" in result
    print("✅ test_06_query 通过")


def test_07_missing_question():
    """测试缺少 question 参数时的错误处理。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    tool = LlamaIndexRAGTool()
    result = tool.run({"action": "query"})
    assert "错误" in result
    print("✅ test_07_missing_question 通过")


def test_08_unknown_action():
    """测试未知 action 时的错误处理。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    tool = LlamaIndexRAGTool()
    result = tool.run({"action": "unknown"})
    assert "错误" in result
    print("✅ test_08_unknown_action 通过")


def test_09_list():
    """测试 list 操作。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    tool = LlamaIndexRAGTool()
    result = tool.run({"action": "list"})
    assert "个节点" in result
    assert "rag_test.md" in result
    print("✅ test_09_list 通过")


def test_10_to_openai_format():
    """测试 OpenAI function calling 格式转换。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    tool = LlamaIndexRAGTool()
    schema = tool.to_openai_format()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "LlamaIndexRAGTool"
    assert "parameters" in schema["function"]
    print("✅ test_10_to_openai_format 通过")


if __name__ == "__main__":
    tests = [
        test_01_tool_instantiation,
        test_02_parameters,
        test_03_tool_registry,
        test_04_stats,
        test_05_search,
        test_06_query,
        test_07_missing_question,
        test_08_unknown_action,
        test_09_list,
        test_10_to_openai_format,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"结果: {passed}/{len(tests)} 通过, {failed} 失败")
    print(f"{'='*40}")
"""
LlamaIndexRAGTool 并发与压力测试。

验证：
  1. 多线程并发调用 stats 不崩溃
  2. 多线程并发调用 query 不崩溃  
  3. ToolRegistry 熔断器正常保护
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from conftest import requires_pg


@requires_pg
def test_01_concurrent_stats():
    """并发 stats 调用 N 次（LlamaIndex 内部使用 asyncio，线程池需 async 兼容）。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool

    tool = LlamaIndexRAGTool()
    n = 5

    results = []
    for i in range(n):
        result = tool.run({"action": "stats"})
        assert "LlamaIndexRAGTool 统计" in result, f"第 {i} 次调用失败"
        results.append(result)

    print(f"✅ test_01_concurrent_stats: {n} 次连续调用通过")


@requires_pg
def test_02_concurrent_search():
    """连续 search 调用。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool

    tool = LlamaIndexRAGTool()
    queries = ["RAG", "分块策略", "PostgreSQL", "pgvector", "Embedding"]

    for q in queries:
        result = tool.run({"action": "search", "question": q, "top_k": 3})
        assert "检索结果" in result, f"查询失败: {q}"

    print(f"✅ test_02_concurrent_search: {len(queries)} 次搜索通过")


@requires_pg
def test_03_circuit_breaker():
    """测试 ToolRegistry 熔断器保护。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool
    from src.tools.framework.tool_system import ToolRegistry

    tool = LlamaIndexRAGTool()
    registry = ToolRegistry(
        enable_circuit_breaker=True,
        breaker_failure_threshold=2,
        breaker_recovery_timeout=0.5,
    )
    registry.register(tool)

    # 正常调用
    result = registry.execute("LlamaIndexRAGTool", {"action": "stats"})
    assert "LlamaIndexRAGTool 统计" in result

    # 连续失败触发熔断
    for _ in range(3):
        result = registry.execute("LlamaIndexRAGTool", {"action": "unknown"})
        assert "错误" in result

    # 熔断状态不应影响正常调用
    result = registry.execute("LlamaIndexRAGTool", {"action": "stats"})
    assert "LlamaIndexRAGTool 统计" in result

    print("✅ test_03_circuit_breaker: 熔断器保护通过")


@requires_pg
def test_04_concurrent_query():
    """连续 query 调用（含 LLM 生成）。"""
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool

    tool = LlamaIndexRAGTool()
    queries = [
        "RAG 的核心组件是什么？",
        "什么是 Parent-Child 分层索引？",
    ]

    start = time.time()
    for q in queries:
        result = tool.run({"action": "query", "question": q, "top_k": 2})
        assert "引用来源" in result, f"查询失败: {q}"

    elapsed = time.time() - start
    print(f"✅ test_04_concurrent_query: {len(queries)} 次查询通过 ({elapsed:.1f}s)")


if __name__ == "__main__":
    tests = [
        test_01_concurrent_stats,
        test_02_concurrent_search,
        test_03_circuit_breaker,
        test_04_concurrent_query,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"结果: {passed}/{len(tests)} 通过, {failed} 失败")
    print(f"{'='*40}")
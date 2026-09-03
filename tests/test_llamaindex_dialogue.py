"""
LlamaIndexRAGTool 多轮对话测试。

验证：
  1. 多轮对话功能正常
  2. 统计命中率（检索到相关结果的比例）
  3. 统计响应时间（平均/最大/最小）
  4. 对比不同模式性能（纯混合检索 vs MQE vs HyDE）

用法：
  cd 项目根目录
  python tests/test_llamaindex_dialogue.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool

# ============================================================
# 测试用例定义
# 每个用例包含：问题、期望的知识域、是否期望有结果
# ============================================================

TEST_CASES = [
    # ---- RAG 原理相关（应该命中） ----
    {"q": "什么是 RAG？", "domain": "RAG 定义", "expect_hit": True},
    {"q": "RAG 解决了大语言模型的哪些问题？", "domain": "RAG 解决的问题", "expect_hit": True},
    {"q": "RAG 的核心组件有哪些？", "domain": "RAG 组成", "expect_hit": True},
    {"q": "什么是检索增强生成？", "domain": "RAG 定义（同义）", "expect_hit": True},
    {"q": "RAG 技术有哪些优势？", "domain": "RAG 优势", "expect_hit": True},

    # ---- 分块策略相关 ----
    {"q": "文本分块有哪些策略？", "domain": "Chunking 策略", "expect_hit": True},
    {"q": "什么是语义分块？", "domain": "语义分块", "expect_hit": True},
    {"q": "固定大小分块和递归分块有什么区别？", "domain": "分块对比", "expect_hit": True},

    # ---- LlamaIndex 相关 ----
    {"q": "LlamaIndex 是什么？", "domain": "LlamaIndex 框架", "expect_hit": True},
    {"q": "LlamaIndex 的核心概念有哪些？", "domain": "LlamaIndex 概念", "expect_hit": True},
    {"q": "如何在 LlamaIndex 中构建索引？", "domain": "LlamaIndex 使用方法", "expect_hit": True},

    # ---- 高级 RAG 技术 ----
    {"q": "什么是 Parent-Child 分层索引？", "domain": "Parent-Child 索引", "expect_hit": True},
    {"q": "MQE 多查询扩展是什么？", "domain": "MQE", "expect_hit": True},
    {"q": "HyDE 假设文档嵌入的原理是什么？", "domain": "HyDE", "expect_hit": True},
    {"q": "混合检索为什么有效？", "domain": "混合检索", "expect_hit": True},

    # ---- 工程相关（应该命中） ----
    {"q": "向量嵌入常用的模型有哪些？", "domain": "嵌入模型", "expect_hit": True},
    {"q": "PostgreSQL 如何存储向量？", "domain": "pgvector", "expect_hit": True},
    {"q": "余弦距离和欧氏距离有什么区别？", "domain": "距离度量", "expect_hit": True},

    # ---- 可能不相关的问题（应该不命中） ----
    {"q": "今天天气怎么样？", "domain": "天气（无关）", "expect_hit": False},
    {"q": "Python 的装饰器怎么用？", "domain": "Python（无关）", "expect_hit": False},
    {"q": "什么是机器学习？", "domain": "ML 定义（部分相关）", "expect_hit": False},
]


def run_test(tool, question, domain, expect_hit, mode="query"):
    """运行单个测试用例。"""
    start = time.time()
    try:
        if mode == "query":
            result = tool.run({"action": "query", "question": question, "top_k": 3})
        else:
            result = tool.run({"action": "search", "question": question, "top_k": 3})
        elapsed = time.time() - start
    except Exception as e:
        return {
            "question": question,
            "domain": domain,
            "expect_hit": expect_hit,
            "success": False,
            "error": str(e),
            "elapsed": time.time() - start,
            "has_result": False,
            "result_preview": "",
        }

    # 判断是否命中
    has_result = "引用来源" in result or "检索结果" in result
    result_preview = result[:200].replace("\n", " ") if result else ""

    return {
        "question": question,
        "domain": domain,
        "expect_hit": expect_hit,
        "success": True,
        "elapsed": round(elapsed, 2),
        "has_result": has_result,
        "result_preview": result_preview,
    }


def print_report(results, mode_name):
    """打印统计报告。"""
    total = len(results)
    success = [r for r in results if r["success"]]
    hits = [r for r in results if r["has_result"]]
    expected_hits = [r for r in results if r["expect_hit"]]
    actual_hits_on_expected = [r for r in results if r["expect_hit"] and r["has_result"]]
    times = [r["elapsed"] for r in results if r["success"]]

    hit_rate = len(actual_hits_on_expected) / len(expected_hits) * 100 if expected_hits else 0
    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0
    min_time = min(times) if times else 0

    # 按领域统计
    domain_stats = {}
    for r in results:
        d = r["domain"]
        if d not in domain_stats:
            domain_stats[d] = {"total": 0, "hits": 0, "times": []}
        domain_stats[d]["total"] += 1
        if r["has_result"]:
            domain_stats[d]["hits"] += 1
        if r["success"]:
            domain_stats[d]["times"].append(r["elapsed"])

    print(f"\n{'='*60}")
    print(f"📊 {mode_name} 模式 - 测试报告")
    print(f"{'='*60}")
    print(f"  总用例: {total}")
    print(f"  成功: {len(success)}")
    print(f"  失败: {total - len(success)}")
    print(f"  期望命中: {len(expected_hits)}")
    print(f"  实际命中（期望域）: {len(actual_hits_on_expected)}")
    print(f"  🎯 命中率: {hit_rate:.1f}%")
    print(f"  ⏱  平均响应: {avg_time:.2f}s")
    print(f"  ⏱  最大响应: {max_time:.2f}s")
    print(f"  ⏱  最小响应: {min_time:.2f}s")

    print(f"\n{'─'*60}")
    print(f"  按领域明细:")
    print(f"{'─'*60}")
    for d, s in sorted(domain_stats.items()):
        avg_d = sum(s["times"]) / len(s["times"]) if s["times"] else 0
        exp = "✅" if s["hits"] > 0 else "❌"
        print(f"    {exp} {d}: {s['hits']}/{s['total']} 命中, {avg_d:.2f}s")

    # 详细结果
    print(f"\n{'─'*60}")
    print(f"  详细结果:")
    print(f"{'─'*60}")
    for r in results:
        icon = "✅" if r["has_result"] == r["expect_hit"] else "⚠️"
        status = "✓" if r["success"] else "✗"
        hit = "HIT" if r["has_result"] else "MISS"
        print(f"    {icon} [{status}/{hit}] ({r['elapsed']:.2f}s) {r['question'][:50]}")

    overall = {
        "mode": mode_name,
        "total": total,
        "success": len(success),
        "hit_rate": round(hit_rate, 1),
        "avg_time": round(avg_time, 2),
        "max_time": round(max_time, 2),
        "min_time": round(min_time, 2),
        "timestamp": datetime.now().isoformat(),
    }

    return overall


def main():
    print("=" * 60)
    print("LlamaIndexRAGTool 多轮对话测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  测试用例: {len(TEST_CASES)} 个")
    print("=" * 60)

    tool = LlamaIndexRAGTool()

    # 测试模式 1: query（含 MQE 和合成回答）
    print("\n▶ 运行模式 1: query（全功能：混合检索 + MQE + HyDE + 合成回答）")
    results_query = []
    for i, tc in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {tc['q'][:50]}...", end=" ", flush=True)
        r = run_test(tool, tc["q"], tc["domain"], tc["expect_hit"], mode="query")
        results_query.append(r)
        print(f"({r['elapsed']:.2f}s)")
    report_query = print_report(results_query, "query（全功能）")

    # 测试模式 2: search（纯检索，不生成回答）
    print("\n▶ 运行模式 2: search（纯检索，无合成回答）")
    tool_search = LlamaIndexRAGTool(config={"enable_mqe": False, "enable_hyde": False})
    results_search = []
    for i, tc in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {tc['q'][:50]}...", end=" ", flush=True)
        r = run_test(tool_search, tc["q"], tc["domain"], tc["expect_hit"], mode="search")
        results_search.append(r)
        print(f"({r['elapsed']:.2f}s)")
    report_search = print_report(results_search, "search（纯检索）")

    # 汇总对比
    print(f"\n{'='*60}")
    print(f"🏆 汇总对比")
    print(f"{'='*60}")
    print(f"  {'指标':<20} {'query（全功能）':<25} {'search（纯检索）':<25}")
    print(f"  {'─'*20} {'─'*25} {'─'*25}")
    print(f"  {'命中率':<20} {report_query['hit_rate']:<25.1f}% {report_search['hit_rate']:<25.1f}%")
    print(f"  {'平均响应':<20} {report_query['avg_time']:<25.2f}s {report_search['avg_time']:<25.2f}s")
    print(f"  {'最大响应':<20} {report_query['max_time']:<25.2f}s {report_search['max_time']:<25.2f}s")
    print(f"  {'最小响应':<20} {report_query['min_time']:<25.2f}s {report_search['min_time']:<25.2f}s")

    # 保存报告
    report = {
        "query": report_query,
        "search": report_search,
        "test_cases": len(TEST_CASES),
    }
    report_path = Path(__file__).resolve().parent.parent / "logs" / "llamaindex_benchmark.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {report_path}")

    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
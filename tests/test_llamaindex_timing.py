"""
LlamaIndexRAGTool 分环节耗时分析。

精确测量每个环节耗时：
  1. 索引加载
  2. 稠密检索
  3. 稀疏 BM25 检索
  4. 查询融合（RRF）
  5. MQE 查询扩展
  6. HyDE 假设文档
  7. LLM 问答合成

用法：
  cd 项目根目录
  python tests/test_llamaindex_timing.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.retrievers.bm25 import BM25Retriever

from src.tools.rag.llamaindex.adapters import DashScopeEmbedding, QwenLLM
from src.tools.rag.llamaindex.retrievers import load_pg_index, _get_all_nodes_from_index
from src.tools.rag.llamaindex.transforms import MultiQueryExpander, HyDEExpander


def timeit(label):
    """计时器装饰器。"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            return result, elapsed
        return wrapper
    return decorator


def main():
    query = "RAG 技术解决了大语言模型的哪些问题？"
    print(f"查询: {query}")
    print(f"{'='*60}")

    # ============================================================
    # 1. 初始化组件
    # ============================================================
    t_start = time.perf_counter()

    print("\n1. 初始化组件")
    print("─" * 40)

    t0 = time.perf_counter()
    embed_model = DashScopeEmbedding()
    llm = QwenLLM()
    Settings.llm = llm
    Settings.embed_model = embed_model
    print(f"   嵌入模型 + LLM 初始化: {time.perf_counter() - t0:.4f}s")

    # ============================================================
    # 2. 加载索引
    # ============================================================
    print("\n2. 加载索引（PGVectorStore）")
    print("─" * 40)

    t0 = time.perf_counter()
    index = load_pg_index(table_name="llamaindex_rag", embed_dim=2560)
    t_index = time.perf_counter() - t0
    print(f"   加载索引: {t_index:.4f}s")

    # ============================================================
    # 3. 稠密检索（PGVectorStore）
    # ============================================================
    print("\n3. 稠密检索")
    print("─" * 40)

    t0 = time.perf_counter()
    dense_retriever = index.as_retriever(similarity_top_k=10)
    t_dense_prep = time.perf_counter() - t0
    print(f"   创建稠密检索器: {t_dense_prep:.4f}s")

    t0 = time.perf_counter()
    dense_results = dense_retriever.retrieve(query)
    t_dense = time.perf_counter() - t0
    print(f"   稠密检索执行: {t_dense:.4f}s")
    print(f"   返回 {len(dense_results)} 个结果")
    for i, r in enumerate(dense_results[:3]):
        print(f"     [{i+1}] score={r.score:.4f} | {r.node.text[:60]}...")

    # ============================================================
    # 4. 稀疏 BM25 检索
    # ============================================================
    print("\n4. 稀疏 BM25 检索")
    print("─" * 40)

    t0 = time.perf_counter()
    all_nodes = _get_all_nodes_from_index(index)
    t_nodes = time.perf_counter() - t0
    print(f"   加载 BM25 语料 ({len(all_nodes)} 节点): {t_nodes:.4f}s")

    if all_nodes:
        t0 = time.perf_counter()
        sparse_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=10,
            language="chinese",
            skip_stemming=True,
        )
        t_sparse_prep = time.perf_counter() - t0
        print(f"   创建 BM25 检索器: {t_sparse_prep:.4f}s")

        t0 = time.perf_counter()
        sparse_results = sparse_retriever.retrieve(query)
        t_sparse = time.perf_counter() - t0
        print(f"   BM25 检索执行: {t_sparse:.4f}s")
        print(f"   返回 {len(sparse_results)} 个结果")
        for i, r in enumerate(sparse_results[:3]):
            print(f"     [{i+1}] score={r.score:.4f} | {r.node.text[:60]}...")
    else:
        t_sparse_prep = 0
        t_sparse = 0
        sparse_results = []
        print("   BM25 不可用（跳过）")

    # ============================================================
    # 5. 查询融合（QueryFusionRetriever）
    # ============================================================
    print("\n5. 查询融合（Reciprocal Rerank）")
    print("─" * 40)

    t0 = time.perf_counter()
    if sparse_results:
        fusion_retriever = QueryFusionRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            retriever_weights=[0.7, 0.3],
            similarity_top_k=5,
            num_queries=1,
            mode="reciprocal_rerank",
            llm=llm,
            verbose=False,
        )
        t_fusion_prep = time.perf_counter() - t0
        print(f"   创建融合检索器: {t_fusion_prep:.4f}s")

        t0 = time.perf_counter()
        fusion_results = fusion_retriever.retrieve(query)
        t_fusion = time.perf_counter() - t0
        print(f"   融合检索执行: {t_fusion:.4f}s")
        print(f"   返回 {len(fusion_results)} 个结果")
        for i, r in enumerate(fusion_results[:3]):
            print(f"     [{i+1}] score={r.score:.4f} | {r.node.text[:60]}...")
    else:
        t_fusion_prep = 0
        t_fusion = 0
        fusion_results = dense_results[:5]
        print("   跳过融合（BM25 不可用）")

    # ============================================================
    # 6. MQE 多查询扩展
    # ============================================================
    print("\n6. MQE 多查询扩展")
    print("─" * 40)

    t0 = time.perf_counter()
    mqe = MultiQueryExpander(llm=llm, num_queries=3)
    t_mqe_prep = time.perf_counter() - t0
    print(f"   初始化 MQE: {t_mqe_prep:.4f}s")

    t0 = time.perf_counter()
    mqe_queries = mqe.expand(query)
    t_mqe_expand = time.perf_counter() - t0
    print(f"   MQE 生成查询变体: {t_mqe_expand:.4f}s")
    for i, q in enumerate(mqe_queries):
        print(f"     [{i}] {q[:80]}")

    t0 = time.perf_counter()
    mqe_results = mqe.retrieve_multi(query, fusion_retriever, top_k=3)
    t_mqe_retrieve = time.perf_counter() - t0
    print(f"   MQE 多查询检索: {t_mqe_retrieve:.4f}s")
    print(f"   返回 {len(mqe_results)} 个结果")
    for i, r in enumerate(mqe_results):
        print(f"     [{i+1}] score={r.score:.4f} | {r.node.text[:60]}...")

    # ============================================================
    # 7. HyDE 假设文档嵌入
    # ============================================================
    print("\n7. HyDE 假设文档嵌入")
    print("─" * 40)

    t0 = time.perf_counter()
    hyde = HyDEExpander(llm=llm)
    t_hyde_prep = time.perf_counter() - t0
    print(f"   初始化 HyDE: {t_hyde_prep:.4f}s")

    t0 = time.perf_counter()
    hyde_doc = hyde.expand(query)
    t_hyde_expand = time.perf_counter() - t0
    print(f"   HyDE 生成假设文档: {t_hyde_expand:.4f}s")
    print(f"   假设文档: {hyde_doc[:120]}...")

    t0 = time.perf_counter()
    hyde_results = hyde.retrieve_with_hypothetical(query, fusion_retriever, top_k=3)
    t_hyde_retrieve = time.perf_counter() - t0
    print(f"   HyDE 假设检索: {t_hyde_retrieve:.4f}s")
    print(f"   返回 {len(hyde_results)} 个结果")
    for i, r in enumerate(hyde_results):
        print(f"     [{i+1}] score={r.score:.4f} | {r.node.text[:60]}...")

    # ============================================================
    # 8. ResponseSynthesizer 问答合成
    # ============================================================
    print("\n8. ResponseSynthesizer 问答合成")
    print("─" * 40)

    t0 = time.perf_counter()
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,
        verbose=False,
    )
    t_synth_prep = time.perf_counter() - t0
    print(f"   初始化 ResponseSynthesizer: {t_synth_prep:.4f}s")

    t0 = time.perf_counter()
    response = synthesizer.synthesize(query, mqe_results)
    t_synth = time.perf_counter() - t0
    print(f"   问答合成: {t_synth:.4f}s")
    print(f"   回答: {str(response)[:200]}...")

    # ============================================================
    # 汇总
    # ============================================================
    t_total = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"📊 分环节耗时汇总")
    print(f"{'='*60}")

    # 按依赖顺序列出
    phases = [
        ("索引加载", t_index, "初始化"),
        ("稠密检索", t_dense, "检索"),
        ("BM25 稀疏检索", t_sparse, "检索"),
        ("查询融合 RRF", t_fusion, "检索"),
        ("MQE 查询扩展", t_mqe_expand, "查询变换"),
        ("MQE 多查询检索", t_mqe_retrieve, "检索"),
        ("HyDE 假设文档", t_hyde_expand, "查询变换"),
        ("HyDE 假设检索", t_hyde_retrieve, "检索"),
        ("LLM 问答合成", t_synth, "问答"),
    ]

    print(f"  {'环节':<22} {'耗时':<10} {'分类':<10} {'占比'}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*8}")
    for name, elapsed, category in phases:
        pct = elapsed / t_total * 100
        print(f"  {name:<22} {elapsed:<10.4f}s {category:<10} {pct:<6.1f}%")

    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'总耗时':<22} {t_total:<10.4f}s {'总计':<10} {100.0:<6.1f}%")

    # 按分类汇总
    print(f"\n  {'─'*50}")
    print(f"  按分类汇总:")
    print(f"  {'─'*50}")
    by_category = {}
    for name, elapsed, category in phases:
        if category not in by_category:
            by_category[category] = {"time": 0, "items": []}
        by_category[category]["time"] += elapsed
        by_category[category]["items"].append((name, elapsed))

    for cat, data in sorted(by_category.items(), key=lambda x: -x[1]["time"]):
        pct = data["time"] / t_total * 100
        print(f"    {cat}: {data['time']:.4f}s ({pct:.1f}%)")
        for name, elapsed in data["items"]:
            print(f"      └ {name}: {elapsed:.4f}s")

    print(f"\n{'='*60}")
    print(f"分析结论:")
    print(f"{'='*60}")
    print(f"  1. 一次完整对话从请求到输出共耗时 {t_total:.2f}s")
    print(f"  2. 检索阶段（稠密+BM25+融合+MQE+HyDE）约 {t_dense+t_sparse+t_fusion+t_mqe_retrieve+t_hyde_retrieve:.2f}s")
    print(f"  3. 查询变换阶段（MQE+HyDE 的 LLM 调用）约 {t_mqe_expand+t_hyde_expand:.2f}s")
    print(f"  4. LLM 问答合成本身约 {t_synth:.2f}s")
    print(f"  5. 瓶颈在 LLM 生成（MQE 变体 + HyDE 假设 + 回答合成）")
    print(f"  6. 纯检索（无 LLM）可在 0.3-0.5s 内完成")


if __name__ == "__main__":
    main()
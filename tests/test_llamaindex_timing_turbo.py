"""
纯混合检索 + qwen-turbo 分环节耗时分析。

对比纯混合检索 + qwen-plus 的数据，评估换用更轻量模型的效果。

用法：
  cd 项目根目录
  python tests/test_llamaindex_timing_turbo.py
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


def main():
    query = "RAG 技术解决了大语言模型的哪些问题？"
    print(f"查询: {query}")
    print(f"{'='*60}")
    print(f"模型: qwen-turbo | 模式: 纯混合检索（禁用 MQE + HyDE）")
    print(f"{'='*60}")

    # ============================================================
    # 1. 初始化组件
    # ============================================================
    t_start = time.perf_counter()

    print("\n1. 初始化组件")
    print("─" * 40)

    t0 = time.perf_counter()
    embed_model = DashScopeEmbedding()
    llm = QwenLLM(model="qwen-turbo")
    Settings.llm = llm
    Settings.embed_model = embed_model
    print(f"   嵌入模型 + LLM(qwen-turbo) 初始化: {time.perf_counter() - t0:.4f}s")

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
    # 3. 稠密检索
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
    print(f"   稠密检索执行: {t_dense:.4f}s | 返回 {len(dense_results)} 个结果")

    # ============================================================
    # 4. BM25 稀疏检索
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
            nodes=all_nodes, similarity_top_k=10,
            language="chinese", skip_stemming=True,
        )
        t_sparse_prep = time.perf_counter() - t0
        t0 = time.perf_counter()
        sparse_results = sparse_retriever.retrieve(query)
        t_sparse = time.perf_counter() - t0
        print(f"   BM25 检索执行: {t_sparse:.4f}s | 返回 {len(sparse_results)} 个结果")
    else:
        t_sparse_prep = 0; t_sparse = 0; sparse_results = []
        print("   BM25 不可用（跳过）")

    # ============================================================
    # 5. 查询融合
    # ============================================================
    print("\n5. 查询融合（Reciprocal Rerank）")
    print("─" * 40)

    t0 = time.perf_counter()
    if sparse_results:
        fusion_retriever = QueryFusionRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            retriever_weights=[0.7, 0.3],
            similarity_top_k=5, num_queries=1,
            mode="reciprocal_rerank", llm=llm, verbose=False,
        )
        t_fusion_prep = time.perf_counter() - t0
        t0 = time.perf_counter()
        fusion_results = fusion_retriever.retrieve(query)
        t_fusion = time.perf_counter() - t0
        print(f"   融合检索执行: {t_fusion:.4f}s | 返回 {len(fusion_results)} 个结果")
    else:
        t_fusion_prep = 0; t_fusion = 0
        fusion_results = dense_results[:5]
        print("   跳过融合")

    # ============================================================
    # 6. 问答合成（qwen-turbo）
    # ============================================================
    print("\n6. ResponseSynthesizer 问答合成（qwen-turbo）")
    print("─" * 40)

    t0 = time.perf_counter()
    synthesizer = get_response_synthesizer(
        llm=llm, response_mode=ResponseMode.COMPACT, verbose=False,
    )
    t_synth_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    response = synthesizer.synthesize(query, fusion_results)
    t_synth = time.perf_counter() - t0
    print(f"   问答合成: {t_synth:.4f}s")
    print(f"   回答: {str(response)[:200]}...")

    # ============================================================
    # 汇总
    # ============================================================
    t_total = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"分环节耗时汇总（qwen-turbo + 纯混合检索）")
    print(f"{'='*60}")

    phases = [
        ("索引加载", t_index, "初始化"),
        ("稠密检索", t_dense, "检索"),
        ("BM25 稀疏检索", t_sparse, "检索"),
        ("查询融合 RRF", t_fusion, "检索"),
        ("LLM 问答合成", t_synth, "问答"),
    ]

    print(f"  {'环节':<22} {'耗时':<10} {'分类':<10} {'占比'}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*8}")
    for name, elapsed, category in phases:
        pct = elapsed / t_total * 100
        print(f"  {name:<22} {elapsed:<10.4f}s {category:<10} {pct:<6.1f}%")

    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'总耗时':<22} {t_total:<10.4f}s {'总计':<10} {100.0:<6.1f}%")

    # 对比
    print(f"\n{'='*60}")
    print(f"模型对比:")
    print(f"{'='*60}")
    print(f"  {'指标':<30} {'qwen-plus':<18} {'qwen-turbo':<18}")
    print(f"  {'─'*30} {'─'*18} {'─'*18}")
    print(f"  {'LLM 问答合成':<30} {'5.6571':<18} {t_synth:<18.4f}s")
    print(f"  {'总耗时':<30} {'6.3512':<18} {t_total:<18.4f}s")
    if t_synth < 5.6571:
        print(f"  {'问答合成加速比':<30} {'1.00x':<18} {5.6571/t_synth:<18.2f}x")


if __name__ == "__main__":
    main()
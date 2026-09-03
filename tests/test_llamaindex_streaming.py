"""
流式输出感知速度测试（qwen-turbo）。

测量：
  - 首 token 延迟（从请求到第一个 token 出现的时间）
  - 总流式完成时间
  - 每秒输出字符数
  - 构建检索上下文（非流式）的耗时

用法：
  cd 项目根目录
  python tests/test_llamaindex_streaming.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from src.tools.rag.llamaindex.adapters import DashScopeEmbedding, QwenLLM
from src.tools.rag.llamaindex.retrievers import load_pg_index, _get_all_nodes_from_index


def main():
    query = "RAG 技术解决了大语言模型的哪些问题？请详细说明每个优势的原理和实际应用场景。"
    print(f"查询: {query}")
    print(f"{'='*60}")
    print(f"模型: qwen-turbo | 模式: 流式输出")
    print(f"{'='*60}")

    # ============================================================
    # 阶段 1: 检索上下文（非流式）
    # ============================================================
    t_start = time.perf_counter()

    print("\n1. 检索阶段（非流式）")
    print("─" * 40)

    embed_model = DashScopeEmbedding()
    llm = QwenLLM(model="qwen-turbo")
    Settings.llm = llm
    Settings.embed_model = embed_model

    t0 = time.perf_counter()
    index = load_pg_index(table_name="llamaindex_rag", embed_dim=2560)
    t_index = time.perf_counter() - t0
    print(f"   加载索引: {t_index:.4f}s")

    t0 = time.perf_counter()
    dense_retriever = index.as_retriever(similarity_top_k=10)
    dense_results = dense_retriever.retrieve(query)
    t_dense = time.perf_counter() - t0
    print(f"   稠密检索: {t_dense:.4f}s")

    t0 = time.perf_counter()
    all_nodes = _get_all_nodes_from_index(index)
    sparse_retriever = BM25Retriever.from_defaults(
        nodes=all_nodes, similarity_top_k=10,
        language="chinese", skip_stemming=True,
    )
    sparse_results = sparse_retriever.retrieve(query)
    t_sparse = time.perf_counter() - t0
    print(f"   BM25 检索: {t_sparse:.4f}s")

    t0 = time.perf_counter()
    fusion_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        retriever_weights=[0.7, 0.3],
        similarity_top_k=5, num_queries=1,
        mode="reciprocal_rerank", llm=llm, verbose=False,
    )
    fusion_results = fusion_retriever.retrieve(query)
    t_fusion = time.perf_counter() - t0
    print(f"   融合检索: {t_fusion:.4f}s")

    # 构建上下文
    context = "\n\n".join([f"[{i+1}] {r.node.text}" for i, r in enumerate(fusion_results)])
    t_retrieval = time.perf_counter() - t_start
    print(f"   检索阶段总耗时: {t_retrieval:.4f}s")
    print(f"   检索到 {len(fusion_results)} 个相关片段")

    # ============================================================
    # 阶段 2: 流式问答
    # ============================================================
    print("\n2. 流式问答阶段（qwen-turbo）")
    print("─" * 40)

    prompt = f"""你是一个知识库问答助手。请严格基于给定的资料片段回答用户问题。

【资料片段】
{context}

【问题】
{query}

【回答要求】
1. 只依据资料片段回答，资料中没有的信息请明确说明「资料中未提及」
2. 回答末尾标注引用的资料编号，如 [1][2]
3. 使用简洁清晰的中文回答"""

    # 首 token 延迟
    print("\n  回答内容（流式输出）:")
    print("  " + "─" * 40)

    t0 = time.perf_counter()
    first_token = True
    chunk_count = 0
    total_chars = 0
    full_text = ""

    for chunk in llm.stream_complete(prompt):
        if first_token:
            t_first_token = time.perf_counter() - t0
            first_token = False
            print(f"  [首 token 到达: {t_first_token:.3f}s]", end="")
        chunk_count += 1
        total_chars += len(chunk.delta or "")
        full_text += chunk.text
        # 显示流式输出（不换行）
        print(chunk.delta or "", end="", flush=True)

    t_stream = time.perf_counter() - t0
    print(f"\n\n  {'─'*40}")

    # ============================================================
    # 汇总
    # ============================================================
    t_total = time.perf_counter() - t_start

    chars_per_sec = total_chars / t_stream if t_stream > 0 else 0

    print(f"\n{'='*60}")
    print(f"流式感知速度汇总")
    print(f"{'='*60}")
    print(f"  {'指标':<30} {'值':<20}")
    print(f"  {'─'*30} {'─'*20}")
    print(f"  {'模型':<30} {'qwen-turbo':<20}")
    print(f"  {'检索阶段耗时':<30} {t_retrieval:<20.4f}s")
    print(f"  {'首 token 延迟':<30} {t_first_token:<20.4f}s")
    print(f"  {'流式完成时间':<30} {t_stream:<20.4f}s")
    print(f"  {'总耗时（请求到输出完整回答）':<30} {t_total:<20.4f}s")
    print(f"  {'回答总字符数':<30} {total_chars:<20d}")
    print(f"  {'Chunk 数':<30} {chunk_count:<20d}")
    print(f"  {'输出速度':<30} {chars_per_sec:<20.1f} chars/s")
    print(f"  {'用户感知等待时间':<30} {t_retrieval + t_first_token:<20.4f}s")
    print(f"  {'（检索 + 首 token）= 用户看到第一个字的时间':<30}")

    # 对比
    print(f"\n{'='*60}")
    print(f"与之前对比（同查询 qwen-plus 非流式）:")
    print(f"{'='*60}")
    print(f"  qwen-plus 非流式: 总耗时 6.35s（用户等待全部文本才看到）")
    print(f"  qwen-turbo 流式:  首 token {t_first_token:.2f}s 后开始输出，"
          f"总 {t_total:.2f}s 完成")
    print(f"  {'='*30}")
    print(f"  感知提升: 用户等待时间从 6.35s → {t_retrieval + t_first_token:.2f}s")
    print(f"  （约缩短 {100 - (t_retrieval + t_first_token)/6.35*100:.0f}%）")


if __name__ == "__main__":
    main()
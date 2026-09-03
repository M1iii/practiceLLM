"""
第三阶段：端到端 QA 脚本。

集成：
  - 混合检索（稠密 PGVectorStore + 稀疏 BM25）
  - MQE 多查询扩展 + HyDE 假设文档嵌入
  - ResponseSynthesizer 完善问答体验

用法：
  cd 项目根目录
  python -m rag.llamaindex.run_qa [--query "你的问题"] [--no-hybrid] [--no-mqe] [--no-hyde]
"""

import os
import sys
import argparse
import time
from pathlib import Path


from dotenv import load_dotenv
load_dotenv()

from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import Settings

from src.tools.rag.llamaindex.adapters import DashScopeEmbedding, QwenLLM
from src.tools.rag.llamaindex.retrievers import load_pg_index, _get_all_nodes_from_index
from src.tools.rag.llamaindex.transforms import MultiQueryExpander, HyDEExpander, QueryTransformPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="LlamaIndex 端到端 QA（第三阶段）")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="单个查询问题（不传则进入交互模式）")
    parser.add_argument("--table-name", type=str, default="llamaindex_rag")
    parser.add_argument("--embed-dim", type=int, default=2560)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-hybrid", action="store_true",
                        help="禁用混合检索（仅使用稠密检索）")
    parser.add_argument("--no-mqe", action="store_true",
                        help="禁用 MQE 多查询扩展")
    parser.add_argument("--no-hyde", action="store_true",
                        help="禁用 HyDE 假设文档嵌入")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("第三阶段：LlamaIndex 端到端 QA")
    print("=" * 60)

    # 1. 初始化组件
    print("\n初始化组件...")
    embed_model = DashScopeEmbedding()
    llm = QwenLLM()
    Settings.llm = llm
    Settings.embed_model = embed_model
    print(f"  嵌入模型: {embed_model.class_name()}")
    print(f"  LLM: {llm.model}")

    # 2. 加载索引
    print(f"\n加载 PostgreSQL 索引 ({args.table_name})...")
    index = load_pg_index(table_name=args.table_name, embed_dim=args.embed_dim)
    print("  索引加载完成")

    # 3. 构建检索器
    print("\n构建检索器...")
    if args.no_hybrid:
        # 仅稠密检索
        retriever = index.as_retriever(similarity_top_k=args.top_k)
        print("  模式: 仅稠密检索")
    else:
        # 混合检索
        dense_retriever = index.as_retriever(similarity_top_k=args.top_k * 2)
        all_nodes = _get_all_nodes_from_index(index)

        if not all_nodes:
            print("  警告: 无法从索引中提取节点，BM25 将使用稠密检索结果作为语料")
            sparse_retriever = None
        else:
            from llama_index.retrievers.bm25 import BM25Retriever
            sparse_retriever = BM25Retriever.from_defaults(
                nodes=all_nodes,
                similarity_top_k=args.top_k * 2,
                language="chinese",
                skip_stemming=True,
            )
            print(f"  BM25 语料: {len(all_nodes)} 个节点")

        if sparse_retriever:
            retriever = QueryFusionRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                retriever_weights=[0.7, 0.3],
                similarity_top_k=args.top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                llm=llm,
                verbose=args.verbose,
            )
            print("  模式: 混合检索（稠密 0.7 + 稀疏 0.3）")
        else:
            retriever = dense_retriever
            print("  模式: 仅稠密检索（BM25 不可用）")

    # 4. 初始化查询变换
    mqe = MultiQueryExpander(llm=llm, num_queries=3) if not args.no_mqe else None
    hyde = HyDEExpander(llm=llm) if not args.no_hyde else None
    pipeline = QueryTransformPipeline(mqe=mqe, hyde=hyde) if (mqe or hyde) else None

    if pipeline:
        print(f"  查询变换: MQE={not args.no_mqe}, HyDE={not args.no_hyde}")

    # 5. 初始化 ResponseSynthesizer
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,
        verbose=args.verbose,
    )

    print()
    print("=" * 60)
    print("准备就绪，开始问答")
    print("=" * 60)

    # 6. 交互或单次查询
    if args.query:
        queries = [args.query]
    else:
        print("\n输入问题（输入 q 退出）\n")
        queries = []
        while True:
            q = input(">>> ").strip()
            if q.lower() in ("q", "quit", "exit"):
                break
            if q:
                queries.append(q)

    for question in queries:
        print(f"\n{'=' * 60}")
        print(f"问题: {question}")
        print(f"{'=' * 60}")

        start = time.time()

        # 6a. 查询变换
        if pipeline:
            transformed = pipeline.transform(question)
            if args.verbose:
                print(f"\n查询变换 ({len(transformed)} 个变体):")
                for t in transformed:
                    print(f"  - {t[:80]}...")

        # 6b. 检索
        retrieval_start = time.time()
        if mqe and not args.no_mqe:
            # MQE 检索
            results = mqe.retrieve_multi(question, retriever, top_k=args.top_k)
        elif hyde and not args.no_hyde:
            # HyDE 检索
            results = hyde.retrieve_with_hypothetical(question, retriever, top_k=args.top_k)
        else:
            # 直接检索
            results = retriever.retrieve(question)

        retrieval_time = time.time() - retrieval_start

        print(f"\n检索结果 ({len(results)} 个, {retrieval_time:.2f}s):")
        for i, r in enumerate(results):
            score = r.score
            src = r.node.metadata.get("file_name", "未知")
            preview = r.node.text[:80].replace("\n", " ")
            if score is not None:
                print(f"  [{i + 1}] score={score:.4f} ({src})\n      {preview}...")
            else:
                print(f"  [{i + 1}] ({src})\n      {preview}...")

        # 6c. 合成回答
        if results:
            synthesize_start = time.time()
            response = synthesizer.synthesize(
                question,
                results,
            )
            synthesize_time = time.time() - synthesize_start

            print(f"\n回答 ({synthesize_time:.2f}s):\n{response}")
        else:
            print("\n未检索到相关内容")

        total_time = time.time() - start
        print(f"\n总耗时: {total_time:.2f}s")

    print("\n" + "=" * 60)
    print("QA 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
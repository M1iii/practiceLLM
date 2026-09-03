"""
混合检索模块：稠密检索（PGVectorStore）+ 稀疏检索（BM25），
通过 QueryFusionRetriever 进行融合。

用法：
    from src.tools.rag.llamaindex.retrievers import build_hybrid_retriever

    retriever = build_hybrid_retriever(
        similarity_top_k=5,
        dense_weight=0.7,
        sparse_weight=0.3,
    )
    results = retriever.retrieve("什么是 RAG？")
"""

import os
from typing import Optional, List

from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

from src.tools.rag.llamaindex.adapters import DashScopeEmbedding


def load_pg_index(
    table_name: str = "llamaindex_rag",
    embed_dim: int = 2560,
) -> VectorStoreIndex:
    """从 PostgreSQL 加载现有索引。"""
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = int(os.getenv("PG_PORT", "5433"))
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    database = os.getenv("PG_DATABASE", "practice_llm")

    vector_store = PGVectorStore.from_params(
        database=database,
        host=host,
        port=port,
        user=user,
        password=password,
        table_name=table_name,
        embed_dim=embed_dim,
        hnsw_kwargs=None,
    )

    embed_model = DashScopeEmbedding()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


def build_hybrid_retriever(
    similarity_top_k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    table_name: str = "llamaindex_rag",
    embed_dim: int = 2560,
    verbose: bool = False,
) -> QueryFusionRetriever:
    """构建混合检索器（稠密 + 稀疏 BM25）。

    参数:
        similarity_top_k: 最终返回的 top-k 结果数
        dense_weight: 稠密检索权重（0-1）
        sparse_weight: 稀疏检索权重（0-1）
        table_name: PostgreSQL 表名
        embed_dim: 嵌入维度
        verbose: 是否打印详细信息
    """
    # 1. 从 PostgreSQL 加载索引
    index = load_pg_index(table_name=table_name, embed_dim=embed_dim)

    # 2. 稠密检索器（PGVectorStore）
    dense_retriever = index.as_retriever(similarity_top_k=similarity_top_k * 2)

    # 3. 稀疏检索器（BM25）
    # 从索引中提取节点用于 BM25
    retriever_mode = index.as_retriever(similarity_top_k=similarity_top_k * 2)
    # 先执行一次检索获取节点列表（用于初始化 BM25 的语料库）
    # 注意：需要从 PGVectorStore 中提取所有节点
    all_nodes = _get_all_nodes_from_index(index)

    sparse_retriever = BM25Retriever.from_defaults(
        nodes=all_nodes,
        similarity_top_k=similarity_top_k * 2,
        language="chinese",
        skip_stemming=True,
    )

    # 4. 融合检索器
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        retriever_weights=[dense_weight, sparse_weight],
        similarity_top_k=similarity_top_k,
        num_queries=1,  # 不生成额外查询变体，直接融合
        mode="reciprocal_rerank",
        verbose=verbose,
    )

    return hybrid_retriever


def _get_all_nodes_from_index(index: VectorStoreIndex) -> list:
    """从 PGVectorStore 中提取所有节点（用于 BM25 初始化）。"""
    import json
    from sqlalchemy import create_engine, text
    from llama_index.core.schema import TextNode

    # 从环境变量读取连接参数
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = int(os.getenv("PG_PORT", "5433"))
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    database = os.getenv("PG_DATABASE", "practice_llm")

    # 获取表名
    try:
        table_name = index.vector_store.table_name if hasattr(index, "vector_store") else "data_llamaindex_rag"
        if not table_name.startswith("data_"):
            table_name = f"data_{table_name}"
    except Exception:
        table_name = "data_llamaindex_rag"

    url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT node_id, text, metadata_ FROM {table_name} LIMIT 1000")
            ).fetchall()

        nodes = []
        for row in rows:
            node_id, text, metadata_val = row
            if isinstance(metadata_val, str):
                metadata = json.loads(metadata_val)
            elif isinstance(metadata_val, dict):
                metadata = metadata_val
            else:
                metadata = {}
            node = TextNode(text=text, node_id=node_id or "", metadata=metadata)
            nodes.append(node)
        return nodes
    except Exception as e:
        print(f"  警告: 从 PostgreSQL 提取节点失败: {e}")
        return []


def retrieve_hybrid(
    query: str,
    top_k: int = 5,
    table_name: str = "llamaindex_rag",
    verbose: bool = False,
) -> list:
    """便捷方法：一步完成混合检索。"""
    retriever = build_hybrid_retriever(
        similarity_top_k=top_k,
        table_name=table_name,
        verbose=verbose,
    )
    return retriever.retrieve(query)
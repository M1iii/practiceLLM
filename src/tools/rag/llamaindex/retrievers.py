"""混合检索模块：稠密检索（PGVectorStore）+ 稀疏检索（BM25），通过 QueryFusionRetriever 融合。"""

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
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = int(os.getenv("PG_PORT", "5433"))
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    database = os.getenv("PG_DATABASE", "practice_llm")

    vector_store = PGVectorStore.from_params(
        database=database, host=host, port=port, user=user,
        password=password, table_name=table_name, embed_dim=embed_dim,
        hnsw_kwargs=None,
    )
    embed_model = DashScopeEmbedding()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model,
    )


def build_hybrid_retriever(
    similarity_top_k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    table_name: str = "llamaindex_rag",
    embed_dim: int = 2560,
    verbose: bool = False,
) -> QueryFusionRetriever:
    index = load_pg_index(table_name=table_name, embed_dim=embed_dim)
    dense_retriever = index.as_retriever(similarity_top_k=similarity_top_k * 2)
    all_nodes = _get_all_nodes_from_index(index)
    if not all_nodes:
        return QueryFusionRetriever(
            retrievers=[dense_retriever], retriever_weights=[1.0],
            similarity_top_k=similarity_top_k, num_queries=1,
            mode="reciprocal_rerank", verbose=verbose,
        )
    sparse_retriever = BM25Retriever.from_defaults(
        nodes=all_nodes, similarity_top_k=similarity_top_k * 2,
        language="chinese", skip_stemming=True,
    )
    return QueryFusionRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        retriever_weights=[dense_weight, sparse_weight],
        similarity_top_k=similarity_top_k, num_queries=1,
        mode="reciprocal_rerank", verbose=verbose,
    )


def _get_all_nodes_from_index(index: VectorStoreIndex) -> list:
    import json
    from sqlalchemy import create_engine, text
    from llama_index.core.schema import TextNode

    host = os.getenv("PG_HOST", "127.0.0.1")
    port = int(os.getenv("PG_PORT", "5433"))
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    database = os.getenv("PG_DATABASE", "practice_llm")

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
    query: str, top_k: int = 5, table_name: str = "llamaindex_rag",
    verbose: bool = False,
) -> list:
    retriever = build_hybrid_retriever(
        similarity_top_k=top_k, table_name=table_name, verbose=verbose,
    )
    return retriever.retrieve(query)
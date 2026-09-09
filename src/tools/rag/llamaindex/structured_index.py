"""结构化索引模块：元数据抽取 + JSONB GIN 索引 + 结构化过滤检索。"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple

from llama_index.core.schema import Document, TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection


class DocumentMetadata:
    """文档结构化元数据。"""
    def __init__(
        self,
        doc_type: str = "general",        # 文档类型: concept/tutorial/reference/technical/general
        tags: Optional[List[str]] = None,  # 关键词标签数组
        section: Optional[str] = None,     # 章节标题（从文档第一行提取）
        file_name: Optional[str] = None,   # 文件名
        file_type: Optional[str] = None,   # 文件类型 (.md/.txt/.py 等)
    ):
        self.doc_type = doc_type
        self.tags = tags or []
        self.section = section
        self.file_name = file_name
        self.file_type = file_type

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，存入 metadata_ 字段。"""
        return {
            "doc_type": self.doc_type,
            "tags": self.tags,
            "section": self.section,
            "file_name": self.file_name,
            "file_type": self.file_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """从字典恢复。"""
        return cls(
            doc_type=data.get("doc_type", "general"),
            tags=data.get("tags", []),
            section=data.get("section"),
            file_name=data.get("file_name"),
            file_type=data.get("file_type"),
        )


class MetadataExtractor:
    """规则抽取结构化元数据，无需 LLM。"""

    DOC_TYPE_KEYWORDS = {
        "concept": [
            "什么是", "定义", "介绍", "概念", "解释", "概述", "简介",
            "原理", "说明", "基础", "入门", "指南",
        ],
        "tutorial": [
            "教程", "步骤", "如何", "怎么", "方法", "手把手", "实践",
            "实现", "代码", "示例", "例子", "演示", "练习",
        ],
        "technical": [
            "技术", "架构", "设计", "实现", "细节", "分析", "对比",
            "比较", "优化", "改进", "性能", "瓶颈", "问题",
        ],
        "reference": [
            "参考", "手册", "API", "文档", "参数", "配置", "选项",
            "命令", "用法", "示例",
        ],
    }

    # 常见技术关键词（用于自动打标签）
    COMMON_TAGS = [
        "RAG", "检索增强生成", "LLM", "大语言模型", "向量", "嵌入",
        "索引", "分块", "chunking", "PostgreSQL", "pgvector",
        "LlamaIndex", "混合检索", "稠密", "稀疏", "BM25",
        "MQE", "多查询扩展", "HyDE", "假设文档", "Parent-Child",
        "分层索引", "智能路由", "流式输出", "Docker",
        "Git", "Python", "测试", "性能优化",
    ]

    def extract(self, doc: Document) -> Dict[str, Any]:
        """从 Document 抽取结构化元数据。"""
        metadata = DocumentMetadata()

        if doc.metadata:
            metadata.file_name = doc.metadata.get("file_name")
            if metadata.file_name and "." in metadata.file_name:
                metadata.file_type = metadata.file_name.split(".")[-1]

        # 2. 从文档第一行提取章节标题
        lines = [line.strip() for line in doc.text.split("\n") if line.strip()]
        if lines:
            first_line = lines[0]
            # 去掉 markdown 标题符号 (# ## 等)
            clean_title = re.sub(r'^[#*]+\s*', '', first_line)
            if len(clean_title) > 3 and len(clean_title) < 100:
                metadata.section = clean_title

        text_lower = doc.text.lower()
        doc_type_scores: Dict[str, int] = {}
        for dtype, keywords in self.DOC_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            doc_type_scores[dtype] = score

        # 取分数最高的文档类型
        if any(doc_type_scores.values()):
            metadata.doc_type = max(doc_type_scores.items(), key=lambda x: x[1])[0]
        else:
            metadata.doc_type = "general"

        # 4. 自动打标签
        tags = []
        for tag in self.COMMON_TAGS:
            if tag.lower() in text_lower:
                tags.append(tag)
        # 从标题追加标签
        if metadata.section:
            section_words = re.findall(r'[a-zA-Z]+', metadata.section.lower())
            for word in section_words:
                if len(word) >= 4 and word in [t.lower() for t in self.COMMON_TAGS]:
                    original_tag = next(t for t in self.COMMON_TAGS if t.lower() == word)
                    if original_tag not in tags:
                        tags.append(original_tag)
        metadata.tags = tags

        # 5. 返回完整字典
        result = metadata.to_dict()

        # 合并到原有 metadata，不覆盖原有键
        if doc.metadata:
            for k, v in doc.metadata.items():
                if k not in result or result[k] is None:
                    result[k] = v

        return result


class StructuredFilter:
    """结构化过滤：使用 JSONB GIN 索引快速过滤节点。

    支持过滤条件：
      - 精确匹配: {"doc_type": "technical"} → metadata_ @> '{"doc_type": "technical"}'::jsonb
      - 标签包含: {"tags": ["RAG"]} → metadata_'tags' ? 'RAG'
      - 多个条件: AND 连接
    """

    def __init__(self, table_name: str = "llamaindex_rag"):
        self.table_name = table_name
        self._engine = None

    def _get_connection(self) -> Connection:
        """获取数据库连接（惰性初始化）。"""
        if self._engine is None:
            host = os.getenv("PG_HOST", "127.0.0.1")
            port = int(os.getenv("PG_PORT", "5433"))
            user = os.getenv("PG_USER", "postgres")
            password = os.getenv("PG_PASSWORD", "")
            database = os.getenv("PG_DATABASE", "practice_llm")
            url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            self._engine = create_engine(url)
        return self._engine.connect()

    def create_gin_index(self) -> bool:
        """在 metadata_ 列上创建 JSONB GIN 索引。

        Returns:
            True 创建成功，False 已存在或失败
        """
        conn = self._get_connection()
        actual_table = f"data_{self.table_name}"

        # 检查索引是否已存在
        check_sql = text("""
            SELECT 1 FROM pg_indexes
            WHERE tablename = :table AND indexname = :idx_name
        """)
        result = conn.execute(check_sql, {
            "table": actual_table,
            "idx_name": f"{actual_table}_metadata_gin"
        }).fetchone()

        if result:
            conn.close()
            return False  # 已存在

        # 创建 GIN 索引（json 列需 cast 为 jsonb）
        create_sql = text(f"""
            CREATE INDEX {actual_table}_metadata_gin
            ON {actual_table} USING GIN ((metadata_::jsonb) jsonb_path_ops)
        """)
        conn.execute(create_sql)
        conn.commit()
        conn.close()
        return True

    def _build_where_clause(self, filters: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """构建 WHERE 子句和参数。

        规则：
          - 数组值 → tags 包含任一标签 (metadata_'tags' ?| array[:key])
          - 标量值 → 精确匹配 (metadata_ @> :key::jsonb)
        """
        conditions = []
        params = {}
        idx = 0

        for key, value in filters.items():
            if isinstance(value, list):
                param_name = f"p{idx}"
                conditions.append(f"(metadata_::jsonb)->'{key}' ?| :{param_name}")
                params[param_name] = value
                idx += 1
            elif value is None:
                conditions.append(f"(metadata_::jsonb)->'{key}' IS NULL")
            else:
                param_name = f"p{idx}"
                conditions.append(f"metadata_::jsonb @> CAST(:{param_name} AS jsonb)")
                params[param_name] = json.dumps({key: value})
                idx += 1

        if not conditions:
            return "TRUE", params

        where = " AND ".join(conditions)
        return where, params

    def filter(self, filters: Dict[str, Any], limit: int = 200) -> List[str]:
        """根据过滤条件获取候选 node_id 列表。

        Args:
            filters: 过滤字典，例如 {"doc_type": "technical", "tags": ["RAG", "chunking"]}
            limit: 最大返回节点数，避免返回过多结果影响检索性能

        Returns:
            node_id 列表，可以用于后续在向量检索中过滤
        """
        if not filters:
            return []

        where, params = self._build_where_clause(filters)
        actual_table = f"data_{self.table_name}"

        sql = text(f"""
            SELECT node_id FROM {actual_table}
            WHERE {where}
            ORDER BY node_id
            LIMIT {limit}
        """)

        conn = self._get_connection()
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        return [row[0] for row in rows if row[0]]

    def count(self, filters: Dict[str, Any]) -> int:
        """统计符合过滤条件的节点数。"""
        where, params = self._build_where_clause(filters)
        actual_table = f"data_{self.table_name}"

        sql = text(f"SELECT COUNT(*) FROM {actual_table} WHERE {where}")
        conn = self._get_connection()
        result = conn.execute(sql, params).fetchone()
        conn.close()

        return result[0] if result else 0


class StructuredFilterRetriever:
    """先筛后查检索器：JSONB 结构化过滤缩小候选集，再执行混合检索。"""

    def __init__(
        self,
        table_name: str = "llamaindex_rag",
        embed_dim: int = 2560,
        similarity_top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        verbose: bool = False,
    ):
        self.table_name = table_name
        self.embed_dim = embed_dim
        self.similarity_top_k = similarity_top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.verbose = verbose

        # 内部分量
        self._structured_filter = StructuredFilter(table_name=table_name)
        self._index = None
        self._full_retriever = None

    def _lazy_init(self):
        """惰性初始化索引和完整检索器。"""
        if self._index is not None:
            return

        from src.tools.rag.llamaindex.retrievers import load_pg_index, _get_all_nodes_from_index
        from src.tools.rag.llamaindex.adapters import DashScopeEmbedding

        embed_model = DashScopeEmbedding()
        self._index = load_pg_index(
            table_name=self.table_name,
            embed_dim=self.embed_dim,
        )

        # 构建完整混合检索器（用于无过滤或全量检索）
        dense_retriever = self._index.as_retriever(
            similarity_top_k=self.similarity_top_k * 2
        )
        all_nodes = _get_all_nodes_from_index(self._index)
        if not all_nodes:
            self._full_retriever = QueryFusionRetriever(
                retrievers=[dense_retriever], retriever_weights=[1.0],
                similarity_top_k=self.similarity_top_k, num_queries=1,
                mode="reciprocal_rerank", verbose=self.verbose,
            )
        else:
            sparse_retriever = BM25Retriever.from_defaults(
                nodes=all_nodes,
                similarity_top_k=self.similarity_top_k * 2,
                language="chinese",
                skip_stemming=True,
            )
            self._full_retriever = QueryFusionRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                retriever_weights=[self.dense_weight, self.sparse_weight],
                similarity_top_k=self.similarity_top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                verbose=self.verbose,
            )

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """执行先筛后查检索。

        Args:
            query: 查询文本
            filters: 可选的结构化过滤条件。
                     如果为 None，则使用完整混合检索（无过滤）。

        Returns:
            检索结果列表 (NodeWithScore)
        """
        self._lazy_init()

        # 判断是否有过滤条件
        has_filter = filters and any(
            isinstance(v, list) and len(v) > 0 or not isinstance(v, list) and v is not None
            for v in filters.values()
        )

        if not has_filter:
            # 无过滤，走完整混合检索
            if self.verbose:
                print("  [StructuredFilterRetriever] 无过滤条件，走全量检索")
            return self._full_retriever.retrieve(query)

        # 有过滤条件：先获取候选节点
        if self.verbose:
            print(f"  [StructuredFilterRetriever] 过滤条件: {filters}")

        candidate_ids = self._structured_filter.filter(filters)
        if not candidate_ids:
            if self.verbose:
                print("  [StructuredFilterRetriever] 无匹配节点，回退到全量检索")
            return self._full_retriever.retrieve(query)

        if self.verbose:
            print(f"  [StructuredFilterRetriever] 候选节点: {len(candidate_ids)} 个")

        # 在候选节点集上做混合检索
        # 获取所有节点，过滤候选集
        from src.tools.rag.llamaindex.retrievers import _get_all_nodes_from_index
        all_nodes = _get_all_nodes_from_index(self._index)
        filtered_nodes = [n for n in all_nodes if n.node_id in candidate_ids]

        if not filtered_nodes:
            if self.verbose:
                print("  [StructuredFilterRetriever] 过滤后无节点，回退到全量检索")
            return self._full_retriever.retrieve(query)

        # 稠密检索：在候选节点上做向量检索
        from llama_index.core.retrievers import VectorIndexRetriever
        filtered_dense = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=min(self.similarity_top_k * 2, len(filtered_nodes)),
        )
        # 先用稠密检索获取候选，再按 node_id 过滤
        dense_results = filtered_dense.retrieve(query)
        dense_results = [r for r in dense_results if r.node.node_id in candidate_ids]

        # 稀疏检索：BM25 在过滤后的节点上
        sparse_filtered = BM25Retriever.from_defaults(
            nodes=filtered_nodes,
            similarity_top_k=min(self.similarity_top_k * 2, len(filtered_nodes)),
            language="chinese",
            skip_stemming=True,
        )

        # 融合检索
        fusion_retriever = QueryFusionRetriever(
            retrievers=[filtered_dense, sparse_filtered],
            retriever_weights=[self.dense_weight, self.sparse_weight],
            similarity_top_k=self.similarity_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            verbose=False,
        )

        return fusion_retriever.retrieve(query)

"""
查询解析器：从自然语言查询中提取结构化过滤条件。

策略：
  - 规则匹配：关键词 → doc_type / tags 映射
  - 返回 (filters, cleaned_query)

用法：
    from src.tools.rag.llamaindex.query_parser import QueryParser
    parser = QueryParser()
    filters, clean_q = parser.parse("RAG 技术文档中关于分块的策略")
    # filters = {"doc_type": "technical", "tags": ["chunking"]}
    # clean_q = "关于分块的策略"
"""

import re
from typing import Dict, List, Tuple, Optional, Any


class QueryParser:
    """从自然语言查询中提取结构化过滤条件。"""

    # doc_type 关键词映射
    DOC_TYPE_PATTERNS: Dict[str, List[str]] = {
        "technical": [
            "技术", "技术文档", "技术方案", "架构", "设计", "实现细节",
            "分析", "对比", "技术对比", "性能", "优化",
        ],
        "tutorial": [
            "教程", "步骤", "如何", "怎么", "方法", "手把手",
            "实践", "示例", "例子", "演示", "练习", "实现",
        ],
        "concept": [
            "概念", "什么是", "定义", "介绍", "解释", "概述",
            "简介", "原理", "说明", "基础", "入门", "指南",
        ],
        "reference": [
            "参考", "手册", "API", "文档", "参数", "配置",
            "命令", "用法",
        ],
    }

    TAG_PATTERNS: Dict[str, List[str]] = {
        "RAG": ["rag", "检索增强生成", "检索增强"],
        "LLM": ["llm", "大语言模型", "大模型", "语言模型"],
        "chunking": ["分块", "chunking", "分词", "文本分割", "节点切分"],
        "向量": ["向量", "嵌入", "embedding", "向量检索", "相似度"],
        "索引": ["索引", "index", "索引构建", "建索引"],
        "PostgreSQL": ["postgresql", "pgvector", "postgres", "pg"],
        "LlamaIndex": ["llamaindex", "llama index"],
        "混合检索": ["混合检索", "hybrid", "稠密检索", "稀疏检索", "bm25"],
        "MQE": ["mqe", "多查询扩展", "查询扩展"],
        "HyDE": ["hyde", "假设文档", "假设文档嵌入"],
        "Parent-Child": ["parent-child", "分层索引", "父子节点", "层次索引"],
        "智能路由": ["智能路由", "路由", "快速通道", "查询分类"],
        "Docker": ["docker", "容器化", "容器"],
        "流式": ["流式", "streaming", "流式输出", "sse"],
    }

    def parse(self, query: str) -> Tuple[Dict[str, Any], str]:
        """解析查询，返回 (filters, cleaned_query)。

        filters 示例:
            {"doc_type": "technical", "tags": ["RAG", "chunking"]}

        cleaned_query: 去掉 doc_type 关键词后的查询文本
        """
        filters: Dict[str, Any] = {}
        query_lower = query.lower()

        # 1. 提取 doc_type
        doc_type = self._extract_doc_type(query_lower)
        if doc_type:
            filters["doc_type"] = doc_type

        # 2. 提取标签
        tags = self._extract_tags(query_lower)
        if tags:
            filters["tags"] = tags

        # 3. 清理查询：去掉 doc_type 关键词（保留标签关键词，因为可能也是语义内容）
        cleaned = self._clean_query(query, doc_type)

        return filters, cleaned

    def _extract_doc_type(self, query_lower: str) -> Optional[str]:
        scores: Dict[str, int] = {}
        for dtype, patterns in self.DOC_TYPE_PATTERNS.items():
            score = sum(1 for p in patterns if p in query_lower)
            if score > 0:
                scores[dtype] = score

        if not scores:
            return None

        return max(scores.items(), key=lambda x: x[1])[0]

    def _extract_tags(self, query_lower: str) -> List[str]:
        """从查询中提取标签。"""
        matched_tags = []
        for tag, patterns in self.TAG_PATTERNS.items():
            if any(p in query_lower for p in patterns):
                matched_tags.append(tag)
        return matched_tags

    def _clean_query(self, query: str, doc_type: Optional[str]) -> str:
        """清理查询：去掉 doc_type 关键词，保留语义核心。"""
        if doc_type and doc_type in self.DOC_TYPE_PATTERNS:
            patterns = self.DOC_TYPE_PATTERNS[doc_type]
            cleaned = query
            for p in patterns:
                cleaned = cleaned.replace(p, "")
            # 清理多余空格
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            # 如果清理后为空，返回原查询
            if not cleaned:
                return query
            return cleaned

        return query

    def format_filters_display(self, filters: Dict[str, Any]) -> str:
        """格式化过滤器用于显示。"""
        parts = []
        if "doc_type" in filters:
            parts.append(f"doc_type={filters['doc_type']}")
        if "tags" in filters and filters["tags"]:
            parts.append(f"tags={filters['tags']}")
        if not parts:
            return "无过滤条件"
        return " | ".join(parts)
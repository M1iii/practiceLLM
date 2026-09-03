"""
LlamaIndexRAGTool: 将 LlamaIndex RAG 封装为 Agent 框架可用的工具。

集成了:
  - PostgreSQL + PGVectorStore 持久化存储
  - 混合检索：稠密向量（PGVector）+ 稀疏 BM25
  - MQE 多查询扩展 + HyDE 假设文档嵌入
  - ResponseSynthesizer 问答合成

用法：
    from src.tools.rag.llamaindex_tool import LlamaIndexRAGTool

    tool = LlamaIndexRAGTool()
    # 工具接口遵循框架抽象基类 Tool 规范
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from src.tools.framework.tool_system import Tool, ToolParameter
from src.tools.framework.tool_system import dual_protocol_execute

from src.tools.rag.llamaindex.adapters import DashScopeEmbedding, QwenLLM
from src.tools.rag.llamaindex.retrievers import load_pg_index, build_hybrid_retriever, _get_all_nodes_from_index
from src.tools.rag.llamaindex.transforms import MultiQueryExpander, HyDEExpander, QueryTransformPipeline, QueryClassifier
from src.tools.rag.llamaindex.structured_index import StructuredFilterRetriever
from src.tools.rag.llamaindex.query_parser import QueryParser
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core import Settings

logger = logging.getLogger(__name__)


class LlamaIndexRAGTool(Tool):
    """LlamaIndex RAG 工具（基于 PostgreSQL + pgvector 持久化存储）。

    支持:
      - 结构化索引 + 先筛后查：从自然语言查询自动提取过滤条件，先缩小范围再检索
      - 混合检索：稠密向量（PGVector）+ 稀疏 BM25
      - MQE 多查询扩展 + HyDE 假设文档嵌入
      - 智能路由：简单问题走快速通道，复杂问题走全量通道

    支持 action:
      - query: 直接问答（question → 回答）
      - search: 仅检索不生成回答
      - rebuild: 重建索引（删除旧表后重新构建 docs/ 目录文档）
      - list: 列出索引中的文档
      - stats: 显示统计信息
    """

    # 问答提示模板（参考原始 RagTool 对齐）
    QA_PROMPT = """你是一个知识库问答助手。请严格基于给定的资料片段回答用户问题。

【资料片段】
{context}

【问题】
{question}

【回答要求】
1. 只依据资料片段回答，资料中没有的信息请明确说明「资料中未提及」
2. 回答末尾标注引用的资料编号，如 [1][2]
3. 使用简洁清晰的中文回答"""

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self._table_name = config.get("table_name", "llamaindex_rag")
        self._embed_dim = config.get("embed_dim", 2560)
        self._top_k = config.get("top_k", 5)
        self._dense_weight = config.get("dense_weight", 0.7)
        self._sparse_weight = config.get("sparse_weight", 0.3)
        self._enable_mqe = config.get("enable_mqe", True)
        self._enable_hyde = config.get("enable_hyde", True)
        self._verbose = config.get("verbose", False)
        # 智能路由：简单问题走快速通道（跳过 MQE/HyDE），复杂问题走全量通道
        self._enable_auto_route = config.get("enable_auto_route", True)
        # 可单独为 MQE/HyDE 指定轻量模型，降低中间 LLM 调用的成本
        # 默认使用 qwen-turbo（比 qwen-plus 快 2 倍，成本更低）
        self._mqe_llm_model = config.get("mqe_llm_model", "qwen-turbo")
        self._hyde_llm_model = config.get("hyde_llm_model", "qwen-turbo")

        # 惰性初始化
        self._index = None
        self._retriever = None
        self._synthesizer = None
        self._mqe = None
        self._hyde = None
        self._classifier = None

    @property
    def name(self) -> str:
        return "LlamaIndexRAGTool"

    @property
    def description(self) -> str:
        return (
            "基于 LlamaIndex 框架的 RAG 检索增强生成工具，使用 PostgreSQL + pgvector "
            "持久化存储，支持结构化索引（先筛后查）、混合检索（稠密向量+稀疏 BM25）、"
            "MQE 多查询扩展、HyDE 假设文档嵌入，"
            "支持 docs/ 目录文档索引重建、知识库问答、语义检索。"
            "适用：基于 docs/ 目录文档的大规模知识库问答、混合检索召回、持久化索引存储。"
            "不适用：动态添加单个文档（当前只支持离线全量重建）、实时网络搜索（请用 AdvancedSearch）。"
            "注意：需要配置 DASHSCOPE_API_KEY，需要先运行离线构建脚本: "
            "python -m rag.llamaindex.build_pg_index --reset"
        )

    def get_parameters(self) -> List[ToolParameter]:
        """声明工具参数。"""
        from src.tools.framework.tool_system import ToolParameter
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型",
                enum=["query", "search", "rebuild", "list", "stats"],
                required=True,
            ),
            ToolParameter(
                name="question",
                type="string",
                description="查询问题（action=query/search 时必填）",
                required=False,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="返回结果数量（默认 5）",
                required=False,
                default=5,
            ),
        ]

    def execute(self, action, **kwargs):
        """兼容新旧协议执行入口。"""
        return dual_protocol_execute(self, action, **kwargs)

    def run(self, args: dict) -> str:
        """核心执行逻辑。"""
        action = args.get("action", "")
        question = args.get("question", "")
        top_k = args.get("top_k", self._top_k)

        try:
            if action == "query":
                if not question:
                    return "错误：action=query 需要 question 参数"
                return self._do_query(question, top_k)
            elif action == "search":
                if not question:
                    return "错误：action=search 需要 question 参数"
                return self._do_search(question, top_k)
            elif action == "rebuild":
                return self._do_rebuild()
            elif action == "list":
                return self._do_list()
            elif action == "stats":
                return self._do_stats()
            else:
                return f"错误：未知 action '{action}'，支持: query, search, rebuild, list, stats"
        except Exception as e:
            logger.error(f"LlamaIndexRAGTool 执行异常: {e}", exc_info=True)
            return f"LlamaIndexRAGTool 执行失败: {type(e).__name__}: {e}"

    def _lazy_init(self):
        """惰性初始化组件（第一次调用时执行）。"""
        if self._index is not None:
            return

        logger.info("LlamaIndexRAGTool 惰性初始化中...")

        # 设置全局 Settings（使用主 LLM）
        embed_model = DashScopeEmbedding()
        main_llm = QwenLLM()
        Settings.llm = main_llm
        Settings.embed_model = embed_model

        # 查询变换使用轻量级 LLM（默认 qwen-turbo），降低中间调用成本
        mqe_llm = QwenLLM(model=self._mqe_llm_model)
        hyde_llm = QwenLLM(model=self._hyde_llm_model)

        # 加载索引
        self._index = load_pg_index(
            table_name=self._table_name,
            embed_dim=self._embed_dim,
        )

        # 构建混合检索器
        self._retriever = build_hybrid_retriever(
            similarity_top_k=self._top_k,
            dense_weight=self._dense_weight,
            sparse_weight=self._sparse_weight,
            table_name=self._table_name,
            embed_dim=self._embed_dim,
            verbose=self._verbose,
        )

        # 查询变换（使用轻量 LLM）
        self._mqe = MultiQueryExpander(llm=mqe_llm, num_queries=3) if self._enable_mqe else None
        self._hyde = HyDEExpander(llm=hyde_llm) if self._enable_hyde else None

        # 智能路由分类器
        self._classifier = QueryClassifier(verbose=self._verbose) if self._enable_auto_route else None

        # 响应合成器（使用主 LLM）
        self._synthesizer = get_response_synthesizer(
            llm=main_llm,
            response_mode=ResponseMode.COMPACT,
            verbose=self._verbose,
        )

        logger.info("LlamaIndexRAGTool 初始化完成")

    def _do_query(self, question: str, top_k: int) -> str:
        """问答：检索 + 合成回答。"""
        self._lazy_init()

        # 检索
        results = self._retrieve(question, top_k)
        if not results:
            return "未检索到相关内容。请确认索引已构建: python -m rag.llamaindex.build_pg_index --reset"

        # 合成回答
        response = self._synthesizer.synthesize(question, results)

        # 格式化输出
        output = f"{str(response)}\n\n---\n引用来源 ({len(results)}):\n"
        for i, r in enumerate(results):
            src = r.node.metadata.get("file_name", "未知")
            preview = r.node.text[:80].replace("\n", " ")
            score = r.score
            output += f"[{i+1}] {src} (score={score:.4f}): {preview}...\n"

        return output.strip()

    def _do_search(self, question: str, top_k: int) -> str:
        """仅检索不生成回答。"""
        self._lazy_init()

        results = self._retrieve(question, top_k)
        if not results:
            return "未检索到相关内容。"

        output = f"检索结果 ({len(results)} 条):\n"
        for i, r in enumerate(results):
            src = r.node.metadata.get("file_name", "未知")
            score = r.score
            text = r.node.text[:120].replace("\n", " ")
            output += f"[{i+1}] score={score:.4f} | {src} | {text}...\n"

        return output.strip()

    def _retrieve(self, question: str, top_k: int):
        """统一检索入口（含查询解析 + 先筛后查 + 智能路由 + 查询变换 + 去重）。

        检索链路：
          1. 解析查询 → 提取结构化过滤条件 (filters, clean_q)
          2. 有过滤条件 → 先筛后查（StructuredFilterRetriever）
          3. 无过滤条件 → 智能路由 + MQE/HyDE/纯混合检索（原有逻辑）
        """
        self._lazy_init()

        # Step 1: 解析查询提取结构化过滤条件
        parser = QueryParser()
        filters, clean_q = parser.parse(question)
        has_filters = bool(filters)

        if has_filters:
            # Step 2: 先筛后查 — 结构化过滤缩小候选集，再执行混合检索
            if self._verbose:
                print(f"  [LlamaIndexRAGTool] 结构化过滤条件: {parser.format_filters_display(filters)}")
                print(f"  [LlamaIndexRAGTool] 清理后查询: {clean_q}")

            sfr = StructuredFilterRetriever(
                table_name=self._table_name,
                embed_dim=self._embed_dim,
                similarity_top_k=top_k,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
                verbose=self._verbose,
            )
            results = sfr.retrieve(clean_q, filters=filters)
        else:
            # Step 3: 无过滤条件，走原有逻辑（智能路由 + MQE/HyDE）
            use_fast = False
            if self._classifier:
                route = self._classifier.classify(question)
                use_fast = (route == "fast")
            else:
                use_fast = not self._mqe and not self._hyde

            if use_fast:
                results = self._retriever.retrieve(question)
            elif self._mqe:
                results = self._mqe.retrieve_multi(question, self._retriever, top_k=top_k * 2)
            elif self._hyde:
                results = self._hyde.retrieve_with_hypothetical(question, self._retriever, top_k=top_k * 2)
            else:
                results = self._retriever.retrieve(question)

        # 去重：按 node_id 去重，保留 score 最高的
        seen = {}
        for r in results:
            nid = r.node.node_id
            if nid not in seen or r.score > seen[nid].score:
                seen[nid] = r
        deduped = sorted(seen.values(), key=lambda x: x.score or 0, reverse=True)
        return deduped[:top_k]

    def _do_rebuild(self) -> str:
        """重建索引（提示用户运行离线脚本）。"""
        cmd = (
            "需要运行离线构建脚本完成全量重建:\n\n"
            "  python -m rag.llamaindex.build_pg_index --reset\n\n"
            "注意:\n"
            "  1. 该操作会删除旧表并重建，需要几分钟时间（取决于文档数量）\n"
            "  2. 构建完成后，重新调用工具即可使用新索引"
        )
        return cmd

    def _do_list(self) -> str:
        """列出索引中的文档。"""
        self._lazy_init()

        nodes = _get_all_nodes_from_index(self._index)
        if not nodes:
            return "索引为空，请先执行 rebuild。"

        docs = {}
        for node in nodes:
            fname = node.metadata.get("file_name", "unknown")
            if fname in docs:
                docs[fname] += 1
            else:
                docs[fname] = 1

        output = f"索引中共有 {len(nodes)} 个节点，来自 {len(docs)} 个文档:\n"
        for fname, count in sorted(docs.items()):
            output += f"  {fname}: {count} 个节点\n"

        return output.strip()

    def _do_stats(self) -> str:
        """显示统计信息。"""
        self._lazy_init()

        nodes = _get_all_nodes_from_index(self._index)
        total_nodes = len(nodes)

        docs = {}
        for node in nodes:
            fname = node.metadata.get("file_name", "unknown")
            docs[fname] = docs.get(fname, 0) + 1

        return (
            f"LlamaIndexRAGTool 统计:\n"
            f"  表名: {self._table_name}\n"
            f"  嵌入维度: {self._embed_dim}\n"
            f"  默认返回 top-k: {self._top_k}\n"
            f"  MQE 多查询扩展: {'启用' if self._enable_mqe else '禁用'}\n"
            f"  HyDE 假设文档: {'启用' if self._enable_hyde else '禁用'}\n"
            f"  智能路由: {'启用' if self._enable_auto_route else '禁用'}\n"
            f"  混合权重: 稠密 {self._dense_weight} + 稀疏 {self._sparse_weight}\n"
            f"  总节点数: {total_nodes}\n"
            f"  文档数: {len(docs)}\n"
        ).strip()

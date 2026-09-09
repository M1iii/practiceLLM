"""RAG 工具主类：基于阿里云百炼（DashScope）的检索增强生成工具。"""

import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src.tools.framework.tool_system import Tool, ToolParameter, dual_protocol_execute
from src.tools.rag.rag_tool.models import RagChunk, RagDocument
from src.tools.rag.rag_tool.loader import DocumentLoader
from src.tools.rag.rag_tool.splitter import TextSplitter, chunk_paragraphs, estimate_tokens, _approx_token_len
from src.tools.rag.rag_tool.index import RagIndex, QwenChatClient, QueryExpander, HydeGenerator
from src.tools.rag.rag_tool.encoder import VectorEncoder, index_chunks
from src.core.cache import SafeFullCache


class RagTool(Tool):
    """RAG 工具：检索增强生成。"""

    RAG_QA_PROMPT = """你是一个知识库问答助手。请严格基于给定的资料片段回答用户问题。

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
        self._loader = DocumentLoader()
        self._splitter = TextSplitter(
            chunk_size=config.get("chunk_size", 500),
            overlap=config.get("overlap", 50),
            chunk_tokens=config.get("chunk_tokens"),
            overlap_tokens=config.get("overlap_tokens"),
        )
        self._index = RagIndex()
        self._llm = None
        self._cache = SafeFullCache(
            max_size=config.get("cache_max_size", 10000),
            default_ttl=config.get("cache_ttl", 3600),
        )
        self._expander = QueryExpander(
            llm_client=None,
            threshold=config.get("mqe_threshold", 5),
            batch_size=config.get("mqe_batch_size", 10),
            cache=self._cache,
        )
        self._hyde = HydeGenerator(llm_client=None, cache=self._cache)

    @property
    def name(self) -> str:
        return "RagTool"

    @property
    def description(self) -> str:
        return (
            "RAG 检索增强生成工具，支持添加多格式文档（PDF/Office/图片/音频/文本）、"
            "智能检索召回、知识库问答（Qwen）。"
            "适用：基于私有文档的知识库问答、多格式文档索引与混合检索。"
            "不适用：实时网络搜索（请用 AdvancedSearch）、短期记忆（请用 MemoryTool）、"
            "小规模结构化笔记（请用 NoteTool）。"
            "注意：问答功能需要配置 DASHSCOPE_API_KEY。"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return self._build_parameters()

    def execute(self, action, **kwargs):
        return dual_protocol_execute(self, action, **kwargs)

    def _get_llm(self) -> QwenChatClient:
        if self._llm is None:
            try:
                self._llm = QwenChatClient()
            except ValueError:
                self._llm = None
        return self._llm

    @staticmethod
    def _build_parameters() -> List[ToolParameter]:
        return [
            ToolParameter(name="file_path", type="string",
                          description="文件路径（add_file 时使用）", required=False),
            ToolParameter(name="content", type="string",
                          description="文本内容（add_text 时使用）", required=False),
            ToolParameter(name="title", type="string",
                          description="文档标题（add_text 时使用）", required=False),
            ToolParameter(name="description", type="string",
                          description="文件描述（图片/音频等元数据语义化时使用）", required=False),
            ToolParameter(name="query", type="string",
                          description="检索查询（search / search_mqe 时使用）", required=False),
            ToolParameter(name="question", type="string",
                          description="问题（query 时使用）", required=False),
            ToolParameter(name="top_k", type="number",
                          description="召回数量（默认 5）", required=False, default=5),
            ToolParameter(name="min_score", type="number",
                          description="最低相关分（默认 0.0）", required=False, default=0.0),
            ToolParameter(name="show_context", type="boolean",
                          description="是否展示召回上下文（query 时，默认 false）", required=False, default=False),
            ToolParameter(name="mqe", type="boolean",
                          description="是否启用多查询扩展（query 时，默认 false，旧接口）", required=False, default=False),
            ToolParameter(name="n_queries", type="number",
                          description="扩展查询数量（search_mqe / mqe 时使用，默认 3）", required=False, default=3),
            ToolParameter(name="hyde", type="boolean",
                          description="是否启用假设文档嵌入检索（search_hyde / query 时，默认 false，旧接口）",
                          required=False, default=False),
            ToolParameter(name="enable_mqe", type="boolean",
                          description="统一框架：启用 MQE 多查询扩展（默认 false）", required=False, default=False),
            ToolParameter(name="enable_hyde", type="boolean",
                          description="统一框架：启用 HyDE 假设文档嵌入（默认 false）", required=False, default=False),
            ToolParameter(name="mqe_expansions", type="number",
                          description="MQE 扩展查询数量（默认 2）", required=False, default=2),
            ToolParameter(name="candidate_pool_multiplier", type="number",
                          description="候选池倍数：pool = max(top_k × 倍数, 20)（默认 4）", required=False, default=4),
            ToolParameter(name="doc_id", type="string",
                          description="文档 ID（delete 时使用）", required=False),
            ToolParameter(name="doc_type", type="string",
                          description="按类型过滤（list 时使用）", required=False),
        ]

    def run(self, args: Dict[str, Any]) -> str:
        action = (args.get("action") or "").lower()
        if action == "add_file":
            return self._add_file(args)
        elif action == "add_text":
            return self._add_text(args)
        elif action == "convert":
            return self._convert(args)
        elif action == "search":
            return self._search(args)
        elif action == "search_mqe":
            return self._search_mqe(args)
        elif action == "search_hyde":
            return self._search_hyde(args)
        elif action == "search_expanded":
            return self._search_expanded(args)
        elif action == "query":
            return self._query(args)
        elif action == "list":
            return self._list(args)
        elif action == "stats":
            return self._stats(args)
        elif action == "delete":
            return self._delete(args)
        elif action == "clear":
            return self._clear(args)
        return f"未知操作: '{action}'，当前支持: add_file, add_text, convert, search, search_mqe, search_hyde, search_expanded, query, list, stats, delete, clear"

    def _add_file(self, args: Dict[str, Any]) -> str:
        file_path = args.get("file_path", "").strip()
        if not file_path:
            return "add_file 操作需要提供 file_path 参数"
        description = (args.get("description") or "").strip()
        try:
            doc_meta, blocks = self._loader.load(file_path, description)
        except FileNotFoundError as e:
            return f"文件不存在: {e}"
        except Exception as e:
            return f"文档加载失败: {e}"
        return self._index_blocks(doc_meta, blocks)

    def _add_text(self, args: Dict[str, Any]) -> str:
        content = (args.get("content") or "").strip()
        if not content:
            return "add_text 操作需要提供 content 参数"
        title = (args.get("title") or "").strip() or "文本片段"
        doc_meta = {
            "name": title, "source_path": None, "doc_type": "text",
            "extension": "", "size": len(content), "description": "",
        }
        blocks = [{"content": content, "metadata": {"section": "全文"}}]
        return self._index_blocks(doc_meta, blocks)

    def _index_blocks(self, doc_meta: Dict[str, Any], blocks: List[Dict[str, Any]]) -> str:
        if not blocks:
            return f"文档 {doc_meta['name']} 未提取到任何文本内容"
        doc_id = str(uuid.uuid4())
        doc = RagDocument(
            doc_id=doc_id, name=doc_meta["name"],
            source_path=doc_meta.get("source_path"),
            doc_type=doc_meta.get("doc_type", "text"),
            created_at=datetime.now().isoformat(),
            metadata={
                "extension": doc_meta.get("extension", ""),
                "size": doc_meta.get("size", 0),
                "description": doc_meta.get("description", ""),
                "converter": doc_meta.get("converter", "fallback"),
            },
        )
        embedded_count = 0
        chunks: List[RagChunk] = []
        if self._splitter.use_tokens:
            paragraph_dicts: List[Dict] = []
            for block in blocks:
                for para in re.split(r'\n\s*\n', block["content"]):
                    para = para.strip()
                    if para:
                        paragraph_dicts.append({
                            "content": para,
                            "metadata": block.get("metadata", {}),
                        })
            chunk_dicts = self._splitter.split_paragraph_dicts(paragraph_dicts)
            for cd in chunk_dicts:
                chunks.append(RagChunk(
                    chunk_id=str(uuid.uuid4()), doc_id=doc_id, doc_name=doc_meta["name"],
                    content=cd["content"],
                    metadata={**cd.get("metadata", {}), "tokens": cd.get("tokens", 0),
                              "paragraph_count": cd.get("paragraph_count", 1)},
                ))
        else:
            for block in blocks:
                for piece in self._splitter.split(block["content"]):
                    chunks.append(RagChunk(
                        chunk_id=str(uuid.uuid4()), doc_id=doc_id, doc_name=doc_meta["name"],
                        content=piece, metadata=dict(block.get("metadata", {})),
                    ))
        chunks, embedded_count = self._index.index_chunks(chunks)
        doc.chunk_count = len(chunks)
        self._index.add_document(doc, chunks)
        self._index.rebuild_idf()
        self._cache.clear()
        embed_status = (f"嵌入向量: {embedded_count}/{len(chunks)} 块"
                        if self._index.embedding_available
                        else "嵌入 API 不可用，使用 TF-IDF 检索")
        if self._splitter.use_tokens:
            chunk_mode = (f"token 分块（chunk_tokens={self._splitter.chunk_tokens}, "
                          f"overlap_tokens={self._splitter.overlap_tokens}）")
        else:
            chunk_mode = (f"字符分块（chunk_size={self._splitter.chunk_size}, "
                          f"overlap={self._splitter.overlap}）")
        return (
            f"文档已添加\n"
            f"   ID: {doc_id[:12]}...\n"
            f"   名称: {doc_meta['name']}\n"
            f"   类型: {doc_meta.get('doc_type', 'text')}\n"
            f"   分块数: {len(chunks)}（{chunk_mode}）\n"
            f"   {embed_status}"
        )

    def _convert(self, args: Dict[str, Any]) -> str:
        file_path = args.get("file_path", "").strip()
        if not file_path:
            return "convert 操作需要提供 file_path 参数"
        if not os.path.exists(file_path):
            return f"文件不存在: {file_path}"
        description = (args.get("description") or "").strip()
        try:
            doc_meta, blocks = self._loader.load(file_path, description)
        except Exception as e:
            return f"转换失败: {e}"
        converter_name = doc_meta.get("converter", "fallback")
        converter_label = "MarkItDown 统一转换" if converter_name == "markitdown" else "轻量回退解析"
        lines = [
            "=" * 60,
            f"文档转换预览: {doc_meta['name']}",
            f"   类型: {doc_meta['doc_type']} | 转换引擎: {converter_label}",
            f"   分块数: {len(blocks)}（后续将进入统一分块 → 向量化 → 入库流程）",
            "=" * 60,
        ]
        for i, block in enumerate(blocks, 1):
            lines.append("")
            lines.append(f"--- 块 {i} [{block['metadata'].get('section', '')}] ---")
            lines.append(block["content"][:300] + ("..." if len(block["content"]) > 300 else ""))
        return "\n".join(lines)

    def _search(self, args: Dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "search 操作需要提供 query 参数"
        top_k = max(1, min(50, int(args.get("top_k", 5))))
        min_score = max(0.0, min(1.0, float(args.get("min_score", 0.0))))
        results = self._index.search(query, top_k=top_k, min_score=min_score)
        if not results:
            return "没有检索到相关分块。"
        method_str = "嵌入向量 + TF-IDF 混合" if self._index.embedding_available else "TF-IDF"
        lines = [
            "=" * 60,
            f"RAG 检索: \"{query}\"",
            f"   检索方式: {method_str}",
            f"   召回结果: {len(results)}/{self._index.get_chunk_count()} 分块",
            "=" * 60,
        ]
        for i, (score, chunk, method) in enumerate(results, 1):
            lines.append("")
            lines.append(f"--- 结果 {i} [{method}] score={score:.4f} ---")
            lines.append(f"  来源: {chunk.doc_name}（{chunk.metadata.get('section', '')}）")
            lines.append(f"  内容: {chunk.content[:120]}{'...' if len(chunk.content) > 120 else ''}")
        return "\n".join(lines)

    def _retrieve_mqe(self, query: str, top_k: int, n_queries: int = 3) -> Tuple[List[tuple], Dict[str, Any]]:
        queries, mode, total = self._expander.plan(query, n=n_queries)
        merged: Dict[str, Tuple[float, RagChunk, str]] = {}
        match_counts: Dict[str, int] = {}
        batches_used = 0
        for batch in self._expander.iter_batches(queries):
            batches_used += 1
            for q in batch:
                for score, chunk, method in self._index.search(q, top_k=top_k):
                    if chunk.chunk_id not in merged or score > merged[chunk.chunk_id][0]:
                        merged[chunk.chunk_id] = (score, chunk, method)
                    match_counts[chunk.chunk_id] = match_counts.get(chunk.chunk_id, 0) + 1
        results = []
        for chunk_id, (score, chunk, method) in merged.items():
            matches = match_counts.get(chunk_id, 1)
            final_score = score * min(1.5, 1.0 + 0.15 * (matches - 1))
            results.append((final_score, chunk, method, matches))
        results.sort(key=lambda x: x[0], reverse=True)
        stats = {"queries": queries, "mode": mode, "total": total,
                 "batches": batches_used, "matched": len(results)}
        return results, stats

    def _search_mqe(self, args: Dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "search_mqe 操作需要提供 query 参数"
        top_k = max(1, min(50, int(args.get("top_k", 5))))
        n_queries = max(1, min(10, int(args.get("n_queries", 3))))
        results, stats = self._retrieve_mqe(query, top_k, n_queries)
        if not results:
            return "MQE 检索没有找到相关分块。"
        mode_label = "批量查询模式" if stats["mode"] == "batch" else "单查询模式"
        lines = [
            "=" * 60,
            f"MQE 多查询扩展检索: \"{query}\"",
            f"   扩展查询({stats['total']}个): {', '.join(stats['queries'])}",
            f"   模式: {mode_label}（threshold={self._expander.threshold}）"
            f" | 分批: {stats['batches']}批（batch_size={self._expander.batch_size}）",
            f"   召回结果: {stats['matched']}/{self._index.get_chunk_count()} 分块",
            "=" * 60,
        ]
        for i, (score, chunk, method, matches) in enumerate(results[:top_k], 1):
            lines.append("")
            lines.append(f"--- 结果 {i} [{method}] score={score:.4f} 命中{matches}查询 ---")
            lines.append(f"  来源: {chunk.doc_name}（{chunk.metadata.get('section', '')}）")
            lines.append(f"  内容: {chunk.content[:120]}{'...' if len(chunk.content) > 120 else ''}")
        return "\n".join(lines)

    def _retrieve_hyde(self, query: str, top_k: int) -> Tuple[List[tuple], Dict[str, Any]]:
        hypo_doc = self._hyde.generate(query)
        results = self._index.search(hypo_doc, top_k=top_k)
        stats = {"hypothesis": hypo_doc, "used_hypothesis": hypo_doc != query}
        return results, stats

    def _search_hyde(self, args: Dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "search_hyde 操作需要提供 query 参数"
        top_k = max(1, min(50, int(args.get("top_k", 5))))
        results, stats = self._retrieve_hyde(query, top_k)
        if not results:
            return "HyDE 检索没有找到相关分块。"
        hypo_note = ("假设文档" if stats["used_hypothesis"] else "原始查询（LLM 兜底）")
        lines = [
            "=" * 60,
            f"HyDE 假设文档嵌入检索: \"{query}\"",
            f"   假设文档（{hypo_note}）: {stats['hypothesis'][:100]}"
            + ("..." if len(stats['hypothesis']) > 100 else ""),
            f"   召回结果: {len(results)}/{self._index.get_chunk_count()} 分块",
            "=" * 60,
        ]
        for i, (score, chunk, method) in enumerate(results[:top_k], 1):
            lines.append("")
            lines.append(f"--- 结果 {i} [{method}] score={score:.4f} ---")
            lines.append(f"  来源: {chunk.doc_name}（{chunk.metadata.get('section', '')}）")
            lines.append(f"  内容: {chunk.content[:120]}{'...' if len(chunk.content) > 120 else ''}")
        return "\n".join(lines)

    def _retrieve_expanded(self, query: str, top_k: int = 8, score_threshold: float = None,
                           enable_mqe: bool = False, mqe_expansions: int = 2,
                           enable_hyde: bool = False, candidate_pool_multiplier: int = 4,
                           min_score: float = 0.0) -> Tuple[List[tuple], Dict[str, Any]]:
        if not query:
            return [], {}
        expansions: List[str] = [query]
        strategy_parts = ["base"]
        if enable_mqe and mqe_expansions > 0:
            mqe_queries = self._expander.expand(query, n=mqe_expansions)
            expansions.extend(mqe_queries[1:])
            strategy_parts.append("MQE")
        if enable_hyde:
            hyde_text = self._hyde.generate(query)
            if hyde_text and hyde_text != query:
                expansions.append(hyde_text)
                strategy_parts.append("HyDE")
        uniq: List[str] = []
        for e in expansions:
            e = (e or "").strip()
            if e and e not in uniq:
                uniq.append(e)
        expansions = uniq
        pool = max(top_k * max(1, candidate_pool_multiplier), 20)
        per = max(1, pool // max(1, len(expansions)))

        def _search_one(q: str):
            return self._index.search(q, top_k=per, min_score=min_score)

        agg: Dict[str, Tuple[float, RagChunk, str]] = {}
        workers = min(8, len(expansions))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_search_one, q) for q in expansions]
                for future in futures:
                    for score, chunk, method in future.result():
                        cid = chunk.chunk_id
                        if cid not in agg or score > agg[cid][0]:
                            agg[cid] = (score, chunk, method)
        else:
            for score, chunk, method in _search_one(expansions[0]):
                agg[chunk.chunk_id] = (score, chunk, method)
        merged = sorted(agg.values(), key=lambda x: x[0], reverse=True)
        results = merged[:top_k]
        stats = {"expansions": expansions, "strategy": "+".join(strategy_parts),
                 "pool": pool, "per": per, "total_candidates": len(agg), "matched": len(results)}
        return results, stats

    def _search_expanded(self, args: Dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "search_expanded 操作需要提供 query 参数"
        top_k = max(1, min(50, int(args.get("top_k", 8))))
        min_score = max(0.0, min(1.0, float(args.get("min_score", 0.0))))
        enable_mqe = bool(args.get("enable_mqe", False))
        enable_hyde = bool(args.get("enable_hyde", False))
        mqe_expansions = max(1, min(10, int(args.get("mqe_expansions", 2))))
        multiplier = max(1, min(20, int(args.get("candidate_pool_multiplier", 4))))
        results, stats = self._retrieve_expanded(
            query, top_k=top_k, score_threshold=None,
            enable_mqe=enable_mqe, mqe_expansions=mqe_expansions,
            enable_hyde=enable_hyde, candidate_pool_multiplier=multiplier, min_score=min_score,
        )
        if not results:
            return "扩展检索没有找到相关分块。"
        lines = [
            "=" * 60,
            f"统一扩展检索: \"{query}\"",
            f"   策略: {stats['strategy']} | 扩展查询({len(stats['expansions'])}个)",
            f"   候选池: {stats['pool']}（top_k×{multiplier}）| 每查询召回{stats['per']} | "
            f"合并候选{stats['total_candidates']} → 返回{stats['matched']}",
            "=" * 60,
        ]
        lines.append(f"   扩展查询: {', '.join(e[:40] for e in stats['expansions'])}")
        for i, (score, chunk, method) in enumerate(results[:top_k], 1):
            lines.append("")
            lines.append(f"--- 结果 {i} [{method}] score={score:.4f} ---")
            lines.append(f"  来源: {chunk.doc_name}（{chunk.metadata.get('section', '')}）")
            lines.append(f"  内容: {chunk.content[:120]}{'...' if len(chunk.content) > 120 else ''}")
        return "\n".join(lines)

    def _query(self, args: Dict[str, Any]) -> str:
        question = (args.get("question") or "").strip()
        if not question:
            return "query 操作需要提供 question 参数"
        top_k = max(1, min(20, int(args.get("top_k", 5))))
        show_context = bool(args.get("show_context", False))
        use_mqe = bool(args.get("mqe", False))
        use_hyde = bool(args.get("hyde", False))
        enable_mqe = bool(args.get("enable_mqe", False))
        enable_hyde = bool(args.get("enable_hyde", False))
        enhance_note = ""
        if enable_mqe or enable_hyde:
            mqe_expansions = max(1, min(10, int(args.get("mqe_expansions", 2))))
            multiplier = max(1, min(20, int(args.get("candidate_pool_multiplier", 4))))
            exp_results, stats = self._retrieve_expanded(
                question, top_k=top_k, enable_mqe=enable_mqe, mqe_expansions=mqe_expansions,
                enable_hyde=enable_hyde, candidate_pool_multiplier=multiplier,
            )
            results = exp_results
            enhance_note = (f" | 扩展检索[{stats['strategy']}]: "
                            f"{len(stats['expansions'])}查询/池{stats['pool']}")
        elif use_mqe:
            n_queries = max(1, min(10, int(args.get("n_queries", 3))))
            mqe_results, stats = self._retrieve_mqe(question, top_k, n_queries)
            results = [(score, chunk, method) for score, chunk, method, _ in mqe_results][:top_k]
            mode_label = "批量" if stats["mode"] == "batch" else "单查询"
            enhance_note = f" | MQE: {stats['total']}查询/{mode_label}/{stats['batches']}批"
        elif use_hyde:
            hyde_results, stats = self._retrieve_hyde(question, top_k)
            results = hyde_results
            hyde_note = "HyDE" if stats["used_hypothesis"] else "HyDE(LLM兜底)"
            enhance_note = f" | {hyde_note}"
        else:
            results = self._index.search(question, top_k=top_k)
        if not results:
            return "知识库中没有检索到相关资料，无法回答。"
        context_parts = []
        for i, (score, chunk, method) in enumerate(results, 1):
            context_parts.append(f"[{i}] (相关度:{score:.3f}) 来源:{chunk.doc_name}\n{chunk.content}")
        context = "\n\n".join(context_parts)
        qa_cache_key = f"qa:{question}:{top_k}:{enhance_note}:{show_context}"
        cached_output = self._cache.get(qa_cache_key)
        if cached_output is not None:
            return cached_output
        llm = self._get_llm()
        if llm is None:
            return ("未配置 DASHSCOPE_API_KEY，无法调用 Qwen 问答。\n"
                    f"以下为检索到的资料：\n\n{context}")
        messages = [
            {"role": "system", "content": "你是一个严谨的知识库问答助手。"},
            {"role": "user", "content": self.RAG_QA_PROMPT.format(context=context, question=question)},
        ]
        try:
            answer = llm.chat(messages, temperature=0.3)
        except Exception as e:
            return f"Qwen 调用失败: {e}"
        lines = [
            "=" * 60,
            f"问题: {question}",
            f"  模型: {llm.model} | 召回: {len(results)} 块{enhance_note}",
            "=" * 60,
            "回答:",
            answer,
        ]
        if show_context:
            lines.append("")
            lines.append("引用资料:")
            for i, (score, chunk, method) in enumerate(results, 1):
                lines.append(f"  [{i}] {chunk.doc_name} ({chunk.metadata.get('section', '')}) "
                             f"score={score:.3f}")
                lines.append(f"      {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")
        output = "\n".join(lines)
        self._cache.set(qa_cache_key, output)
        return output

    def _list(self, args: Dict[str, Any]) -> str:
        docs = self._index.get_docs()
        doc_type = args.get("doc_type")
        if doc_type:
            docs = [d for d in docs if d.doc_type == doc_type]
        if not docs:
            return "知识库为空。"
        lines = [
            "=" * 60,
            f"知识库文档列表（共 {len(docs)} 份）",
            "=" * 60,
        ]
        for d in docs:
            lines.append(f"  ID: {d.doc_id[:12]}... | {d.name} "
                         f"[{d.doc_type}] {d.chunk_count}块 {d.created_at[:10]}")
        return "\n".join(lines)

    def _stats(self, args: Dict[str, Any]) -> str:
        docs = self._index.get_docs()
        type_stats: Counter = Counter(d.doc_type for d in docs)
        embed_note = "可用" if self._index.embedding_available else "不可用（TF-IDF 模式）"
        cache_stats = self._cache.stats()
        return (
            "知识库统计\n"
            f"  文档数: {self._index.get_doc_count()}\n"
            f"  分块数: {self._index.get_chunk_count()}\n"
            f"  嵌入 API: {embed_note}\n"
            f"  类型分布: {dict(type_stats) if type_stats else '(空)'}\n"
            f"  缓存层: {cache_stats['size']}/{cache_stats['max_size']} 项 "
            f"(命中率 {cache_stats['hit_rate']:.1%}, 命中{cache_stats['hits']} "
            f"未命中{cache_stats['misses']})"
        )

    def _delete(self, args: Dict[str, Any]) -> str:
        doc_id = args.get("doc_id", "").strip()
        if not doc_id:
            return "delete 操作需要提供 doc_id 参数"
        docs = self._index.get_docs()
        matched = [d for d in docs if d.doc_id.startswith(doc_id)]
        if len(matched) != 1:
            return (f"文档 ID 匹配到 {len(matched)} 份，请提供更精确的 ID"
                    if matched else f"未找到文档: {doc_id}")
        doc = matched[0]
        self._index.delete_doc(doc.doc_id)
        self._cache.clear()
        return f"已删除文档: {doc.name}（{doc.chunk_count} 块）"

    def _clear(self, args: Dict[str, Any]) -> str:
        count = self._index.get_doc_count()
        self._index.clear()
        self._cache.clear()
        return f"已清空知识库（删除 {count} 份文档）"

    def add_file(self, file_path: str, description: str = None) -> str:
        return self.execute("add_file", file_path=file_path, description=description)

    def ask(self, question: str, top_k: int = 5) -> str:
        return self.execute("query", question=question, top_k=top_k)
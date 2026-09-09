"""记忆工具 (MemoryTool)：管理多类型记忆的存储与检索。"""

import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from src.tools.framework.tool_system import Tool, ToolParameter, dual_protocol_execute
from src.tools.memory.modules import (
    MemoryEntry, DEFAULT_IMPORTANCE, FILE_MODALITY_MAP,
    MemoryManager,
)
from src.tools.memory.memory_tool.reranker import LLMReranker


class MemoryTool(Tool):
    """管理多类型记忆的存储与检索。"""

    def __init__(self, session_id: str = None, config: Dict[str, Any] = None):
        self._session_id = session_id or str(uuid.uuid4())
        self._manager = MemoryManager(config)

    @property
    def name(self) -> str:
        return "MemoryTool"

    @property
    def description(self) -> str:
        return (
            "记忆管理工具，支持添加、搜索、遗忘、整合等操作。"
            "记忆类型: working(工作记忆), episodic(情景记忆), semantic(语义记忆), perceptual(感知记忆)。"
            "适用：跨会话短期信息存取、用户偏好与事实记忆、任务上下文保持。"
            "不适用：长文档与结构化笔记归档（请用 NoteTool）、文档知识库问答（请用 RagTool）。"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="action", type="string",
                          description="操作类型: add（添加）、search（搜索）、forget（遗忘）、consolidate（整合）",
                          required=True, enum=["add", "search", "forget", "consolidate"]),
            ToolParameter(name="content", type="string",
                          description="记忆内容文本（add 时必填）", required=False),
            ToolParameter(name="memory_type", type="string",
                          description="记忆类型（add 时使用，search 时作为单个类型过滤器）",
                          required=False, default="working",
                          enum=["working", "episodic", "semantic", "perceptual"]),
            ToolParameter(name="importance", type="number",
                          description="重要程度 0.0-1.0（add 时使用，不传则按记忆类型设置默认值）",
                          required=False, default=None),
            ToolParameter(name="file_path", type="string",
                          description="关联文件路径（用于多模态记忆）", required=False, default=None),
            ToolParameter(name="modality", type="string",
                          description="信息模态: text/image/audio/video/code，不传则根据 file_path 自动推断",
                          required=False, default=None),
            ToolParameter(name="session_id", type="string",
                          description="会话 ID，不传则使用当前默认会话", required=False, default=None),
            ToolParameter(name="query", type="string",
                          description="搜索查询文本（search 时必填）", required=False),
            ToolParameter(name="limit", type="number",
                          description="返回结果数量上限（search 时使用）", required=False, default=5),
            ToolParameter(name="memory_types", type="array",
                          description="按多个记忆类型过滤（search 时使用，与 memory_type 可组合）",
                          required=False, default=None,
                          items=ToolParameter(name="type", type="string", description="记忆类型",
                                              required=True,
                                              enum=["working", "episodic", "semantic", "perceptual"])),
            ToolParameter(name="min_importance", type="number",
                          description="最低重要度阈值（search 时使用）", required=False, default=0.1),
            ToolParameter(name="target_modality", type="string",
                          description="目标模态过滤（search perceptual 时使用）",
                          required=False, default=None,
                          enum=["text", "image", "audio", "video", "code", "document", "data"]),
            ToolParameter(name="strategy", type="string",
                          description="遗忘策略（forget 时使用）",
                          required=False, default="importance",
                          enum=["importance", "time", "capacity", "mixed", "dedup"]),
            ToolParameter(name="threshold", type="number",
                          description="阈值：importance 策略下限/dedup 相似度上限/capacity 容量上限",
                          required=False, default=0.3),
            ToolParameter(name="max_age_days", type="number",
                          description="最大保留天数（time/mixed 策略使用）", required=False, default=7),
            ToolParameter(name="from_type", type="string",
                          description="源记忆类型（consolidate 时使用，如 working→episodic）",
                          required=False, default=None,
                          enum=["working", "episodic", "semantic", "perceptual"]),
            ToolParameter(name="to_type", type="string",
                          description="目标记忆类型（consolidate 时使用）",
                          required=False, default=None,
                          enum=["working", "episodic", "semantic", "perceptual"]),
            ToolParameter(name="importance_threshold", type="number",
                          description="重要性阈值，仅整合 importance ≥ 此值的记忆（consolidate 时使用）",
                          required=False, default=0.7),
            ToolParameter(name="time_window_minutes", type="number",
                          description="时间窗口（分钟），将过去 N 分钟内的记忆打包整合并提醒用户",
                          required=False, default=None),
        ]

    def execute(self, action, **kwargs):
        return dual_protocol_execute(self, action, **kwargs)

    def run(self, args: Dict[str, Any]) -> str:
        action = args.get("action", "").lower()
        if action == "add":
            return self._add_memory(args)
        elif action == "search":
            return self._search_memory(args)
        elif action == "forget":
            return self._forget_memory(args)
        elif action == "consolidate":
            return self._consolidate_memory(args)
        elif action == "rerank":
            return self._rerank_memory(args)
        return f"未知操作: '{action}'，当前支持: add, search, forget, consolidate, rerank"

    # ----------------------------------------------------------
    # 添加记忆
    # ----------------------------------------------------------

    def _add_memory(self, args: Dict[str, Any]) -> str:
        content = args.get("content", "").strip()
        if not content:
            return "content 不能为空"

        memory_type = args.get("memory_type", "working").lower()
        if memory_type not in DEFAULT_IMPORTANCE:
            return f"无效的 memory_type: '{memory_type}'，可选: {list(DEFAULT_IMPORTANCE.keys())}"

        if not self._manager.is_enabled(memory_type):
            return (f"记忆类型 '{memory_type}' 未启用，"
                    f"当前已启用: {self._manager.get_enabled_types()}")

        importance = args.get("importance")
        if importance is None:
            importance = DEFAULT_IMPORTANCE[memory_type]
        importance = max(0.0, min(1.0, float(importance)))

        session_id = args.get("session_id") or self._session_id

        file_path = args.get("file_path")
        modality = args.get("modality")

        if file_path and not modality:
            modality = self._detect_modality(file_path)

        if not modality:
            modality = "text"

        known_keys = {"action", "content", "memory_type", "importance",
                      "file_path", "modality", "session_id"}
        extra_metadata = {k: v for k, v in args.items() if k not in known_keys}

        if file_path:
            extra_metadata["file_exists"] = os.path.exists(file_path)
            extra_metadata["file_extension"] = os.path.splitext(file_path)[1].lower()

        entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            importance=importance,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            file_path=file_path,
            modality=modality,
            metadata=extra_metadata,
        )

        add_msg = self._manager.add(entry)

        return (
            f"记忆已添加\n"
            f"   ID: {entry.memory_id[:12]}...\n"
            f"   类型: {entry.memory_type}\n"
            f"   重要度: {entry.importance:.1f}\n"
            f"   模态: {entry.modality}\n"
            f"   会话: {entry.session_id[:12]}...\n"
            f"   时间: {entry.timestamp}\n"
            f"   总记忆数: {self._manager.get_count()}"
            + (f"\n   {add_msg}" if add_msg else "")
        )

    # ----------------------------------------------------------
    # 搜索记忆
    # ----------------------------------------------------------

    def _search_memory(self, args: Dict[str, Any]) -> str:
        params = self._normalize_search_params(args)
        if params["error"]:
            return params["error"]

        query = params["query"]
        limit = params["limit"]
        memory_types = params["memory_types"]
        min_importance = params["min_importance"]

        if self._manager.get_count() == 0:
            return "记忆库为空，无法搜索。"

        search_types = memory_types or set(self._manager.get_enabled_types())

        scored = []
        retrieval_methods = []

        if "working" in search_types and self._manager.is_enabled("working"):
            working_module = self._manager.get_module("working")
            if working_module and working_module.get_count() > 0:
                retrieval_methods.append("working(混合检索)")
                results = working_module.retrieve(query, limit=limit)
                for final_score, base_relevance, mem, details in results:
                    if mem.importance >= min_importance:
                        scored.append((mem, base_relevance, final_score,
                                       details.get("method", "hybrid"), details))

        if "episodic" in search_types and self._manager.is_enabled("episodic"):
            episodic_module = self._manager.get_module("episodic")
            if episodic_module and episodic_module.get_count() > 0:
                retrieval_methods.append("episodic(结构化+向量)")
                retrieve_kwargs = {"min_importance": min_importance}
                session_id = args.get("session_id")
                if session_id:
                    retrieve_kwargs["session_id"] = session_id
                results = episodic_module.retrieve(query, limit=limit, **retrieve_kwargs)
                for final_score, base_relevance, mem, details in results:
                    scored.append((mem, base_relevance, final_score,
                                   details.get("method", "hybrid"), details))

        if "semantic" in search_types and self._manager.is_enabled("semantic"):
            semantic_module = self._manager.get_module("semantic")
            if semantic_module and semantic_module.get_count() > 0:
                retrieval_methods.append("semantic(向量+图谱)")
                results = semantic_module.retrieve(query, limit=limit, min_importance=min_importance)
                for final_score, base_relevance, mem, details in results:
                    scored.append((mem, base_relevance, final_score,
                                   details.get("method", "hybrid"), details))

        if "perceptual" in search_types and self._manager.is_enabled("perceptual"):
            perceptual_module = self._manager.get_module("perceptual")
            if perceptual_module and perceptual_module.get_count() > 0:
                retrieval_methods.append("perceptual(多模态向量)")
                retrieve_kwargs = {"min_importance": min_importance}
                target_modality = args.get("target_modality")
                if target_modality:
                    retrieve_kwargs["target_modality"] = target_modality
                results = perceptual_module.retrieve(query, limit=limit, **retrieve_kwargs)
                for final_score, base_relevance, mem, details in results:
                    scored.append((mem, base_relevance, final_score,
                                   details.get("method", "hybrid"), details))

        other_types = search_types - {"working", "episodic", "semantic", "perceptual"}
        if other_types:
            retrieval_methods.append("其他类型(关键词匹配)")
            keywords = self._tokenize(query)
            for mem in self._manager.get_all():
                if mem.memory_type not in other_types:
                    continue
                if mem.importance < min_importance:
                    continue
                relevance = self._calculate_relevance(mem, keywords, query)
                if relevance > 0:
                    combined = relevance * 0.7 + mem.importance * 0.3
                    scored.append((mem, relevance, combined, "keyword", None))

        scored.sort(key=lambda x: x[2], reverse=True)
        results = scored[:limit]

        if not results:
            return (f"未找到匹配的记忆（已启用: {', '.join(retrieval_methods) if retrieval_methods else '无'}）\n"
                    f"   查询: {query[:80]}{'...' if len(query) > 80 else ''}")

        lines = [
            f"找到 {len(results)} 条相关记忆（查询: {query[:50]}）"
        ]
        for i, (mem, base_relevance, final_score, method, details) in enumerate(results, 1):
            lines.append(f"\n  [{i}] ({mem.memory_type}, {method})")
            lines.append(f"      内容: {mem.content[:120]}{'...' if len(mem.content) > 120 else ''}")
            if details:
                lines.append(f"      评分: {details.get('final_score', final_score):.3f} = "
                             f"基础相关 {details.get('base_relevance', base_relevance):.3f} × "
                             f"重要权重 {details.get('importance_weight', 1.0):.2f}")
            else:
                lines.append(f"      评分: {final_score:.3f} = 相关 {base_relevance:.3f} × 重要权重")
            lines.append(f"      时间: {mem.timestamp}")

        return "\n".join(lines)

    # ----------------------------------------------------------
    # 遗忘策略
    # ----------------------------------------------------------

    def _forget_memory(self, args: Dict[str, Any]) -> str:
        strategy = args.get("strategy", "importance")
        threshold = args.get("threshold", 0.3)
        max_age_days = args.get("max_age_days", 7)

        all_memories = self._manager.get_all()
        if not all_memories:
            return "记忆库为空，无需遗忘。"

        if strategy == "importance":
            removed = [m for m in all_memories if m.importance < threshold]
            kept = [m for m in all_memories if m.importance >= threshold]
        elif strategy == "time":
            cutoff = datetime.now() - _timedelta(days=max_age_days)
            removed = []
            kept = []
            for m in all_memories:
                try:
                    mt = datetime.fromisoformat(m.timestamp)
                    if mt < cutoff:
                        removed.append(m)
                    else:
                        kept.append(m)
                except (ValueError, TypeError):
                    kept.append(m)
        elif strategy == "capacity":
            capacity = int(threshold)
            if len(all_memories) <= capacity:
                return f"当前记忆数 ({len(all_memories)}) 未超过容量上限 ({capacity})，无需遗忘。"
            sorted_mem = sorted(all_memories, key=lambda m: (m.importance, m.timestamp))
            removed = sorted_mem[:len(all_memories) - capacity]
            kept = sorted_mem[len(all_memories) - capacity:]
        elif strategy == "mixed":
            cutoff = datetime.now() - _timedelta(days=max_age_days)
            removed = []
            kept = []
            for m in all_memories:
                try:
                    mt = datetime.fromisoformat(m.timestamp)
                    is_old = mt < cutoff
                    is_low_importance = m.importance < threshold
                    if is_old and is_low_importance:
                        removed.append(m)
                    else:
                        kept.append(m)
                except (ValueError, TypeError):
                    kept.append(m)
        elif strategy == "dedup":
            removed = []
            kept = []
            seen = []
            for m in all_memories:
                duplicate = False
                for existing in seen:
                    sim = self._compute_similarity(m.content, existing.content)
                    if sim >= threshold:
                        duplicate = True
                        if m.importance > existing.importance:
                            kept.remove(existing)
                            kept.append(m)
                        break
                if not duplicate:
                    kept.append(m)
                    seen.append(m)
                else:
                    removed.append(m)
        else:
            return f"未知遗忘策略: '{strategy}'，可选: importance, time, capacity, mixed, dedup"

        self._manager.replace_all(kept)

        return (
            f"遗忘完成（策略: {strategy}）\n"
            f"   已遗忘: {len(removed)} 条\n"
            f"   剩余: {len(kept)} 条"
        )

    # ----------------------------------------------------------
    # 整合操作
    # ----------------------------------------------------------

    def _consolidate_memory(self, args: Dict[str, Any]) -> str:
        from_type = args.get("from_type")
        to_type = args.get("to_type")
        importance_threshold = args.get("importance_threshold", 0.7)
        time_window_minutes = args.get("time_window_minutes")
        modality_filter = args.get("modality_filter")

        if time_window_minutes:
            return self._time_window_consolidate(time_window_minutes)

        if not from_type or not to_type:
            return "consolidate 需要 from_type + to_type（类型转换）或 time_window_minutes（时间窗口）。"

        if not self._manager.is_enabled(from_type):
            return f"源类型 '{from_type}' 未启用。"
        if not self._manager.is_enabled(to_type):
            return f"目标类型 '{to_type}' 未启用。"

        source_module = self._manager.get_module(from_type)
        source_memories = source_module.get_all() if source_module else []

        if modality_filter == "non_text":
            candidates = [m for m in source_memories
                          if m.modality and m.modality != "text"]
            if not candidates:
                return f"没有非文本模态的 '{from_type}' 记忆需要整合到 perceptual。"
            for m in candidates:
                new_entry = MemoryEntry(
                    memory_id=str(uuid.uuid4()),
                    content=m.content,
                    memory_type=to_type,
                    importance=m.importance,
                    timestamp=m.timestamp,
                    session_id=m.session_id,
                    modality=m.modality,
                    file_path=m.file_path,
                    metadata={**m.metadata,
                              "source_type": from_type,
                              "source_id": m.memory_id},
                )
                self._manager.add(new_entry)
            return (
                f"感知整合完成\n"
                f"   来源: {from_type} → 目标: {to_type}\n"
                f"   整合了 {len(candidates)} 条非文本模态记忆"
            )

        candidates = [m for m in source_memories if m.importance >= importance_threshold]
        if not candidates:
            return f"没有重要性 ≥ {importance_threshold} 的 '{from_type}' 记忆需要整合。"

        merged = f"【{from_type}→{to_type}整合】\n" + "\n".join(
            f"- [{m.importance:.1f}] {m.content[:100]}"
            for m in candidates
        )

        avg_importance = sum(m.importance for m in candidates) / len(candidates)

        new_entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content=merged,
            memory_type=to_type,
            importance=min(1.0, avg_importance + 0.1),
            timestamp=datetime.now().isoformat(),
            session_id=candidates[0].session_id,
            modality="text",
            metadata={
                "source_type": from_type,
                "source_count": len(candidates),
                "source_ids": [m.memory_id for m in candidates],
            },
        )

        self._manager.add(new_entry)

        source_module.set_memories(
            [m for m in source_memories if m.importance < importance_threshold]
        )

        return (
            f"整合完成\n"
            f"   来源: {from_type} → 目标: {to_type}\n"
            f"   整合了 {len(candidates)} 条重要性 ≥ {importance_threshold} 的记忆\n"
            f"   新记忆 ID: {new_entry.memory_id[:12]}...\n"
            f"   新记忆重要度: {new_entry.importance:.2f}"
        )

    def _time_window_consolidate(self, time_window_minutes: int) -> str:
        cutoff = datetime.now() - _timedelta(minutes=time_window_minutes)
        recent = []
        for mem in self._manager.get_all():
            try:
                mt = datetime.fromisoformat(mem.timestamp)
                if mt >= cutoff:
                    recent.append(mem)
            except (ValueError, TypeError):
                pass

        if not recent:
            return f"过去 {time_window_minutes} 分钟内没有新的记忆。"

        by_type: Dict[str, List[MemoryEntry]] = {}
        for mem in recent:
            by_type.setdefault(mem.memory_type, []).append(mem)

        lines = [f"时间窗口整合（过去 {time_window_minutes} 分钟）"]
        for mem_type, mems in by_type.items():
            avg_imp = sum(m.importance for m in mems) / len(mems)
            lines.append(f"   {mem_type}: {len(mems)} 条, 平均重要度 {avg_imp:.2f}")
        return "\n".join(lines)

    # ----------------------------------------------------------
    # 重排序操作
    # ----------------------------------------------------------

    def _rerank_memory(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "").strip()
        if not query:
            return "rerank 需要 query 参数。"

        limit = args.get("limit", 5)
        coarse_limit = args.get("coarse_limit", 15)

        memory_types = args.get("memory_types")
        search_types = memory_types or set(self._manager.get_enabled_types())

        candidates = []
        for mem_type in search_types:
            if self._manager.is_enabled(mem_type):
                candidates.extend(self._manager.get_by_type(mem_type))

        if not candidates:
            return "候选记忆为空。"

        reranker = LLMReranker()
        results = reranker.rerank(candidates, query,
                                  coarse_limit=coarse_limit,
                                  rerank_limit=limit)

        if not results:
            return f"未找到匹配的记忆（已检索 {len(candidates)} 条候选）"

        lines = [
            f"LLM 重排序完成（{len(results)} 条结果，候选 {len(candidates)} 条）"
        ]
        for i, (mem, relevance, reason) in enumerate(results, 1):
            lines.append(f"\n  [{i}] 相关度: {relevance:.2f} | {reason}")
            lines.append(f"      内容: {mem.content[:120]}{'...' if len(mem.content) > 120 else ''}")

        return "\n".join(lines)

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    @staticmethod
    def _detect_modality(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        return FILE_MODALITY_MAP.get(ext, "text")

    @staticmethod
    def _normalize_search_params(args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "").strip()
        if not query:
            return {"error": "search 操作需要 query 参数。"}

        limit = max(1, min(50, int(args.get("limit", 5))))
        min_importance = float(args.get("min_importance", 0.1))

        memory_type = args.get("memory_type")
        memory_types = args.get("memory_types")

        if memory_type:
            if isinstance(memory_types, list):
                memory_types = set(memory_types) | {memory_type}
            else:
                memory_types = {memory_type}

        return {
            "error": None,
            "query": query,
            "limit": limit,
            "memory_types": memory_types,
            "min_importance": min_importance,
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text.lower())

    @staticmethod
    def _calculate_relevance(mem: MemoryEntry, query_keywords: List[str], query: str) -> float:
        if not query_keywords:
            return 0.0

        content_keywords = set(
            re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', mem.content.lower())
        )

        if not content_keywords:
            return 0.0

        mid = set(query_keywords) & content_keywords
        jaccard = len(mid) / len(set(query_keywords) | content_keywords) if set(query_keywords) | content_keywords else 0.0
        bonus = 0.1 if query.lower() in mem.content.lower() else 0.0
        return min(1.0, jaccard + bonus)

    @staticmethod
    def _compute_similarity(text_a: str, text_b: str) -> float:
        tokens_a = set(re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text_a.lower()))
        tokens_b = set(re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0


def _timedelta(**kwargs):
    """辅助函数，避免模块级 timedelta 导入。"""
    from datetime import timedelta
    return timedelta(**kwargs)
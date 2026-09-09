"""Gather-Select-Structure-Compress（GSSC）上下文构建流水线。

将多源候选信息（RAG 检索 / 记忆系统 / 对话历史 / 自定义信息包）加工为
面向 LLM 的结构化上下文，并在超限时压缩，保证质量与预算。

四个阶段：
  1. Gather    多源汇集候选信息（系统指令最高优先级，始终保留）
  2. Select    相关性 + 新近性加权评分，按 token 预算填充选择
  3. Structure 组装结构化上下文模板（Role/Task/State/Evidence/Context/Output）
  4. Compress  超限压缩（优先保留 Task/State/Evidence，仍超限则截断）

依赖：context_builder.ContextPacket（候选信息包）
"""

import sys
import os
import re
import math
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable


from src.tools.context.context_builder import ContextPacket, ContextBuilder, ContextConfig
from src.core.cache import SafeFullCache   # 复用缓存层
from src.tools.context.context_monitor import ContextMonitor   # 上下文构建监控日志


# ============================================================
# 缓存工具函数
# ============================================================

def _fingerprint(*parts) -> str:
    """生成缓存键指纹（md5 前 16 位）。"""
    text = "|".join(str(p) for p in parts)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _packets_to_dicts(packets: List[ContextPacket]) -> List[Dict[str, Any]]:
    """ContextPacket 列表 → dict 列表（可缓存）。"""
    return [{
        "content": p.content,
        "timestamp": p.timestamp,
        "relevance_score": p.relevance_score,
        "metadata": dict(p.metadata),
    } for p in packets]


def _dicts_to_packets(dicts: List[Dict[str, Any]]) -> List[ContextPacket]:
    """dict 列表 → ContextPacket 列表（缓存恢复）。"""
    return [ContextPacket(
        content=d["content"],
        timestamp=d.get("timestamp"),
        relevance_score=d.get("relevance_score", 0.0),
        metadata=d.get("metadata", {}),
    ) for d in dicts]


# ============================================================
# 1) Gather：多源汇集候选信息
# ============================================================

class Gatherer:
    """Gather：从多个来源汇集候选信息。

    来源（按优先级）：
      1. 系统指令 —— 最高优先级，不参与评分，始终保留（metadata.priority=system）
      2. RAG 系统检索相关片段（rag_search 可调用对象）
      3. 记忆系统检索相关记忆（memory_search 可调用对象）
      4. 对话历史 —— 保留最近 n 条（n <= 5）
      5. 自定义信息包
    """

    def __init__(self, rag_search: Optional[Callable] = None,
                 memory_search: Optional[Callable] = None,
                 max_history: int = 5,
                 cache: Optional[SafeFullCache] = None,
                 cache_ttl: int = 300):
        self.rag_search = rag_search      # (query, top_k) -> [(score, content, meta_dict)]
        self.memory_search = memory_search  # 同上
        self.max_history = max(0, min(5, max_history))  # 最近 n 条对话历史（n<=5）
        self._cache = cache               # 缓存层：汇集结果可复用
        self.cache_ttl = max(1, cache_ttl)

    def gather(self, query: str, history: Optional[List[str]] = None,
               system_instruction: Optional[str] = None,
               custom_packets: Optional[List[ContextPacket]] = None,
               top_k: int = 5) -> List[ContextPacket]:
        """汇集候选信息，返回 ContextPacket 列表（结果走缓存层）。"""
        cache_key = f"gather:{_fingerprint(
            query, history or [], system_instruction or "",
            _packets_to_dicts(custom_packets or []), top_k, self.max_history)}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return _dicts_to_packets(cached)

        packets: List[ContextPacket] = []

        # 1. 系统指令：最高优先级，不参与评分，始终保留
        if system_instruction:
            packets.append(ContextPacket(
                content=system_instruction,
                timestamp=time.time(),
                relevance_score=1.0,
                metadata={"source": "system", "priority": "system"},
            ))

        # 2. RAG 系统检索
        if self.rag_search:
            try:
                for score, content, meta in self.rag_search(query, top_k):
                    packets.append(ContextPacket(
                        content=content,
                        relevance_score=float(score),
                        metadata={"source": "rag", **meta},
                    ))
            except Exception as e:
                print(f"  ⚠️ RAG 检索失败: {e}")

        # 3. 记忆系统检索
        if self.memory_search:
            try:
                for score, content, meta in self.memory_search(query, top_k):
                    packets.append(ContextPacket(
                        content=content,
                        relevance_score=float(score),
                        metadata={"source": "memory", **meta},
                    ))
            except Exception as e:
                print(f"  ⚠️ 记忆检索失败: {e}")

        # 4. 对话历史：保留最近 max_history 条
        if history:
            for item in history[-self.max_history:]:
                packets.append(ContextPacket(
                    content=item,
                    timestamp=time.time(),
                    metadata={"source": "history"},
                ))

        # 5. 自定义信息包
        if custom_packets:
            packets.extend(custom_packets)

        if self._cache is not None:
            self._cache.set(cache_key, _packets_to_dicts(packets), ttl=self.cache_ttl)
        return packets


# ============================================================
# 2) Select：相关性 + 新近性评分选择
# ============================================================

class Selector:
    """Select：对候选信息进行相关性（关键词重叠）与新近性（指数衰减）评分。

    流程：
      1. 分离系统指令，计算其 token 占用（最高优先级，始终保留）
      2. calculate_relevance() 关键词重叠算法，低于阈值过滤
      3. calculate_recency() 指数衰减模型（24h 内高分，之后衰减）
      4. 加权组合综合分数（relevance_weight + recency_weight = 1.0）
      5. 按分数从高到低填充，直到达到 token 上限
    """

    def __init__(self, config: Optional[ContextConfig] = None,
                 relevance_threshold: float = 0.1,
                 cache: Optional[SafeFullCache] = None,
                 cache_ttl: int = 300):
        self.config = config or ContextConfig()
        self.relevance_threshold = max(0.0, min(1.0, relevance_threshold))
        self._counter = ContextBuilder()
        self._cache = cache               # 缓存层：选中结果可复用
        self.cache_ttl = max(1, cache_ttl)

    def _count_tokens(self, text: str) -> int:
        return self._counter.estimate_tokens(text)

    # --- 分词（英文单词 + 中文 2-gram）---

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        tokens = re.findall(r'[a-zA-Z0-9]+', text)
        cjk = re.findall(r'[\u4e00-\u9fff]+', text)
        for seg in cjk:
            tokens.append(seg)
            if len(seg) >= 2:
                for i in range(len(seg) - 1):
                    tokens.append(seg[i:i + 2])
        return tokens

    # --- 相关性：关键词重叠算法 ---

    def calculate_relevance(self, content: str, query: str) -> float:
        """计算内容与查询的相关性（0.0-1.0）：查询词在内容中的覆盖率。

        score = 查询词与内容词的交集数 / 查询词总数
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return 0.0
        c_tokens = set(self._tokenize(content))
        overlap = sum(1 for t in q_tokens if t in c_tokens)
        return overlap / len(q_tokens)

    # --- 新近性：指数衰减模型 ---

    def calculate_recency(self, timestamp: Optional[float], now: float = None) -> float:
        """计算时间近因性分数（0.0-1.0）。

        指数衰减模型：24 小时内保持高分（≈1.0），之后按 exp 逐渐衰减。
        score = exp(-(age_hours - 24) / 72)  （age <= 24h 时为 1.0）
        """
        if timestamp is None:
            return 1.0
        now = now or time.time()
        age_hours = max(0.0, (now - timestamp) / 3600.0)
        if age_hours <= 24.0:
            return 1.0
        return max(0.0, math.exp(-(age_hours - 24.0) / 72.0))

    # --- 选择主流程 ---

    def select(self, packets: List[ContextPacket], query: str,
               available_tokens: int) -> List[ContextPacket]:
        """评分并选择信息包，返回选中列表（系统指令始终在首位）。

        Args:
            packets: 候选信息包列表
            query: 用户查询（用于计算相关性）
            available_tokens: 可用的 token 数量
        """
        if available_tokens <= 0:
            return []

        cache_key = f"select:{_fingerprint(query, available_tokens,
                                           _packets_to_dicts(packets))}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return _dicts_to_packets(cached)

        # 1. 分离系统指令（最高优先级，不参与评分，始终保留）
        system_packets = [p for p in packets
                          if p.metadata.get("priority") == "system"]
        others = [p for p in packets
                  if p.metadata.get("priority") != "system"]
        system_tokens = sum(self._count_tokens(p.content) for p in system_packets)
        remaining = available_tokens - system_tokens
        if remaining <= 0:
            return system_packets

        # 2. 相关性计算 + 阈值过滤
        scored: List[Tuple[float, ContextPacket, float, float]] = []
        now = time.time()
        for p in others:
            rel = self.calculate_relevance(p.content, query)
            if rel < self.relevance_threshold:
                continue
            rec = self.calculate_recency(p.timestamp, now)
            score = (self.config.relevance_weight * rel
                     + self.config.recency_weight * rec)
            scored.append((score, p, rel, rec))

        # 3. 按综合分数从高到低填充，直到 token 上限
        scored.sort(key=lambda x: x[0], reverse=True)
        selected: List[ContextPacket] = []
        used = 0
        for score, packet, rel, rec in scored:
            tokens = self._count_tokens(packet.content)
            if used + tokens > remaining:
                continue
            packet.metadata["score"] = round(score, 4)
            packet.metadata["relevance"] = round(rel, 4)
            packet.metadata["recency"] = round(rec, 4)
            packet.metadata["tokens"] = tokens
            selected.append(packet)
            used += tokens

        if self._cache is not None:
            self._cache.set(cache_key, _packets_to_dicts(selected), ttl=self.cache_ttl)
        return system_packets + selected


# ============================================================
# 3) Structure：结构化上下文模板
# ============================================================

class Structurer:
    """Structure：将选中信息组织成结构化上下文模板。

    模板结构：
      [Role & Policies]  Agent 的角色定位和行为准则（来自系统指令）
      [Task]             当前需要完成的具体任务（用户查询）
      [State]            Agent 的当前状态
      [Evidence]         从外部知识库检索的证据信息（RAG 来源）
      [Context]          历史对话和相关记忆
      [Output]           期望的输出格式和要求
    """

    def __init__(self, role_template: str = None, output_template: str = None,
                 state_template: str = None,
                 cache: Optional[SafeFullCache] = None,
                 cache_ttl: int = 300):
        self.role_template = role_template or "你是一个严谨的知识库问答助手。"
        self.output_template = output_template or (
            "简洁清晰地回答用户问题；引用证据时标注来源编号 [1][2]；"
            "资料中没有的信息请明确说明「资料中未提及」。")
        self.state_template = state_template or "已获取检索证据与相关记忆，准备生成回答。"
        self._cache = cache               # 缓存层：结构化结果可复用
        self.cache_ttl = max(1, cache_ttl)

    def structure(self, packets: List[ContextPacket], query: str) -> Dict[str, str]:
        """组织结构化上下文，返回 {section_name: content} 映射（结果走缓存层）。"""
        cache_key = f"structure:{_fingerprint(query, _packets_to_dicts(packets))}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        sections = {name: "" for name in
                    ["Role & Policies", "Task", "State", "Evidence", "Context", "Output"]}

        # 系统指令 → Role & Policies
        system_parts = [p.content for p in packets
                        if p.metadata.get("source") == "system"]
        sections["Role & Policies"] = self.role_template
        if system_parts:
            sections["Role & Policies"] = "\n".join(system_parts)

        sections["Task"] = query
        sections["State"] = self.state_template

        # RAG 证据 → Evidence
        evidence = [p.content for p in packets
                    if p.metadata.get("source") == "rag"]
        sections["Evidence"] = "\n".join(
            f"[{i}] {c}" for i, c in enumerate(evidence, 1)) if evidence else "（无检索证据）"

        # 历史对话 / 记忆 / 自定义 → Context
        context_parts = [p.content for p in packets
                         if p.metadata.get("source") not in ("system", "rag")]
        sections["Context"] = "\n".join(
            f"- {c}" for c in context_parts) if context_parts else "（无历史上下文）"

        sections["Output"] = self.output_template

        if self._cache is not None:
            self._cache.set(cache_key, sections, ttl=self.cache_ttl)
        return sections

    def render(self, sections: Dict[str, str]) -> str:
        """将 section 映射渲染为模板字符串。"""
        blocks = []
        for name in ["Role & Policies", "Task", "State", "Evidence", "Context", "Output"]:
            blocks.append(f"[{name}]\n{sections.get(name, '')}")
        return "\n\n".join(blocks)


# ============================================================
# 4) Compress：超限上下文压缩
# ============================================================

class Compressor:
    """Compress：对超限上下文进行压缩处理。

    - 仅当前 token 数大于最大限制时启用压缩
    - 为 max_tokens 设置 10% 预留：超过 90% 时压缩
    - 压缩优先保留重要部分：[Task]、[State]、[Evidence]
    - 压缩后仍超限 → _truncate_text() 截断
    """

    KEEP_SECTIONS = ("Task", "State", "Evidence")

    def __init__(self, reserve_ratio: float = 0.1,
                 cache: Optional[SafeFullCache] = None,
                 cache_ttl: int = 300):
        self.reserve_ratio = max(0.0, min(0.5, reserve_ratio))
        self._counter = ContextBuilder()
        self._cache = cache               # 缓存层：压缩结果可复用
        self.cache_ttl = max(1, cache_ttl)

    def _count_tokens(self, text: str) -> int:
        """估算文本的 token 数量。"""
        return self._counter.estimate_tokens(text)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数（为后缀标记预留预算，硬保证不超限）。"""
        tokens = self._count_tokens(text)
        if tokens <= max_tokens:
            return text
        suffix = "…[截断]"
        reserve = min(max_tokens - 1, 6)
        ratio = max(0.0, (max_tokens - reserve)) / max(1, tokens)
        cut = max(1, int(len(text) * ratio))
        result = text[:cut] + suffix
        # 微调保证严格 ≤ max_tokens（估算按字符比例存在误差）
        while self._count_tokens(result) > max_tokens and cut > 1:
            cut -= 1
            result = text[:cut] + suffix
        return result

    def _split_sections(self, context: str) -> List[Tuple[str, str]]:
        """按 [Section] 标题拆分上下文为 (section_name, content) 列表。"""
        pattern = re.compile(r'^\[([^\]]+)\]\s*$', re.MULTILINE)
        matches = list(pattern.finditer(context))
        if not matches:
            return [("全文", context)]
        sections = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
            sections.append((m.group(1), context[start:end].strip("\n")))
        return sections

    def compress(self, context: str, max_tokens: int) -> Tuple[str, bool, int]:
        """压缩上下文。

        Returns:
            (压缩后的上下文, 是否发生压缩, 压缩后 token 数)
        """
        cache_key = f"compress:{_fingerprint(max_tokens, context)}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return tuple(cached)

        tokens = self._count_tokens(context)

        # 未超过 90% 阈值（预留 10%）→ 不压缩
        soft_limit = max_tokens * (1.0 - self.reserve_ratio)
        if tokens <= soft_limit:
            return context, False, tokens

        # 超过 90%：压缩，优先保留 [Task]/[State]/[Evidence]
        budget = max(1, int(soft_limit))
        sections = self._split_sections(context)

        # 重要 section 完整保留，其余按剩余预算截断
        kept: List[str] = []
        important_tokens = 0
        others: List[Tuple[str, str]] = []
        for name, content in sections:
            if name in self.KEEP_SECTIONS:
                kept.append(f"[{name}]\n{content}")
                important_tokens += self._count_tokens(f"[{name}]\n{content}")
            else:
                others.append((name, content))

        remaining = budget - important_tokens
        for name, content in others:
            if remaining <= 0:
                break
            t = self._count_tokens(content)
            if t <= remaining:
                kept.append(f"[{name}]\n{content}")
                remaining -= t
            else:
                kept.append(f"[{name}]\n{self._truncate_text(content, remaining)}")
                remaining = 0

        compressed = "\n\n".join(kept)
        compressed_tokens = self._count_tokens(compressed)

        # 仍超限 → 截断到 max_tokens
        if compressed_tokens > max_tokens:
            compressed = self._truncate_text(compressed, max_tokens)
            compressed_tokens = self._count_tokens(compressed)

        if self._cache is not None:
            self._cache.set(cache_key, (compressed, True, compressed_tokens),
                            ttl=self.cache_ttl)
        return compressed, True, compressed_tokens


# ============================================================
# 流水线编排
# ============================================================

@dataclass
class GSSCResult:
    """流水线输出结果。"""
    context: str                                  # 最终上下文
    sections: Dict[str, str] = field(default_factory=dict)  # 结构化 section 映射
    packets: List[ContextPacket] = field(default_factory=list)  # 选中的信息包
    tokens: int = 0                               # 实际 token 数
    compressed: bool = False                      # 是否发生压缩


class GSSCPipeline:
    """Gather-Select-Structure-Compress 流水线（集成缓存层 + 监控日志）。

    - 创建共享缓存并注入各阶段（Gather/Select/Structure/Compress 结果均可复用）
    - 端到端 run() 结果按输入指纹缓存，相同输入直接命中
    - 每次上下文构建记录监控日志（选中数 / token 使用率 / 阶段耗时等）
    - 知识库/记忆变更时调用 clear_cache() 保证一致性
    """

    def __init__(self, gatherer: Optional[Gatherer] = None,
                 selector: Optional[Selector] = None,
                 structurer: Optional[Structurer] = None,
                 compressor: Optional[Compressor] = None,
                 max_tokens: int = 4096,
                 cache: Optional[SafeFullCache] = None,
                 cache_ttl: int = 300,
                 monitor: Optional[ContextMonitor] = None):
        self.max_tokens = max_tokens
        # 共享缓存层（未提供时自动创建）
        self._cache = cache or SafeFullCache(max_size=10000, default_ttl=cache_ttl)
        self.cache_ttl = max(1, cache_ttl)
        # 监控日志（未提供时自动创建，默认落盘 logs/context_monitor.jsonl）
        self.monitor = monitor or ContextMonitor()
        # 注入缓存到各阶段（阶段已有缓存则保留）
        self.gatherer = gatherer or Gatherer(cache=self._cache, cache_ttl=self.cache_ttl)
        self.selector = selector or Selector(cache=self._cache, cache_ttl=self.cache_ttl)
        self.structurer = structurer or Structurer(cache=self._cache, cache_ttl=self.cache_ttl)
        self.compressor = compressor or Compressor(cache=self._cache, cache_ttl=self.cache_ttl)
        for stage in (self.gatherer, self.selector, self.structurer, self.compressor):
            if getattr(stage, "_cache", None) is None:
                stage._cache = self._cache

    def clear_cache(self):
        """清空流水线缓存（知识库/记忆变更时调用，保证一致性）。"""
        self._cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """缓存层统计。"""
        return self._cache.stats()

    def run(self, query: str, history: Optional[List[str]] = None,
            system_instruction: Optional[str] = None,
            custom_packets: Optional[List[ContextPacket]] = None,
            max_tokens: Optional[int] = None) -> GSSCResult:
        """执行完整流水线（结果按输入指纹走缓存层，构建过程记录监控日志）。"""
        limit = max_tokens or self.max_tokens
        t0 = time.perf_counter()

        cache_key = f"gssc:{_fingerprint(
            query, history or [], system_instruction or "", limit,
            _packets_to_dicts(custom_packets or []))}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            result = self._result_from_dict(cached)
            # 监控：缓存命中路径（无阶段耗时）
            self.monitor.record({
                "query": query[:60], "cache_hit": True,
                "max_tokens": limit,
                "tokens_used": result.tokens,
                "token_utilization": result.tokens / max(1, limit),
                "compressed": result.compressed,
                "total_time_ms": round((time.perf_counter() - t0) * 1000, 1),
            })
            return result

        # 1. Gather 多源汇集（计时）
        t1 = time.perf_counter()
        candidates = self.gatherer.gather(
            query=query, history=history,
            system_instruction=system_instruction,
            custom_packets=custom_packets)
        t2 = time.perf_counter()

        # 2. Select 评分选择（token 预算 = max_tokens）
        selected = self.selector.select(candidates, query,
                                        available_tokens=limit)
        t3 = time.perf_counter()

        # 3. Structure 结构化模板
        sections = self.structurer.structure(selected, query)
        context = self.structurer.render(sections)
        t4 = time.perf_counter()

        # 4. Compress 超限压缩
        compressed, did_compress, tokens = self.compressor.compress(context, limit)
        t5 = time.perf_counter()

        result = GSSCResult(
            context=compressed,
            sections=sections,
            packets=selected,
            tokens=tokens,
            compressed=did_compress,
        )
        self._cache.set(cache_key, self._result_to_dict(result), ttl=self.cache_ttl)

        # 监控：记录上下文构建统计信息
        self.monitor.record(self._build_stats(
            query, candidates, selected, sections, tokens, limit,
            did_compress, t0, t1, t2, t3, t4, t5))
        return result

    # --- 监控统计构建 ---

    @staticmethod
    def _build_stats(query, candidates, selected, sections, tokens, limit,
                     did_compress, t0, t1, t2, t3, t4, t5) -> Dict[str, Any]:
        """汇总一次上下文构建的统计信息。"""
        from collections import Counter
        src_counts = Counter(p.metadata.get("source", "?") for p in candidates)

        # 系统指令占用（选中包中 priority=system）
        system_tokens = sum(
            ContextBuilder().estimate_tokens(p.content)
            for p in selected if p.metadata.get("priority") == "system")

        # 选中包质量指标（非系统包）
        scored = [p.metadata.get("score") for p in selected
                  if p.metadata.get("score") is not None]
        rels = [p.metadata.get("relevance") for p in selected
                if p.metadata.get("relevance") is not None]
        recs = [p.metadata.get("recency") for p in selected
                if p.metadata.get("recency") is not None]

        return {
            "query": query[:60],
            "cache_hit": False,
            "max_tokens": limit,
            "gather_count": len(candidates),
            "gather_sources": dict(src_counts),
            "selected_count": len(selected),
            "system_tokens": system_tokens,
            "tokens_used": tokens,
            "token_utilization": tokens / max(1, limit),
            "compressed": did_compress,
            "avg_score": round(sum(scored) / len(scored), 4) if scored else None,
            "avg_relevance": round(sum(rels) / len(rels), 4) if rels else None,
            "avg_recency": round(sum(recs) / len(recs), 4) if recs else None,
            "stage_times_ms": {
                "gather": round((t2 - t1) * 1000, 1),
                "select": round((t3 - t2) * 1000, 1),
                "structure": round((t4 - t3) * 1000, 1),
                "compress": round((t5 - t4) * 1000, 1),
            },
            "total_time_ms": round((t5 - t0) * 1000, 1),
        }

    # --- 结果序列化（缓存存取） ---

    @staticmethod
    def _result_to_dict(result: GSSCResult) -> Dict[str, Any]:
        return {
            "context": result.context,
            "sections": result.sections,
            "packets": _packets_to_dicts(result.packets),
            "tokens": result.tokens,
            "compressed": result.compressed,
        }

    @staticmethod
    def _result_from_dict(data: Dict[str, Any]) -> GSSCResult:
        return GSSCResult(
            context=data.get("context", ""),
            sections=data.get("sections", {}),
            packets=_dicts_to_packets(data.get("packets", [])),
            tokens=data.get("tokens", 0),
            compressed=data.get("compressed", False),
        )


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧬 GSSC 流水线演示（Gather-Select-Structure-Compress）")
    print("=" * 60)

    # --- 准备 RAG 知识库与记忆 ---
    from src.tools.rag.rag_tool import RagTool
    rag = RagTool(config={"cache_max_size": 500, "cache_ttl": 3600})
    rag.execute("add_text", content="RAG 是检索增强生成，先从知识库检索相关片段，"
                                    "再注入提示词由大模型生成答案。",
                title="rag_guide")
    rag.execute("add_text", content="MQE 多查询扩展独立调用 LLM 生成多样化查询提升召回；"
                                    "HyDE 生成假设文档用于向量检索。",
                title="rag_enhance")

    def rag_search(query, top_k):
        return [(score, c.content,
                 {"doc": c.doc_name, "section": c.metadata.get("section", "")})
                for score, c, _ in rag._index.search(query, top_k=top_k)]

    # 记忆系统
    try:
        from src.tools.memory.modules import MemoryManager, MemoryEntry
        from datetime import datetime as _dt
        memory = MemoryManager(config={"enabled_types": ["working", "episodic"],
                                       "working_capacity": 30})
        for i, text in enumerate(["用户偏好中文回答",
                                  "用户正在搭建 RAG 系统",
                                  "之前讨论过缓存层设计"]):
            memory.add(MemoryEntry(
                memory_id=f"m{i}", memory_type="episodic",
                content=text, importance=0.5,
                timestamp=_dt.now().isoformat(), session_id="demo",
                metadata={}))
        mem_search_impl = memory.get_module("episodic")

        def memory_search(query, top_k):
            out = []
            for _, _, entry, _ in mem_search_impl.retrieve(query, limit=top_k):
                out.append((0.5, entry.content, {"memory_type": "episodic"}))
            return out
    except Exception as e:
        memory_search = None
        print(f"  ⚠️ 记忆系统不可用: {e}")

    # --- 流水线参数 ---
    query = "多查询扩展和假设文档嵌入如何提升 RAG 检索效果"
    history = ["用户: RAG 是什么", "助手: 检索增强生成，结合检索与大模型",
               "用户: 最近在优化检索", "助手: 可以试试查询扩展策略"]
    system_instruction = "你是 RAG 检索增强专家，回答需专业、简洁，引用证据编号。"
    custom = [ContextPacket(content="补充说明：统一扩展检索框架支持 enable_mqe/enable_hyde 组合。",
                            timestamp=time.time(), relevance_score=0.4,
                            metadata={"source": "custom"})]

    # 1. Gather
    print("\n--- 1) Gather 多源汇集 ---")
    gatherer = Gatherer(rag_search=rag_search, memory_search=memory_search, max_history=5)
    candidates = gatherer.gather(query, history=history,
                                 system_instruction=system_instruction,
                                 custom_packets=custom, top_k=3)
    from collections import Counter
    src_counts = Counter(p.metadata.get("source", "?") for p in candidates)
    print(f"   候选信息: {len(candidates)} 个 | 来源分布: {dict(src_counts)}")

    # 2. Select
    print("\n--- 2) Select 评分选择（相关性阈值 0.15）---")
    selector = Selector(ContextConfig(relevance_weight=0.6, recency_weight=0.4),
                        relevance_threshold=0.15)
    selected = selector.select(candidates, query, available_tokens=400)
    for p in selected:
        tag = p.metadata.get("source", "?")
        if tag == "system":
            print(f"   [system] 最高优先级始终保留 | {p.content[:24]}...")
        else:
            print(f"   [{tag}] score={p.metadata.get('score')} "
                  f"rel={p.metadata.get('relevance')} rec={p.metadata.get('recency')} "
                  f"| {p.content[:24]}...")

    # 3. Structure
    print("\n--- 3) Structure 结构化模板 ---")
    structurer = Structurer()
    sections = structurer.structure(selected, query)
    rendered = structurer.render(sections)
    for line in rendered.splitlines()[:14]:
        print(f"   {line[:72]}")

    # 4. Compress（正常不压缩）
    print("\n--- 4) Compress（正常上下文，未超 90% → 不压缩）---")
    compressor = Compressor()
    ctx_out, did_c, tok = compressor.compress(rendered, max_tokens=500)
    print(f"   tokens={tok} ≤ 500×0.9 → 压缩: {did_c}")

    # 4b. Compress（超限 → 压缩优先保留重要部分 → 截断）
    print("\n--- 4b) Compress（max_tokens=250 超限 → 优先保留重要 section）---")
    big_context = "\n\n".join([
        f"[Role & Policies]\n" + "次要的角色说明。" * 60,
        f"[Task]\n" + "重要任务内容。" * 8,
        f"[State]\n" + "重要状态内容。" * 8,
        f"[Evidence]\n" + "重要证据内容。" * 8,
        f"[Context]\n" + "次要的历史内容。" * 60,
        f"[Output]\n" + "次要的输出要求。" * 60,
    ])
    print(f"   原始 tokens={compressor._count_tokens(big_context)}")
    c_out, c_flag, c_tok = compressor.compress(big_context, max_tokens=250)
    print(f"   压缩后={c_tok}（≤250）| 发生压缩: {c_flag}")
    kept_secs = re.findall(r'^\[([^\]]+)\]', c_out, re.MULTILINE)
    print(f"   保留的 section: {kept_secs}（重要部分 Task/State/Evidence 完整保留）")
    print(f"   压缩后内容: {c_out[:120]}...")

    # 5. 完整流水线
    print("\n--- 5) GSSCPipeline 端到端 ---")
    pipeline = GSSCPipeline(gatherer=gatherer, selector=selector,
                            structurer=structurer, compressor=compressor,
                            max_tokens=500)
    result = pipeline.run(query, history=history,
                          system_instruction=system_instruction,
                          custom_packets=custom)
    print(f"   选中包: {len(result.packets)} | tokens={result.tokens} | "
          f"压缩={result.compressed}")
    print("   最终上下文:")
    for line in result.context.splitlines()[:10]:
        print(f"     {line[:72]}")
    print()

    # 5b. 缓存层集成验证
    print("--- 5b) 缓存层集成（重复运行命中缓存）---")
    s0 = pipeline.cache_stats()
    pipeline.run(query, history=history, system_instruction=system_instruction,
                 custom_packets=custom)
    s1 = pipeline.cache_stats()
    r2 = pipeline.run(query, history=history, system_instruction=system_instruction,
                      custom_packets=custom)
    s2 = pipeline.cache_stats()
    print(f"   首轮（未命中）: 新增缓存项 +{s1['size'] - s0['size']}，命中 +{s1['hits'] - s0['hits']}")
    print(f"   二轮（命中缓存）: 新增缓存项 +{s2['size'] - s1['size']}，命中 +{s2['hits'] - s1['hits']} ✅")
    print(f"   两次结果一致: {result.context == r2.context}")
    print(f"   缓存统计: size={s2['size']} 命中率={s2['hit_rate']:.1%}")
    print(f"   变更知识库后 clear_cache():")
    pipeline.clear_cache()
    print(f"     size={pipeline.cache_stats()['size']}（已清空，保证一致性）")
    print()

    # 5c. 监控日志（每次上下文构建的统计信息）
    print("--- 5c) 监控日志（每次上下文构建统计）---")
    # 新增一次压缩触发场景，丰富监控样本
    pipeline.run("超长查询触发压缩，" + "测试内容" * 30,
                 history=history, system_instruction=system_instruction,
                 custom_packets=custom, max_tokens=120)
    recent = pipeline.monitor.get_recent(3)
    for e in recent:
        stage = e.get("stage_times_ms")
        stage_note = (f" 阶段耗时: {stage}" if stage else " 缓存命中路径")
        print(f"   [{e['ts'][11:]}]{'✅缓存' if e.get('cache_hit') else ' 构建 '} "
              f"选中{e.get('selected_count', '-')} | "
              f"tokens={e.get('tokens_used', '-')}/{e.get('max_tokens')} "
              f"利用率={e.get('token_utilization', 0):.0%} "
              f"压缩={e.get('compressed', '-')} 耗时={e.get('total_time_ms')}ms")
        print(f"      {stage_note}")
    print()
    agg = pipeline.monitor.get_aggregate()
    print(f"   聚合统计: 记录{agg['entries']}条 | 平均选中{agg['avg_selected_count']} | "
          f"平均token利用率{agg['avg_token_utilization']:.1%} | "
          f"缓存命中率{agg['cache_hit_rate']:.0%} | "
          f"平均耗时{agg['avg_total_time_ms']:.1f}ms")
    print(f"   来源分布: {agg['source_distribution']}")
    print(f"   监控器: {pipeline.monitor.stats()}")
    print()
    print("✅ GSSC 流水线演示完成")
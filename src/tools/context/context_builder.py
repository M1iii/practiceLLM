"""上下文构建器（ContextBuilder）。

将检索/记忆召回的信息封装为标准上下文，供 LLM 使用。
核心数据结构：
  1. ContextPacket：候选信息包（内容、时间戳、token 数、相关性分数、可选元数据）
  2. ContextConfig：构建配置（最大 token、系统指令预留比例、最低相关性阈值、
     压缩开关、新近性权重 + 相关性权重 = 1.0）

构建流程：筛选（最低相关性）→ 综合评分排序（相关性 + 新近性加权）
        → 预算分配（系统指令预留 + 可选压缩）→ 组装上下文。

使用方式:
    builder = ContextBuilder(ContextConfig(max_tokens=2048, min_relevance=0.3,
                                           recency_weight=0.3, relevance_weight=0.7))
    result = builder.build(packets)
    prompt = result["system"] + "\n\n" + result["context"]
"""

import sys
import os
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple



# ============================================================
# 核心数据结构
# ============================================================

@dataclass
class ContextPacket:
    """候选信息包：将每条候选信息封装为一个包。

    Attributes:
        content: 信息内容（文本）
        timestamp: 信息产生时间（Unix 时间戳，秒），None 表示未知
        token_count: token 数量（None 时由构建器自动估算）
        relevance_score: 相关性分数（0.0-1.0）
        metadata: 可选元数据（来源、章节、类型等）
    """
    content: str
    timestamp: Optional[float] = None
    token_count: Optional[int] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextConfig:
    """上下文构建配置。

    Attributes:
        max_tokens: 上下文最大 token 数量
        system_instruction_ratio: 为系统指令预留的比例（0.0-1.0），
                                  实际内容预算 = max_tokens × (1 - ratio)
        min_relevance: 最低相关性阈值，低于该值的包被过滤
        enable_compression: 是否启用压缩（超预算时截断保留）
        recency_weight: 新近性权重（0.0-1.0）
        relevance_weight: 相关性权重（0.0-1.0），
                          注意：recency_weight + relevance_weight 必须为 1.0
    """
    max_tokens: int = 4096
    system_instruction_ratio: float = 0.1
    min_relevance: float = 0.0
    enable_compression: bool = False
    recency_weight: float = 0.3
    relevance_weight: float = 0.7

    def __post_init__(self):
        """配置校验：范围检查 + 权重和必须为 1.0。"""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正数，当前: {self.max_tokens}")
        if not 0.0 <= self.system_instruction_ratio <= 1.0:
            raise ValueError(f"system_instruction_ratio 必须在 0.0~1.0，"
                             f"当前: {self.system_instruction_ratio}")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError(f"min_relevance 必须在 0.0~1.0，当前: {self.min_relevance}")
        for name, value in (("recency_weight", self.recency_weight),
                            ("relevance_weight", self.relevance_weight)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 0.0~1.0，当前: {value}")
        total = self.recency_weight + self.relevance_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"recency_weight 与 relevance_weight 之和必须为 1.0，"
                f"当前: {self.recency_weight} + {self.relevance_weight} = {total}")


# ============================================================
# 上下文构建器
# ============================================================

class ContextBuilder:
    """上下文构建器：筛选 → 评分排序 → 预算分配 → 组装。"""

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()

    # ------------------------------------------------------------
    # Token 估算
    # ------------------------------------------------------------

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or
            0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F or
            0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF or
            0xF900 <= code <= 0xFAFF
        )

    def estimate_tokens(self, text: str) -> int:
        """近似 Token 估算：CJK 1 字 ≈ 1 token，其他按空白分词每个 1 token。"""
        if not text:
            return 0
        cjk = sum(1 for ch in text if self._is_cjk(ch))
        non_cjk = len([t for t in text.split() if t])
        return cjk + non_cjk

    def _ensure_tokens(self, packet: ContextPacket) -> int:
        """获取包的 token 数（未提供时自动估算）。"""
        if packet.token_count is None or packet.token_count <= 0:
            return self.estimate_tokens(packet.content)
        return packet.token_count

    # ------------------------------------------------------------
    # 筛选与评分
    # ------------------------------------------------------------

    def select(self, packets: List[ContextPacket]) -> List[ContextPacket]:
        """按最低相关性阈值筛选候选包。"""
        return [p for p in packets if p.relevance_score >= self.config.min_relevance]

    @staticmethod
    def _recency_score(ts: Optional[float], min_ts: float, max_ts: float) -> float:
        """新近度归一化（0.0-1.0）：时间戳越新越接近 1.0。

        基于候选池内时间范围线性归一化；时间戳未知或范围为零时视为最新。
        """
        if ts is None or max_ts <= min_ts:
            return 1.0
        return max(0.0, min(1.0, (ts - min_ts) / (max_ts - min_ts)))

    def rank(self, packets: List[ContextPacket]) -> List[Tuple[float, ContextPacket, float]]:
        """综合评分排序：score = relevance_weight×相关性 + recency_weight×新近度。

        Returns:
            [(score, packet, recency_score), ...] 按 score 降序
        """
        now = time.time()
        timestamps = [p.timestamp if p.timestamp is not None else now for p in packets]
        min_ts, max_ts = min(timestamps), max(timestamps)

        scored = []
        for p in packets:
            recency = self._recency_score(p.timestamp, min_ts, max_ts)
            score = (self.config.relevance_weight * max(0.0, min(1.0, p.relevance_score))
                     + self.config.recency_weight * recency)
            scored.append((score, p, recency))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ------------------------------------------------------------
    # 压缩与组装
    # ------------------------------------------------------------

    def _compress(self, content: str, budget: int) -> str:
        """压缩：超出预算时按 token 比例截断并标记（为后缀预留预算，硬保证不超）。"""
        tokens = self.estimate_tokens(content)
        if tokens <= budget:
            return content
        suffix = "…[已压缩]"
        reserve = min(budget - 1, 8)                 # 为后缀标记预留 token
        ratio = max(0.0, (budget - reserve)) / max(1, tokens)
        cut = max(1, int(len(content) * ratio))
        result = content[:cut] + suffix
        # 微调保证严格 ≤ budget（估算按字符比例存在误差）
        while self.estimate_tokens(result) > budget and cut > 1:
            cut -= 1
            result = content[:cut] + suffix
        return result

    def build(self, packets: List[ContextPacket]) -> Dict[str, Any]:
        """构建上下文：筛选 → 评分排序 → 预算分配 → 组装。

        Returns:
            {
              system: 系统指令占位区（预留 token 说明）
              context: 组装后的上下文文本
              packets: 实际入选的 ContextPacket 列表
              tokens: 实际使用的 token 数
              budget: 内容区预算（max_tokens × (1 - ratio)）
            }
        """
        # 1. 筛选（最低相关性阈值）
        selected = self.select(packets)
        if not selected:
            return {"system": "", "context": "", "packets": [], "tokens": 0, "budget": 0}

        # 2. 综合评分排序（相关性 + 新近性加权）
        scored = self.rank(selected)

        # 3. 预算分配：系统指令预留
        system_tokens = int(self.config.max_tokens * self.config.system_instruction_ratio)
        context_budget = self.config.max_tokens - system_tokens

        # 4. 按分数装入预算（可选压缩）
        kept: List[Dict[str, Any]] = []
        used = 0
        for score, packet, recency in scored:
            tokens = self._ensure_tokens(packet)
            remaining = context_budget - used

            if tokens <= remaining:
                kept.append({"score": score, "packet": packet,
                             "recency": recency, "content": packet.content,
                             "tokens": tokens, "compressed": False})
                used += tokens
                continue

            # 超出剩余预算：启用压缩时截断容纳，否则停止（后续分数更低）
            if self.config.enable_compression and remaining > 8:
                compressed = self._compress(packet.content, remaining)
                c_tokens = self.estimate_tokens(compressed)
                if c_tokens > 0:
                    kept.append({"score": score, "packet": packet,
                                 "recency": recency, "content": compressed,
                                 "tokens": c_tokens, "compressed": True})
                    used += c_tokens
                continue
            break

        # 5. 组装上下文
        parts = []
        for i, k in enumerate(kept, 1):
            header = f"[{i}] 相关度:{k['packet'].relevance_score:.2f} 新近度:{k['recency']:.2f}"
            if k.get("compressed"):
                header += " [已压缩]"
            parts.append(f"{header}\n{k['content']}")
        context = "\n\n".join(parts)

        return {
            "system": (f"【系统指令区】预留 {system_tokens} tokens "
                       f"（{self.config.system_instruction_ratio:.0%}）"),
            "context": context,
            "packets": [k["packet"] for k in kept],
            "tokens": used,
            "budget": context_budget,
        }

    def build_text(self, packets: List[ContextPacket]) -> str:
        """便捷方法：返回可直接拼入提示词的文本（系统区 + 上下文）。"""
        result = self.build(packets)
        if not result["context"]:
            return result["system"] or ""
        return f"{result['system']}\n\n{result['context']}"


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧩 ContextBuilder 上下文构建器演示")
    print("=" * 60)

    # 1. 配置校验：权重和必须为 1.0
    print("--- 配置校验（权重和必须为 1.0）---")
    try:
        ContextConfig(recency_weight=0.5, relevance_weight=0.3)
        print("   ❌ 未触发校验")
    except ValueError as e:
        print(f"   ✅ 捕获: {e}")
    print()

    # 2. 构建候选信息包（不同相关性 / 新近性）
    print("--- 构建候选 ContextPacket（6 个，相关性 + 时间戳差异）---")
    packets = [
        ContextPacket(content="RAG 是检索增强生成，结合信息检索与大语言模型。",
                      timestamp=time.time() - 3600 * 24 * 5, relevance_score=0.62),
        ContextPacket(content="MQE 多查询扩展：独立调用 LLM 生成多样化查询，提升召回。",
                      timestamp=time.time() - 3600 * 1, relevance_score=0.45),
        ContextPacket(content="HyDE 假设文档嵌入：生成假设性答案文档用于向量检索。",
                      timestamp=time.time() - 3600 * 3, relevance_score=0.38),
        ContextPacket(content="缓存层 SafeFullCache：TTL 淘汰 + 容量限制 + 一致性更新。",
                      timestamp=time.time() - 3600 * 2, relevance_score=0.51),
        ContextPacket(content="旧闻：去年的过时技术讨论，相关性较低。",
                      timestamp=time.time() - 3600 * 24 * 30, relevance_score=0.28),
        ContextPacket(content="低相关噪声信息，不应被选中。",
                      timestamp=time.time() - 3600 * 24, relevance_score=0.05),
    ]
    for i, p in enumerate(packets, 1):
        print(f"   包{i}: 相关={p.relevance_score:.2f} "
              f"新近={(time.time() - p.timestamp) / 3600:.0f}h前 | {p.content[:20]}...")
    print()

    # 3. 筛选 + 综合评分
    print("--- 筛选（min_relevance=0.3）→ 综合评分（relevance 0.7 + recency 0.3）---")
    config = ContextConfig(max_tokens=200, system_instruction_ratio=0.1,
                           min_relevance=0.3, enable_compression=False,
                           recency_weight=0.3, relevance_weight=0.7)
    builder = ContextBuilder(config)
    selected = builder.select(packets)
    print(f"   筛选后: {len(selected)}/{len(packets)} 个包")
    for score, packet, recency in builder.rank(selected):
        print(f"   score={score:.3f} | 相关={packet.relevance_score:.2f} "
              f"新近={recency:.2f} | {packet.content[:16]}...")
    print()

    # 4. 构建上下文（预算分配）
    print("--- 构建上下文（max_tokens=200, 系统预留 10%）---")
    result = builder.build(packets)
    print(f"   内容预算: {result['budget']} tokens | 实际使用: {result['tokens']} | "
          f"入选 {len(result['packets'])} 包")
    print(f"   {result['system']}")
    print("   上下文:")
    for line in result["context"].splitlines()[:12]:
        print(f"     {line[:70]}")
    print()

    # 5. 压缩模式对比
    print("--- 压缩模式（enable_compression=True）对比 ---")
    long_packet = ContextPacket(
        content="压缩测试。" * 200,  # 800 tokens
        timestamp=time.time(), relevance_score=0.9)
    compact_cfg = ContextConfig(max_tokens=150, system_instruction_ratio=0.0,
                                enable_compression=True)
    compact_result = ContextBuilder(compact_cfg).build([long_packet])
    print(f"   未压缩: {builder.estimate_tokens(long_packet.content)} tokens")
    print(f"   压缩后: {compact_result['tokens']} tokens（预算 {compact_result['budget']}）")
    print(f"   内容: {compact_result['context'][-20:]}")
    print()

    # 6. 低相关性过滤（min_relevance=0.5）
    print("--- 高阈值过滤（min_relevance=0.5）---")
    strict = ContextBuilder(ContextConfig(max_tokens=500, min_relevance=0.5))
    strict_result = strict.build(packets)
    print(f"   入选 {len(strict_result['packets'])} 包: "
          f"{[p.content[:10] + '...' for p in strict_result['packets']]}")
    print()
    print("✅ ContextBuilder 演示完成")
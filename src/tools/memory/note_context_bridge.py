"""NoteTool × ContextBuilder 桥接（NoteContextBridge）。

在每轮对话前检索相关笔记，并利用 ContextBuilder 构建上下文注入提示词，
为长时程任务提供"记忆注入"能力。

特性：
  1. 检索：复用 NoteTool.find_notes（标题+内容关键词 + 时间混合评分）
  2. 类型加权：不同笔记类型设置不同相关性加成（blocker > action > conclusion），
     使阻塞/行动类笔记更容易被优先纳入上下文
  3. 数量限制：limit 参数限制笔记条数 + ContextBuilder token 预算限制，
     双重避免上下文过载
  4. 输出：ContextPacket 列表 / ContextBuilder 构建结果 / 可直接拼接的文本

使用方式:
    bridge = NoteContextBridge(notes_dir="data/notes")
    bridge.create_note_proxy(...)          # 或直接操作 note_tool
    context_text = bridge.prepare("当前任务：接入 GSSC 流水线")
    prompt = system_prompt + context_text + user_query
"""

import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


from src.tools.context.context_builder import ContextPacket, ContextBuilder, ContextConfig
from src.tools.memory.note_tool import NoteTool, NOTE_TYPES


class NoteContextBridge:
    """NoteTool × ContextBuilder 桥接：每轮对话前检索相关笔记并构建上下文。"""

    # 类型相关性加成：blocker > action > conclusion > task_state > reference > general
    TYPE_BOOST: Dict[str, float] = {
        "blocker": 0.25,     # 阻塞事项最高优先级
        "action": 0.15,      # 行动事项
        "conclusion": 0.10,  # 结论
        "task_state": 0.08,  # 任务状态
        "reference": 0.05,   # 参考资料
        "general": 0.0,      # 通用
    }

    def __init__(self, note_tool: Optional[NoteTool] = None,
                 builder: Optional[ContextBuilder] = None,
                 config: Optional[ContextConfig] = None,
                 default_limit: int = 5,
                 notes_dir: Optional[str] = None):
        """初始化桥接。

        Args:
            note_tool: NoteTool 实例（未提供则用 notes_dir 创建）
            builder: ContextBuilder 实例（未提供则用 config 创建）
            config: ContextBuilder 配置（max_tokens / min_relevance 等）
            default_limit: 默认笔记数量上限（避免上下文过载）
            notes_dir: 笔记存储目录（创建 NoteTool 时使用）
        """
        self.note_tool = note_tool or NoteTool(notes_dir=notes_dir)
        self.config = config or ContextConfig(
            max_tokens=800,          # 笔记上下文 token 预算
            system_instruction_ratio=0.0,
            min_relevance=0.05,      # 低相关性笔记过滤
            enable_compression=True, # 超限时压缩
            relevance_weight=0.7,
            recency_weight=0.3,
        )
        self.builder = builder or ContextBuilder(self.config)
        self.default_limit = max(1, min(20, default_limit))

    # ------------------------------------------------------------
    # 类型加权相关性
    # ------------------------------------------------------------

    def type_boost(self, note_type: str) -> float:
        """获取笔记类型的相关性加成（blocker > action > conclusion）。"""
        return self.TYPE_BOOST.get(note_type, 0.0)

    def relevance_with_type(self, base_score: float, note_type: str) -> float:
        """基础相关性 + 类型加成（clamp 到 0.0-1.0）。"""
        return max(0.0, min(1.0, base_score + self.type_boost(note_type)))

    # ------------------------------------------------------------
    # 检索 → ContextPacket
    # ------------------------------------------------------------

    def retrieve(self, query: str, limit: Optional[int] = None,
                 note_type: Optional[str] = None,
                 tags: Optional[List[str]] = None) -> List[ContextPacket]:
        """检索相关笔记并封装为 ContextPacket（含类型加权相关性）。

        Returns:
            ContextPacket 列表，最多 limit 条（避免上下文过载）
        """
        limit = limit or self.default_limit
        limit = max(1, min(20, limit))

        packets: List[ContextPacket] = []
        # 多取 2 倍候选，类型加权后再截断，保证加权排序生效
        raw = self.note_tool.find_notes(query, limit=limit * 2,
                                        note_type=note_type, tags=tags)
        for base_score, meta, content in raw:
            note_type_str = meta.get("type", "general")
            boosted = self.relevance_with_type(base_score, note_type_str)

            timestamp = None
            try:
                timestamp = datetime.fromisoformat(
                    str(meta.get("updated_at", ""))).timestamp()
            except (ValueError, TypeError):
                timestamp = None

            packets.append(ContextPacket(
                content=(f"[{note_type_str}] {meta.get('title', '')}\n{content}"
                         if content else f"[{note_type_str}] {meta.get('title', '')}"),
                timestamp=timestamp,
                relevance_score=boosted,
                token_count=None,
                metadata={
                    "source": "note",
                    "note_id": meta.get("id", ""),
                    "note_type": note_type_str,
                    "tags": meta.get("tags", []),
                    "base_score": round(base_score, 4),
                },
            ))

        # 按加权后相关性降序，截断到 limit（数量限制）
        packets.sort(key=lambda p: p.relevance_score, reverse=True)
        return packets[:limit]

    # ------------------------------------------------------------
    # 构建上下文
    # ------------------------------------------------------------

    def build(self, query: str, limit: Optional[int] = None,
              note_type: Optional[str] = None,
              tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """检索相关笔记 → ContextBuilder 构建（token 预算限制）。

        Returns:
            ContextBuilder.build() 结果：{system, context, packets, tokens, budget}
        """
        packets = self.retrieve(query, limit=limit, note_type=note_type, tags=tags)
        return self.builder.build(packets)

    def prepare(self, query: str, limit: Optional[int] = None,
                note_type: Optional[str] = None,
                tags: Optional[List[str]] = None) -> str:
        """每轮对话前调用：返回可直接拼入提示词的笔记上下文文本。"""
        return self.builder.build_text(
            self.retrieve(query, limit=limit, note_type=note_type, tags=tags))

    # ------------------------------------------------------------
    # NoteTool 便捷代理
    # ------------------------------------------------------------

    def create_note(self, title: str, content: str, note_type: str = "general",
                    tags: Optional[List[str]] = None) -> str:
        return self.note_tool.create_note(title, content, note_type, tags)

    def update_note(self, note_id: str, **kwargs) -> str:
        return self.note_tool.update_note(note_id, **kwargs)

    def delete_note(self, note_id: str) -> str:
        return self.note_tool.delete_note(note_id)

    def summary(self, recent: int = 5) -> Dict[str, Any]:
        return self.note_tool.summary(recent=recent)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import shutil
    import tempfile

    print("=" * 60)
    print("🔗 NoteTool × ContextBuilder 桥接演示")
    print("=" * 60)

    demo_dir = os.path.join(tempfile.gettempdir(), "note_bridge_demo")
    shutil.rmtree(demo_dir, ignore_errors=True)
    bridge = NoteContextBridge(notes_dir=demo_dir, default_limit=4)

    # 准备笔记（不同类型，相同主题便于观察类型加权）
    print("--- 准备笔记（同主题不同类型）---")
    print(bridge.create_note("阻塞：嵌入 API 触发限流",
                             "text-embedding-v3 QPS 超限，需要退避重试，否则检索失败。",
                             note_type="blocker", tags=["api"]))
    print(bridge.create_note("行动：接入 GSSC 流水线",
                             "将 NoteTool 记忆注入与上下文构建器接入对话循环。",
                             note_type="action", tags=["gssc"]))
    print(bridge.create_note("结论：MQE 提升检索效果",
                             "多查询扩展将最佳相关分提升 35%，优于基础检索。",
                             note_type="conclusion", tags=["rag"]))
    print(bridge.create_note("参考：SafeFullCache 用法",
                             "缓存层支持 TTL 淘汰与容量限制，可复用。",
                             note_type="reference", tags=["缓存"]))
    print()

    # 1. 类型加权相关性
    print("--- 1) 类型加权相关性（blocker > action > conclusion）---")
    query = "检索效果优化与流水线接入"
    for score, meta, _ in bridge.note_tool.find_notes(query, limit=10):
        boosted = bridge.relevance_with_type(score, meta.get("type", "general"))
        print(f"   {meta.get('type', ''):<10} 基础={score:.3f} + 加成{bridge.type_boost(meta.get('type','')):.2f} "
              f"→ 加权={boosted:.3f} | {meta.get('title', '')[:16]}")
    print()

    # 2. 检索 → ContextPacket（数量限制）
    print("--- 2) retrieve 检索（limit=3，避免上下文过载）---")
    packets = bridge.retrieve(query, limit=3)
    print(f"   返回 {len(packets)} 条（≤ limit=3）")
    for p in packets:
        print(f"   [{p.metadata['note_type']}] rel={p.relevance_score:.3f} "
              f"| {p.metadata['note_id']} | {p.content.splitlines()[0]}")

    # 3. 构建上下文（token 预算）
    print()
    print("--- 3) build 构建上下文（max_tokens=800）---")
    result = bridge.build(query, limit=3)
    print(f"   入选 {len(result['packets'])} 条 | tokens={result['tokens']}/{result['budget']} "
          f"| 压缩={len(result['context']) > 0 and result['tokens'] >= result['budget']}")
    for line in result["context"].splitlines()[:8]:
        print(f"     {line[:64]}")

    # 4. prepare 每轮对话前注入
    print()
    print("--- 4) prepare 每轮对话前注入 ---")
    text = bridge.prepare(query, limit=2)
    print(f"   注入文本（{len(text.splitlines())} 行，前 5 行）:")
    for line in text.splitlines()[:5]:
        print(f"     {line[:64]}")
    print()
    print("✅ NoteTool × ContextBuilder 桥接演示完成")
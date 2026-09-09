"""结构化笔记工具（NoteTool）：为长时程任务提供的结构化外部记忆。

每则笔记以「Markdown + YAML Front Matter」混合格式存储为独立 .md 文件，
包含元数据（id / title / type / tags / created_at / updated_at）与正文。

功能：
  1. create_note  创建笔记（唯一 id：note_{时间戳}_{len(index)}）
  2. read_note    按 id 读取（60s 缓存，更新/删除时清除对应缓存）
  3. update_note  更新字段（刷新时间戳、重构保存、更新索引）
  4. search_notes 标题+内容搜索（类型/标签过滤，相关性+时间混合排序，
                   仅热门搜索词 60s 缓存）
  5. list_notes   列出元数据（过滤/更新时间排序/分页，15s 缓存）
  6. summary      统计摘要（按类型/标签统计 + 最近 n 条，30s 缓存）
  7. delete_note  删除笔记（删文件 + 移除索引）

缓存策略：复用 SafeFullCache，各操作独立 TTL；
创建/更新/删除时按规则清除相关缓存保证一致性。

使用方式:
    note_tool = NoteTool()
    note_tool.execute("create_note", title="任务状态", content="## 进行中",
                      note_type="task_state", tags=["rag", "进行中"])
    note_tool.execute("read_note", note_id="note_1750000000_0")
"""

import sys
import os
import re
import time
import math
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple


from src.tools.framework.tool_system import Tool, ToolParameter, dual_protocol_execute
from src.core.cache import SafeFullCache   # 复用缓存层

NOTE_TYPES = ("task_state", "conclusion", "blocker", "action", "reference", "general")

DEFAULT_NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "data", "notes")

CACHE_TTL = {
    "note": 60,      # read_note 缓存 60s
    "search": 60,    # search_notes 热门词缓存 60s
    "list": 15,      # list_notes 缓存 15s
    "summary": 30,   # summary 缓存 30s
}


class NoteTool(Tool):
    """结构化笔记工具：Markdown+YAML 混合格式，长时程任务外部记忆。"""

    def __init__(self, notes_dir: Optional[str] = None):
        self.notes_dir = notes_dir or DEFAULT_NOTES_DIR
        os.makedirs(self.notes_dir, exist_ok=True)
        self.index: Dict[str, Dict[str, Any]] = {}   # note_id -> 元数据(+path)
        self._hot_queries: Counter = Counter()       # 热门搜索词统计
        # 各操作独立缓存（TTL 见 CACHE_TTL）
        self._note_cache = SafeFullCache(max_size=500, default_ttl=CACHE_TTL["note"])
        self._search_cache = SafeFullCache(max_size=200, default_ttl=CACHE_TTL["search"])
        self._list_cache = SafeFullCache(max_size=200, default_ttl=CACHE_TTL["list"])
        self._summary_cache = SafeFullCache(max_size=50, default_ttl=CACHE_TTL["summary"])
        self._load_index()

    # ------------------------------------------------------------
    # Markdown + YAML 混合格式
    # ------------------------------------------------------------

    def build_markdown(self, metadata: Dict[str, Any], content: str) -> str:
        """构建 Markdown + YAML 混合文件内容（YAML Front Matter + 正文）。"""
        yaml_lines = ["---"]
        for key in ("id", "title", "type", "tags", "created_at", "updated_at"):
            if key not in metadata:
                continue
            value = metadata[key]
            if key == "tags":
                tags = ", ".join(value) if value else ""
                yaml_lines.append(f"tags: [{tags}]")
            else:
                yaml_lines.append(f"{key}: {value}")
        yaml_lines.append("---")
        return "\n".join(yaml_lines) + "\n\n" + (content or "").strip() + "\n"

    def parse_markdown(self, raw_content: str) -> Tuple[Dict[str, Any], str]:
        """解析 Markdown 文件：分离 YAML Front Matter 与正文。

        Returns:
            (metadata, body)；无 YAML 头时返回 ({}, 全文)
        """
        raw = raw_content.strip()
        if not raw.startswith("---"):
            return {}, raw

        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw

        yaml_block = parts[1].strip()
        body = parts[2].strip()
        metadata: Dict[str, Any] = {}
        for line in yaml_block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "tags":
                metadata[key] = [t.strip() for t in value.strip("[]").split(",") if t.strip()]
            else:
                metadata[key] = value
        return metadata, body

    # ------------------------------------------------------------
    # 文件读写
    # ------------------------------------------------------------

    def _note_path(self, note_id: str) -> str:
        return os.path.join(self.notes_dir, f"{note_id}.md")

    def _read_file(self, note_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        path = self._note_path(note_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return self.parse_markdown(f.read())

    def _write_file(self, note_id: str, metadata: Dict[str, Any], content: str):
        with open(self._note_path(note_id), "w", encoding="utf-8") as f:
            f.write(self.build_markdown(metadata, content))

    def _load_index(self):
        """启动时扫描已有笔记文件，重建索引。"""
        for filename in sorted(os.listdir(self.notes_dir)):
            if not filename.endswith(".md"):
                continue
            note_id = filename[:-3]
            meta, _ = self._read_file(note_id)
            if not meta or not meta.get("id"):
                continue
            self.index[note_id] = {
                **meta,
                "id": note_id,
                "tags": meta.get("tags", []),
                "path": self._note_path(note_id),
            }

    # ------------------------------------------------------------
    # 缓存失效
    # ------------------------------------------------------------

    def _evict_search_by_note(self, note_id: str):
        """清除包含该笔记的搜索缓存（更新/删除时调用）。"""
        for key in list(self._search_cache._backend.keys):
            try:
                value = self._search_cache._backend.get(key)
            except KeyError:
                continue
            if isinstance(value, str) and note_id in value:
                self._search_cache._delete(key)

    def _invalidate_note(self, note_id: str):
        """更新/删除笔记：清除该笔记读取缓存 + 相关搜索缓存。"""
        self._note_cache._delete(f"note:{note_id}")
        self._evict_search_by_note(note_id)

    def _invalidate_aggregate(self):
        """创建/更新/删除笔记：清除所有列表与摘要缓存。"""
        self._list_cache.clear()
        self._summary_cache.clear()

    # ------------------------------------------------------------
    # 搜索相关工具函数
    # ------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        tokens = re.findall(r'[a-zA-Z0-9]+', text)
        for seg in re.findall(r'[\u4e00-\u9fff]+', text):
            tokens.append(seg)
            if len(seg) >= 2:
                tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        return tokens

    def _relevance(self, content: str, query: str) -> float:
        """关键词重叠相关性（0.0-1.0）：查询词在内容中的覆盖率。"""
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return 0.0
        c_tokens = set(self._tokenize(content))
        return sum(1 for t in q_tokens if t in c_tokens) / len(q_tokens)

    def _recency(self, updated_at: str, now: float = None) -> float:
        """时间近因性（指数衰减）：24h 内 1.0，之后 exp(-(age-24)/72)。"""
        try:
            age_hours = (now or time.time()) - datetime.fromisoformat(updated_at).timestamp()
            age_hours = max(0.0, age_hours / 3600.0)
        except (ValueError, TypeError):
            return 1.0
        if age_hours <= 24.0:
            return 1.0
        return max(0.0, math.exp(-(age_hours - 24.0) / 72.0))

    # ------------------------------------------------------------
    # 1) 创建笔记
    # ------------------------------------------------------------

    def create_note(self, title: str, content: str, note_type: str = "general",
                    tags: Optional[List[str]] = None) -> str:
        """创建笔记，返回操作结果消息。"""
        if not title or not title.strip():
            return "❌ create_note 需要提供 title"
        if note_type not in NOTE_TYPES:
            return (f"❌ 无效的 note_type '{note_type}'，可选: {', '.join(NOTE_TYPES)}")

        note_id = f"note_{int(time.time())}_{len(self.index)}"
        now = datetime.now().isoformat(timespec="seconds")
        metadata = {
            "id": note_id, "title": title.strip(),
            "type": note_type, "tags": list(tags or []),
            "created_at": now, "updated_at": now,
        }
        self._write_file(note_id, metadata, content)
        self.index[note_id] = {**metadata, "path": self._note_path(note_id)}
        self._invalidate_aggregate()   # 新建笔记影响列表/摘要
        return f"✅ 已创建笔记 {note_id}（{metadata['title']}）"

    # ------------------------------------------------------------
    # 2) 读取笔记
    # ------------------------------------------------------------

    def read_note(self, note_id: str) -> str:
        """按 id 读取笔记并解析（60s 缓存）。"""
        if note_id not in self.index:
            return f"❌ 未找到笔记: {note_id}"

        cache_key = f"note:{note_id}"
        cached = self._note_cache.get(cache_key)
        if cached is not None:
            metadata, content = cached
        else:
            parsed = self._read_file(note_id)
            if parsed is None:
                return f"❌ 笔记文件缺失: {note_id}"
            metadata, content = parsed
            self._note_cache.set(cache_key, (metadata, content))

        lines = [
            "=" * 56,
            f"📝 {metadata.get('title', note_id)} [{metadata.get('type', 'general')}]",
            f"   id: {metadata.get('id', note_id)}",
            f"   标签: {', '.join(metadata.get('tags', [])) if metadata.get('tags') else '(无)'}",
            f"   创建: {metadata.get('created_at', '')} | 更新: {metadata.get('updated_at', '')}",
            "=" * 56,
        ]
        if content:
            lines.append("")
            lines.append(content)
        return "\n".join(lines)

    # ------------------------------------------------------------
    # 3) 更新笔记
    # ------------------------------------------------------------

    def update_note(self, note_id: str, title: Optional[str] = None,
                    content: Optional[str] = None,
                    note_type: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> str:
        """更新笔记字段，返回操作结果消息。"""
        if note_id not in self.index:
            return f"❌ 未找到笔记: {note_id}"
        if note_type is not None and note_type not in NOTE_TYPES:
            return (f"❌ 无效的 note_type '{note_type}'，可选: {', '.join(NOTE_TYPES)}")
        if not any(v is not None for v in (title, content, note_type, tags)):
            return "❌ 未提供任何更新字段（title/content/note_type/tags）"

        parsed = self._read_file(note_id)
        if parsed is None:
            return f"❌ 笔记文件缺失: {note_id}"
        metadata, body = parsed

        # 更新参数或字段
        if title is not None:
            metadata["title"] = title.strip()
        if note_type is not None:
            metadata["type"] = note_type
        if tags is not None:
            metadata["tags"] = list(tags)
        if content is not None:
            body = content
        metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")

        # 重构并保存 + 更新索引
        self._write_file(note_id, metadata, body)
        self.index[note_id] = {**metadata, "path": self._note_path(note_id)}
        self._invalidate_note(note_id)
        self._invalidate_aggregate()
        return f"✅ 已更新笔记 {note_id}（{metadata['title']}）"

    # ------------------------------------------------------------
    # 4) 搜索笔记
    # ------------------------------------------------------------

    def find_notes(self, query: str, limit: int = 10,
                   note_type: Optional[str] = None,
                   tags: Optional[List[str]] = None
                   ) -> List[Tuple[float, Dict[str, Any], str]]:
        """原始检索：返回 (score, metadata, content) 列表（供桥接层/内部复用）。

        标题+内容关键词相关性 + 时间近因性混合评分（0.7×rel + 0.3×rec）。
        """
        query = (query or "").strip()
        if not query:
            return []
        limit = max(1, min(50, limit))

        results: List[Tuple[float, Dict[str, Any], str]] = []
        for note_id, meta in self.index.items():
            if note_type and meta.get("type") != note_type:
                continue
            if tags and not set(tags).issubset(set(meta.get("tags", []))):
                continue
            parsed = self._read_file(note_id)
            if parsed is None:
                continue
            _, body = parsed
            text = f"{meta.get('title', '')}\n{body}"
            rel = self._relevance(text, query)
            if rel <= 0.0:
                continue
            rec = self._recency(str(meta.get("updated_at", "")))
            score = 0.7 * rel + 0.3 * rec   # 相关性 + 时间混合
            results.append((score, meta, body))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:limit]

    def search_notes(self, query: str, limit: int = 10,
                     note_type: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> str:
        """搜索笔记：标题+内容关键词，类型/标签过滤，相关性+时间混合排序。"""
        query = (query or "").strip()
        if not query:
            return "❌ search_notes 需要提供 query"
        limit = max(1, min(50, limit))

        # 热门搜索词统计：仅当 query 出现 >= 2 次才启用 60s 缓存
        self._hot_queries[query] += 1
        cache_key = f"search:{query}:{note_type}:{sorted(tags or [])}:{limit}"
        is_hot = self._hot_queries[query] >= 2
        if is_hot:
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                return cached

        # 复用 find_notes 原始检索
        results = self.find_notes(query, limit=limit, note_type=note_type, tags=tags)
        top = results

        lines = [
            "=" * 56,
            f"🔍 笔记搜索: \"{query}\"（命中 {len(results)}，显示 {len(top)}）",
            f"   类型过滤: {note_type or '全部'} | 标签过滤: {tags or '全部'} | "
            f"热门词缓存: {'开' if is_hot else '关'}",
            "=" * 56,
        ]
        for i, (score, meta, _body) in enumerate(top, 1):
            lines.append(f"[{i}] {meta.get('title', '')} ({meta.get('type', '')}) "
                         f"score={score:.3f}")
            lines.append(f"    id: {meta.get('id')} | 更新: {meta.get('updated_at', '')}")
        if not top:
            lines.append("（无匹配笔记）")

        output = "\n".join(lines)
        if is_hot:
            self._search_cache.set(cache_key, output)
        return output

    # ------------------------------------------------------------
    # 5) 列出笔记
    # ------------------------------------------------------------

    def list_notes(self, limit: int = 15, note_type: Optional[str] = None,
                   tags: Optional[List[str]] = None, offset: int = 0) -> str:
        """列出笔记元数据：过滤 + 按更新时间排序 + 分页（15s 缓存）。"""
        limit = max(1, min(100, limit))
        offset = max(0, offset)

        cache_key = f"list:{note_type}:{sorted(tags or [])}:{limit}:{offset}"
        cached = self._list_cache.get(cache_key)
        if cached is not None:
            return cached

        # 过滤
        matched = [meta for meta in self.index.values()
                   if (not note_type or meta.get("type") == note_type)
                   and (not tags or set(tags).issubset(set(meta.get("tags", []))))]
        # 按更新时间排序（降序）
        matched.sort(key=lambda m: str(m.get("updated_at", "")), reverse=True)

        page = matched[offset:offset + limit]
        total = len(matched)
        lines = [
            "=" * 56,
            f"📚 笔记列表（共 {total} 条，显示 {len(page)} 条，offset={offset}）",
            f"   类型过滤: {note_type or '全部'} | 标签过滤: {tags or '全部'}",
            "=" * 56,
        ]
        for meta in page:
            lines.append(f"  {meta.get('id')} | {meta.get('title', '')} "
                         f"[{meta.get('type', '')}] 更新:{meta.get('updated_at', '')}")
        if not page:
            lines.append("（无笔记）")

        output = "\n".join(lines)
        self._list_cache.set(cache_key, output)
        return output

    # ------------------------------------------------------------
    # 6) 生成笔记摘要
    # ------------------------------------------------------------

    def summary(self, recent: int = 5) -> Dict[str, Any]:
        """生成笔记摘要：按类型/标签统计 + 最近 n 条（30s 缓存）。

        Returns:
            {total_notes, type_distribution, tag_distribution,
             recent_notes: [{id, title, type, updated_at}]}
        """
        cache_key = f"summary:{recent}"
        cached = self._summary_cache.get(cache_key)
        if cached is not None:
            return cached

        type_dist: Counter = Counter(m.get("type", "general") for m in self.index.values())
        tag_dist: Counter = Counter()
        for m in self.index.values():
            for t in m.get("tags", []):
                tag_dist[t] += 1

        recent_notes = sorted(
            self.index.values(),
            key=lambda m: str(m.get("updated_at", "")), reverse=True)[:recent]
        result = {
            "total_notes": len(self.index),
            "type_distribution": dict(type_dist),
            "tag_distribution": dict(tag_dist),
            "recent_notes": [
                {"id": m.get("id"), "title": m.get("title", ""),
                 "type": m.get("type", ""), "updated_at": m.get("updated_at", "")}
                for m in recent_notes
            ],
        }
        self._summary_cache.set(cache_key, result)
        return result

    def _summary_action(self, recent: int = 5) -> str:
        """summary 动作：格式化输出摘要。"""
        s = self.summary(recent=recent)
        lines = [
            "=" * 56,
            f"📊 笔记摘要（共 {s['total_notes']} 条）",
            f"   类型分布: {s['type_distribution']}",
            f"   标签分布: {s['tag_distribution']}",
            "   最近笔记:",
        ]
        for n in s["recent_notes"]:
            lines.append(f"     {n['id']} | {n['title']} [{n['type']}] "
                         f"更新:{n['updated_at']}")
        return "\n".join(lines)

    # ------------------------------------------------------------
    # 7) 删除笔记
    # ------------------------------------------------------------

    def delete_note(self, note_id: str) -> str:
        """删除笔记：删除文件并移除索引，返回操作结果消息。"""
        if note_id not in self.index:
            return f"❌ 未找到笔记: {note_id}"
        meta = self.index.pop(note_id)
        path = self._note_path(note_id)
        if os.path.exists(path):
            os.remove(path)
        self._invalidate_note(note_id)
        self._invalidate_aggregate()
        return f"✅ 已删除笔记 {note_id}（{meta.get('title', '')}）"

    # ------------------------------------------------------------
    # Tool 接口
    # ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "NoteTool"

    @property
    def description(self) -> str:
        return (
            "结构化笔记工具（长时程任务外部记忆）：Markdown+YAML 混合格式存储，"
            "支持创建/读取/更新/搜索/列出/摘要/删除笔记。"
            "适用：任务状态、结论、阻塞项、待办、参考资料的跨会话归档与检索。"
            "不适用：临时性短期记忆（请用 MemoryTool）、大规模文档问答（请用 RagTool）、"
            "二进制文件存储。"
            "注意：每则笔记为独立 .md 文件，含 YAML 元数据（id/type/tags/时间戳）。"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="title", type="string",
                          description="笔记标题（create_note 时使用）", required=False),
            ToolParameter(name="content", type="string",
                          description="笔记内容（Markdown，create_note/update_note 时使用）",
                          required=False),
            ToolParameter(name="note_type", type="string",
                          description=f"笔记类型，可选: {', '.join(NOTE_TYPES)}",
                          required=False, enum=list(NOTE_TYPES)),
            ToolParameter(name="tags", type="array",
                          description="标签列表（如 ['rag','进行中']）", required=False,
                          items=ToolParameter(name="tag", type="string",
                                              description="标签", required=False)),
            ToolParameter(name="note_id", type="string",
                          description="笔记 ID（read/update/delete 时使用）", required=False),
            ToolParameter(name="query", type="string",
                          description="搜索关键词（search_notes 时使用）", required=False),
            ToolParameter(name="limit", type="number",
                          description="返回数量（search_notes 默认 10，list_notes 默认 15）",
                          required=False, default=10),
            ToolParameter(name="offset", type="number",
                          description="分页偏移（list_notes 时使用，默认 0）",
                          required=False, default=0),
            ToolParameter(name="recent", type="number",
                          description="摘要展示最近 n 条（summary 时使用，默认 5）",
                          required=False, default=5),
        ]

    def execute(self, action, **kwargs):
        """统一入口：兼容旧协议 execute(action, **kwargs) 与新协议 execute(args)。"""
        return dual_protocol_execute(self, action, **kwargs)

    def run(self, args: Dict[str, Any]) -> str:
        action = (args.get("action") or "").lower()
        if action == "create_note":
            return self.create_note(
                title=args.get("title", ""),
                content=args.get("content", ""),
                note_type=args.get("note_type", "general"),
                tags=args.get("tags") or None)
        elif action == "read_note":
            return self.read_note(args.get("note_id", ""))
        elif action == "update_note":
            return self.update_note(
                note_id=args.get("note_id", ""),
                title=args.get("title"),
                content=args.get("content"),
                note_type=args.get("note_type"),
                tags=args.get("tags"))
        elif action == "search_notes":
            return self.search_notes(
                query=args.get("query", ""),
                limit=int(args.get("limit", 10)),
                note_type=args.get("note_type"),
                tags=args.get("tags"))
        elif action == "list_notes":
            return self.list_notes(
                limit=int(args.get("limit", 15)),
                note_type=args.get("note_type"),
                tags=args.get("tags"),
                offset=int(args.get("offset", 0)))
        elif action == "summary":
            return self._summary_action(recent=int(args.get("recent", 5)))
        elif action == "delete_note":
            return self.delete_note(args.get("note_id", ""))
        return f"❌ 未知操作: '{action}'，当前支持: create_note, read_note, update_note, search_notes, list_notes, summary, delete_note"



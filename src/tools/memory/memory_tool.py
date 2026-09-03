"""
记忆工具 (MemoryTool)
=====================
支持多类型记忆的添加、搜索、摘要、统计、更新、删除、遗忘、整合与清空。
当前实现：add_memory（添加）、search_memory（搜索）、forget_memory（遗忘）、consolidate_memory（整合）

记忆类型:
  - working: 工作记忆（临时信息，默认 importance=0.6）
    * 容量有限（默认50条）+ TTL自动清理（默认60分钟）
    * 混合检索：TF-IDF向量化语义检索 + 关键词匹配回退
    * 评分公式：(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)
  - episodic: 情景记忆（具体事件，默认 importance=0.8）
    * SQLite 持久化存储，支持崩溃恢复
    * 会话级索引，支持按会话检索
    * 混合检索：结构化过滤（时间/会话/重要度） + TF-IDF向量检索
    * 评分公式：(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)
  - semantic: 语义记忆（抽象知识，默认 importance=0.9）
    * 内存知识图谱：实体（Entity）+ 关系（Relation），支持图检索
    * 嵌入向量检索（百炼 text-embedding-v3），API 不可用时自动回退 TF-IDF
    * 轻量级实体/关系抽取（正则+关键词匹配，无外部 NLP 依赖）
    * 混合检索：向量检索（0.7权重） + 知识图谱检索（0.3权重）
    * 评分公式：(向量相似度 × 0.7 + 图谱得分 × 0.3) × (0.8 + 重要性 × 0.4)
  - perceptual: 感知记忆（多模态信息，默认 importance=0.7）
    * 多模态支持：text/image/audio/video/code/document/data
    * 按模态分离的向量存储，非文本模态自动拼接「内容+文件名关键词+元数据标签」编码
    * 跨模态语义检索：嵌入向量优先，API 不可用时回退 TF-IDF
    * 支持 target_modality 模态过滤
    * 评分公式：(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)

遗忘策略:
  - importance: 基于重要性（删除 importance < threshold 的记忆）
  - time: 基于时间（删除超过 max_age_days 天的记忆）
  - capacity: 基于容量（超过 threshold 条时删除最不重要的）
  - mixed: 混合策略（低重要度且超时的记忆被遗忘）
  - dedup: 语义去重（相似度 ≥ threshold 的重复记忆，保留重要度更高的）

整合模式:
  - 类型转换: 将重要的工作记忆转为情景记忆，将重要的情景记忆转为语义记忆
  - 时间窗口: 将过去 N 分钟内的记忆打包整合，窗口结束时主动提醒用户

使用方式:
    tool = MemoryTool()
    tool.execute("add", content="...", memory_type="working")
    tool.execute("search", query="Python", limit=5)
    tool.execute("forget", strategy="importance", threshold=0.3)
    tool.execute("consolidate", from_type="working", to_type="episodic")
    tool.execute("consolidate", time_window_minutes=30)
"""

import sys
import os


import uuid
import re
import math
import json
import base64
import sqlite3
import hashlib
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from src.tools.framework.tool_system import Tool, ToolParameter, dual_protocol_execute

# PostgreSQL 后端（可选，导入失败时回退）
try:
    from src.core.storage import PostgreSQLBackend
    _HAS_PG = True
except ImportError:
    PostgreSQLBackend = None
    _HAS_PG = False
except Exception:
    PostgreSQLBackend = None
    _HAS_PG = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# 嵌入模型客户端（阿里云百炼 DashScope）
# ============================================================

class EmbeddingClient:
    """阿里云百炼多模态嵌入 API 客户端，支持离线回退。

    特点：
      - 调用 DashScope qwen3-vl-embedding 多模态嵌入模型
      - 支持文本 + 图片多模态输入（独立嵌入 / 融合嵌入）
      - 纯文本调用向后兼容 text-embedding-v3 用法
      - MD5 内容缓存，避免重复 API 调用
      - 自动检测 API Key 可用性，不可用时返回 None 触发 fallback
      - 批量嵌入支持，减少网络请求次数
    """

    DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        self._available = self._check_available()
        self._cache: Dict[str, List[float]] = {}

    def _check_available(self) -> bool:
        """检查 API 是否可用（key 非空且非占位符）。"""
        if not self.api_key:
            return False
        placeholders = {"your_key", "your_api_key", "placeholder", "xxx", ""}
        if self.api_key.lower().strip() in placeholders:
            return False
        return True

    @property
    def is_available(self) -> bool:
        return self._available

    def embed(self, text: str = None, images: List[str] = None,
              enable_fusion: bool = False) -> Optional[List[float]]:
        """获取嵌入向量，支持多模态输入（文本 + 图片）。

        向后兼容：embed("text") 纯文本调用仍正常工作。
        多模态：embed(text="描述", images=["img.jpg"]) 生成图文融合嵌入。

        Args:
            text: 文本内容
            images: 图片文件路径列表
            enable_fusion: 是否融合嵌入（将图文融合为一个向量）
        """
        if not self._available:
            return None

        contents = []
        if text:
            contents.append({"text": text})
        if images:
            for img_path in images:
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(img_path)[1].lstrip(".") or "png"
                img_data_uri = f"data:image/{ext};base64,{img_b64}"
                contents.append({"image": img_data_uri})

        if not contents:
            return None

        cache_key = hashlib.md5(
            json.dumps(contents, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            embeddings = self._call_api(contents, enable_fusion=enable_fusion)
            if embeddings:
                result = embeddings[0]  # 返回第一个嵌入（融合时唯一，独立时取文本）
                self._cache[cache_key] = result
                return result
        except Exception:
            pass

        return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量获取嵌入向量，减少 API 调用次数。"""
        if not self._available:
            return [None] * len(texts)

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_texts = []
        uncached_keys = []

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_keys.append((i, cache_key))

        if uncached_texts:
            try:
                embeddings = self._call_api(uncached_texts)
                for (idx, cache_key), emb in zip(uncached_keys, embeddings):
                    if emb:
                        results[idx] = emb
                        self._cache[cache_key] = emb
            except Exception:
                pass

        return results

    def _call_api(self, inputs: Any, embedding_type: str = "query",
                  enable_fusion: bool = False) -> List[List[float]]:
        """调用 DashScope 多模态嵌入 API（qwen3-vl-embedding）。

        自动检测：传入 List[str] 时自动包装为 text content（保持 embed_batch 兼容）；
        传入 List[dict] 时作为 contents 直接使用。

        Args:
            inputs: List[str] 纯文本列表 或 List[dict] contents 结构
            embedding_type: 保留参数，暂未使用
            enable_fusion: 是否融合嵌入
        """
        # 自动检测：字符串列表 → 包装为 text content
        if inputs and isinstance(inputs[0], str):
            contents = [{"text": t} for t in inputs]
        else:
            contents = inputs

        params = {}
        if enable_fusion:
            params["enable_fusion"] = True

        data = json.dumps({
            "model": self.model,
            "input": {"contents": contents},
            "parameters": params,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.DASHSCOPE_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        embeddings = []
        for item in result.get("output", {}).get("embeddings", []):
            embeddings.append(item["embedding"])

        return embeddings

    @staticmethod
    def cosine_similarity_dense(vec_a: List[float], vec_b: List[float]) -> float:
        """计算两个稠密向量的余弦相似度。"""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ============================================================
# 记忆类型默认重要度
# ============================================================

DEFAULT_IMPORTANCE = {
    "working": 0.6,
    "episodic": 0.8,
    "semantic": 0.9,
    "perceptual": 0.7,
}

# 文件扩展名 → 模态映射
FILE_MODALITY_MAP = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".bmp": "image", ".webp": "image",
    ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".aac": "audio",
    ".py": "code", ".js": "code", ".java": "code", ".cpp": "code", ".go": "code", ".rs": "code",
    ".txt": "text", ".md": "text", ".pdf": "document", ".doc": "document", ".docx": "document",
    ".csv": "data", ".json": "data", ".xml": "data",
}


# ============================================================
# 记忆条目数据结构
# ============================================================

@dataclass
class MemoryEntry:
    """单条记忆的完整数据结构。"""
    memory_id: str
    content: str
    memory_type: str                          # working/episodic/semantic/perceptual
    importance: float                         # 0.0-1.0
    timestamp: str                            # ISO 格式时间戳
    session_id: str                           # 会话归属
    file_path: Optional[str] = None           # 关联文件路径
    modality: Optional[str] = None            # 模态: text/image/audio/video/code/document/data
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """返回单条记忆的简要描述。"""
        parts = [
            f"[{self.memory_id[:8]}] ({self.memory_type}, {self.importance:.1f})",
            f"  内容: {self.content[:100]}{'...' if len(self.content) > 100 else ''}",
            f"  时间: {self.timestamp}",
            f"  会话: {self.session_id[:8]}",
        ]
        if self.modality:
            parts.append(f"  模态: {self.modality}")
        if self.file_path:
            parts.append(f"  文件: {self.file_path}")
        if self.metadata:
            parts.append(f"  元数据: {self.metadata}")
        return "\n".join(parts)


# ============================================================
# 情景记忆条目数据结构
# ============================================================

@dataclass
class Episode:
    """情景记忆条目，包含会话和上下文信息。"""
    episode_id: str
    session_id: str
    timestamp: str
    content: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 语义记忆：实体与关系数据结构
# ============================================================

@dataclass
class Entity:
    """知识图谱实体。"""
    entity_id: str
    name: str
    entity_type: str          # concept / technology / person / tool / language / other
    memory_ids: List[str] = field(default_factory=list)  # 提及该实体的记忆ID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Relation:
    """知识图谱关系（三元组）。"""
    relation_id: str
    source_entity: str        # 实体 name
    target_entity: str        # 实体 name
    relation_type: str        # is_a / uses / part_of / related_to / co_occurs
    memory_id: str            # 来源记忆ID
    weight: float = 1.0       # 关系强度

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 记忆模块（MemoryModule）
# ============================================================

class MemoryModule:
    """单类型记忆模块基类，负责某一类记忆的存储与管理。"""

    def __init__(self, memory_type: str, max_capacity: int = None):
        self.memory_type = memory_type
        self.max_capacity = max_capacity
        self._memories: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> str:
        """添加记忆，返回状态消息。子类可覆写以实现容量策略。"""
        self._memories.append(entry)
        return ""

    def get_all(self) -> List[MemoryEntry]:
        return list(self._memories)

    def get_count(self) -> int:
        return len(self._memories)

    def clear(self):
        self._memories.clear()

    def set_memories(self, memories: List[MemoryEntry]):
        """直接替换本模块的记忆列表。"""
        self._memories = list(memories)


class WorkingMemory(MemoryModule):
    """工作记忆模块 —— 容量有限 + TTL自动清理 + 混合检索。

    特点：
      - 容量有限（默认50条），超出时淘汰最不重要的
      - TTL自动清理：超过 max_age_minutes 分钟的记忆自动过期
      - 纯内存存储，访问速度极快
      - 混合检索：优先 TF-IDF 向量化语义检索，失败则回退关键词匹配
      - 评分公式：(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)
    """

    def __init__(self, max_capacity: int = 50, max_age_minutes: int = 60):
        super().__init__("working", max_capacity)
        self.max_age_minutes = max_age_minutes

    def add(self, entry: MemoryEntry) -> str:
        """添加工作记忆：先过期清理，再容量管理，最后追加。"""
        self._expire_old_memories()

        evicted_msg = ""
        while len(self._memories) >= self.max_capacity:
            evicted = self._remove_lowest_priority_memory()
            if evicted and not evicted_msg:
                evicted_msg = (f"⚠️ 工作记忆已达容量上限({self.max_capacity})，"
                               f"已淘汰最不重要的记忆: {evicted.content[:40]}...")

        self._memories.append(entry)
        return evicted_msg

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[Tuple[float, float, "MemoryEntry", Dict]]:
        """混合检索：TF-IDF向量化 + 关键词匹配。

        评分算法：
          base_relevance = tfidf_score * 0.7 + keyword_score * 0.3  (有TF-IDF时)
                         = keyword_score                             (TF-IDF失败时)
          time_decay = exp(-age_minutes / max_age_minutes)
          importance_weight = 0.8 + importance * 0.4
          final_score = (base_relevance * time_decay) * importance_weight

        Returns:
            [(final_score, base_relevance, memory, details_dict), ...]
            按分数降序排列，截断至 limit 条。
        """
        self._expire_old_memories()

        if not self._memories:
            return []

        # 尝试 TF-IDF 向量检索
        vector_scores = self._try_tfidf_search(query)

        # 计算综合分数
        scored_memories = []
        for memory in self._memories:
            vector_score = vector_scores.get(memory.memory_id, 0.0)
            keyword_score = self._calculate_keyword_score(query, memory.content)

            # 混合评分：有TF-IDF分数时加权融合，否则纯关键词
            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
                method = "hybrid"
            else:
                base_relevance = keyword_score
                method = "keyword"

            # 时间衰减
            time_decay = self._calculate_time_decay(memory.timestamp)

            # 重要性权重
            importance_weight = 0.8 + (memory.importance * 0.4)

            # 最终分数：(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)
            final_score = base_relevance * time_decay * importance_weight

            if final_score > 0:
                details = {
                    "method": method,
                    "tfidf_score": vector_score,
                    "keyword_score": keyword_score,
                    "time_decay": time_decay,
                    "importance_weight": importance_weight,
                    "base_relevance": base_relevance,
                    "final_score": final_score,
                }
                scored_memories.append((final_score, base_relevance, memory, details))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return scored_memories[:limit]

    # ============================================================
    # 过期清理与容量管理
    # ============================================================

    def _expire_old_memories(self):
        """清理超过 TTL 的过期记忆。"""
        if self.max_age_minutes is None or self.max_age_minutes <= 0:
            return

        now = datetime.now()
        kept = []
        for mem in self._memories:
            try:
                mem_time = datetime.fromisoformat(mem.timestamp)
                age_minutes = (now - mem_time).total_seconds() / 60
                if age_minutes < self.max_age_minutes:
                    kept.append(mem)
            except (ValueError, TypeError):
                kept.append(mem)
        self._memories = kept

    def _remove_lowest_priority_memory(self) -> Optional[MemoryEntry]:
        """淘汰重要度最低的记忆（相同则淘汰最早的）。"""
        if not self._memories:
            return None
        self._memories.sort(key=lambda m: (m.importance, m.timestamp))
        return self._memories.pop(0)

    # ============================================================
    # TF-IDF 语义检索
    # ============================================================

    def _try_tfidf_search(self, query: str) -> Dict[str, float]:
        """尝试 TF-IDF 向量化语义检索，失败则返回空字典（触发关键词回退）。

        Returns:
            {memory_id: cosine_similarity} 映射，分数范围 0.0-1.0。
        """
        try:
            if len(self._memories) < 2:
                return {}

            query_tokens = self._tokenize_for_tfidf(query)
            if not query_tokens:
                return {}

            # 构建文档集合
            documents = [self._tokenize_for_tfidf(mem.content) for mem in self._memories]

            # 文档频率
            df = Counter()
            for doc_tokens in documents:
                for token in set(doc_tokens):
                    df[token] += 1

            # IDF（平滑处理）
            total_docs = len(documents)
            idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1
                   for token, freq in df.items()}

            # 查询向量
            query_tf = Counter(query_tokens)
            query_len = len(query_tokens)
            query_vector = {token: (count / query_len) * idf.get(token, 0.0)
                            for token, count in query_tf.items()}

            if all(v == 0 for v in query_vector.values()):
                return {}

            # 逐文档计算余弦相似度
            scores = {}
            for mem, doc_tokens in zip(self._memories, documents):
                doc_len = len(doc_tokens) if doc_tokens else 1
                doc_tf = Counter(doc_tokens)
                doc_vector = {token: (count / doc_len) * idf.get(token, 0.0)
                              for token, count in doc_tf.items()}

                similarity = self._cosine_similarity(query_vector, doc_vector)
                if similarity > 0:
                    scores[mem.memory_id] = similarity

            return scores
        except Exception:
            return {}

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """计算两个稀疏向量的余弦相似度。"""
        if not vec_a or not vec_b:
            return 0.0

        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def _tokenize_for_tfidf(text: str) -> List[str]:
        """TF-IDF 专用分词：英文按单词，中文按整词 + 2-gram 展开。"""
        tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text.lower())
        expanded = []
        for token in tokens:
            if token and '\u4e00' <= token[0] <= '\u9fff' and len(token) >= 2:
                expanded.append(token)
                for i in range(len(token) - 1):
                    expanded.append(token[i:i + 2])
            elif token:
                expanded.append(token)
        return expanded

    # ============================================================
    # 关键词匹配
    # ============================================================

    @staticmethod
    def _calculate_keyword_score(query: str, content: str) -> float:
        """关键词匹配评分（0.0-1.0）：Jaccard相似度 + 子串匹配加成。"""
        query_tokens = set(re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', query.lower()))
        content_tokens = set(re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', content.lower()))

        if not query_tokens or not content_tokens:
            return 0.0

        # 中文 2-gram 展开
        def expand(tokens):
            expanded = set()
            for t in tokens:
                if t and '\u4e00' <= t[0] <= '\u9fff' and len(t) >= 2:
                    expanded.add(t)
                    for i in range(len(t) - 1):
                        expanded.add(t[i:i + 2])
                else:
                    expanded.add(t)
            return expanded

        q_expanded = expand(query_tokens)
        c_expanded = expand(content_tokens)

        intersection = q_expanded & c_expanded
        union = q_expanded | c_expanded
        jaccard = len(intersection) / len(union) if union else 0.0

        bonus = 0.15 if query.lower() in content.lower() else 0.0
        return min(1.0, jaccard + bonus)

    # ============================================================
    # 时间衰减
    # ============================================================

    def _calculate_time_decay(self, timestamp: str) -> float:
        """指数时间衰减：decay = exp(-age_minutes / max_age_minutes)。

        新记忆 ≈ 1.0，接近TTL的记忆 ≈ 1/e ≈ 0.368。
        """
        try:
            mem_time = datetime.fromisoformat(timestamp)
            age_minutes = (datetime.now() - mem_time).total_seconds() / 60
        except (ValueError, TypeError):
            return 0.5

        if self.max_age_minutes is None or self.max_age_minutes <= 0:
            return 1.0

        decay = math.exp(-age_minutes / self.max_age_minutes)
        return max(0.0, min(1.0, decay))


class EpisodicMemory(MemoryModule):
    """情景记忆模块 —— SQLite / PostgreSQL 持久化 + 会话索引 + 混合/向量检索。

    特点：
      - SQLite 持久化存储（默认），支持崩溃恢复
      - PostgreSQL 可选（JSONB + pgvector），支持向量级联检索
      - 会话级索引（sessions），支持按会话检索
      - 混合检索：结构化过滤 + TF-IDF 语义向量检索（SQLite）
      - pgvector 余弦距离检索（PostgreSQL）
      - 评分公式：(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)
    """

    def __init__(self, db_path: str = ":memory:", pg_backend=None):
        super().__init__("episodic", None)
        self._pg_backend = pg_backend
        self.sessions: Dict[str, List[str]] = {}  # session_id -> [episode_id]

        if self._pg_backend:
            # PostgreSQL 模式：直接从 pg_backend 管理
            self.db_path = None
            self._conn = None
            self._embedding_client = EmbeddingClient()
        else:
            # SQLite 模式（默认）
            self.db_path = db_path
            self._conn: Optional[sqlite3.Connection] = None
            self._init_db()
            self._load_from_db()

    # ============================================================
    # SQLite 持久化
    # ============================================================

    def _init_db(self):
        """初始化 SQLite 数据库表。"""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id   TEXT PRIMARY KEY,
                session_id   TEXT,
                timestamp    TEXT,
                content      TEXT,
                importance   REAL,
                memory_type  TEXT,
                modality     TEXT,
                file_path    TEXT,
                metadata     TEXT
            )
        """)
        self._conn.commit()

    def _load_from_db(self):
        """从 SQLite 加载已有记忆到内存（启动恢复）。"""
        cursor = self._conn.execute("SELECT * FROM episodes")
        for row in cursor:
            entry = MemoryEntry(
                memory_id=row[0],
                content=row[3],
                memory_type=row[5] or "episodic",
                importance=row[4] if row[4] is not None else 0.8,
                timestamp=row[2],
                session_id=row[1] or "default",
                file_path=row[7],
                modality=row[6],
                metadata=json.loads(row[8]) if row[8] else {},
            )
            self._memories.append(entry)
            self._index_session(entry)
        self._conn.commit()

    def _persist_episode(self, entry: MemoryEntry):
        """将单条记忆写入 SQLite。"""
        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, session_id, timestamp, content, importance,
                memory_type, modality, file_path, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry.memory_id, entry.session_id, entry.timestamp,
             entry.content, entry.importance, entry.memory_type,
             entry.modality, entry.file_path, json.dumps(entry.metadata, ensure_ascii=False))
        )
        self._conn.commit()

    def _flush_db(self):
        """清空 SQLite 并按当前 _memories 重新写入（用于批量同步）。"""
        self._conn.execute("DELETE FROM episodes")
        for mem in self._memories:
            self._conn.execute(
                """INSERT OR REPLACE INTO episodes
                   (episode_id, session_id, timestamp, content, importance,
                    memory_type, modality, file_path, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mem.memory_id, mem.session_id, mem.timestamp,
                 mem.content, mem.importance, mem.memory_type,
                 mem.modality, mem.file_path, json.dumps(mem.metadata, ensure_ascii=False))
            )
        self._conn.commit()

    def close(self):
        """关闭数据库连接。"""
        if self._pg_backend:
            self._pg_backend.close()
        elif self._conn:
            self._conn.close()
            self._conn = None

    # ============================================================
    # 会话索引
    # ============================================================

    def _index_session(self, entry: MemoryEntry):
        """将会话 ID 加入索引。"""
        sid = entry.session_id
        if sid not in self.sessions:
            self.sessions[sid] = []
        self.sessions[sid].append(entry.memory_id)

    def _rebuild_session_index(self):
        """根据 _memories 重建会话索引。"""
        self.sessions.clear()
        for mem in self._memories:
            self._index_session(mem)

    def get_session_episodes(self, session_id: str) -> List[MemoryEntry]:
        """获取指定会话的所有记忆。"""
        ids = set(self.sessions.get(session_id, []))
        return [m for m in self._memories if m.memory_id in ids]

    # ============================================================
    # 增删操作（覆写基类）
    # ============================================================

    def add(self, entry: MemoryEntry) -> str:
        """添加情景记忆：内存 + 会话索引 + 持久化。"""
        self._memories.append(entry)
        self._index_session(entry)

        if self._pg_backend:
            self._pg_add(entry)
        else:
            self._persist_episode(entry)
        return ""

    def _pg_add(self, entry: MemoryEntry):
        """PostgreSQL 模式：写入 episodes 表（JSONB metadata + pgvector embedding）。"""
        # 生成向量嵌入
        vec = self._embedding_client.embed(entry.content)
        vec_str = "[" + ",".join(str(v) for v in vec) + "]" if vec else None

        embedding_cast = "::vector" if self._pg_backend._has_pgvector else ""

        self._pg_backend.execute(
            f"""INSERT INTO episodes (episode_id, session_id, timestamp, content,
               importance, memory_type, modality, file_path, metadata, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s{embedding_cast})""",
            (entry.memory_id, entry.session_id, entry.timestamp,
             entry.content, entry.importance, entry.memory_type,
             entry.modality, entry.file_path,
             json.dumps(entry.metadata, ensure_ascii=False),
             vec_str)
        )

    def clear(self):
        """清空所有记忆（内存 + 数据库）。"""
        self._memories.clear()
        self.sessions.clear()
        if self._pg_backend:
            self._pg_backend.execute("DELETE FROM episodes")
        else:
            self._conn.execute("DELETE FROM episodes")
            self._conn.commit()

    def set_memories(self, memories: List[MemoryEntry]):
        """直接替换记忆列表，同步数据库和会话索引。"""
        self._memories = list(memories)
        self._rebuild_session_index()
        if self._pg_backend:
            self._pg_backend.execute("DELETE FROM episodes")
            for mem in self._memories:
                self._pg_add(mem)
        else:
            self._flush_db()

    # ============================================================
    # 混合检索
    # ============================================================

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[Tuple[float, float, "MemoryEntry", Dict]]:
        """混合检索：结构化过滤 + TF-IDF / pgvector 语义向量检索。

        PostgreSQL 模式使用 pgvector 余弦距离检索；
        SQLite 模式使用 TF-IDF 余弦相似度检索。

        评分算法：
          vector_score  = TF-IDF 余弦相似度 / pgvector 余弦距离转换
          recency_score = exp(-age_days / 30)     # 30天半衰期
          base_relevance = vector_score × 0.8 + recency_score × 0.2
          importance_weight = 0.8 + importance × 0.4
          final_score = base_relevance × importance_weight

        Returns:
            [(final_score, base_relevance, memory, details_dict), ...]
        """
        if self._pg_backend:
            return self._pgvector_retrieve(query, limit, **kwargs)

        if not self._memories:
            return []

        # 1. 结构化预过滤
        candidates = self._structured_filter(**kwargs)
        if not candidates:
            return []

        # 2. TF-IDF 向量语义检索
        vector_scores = self._vector_search(query, candidates)
        tfidf_available = len(vector_scores) > 0

        # 3. 综合评分与排序
        scored = []
        for mem in candidates:
            keyword_score = WorkingMemory._calculate_keyword_score(query, mem.content)

            if tfidf_available:
                vec_score = vector_scores.get(mem.memory_id, 0.0)
                if vec_score > 0:
                    method = "hybrid"
                    effective_vec = vec_score
                else:
                    method = "keyword"
                    effective_vec = keyword_score
            else:
                method = "keyword"
                effective_vec = keyword_score

            # 时间近因性
            recency_score = self._calculate_recency(mem.timestamp)

            # base_relevance = vector_score × 0.8 + recency × 0.2
            base_relevance = effective_vec * 0.8 + recency_score * 0.2

            # importance_weight = 0.8 + importance × 0.4
            importance_weight = 0.8 + (mem.importance * 0.4)

            # final_score
            final_score = base_relevance * importance_weight

            if final_score > 0:
                details = {
                    "method": method,
                    "vector_score": effective_vec,
                    "tfidf_score": vec_score if tfidf_available else 0.0,
                    "keyword_score": keyword_score,
                    "recency_score": recency_score,
                    "importance_weight": importance_weight,
                    "base_relevance": base_relevance,
                    "final_score": final_score,
                }
                scored.append((final_score, base_relevance, mem, details))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    def _pgvector_retrieve(self, query: str, limit: int,
                           **kwargs) -> List[Tuple[float, float, "MemoryEntry", Dict]]:
        """PostgreSQL pgvector 模式检索：使用余弦距离运算符 <=>。

        直接从数据库级向量检索，返回按余弦距离升序排列的结果。
        """
        vec = self._embedding_client.embed(query)
        if not vec:
            return []

        session_id = kwargs.get("session_id")
        rows = self._pg_backend.vector_search(vec, limit=limit,
                                                session_id=session_id)
        if not rows:
            return []

        scored = []
        for row in rows:
            # 余弦距离 → 相似度转换（distance 0 = 完全相同, 2 = 完全相反）
            distance = row.get("distance", 1.0)
            vector_score = max(0.0, 1.0 - distance)

            mem = MemoryEntry(
                memory_id=row["episode_id"],
                content=row["content"],
                memory_type=row.get("memory_type") or "episodic",
                importance=row.get("importance", 0.8),
                timestamp=row.get("timestamp", ""),
                session_id=row.get("session_id", "default"),
                file_path=row.get("file_path", ""),
                modality=row.get("modality", ""),
                metadata=row.get("metadata", {}),
            )

            recency_score = self._calculate_recency(mem.timestamp)
            base_relevance = vector_score * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (mem.importance * 0.4)
            final_score = base_relevance * importance_weight

            if final_score > 0:
                details = {
                    "method": "pgvector",
                    "vector_score": vector_score,
                    "pgvector_distance": distance,
                    "recency_score": recency_score,
                    "importance_weight": importance_weight,
                    "base_relevance": base_relevance,
                    "final_score": final_score,
                }
                scored.append((final_score, base_relevance, mem, details))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    # ============================================================
    # 结构化预过滤
    # ============================================================

    def _structured_filter(self, **kwargs) -> List[MemoryEntry]:
        """结构化预过滤：时间范围 / 会话 / 重要度。"""
        candidates = list(self._memories)

        # 时间范围
        time_start = kwargs.get("time_start")
        time_end = kwargs.get("time_end")
        if time_start or time_end:
            candidates = [m for m in candidates
                          if self._in_time_range(m.timestamp, time_start, time_end)]

        # 会话过滤
        session_id = kwargs.get("session_id")
        if session_id:
            candidates = [m for m in candidates if m.session_id == session_id]

        # 重要度过滤
        min_importance = kwargs.get("min_importance")
        if min_importance is not None:
            candidates = [m for m in candidates if m.importance >= min_importance]

        return candidates

    @staticmethod
    def _in_time_range(timestamp: str, time_start: Any, time_end: Any) -> bool:
        """判断时间戳是否在指定范围内。"""
        try:
            ts = datetime.fromisoformat(timestamp)
        except (ValueError, TypeError):
            return True  # 无法解析时间的保留

        if time_start:
            try:
                start = datetime.fromisoformat(str(time_start)) if isinstance(time_start, str) else time_start
                if ts < start:
                    return False
            except (ValueError, TypeError):
                pass

        if time_end:
            try:
                end = datetime.fromisoformat(str(time_end)) if isinstance(time_end, str) else time_end
                if ts > end:
                    return False
            except (ValueError, TypeError):
                pass

        return True

    # ============================================================
    # TF-IDF 向量检索（复用 WorkingMemory 静态方法）
    # ============================================================

    def _vector_search(self, query: str, candidates: List[MemoryEntry]) -> Dict[str, float]:
        """TF-IDF 向量语义检索，返回 {memory_id: cosine_similarity}。"""
        try:
            if len(candidates) < 2:
                return {}

            query_tokens = WorkingMemory._tokenize_for_tfidf(query)
            if not query_tokens:
                return {}

            documents = [WorkingMemory._tokenize_for_tfidf(mem.content) for mem in candidates]

            # 文档频率
            df = Counter()
            for doc_tokens in documents:
                for token in set(doc_tokens):
                    df[token] += 1

            # IDF（平滑）
            total_docs = len(documents)
            idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1
                   for token, freq in df.items()}

            # 查询向量
            query_tf = Counter(query_tokens)
            query_len = len(query_tokens)
            query_vector = {token: (count / query_len) * idf.get(token, 0.0)
                            for token, count in query_tf.items()}

            if all(v == 0 for v in query_vector.values()):
                return {}

            # 逐文档余弦相似度
            scores = {}
            for mem, doc_tokens in zip(candidates, documents):
                doc_len = len(doc_tokens) if doc_tokens else 1
                doc_tf = Counter(doc_tokens)
                doc_vector = {token: (count / doc_len) * idf.get(token, 0.0)
                              for token, count in doc_tf.items()}

                similarity = WorkingMemory._cosine_similarity(query_vector, doc_vector)
                if similarity > 0:
                    scores[mem.memory_id] = similarity

            return scores
        except Exception:
            return {}

    # ============================================================
    # 时间近因性
    # ============================================================

    @staticmethod
    def _calculate_recency(timestamp: str) -> float:
        """时间近因性评分（0.0-1.0）。

        指数衰减：recency = exp(-age_days / 30)
        30天半衰期，适合情景记忆的长周期特性。
        """
        try:
            mem_time = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - mem_time).total_seconds() / 86400
        except (ValueError, TypeError):
            return 0.5

        return max(0.0, min(1.0, math.exp(-age_days / 30.0)))


class SemanticMemory(MemoryModule):
    """语义记忆模块 —— 知识图谱 + 嵌入向量/TF-IDF检索 + 混合排序。

    特点：
      - 内存知识图谱：实体（Entity）+ 关系（Relation），支持图检索
      - 嵌入向量检索（百炼 text-embedding-v3，API 不可用时自动回退 TF-IDF）
      - 轻量级实体/关系抽取（正则 + 关键词匹配，无外部 NLP 依赖）
      - 混合检索：向量检索（0.7权重） + 图检索（0.3权重）
      - 评分公式：base_relevance = vector_score × 0.7 + graph_score × 0.3
                  importance_weight = 0.8 + importance × 0.4
                  combined_score = base_relevance × importance_weight
    """

    # 已知技术词汇表（用于实体类型推断）
    _TECH_KEYWORDS = {
        "python": "language", "java": "language", "javascript": "language",
        "golang": "language", "go": "language", "rust": "language",
        "c++": "language", "c#": "language", "ruby": "language",
        "ai": "concept", "人工智能": "concept", "ml": "concept",
        "机器学习": "concept", "深度学习": "concept", "nlp": "concept",
        "自然语言处理": "concept", "llm": "concept", "大模型": "concept",
        "tensorflow": "tool", "pytorch": "tool", "numpy": "tool",
        "pandas": "tool", "scikit-learn": "tool", "docker": "tool",
        "kubernetes": "tool", "git": "tool",
        "react": "technology", "vue": "technology", "angular": "technology",
        "flask": "technology", "django": "technology", "fastapi": "technology",
        "redis": "technology", "mysql": "technology", "postgresql": "technology",
        "mongodb": "technology", "elasticsearch": "technology",
    }

    def __init__(self):
        super().__init__("semantic", None)
        self.entities: Dict[str, Entity] = {}       # name -> Entity
        self.relations: List[Relation] = []
        self._entity_index: Dict[str, set] = {}     # memory_id -> {entity_name, ...}
        self._embedding_client = EmbeddingClient()   # 百炼嵌入客户端
        self._embeddings: Dict[str, List[float]] = {}  # memory_id -> embedding vector

    # ============================================================
    # 增删操作（覆写基类）
    # ============================================================

    def add(self, entry: MemoryEntry) -> str:
        """添加语义记忆：存储 + 嵌入向量 + 抽取实体/关系 + 构建知识图谱。"""
        self._memories.append(entry)

        # 生成嵌入向量（API 不可用时跳过，检索时回退 TF-IDF）
        embedding = self._embedding_client.embed(entry.content)
        if embedding:
            self._embeddings[entry.memory_id] = embedding

        # 抽取实体和关系
        entity_names = self._extract_entities(entry.content)
        relations = self._extract_relations(entry.content, entity_names)

        # 注册实体
        matched_names = set()
        for name in entity_names:
            self._register_entity(name, entry.memory_id)
            matched_names.add(name)

        self._entity_index[entry.memory_id] = matched_names

        # 注册关系
        for rel in relations:
            self.relations.append(rel)

        # 共现关系：同一记忆中的所有实体对
        entity_list = list(matched_names)
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                self.relations.append(Relation(
                    relation_id=str(uuid.uuid4()),
                    source_entity=entity_list[i],
                    target_entity=entity_list[j],
                    relation_type="co_occurs",
                    memory_id=entry.memory_id,
                    weight=0.5,
                ))

        return ""

    def clear(self):
        """清空所有记忆和知识图谱。"""
        self._memories.clear()
        self.entities.clear()
        self._embeddings.clear()
        self.relations.clear()
        self._entity_index.clear()

    def set_memories(self, memories: List[MemoryEntry]):
        """直接替换记忆列表，重建知识图谱和嵌入向量。"""
        self._memories = list(memories)
        self.entities.clear()
        self.relations.clear()
        self._entity_index.clear()
        self._embeddings.clear()

        # 批量生成嵌入向量
        if memories and self._embedding_client.is_available:
            texts = [m.content for m in memories]
            embeddings = self._embedding_client.embed_batch(texts)
            for mem, emb in zip(memories, embeddings):
                if emb:
                    self._embeddings[mem.memory_id] = emb

        for entry in memories:
            entity_names = self._extract_entities(entry.content)
            relations = self._extract_relations(entry.content, entity_names)
            matched = set()
            for name in entity_names:
                self._register_entity(name, entry.memory_id)
                matched.add(name)
            self._entity_index[entry.memory_id] = matched
            self.relations.extend(relations)
            entity_list = list(matched)
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    self.relations.append(Relation(
                        relation_id=str(uuid.uuid4()),
                        source_entity=entity_list[i],
                        target_entity=entity_list[j],
                        relation_type="co_occurs",
                        memory_id=entry.memory_id,
                        weight=0.5,
                    ))

    # ============================================================
    # 实体与关系抽取
    # ============================================================

    def _extract_entities(self, content: str) -> List[str]:
        """轻量级实体抽取：技术词汇 + 中文专有名词 + 英文专有名词。

        策略：
          1. 匹配已知技术关键词表
          2. 匹配英文大写开头的词（专有名词）
          3. 匹配中文特定模式（X是Y、X使用Y等结构中的X/Y）
          4. 匹配引号内的内容
        """
        entities = []
        content_lower = content.lower()

        # 1. 已知技术关键词
        for keyword, etype in self._TECH_KEYWORDS.items():
            if keyword in content_lower:
                entities.append(keyword)

        # 2. 英文专有名词（大写开头，≥2字符）
        proper_nouns = re.findall(r'\b([A-Z][a-zA-Z]{2,})\b', content)
        for noun in proper_nouns:
            if noun.lower() not in [e.lower() for e in entities]:
                entities.append(noun)

        # 3. 中文模式抽取：「X是Y」「X使用Y」「X包含Y」「X属于Y」
        cn_patterns = [
            r'([\u4e00-\u9fff]{2,8})是(?:一种|一个|一类)?',
            r'([\u4e00-\u9fff]{2,8})使用',
            r'([\u4e00-\u9fff]{2,8})包含',
            r'([\u4e00-\u9fff]{2,8})属于',
            r'([\u4e00-\u9fff]{2,8})支持',
            r'([\u4e00-\u9fff]{2,8})提供了',
        ]
        for pattern in cn_patterns:
            matches = re.findall(pattern, content)
            for m in matches:
                if m not in entities and len(m) >= 2:
                    entities.append(m)

        # 提取「是Y」中的Y
        is_target = re.findall(r'是(?:一种|一个|一类)?([\u4e00-\u9fff]{2,8})', content)
        for m in is_target:
            if m not in entities:
                entities.append(m)

        # 4. 引号内的内容
        quoted = re.findall(r'[「"\'](.+?)[」"\']', content)
        for q in quoted:
            if q not in entities and len(q) >= 2:
                entities.append(q)

        return entities

    def _extract_relations(self, content: str, entity_names: List[str]) -> List[Relation]:
        """轻量级关系抽取：基于模式匹配提取三元组。

        支持的关系模式：
          - is_a:    X是Y / X是一种Y
          - uses:    X使用Y / X利用Y
          - part_of: X包含Y / X是Y的一部分
          - related_to: X和Y / X与Y
        """
        relations = []
        if len(entity_names) < 2:
            return relations

        entity_set = set(e.lower() for e in entity_names)
        content_lower = content.lower()

        # is_a 关系：「X是Y」
        for src in entity_names:
            pattern = re.escape(src) + r'是(?:一种|一个|一类)?(.{2,20})'
            match = re.search(pattern, content)
            if match:
                target_text = match.group(1).strip()
                for tgt in entity_names:
                    if tgt != src and (tgt in target_text or target_text.startswith(tgt)):
                        relations.append(Relation(
                            relation_id=str(uuid.uuid4()),
                            source_entity=src,
                            target_entity=tgt,
                            relation_type="is_a",
                            memory_id="",  # 调用方会在 add 中补充
                            weight=1.0,
                        ))

        # uses 关系：「X使用Y」「X利用Y」
        for src in entity_names:
            for verb in ['使用', '利用', '用']:
                pattern = re.escape(src) + verb + r'(.{2,20})'
                match = re.search(pattern, content)
                if match:
                    target_text = match.group(1).strip()
                    for tgt in entity_names:
                        if tgt != src and (tgt in target_text or target_text.startswith(tgt)):
                            relations.append(Relation(
                                relation_id=str(uuid.uuid4()),
                                source_entity=src,
                                target_entity=tgt,
                                relation_type="uses",
                                memory_id="",
                                weight=1.0,
                            ))

        # related_to 关系：「X和Y」「X与Y」
        for i, src in enumerate(entity_names):
            for j, tgt in enumerate(entity_names):
                if i >= j:
                    continue
                pattern = re.escape(src) + r'[与和].{0,5}' + re.escape(tgt)
                if re.search(pattern, content):
                    relations.append(Relation(
                        relation_id=str(uuid.uuid4()),
                        source_entity=src,
                        target_entity=tgt,
                        relation_type="related_to",
                        memory_id="",
                        weight=0.8,
                    ))

        return relations

    def _register_entity(self, name: str, memory_id: str):
        """注册实体到知识图谱。"""
        key = name.lower()
        if key not in self.entities:
            etype = self._infer_entity_type(name)
            self.entities[key] = Entity(
                entity_id=str(uuid.uuid4()),
                name=name,
                entity_type=etype,
                memory_ids=[],
            )
        if memory_id not in self.entities[key].memory_ids:
            self.entities[key].memory_ids.append(memory_id)

    @classmethod
    def _infer_entity_type(cls, name: str) -> str:
        """推断实体类型。"""
        name_lower = name.lower()
        if name_lower in cls._TECH_KEYWORDS:
            return cls._TECH_KEYWORDS[name_lower]
        if re.match(r'^[A-Z][a-zA-Z]+$', name):
            return "person"  # 英文专有名词默认为人物/组织
        return "concept"

    # ============================================================
    # 混合检索
    # ============================================================

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[Tuple[float, float, "MemoryEntry", Dict]]:
        """混合检索：嵌入向量/TF-IDF向量检索 + 知识图谱检索。

        评分算法：
          vector_score  = 嵌入向量余弦相似度（API可用时）/ TF-IDF余弦相似度（回退）
          graph_score   = 实体重叠度 + 关系连接度
          base_relevance = vector_score × 0.7 + graph_score × 0.3
          importance_weight = 0.8 + importance × 0.4
          combined_score = base_relevance × importance_weight

        Returns:
            [(combined_score, base_relevance, memory, details_dict), ...]
        """
        if not self._memories:
            return []

        min_importance = kwargs.get("min_importance", 0.0)

        # 1. 向量检索（嵌入优先，TF-IDF 回退）
        vector_scores, vector_method = self._vector_search(query)

        # 2. 图检索
        graph_scores, matched_entities = self._graph_search(query)

        # 3. 混合评分
        scored = []
        for mem in self._memories:
            if mem.importance < min_importance:
                continue

            vec_score = vector_scores.get(mem.memory_id, 0.0)
            graph_score = graph_scores.get(mem.memory_id, 0.0)

            # 如果向量和图都为0，跳过
            if vec_score == 0 and graph_score == 0:
                continue

            base_relevance = vec_score * 0.7 + graph_score * 0.3
            importance_weight = 0.8 + (mem.importance * 0.4)
            combined_score = base_relevance * importance_weight

            if combined_score > 0:
                # 确定检索方式
                if vec_score > 0 and graph_score > 0:
                    method = "hybrid"
                elif vec_score > 0:
                    method = "vector"
                else:
                    method = "graph"

                # 该记忆匹配到的实体
                mem_entities = self._entity_index.get(mem.memory_id, set())
                mem_matched = [e for e in matched_entities if e in mem_entities]

                details = {
                    "method": method,
                    "vector_method": vector_method,  # "embedding" or "tfidf"
                    "vector_score": vec_score,
                    "graph_score": graph_score,
                    "base_relevance": base_relevance,
                    "importance_weight": importance_weight,
                    "combined_score": combined_score,
                    "matched_entities": mem_matched,
                    "entity_count": len(mem_entities),
                }
                scored.append((combined_score, base_relevance, mem, details))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    # ============================================================
    # 向量检索（嵌入优先 + TF-IDF 回退）
    # ============================================================

    def _vector_search(self, query: str) -> Tuple[Dict[str, float], str]:
        """向量语义检索：嵌入向量优先，TF-IDF 回退。

        Returns:
            ({memory_id: similarity}, vector_method)
            vector_method: "embedding" 或 "tfidf"
        """
        # 优先尝试嵌入向量检索
        if self._embedding_client.is_available and self._embeddings:
            query_embedding = self._embedding_client.embed(query)
            if query_embedding:
                scores = {}
                for mem in self._memories:
                    mem_embedding = self._embeddings.get(mem.memory_id)
                    if mem_embedding:
                        similarity = EmbeddingClient.cosine_similarity_dense(
                            query_embedding, mem_embedding
                        )
                        if similarity > 0:
                            scores[mem.memory_id] = similarity
                if scores:
                    return scores, "embedding"

        # 回退到 TF-IDF
        tfidf_scores = self._tfidf_search(query)
        return tfidf_scores, "tfidf"

    def _tfidf_search(self, query: str) -> Dict[str, float]:
        """TF-IDF 向量语义检索（回退方案），返回 {memory_id: cosine_similarity}。"""
        try:
            if len(self._memories) < 1:
                return {}

            query_tokens = WorkingMemory._tokenize_for_tfidf(query)
            if not query_tokens:
                return {}

            documents = [WorkingMemory._tokenize_for_tfidf(mem.content) for mem in self._memories]

            # 文档频率
            df = Counter()
            for doc_tokens in documents:
                for token in set(doc_tokens):
                    df[token] += 1

            # IDF（平滑）
            total_docs = len(documents)
            idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1
                   for token, freq in df.items()}

            # 查询向量
            query_tf = Counter(query_tokens)
            query_len = len(query_tokens)
            query_vector = {token: (count / query_len) * idf.get(token, 0.0)
                            for token, count in query_tf.items()}

            if all(v == 0 for v in query_vector.values()):
                return {}

            # 逐文档余弦相似度
            scores = {}
            for mem, doc_tokens in zip(self._memories, documents):
                doc_len = len(doc_tokens) if doc_tokens else 1
                doc_tf = Counter(doc_tokens)
                doc_vector = {token: (count / doc_len) * idf.get(token, 0.0)
                              for token, count in doc_tf.items()}

                similarity = WorkingMemory._cosine_similarity(query_vector, doc_vector)
                if similarity > 0:
                    scores[mem.memory_id] = similarity

            return scores
        except Exception:
            return {}

    # ============================================================
    # 知识图谱检索
    # ============================================================

    def _graph_search(self, query: str) -> Tuple[Dict[str, float], List[str]]:
        """图检索：从查询中抽取实体，通过实体重叠和关系连接度评分。

        Returns:
            ({memory_id: graph_score}, [matched_entity_names])
        """
        # 从查询中抽取实体
        query_entities = self._extract_entities(query)
        query_entity_keys = set(e.lower() for e in query_entities)

        if not query_entity_keys:
            return {}, []

        # 匹配到的实体名（原始大小写）
        matched_entities = []
        for name in query_entities:
            if name.lower() in self.entities:
                matched_entities.append(name)

        # 计算每条记忆的图得分
        scores: Dict[str, float] = {}

        for mem in self._memories:
            mem_entities = self._entity_index.get(mem.memory_id, set())
            if not mem_entities:
                continue

            # 实体重叠度：查询实体与记忆实体的 Jaccard 相似度
            mem_entity_keys = set(e.lower() for e in mem_entities)
            overlap = query_entity_keys & mem_entity_keys
            union = query_entity_keys | mem_entity_keys
            entity_overlap_score = len(overlap) / len(union) if union else 0.0

            # 关系连接度：通过关系图扩展，检查间接连接
            relation_score = 0.0
            for rel in self.relations:
                if rel.memory_id == mem.memory_id:
                    # 直接关系加分
                    if (rel.source_entity.lower() in query_entity_keys or
                            rel.target_entity.lower() in query_entity_keys):
                        relation_score += rel.weight * 0.3
                else:
                    # 间接关系（其他记忆中的实体与当前记忆实体有连接）
                    src_in_query = rel.source_entity.lower() in query_entity_keys
                    tgt_in_mem = rel.target_entity.lower() in mem_entity_keys
                    tgt_in_query = rel.target_entity.lower() in query_entity_keys
                    src_in_mem = rel.source_entity.lower() in mem_entity_keys

                    if (src_in_query and tgt_in_mem) or (tgt_in_query and src_in_mem):
                        relation_score += rel.weight * 0.15

            # 图得分 = 实体重叠度（0.6权重） + 关系连接度（0.4权重），归一化到 [0, 1]
            relation_score = min(1.0, relation_score)
            graph_score = entity_overlap_score * 0.6 + relation_score * 0.4

            if graph_score > 0:
                scores[mem.memory_id] = graph_score

        return scores, matched_entities

    # ============================================================
    # 知识图谱查询辅助
    # ============================================================

    def get_entity_relations(self, entity_name: str) -> List[Dict[str, Any]]:
        """获取指定实体的所有关系。"""
        key = entity_name.lower()
        result = []
        for rel in self.relations:
            if rel.source_entity.lower() == key or rel.target_entity.lower() == key:
                result.append(rel.to_dict())
        return result

    def get_entity_count(self) -> int:
        return len(self.entities)

    def get_relation_count(self) -> int:
        return len(self.relations)


class PerceptualMemory(MemoryModule):
    """感知记忆模块 —— 多模态存储 + 跨模态语义检索 + 时间/重要性融合。

    特点：
      - 多模态支持：text / image / audio / video / code / document / data
      - 按模态分离的向量存储：vector_stores[modality] = {memory_id: embedding}
      - 语义编码：
          text 模态：直接嵌入 content（百炼 text-embedding-v3）
          非文本模态：将「内容描述 + 文件名关键词 + 元数据标签 + 模态标记」
                     拼接为语义文本再编码，实现跨模态语义匹配
      - 嵌入 API 不可用时自动回退 TF-IDF 检索
      - 跨模态检索：查询编码为语义向量后匹配任意模态的记忆
      - 评分公式：base_relevance = vector_score × 0.8 + recency_score × 0.2
                  importance_weight = 0.8 + importance × 0.4
                  combined_score = base_relevance × importance_weight
    """

    def __init__(self):
        super().__init__("perceptual", None)
        self._embedding_client = EmbeddingClient()          # 百炼嵌入客户端
        # 按模态分离的向量存储：modality -> {memory_id: [float, ...]}
        self.vector_stores: Dict[str, Dict[str, List[float]]] = {}
        # 语义文本缓存：memory_id -> 拼接后的语义文本
        self._semantic_texts: Dict[str, str] = {}

    # ============================================================
    # 增删操作（覆写基类）
    # ============================================================

    def add(self, entry: MemoryEntry) -> str:
        """添加感知记忆：存储 + 构建语义文本 + 编码向量 + 按模态归档。"""
        self._memories.append(entry)

        # 构建语义文本（内容 + 文件名关键词 + 元数据标签 + 模态标记）
        semantic_text = self._build_semantic_text(entry)
        self._semantic_texts[entry.memory_id] = semantic_text

        # 生成嵌入向量（API 不可用时跳过，检索时回退 TF-IDF）
        embedding = self._embedding_client.embed(semantic_text)
        if embedding:
            store = self._ensure_store(entry.modality or "text")
            store[entry.memory_id] = embedding

        return ""

    def clear(self):
        """清空所有记忆和向量存储。"""
        self._memories.clear()
        self.vector_stores.clear()
        self._semantic_texts.clear()

    def set_memories(self, memories: List[MemoryEntry]):
        """直接替换记忆列表，重建向量存储。"""
        self._memories = list(memories)
        self.vector_stores.clear()
        self._semantic_texts.clear()

        # 批量生成嵌入向量
        if memories and self._embedding_client.is_available:
            entries_by_modality: Dict[str, List[MemoryEntry]] = {}
            for mem in memories:
                mod = mem.modality or "text"
                entries_by_modality.setdefault(mod, []).append(mem)

            for mod, mod_memories in entries_by_modality.items():
                semantic_texts = [self._build_semantic_text(m) for m in mod_memories]
                for mem, text in zip(mod_memories, semantic_texts):
                    self._semantic_texts[mem.memory_id] = text
                embeddings = self._embedding_client.embed_batch(semantic_texts)
                store = self._ensure_store(mod)
                for mem, emb in zip(mod_memories, embeddings):
                    if emb:
                        store[mem.memory_id] = emb
        else:
            for mem in memories:
                self._semantic_texts[mem.memory_id] = self._build_semantic_text(mem)

    # ============================================================
    # 语义文本构建与关键词提取
    # ============================================================

    def _ensure_store(self, modality: str) -> Dict[str, List[float]]:
        """获取（必要时创建）指定模态的向量存储。"""
        if modality not in self.vector_stores:
            self.vector_stores[modality] = {}
        return self.vector_stores[modality]

    def _build_semantic_text(self, entry: MemoryEntry) -> str:
        """构建语义文本：内容 + 文件名关键词 + 元数据标签 + 模态标记。"""
        parts = [entry.content or ""]
        if entry.file_path:
            file_kw = self._extract_file_keywords(entry.file_path)
            if file_kw:
                parts.append(" ".join(file_kw))
        # 元数据标签（tags / labels / caption / keywords）
        for key in ("tags", "labels", "caption", "keywords"):
            val = entry.metadata.get(key)
            if val:
                if isinstance(val, list):
                    parts.append(" ".join(str(v) for v in val))
                else:
                    parts.append(str(val))
        parts.append(entry.modality or "text")
        return " ".join(p for p in parts if p)

    @staticmethod
    def _extract_file_keywords(file_path: str) -> List[str]:
        """从文件路径提取关键词：去扩展名 + 按分隔符拆分。"""
        try:
            filename = os.path.basename(file_path)
            name = os.path.splitext(filename)[0]
            # 按常见分隔符拆分（- _ . 空格）
            tokens = re.split(r'[-_.\s]+', name)
            return [t for t in tokens if t and len(t) >= 2][:8]
        except Exception:
            return []

    # ============================================================
    # 时间近因性
    # ============================================================

    @staticmethod
    def _calculate_recency_score(timestamp: str) -> float:
        """计算时间近因性得分（指数衰减）。

        24小时内保持高分，之后逐渐衰减；最低保持 0.1 基础分。
        """
        try:
            memory_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            age_hours = (current_time - memory_time).total_seconds() / 3600
            decay_factor = 0.1  # 衰减系数
            recency_score = math.exp(-decay_factor * age_hours / 24)
            return max(0.1, recency_score)  # 最低保持 0.1 的基础分数
        except Exception:
            return 0.5  # 默认中等分数

    # ============================================================
    # 混合检索
    # ============================================================

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[Tuple[float, float, "MemoryEntry", Dict]]:
        """检索感知记忆：向量检索（跨模态）+ 时间近因性 + 重要性融合。

        kwargs:
          target_modality: 限定目标记忆模态（None=全部模态）
          min_importance: 最低重要度阈值
        """
        if not self._memories:
            return []

        target_modality = kwargs.get("target_modality")
        min_importance = kwargs.get("min_importance", 0.0)

        # 1. 向量检索（嵌入优先，TF-IDF 回退）
        vector_scores, vector_method = self._vector_search(query, target_modality)

        # 2. 融合排序（向量相似度 + 时间近因性 + 重要性权重）
        scored = []
        for mem in self._memories:
            if mem.importance < min_importance:
                continue
            if target_modality and (mem.modality or "text") != target_modality:
                continue

            vec_score = vector_scores.get(mem.memory_id, 0.0)
            recency_score = self._calculate_recency_score(mem.timestamp)

            # 评分公式
            base_relevance = vec_score * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (mem.importance * 0.4)
            combined_score = base_relevance * importance_weight

            if combined_score > 0:
                details = {
                    "method": "hybrid" if vec_score > 0 else "keyword",
                    "vector_method": vector_method,  # "embedding" / "tfidf"
                    "vector_score": vec_score,
                    "recency_score": recency_score,
                    "base_relevance": base_relevance,
                    "importance_weight": importance_weight,
                    "combined_score": combined_score,
                    "modality": mem.modality or "text",
                }
                scored.append((combined_score, base_relevance, mem, details))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    # ============================================================
    # 向量检索（嵌入优先 + TF-IDF 回退）
    # ============================================================

    def _vector_search(self, query: str, target_modality: str = None) -> Tuple[Dict[str, float], str]:
        """向量语义检索：嵌入向量优先，TF-IDF 回退。

        Returns:
            ({memory_id: similarity}, vector_method)
        """
        # 候选记忆池（模态过滤后）
        if target_modality:
            candidate_ids = {
                mem.memory_id for mem in self._memories
                if (mem.modality or "text") == target_modality
            }
        else:
            candidate_ids = {mem.memory_id for mem in self._memories}

        if not candidate_ids:
            return {}, "tfidf"

        # 优先尝试嵌入向量检索
        if self._embedding_client.is_available and self.vector_stores:
            query_embedding = self._embedding_client.embed(query)
            if query_embedding:
                scores = {}
                for store in self.vector_stores.values():
                    for memory_id, vec in store.items():
                        if memory_id not in candidate_ids:
                            continue
                        similarity = EmbeddingClient.cosine_similarity_dense(
                            query_embedding, vec
                        )
                        if similarity > 0:
                            scores[memory_id] = similarity
                if scores:
                    return scores, "embedding"

        # 回退到 TF-IDF（对语义文本）
        return self._tfidf_search(query, candidate_ids), "tfidf"

    def _tfidf_search(self, query: str, candidate_ids: set) -> Dict[str, float]:
        """TF-IDF 向量语义检索（回退方案），基于语义文本计算。"""
        try:
            if not candidate_ids:
                return {}

            query_tokens = WorkingMemory._tokenize_for_tfidf(query)
            if not query_tokens:
                return {}

            candidates = [m for m in self._memories if m.memory_id in candidate_ids]
            doc_texts = [self._semantic_texts.get(m.memory_id, m.content) for m in candidates]
            documents = [WorkingMemory._tokenize_for_tfidf(t) for t in doc_texts]

            # 文档频率
            df = Counter()
            for doc_tokens in documents:
                for token in set(doc_tokens):
                    df[token] += 1

            # IDF（平滑）
            total_docs = len(documents)
            idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1
                   for token, freq in df.items()}

            # 查询向量
            query_tf = Counter(query_tokens)
            query_len = len(query_tokens)
            query_vector = {token: (count / query_len) * idf.get(token, 0.0)
                            for token, count in query_tf.items()}

            if all(v == 0 for v in query_vector.values()):
                return {}

            # 逐文档余弦相似度
            scores = {}
            for mem, doc_tokens in zip(candidates, documents):
                doc_len = len(doc_tokens) if doc_tokens else 1
                doc_tf = Counter(doc_tokens)
                doc_vector = {token: (count / doc_len) * idf.get(token, 0.0)
                              for token, count in doc_tf.items()}
                similarity = WorkingMemory._cosine_similarity(query_vector, doc_vector)
                if similarity > 0:
                    scores[mem.memory_id] = similarity

            return scores
        except Exception:
            return {}

    # ============================================================
    # 辅助查询
    # ============================================================

    def get_modality_stats(self) -> Dict[str, int]:
        """返回各模态的记忆数量统计。"""
        stats: Dict[str, int] = {}
        for mem in self._memories:
            mod = mem.modality or "text"
            stats[mod] = stats.get(mod, 0) + 1
        return stats


# 模块类映射
MODULE_CLASSES = {
    "working": WorkingMemory,
    "episodic": EpisodicMemory,
    "semantic": SemanticMemory,
    "perceptual": PerceptualMemory,
}


# ============================================================
# MemoryManager —— 记忆操作统一接口
# ============================================================

class MemoryManager:
    """
    记忆管理器：作为 MemoryTool 的底层引擎，
    根据配置启用不同类型的记忆模块，统一调度 add / get / remove 等操作。
    """

    DEFAULT_CONFIG = {
        "enabled_types": ["working", "episodic", "semantic", "perceptual"],
        "working_capacity": 50,
        "working_memory_ttl": 60,
        "episodic_db_path": ":memory:",
    }

    def __init__(self, config: Dict[str, Any] = None):
        config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._config = config
        self._modules: Dict[str, MemoryModule] = {}

        enabled = config["enabled_types"]
        for mem_type in enabled:
            if mem_type not in MODULE_CLASSES:
                continue
            if mem_type == "working":
                self._modules[mem_type] = WorkingMemory(
                    max_capacity=config.get("working_capacity", 50),
                    max_age_minutes=config.get("working_memory_ttl", 60),
                )
            elif mem_type == "episodic":
                self._modules[mem_type] = EpisodicMemory(
                    db_path=config.get("episodic_db_path", ":memory:"),
                )
            else:
                self._modules[mem_type] = MODULE_CLASSES[mem_type]()

    # --- 模块查询 ---

    def get_enabled_types(self) -> List[str]:
        return list(self._modules.keys())

    def is_enabled(self, memory_type: str) -> bool:
        return memory_type in self._modules

    def get_module(self, memory_type: str) -> MemoryModule:
        return self._modules.get(memory_type)

    # --- 统一操作 ---

    def add(self, entry: MemoryEntry) -> str:
        """将记忆添加到对应类型的模块，返回模块的附加消息（如容量淘汰提示）。"""
        module = self._modules.get(entry.memory_type)
        if module is None:
            return (f"❌ 记忆类型 '{entry.memory_type}' 未启用，"
                    f"当前已启用: {list(self._modules.keys())}")
        return module.add(entry)

    def get_all(self) -> List[MemoryEntry]:
        """聚合所有模块的记忆，返回扁平列表。"""
        all_memories = []
        for module in self._modules.values():
            all_memories.extend(module.get_all())
        return all_memories

    def get_count(self) -> int:
        return sum(m.get_count() for m in self._modules.values())

    def get_by_type(self, memory_type: str) -> List[MemoryEntry]:
        module = self._modules.get(memory_type)
        return module.get_all() if module else []

    def replace_all(self, memories: List[MemoryEntry]):
        """清空所有模块，按类型重新分发记忆（用于 forget 操作后回写）。

        使用 set_memories() 而非直接操作 _memories，
        确保各模块（如 EpisodicMemory）能同步持久层和索引。
        """
        by_type: Dict[str, List[MemoryEntry]] = {}
        for mem in memories:
            by_type.setdefault(mem.memory_type, []).append(mem)

        for mem_type, module in self._modules.items():
            module.set_memories(by_type.get(mem_type, []))

    def remove_ids(self, ids: set) -> List[MemoryEntry]:
        """按 ID 集合删除记忆，返回被删除的列表。

        使用 set_memories() 回写，确保持久层同步。
        """
        removed = []
        for module in self._modules.values():
            kept = []
            for mem in module._memories:
                if mem.memory_id in ids:
                    removed.append(mem)
                else:
                    kept.append(mem)
            module.set_memories(kept)
        return removed

    def get_config(self) -> Dict[str, Any]:
        return dict(self._config)


# ============================================================
# LLM 重排序器（TF-IDF 粗筛 + LLM 精排）
# ============================================================

class LLMReranker:
    """TF-IDF 粗筛 + LLM 精排 两阶段检索重排序器。

    流程：
      1. TF-IDF 粗筛：对所有候选记忆计算 TF-IDF 余弦相似度，取 top-k 作为粗筛结果
      2. 关键词快速抽取：从粗筛结果中提取关键词，辅助 LLM 理解重点
      3. LLM 精排：将粗筛候选 + 查询发送给 LLM，按语义相关性重新排序并打分

    使用方式：
        reranker = LLMReranker()
        results = reranker.rerank(memories, query, coarse_limit=15, rerank_limit=5)
        # results: [(MemoryEntry, relevance_score, reason), ...]
    """

    RERANK_PROMPT = """你是一个记忆检索排序助手。请根据用户的查询，对以下候选记忆按相关性从高到低排序。

【查询】
{query}

【候选记忆列表】
{memory_list}

【任务】
分析每条记忆与查询的语义相关性，按相关度从高到低排序。只返回 JSON 数组，格式严格如下：
[{{"index": 原始编号, "relevance": 0.0-1.0的相关性评分, "reason": "10字以内的简短理由"}}, ...]

要求：
- relevance 精确到两位小数，1.0 表示完全匹配，0.0 表示完全不相关
- 只返回 JSON 数组，不要任何其他文字
- 按 relevance 从高到低排列"""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from src.core.llm import practiceLLM
            self._llm = practiceLLM()
        return self._llm

    def rerank(
        self,
        memories: list,
        query: str,
        coarse_limit: int = 15,
        rerank_limit: int = 5,
    ) -> list:
        """两阶段检索：TF-IDF 粗筛 → LLM 精排。

        Args:
            memories: 候选记忆列表（MemoryEntry 对象）
            query: 查询文本
            coarse_limit: TF-IDF 粗筛保留的候选数量
            rerank_limit: LLM 精排后返回的最终数量

        Returns:
            [(MemoryEntry, relevance_score, reason), ...] 按相关性降序排列
        """
        if not memories:
            return []

        # 阶段一：TF-IDF 粗筛
        coarse_results = self._coarse_filter(memories, query, coarse_limit)

        if not coarse_results:
            return []

        # 阶段二：LLM 精排
        llm_results = self._llm_rerank(coarse_results, query, rerank_limit)

        return llm_results

    # ============================================================
    # 阶段一：TF-IDF 粗筛
    # ============================================================

    def _coarse_filter(self, memories: list, query: str, top_k: int) -> list:
        """TF-IDF 粗筛：对每条记忆计算 TF-IDF 余弦相似度，返回 top_k。"""
        try:
            # 分词
            query_tokens = WorkingMemory._tokenize_for_tfidf(query)
            if not query_tokens:
                return [(mem, 0.0) for mem in memories[:top_k]]

            doc_tokens_list = [WorkingMemory._tokenize_for_tfidf(m.content) for m in memories]

            # 文档频率
            df = Counter()
            for tokens in doc_tokens_list:
                for token in set(tokens):
                    df[token] += 1

            total_docs = len(memories)
            idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1
                   for token, freq in df.items()}

            # 查询向量
            query_tf = Counter(query_tokens)
            query_len = len(query_tokens)
            query_vector = {token: (count / query_len) * idf.get(token, 0.0)
                            for token, count in query_tf.items()}

            # 逐文档计算余弦相似度
            scores = []
            for mem, doc_tokens in zip(memories, doc_tokens_list):
                doc_len = len(doc_tokens) if doc_tokens else 1
                doc_tf = Counter(doc_tokens)
                doc_vector = {token: (count / doc_len) * idf.get(token, 0.0)
                              for token, count in doc_tf.items()}
                similarity = WorkingMemory._cosine_similarity(query_vector, doc_vector)
                scores.append((mem, similarity))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        except Exception:
            return [(mem, 0.0) for mem in memories[:top_k]]

    # ============================================================
    # 阶段二：LLM 精排
    # ============================================================

    def _llm_rerank(self, coarse_results: list, query: str, top_k: int) -> list:
        """LLM 精排：将粗筛候选发送给 LLM 进行语义重排序。"""
        # 构建候选记忆列表文本
        memory_lines = []
        for i, (mem, tfidf_score) in enumerate(coarse_results, 1):
            memory_lines.append(
                f"[{i}] (TF-IDF:{tfidf_score:.3f}) {mem.content}"
            )
        memory_list_text = "\n".join(memory_lines)

        prompt = self.RERANK_PROMPT.format(
            query=query,
            memory_list=memory_list_text,
        )

        messages = [
            {"role": "system", "content": "你是一个精确的记忆检索排序助手。只返回 JSON 数组，不输出任何其他内容。"},
            {"role": "user", "content": prompt},
        ]

        try:
            llm = self._get_llm()
            response = llm.invoke(messages, temperature=0, stream=False)

            if not response:
                return self._fallback_rerank(coarse_results, top_k)

            ranked = self._parse_llm_response(response, coarse_results)
            return ranked[:top_k]

        except Exception:
            return self._fallback_rerank(coarse_results, top_k)

    def _parse_llm_response(self, response: str, coarse_results: list) -> list:
        """解析 LLM 返回的 JSON 排序结果。"""
        # 尝试提取 JSON 数组
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return self._fallback_rerank(coarse_results, len(coarse_results))

        try:
            rankings = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return self._fallback_rerank(coarse_results, len(coarse_results))

        if not isinstance(rankings, list):
            return self._fallback_rerank(coarse_results, len(coarse_results))

        results = []
        seen_indices = set()
        for item in rankings:
            idx = item.get("index", -1) - 1  # LLM 返回的编号从 1 开始
            if 0 <= idx < len(coarse_results) and idx not in seen_indices:
                mem, _ = coarse_results[idx]
                relevance = max(0.0, min(1.0, float(item.get("relevance", 0.0))))
                reason = str(item.get("reason", ""))[:20]
                results.append((mem, relevance, reason))
                seen_indices.add(idx)

        return results

    def _fallback_rerank(self, coarse_results: list, top_k: int) -> list:
        """LLM 调用失败时的回退：按 TF-IDF 分排序。"""
        return [(mem, score, "TF-IDF回退") for mem, score in coarse_results[:top_k]]

    # ============================================================
    # 关键词抽取（辅助调试/展示）
    # ============================================================

    @staticmethod
    def extract_keywords(text: str, top_k: int = 5) -> list:
        """从文本中抽取 TF-IDF 关键词，用于辅助展示。"""
        try:
            tokens = WorkingMemory._tokenize_for_tfidf(text)
            if not tokens:
                return []
            tf = Counter(tokens)
            total = len(tokens)
            return [word for word, _ in tf.most_common(top_k)]
        except Exception:
            return []


# ============================================================
# MemoryTool
# ============================================================

class MemoryTool(Tool):
    """
    记忆工具：管理多类型记忆的存储与检索。
    初始化时创建 MemoryManager 实例作为统一操作接口，根据 config 启用不同记忆模块。
    当前支持 action: add, search, forget, consolidate, rerank
    """

    def __init__(self, session_id: str = None, config: Dict[str, Any] = None):
        """
        Args:
            session_id: 指定会话 ID，不传则自动生成
            config: 记忆管理器配置，支持:
                - enabled_types: List[str] 启用的记忆类型，默认全部启用
                - working_capacity: int 工作记忆容量上限，默认 7
        """
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
            ToolParameter(
                name="action",
                type="string",
                description="操作类型: add（添加）、search（搜索）、forget（遗忘）、consolidate（整合）",
                required=True,
                enum=["add", "search", "forget", "consolidate"],
            ),
            # --- add 参数 ---
            ToolParameter(
                name="content",
                type="string",
                description="记忆内容文本（add 时必填）",
                required=False,
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="记忆类型（add 时使用，search 时作为单个类型过滤器）",
                required=False,
                default="working",
                enum=["working", "episodic", "semantic", "perceptual"],
            ),
            ToolParameter(
                name="importance",
                type="number",
                description="重要程度 0.0-1.0（add 时使用，不传则按记忆类型设置默认值）",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="关联文件路径（用于多模态记忆）",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="modality",
                type="string",
                description="信息模态: text/image/audio/video/code，不传则根据 file_path 自动推断",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="session_id",
                type="string",
                description="会话 ID，不传则使用当前默认会话",
                required=False,
                default=None,
            ),
            # --- search 参数 ---
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询文本（search 时必填）",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="number",
                description="返回结果数量上限（search 时使用）",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="memory_types",
                type="array",
                description="按多个记忆类型过滤（search 时使用，与 memory_type 可组合）",
                required=False,
                default=None,
                items=ToolParameter(
                    name="type",
                    type="string",
                    description="记忆类型",
                    required=True,
                    enum=["working", "episodic", "semantic", "perceptual"],
                ),
            ),
            ToolParameter(
                name="min_importance",
                type="number",
                description="最低重要度阈值（search 时使用）",
                required=False,
                default=0.1,
            ),
            ToolParameter(
                name="target_modality",
                type="string",
                description="目标模态过滤（search perceptual 时使用）",
                required=False,
                default=None,
                enum=["text", "image", "audio", "video", "code", "document", "data"],
            ),
            # --- forget 参数 ---
            ToolParameter(
                name="strategy",
                type="string",
                description="遗忘策略（forget 时使用）",
                required=False,
                default="importance",
                enum=["importance", "time", "capacity", "mixed", "dedup"],
            ),
            ToolParameter(
                name="threshold",
                type="number",
                description="阈值：importance 策略下限/dedup 相似度上限/capacity 容量上限",
                required=False,
                default=0.3,
            ),
            ToolParameter(
                name="max_age_days",
                type="number",
                description="最大保留天数（time/mixed 策略使用）",
                required=False,
                default=7,
            ),
            # --- consolidate 参数 ---
            ToolParameter(
                name="from_type",
                type="string",
                description="源记忆类型（consolidate 时使用，如 working→episodic）",
                required=False,
                default=None,
                enum=["working", "episodic", "semantic", "perceptual"],
            ),
            ToolParameter(
                name="to_type",
                type="string",
                description="目标记忆类型（consolidate 时使用）",
                required=False,
                default=None,
                enum=["working", "episodic", "semantic", "perceptual"],
            ),
            ToolParameter(
                name="importance_threshold",
                type="number",
                description="重要性阈值，仅整合 importance ≥ 此值的记忆（consolidate 时使用）",
                required=False,
                default=0.7,
            ),
            ToolParameter(
                name="time_window_minutes",
                type="number",
                description="时间窗口（分钟），将过去 N 分钟内的记忆打包整合并提醒用户",
                required=False,
                default=None,
            ),
        ]

    # ============================================================
    # 统一入口
    # ============================================================

    def execute(self, action, **kwargs):
        """
        统一入口（双协议）：旧协议 execute("add", content=...) 返回 str；
        新协议 execute({"action": "add", "content": ...}) 返回 ToolResponse。
        内部统一组装字典后委托给 run()。
        """
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
        return f"❌ 未知操作: '{action}'，当前支持: add, search, forget, consolidate, rerank"

    # ============================================================
    # 添加记忆
    # ============================================================

    def _add_memory(self, args: Dict[str, Any]) -> str:
        """
        添加一条记忆，自动补充时间戳、会话 ID、模态推断。
        支持通过 **metadata 传入额外字段（如 event_type, location, knowledge_type 等）。
        """
        content = args.get("content", "").strip()
        if not content:
            return "❌ content 不能为空"

        memory_type = args.get("memory_type", "working").lower()
        if memory_type not in DEFAULT_IMPORTANCE:
            return f"❌ 无效的 memory_type: '{memory_type}'，可选: {list(DEFAULT_IMPORTANCE.keys())}"

        if not self._manager.is_enabled(memory_type):
            return (f"❌ 记忆类型 '{memory_type}' 未启用，"
                    f"当前已启用: {self._manager.get_enabled_types()}")

        # importance: 显式传入 > 类型默认值 > 0.5
        importance = args.get("importance")
        if importance is None:
            importance = DEFAULT_IMPORTANCE[memory_type]
        importance = max(0.0, min(1.0, float(importance)))

        # 会话 ID
        session_id = args.get("session_id") or self._session_id

        # 文件路径 & 模态推断
        file_path = args.get("file_path")
        modality = args.get("modality")

        if file_path and not modality:
            modality = self._detect_modality(file_path)

        if not modality:
            modality = "text"

        # 提取额外元数据（排除已处理的字段）
        known_keys = {"action", "content", "memory_type", "importance",
                      "file_path", "modality", "session_id"}
        extra_metadata = {k: v for k, v in args.items() if k not in known_keys}

        # 文件元数据补充
        if file_path:
            extra_metadata["file_exists"] = os.path.exists(file_path)
            extra_metadata["file_extension"] = os.path.splitext(file_path)[1].lower()

        # 创建记忆条目
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
            f"✅ 记忆已添加\n"
            f"   ID: {entry.memory_id[:12]}...\n"
            f"   类型: {entry.memory_type}\n"
            f"   重要度: {entry.importance:.1f}\n"
            f"   模态: {entry.modality}\n"
            f"   会话: {entry.session_id[:12]}...\n"
            f"   时间: {entry.timestamp}\n"
            f"   总记忆数: {self._manager.get_count()}"
            + (f"\n   {add_msg}" if add_msg else "")
        )

    # ============================================================
    # 搜索记忆
    # ============================================================

    def _search_memory(self, args: Dict[str, Any]) -> str:
        """
        搜索记忆：工作记忆、情景记忆和语义记忆使用混合检索，其他类型使用关键词匹配。
        支持按记忆类型、重要度阈值过滤，并格式化输出。
        """
        # 1. 参数标准化
        params = self._normalize_search_params(args)

        if params["error"]:
            return params["error"]

        query = params["query"]
        limit = params["limit"]
        memory_types = params["memory_types"]
        min_importance = params["min_importance"]

        # 2. 无记忆时直接返回
        if self._manager.get_count() == 0:
            return "📭 记忆库为空，无法搜索。"

        # 3. 确定搜索范围
        search_types = memory_types or set(self._manager.get_enabled_types())

        # 4. 分类型检索
        scored = []
        retrieval_methods = []

        # 4a. 工作记忆使用混合检索（TF-IDF + 关键词）
        if "working" in search_types and self._manager.is_enabled("working"):
            working_module = self._manager.get_module("working")
            if working_module and working_module.get_count() > 0:
                retrieval_methods.append("working(混合检索)")
                results = working_module.retrieve(query, limit=limit)
                for final_score, base_relevance, mem, details in results:
                    if mem.importance >= min_importance:
                        scored.append((mem, base_relevance, final_score,
                                       details.get("method", "hybrid"), details))

        # 4b. 情景记忆使用混合检索（结构化过滤 + TF-IDF向量）
        if "episodic" in search_types and self._manager.is_enabled("episodic"):
            episodic_module = self._manager.get_module("episodic")
            if episodic_module and episodic_module.get_count() > 0:
                retrieval_methods.append("episodic(结构化+向量)")
                retrieve_kwargs = {"min_importance": min_importance}
                # 传递可选的会话过滤
                session_id = args.get("session_id")
                if session_id:
                    retrieve_kwargs["session_id"] = session_id
                results = episodic_module.retrieve(query, limit=limit, **retrieve_kwargs)
                for final_score, base_relevance, mem, details in results:
                    scored.append((mem, base_relevance, final_score,
                                   details.get("method", "hybrid"), details))

        # 4c. 语义记忆使用混合检索（TF-IDF向量 + 知识图谱）
        if "semantic" in search_types and self._manager.is_enabled("semantic"):
            semantic_module = self._manager.get_module("semantic")
            if semantic_module and semantic_module.get_count() > 0:
                retrieval_methods.append("semantic(向量+图谱)")
                results = semantic_module.retrieve(query, limit=limit, min_importance=min_importance)
                for final_score, base_relevance, mem, details in results:
                    scored.append((mem, base_relevance, final_score,
                                   details.get("method", "hybrid"), details))

        # 4d. 感知记忆使用多模态向量检索（嵌入 + 时间近因性 + 重要性融合）
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

        # 4e. 其他类型使用关键词匹配
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

        # 5. 排序（综合分降序）+ 截断
        scored.sort(key=lambda x: x[2], reverse=True)
        results = scored[:limit]

        # 6. 格式化输出
        return self._format_search_results(
            query, results, self._manager.get_count(), params, retrieval_methods
        )

    def _normalize_search_params(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        对搜索参数进行标准化处理：
        - query: 去空白、非空校验
        - limit: 整数转换，范围限制 1-100
        - memory_type + memory_types: 合并去重，小写化，校验有效性
        - min_importance: 浮点转换，范围限制 0.0-1.0
        """
        result = {"error": None}

        # query
        query = (args.get("query") or "").strip()
        if not query:
            result["error"] = "❌ search 操作需要提供 query 参数"
            return result
        result["query"] = query

        # limit
        try:
            limit = int(args.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        result["limit"] = max(1, min(100, limit))

        # memory_types: 合并 memory_type（单个）和 memory_types（数组）
        types_set = set()

        single_type = args.get("memory_type")
        if single_type:
            if isinstance(single_type, str):
                types_set.add(single_type.lower())
            elif isinstance(single_type, list):
                for t in single_type:
                    types_set.add(str(t).lower())

        multi_types = args.get("memory_types")
        if multi_types:
            if isinstance(multi_types, str):
                types_set.add(multi_types.lower())
            elif isinstance(multi_types, list):
                for t in multi_types:
                    types_set.add(str(t).lower())

        # 校验类型有效性
        valid_types = set(DEFAULT_IMPORTANCE.keys())
        invalid_types = types_set - valid_types
        if invalid_types:
            result["error"] = f"❌ 无效的记忆类型: {list(invalid_types)}，可选: {list(valid_types)}"
            return result

        result["memory_types"] = types_set if types_set else None

        # min_importance
        try:
            min_importance = float(args.get("min_importance", 0.1))
        except (TypeError, ValueError):
            min_importance = 0.1
        result["min_importance"] = max(0.0, min(1.0, min_importance))

        return result

    @staticmethod
    def _tokenize(query: str) -> List[str]:
        """将查询文本分词为关键词列表（支持中英文混合）。"""
        import re
        # 英文：按非字母数字分割；中文：逐字提取（2字以上组合更好匹配）
        tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', query.lower())
        # 中文按 2-gram 展开，提升匹配率
        expanded = []
        for token in tokens:
            if '\u4e00' <= token[0] <= '\u9fff' and len(token) >= 2:
                # 整词保留 + 2-gram
                expanded.append(token)
                for i in range(len(token) - 1):
                    expanded.append(token[i:i+2])
            else:
                expanded.append(token)
        return expanded if expanded else [query.lower()]

    @staticmethod
    def _calculate_relevance(mem: "MemoryEntry", keywords: List[str], raw_query: str) -> float:
        """
        计算单条记忆与查询的相关度分数（0.0-1.0）。
        匹配范围：content（权重 0.6）+ metadata 值（权重 0.3）+ memory_type（权重 0.1）。
        """
        if not keywords:
            return 0.0

        content_lower = mem.content.lower()
        meta_text = " ".join(str(v) for v in mem.metadata.values()).lower()
        type_lower = mem.memory_type.lower()

        content_hits = sum(1 for kw in keywords if kw in content_lower)
        meta_hits = sum(1 for kw in keywords if kw in meta_text)
        type_hits = sum(1 for kw in keywords if kw in type_lower)

        total_kw = len(keywords)
        if total_kw == 0:
            return 0.0

        content_score = content_hits / total_kw
        meta_score = meta_hits / total_kw
        type_score = type_hits / total_kw

        # 完整 query 子串匹配加分
        bonus = 0.0
        if raw_query.lower() in content_lower:
            bonus = 0.15

        relevance = content_score * 0.6 + meta_score * 0.3 + type_score * 0.1 + bonus
        return min(1.0, relevance)

    @staticmethod
    def _format_search_results(
        query: str,
        scored: List[tuple],
        total: int,
        params: Dict[str, Any],
        retrieval_methods: List[str] = None,
    ) -> str:
        """格式化搜索结果输出，混合检索结果显示详细评分。"""
        method_str = " + ".join(retrieval_methods) if retrieval_methods else "关键词匹配"
        lines = [
            "=" * 60,
            f"🔍 搜索记忆: \"{query}\"",
            f"   筛选条件: 类型={params['memory_types'] or '全部'}, "
            f"最低重要度={params['min_importance']:.1f}, 上限={params['limit']}",
            f"   检索方式: {method_str}",
            f"   记忆库总数: {total}，匹配结果: {len(scored)}",
            "=" * 60,
        ]

        if not scored:
            lines.append("📭 没有找到匹配的记忆。")
            return "\n".join(lines)

        for i, item in enumerate(scored, 1):
            mem, relevance, combined = item[0], item[1], item[2]
            method = item[3] if len(item) > 3 else "keyword"
            details = item[4] if len(item) > 4 else None

            lines.append("")
            lines.append(f"--- 结果 {i} ---")
            lines.append(f"  ID: {mem.memory_id[:12]}...")
            lines.append(f"  类型: {mem.memory_type}")
            lines.append(f"  重要度: {mem.importance:.1f}")

            if details and "modality" in details:
                # 感知记忆：多模态向量检索（嵌入/TF-IDF + 时间近因性 + 重要性）
                # 评分公式：(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)
                vec_method = details.get("vector_method", "tfidf")
                vec_label = "嵌入向量" if vec_method == "embedding" else "TF-IDF"
                lines.append(f"  检索方式: 🌐 多模态向量检索 ({vec_label} + 时间近因性)")
                lines.append(f"  向量方法: {vec_label}")
                lines.append(f"  向量相似度: {details['vector_score']:.4f}")
                lines.append(f"  时间近因性: {details['recency_score']:.4f}")
                lines.append(f"  基础相关度: {details['base_relevance']:.4f}")
                lines.append(f"  重要性权重: {details['importance_weight']:.4f}")
                lines.append(f"  最终得分: {details['combined_score']:.4f}")
            elif details and "recency_score" in details:
                # 情景记忆：结构化过滤 + TF-IDF向量检索
                # 评分公式：(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)
                if method == "hybrid":
                    lines.append(f"  检索方式: 🔀 混合检索 (结构化过滤 + TF-IDF向量)")
                    lines.append(f"  向量相似度: {details['vector_score']:.4f}")
                    lines.append(f"  TF-IDF得分: {details['tfidf_score']:.4f}")
                    lines.append(f"  关键词匹配分: {details['keyword_score']:.4f}")
                else:
                    lines.append(f"  检索方式: 📝 关键词匹配 (TF-IDF回退)")
                    lines.append(f"  关键词匹配分: {details['keyword_score']:.4f}")
                lines.append(f"  时间近因性: {details['recency_score']:.4f}")
                lines.append(f"  基础相关度: {details['base_relevance']:.4f}")
                lines.append(f"  重要性权重: {details['importance_weight']:.4f}")
                lines.append(f"  最终得分: {details['final_score']:.4f}")
            elif details and "graph_score" in details:
                # 语义记忆：嵌入向量/TF-IDF + 知识图谱检索
                # 评分公式：(向量相似度 × 0.7 + 图谱得分 × 0.3) × (0.8 + 重要性 × 0.4)
                vec_method = details.get("vector_method", "tfidf")
                vec_label = "嵌入向量" if vec_method == "embedding" else "TF-IDF"
                if method == "hybrid":
                    lines.append(f"  检索方式: 🔀 混合检索 ({vec_label} + 知识图谱)")
                elif method == "vector":
                    lines.append(f"  检索方式: 📊 向量检索 ({vec_label})")
                else:
                    lines.append(f"  检索方式: 🕸️ 图谱检索 (实体+关系)")
                lines.append(f"  向量方法: {vec_label}")
                lines.append(f"  向量相似度: {details['vector_score']:.4f}")
                lines.append(f"  图谱得分: {details['graph_score']:.4f}")
                lines.append(f"  基础相关度: {details['base_relevance']:.4f}")
                lines.append(f"  重要性权重: {details['importance_weight']:.4f}")
                lines.append(f"  最终得分: {details['combined_score']:.4f}")
                if details.get("matched_entities"):
                    lines.append(f"  匹配实体: {', '.join(details['matched_entities'])}")
                lines.append(f"  实体总数: {details.get('entity_count', 0)}")
            elif details and method == "hybrid":
                # 工作记忆：TF-IDF + 关键词混合检索
                # 评分公式：(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)
                lines.append(f"  检索方式: 🔀 混合检索 (TF-IDF + 关键词)")
                lines.append(f"  TF-IDF相似度: {details['tfidf_score']:.4f}")
                lines.append(f"  关键词匹配分: {details['keyword_score']:.4f}")
                lines.append(f"  混合相关度: {details['base_relevance']:.4f}")
                lines.append(f"  时间衰减: {details['time_decay']:.4f}")
                lines.append(f"  重要性权重: {details['importance_weight']:.4f}")
                lines.append(f"  最终得分: {combined:.4f}")
            elif details and method == "keyword":
                lines.append(f"  检索方式: 📝 关键词匹配 (TF-IDF回退)")
                lines.append(f"  关键词匹配分: {details['keyword_score']:.4f}")
                lines.append(f"  混合相关度: {details['base_relevance']:.4f}")
                lines.append(f"  时间衰减: {details['time_decay']:.4f}")
                lines.append(f"  重要性权重: {details['importance_weight']:.4f}")
                lines.append(f"  最终得分: {combined:.4f}")
            else:
                lines.append(f"  检索方式: 📝 关键词匹配")
                lines.append(f"  相关度: {relevance:.2f}")
                lines.append(f"  综合分: {combined:.2f}")

            lines.append(f"  内容: {mem.content}")
            lines.append(f"  时间: {mem.timestamp}")
            if mem.modality and mem.modality != "text":
                lines.append(f"  模态: {mem.modality}")
            if mem.file_path:
                lines.append(f"  文件: {mem.file_path}")
            if mem.metadata:
                lines.append(f"  元数据: {mem.metadata}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ============================================================
    # 遗忘记忆
    # ============================================================

    def _forget_memory(self, args: Dict[str, Any]) -> str:
        """
        根据指定策略遗忘（删除）记忆，返回操作摘要。
        支持五种策略：importance / time / capacity / mixed / dedup。
        """
        if self._manager.get_count() == 0:
            return "📭 记忆库为空，无需遗忘。"

        strategy = str(args.get("strategy", "importance")).lower()
        threshold = float(args.get("threshold", 0.3))
        max_age_days = int(args.get("max_age_days", 7))

        before_count = self._manager.get_count()

        if strategy == "importance":
            removed = self._forget_by_importance(threshold)
        elif strategy == "time":
            removed = self._forget_by_time(max_age_days)
        elif strategy == "capacity":
            removed = self._forget_by_capacity(int(threshold))
        elif strategy == "mixed":
            removed = self._forget_mixed(threshold, max_age_days)
        elif strategy == "dedup":
            removed = self._forget_by_dedup(threshold)
        else:
            return (f"❌ 未知策略: '{strategy}'，"
                    f"可选: importance, time, capacity, mixed, dedup")

        after_count = self._manager.get_count()

        return self._format_forget_result(strategy, threshold, max_age_days,
                                          before_count, after_count, removed)

    def _forget_by_importance(self, threshold: float) -> List["MemoryEntry"]:
        """基于重要性遗忘：删除 importance < threshold 的记忆。"""
        threshold = max(0.0, min(1.0, threshold))
        kept = []
        removed = []
        for mem in self._manager.get_all():
            if mem.importance < threshold:
                removed.append(mem)
            else:
                kept.append(mem)
        self._manager.replace_all(kept)
        return removed

    def _forget_by_time(self, max_age_days: int) -> List["MemoryEntry"]:
        """基于时间遗忘：删除超过 max_age_days 天的记忆。"""
        now = datetime.now()
        kept = []
        removed = []
        for mem in self._manager.get_all():
            try:
                mem_time = datetime.fromisoformat(mem.timestamp)
            except (ValueError, TypeError):
                kept.append(mem)
                continue
            age_days = (now - mem_time).total_seconds() / 86400
            if age_days >= max_age_days:
                removed.append(mem)
            else:
                kept.append(mem)
        self._manager.replace_all(kept)
        return removed

    def _forget_by_capacity(self, capacity: int) -> List["MemoryEntry"]:
        """基于容量遗忘：超过 capacity 条时，按重要度从低到高删除多余记忆。"""
        if capacity < 0:
            capacity = 0
        all_memories = self._manager.get_all()
        if len(all_memories) <= capacity:
            return []

        # 按重要度升序排列，删除最不重要的 (count - capacity) 条
        sorted_memories = sorted(all_memories, key=lambda m: m.importance)
        remove_count = len(all_memories) - capacity
        removed = sorted_memories[:remove_count]
        removed_ids = {m.memory_id for m in removed}
        self._manager.remove_ids(removed_ids)
        return removed

    def _forget_mixed(self, threshold: float, max_age_days: int) -> List["MemoryEntry"]:
        """混合策略：同时满足「低重要度」且「超时」的记忆被遗忘。"""
        threshold = max(0.0, min(1.0, threshold))
        now = datetime.now()
        kept = []
        removed = []
        for mem in self._manager.get_all():
            try:
                mem_time = datetime.fromisoformat(mem.timestamp)
                age_days = (now - mem_time).total_seconds() / 86400
            except (ValueError, TypeError):
                age_days = 0.0

            is_low_importance = mem.importance < threshold
            is_old = age_days >= max_age_days

            if is_low_importance and is_old:
                removed.append(mem)
            else:
                kept.append(mem)
        self._manager.replace_all(kept)
        return removed

    def _forget_by_dedup(self, similarity_threshold: float) -> List["MemoryEntry"]:
        """语义相似性去重：相似度高于阈值的记忆，保留重要度更高的那条。"""
        similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        memories = self._manager.get_all()
        if len(memories) < 2:
            return []

        removed = []
        removed_ids = set()

        for i in range(len(memories)):
            if memories[i].memory_id in removed_ids:
                continue
            for j in range(i + 1, len(memories)):
                if memories[j].memory_id in removed_ids:
                    continue

                sim = self._text_similarity(
                    memories[i].content,
                    memories[j].content,
                )
                if sim >= similarity_threshold:
                    # 保留重要度更高的，遗忘较低的；重要度相同则遗忘后者
                    if memories[i].importance >= memories[j].importance:
                        victim = memories[j]
                    else:
                        victim = memories[i]
                    if victim.memory_id not in removed_ids:
                        removed_ids.add(victim.memory_id)
                        removed.append(victim)

        self._manager.remove_ids(removed_ids)
        return removed

    # --- 遗忘策略辅助 ---

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """
        计算两段文本的相似度（0.0-1.0）。
        使用 Jaccard 相似度：交集词数 / 并集词数。
        支持中英文混合分词。
        """
        import re
        # 中英文分词
        tokens_a = set(re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text_a.lower()))
        tokens_b = set(re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text_b.lower()))

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    @staticmethod
    def _format_forget_result(
        strategy: str,
        threshold: float,
        max_age_days: int,
        before: int,
        after: int,
        removed: List["MemoryEntry"],
    ) -> str:
        """格式化遗忘操作的结果输出。"""
        strategy_labels = {
            "importance": f"基于重要性（阈值 < {threshold:.1f}）",
            "time": f"基于时间（超过 {max_age_days} 天）",
            "capacity": f"基于容量（上限 {int(threshold)} 条）",
            "mixed": f"混合策略（重要性 < {threshold:.1f} 且 超过 {max_age_days} 天）",
            "dedup": f"语义去重（相似度 ≥ {threshold:.2f}）",
        }

        lines = [
            "=" * 60,
            f"🗑️ 遗忘记忆",
            f"   策略: {strategy_labels.get(strategy, strategy)}",
            f"   遗忘前: {before} 条 → 遗忘后: {after} 条（删除 {len(removed)} 条）",
            "=" * 60,
        ]

        if removed:
            lines.append("")
            for i, mem in enumerate(removed, 1):
                lines.append(f"  [{i}] {mem.memory_id[:8]} ({mem.memory_type}, "
                             f"imp={mem.importance:.1f}) {mem.content[:60]}"
                             f"{'...' if len(mem.content) > 60 else ''}")
        else:
            lines.append("  （没有记忆被遗忘）")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ============================================================
    # 整合记忆
    # ============================================================

    # 允许的类型转换路径
    VALID_CONVERSIONS = {
        ("working", "episodic"),
        ("episodic", "semantic"),
        ("working", "semantic"),
        ("perceptual", "semantic"),
        ("perceptual", "episodic"),
    }

    def _consolidate_memory(self, args: Dict[str, Any]) -> str:
        """
        整合记忆：支持两种模式。
        1. 类型转换：将重要的 from_type 记忆提升为 to_type（如 working→episodic）
        2. 时间窗口：将过去 N 分钟内的记忆打包整合，并主动提醒用户
        """
        if self._manager.get_count() == 0:
            return "📭 记忆库为空，无需整合。"

        from_type = str(args.get("from_type", "")).lower() or None
        to_type = str(args.get("to_type", "")).lower() or None
        importance_threshold = max(0.0, min(1.0, float(args.get("importance_threshold", 0.7))))
        time_window_minutes = args.get("time_window_minutes")

        # 模式判断
        if time_window_minutes is not None:
            return self._consolidate_by_time_window(
                int(time_window_minutes), from_type, to_type, importance_threshold
            )
        else:
            return self._consolidate_by_type(from_type, to_type, importance_threshold)

    def _consolidate_by_type(
        self, from_type: str, to_type: str, importance_threshold: float
    ) -> str:
        """类型转换整合：将重要的 from_type 记忆转为 to_type。"""
        # 参数校验
        if not from_type or not to_type:
            return "❌ 类型转换模式需要同时提供 from_type 和 to_type 参数"
        if from_type == to_type:
            return f"❌ from_type 和 to_type 不能相同（均为 '{from_type}'）"
        if from_type not in DEFAULT_IMPORTANCE:
            return f"❌ 无效的 from_type: '{from_type}'，可选: {list(DEFAULT_IMPORTANCE.keys())}"
        if to_type not in DEFAULT_IMPORTANCE:
            return f"❌ 无效的 to_type: '{to_type}'，可选: {list(DEFAULT_IMPORTANCE.keys())}"
        if (from_type, to_type) not in self.VALID_CONVERSIONS:
            return (f"❌ 不支持的类型转换: {from_type}→{to_type}，"
                    f"允许的转换: {['→'.join(c) for c in self.VALID_CONVERSIONS]}")

        # 筛选符合条件的记忆
        candidates = [
            m for m in self._manager.get_all()
            if m.memory_type == from_type and m.importance >= importance_threshold
        ]

        if not candidates:
            return (f"ℹ️ 没有符合条件的记忆需要整合\n"
                    f"   筛选: type={from_type}, importance≥{importance_threshold:.1f}\n"
                    f"   当前记忆库中 {from_type} 类型且重要度达标的记忆: 0 条")

        # 执行转换：修改类型 + 微调重要度（整合后略有提升）
        converted = []
        for mem in candidates:
            old_type = mem.memory_type
            old_importance = mem.importance
            mem.memory_type = to_type
            # 重要度提升 0.05，但不超过 1.0
            mem.importance = min(1.0, mem.importance + 0.05)
            mem.metadata["consolidated_from"] = old_type
            mem.metadata["consolidated_at"] = datetime.now().isoformat()
            mem.metadata["pre_consolidation_importance"] = old_importance
            converted.append(mem)

        # 类型已变更，重新分发到对应模块
        self._manager.replace_all(self._manager.get_all())

        return self._format_consolidate_type_result(
            from_type, to_type, importance_threshold, converted
        )

    def _consolidate_by_time_window(
        self,
        time_window_minutes: int,
        from_type: str,
        to_type: str,
        importance_threshold: float,
    ) -> str:
        """
        时间窗口整合：将过去 N 分钟内的记忆打包整合为一条新记忆。
        原始记忆保留但标记为已整合，新记忆类型由 to_type 决定（默认 semantic）。
        整合完成后主动提醒用户。
        """
        if time_window_minutes < 0:
            time_window_minutes = 0

        now = datetime.now()
        window_start = now.timestamp() - time_window_minutes * 60

        # 筛选时间窗口内的记忆
        in_window = []
        for mem in self._manager.get_all():
            try:
                mem_time = datetime.fromisoformat(mem.timestamp).timestamp()
            except (ValueError, TypeError):
                continue
            if mem_time >= window_start:
                # 可选：按 from_type 过滤
                if from_type and mem.memory_type != from_type:
                    continue
                if mem.importance < importance_threshold:
                    continue
                in_window.append(mem)

        if not in_window:
            return (f"ℹ️ 时间窗口（过去 {time_window_minutes} 分钟）内没有需要整合的记忆\n"
                    f"   筛选: type={from_type or '全部'}, importance≥{importance_threshold:.1f}")

        # 确定目标类型
        target_type = to_type or "semantic"

        # 构建整合记忆的内容
        summary_lines = [f"[整合记忆] 以下 {len(in_window)} 条记忆在过去 {time_window_minutes} 分钟内产生，已打包整合："]
        for i, mem in enumerate(in_window, 1):
            summary_lines.append(f"  {i}. [{mem.memory_type}] (imp={mem.importance:.1f}) {mem.content}")
        consolidated_content = "\n".join(summary_lines)

        # 计算整合后的重要度（取窗口内最高重要度）
        max_importance = max(m.importance for m in in_window)

        # 创建整合记忆条目
        consolidated_entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content=consolidated_content,
            memory_type=target_type,
            importance=max_importance,
            timestamp=now.isoformat(),
            session_id=self._session_id,
            modality="text",
            metadata={
                "consolidation": True,
                "source_count": len(in_window),
                "time_window_minutes": time_window_minutes,
                "source_types": list({m.memory_type for m in in_window}),
                "source_ids": [m.memory_id for m in in_window],
            },
        )

        # 标记原始记忆为已整合
        for mem in in_window:
            mem.metadata["consolidated"] = True
            mem.metadata["consolidated_into"] = consolidated_entry.memory_id

        # 添加整合记忆到记忆库
        self._manager.add(consolidated_entry)

        return self._format_consolidate_window_result(
            time_window_minutes, target_type, importance_threshold,
            in_window, consolidated_entry
        )

    # --- 整合结果格式化 ---

    @staticmethod
    def _format_consolidate_type_result(
        from_type: str,
        to_type: str,
        importance_threshold: float,
        converted: List["MemoryEntry"],
    ) -> str:
        """格式化类型转换整合结果。"""
        lines = [
            "=" * 60,
            f"🔄 整合记忆（类型转换）",
            f"   转换: {from_type} → {to_type}",
            f"   阈值: importance ≥ {importance_threshold:.1f}",
            f"   整合数量: {len(converted)} 条",
            "=" * 60,
        ]

        if converted:
            lines.append("")
            for i, mem in enumerate(converted, 1):
                old_imp = mem.metadata.get("pre_consolidation_importance", "?")
                lines.append(
                    f"  [{i}] {mem.memory_id[:8]} "
                    f"{from_type}(imp={old_imp:.1f}) → {to_type}(imp={mem.importance:.1f})  "
                    f"{mem.content[:50]}{'...' if len(mem.content) > 50 else ''}"
                )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _format_consolidate_window_result(
        time_window_minutes: int,
        target_type: str,
        importance_threshold: float,
        source_memories: List["MemoryEntry"],
        consolidated: "MemoryEntry",
    ) -> str:
        """格式化时间窗口整合结果（含主动提醒）。"""
        lines = [
            "=" * 60,
            f"⏰ 整合记忆（时间窗口）",
            f"   窗口: 过去 {time_window_minutes} 分钟",
            f"   目标类型: {target_type}",
            f"   阈值: importance ≥ {importance_threshold:.1f}",
            f"   整合数量: {len(source_memories)} 条 → 1 条整合记忆",
            "=" * 60,
            "",
            "📦 整合记忆内容:",
            consolidated.content,
            "",
            f"   整合记忆 ID: {consolidated.memory_id[:12]}...",
            f"   整合记忆类型: {consolidated.memory_type}",
            f"   整合记忆重要度: {consolidated.importance:.1f}",
            "",
            "🔔 【主动提醒】时间窗口已结束，上述记忆已完成整合。",
            "   原始记忆已标记为「已整合」，整合记忆已存入记忆库。",
            "   您可以通过搜索功能检索整合后的记忆。",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ============================================================
    # LLM 重排序搜索
    # ============================================================

    def _rerank_memory(self, args: Dict[str, Any]) -> str:
        """TF-IDF 粗筛 + LLM 精排 两阶段检索。

        Args:
            query: 查询文本
            memory_type: 限定记忆类型（可选，不指定则搜索全部已启用类型）
            memory_types: 同 memory_type 的别名（兼容原有参数名）
            coarse_limit: TF-IDF 粗筛保留的候选数量（默认 15）
            rerank_limit: LLM 精排后返回的最终数量（默认 5）
            min_importance: 最低重要度过滤（默认 0.0）
        """
        query = (args.get("query") or "").strip()
        if not query:
            return "❌ rerank 操作需要提供 query 参数"

        # 参数标准化
        coarse_limit = max(1, min(50, int(args.get("coarse_limit", 15))))
        rerank_limit = max(1, min(20, int(args.get("rerank_limit", 5))))
        min_importance = max(0.0, min(1.0, float(args.get("min_importance", 0.0))))

        # 记忆类型过滤
        memory_type = args.get("memory_type") or args.get("memory_types")
        if memory_type:
            if isinstance(memory_type, list):
                memory_types = [t.lower() for t in memory_type]
            else:
                memory_types = [memory_type.lower()]
            for mt in memory_types:
                if mt not in DEFAULT_IMPORTANCE:
                    return f"❌ 无效的记忆类型: '{mt}'，可选: {list(DEFAULT_IMPORTANCE.keys())}"
        else:
            memory_types = list(self._manager.get_enabled_types())

        # 收集候选记忆
        candidates = []
        for mem in self._manager.get_all():
            if mem.memory_type in memory_types and mem.importance >= min_importance:
                candidates.append(mem)

        if not candidates:
            return "📭 没有符合条件的记忆可供重排序。"

        # TF-IDF 关键词抽取（展示用）
        keywords = LLMReranker.extract_keywords(query, top_k=5)

        # 执行两阶段检索
        reranker = LLMReranker()
        results = reranker.rerank(
            candidates, query,
            coarse_limit=coarse_limit,
            rerank_limit=rerank_limit,
        )

        return self._format_rerank_results(
            query, results, len(candidates), keywords,
            coarse_limit, rerank_limit, memory_types, min_importance,
        )

    @staticmethod
    def _format_rerank_results(
        query: str,
        results: list,
        total_candidates: int,
        keywords: list,
        coarse_limit: int,
        rerank_limit: int,
        memory_types: list,
        min_importance: float,
    ) -> str:
        """格式化重排序结果输出。"""
        lines = [
            "=" * 60,
            f"🎯 LLM 重排序检索: \"{query}\"",
            f"   TF-IDF 关键词: {', '.join(keywords) if keywords else '(无)'}",
            f"   记忆类型: {', '.join(memory_types)}, 最低重要度: {min_importance:.1f}",
            f"   候选池: {total_candidates} 条 → TF-IDF 粗筛: {coarse_limit} 条 → LLM 精排: {rerank_limit} 条",
            "=" * 60,
        ]

        if not results:
            lines.append("📭 LLM 重排序后没有匹配的记忆。")
            return "\n".join(lines)

        for i, (mem, relevance, reason) in enumerate(results, 1):
            lines.append("")
            lines.append(f"--- 结果 {i} ---")
            lines.append(f"  ID: {mem.memory_id[:12]}...")
            lines.append(f"  类型: {mem.memory_type}")
            lines.append(f"  LLM 相关性: {relevance:.2f}")
            lines.append(f"  理由: {reason}")
            lines.append(f"  重要度: {mem.importance:.1f}")
            lines.append(f"  内容: {mem.content}")
            lines.append(f"  时间: {mem.timestamp}")

        return "\n".join(lines)

    # ============================================================
    # 辅助方法
    # ============================================================

    @staticmethod
    def _detect_modality(file_path: str) -> str:
        """根据文件扩展名推断模态。"""
        ext = os.path.splitext(file_path)[1].lower()
        return FILE_MODALITY_MAP.get(ext, "unknown")

    def get_all_memories(self) -> List[MemoryEntry]:
        """获取所有记忆（供外部调用）。"""
        return self._manager.get_all()

    def get_memory_count(self) -> int:
        """获取记忆总数。"""
        return self._manager.get_count()


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import json

    # ========== 第零部分：MemoryManager 配置演示 ==========
    print("=" * 60)
    print("⚙️ MemoryTool - MemoryManager 配置演示")
    print("=" * 60)
    print()

    # 默认配置（全部模块启用）
    print("--- 配置 1: 默认配置（全部启用）---")
    default_tool = MemoryTool()
    print(f"   已启用模块: {default_tool._manager.get_enabled_types()}")
    print(f"   工作记忆容量: {default_tool._manager.get_config().get('working_capacity')}")
    print(f"   工作记忆TTL: {default_tool._manager.get_config().get('working_memory_ttl')} 分钟")
    print()

    # 自定义配置：仅启用 working + semantic，工作记忆容量=3，TTL=30分钟
    print("--- 配置 2: 自定义配置（仅 working + semantic，容量=3，TTL=30分钟）---")
    custom_tool = MemoryTool(config={
        "enabled_types": ["working", "semantic"],
        "working_capacity": 3,
        "working_memory_ttl": 30,
    })
    print(f"   已启用模块: {custom_tool._manager.get_enabled_types()}")
    print(f"   工作记忆容量: {custom_tool._manager.get_config().get('working_capacity')}")
    print(f"   工作记忆TTL: {custom_tool._manager.get_config().get('working_memory_ttl')} 分钟")
    print()

    # 工作记忆容量淘汰测试
    print("--- 配置 3: 工作记忆容量淘汰（容量=3，添加 5 条）---")
    for i in range(5):
        r = custom_tool.execute("add", content=f"工作记忆 #{i+1}", memory_type="working", importance=0.1 * (i + 1))
    print(f"   添加 5 条后实际保留: {custom_tool.get_memory_count()} 条（容量=3）")
    print()

    # 未启用类型拒绝测试
    print("--- 配置 4: 未启用类型拒绝（episodic 未启用）---")
    r = custom_tool.execute("add", content="测试情景记忆", memory_type="episodic")
    print(f"   {r}")
    print()

    # ========== 第一部分：添加记忆（使用 execute 入口）==========
    print("=" * 60)
    print("🧠 MemoryTool - 添加记忆演示")
    print("=" * 60)
    print()

    tool = MemoryTool()

    # 1. 工作记忆
    print("--- 1. 工作记忆 ---")
    result = tool.execute(
        "add",
        content="用户刚才问了关于Python函数的问题",
        memory_type="working",
        importance=0.6,
    )
    print(result)
    print()

    # 2. 情景记忆（含额外元数据）
    print("--- 2. 情景记忆 ---")
    result = tool.execute(
        "add",
        content="2024年3月15日，用户张三完成了第一个Python项目",
        memory_type="episodic",
        importance=0.8,
        event_type="milestone",
        location="在线学习平台",
    )
    print(result)
    print()

    # 3. 语义记忆（含额外元数据）
    print("--- 3. 语义记忆 ---")
    result = tool.execute(
        "add",
        content="Python是一种解释型、面向对象的编程语言",
        memory_type="semantic",
        importance=0.9,
        knowledge_type="factual",
    )
    print(result)
    print()

    # 4. 感知记忆（多模态，自动推断模态）
    print("--- 4. 感知记忆（自动推断模态）---")
    result = tool.execute(
        "add",
        content="用户上传了一张Python代码截图，包含函数定义",
        memory_type="perceptual",
        importance=0.7,
        file_path="./uploads/code_screenshot.png",
    )
    print(result)
    print()

    # 5. 默认重要度测试（不传 importance）
    print("--- 5. 默认重要度测试 ---")
    result = tool.execute(
        "add",
        content="用户正在学习数据结构",
        memory_type="working",
    )
    print(result)
    print()

    # 6. 代码文件模态推断
    print("--- 6. 代码文件模态推断 ---")
    result = tool.execute(
        "add",
        content="用户提交了一个Python脚本",
        memory_type="perceptual",
        file_path="./solution.py",
    )
    print(result)
    print()

    # ========== 第二部分：搜索记忆（使用 execute 入口）==========
    print("=" * 60)
    print("🔍 MemoryTool - 搜索记忆演示")
    print("=" * 60)
    print()

    # 测试 1: 基本搜索
    print("--- 搜索 1: 搜索「Python」---")
    result = tool.execute("search", query="Python", limit=5)
    print(result)
    print()

    # 测试 2: 按类型过滤
    print("--- 搜索 2: 搜索「用户」，仅 semantic 类型 ---")
    result = tool.execute("search", query="用户", memory_type="semantic")
    print(result)
    print()

    # 测试 3: 多类型过滤 + 重要度阈值
    print("--- 搜索 3: 搜索「Python」，类型=[working, episodic]，重要度≥0.7 ---")
    result = tool.execute(
        "search",
        query="Python",
        memory_types=["working", "episodic"],
        min_importance=0.7,
    )
    print(result)
    print()

    # 测试 4: 无匹配结果
    print("--- 搜索 4: 搜索「Java」（无匹配）---")
    result = tool.execute("search", query="Java")
    print(result)
    print()

    # 测试 5: 参数标准化（memory_type + memory_types 组合）
    print("--- 搜索 5: memory_type=episodic + memory_types=[semantic, perceptual] ---")
    result = tool.execute(
        "search",
        query="Python",
        memory_type="episodic",
        memory_types=["semantic", "perceptual"],
    )
    print(result)
    print()

    # 测试 6: 无效类型校验
    print("--- 搜索 6: 无效类型校验 ---")
    result = tool.execute("search", query="Python", memory_types=["invalid_type"])
    print(result)
    print()

    # 测试 7: limit 限制
    print("--- 搜索 7: limit=2 ---")
    result = tool.execute("search", query="Python", limit=2)
    print(result)

    # ========== 第三部分：遗忘记忆 ==========
    print()
    print("=" * 60)
    print("🗑️ MemoryTool - 遗忘记忆演示")
    print("=" * 60)
    print()

    # --- 测试 1: 基于重要性遗忘 ---
    print("--- 遗忘 1: importance 策略（threshold=0.7）---")
    print(f"   遗忘前: {tool.get_memory_count()} 条")
    result = tool.execute("forget", strategy="importance", threshold=0.7)
    print(result)
    print()

    # --- 测试 2: 基于容量遗忘 ---
    print("--- 遗忘 2: capacity 策略（上限 2 条）---")
    print(f"   遗忘前: {tool.get_memory_count()} 条")
    result = tool.execute("forget", strategy="capacity", threshold=2)
    print(result)
    print()

    # --- 测试 3: 语义去重（新建工具，添加相似记忆）---
    print("--- 遗忘 3: dedup 策略（相似度 ≥ 0.5）---")
    dedup_tool = MemoryTool()
    dedup_tool.execute("add", content="Python是一种编程语言", memory_type="semantic", importance=0.9)
    dedup_tool.execute("add", content="Python是一种编程语言", memory_type="semantic", importance=0.5)
    dedup_tool.execute("add", content="Java是一种编程语言", memory_type="semantic", importance=0.8)
    dedup_tool.execute("add", content="今天天气很好", memory_type="working", importance=0.6)
    print(f"   遗忘前: {dedup_tool.get_memory_count()} 条")
    result = dedup_tool.execute("forget", strategy="dedup", threshold=0.5)
    print(result)
    print()

    # --- 测试 4: 混合策略（新建工具）---
    print("--- 遗忘 4: mixed 策略（importance < 0.7 且 age > 0 天）---")
    mixed_tool = MemoryTool()
    mixed_tool.execute("add", content="低重要度记忆", memory_type="working", importance=0.3)
    mixed_tool.execute("add", content="高重要度记忆", memory_type="semantic", importance=0.9)
    print(f"   遗忘前: {mixed_tool.get_memory_count()} 条")
    result = mixed_tool.execute("forget", strategy="mixed", threshold=0.7, max_age_days=0)
    print(result)
    print()

    # --- 测试 5: 基于时间遗忘（max_age_days=0 删除全部）---
    print("--- 遗忘 5: time 策略（max_age_days=0，删除全部刚创建的记忆）---")
    time_tool = MemoryTool()
    time_tool.execute("add", content="临时记忆1", memory_type="working", importance=0.5)
    time_tool.execute("add", content="临时记忆2", memory_type="working", importance=0.9)
    print(f"   遗忘前: {time_tool.get_memory_count()} 条")
    result = time_tool.execute("forget", strategy="time", max_age_days=0)
    print(result)
    print()

    # --- 测试 6: 无效策略 ---
    print("--- 遗忘 6: 无效策略校验 ---")
    result = tool.execute("forget", strategy="invalid_strategy")
    print(result)

    # ========== 第四部分：整合记忆 ==========
    print()
    print("=" * 60)
    print("🔄 MemoryTool - 整合记忆演示")
    print("=" * 60)
    print()

    # --- 测试 1: 类型转换 working→episodic ---
    print("--- 整合 1: 类型转换 working→episodic（importance≥0.6）---")
    consol_tool = MemoryTool()
    consol_tool.execute("add", content="用户问了Python函数的问题", memory_type="working", importance=0.8)
    consol_tool.execute("add", content="用户正在调试代码", memory_type="working", importance=0.6)
    consol_tool.execute("add", content="低优先级临时笔记", memory_type="working", importance=0.3)
    consol_tool.execute("add", content="Python是解释型语言", memory_type="semantic", importance=0.9)
    print(f"   整合前: {consol_tool.get_memory_count()} 条")
    result = consol_tool.execute("consolidate", from_type="working", to_type="episodic", importance_threshold=0.6)
    print(result)
    print()

    # --- 测试 2: 类型转换 episodic→semantic ---
    print("--- 整合 2: 类型转换 episodic→semantic（importance≥0.8）---")
    print(f"   整合前: {consol_tool.get_memory_count()} 条")
    result = consol_tool.execute("consolidate", from_type="episodic", to_type="semantic", importance_threshold=0.8)
    print(result)
    print()

    # --- 测试 3: 不支持的转换路径 ---
    print("--- 整合 3: 不支持的转换路径 semantic→working ---")
    result = consol_tool.execute("consolidate", from_type="semantic", to_type="working")
    print(result)
    print()

    # --- 测试 4: 时间窗口整合 ---
    print("--- 整合 4: 时间窗口整合（过去 60 分钟，全类型）---")
    window_tool = MemoryTool()
    window_tool.execute("add", content="用户学习了Python基础", memory_type="working", importance=0.8)
    window_tool.execute("add", content="用户完成了一个练习题", memory_type="episodic", importance=0.7)
    window_tool.execute("add", content="Python支持面向对象", memory_type="semantic", importance=0.9)
    print(f"   整合前: {window_tool.get_memory_count()} 条")
    result = window_tool.execute("consolidate", time_window_minutes=60, importance_threshold=0.1)
    print(result)
    print()

    # --- 测试 5: 时间窗口 + 类型过滤 ---
    print("--- 整合 5: 时间窗口整合（过去 60 分钟，仅 working 类型）---")
    window_tool2 = MemoryTool()
    window_tool2.execute("add", content="临时任务A", memory_type="working", importance=0.8)
    window_tool2.execute("add", content="临时任务B", memory_type="working", importance=0.6)
    window_tool2.execute("add", content="持久知识", memory_type="semantic", importance=0.9)
    print(f"   整合前: {window_tool2.get_memory_count()} 条")
    result = window_tool2.execute(
        "consolidate", time_window_minutes=60, from_type="working",
        to_type="episodic", importance_threshold=0.5
    )
    print(result)
    print()

    # --- 测试 6: 无符合条件记忆 ---
    print("--- 整合 6: 无符合条件记忆（importance≥0.95）---")
    result = consol_tool.execute("consolidate", from_type="working", to_type="episodic", importance_threshold=0.95)
    print(result)

    # ========== 第五部分：工作记忆混合检索 ==========
    print()
    print("=" * 60)
    print("🔀 MemoryTool - 工作记忆混合检索演示")
    print("    TF-IDF向量化语义检索 + 关键词匹配回退")
    print("    评分公式: (相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)")
    print("=" * 60)
    print()

    # --- 测试 1: TF-IDF 语义检索 ---
    print("--- 混合检索 1: TF-IDF 语义检索 ---")
    print("   添加5条工作记忆（语义相关 vs 无关）...")
    tfidf_tool = MemoryTool()
    tfidf_tool.execute("add", content="Python是一种解释型编程语言", memory_type="working", importance=0.8)
    tfidf_tool.execute("add", content="Java也是一种面向对象的编程语言", memory_type="working", importance=0.7)
    tfidf_tool.execute("add", content="用户正在学习数据结构与算法", memory_type="working", importance=0.6)
    tfidf_tool.execute("add", content="今天午饭吃了面条", memory_type="working", importance=0.3)
    tfidf_tool.execute("add", content="Golang是Google开发的编程语言", memory_type="working", importance=0.5)
    print(f"   已添加 {tfidf_tool.get_memory_count()} 条工作记忆")
    print()
    print("   搜索「编程语言」（应匹配3条编程相关记忆）:")
    result = tfidf_tool.execute("search", query="编程语言", memory_type="working", limit=5)
    print(result)
    print()

    # --- 测试 2: 关键词回退（TF-IDF失败：仅1条记忆）---
    print("--- 混合检索 2: 关键词回退（仅1条记忆，TF-IDF需要≥2条）---")
    single_tool = MemoryTool()
    single_tool.execute("add", content="用户问了关于Python函数的问题", memory_type="working", importance=0.7)
    result = single_tool.execute("search", query="Python", memory_type="working")
    print(result)
    print()

    # --- 测试 3: 时间衰减效果 ---
    print("--- 混合检索 3: 时间衰减效果 ---")
    print("   添加3条相似记忆，手动修改时间戳使其中1条更早...")
    decay_tool = MemoryTool(config={"enabled_types": ["working"], "working_memory_ttl": 60})
    decay_tool.execute("add", content="Python编程语言基础知识", memory_type="working", importance=0.7)
    decay_tool.execute("add", content="Python编程语言进阶用法", memory_type="working", importance=0.7)
    decay_tool.execute("add", content="Python编程语言高级特性", memory_type="working", importance=0.7)

    # 手动将第一条记忆的时间戳设为30分钟前
    working_mod = decay_tool._manager.get_module("working")
    if working_mod and len(working_mod._memories) > 0:
        old_time = datetime.now() - timedelta(minutes=30)
        working_mod._memories[0].timestamp = old_time.isoformat()
        print(f"   第一条记忆时间戳已修改为30分钟前: {working_mod._memories[0].content}")
    print()
    print("   搜索「Python编程」（观察时间衰减差异）:")
    result = decay_tool.execute("search", query="Python编程", memory_type="working", limit=5)
    print(result)
    print()

    # --- 测试 4: TTL 自动过期清理 ---
    print("--- 混合检索 4: TTL 自动过期清理 ---")
    ttl_tool = MemoryTool(config={"enabled_types": ["working"], "working_memory_ttl": 1})
    ttl_tool.execute("add", content="这条记忆很快会过期", memory_type="working", importance=0.9)
    print(f"   添加1条记忆，TTL=1分钟，当前记忆数: {ttl_tool.get_memory_count()}")

    # 手动将时间戳设为2分钟前（超过TTL）
    working_mod = ttl_tool._manager.get_module("working")
    if working_mod and len(working_mod._memories) > 0:
        old_time = datetime.now() - timedelta(minutes=2)
        working_mod._memories[0].timestamp = old_time.isoformat()

    print("   已将记忆时间戳设为2分钟前（超过TTL=1分钟）")
    # 再次添加触发过期清理
    ttl_tool.execute("add", content="新记忆触发过期清理", memory_type="working", importance=0.5)
    print(f"   添加新记忆后，当前记忆数: {ttl_tool.get_memory_count()}（旧记忆已过期清理）")
    print()

    # --- 测试 5: 容量管理（默认50条，超出淘汰最不重要的）---
    print("--- 混合检索 5: 容量管理（容量=5，添加8条，观察淘汰）---")
    cap_tool = MemoryTool(config={"enabled_types": ["working"], "working_capacity": 5})
    for i in range(8):
        r = cap_tool.execute(
            "add",
            content=f"工作记忆条目 #{i+1}",
            memory_type="working",
            importance=0.1 * (i + 1),  # 0.1 ~ 0.8
        )
    print(f"   添加8条后实际保留: {cap_tool.get_memory_count()} 条（容量=5）")
    print("   被淘汰的是重要度最低的记忆（importance=0.1, 0.2, 0.3）")
    remaining = [m.content for m in cap_tool.get_all_memories()]
    print(f"   保留的记忆: {remaining}")
    print()

    # --- 测试 6: 直接调用 retrieve 方法 ---
    print("--- 混合检索 6: 直接调用 WorkingMemory.retrieve() ---")
    direct_tool = MemoryTool()
    direct_tool.execute("add", content="机器学习是人工智能的子领域", memory_type="working", importance=0.9)
    direct_tool.execute("add", content="深度学习使用神经网络", memory_type="working", importance=0.8)
    direct_tool.execute("add", content="自然语言处理处理文本数据", memory_type="working", importance=0.7)
    direct_tool.execute("add", content="今天去超市买了水果", memory_type="working", importance=0.3)

    working_mod = direct_tool._manager.get_module("working")
    print(f"   工作记忆数: {working_mod.get_count()}")
    print()
    print("   retrieve('学习', limit=3):")
    results = working_mod.retrieve("学习", limit=3)
    for score, base_rel, mem, details in results:
        print(f"   [{details['method']:8s}] score={score:.4f}  "
              f"tfidf={details['tfidf_score']:.4f}  kw={details['keyword_score']:.4f}  "
              f"decay={details['time_decay']:.4f}  imp_w={details['importance_weight']:.4f}  "
              f"| {mem.content[:30]}")
    print()
    print("   retrieve('水果', limit=3)（TF-IDF精准匹配含'水果'的记忆，排除无关记忆）:")
    results = working_mod.retrieve("水果", limit=3)
    for score, base_rel, mem, details in results:
        print(f"   [{details['method']:8s}] score={score:.4f}  "
              f"tfidf={details['tfidf_score']:.4f}  kw={details['keyword_score']:.4f}  "
              f"decay={details['time_decay']:.4f}  imp_w={details['importance_weight']:.4f}  "
              f"| {mem.content[:30]}")

    # ========== 第六部分：情景记忆混合检索 ==========
    print()
    print("=" * 60)
    print("🎭 MemoryTool - 情景记忆混合检索演示")
    print("    SQLite持久化 + 会话索引 + 结构化过滤 + TF-IDF向量检索")
    print("    评分公式: (向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)")
    print("=" * 60)
    print()

    # --- 测试 1: 结构化过滤 + TF-IDF 向量检索 ---
    print("--- 情景检索 1: 结构化过滤 + TF-IDF向量检索 ---")
    print("   添加5条情景记忆（含不同会话和重要度）...")
    epi_tool = MemoryTool()
    epi_tool.execute("add", content="用户在 北京 完成了Python项目部署",
                     memory_type="episodic", importance=0.9, session_id="sess_A")
    epi_tool.execute("add", content="用户在 上海 参加了AI技术大会",
                     memory_type="episodic", importance=0.8, session_id="sess_B")
    epi_tool.execute("add", content="用户在深圳学习了机器学习课程",
                     memory_type="episodic", importance=0.7, session_id="sess_A")
    epi_tool.execute("add", content="今天午饭吃了牛肉面",
                     memory_type="episodic", importance=0.3, session_id="sess_C")
    epi_tool.execute("add", content="用户在杭州完成了Go语言项目",
                     memory_type="episodic", importance=0.6, session_id="sess_B")
    print(f"   已添加 {epi_tool.get_memory_count()} 条情景记忆")
    print()
    print("   搜索「项目」（应匹配含'项目'的记忆）:")
    result = epi_tool.execute("search", query="项目", memory_type="episodic", limit=5)
    print(result)
    print()

    # --- 测试 2: 会话级检索 ---
    print("--- 情景检索 2: 会话级检索（session_id=sess_A）---")
    print("   仅搜索 sess_A 会话的记忆...")
    result = epi_tool.execute("search", query="用户", memory_type="episodic",
                              limit=5, session_id="sess_A")
    print(result)
    print()

    # --- 测试 3: 重要度过滤 ---
    print("--- 情景检索 3: 重要度过滤（min_importance=0.7）---")
    result = epi_tool.execute("search", query="用户", memory_type="episodic",
                              limit=5, min_importance=0.7)
    print(result)
    print()

    # --- 测试 4: 时间近因性效果 ---
    print("--- 情景检索 4: 时间近因性效果 ---")
    print("   添加3条相似记忆，手动修改时间戳使其中1条更早...")
    recency_tool = MemoryTool(config={"enabled_types": ["episodic"]})
    recency_tool.execute("add", content="Python编程项目开发", memory_type="episodic", importance=0.7)
    recency_tool.execute("add", content="Python编程项目测试", memory_type="episodic", importance=0.7)
    recency_tool.execute("add", content="Python编程项目部署", memory_type="episodic", importance=0.7)

    # 手动将第一条记忆的时间戳设为60天前
    epi_mod = recency_tool._manager.get_module("episodic")
    if epi_mod and len(epi_mod._memories) > 0:
        old_time = datetime.now() - timedelta(days=60)
        epi_mod._memories[0].timestamp = old_time.isoformat()
        epi_mod._persist_episode(epi_mod._memories[0])
        print(f"   第一条记忆时间戳已修改为60天前: {epi_mod._memories[0].content}")
    print()
    print("   搜索「Python编程」（观察时间近因性差异）:")
    result = recency_tool.execute("search", query="Python编程", memory_type="episodic", limit=5)
    print(result)
    print()

    # --- 测试 5: SQLite 持久化验证 ---
    print("--- 情景检索 5: SQLite 持久化验证 ---")
    print("   创建新 MemoryTool 复用同一个 SQLite 文件...")
    import tempfile, os
    db_file = os.path.join(tempfile.gettempdir(), "episodic_test.db")
    if os.path.exists(db_file):
        os.remove(db_file)

    persist_tool = MemoryTool(config={"enabled_types": ["episodic"], "episodic_db_path": db_file})
    persist_tool.execute("add", content="这条记忆会被持久化到SQLite", memory_type="episodic", importance=0.9)
    persist_tool.execute("add", content="另一条持久化记忆", memory_type="episodic", importance=0.7)
    print(f"   写入 {persist_tool.get_memory_count()} 条记忆到 {db_file}")

    # 创建新工具，加载同一个数据库
    reload_tool = MemoryTool(config={"enabled_types": ["episodic"], "episodic_db_path": db_file})
    print(f"   重新加载后记忆数: {reload_tool.get_memory_count()}（应为2）")
    result = reload_tool.execute("search", query="持久化", memory_type="episodic")
    print(result)

    # 清理临时数据库（关闭所有连接后删除）
    for t in [persist_tool, reload_tool]:
        m = t._manager.get_module("episodic")
        if m:
            m.close()
    if os.path.exists(db_file):
        os.remove(db_file)
    print()

    # --- 测试 6: 直接调用 EpisodicMemory.retrieve() ---
    print("--- 情景检索 6: 直接调用 EpisodicMemory.retrieve() ---")
    direct_epi_tool = MemoryTool()
    direct_epi_tool.execute("add", content="用户完成了机器学习培训课程", memory_type="episodic", importance=0.9)
    direct_epi_tool.execute("add", content="用户参加了深度学习研讨会", memory_type="episodic", importance=0.8)
    direct_epi_tool.execute("add", content="用户阅读了强化学习论文", memory_type="episodic", importance=0.7)
    direct_epi_tool.execute("add", content="今天去公园散步了", memory_type="episodic", importance=0.3)

    epi_mod = direct_epi_tool._manager.get_module("episodic")
    print(f"   情景记忆数: {epi_mod.get_count()}")
    print(f"   会话索引: {list(epi_mod.sessions.keys())}")
    print()
    print("   retrieve('学习', limit=3):")
    results = epi_mod.retrieve("学习", limit=3)
    for score, base_rel, mem, details in results:
        print(f"   [{details['method']:8s}] score={score:.4f}  "
              f"vec={details['vector_score']:.4f}  tfidf={details['tfidf_score']:.4f}  "
              f"kw={details['keyword_score']:.4f}  recency={details['recency_score']:.4f}  "
              f"imp_w={details['importance_weight']:.4f}  | {mem.content[:30]}")
    print()
    print("   retrieve('散步', limit=3)（精准匹配含'散步'的记忆）:")
    results = epi_mod.retrieve("散步", limit=3)
    for score, base_rel, mem, details in results:
        print(f"   [{details['method']:8s}] score={score:.4f}  "
              f"vec={details['vector_score']:.4f}  tfidf={details['tfidf_score']:.4f}  "
              f"kw={details['keyword_score']:.4f}  recency={details['recency_score']:.4f}  "
              f"imp_w={details['importance_weight']:.4f}  | {mem.content[:30]}")

    # ========== 第七部分：语义记忆混合检索 ==========
    print()
    print("=" * 60)
    print("🧩 MemoryTool - 语义记忆混合检索演示")
    print("    知识图谱(实体+关系) + 嵌入向量/TF-IDF检索 + 混合排序")
    print("    评分公式: (向量相似度 × 0.7 + 图谱得分 × 0.3) × (0.8 + 重要性 × 0.4)")
    print("=" * 60)
    print()

    # --- 测试 0: 嵌入 API 可用性检测 ---
    print("--- 语义检索 0: 嵌入 API 可用性检测 ---")
    test_client = EmbeddingClient()
    if test_client.is_available:
        print(f"   ✅ 百炼嵌入 API 可用 (model={test_client.model})")
        test_emb = test_client.embed("测试文本")
        if test_emb:
            print(f"   ✅ API 调用成功，向量维度: {len(test_emb)}")
        else:
            print(f"   ⚠️ API 可用但调用失败，将回退 TF-IDF")
    else:
        print(f"   ⚠️ 百炼嵌入 API 不可用（DASHSCOPE_API_KEY 未配置），将使用 TF-IDF")
    print()

    # --- 测试 1: 实体/关系抽取 + 知识图谱构建 ---
    print("--- 语义检索 1: 实体/关系抽取 + 知识图谱构建 ---")
    print("   添加5条语义记忆（含技术概念和关系）...")
    sem_tool = MemoryTool()
    sem_tool.execute("add", content="Python是一种解释型编程语言，支持面向对象编程",
                     memory_type="semantic", importance=0.9)
    sem_tool.execute("add", content="Java也是一种面向对象的编程语言，使用JVM运行",
                     memory_type="semantic", importance=0.8)
    sem_tool.execute("add", content="机器学习是人工智能的子领域，使用Python进行开发",
                     memory_type="semantic", importance=0.9)
    sem_tool.execute("add", content="深度学习使用神经网络，是机器学习的一部分",
                     memory_type="semantic", importance=0.85)
    sem_tool.execute("add", content="今天午饭吃了面条", memory_type="semantic", importance=0.3)

    sem_mod = sem_tool._manager.get_module("semantic")
    print(f"   语义记忆数: {sem_mod.get_count()}")
    print(f"   知识图谱实体数: {sem_mod.get_entity_count()}")
    print(f"   知识图谱关系数: {sem_mod.get_relation_count()}")
    print(f"   实体列表: {[e.name for e in sem_mod.entities.values()]}")
    print()

    # --- 测试 2: 混合检索（向量 + 图谱）---
    print("--- 语义检索 2: 混合检索「Python机器学习」---")
    print("   （向量匹配语义相似 + 图谱匹配实体重叠）")
    result = sem_tool.execute("search", query="Python机器学习", memory_type="semantic", limit=5)
    print(result)
    print()

    # --- 测试 3: 图谱检索优势（实体连接但文本不相似）---
    print("--- 语义检索 3: 图谱检索「深度学习神经网络」---")
    print("   （图谱应通过实体关系找到关联记忆）")
    result = sem_tool.execute("search", query="深度学习神经网络", memory_type="semantic", limit=5)
    print(result)
    print()

    # --- 测试 4: 重要度过滤 ---
    print("--- 语义检索 4: 重要度过滤（min_importance=0.8）---")
    result = sem_tool.execute("search", query="编程语言", memory_type="semantic",
                              limit=5, min_importance=0.8)
    print(result)
    print()

    # --- 测试 5: 无关查询（午饭面条应排除技术记忆）---
    print("--- 语义检索 5: 无关查询「午饭面条」---")
    result = sem_tool.execute("search", query="午饭面条", memory_type="semantic", limit=5)
    print(result)
    print()

    # --- 测试 6: 直接调用 SemanticMemory.retrieve() + 知识图谱查询 ---
    print("--- 语义检索 6: 直接调用 SemanticMemory.retrieve() + 实体关系查询 ---")
    direct_sem_tool = MemoryTool()
    direct_sem_tool.execute("add", content="React是一种前端框架，使用JavaScript开发",
                            memory_type="semantic", importance=0.85)
    direct_sem_tool.execute("add", content="Vue也是一种前端框架，与React类似",
                            memory_type="semantic", importance=0.8)
    direct_sem_tool.execute("add", content="Django是Python的Web框架，使用ORM进行数据库操作",
                            memory_type="semantic", importance=0.75)
    direct_sem_tool.execute("add", content="TensorFlow是Google开发的机器学习框架",
                            memory_type="semantic", importance=0.9)

    sem_mod = direct_sem_tool._manager.get_module("semantic")
    print(f"   语义记忆数: {sem_mod.get_count()}")
    print(f"   实体数: {sem_mod.get_entity_count()}")
    print(f"   关系数: {sem_mod.get_relation_count()}")
    print()

    # 查询实体关系
    print("   实体 'python' 的关系:")
    relations = sem_mod.get_entity_relations("python")
    for rel in relations:
        print(f"     {rel['source_entity']} --[{rel['relation_type']}]--> {rel['target_entity']}  (weight={rel['weight']})")
    print()

    print("   retrieve('前端框架', limit=3):")
    results = sem_mod.retrieve("前端框架", limit=3)
    for score, base_rel, mem, details in results:
        print(f"   [{details['method']:8s}|{details.get('vector_method', 'tfidf'):8s}] score={score:.4f}  "
              f"vec={details['vector_score']:.4f}  graph={details['graph_score']:.4f}  "
              f"base={details['base_relevance']:.4f}  imp_w={details['importance_weight']:.4f}  "
              f"entities={details.get('matched_entities', [])}  | {mem.content[:35]}")
    print()
    print("   retrieve('机器学习', limit=3)（应匹配 TensorFlow 相关记忆）:")
    results = sem_mod.retrieve("机器学习", limit=3)
    for score, base_rel, mem, details in results:
        print(f"   [{details['method']:8s}|{details.get('vector_method', 'tfidf'):8s}] score={score:.4f}  "
              f"vec={details['vector_score']:.4f}  graph={details['graph_score']:.4f}  "
              f"base={details['base_relevance']:.4f}  imp_w={details['importance_weight']:.4f}  "
              f"entities={details.get('matched_entities', [])}  | {mem.content[:35]}")
    print()

    # --- 测试 7: 嵌入向量 vs TF-IDF fallback 对比 ---
    print("--- 语义检索 7: 嵌入向量 vs TF-IDF fallback 对比 ---")
    sem_mod2 = direct_sem_tool._manager.get_module("semantic")
    query = "深度学习用什么框架"
    print(f"   查询: {query}")
    print()

    # 当前检索（嵌入优先，不可用自动回退 TF-IDF）
    print("   [当前模式] 嵌入优先 + 自动回退:")
    results_normal = sem_mod2.retrieve(query, limit=3)
    for score, _, mem, details in results_normal:
        print(f"     [{details.get('vector_method', 'tfidf'):8s}] score={score:.4f}  "
              f"vec={details['vector_score']:.4f}  | {mem.content[:40]}")
    print()

    # 强制 TF-IDF 检索（临时禁用嵌入）
    print("   [强制 TF-IDF] 临时禁用嵌入向量:")
    saved_embeddings = sem_mod2._embeddings.copy()
    sem_mod2._embeddings.clear()
    results_tfidf = sem_mod2.retrieve(query, limit=3)
    for score, _, mem, details in results_tfidf:
        print(f"     [{details.get('vector_method', 'tfidf'):8s}] score={score:.4f}  "
              f"vec={details['vector_score']:.4f}  | {mem.content[:40]}")
    sem_mod2._embeddings = saved_embeddings  # 恢复
    print()

    # 对比总结
    if results_normal and results_normal[0][3].get("vector_method") == "embedding":
        print("   📊 对比: 嵌入向量检索成功，结果排序可能与 TF-IDF 不同")
        print("      嵌入向量能捕捉语义相似度（如「深度学习」≈「机器学习」）")
        print("      TF-IDF 仅依赖词汇重叠，语义理解能力较弱")
    else:
        print("   📊 对比: 嵌入 API 不可用，两种模式均使用 TF-IDF")
        print("      配置 DASHSCOPE_API_KEY 后可启用嵌入向量检索以提升语义理解")
    print()

    # --- 测试 8: 嵌入缓存验证 ---
    print("--- 语义检索 8: 嵌入缓存验证 ---")
    if test_client.is_available:
        import time
        text = "缓存测试文本"
        t1 = time.time()
        emb1 = test_client.embed(text)
        t1_elapsed = time.time() - t1
        t2 = time.time()
        emb2 = test_client.embed(text)
        t2_elapsed = time.time() - t2
        print(f"   首次调用: {t1_elapsed*1000:.1f}ms (API 请求)")
        print(f"   二次调用: {t2_elapsed*1000:.1f}ms (缓存命中)")
        print(f"   向量一致: {emb1 == emb2}")
    else:
        print("   ⚠️ 嵌入 API 不可用，跳过缓存验证")
    print()

    # ========== 第八部分：LLM 重排序检索（TF-IDF 粗筛 + LLM 精排）==========
    print()
    print("=" * 60)
    print("🎯 MemoryTool - LLM 重排序检索演示")
    print("    两阶段检索：TF-IDF 粗筛 → LLM 精排")
    print("=" * 60)
    print()

    # 准备测试数据：添加一批语义记忆
    rerank_tool = MemoryTool()
    test_memories = [
        ("Python 是一种解释型编程语言，广泛用于数据科学和 Web 开发", "semantic", 0.9),
        ("Java 是强类型编译型语言，常用于企业级应用开发", "semantic", 0.8),
        ("机器学习使用 Python 和 R 语言进行数据建模和预测", "semantic", 0.9),
        ("深度学习依赖 GPU 进行大规模矩阵运算训练神经网络", "semantic", 0.85),
        ("React 是前端框架，由 Meta 开发，使用 JSX 语法", "semantic", 0.8),
        ("Vue 是渐进式前端框架，支持模板语法和单文件组件", "semantic", 0.75),
        ("Django 是 Python Web 框架，内置 ORM 和管理后台", "semantic", 0.85),
        ("TensorFlow 是 Google 开发的深度学习框架", "semantic", 0.9),
        ("PyTorch 是 Meta 开发的深度学习框架，动态计算图", "semantic", 0.9),
        ("今天午饭吃了面条，味道不错", "semantic", 0.2),
        ("Docker 容器化技术用于应用打包和部署", "semantic", 0.8),
        ("Kubernetes 是容器编排平台，管理大规模容器集群", "semantic", 0.8),
        ("Redis 是内存键值数据库，常用于缓存和消息队列", "semantic", 0.7),
        ("MySQL 是关系型数据库，支持 SQL 查询和事务", "semantic", 0.7),
        ("Git 是分布式版本控制系统，用于代码协作", "semantic", 0.9),
        ("NLP 自然语言处理是 AI 的重要分支，研究文本理解", "semantic", 0.85),
        ("大语言模型 LLM 使用 Transformer 架构处理自然语言", "semantic", 0.9),
        ("FastAPI 是高性能 Python Web 框架，支持异步处理", "semantic", 0.8),
        ("Go 语言由 Google 开发，适合高并发网络服务", "semantic", 0.75),
        ("Rust 是系统编程语言，注重内存安全和性能", "semantic", 0.8),
    ]
    for content, mtype, imp in test_memories:
        rerank_tool.execute("add", content=content, memory_type=mtype, importance=imp)
    print(f"   已添加 {len(test_memories)} 条语义记忆用于测试")
    print()

    # --- 测试 1: 关键词抽取 ---
    print("--- 重排序 1: TF-IDF 关键词抽取 ---")
    query1 = "Python 框架用于 Web 开发"
    keywords = LLMReranker.extract_keywords(query1, top_k=5)
    print(f"   查询: {query1}")
    print(f"   关键词: {keywords}")
    print()

    # --- 测试 2: 基本 rerank 操作 ---
    print("--- 重排序 2: 基本 rerank 操作「Python 深度学习框架」---")
    query2 = "Python 深度学习框架"
    result = rerank_tool.execute("rerank", query=query2, coarse_limit=10, rerank_limit=5)
    print(result)
    print()

    # --- 测试 3: 对比 search vs rerank ---
    print("--- 重排序 3: search vs rerank 对比「前端框架」---")
    query3 = "前端框架"
    print("   [传统 search]:")
    result_search = rerank_tool.execute("search", query=query3, memory_type="semantic", limit=3)
    print(result_search)
    print()
    print("   [LLM rerank]:")
    result_rerank = rerank_tool.execute("rerank", query=query3, memory_type="semantic",
                                         coarse_limit=10, rerank_limit=3)
    print(result_rerank)
    print()

    # --- 测试 4: 语义理解验证（词汇不重叠但语义相关）---
    print("--- 重排序 4: 语义理解「容器技术」→ 应匹配 Docker/K8s ---")
    query4 = "容器技术"
    result = rerank_tool.execute("rerank", query=query4, coarse_limit=10, rerank_limit=3)
    print(result)
    print()

    # --- 测试 5: 无关查询排除 ---
    print("--- 重排序 5: 无关查询「午饭」→ LLM 应降低其相关性 ---")
    query5 = "午饭"
    result = rerank_tool.execute("rerank", query=query5, coarse_limit=10, rerank_limit=3)
    print(result)
    print()

    # --- 测试 6: 直接调用 LLMReranker ---
    print("--- 重排序 6: 直接调用 LLMReranker.rerank() ---")
    direct_reranker = LLMReranker()
    all_mems = rerank_tool._manager.get_all()
    direct_results = direct_reranker.rerank(all_mems, "数据存储", coarse_limit=8, rerank_limit=3)
    for i, (mem, relevance, reason) in enumerate(direct_results, 1):
        print(f"   [{i}] relevance={relevance:.2f}  reason={reason}  | {mem.content[:40]}")
    print()

    # --- 测试 7: 关键词抽取辅助验证 ---
    print("--- 重排序 7: 关键词抽取辅助验证 ---")
    test_queries = [
        "Python 机器学习框架",
        "容器化部署和编排",
        "前端用户界面开发",
        "版本控制和代码管理",
    ]
    for q in test_queries:
        kws = LLMReranker.extract_keywords(q, top_k=3)
        print(f"   '{q}' → {kws}")
    print()

    # ========== 第九部分：感知记忆多模态检索 ==========
    print()
    print("=" * 60)
    print("🌐 MemoryTool - 感知记忆多模态检索演示")
    print("    多模态存储 + 跨模态语义检索 + 时间近因性/重要性融合")
    print("    评分公式: (向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)")
    print("=" * 60)
    print()

    # --- 测试 1: 多模态添加（文本/图像/音频/视频/文档）---
    print("--- 感知记忆 1: 多模态添加（text/image/audio/video/document）---")
    perc_tool = MemoryTool()
    perc_adds = [
        ("会议纪要：讨论了Q3季度产品发布计划，重点在AI功能上线", "text", None),
        ("猫咪在窗台上晒太阳的照片", "image", "d:/photos/cat_sunny_window.jpg"),
        ("生日派对的现场录音，有欢快的背景音乐", "audio", "d:/recordings/birthday_party.mp3"),
        ("产品演示视频，展示新版本的用户界面交互", "video", "d:/videos/product_demo_ui.mp4"),
        ("用户调研报告，包含50份问卷统计分析", "document", "d:/docs/user_research_report.pdf"),
        ("神经网络训练日志，记录损失值变化曲线", "code", "d:/code/training_log.txt"),
        ("海边日落的照片，金黄色的天空", "image", "d:/photos/beach_sunset_golden.jpg"),
        ("咖啡馆的背景环境音", "audio", "d:/recordings/cafe_ambient.wav"),
    ]
    for content, modality, file_path in perc_adds:
        perc_tool.execute("add", content=content, memory_type="perceptual",
                          importance=0.7, file_path=file_path, modality=modality)
    perc_mod = perc_tool._manager.get_module("perceptual")
    print(f"   感知记忆数: {perc_mod.get_count()}")
    print(f"   模态分布: {perc_mod.get_modality_stats()}")
    print(f"   向量存储模态: {list(perc_mod.vector_stores.keys())}")
    print()

    # --- 测试 2: 跨模态语义检索（文本查询匹配图像/音频）---
    print("--- 感知记忆 2: 跨模态检索「宠物照片」→ 应匹配猫咪图片 ---")
    result = perc_tool.execute("search", query="宠物照片", memory_type="perceptual", limit=3)
    print(result)
    print()

    # --- 测试 3: 跨模态检索「派对氛围音乐」→ 应匹配生日录音 ---
    print("--- 感知记忆 3: 跨模态检索「派对氛围音乐」→ 应匹配生日录音 ---")
    result = perc_tool.execute("search", query="派对氛围音乐", memory_type="perceptual", limit=3)
    print(result)
    print()

    # --- 测试 4: 模态过滤（target_modality=image）---
    print("--- 感知记忆 4: 模态过滤（target_modality=image）「日落」---")
    result = perc_tool.execute("search", query="日落", memory_type="perceptual",
                               target_modality="image", limit=3)
    print(result)
    print()

    # --- 测试 5: 无关查询排除 ---
    print("--- 感知记忆 5: 无关查询「量子计算论文」---")
    result = perc_tool.execute("search", query="量子计算论文", memory_type="perceptual", limit=3)
    print(result)
    print()

    # --- 测试 6: 直接调用 PerceptualMemory.retrieve() + 时间近因性验证 ---
    print("--- 感知记忆 6: 直接调用 retrieve() + 时间近因性验证 ---")
    from datetime import timedelta
    direct_perc = MemoryTool()
    # 模拟一条很久以前的记忆（手动构造）
    old_entry = MemoryEntry(
        memory_id="old-perc-001",
        content="老照片：十年前的全家福",
        memory_type="perceptual",
        importance=0.5,
        timestamp=(datetime.now() - timedelta(days=365)).isoformat(),
        session_id="s1",
        modality="image",
        metadata={},
    )
    direct_perc._manager.get_module("perceptual").add(old_entry)
    direct_perc.execute("add", content="今天拍摄的风景照片", memory_type="perceptual",
                        importance=0.5, modality="image")
    perc_mod2 = direct_perc._manager.get_module("perceptual")
    print(f"   老记忆 recency: {perc_mod2._calculate_recency_score(old_entry.timestamp):.4f} (365天前)")
    print(f"   新记忆 recency: {perc_mod2._calculate_recency_score(direct_perc.get_all_memories()[-1].timestamp):.4f} (刚刚)")
    print()
    results = perc_mod2.retrieve("风景照片", limit=5)
    for score, base_rel, mem, details in results:
        print(f"   [{details['vector_method']:8s}] score={score:.4f}  "
              f"vec={details['vector_score']:.4f}  recency={details['recency_score']:.4f}  "
              f"modality={details['modality']}  | {mem.content[:30]}")
    print()

    # --- 测试 7: 直接调用 PerceptualMemory（无嵌入 API 场景：TF-IDF 回退验证）---
    print("--- 感知记忆 7: 语义文本构建与文件名关键词提取 ---")
    sample_entry = MemoryEntry(
        memory_id="sample-001",
        content="会议录音：产品评审",
        memory_type="perceptual",
        importance=0.8,
        timestamp=datetime.now().isoformat(),
        session_id="s1",
        file_path="d:/meetings/product_review_2026.mp3",
        modality="audio",
        metadata={"tags": ["会议", "评审"]},
    )
    sem_text = perc_mod._build_semantic_text(sample_entry)
    print(f"   语义文本: {sem_text}")
    print(f"   文件名关键词: {perc_mod._extract_file_keywords(sample_entry.file_path)}")
    print()

    print("=" * 60)
    print("✅ 全部演示完成")
    print("=" * 60)
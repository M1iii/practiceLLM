"""
PerceptualMemory：感知记忆模块
============================

多模态存储 + 跨模态语义检索 + 时间/重要性融合。
"""

import os
import re
import json
import math
import sqlite3
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from .base import MemoryEntry, MemoryModule
from .embedding_client import EmbeddingClient
from .working_memory import WorkingMemory


class PerceptualMemory(MemoryModule):
    """感知记忆模块 —— 多模态存储 + 跨模态语义检索 + 时间/重要性融合。

    特点：
      - 多模态支持：text / image / audio / video / code / document / data
      - 按模态分离的向量存储：vector_stores[modality] = {memory_id: embedding}
      - SQLite 持久化选项：记忆、嵌入向量、语义文本持久化，启动时自动加载
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

    def __init__(self, db_path: str = None):
        super().__init__("perceptual", None)
        self._embedding_client = EmbeddingClient()
        self.vector_stores: Dict[str, Dict[str, List[float]]] = {}
        self._semantic_texts: Dict[str, str] = {}
        self._memories: List[MemoryEntry] = []
        self._db_path = db_path

        if db_path:
            self._init_db()
            self._load_from_db()

    # ============================================================
    # SQLite 持久化
    # ============================================================

    def _init_db(self):
        """初始化 SQLite 数据库表。"""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS perceptual_memories (
                memory_id   TEXT PRIMARY KEY,
                content     TEXT,
                memory_type TEXT,
                importance  REAL,
                timestamp   TEXT,
                session_id  TEXT,
                file_path   TEXT,
                modality    TEXT,
                metadata    TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS perceptual_embeddings (
                memory_id   TEXT PRIMARY KEY,
                modality    TEXT,
                embedding   TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS perceptual_semantic_texts (
                memory_id TEXT PRIMARY KEY,
                text      TEXT
            )
        """)
        self._conn.commit()

    def _load_from_db(self):
        """从 SQLite 加载记忆、嵌入向量和语义文本到内存。"""
        # 加载记忆
        cursor = self._conn.execute("SELECT * FROM perceptual_memories")
        for row in cursor:
            entry = MemoryEntry(
                memory_id=row[0],
                content=row[1],
                memory_type=row[2] or "perceptual",
                importance=row[3] if row[3] is not None else 0.7,
                timestamp=row[4],
                session_id=row[5] or "default",
                file_path=row[6],
                modality=row[7],
                metadata=json.loads(row[8]) if row[8] else {},
            )
            self._memories.append(entry)

        # 加载嵌入向量
        cursor = self._conn.execute("SELECT * FROM perceptual_embeddings")
        for row in cursor:
            try:
                emb = json.loads(row[2])
                store = self._ensure_store(row[1])
                store[row[0]] = emb
            except (json.JSONDecodeError, TypeError):
                pass

        # 加载语义文本
        cursor = self._conn.execute("SELECT * FROM perceptual_semantic_texts")
        for row in cursor:
            self._semantic_texts[row[0]] = row[1]

        self._conn.commit()

    def _persist_memory_entry(self, entry: MemoryEntry):
        """持久化记忆条目。"""
        self._conn.execute(
            "INSERT OR REPLACE INTO perceptual_memories "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entry.memory_id, entry.content, entry.memory_type,
             entry.importance, entry.timestamp, entry.session_id,
             entry.file_path, entry.modality,
             json.dumps(entry.metadata, ensure_ascii=False))
        )

    def _persist_embedding(self, memory_id: str, modality: str,
                           embedding: List[float]):
        """持久化嵌入向量。"""
        self._conn.execute(
            "INSERT OR REPLACE INTO perceptual_embeddings VALUES (?, ?, ?)",
            (memory_id, modality, json.dumps(embedding))
        )

    def _persist_semantic_text(self, memory_id: str, text: str):
        """持久化语义文本。"""
        self._conn.execute(
            "INSERT OR REPLACE INTO perceptual_semantic_texts VALUES (?, ?)",
            (memory_id, text)
        )

    def _flush_db(self):
        """清空所有表并按当前状态重新写入。"""
        for table in ["perceptual_memories", "perceptual_embeddings",
                       "perceptual_semantic_texts"]:
            self._conn.execute(f"DELETE FROM {table}")
        for mem in self._memories:
            self._persist_memory_entry(mem)
        for mod, store in self.vector_stores.items():
            for mid, emb in store.items():
                self._persist_embedding(mid, mod, emb)
        for mid, text in self._semantic_texts.items():
            self._persist_semantic_text(mid, text)
        self._conn.commit()

    def close(self):
        """关闭数据库连接。"""
        if self._db_path and hasattr(self, '_conn') and self._conn:
            self._conn.close()
            self._conn = None

    # ============================================================
    # 增删操作（覆写基类）
    # ============================================================

    def add(self, entry: MemoryEntry) -> str:
        """添加感知记忆：存储 + 构建语义文本 + 编码向量 + 按模态归档 + 持久化。"""
        self._memories.append(entry)

        semantic_text = self._build_semantic_text(entry)
        self._semantic_texts[entry.memory_id] = semantic_text

        embedding = self._embedding_client.embed(semantic_text)
        if embedding:
            store = self._ensure_store(entry.modality or "text")
            store[entry.memory_id] = embedding

        if self._db_path:
            self._persist_memory_entry(entry)
            self._persist_semantic_text(entry.memory_id, semantic_text)
            if embedding:
                self._persist_embedding(entry.memory_id,
                                        entry.modality or "text", embedding)
            self._conn.commit()

        return ""

    def clear(self):
        """清空所有记忆和向量存储。"""
        self._memories.clear()
        self.vector_stores.clear()
        self._semantic_texts.clear()
        if self._db_path:
            self._flush_db()

    def set_memories(self, memories: List[MemoryEntry]):
        """直接替换记忆列表，重建向量存储。"""
        self._memories = list(memories)
        self.vector_stores.clear()
        self._semantic_texts.clear()

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

        if self._db_path:
            self._flush_db()

    def get_all(self) -> List[MemoryEntry]:
        return list(self._memories)

    def get_count(self) -> int:
        return len(self._memories)

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
            decay_factor = 0.1
            recency_score = math.exp(-decay_factor * age_hours / 24)
            return max(0.1, recency_score)
        except Exception:
            return 0.5

    # ============================================================
    # 混合检索
    # ============================================================

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[Tuple[float, float, MemoryEntry, Dict]]:
        """检索感知记忆：向量检索（跨模态）+ 时间近因性 + 重要性融合。

        kwargs:
          target_modality: 限定目标记忆模态（None=全部模态）
          min_importance: 最低重要度阈值
        """
        if not self._memories:
            return []

        target_modality = kwargs.get("target_modality")
        min_importance = kwargs.get("min_importance", 0.0)

        vector_scores, vector_method = self._vector_search(query, target_modality)

        scored = []
        for mem in self._memories:
            if mem.importance < min_importance:
                continue
            if target_modality and (mem.modality or "text") != target_modality:
                continue

            vec_score = vector_scores.get(mem.memory_id, 0.0)
            recency_score = self._calculate_recency_score(mem.timestamp)

            base_relevance = vec_score * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (mem.importance * 0.4)
            combined_score = base_relevance * importance_weight

            if combined_score > 0:
                details = {
                    "method": "hybrid" if vec_score > 0 else "keyword",
                    "vector_method": vector_method,
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
        if target_modality:
            candidate_ids = {
                mem.memory_id for mem in self._memories
                if (mem.modality or "text") == target_modality
            }
        else:
            candidate_ids = {mem.memory_id for mem in self._memories}

        if not candidate_ids:
            return {}, "tfidf"

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

            df = Counter()
            for doc_tokens in documents:
                for token in set(doc_tokens):
                    df[token] += 1

            total_docs = len(documents)
            idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1
                   for token, freq in df.items()}

            query_tf = Counter(query_tokens)
            query_len = len(query_tokens)
            query_vector = {token: (count / query_len) * idf.get(token, 0.0)
                            for token, count in query_tf.items()}

            if all(v == 0 for v in query_vector.values()):
                return {}

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

"""
EpisodicMemory：情景记忆模块
============================

SQLite / PostgreSQL 持久化 + 会话索引 + 混合/向量检索。
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from .base import MemoryEntry, MemoryModule
from .embedding_client import EmbeddingClient
from .working_memory import WorkingMemory
from src.core.storage import PostgreSQLBackend


class EpisodicMemory(MemoryModule):
    """情景记忆模块 —— SQLite / PostgreSQL 持久化 + 会话索引 + 混合/向量检索。

    特点：
      - SQLite 持久化存储（默认），支持崩溃恢复
      - PostgreSQL 可选（JSONB + pgvector），支持向量级联检索
      - 会话级索引（sessions），支持按会话检索
      - 混合检索：结构化过滤 + TF-IDF 语义向量检索（SQLite）
      - pgvector 余弦距离检索（PostgreSQL）
      - 嵌入向量缓存，与记忆条目一同持久化，避免重启后重复 API 调用
      - 评分公式：(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)
    """

    def __init__(self, db_path: str = ":memory:", pg_backend=None):
        super().__init__("episodic", None)
        self._pg_backend = pg_backend
        self.sessions: Dict[str, List[str]] = {}  # session_id -> [episode_id]
        self._memories: List[MemoryEntry] = []
        self._embeddings: Dict[str, List[float]] = {}  # memory_id -> embedding
        self._embedding_client = EmbeddingClient()

        if self._pg_backend:
            self.db_path = None
            self._conn = None
        else:
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
                metadata     TEXT,
                embedding    TEXT
            )
        """)
        # 兼容旧表：尝试添加 embedding 列（已存在则忽略）
        try:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN embedding TEXT")
        except sqlite3.OperationalError:
            pass  # 列已存在，忽略
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
            # 加载缓存的嵌入向量
            if len(row) > 9 and row[9]:
                try:
                    self._embeddings[entry.memory_id] = json.loads(row[9])
                except (json.JSONDecodeError, TypeError):
                    pass
        self._conn.commit()

    def _persist_episode(self, entry: MemoryEntry,
                         embedding: Optional[List[float]] = None):
        """将单条记忆写入 SQLite（含嵌入向量缓存）。"""
        emb_json = json.dumps(embedding) if embedding else None
        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, session_id, timestamp, content, importance,
                memory_type, modality, file_path, metadata, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry.memory_id, entry.session_id, entry.timestamp,
             entry.content, entry.importance, entry.memory_type,
             entry.modality, entry.file_path,
             json.dumps(entry.metadata, ensure_ascii=False), emb_json)
        )
        self._conn.commit()

    def _flush_db(self):
        """清空 SQLite 并按当前 _memories 重新写入（用于批量同步）。"""
        self._conn.execute("DELETE FROM episodes")
        for mem in self._memories:
            emb = self._embeddings.get(mem.memory_id)
            emb_json = json.dumps(emb) if emb else None
            self._conn.execute(
                """INSERT OR REPLACE INTO episodes
                   (episode_id, session_id, timestamp, content, importance,
                    memory_type, modality, file_path, metadata, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mem.memory_id, mem.session_id, mem.timestamp,
                 mem.content, mem.importance, mem.memory_type,
                 mem.modality, mem.file_path,
                 json.dumps(mem.metadata, ensure_ascii=False), emb_json)
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
        """添加情景记忆：内存 + 会话索引 + 嵌入向量缓存 + 持久化。"""
        self._memories.append(entry)
        self._index_session(entry)

        if self._pg_backend:
            self._pg_add(entry)
        else:
            embedding = self._embedding_client.embed(entry.content)
            if embedding:
                self._embeddings[entry.memory_id] = embedding
            self._persist_episode(entry, embedding=embedding)
        return ""

    def _pg_add(self, entry: MemoryEntry):
        """PostgreSQL 模式：写入 episodes 表（JSONB metadata + pgvector embedding）。"""
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

    def get_all(self) -> List[MemoryEntry]:
        return list(self._memories)

    def get_count(self) -> int:
        return len(self._memories)

    def get_embeddings(self) -> Dict[str, List[float]]:
        """获取缓存的所有嵌入向量 {memory_id: embedding}。"""
        return dict(self._embeddings)

    # ============================================================
    # 混合检索
    # ============================================================

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[Tuple[float, float, MemoryEntry, Dict]]:
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

        candidates = self._structured_filter(**kwargs)
        if not candidates:
            return []

        vector_scores = self._vector_search(query, candidates)
        tfidf_available = len(vector_scores) > 0

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

            recency_score = self._calculate_recency(mem.timestamp)
            base_relevance = effective_vec * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (mem.importance * 0.4)
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
                           **kwargs) -> List[Tuple[float, float, MemoryEntry, Dict]]:
        """PostgreSQL pgvector 模式检索：使用余弦距离运算符 <=>。"""
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

        time_start = kwargs.get("time_start")
        time_end = kwargs.get("time_end")
        if time_start or time_end:
            candidates = [m for m in candidates
                          if self._in_time_range(m.timestamp, time_start, time_end)]

        session_id = kwargs.get("session_id")
        if session_id:
            candidates = [m for m in candidates if m.session_id == session_id]

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
            return True

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

            from collections import Counter
            df = Counter()
            for doc_tokens in documents:
                for token in set(doc_tokens):
                    df[token] += 1

            total_docs = len(documents)
            import math
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

        import math
        return max(0.0, min(1.0, math.exp(-age_days / 30.0)))

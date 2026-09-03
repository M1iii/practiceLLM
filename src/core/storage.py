"""
PostgreSQL 存储后端（JSONB + pgvector）。

提供：
  - PostgreSQLBackend：连接管理 + 自动建表（JSONB + vector 列）
  - EpisodicMemory 通过注入后端实现 PostgreSQL 持久化

环境变量：
  PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class PostgreSQLBackend:
    """PostgreSQL 连接管理器，支持 JSONB 和 pgvector。

    pgvector 可选：当扩展未安装时，自动降级为内存余弦相似度检索。
    """

    VECTOR_DIM = 2560  # qwen3-vl-embedding 默认维度

    def __init__(self, host: str = None, port: int = None,
                 user: str = None, password: str = None, database: str = None):
        import psycopg2

        self.host = host or os.getenv("PG_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("PG_PORT", "5432"))
        self.user = user or os.getenv("PG_USER", "postgres")
        self.password = password or os.getenv("PG_PASSWORD", "")
        self.database = database or os.getenv("PG_DATABASE", "practice_llm")

        self.conn = psycopg2.connect(
            host=self.host, port=self.port,
            user=self.user, password=self.password,
            dbname=self.database,
        )
        self.conn.autocommit = True
        self._has_pgvector = self._init_vector_extension()
        self._init_episodes_table()

    def _init_vector_extension(self) -> bool:
        """尝试启用 pgvector 扩展。返回 True 表示可用。"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                return cur.fetchone() is not None
        except Exception:
            return False

    def _init_episodes_table(self):
        """创建 episodes 表（JSONB metadata + 可选 pgvector embedding）。"""
        if self._has_pgvector:
            vector_col = f"embedding    vector({self.VECTOR_DIM})"
        else:
            vector_col = "embedding    TEXT"
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id   TEXT PRIMARY KEY,
                    session_id   TEXT,
                    timestamp    TEXT,
                    content      TEXT,
                    importance   REAL,
                    memory_type  TEXT,
                    modality     TEXT,
                    file_path    TEXT,
                    metadata     JSONB DEFAULT '{{}}',
                    {vector_col}
                )
            """)
            if self._has_pgvector:
                # 清理旧索引（ivfflat/hnsw 均不支持 2560 维），使用精确余弦距离搜索
                cur.execute("DROP INDEX IF EXISTS idx_episodes_embedding")
                pass
            # 为 metadata 的 JSONB 路径查询创建 GIN 索引
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodes_metadata
                ON episodes USING gin (metadata)
            """)

    def execute(self, sql: str, params: tuple = ()) -> None:
        """执行无返回值的 SQL。"""
        with self.conn.cursor() as cur:
            cur.execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """查询单行，返回字典或 None。"""
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))

    def fetchall(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """查询多行，返回字典列表。"""
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, r)) for r in rows]

    def vector_search(self, embedding: List[float], limit: int = 5,
                      session_id: str = None) -> List[Dict[str, Any]]:
        """向量相似度检索。

        pgvector 可用时使用余弦距离 <=> 运算符，否则降级为内存余弦相似度。
        """
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

        if self._has_pgvector:
            return self._pgvector_search(vec_str, limit, session_id)
        return self._fallback_search(vec_str, limit, session_id)

    def _pgvector_search(self, vec_str: str, limit: int,
                         session_id: str = None) -> List[Dict[str, Any]]:
        """pgvector 余弦距离检索（排除 embedding 为 NULL 的行）。"""
        sql = """
            SELECT episode_id, session_id, timestamp, content, importance,
                   memory_type, modality, file_path, metadata::text,
                   (embedding <=> %s::vector) AS distance
            FROM episodes
            WHERE embedding IS NOT NULL
        """
        params: list = [vec_str]
        if session_id:
            sql += " AND session_id = %s"
            params.append(session_id)
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params += [vec_str, int(limit)]

        rows = self.fetchall(sql, tuple(params))
        for row in rows:
            row["metadata"] = json.loads(row["metadata"]) if row.get("metadata") else {}
        return rows

    def _fallback_search(self, vec_str: str, limit: int,
                         session_id: str = None) -> List[Dict[str, Any]]:
        """降级检索：加载所有行，在内存中计算余弦相似度。"""
        import math

        query_vec = [float(v) for v in vec_str.strip("[]").split(",")]

        sql = """SELECT episode_id, session_id, timestamp, content, importance,
                        memory_type, modality, file_path, metadata::text, embedding
                 FROM episodes"""
        params: list = []
        if session_id:
            sql += " WHERE session_id = %s"
            params.append(session_id)

        rows = self.fetchall(sql, tuple(params))
        if not rows:
            return []

        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = []
        for row in rows:
            emb_str = row.get("embedding")
            if not emb_str:
                continue
            try:
                emb_vec = json.loads(emb_str) if emb_str.startswith("[") else \
                    [float(v) for v in emb_str.strip("[]").split(",")]
            except (ValueError, json.JSONDecodeError):
                continue
            distance = 1.0 - cosine_similarity(query_vec, emb_vec)
            row["distance"] = distance
            row["metadata"] = json.loads(row["metadata"]) if row.get("metadata") else {}
            scored.append((distance, row))

        scored.sort(key=lambda x: x[0])
        return [r for _, r in scored[:limit]]

    def close(self):
        """关闭连接。"""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
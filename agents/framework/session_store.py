"""SessionStore：Agent 会话持久化（SQLite）。

持久化内容：
  - 会话元信息（session_id / agent_type / meta / created_at / updated_at）
  - 消息历史（role / content / extra / timestamp），按会话可整体恢复

特性：
  - SQLite 单文件存储（默认 data/sessions.db），线程安全（每操作独立连接）
  - 完整 CRUD：创建会话 / 追加消息 / 恢复历史 / 列出 / 删除 / 清空
  - 防无限增长：trim_messages() 裁剪旧消息（保留最近 N 条）
  - 断点续聊：重启后可用相同 session_id 恢复完整对话历史

使用方式:
    store = SessionStore()                       # data/sessions.db
    sid = store.create_session(agent_type="simple")
    store.append_message(sid, "user", "你好")
    store.append_message(sid, "assistant", "你好！有什么可以帮你")
    history = store.get_messages(sid)            # 恢复完整历史
"""

import sys
import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional


class SessionStore:
    """SQLite 会话持久化存储。"""

    def __init__(self, db_path: Optional[str] = None):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = db_path or os.path.join(root, "data", "sessions.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------
    # 底层
    # ------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL DEFAULT '',
                    meta TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    extra TEXT NOT NULL DEFAULT '{}',
                    timestamp TEXT NOT NULL
                )""")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id)""")

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    @staticmethod
    def _loads(text: str, default: Any = None) -> Any:
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return default

    # ------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------

    def create_session(self, session_id: Optional[str] = None,
                       agent_type: str = "",
                       meta: Optional[Dict[str, Any]] = None) -> str:
        """创建会话，返回 session_id（已存在则复用并更新 agent_type）。"""
        sid = session_id or f"session_{int(datetime.now().timestamp() * 1000)}"
        now = self._now()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, agent_type, meta, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    agent_type=excluded.agent_type,
                    meta=excluded.meta,
                    updated_at=excluded.updated_at
            """, (sid, agent_type, self._dumps(meta or {}), now, now))
        return sid

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话元信息（含消息数）。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id=?",
                (session_id,)).fetchone()["c"]
        return {
            "session_id": row["session_id"],
            "agent_type": row["agent_type"],
            "meta": self._loads(row["meta"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": count,
        }

    def list_sessions(self, agent_type: Optional[str] = None,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """列出会话（可按 agent_type 过滤，最新在前）。"""
        sql = "SELECT * FROM sessions"
        params: tuple = ()
        if agent_type:
            sql += " WHERE agent_type=?"
            params = (agent_type,)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(sql, params + (limit,)).fetchall()
        result = []
        for row in rows:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id=?",
                (row["session_id"],)).fetchone()["c"]
            result.append({
                "session_id": row["session_id"],
                "agent_type": row["agent_type"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": count,
            })
        return result

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其全部消息。"""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE session_id=?",
                               (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id=?",
                         (session_id,))
        return cur.rowcount > 0

    # ------------------------------------------------------------
    # 消息管理
    # ------------------------------------------------------------

    def append_message(self, session_id: str, role: str, content: str,
                       extra: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """追加一条消息，返回消息 id（会话不存在时返回 None）。"""
        if not self.session_exists(session_id):
            return None
        now = self._now()
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO messages (session_id, role, content, extra, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, content, self._dumps(extra or {}), now))
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now, session_id))
        return cur.lastrowid

    def get_messages(self, session_id: str, limit: Optional[int] = None,
                     role: Optional[str] = None) -> List[Dict[str, Any]]:
        """恢复会话消息历史（按时间正序；limit 取最近 N 条）。"""
        sql = "SELECT * FROM messages WHERE session_id=?"
        params: list = [session_id]
        if role:
            sql += " AND role=?"
            params.append(role)
        if limit is not None:
            # 取最近 limit 条（先倒序取再正序返回）
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(int(limit))
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            rows = list(reversed(rows))
        else:
            sql += " ORDER BY id ASC"
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        return [
            {"id": r["id"], "role": r["role"], "content": r["content"],
             "extra": self._loads(r["extra"], {}), "timestamp": r["timestamp"]}
            for r in rows
        ]

    def count_messages(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id=?",
                (session_id,)).fetchone()
        return row["c"] if row else 0

    def trim_messages(self, session_id: str, keep_last: int = 100) -> int:
        """裁剪会话旧消息（保留最近 keep_last 条），返回删除条数。"""
        with self._connect() as conn:
            cur = conn.execute("""
                DELETE FROM messages WHERE session_id=? AND id NOT IN (
                    SELECT id FROM messages WHERE session_id=?
                    ORDER BY id DESC LIMIT ?
                )
            """, (session_id, session_id, max(1, int(keep_last))))
        return cur.rowcount

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------

    def clear(self) -> None:
        """清空全部会话与消息。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM sessions")

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions"
                                    ).fetchone()["c"]
            messages = conn.execute("SELECT COUNT(*) AS c FROM messages"
                                    ).fetchone()["c"]
        return {"db_path": self.db_path, "sessions": sessions,
                "messages": messages}


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("💾 SessionStore 会话持久化演示")
    print("=" * 60)

    store = SessionStore(db_path=os.path.join(tempfile.gettempdir(),
                                              "session_demo.db"))
    store.clear()

    # 1. 创建会话并追加消息
    print("--- 1) 创建会话 + 追加消息 ---")
    sid = store.create_session(agent_type="simple", meta={"name": "演示会话"})
    print(f"   会话: {sid}")
    for role, content in [("user", "你好"), ("assistant", "你好！我是 SimpleAgent"),
                          ("user", "介绍一下 RAG"), ("assistant",
                          "RAG 是检索增强生成技术")]:
        mid = store.append_message(sid, role, content)
        print(f"   [{mid}] {role}: {content}")

    # 2. 恢复历史
    print("--- 2) 恢复完整历史 ---")
    for m in store.get_messages(sid):
        print(f"   [{m['id']}] {m['role']}: {m['content']}")

    # 3. 断点续聊（新实例同 session_id）
    print("--- 3) 断点续聊（重启后恢复）---")
    store2 = SessionStore(db_path=store.db_path)   # 模拟重启
    restored = store2.get_messages(sid)
    print(f"   新实例恢复 {len(restored)} 条消息（第1条: "
          f"{restored[0]['role']}: {restored[0]['content']}）")

    # 4. 会话查询 + 裁剪 + 删除
    print("--- 4) 查询 / 裁剪 / 删除 ---")
    print(f"   会话列表: {store2.list_sessions()}")
    store2.append_message(sid, "user", "补充第5条")
    trimmed = store2.trim_messages(sid, keep_last=2)
    print(f"   trim(keep_last=2) 删除 {trimmed} 条，剩余 "
          f"{store2.count_messages(sid)} 条")
    for m in store2.get_messages(sid):
        print(f"     [剩余] {m['role']}: {m['content']}")
    print(f"   删除会话: {store2.delete_session(sid)}")
    print(f"   删除后消息数: {store2.count_messages(sid)}")

    print(f"\n   stats: {store2.stats()}")
    print("\n✅ SessionStore 演示完成")

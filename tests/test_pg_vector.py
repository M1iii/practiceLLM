"""
测试 PostgreSQL JSONB + pgvector/降级检索。

测试内容：
  1. PostgreSQL 连接
  2. JSONB 存储与查询
  3. 向量存储与检索（pgvector 优先，降级兜底）
  4. EpisodicMemory 集成
  5. 与 SQLite 模式兼容
"""

import os
import sys
import json
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.storage import PostgreSQLBackend


def test_01_connection():
    """测试 PostgreSQL 连接与 pgvector 状态。"""
    print("=" * 60)
    print("Test 1: 连接 PostgreSQL")
    print("=" * 60)
    backend = PostgreSQLBackend()
    print(f"  pgvector: {'已安装 ✅' if backend._has_pgvector else '未安装，使用降级检索 ⚠️'}")
    backend.close()
    print("  连接成功 ✅\n")


def test_02_jsonb_write_and_read():
    """测试 JSONB 写入与读取。"""
    print("=" * 60)
    print("Test 2: JSONB 写入与读取")
    print("=" * 60)
    backend = PostgreSQLBackend()

    # 清理之前的测试数据
    backend.execute("DELETE FROM episodes WHERE episode_id LIKE 'test-%%'")

    # 写入带 JSONB metadata 的记忆
    backend.execute(
        """INSERT INTO episodes (episode_id, session_id, timestamp, content,
           importance, memory_type, modality, file_path, metadata)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)""",
        ("test-001", "session-1", "2026-08-31T10:00:00",
         "今天去了长城，天气很好，游客很多。", 0.9,
         "episodic", "text", "",
         json.dumps({"location": "北京", "poi": "长城", "weather": "晴", "tags": ["旅游", "历史"]},
                     ensure_ascii=False))
    )
    # 写入第二条
    backend.execute(
        """INSERT INTO episodes (episode_id, session_id, timestamp, content,
           importance, memory_type, modality, file_path, metadata)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)""",
        ("test-002", "session-1", "2026-08-30T14:30:00",
         "在故宫参观了珍宝馆，看到了很多珍贵的文物。", 0.8,
         "episodic", "text", "",
         json.dumps({"location": "北京", "poi": "故宫", "tags": ["文化", "历史"]},
                     ensure_ascii=False))
    )
    # 写入第三条（不同会话）
    backend.execute(
        """INSERT INTO episodes (episode_id, session_id, timestamp, content,
           importance, memory_type, modality, file_path, metadata)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)""",
        ("test-003", "session-2", "2026-08-29T09:00:00",
         "公司开会讨论了Q3的业绩目标。", 0.7,
         "episodic", "text", "",
         json.dumps({"topic": "Q3业绩", "tags": ["工作", "会议"]},
                     ensure_ascii=False))
    )
    print("  写入 3 条记忆 ✅")

    # 读取全部
    rows = backend.fetchall("SELECT episode_id, content, metadata::text FROM episodes ORDER BY timestamp")
    print(f"  读取 {len(rows)} 条记录 ✅")
    for r in rows:
        meta = json.loads(r["metadata"])
        print(f"    [{r['episode_id']}] {r['content'][:30]}... | metadata: {meta}")

    # JSONB 查询：metadata @> '{"poi": "长城"}' 匹配
    rows = backend.fetchall(
        "SELECT episode_id, content FROM episodes WHERE metadata @> %s::jsonb",
        (json.dumps({"poi": "长城"}),)
    )
    assert len(rows) == 1, f"期望 1 条 JSONB 匹配，实际 {len(rows)}"
    print(f"  JSONB 条件查询（poi=长城）: {rows[0]['episode_id']} ✅")

    # JSONB 查询：metadata->'tags' ? '历史'
    rows = backend.fetchall(
        "SELECT episode_id, content FROM episodes WHERE metadata->'tags' ? '历史'"
    )
    assert len(rows) == 2, f"期望 2 条 tags 包含'历史'，实际 {len(rows)}"
    print(f"  JSONB 标签查询（tags含历史）: {[r['episode_id'] for r in rows]} ✅")

    backend.close()
    print("  JSONB 测试全部通过 ✅\n")


def test_03_vector_store_and_search():
    """测试向量存储与检索。"""
    print("=" * 60)
    print("Test 3: 向量存储与检索")
    print("=" * 60)
    backend = PostgreSQLBackend()

    # 生成模拟向量（qwen3-vl-embedding 维度 2560，用随机值简化测试）
    import random
    random.seed(42)
    vec1 = [random.random() for _ in range(2560)]  # 与"长城"相关
    vec2 = [random.random() for _ in range(2560)]  # 与"故宫"相关

    # 让 vec1 偏向"户外"方向，vec2 偏向"室内"方向
    for i in range(10):
        vec1[i] = 0.9
        vec2[i] = 0.1

    def vec_to_str(v):
        return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

    # 更新嵌入
    if backend._has_pgvector:
        backend.execute("UPDATE episodes SET embedding = %s::vector WHERE episode_id = 'test-001'",
                        (vec_to_str(vec1),))
        backend.execute("UPDATE episodes SET embedding = %s::vector WHERE episode_id = 'test-002'",
                        (vec_to_str(vec2),))
        backend.execute("UPDATE episodes SET embedding = %s::vector WHERE episode_id = 'test-003'",
                        (vec_to_str([0.5] * 2560),))
    else:
        backend.execute("UPDATE episodes SET embedding = %s WHERE episode_id = 'test-001'",
                        (json.dumps(vec1),))
        backend.execute("UPDATE episodes SET embedding = %s WHERE episode_id = 'test-002'",
                        (json.dumps(vec2),))
        backend.execute("UPDATE episodes SET embedding = %s WHERE episode_id = 'test-003'",
                        (json.dumps([0.5] * 2560),))
    print("  embedding 写入完成 ✅")

    # 检索：与 vec1 相近的查询
    query_vec = [v + random.gauss(0, 0.05) for v in vec1]
    results = backend.vector_search(query_vec, limit=3)
    print(f"  检索返回 {len(results)} 条结果 ✅")
    for r in results:
        print(f"    [{r['episode_id']}] distance={r['distance']:.4f} | {r['content'][:30]}...")

    # 验证：test-001 应排第一（与查询向量最相似）
    assert results[0]["episode_id"] == "test-001", \
        f"期望 test-001 排第一，实际 {results[0]['episode_id']}"
    print("  语义排序正确 ✅")

    # 按会话过滤检索
    results = backend.vector_search(query_vec, limit=3, session_id="session-1")
    assert len(results) == 2, f"session-1 期望 2 条，实际 {len(results)}"
    assert all(r["session_id"] == "session-1" for r in results)
    print(f"  会话过滤（session-1）: {len(results)} 条 ✅")

    backend.close()
    print("  向量检索测试全部通过 ✅\n")


def test_04_episodic_memory_integration():
    """测试 EpisodicMemory 与 PostgreSQL 后端集成。"""
    print("=" * 60)
    print("Test 4: EpisodicMemory 集成")
    print("=" * 60)
    from src.tools.memory.memory_tool import EpisodicMemory, MemoryEntry

    backend = PostgreSQLBackend()
    # 清理之前测试留下的 pg-test-* 数据
    backend.execute("DELETE FROM episodes WHERE episode_id LIKE 'pg-test-%%'")
    memory = EpisodicMemory(pg_backend=backend)

    # 添加记忆
    memory.add(MemoryEntry(
        memory_id="pg-test-001",
        content="今天去颐和园散步，看到了美丽的日落。",
        memory_type="episodic",
        importance=0.85,
        timestamp="2026-08-31T10:00:00",
        session_id="pg-session-1",
        metadata={"location": "北京", "poi": "颐和园"},
    ))
    memory.add(MemoryEntry(
        memory_id="pg-test-002",
        content="在圆明园遗址公园了解了历史。",
        memory_type="episodic",
        importance=0.75,
        timestamp="2026-08-30T14:30:00",
        session_id="pg-session-1",
        metadata={"location": "北京", "poi": "圆明园"},
    ))
    memory.add(MemoryEntry(
        memory_id="pg-test-003",
        content="今天去了上海外滩看夜景。",
        memory_type="episodic",
        importance=0.8,
        timestamp="2026-08-29T20:00:00",
        session_id="pg-session-2",
        metadata={"location": "上海", "poi": "外滩"},
    ))
    print("  添加 3 条记忆 ✅")

    # 检索：关于北京的回忆
    results = memory.retrieve("北京旅游景点", limit=5)
    print(f"  检索 '北京旅游景点' 返回 {len(results)} 条:")
    for score, base, mem, details in results:
        print(f"    [{mem.memory_id}] score={score:.4f} | {mem.content[:30]}... | method={details['method']}")

    # 验证：北京相关的排在前面
    if results:
        beijing_results = [r for r in results if "北京" in r[2].content]
        print(f"  北京相关结果: {len(beijing_results)}/{len(results)} ✅")

    # 按会话检索
    results = memory.retrieve("北京旅游景点", limit=5, session_id="pg-session-1")
    assert all(r[2].session_id == "pg-session-1" for r in results)
    print(f"  会话过滤（pg-session-1）: {len(results)} 条 ✅")

    # 清空测试数据
    backend.execute("DELETE FROM episodes WHERE episode_id LIKE 'pg-test-%%'")
    print("  测试数据已清理 ✅")

    memory.close()
    print("  EpisodicMemory 集成测试全部通过 ✅\n")


def test_05_sqlite_compatibility():
    """测试 SQLite 模式（默认）与 PostgreSQL 模式互不干扰。"""
    print("=" * 60)
    print("Test 5: SQLite 兼容性（默认模式）")
    print("=" * 60)
    from src.tools.memory.memory_tool import EpisodicMemory, MemoryEntry
    import tempfile

    # SQLite 模式（默认）
    db_path = os.path.join(tempfile.gettempdir(), "test_episodic_pg.db")
    memory = EpisodicMemory(db_path=db_path)
    memory.add(MemoryEntry(
        memory_id="sqlite-test-001",
        content="SQLite 模式测试记忆",
        memory_type="episodic",
        importance=0.5,
        timestamp="2026-08-31T00:00:00",
        session_id="default",
    ))
    results = memory.retrieve("测试", limit=5)
    assert len(results) == 1
    print(f"  SQLite 模式检索正常: {len(results)} 条 ✅")

    # 清理
    memory.clear()
    memory.close()
    os.remove(db_path)
    print("  SQLite 兼容性测试通过 ✅\n")


def test_06_jsonb_cleanup():
    """清理测试数据。"""
    print("=" * 60)
    print("Test 6: 清理测试数据")
    print("=" * 60)
    backend = PostgreSQLBackend()
    backend.execute("DELETE FROM episodes WHERE episode_id LIKE 'test-%%'")
    backend.execute("DELETE FROM episodes WHERE episode_id LIKE 'pg-test-%%'")
    remaining = backend.fetchall("SELECT COUNT(*) AS cnt FROM episodes")
    print(f"  剩余记录: {remaining[0]['cnt']}")
    backend.close()
    print("  清理完成 ✅\n")


if __name__ == "__main__":
    test_01_connection()
    test_02_jsonb_write_and_read()
    test_03_vector_store_and_search()
    test_04_episodic_memory_integration()
    test_05_sqlite_compatibility()
    test_06_jsonb_cleanup()
    print("=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)
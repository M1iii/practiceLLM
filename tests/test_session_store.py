"""会话持久化测试（使用临时数据库，无网络）。"""
from src.agents.framework.session_store import SessionStore


def test_create_and_get_session(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "t.db"))
    sid = store.create_session(agent_type="simple", meta={"name": "测试"})
    info = store.get_session(sid)
    assert info["agent_type"] == "simple"
    assert info["meta"]["name"] == "测试"
    assert info["message_count"] == 0


def test_append_and_get_messages(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "t.db"))
    sid = store.create_session()
    store.append_message(sid, "user", "你好")
    store.append_message(sid, "assistant", "你好！", extra={"status": "success"})
    msgs = store.get_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "你好"
    assert msgs[1]["extra"]["status"] == "success"
    # 按角色过滤 + 最近 N 条
    assert len(store.get_messages(sid, role="user")) == 1
    assert len(store.get_messages(sid, limit=1)) == 1


def test_restore_continuity(tmp_path):
    """断点续聊：新实例同库同会话恢复完整历史。"""
    db = str(tmp_path / "t.db")
    store1 = SessionStore(db_path=db)
    sid = store1.create_session()
    store1.append_message(sid, "user", "第1轮")
    store1.append_message(sid, "assistant", "回答1")

    store2 = SessionStore(db_path=db)      # 模拟重启
    restored = store2.get_messages(sid)
    assert len(restored) == 2
    assert restored[1]["content"] == "回答1"


def test_trim_messages(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "t.db"))
    sid = store.create_session()
    for i in range(5):
        store.append_message(sid, "user", f"m{i}")
    deleted = store.trim_messages(sid, keep_last=2)
    assert deleted == 3
    remaining = store.get_messages(sid)
    assert [m["content"] for m in remaining] == ["m3", "m4"]


def test_delete_and_clear(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "t.db"))
    sid = store.create_session()
    store.append_message(sid, "user", "x")
    assert store.delete_session(sid) is True
    assert store.count_messages(sid) == 0
    assert store.get_session(sid) is None


def test_append_to_nonexistent_session(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "t.db"))
    assert store.append_message("ghost", "user", "x") is None


def test_stats(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "t.db"))
    s1 = store.create_session()
    store.append_message(s1, "user", "a")
    assert store.stats()["sessions"] == 1
    assert store.stats()["messages"] == 1

"""SessionStore 极端场景测试。"""
import sys, uuid
sys.path.insert(0, "d:\\pycharm_project\\practiceLLM")
from src.agents.framework.session_store import SessionStore

store = SessionStore()

# 1. 空会话
s = store.get_session("nonexistent")
assert s is None
print("[OK] 获取不存在会话返回None")

# 2. 空会话列表
sessions = store.list_sessions()
assert isinstance(sessions, list)
print(f"[OK] 空列表: {len(sessions)}")

# 3. 创建会话（API: meta= 而非 metadata=）
sid = str(uuid.uuid4())
s = store.create_session(session_id=sid, meta={"key": "val"})
assert s == sid  # create_session 返回 session_id
print("[OK] 创建会话")

# 4. 添加空消息
msg_id = store.append_message(session_id=sid, role="user", content="")
assert msg_id is not None
print("[OK] 空消息添加成功")

# 5. 超大消息
large = "x" * 100000
msg_id = store.append_message(session_id=sid, role="user", content=large)
assert msg_id is not None
print("[OK] 超大消息（10万字符）")

# 6. 获取会话历史
msgs = store.get_messages(sid)
assert len(msgs) >= 2
print(f"[OK] 会话历史: {len(msgs)} 条")

# 7. 删除会话
ok = store.delete_session(sid)
assert ok == True
assert store.get_session(sid) is None
print("[OK] 删除会话")

# 8. 重复删除
ok = store.delete_session(sid)
assert ok == False
print("[OK] 重复删除返回False")

# 9. 大量会话
sids = []
for i in range(50):
    sid = str(uuid.uuid4())
    store.create_session(session_id=sid, meta={"index": i})
    sids.append(sid)
sessions = store.list_sessions()
assert len(sessions) >= 50
print(f"[OK] 50个会话: {len(sessions)}")

# 10. 特殊字符消息
msg_id = store.append_message(session_id=sids[0], role="user",
                        content="<script>alert('test')</script>\n```\nprint('hello')\n```")
assert msg_id is not None
print("[OK] 特殊字符消息")

# 11. 元数据持久化验证
s = store.get_session(sids[0])
assert "meta" in s
print(f"[OK] 会话元数据: {s['meta']}")

# 12. 追加到不存在会话
msg_id = store.append_message("nonexistent_session", "user", "test")
assert msg_id is None
print("[OK] 追加到不存在会话返回None")

# 13. 裁剪消息
for i in range(200):
    store.append_message(session_id=sids[0], role="user", content=f"消息{i}")
deleted = store.trim_messages(sids[0], keep_last=10)
assert deleted > 0
msgs = store.get_messages(sids[0])
assert len(msgs) <= 10
print(f"[OK] 裁剪消息: 删除{deleted}条, 剩余{len(msgs)}条")

# 14. 清空
store.clear()
assert len(store.list_sessions()) == 0
print("[OK] 清空全部")

# 15. 自动生成 session_id
sid = store.create_session()
assert sid.startswith("session_")
print("[OK] 自动生成 session_id")

print("--- SessionStore 极端场景完成 ---")
"""MemoryTool 极端场景测试。"""
import sys, uuid
sys.path.insert(0, "d:\\pycharm_project\\practiceLLM")
from src.tools.memory.memory_tool import MemoryTool

tool = MemoryTool(session_id="edge_test")

# 1. 无效操作
r = tool.run({"action": "unknown"})
assert "未知操作" in r
print("[OK] 无效操作返回错误")

# 2. 空内容添加
r = tool.run({"action": "add", "content": ""})
assert "不能为空" in r
print("[OK] 空内容添加拒绝")

# 3. 无效类型
r = tool.run({"action": "add", "content": "测试", "memory_type": "invalid"})
assert "无效的 memory_type" in r
print("[OK] 无效类型拒绝")

# 4. 空搜索
r = tool.run({"action": "search", "query": ""})
assert "需要 query" in r
print("[OK] 空搜索拒绝")

# 5. 搜索无匹配（空库）
r = tool.run({"action": "search", "query": "不存在的查询"})
assert "记忆库为空" in r or "未找到匹配" in r
print("[OK] 空库搜索")

# 6. 添加并检索
for i in range(5):
    tool.run({"action": "add", "content": f"记忆内容{i}", "importance": i * 0.2})
r = tool.run({"action": "search", "query": "记忆内容", "limit": 3})
assert "找到" in r
print("[OK] 添加并检索正常")

# 7. 重要性过滤
r = tool.run({"action": "search", "query": "记忆", "min_importance": 0.6})
assert "找到" in r
print("[OK] 重要性过滤")

# 8. 遗忘策略
r = tool.run({"action": "forget", "strategy": "importance", "threshold": 0.3})
assert "遗忘完成" in r
print("[OK] 遗忘策略")

# 9. 无效遗忘策略
r = tool.run({"action": "forget", "strategy": "invalid"})
assert "未知遗忘策略" in r
print("[OK] 无效遗忘策略")

# 10. 整合操作
tool.run({"action": "add", "content": "待整合的内容", "importance": 0.8, "memory_type": "working"})
r = tool.run({"action": "consolidate", "from_type": "working", "to_type": "episodic", "importance_threshold": 0.5})
assert "整合完成" in r
print("[OK] 整合操作")

# 11. 重排序
r = tool.run({"action": "rerank", "query": "测试"})
print("[OK] 重排序")

# 12. 超大内容
r = tool.run({"action": "add", "content": "x" * 10000})
assert "记忆已添加" in r
print("[OK] 超大内容（1万字符）")

# 13. 特殊字符
r = tool.run({"action": "add", "content": "<script>alert('xss')</script>\n```python\nprint('hello')\n```"})
assert "记忆已添加" in r
print("[OK] 特殊字符记忆")

# 14. 多类型记忆
for t in ["working", "episodic", "semantic", "perceptual"]:
    r = tool.run({"action": "add", "content": f"{t}记忆", "memory_type": t})
    assert "记忆已添加" in r
print("[OK] 多类型记忆添加")

# 15. 按类型搜索
r = tool.run({"action": "search", "query": "记忆", "memory_type": "semantic"})
print("[OK] 按类型搜索")

print("--- MemoryTool 极端场景完成 ---")
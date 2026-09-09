"""NoteTool 极端场景测试。"""
import sys, os, tempfile, shutil
sys.path.insert(0, "d:\\pycharm_project\\practiceLLM")
from src.tools.memory.note_tool import NoteTool

demo_dir = os.path.join(tempfile.gettempdir(), "edge_note_tool")
shutil.rmtree(demo_dir, ignore_errors=True)
tool = NoteTool(notes_dir=demo_dir)

# 1. 空标题
r = tool.create_note(title="", content="内容")
assert "需要提供 title" in r
print("[OK] 空标题拒绝")

# 2. 无效类型
r = tool.create_note(title="测试", content="内容", note_type="invalid")
assert "无效的 note_type" in r
print("[OK] 无效类型拒绝")

# 3. 空搜索
r = tool.search_notes(query="")
assert "需要提供 query" in r
print("[OK] 空搜索返回错误")

# 4. 搜索不存在的笔记
r = tool.search_notes(query="不存在的关键词", limit=10)
assert "无匹配" in r or "无匹配" in r
print("[OK] 无匹配搜索返回空结果")

# 5. 读取不存在的笔记
r = tool.read_note("nonexistent_id")
assert "未找到笔记" in r
print("[OK] 读取不存在笔记")

# 6. 更新不存在的笔记
r = tool.update_note(note_id="nonexistent", title="新标题")
assert "未找到笔记" in r
print("[OK] 更新不存在笔记")

# 7. 不提供任何更新字段
tool.create_note(title="更新测试", content="内容")
note_id = list(tool.index.keys())[0]
r = tool.update_note(note_id=note_id)
assert "未提供任何更新字段" in r
print("[OK] 空更新拒绝")

# 8. 删除不存在的笔记
r = tool.delete_note("nonexistent")
assert "未找到笔记" in r
print("[OK] 删除不存在笔记")

# 9. 超大笔记内容
large_content = "测试内容\n" * 10000
r = tool.create_note(title="超大笔记", content=large_content)
assert "已创建笔记" in r
r = tool.read_note(note_id=list(tool.index.keys())[-1])
assert "超大笔记" in r
print("[OK] 超大笔记（10万字符）读写正常")

# 10. 大量笔记
for i in range(100):
    tool.create_note(title=f"批量笔记{i}", content=f"内容{i}", note_type="reference")
assert len(tool.index) >= 100
print(f"[OK] 100条笔记: index={len(tool.index)}")

# 11. 复杂标签
r = tool.create_note(title="标签测试", content="内容",
                     tags=["中文", "tag-with-dash", "tag with space", "CAPITAL", "重复,逗号"])
assert "已创建笔记" in r
print("[OK] 复杂标签创建成功")

# 12. 特殊字符
r = tool.create_note(title="特殊<字符> & '测试'", content="<script>alert('xss')</script>\n```python\nprint('hello')\n```")
assert "已创建笔记" in r
r = tool.read_note(note_id=list(tool.index.keys())[-1])
assert "特殊" in r
print("[OK] 特殊字符笔记读写正常")

# 13. 大规模重复搜索
for i in range(10):
    for j in range(5):
        r = tool.search_notes(query="批量笔记", limit=20)
        assert "命中" in r or "无匹配" in r
print(f"[OK] 50次重复搜索无异常")

# 14. 缓存一致性
before = tool._note_cache.stats()["size"]
tool.delete_note(note_id=note_id)
after = tool._note_cache.stats()["size"]
assert after <= before
print(f"[OK] 删除后缓存一致性: {before} -> {after}")

print("--- NoteTool 极端场景完成 ---")
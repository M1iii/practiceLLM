"""ToolFilter 极端场景测试。"""
import sys
sys.path.insert(0, "d:\\pycharm_project\\practiceLLM")
from src.tools.framework.tool_filter import ToolFilter

# 1. 空策略
f = ToolFilter()
assert f.is_allowed("any_agent", "any_tool") == True
print("[OK] 空策略允许所有")

# 2. 拒绝指定工具
f.set_policy("agent_a", allow=["*"], deny=["危险工具"])
assert f.is_allowed("agent_a", "安全工具") == True
assert f.is_allowed("agent_a", "危险工具") == False
print("[OK] 拒绝指定工具")

# 3. 仅允许指定工具
f = ToolFilter()
f.set_policy("agent_b", allow=["工具1", "工具2"])
assert f.is_allowed("agent_b", "工具1") == True
assert f.is_allowed("agent_b", "工具3") == False
print("[OK] 仅允许指定工具")

# 4. 通配符允许
f = ToolFilter()
f.set_policy("agent_c", allow=["*"])
assert f.is_allowed("agent_c", "任何工具") == True
print("[OK] 通配符允许所有")

# 5. 拒绝优先（同时允许和拒绝）
f = ToolFilter()
f.set_policy("agent_d", allow=["*"], deny=["禁止工具"])
assert f.is_allowed("agent_d", "禁止工具") == False
assert f.is_allowed("agent_d", "其他工具") == True
print("[OK] 拒绝优先于允许")

# 6. get_visible 过滤
f = ToolFilter()
f.set_policy("agent_e", allow=["工具A", "工具B"])
visible = f.get_visible("agent_e", ["工具A", "工具B", "工具C"])
assert visible == ["工具A", "工具B"]
print("[OK] get_visible 正确过滤")

# 7. 不同代理类型不同策略
f = ToolFilter()
f.set_policy("admin", allow=["*"])
f.set_policy("user", allow=["工具A"])
assert f.is_allowed("admin", "任何工具") == True
assert f.is_allowed("user", "任何工具") == False
assert f.is_allowed("user", "工具A") == True
print("[OK] 不同代理类型不同策略")

# 8. 空工具名
f = ToolFilter()
f.set_policy("agent", allow=[""], deny=[""])
assert f.is_allowed("agent", "") == False
print("[OK] 空工具名处理")

# 9. None 策略（默认为允许）
f = ToolFilter()
assert f.is_allowed("nonexistent", "任何工具") == True
print("[OK] 无策略代理类型默认允许")

# 10. 大批量工具过滤
f = ToolFilter()
f.set_policy("batch", allow=[f"tool_{i}" for i in range(100)])
tools = [f"tool_{i}" for i in range(200)]
visible = f.get_visible("batch", tools)
assert len(visible) == 100
print(f"[OK] 200个工具中过滤出100个")

print("--- ToolFilter 极端场景完成 ---")
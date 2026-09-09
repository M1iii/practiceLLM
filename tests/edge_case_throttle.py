"""AdvancedSearchTool 节流机制极端场景测试。"""
import sys, time
sys.path.insert(0, "d:\\pycharm_project\\practiceLLM")
from src.tools.search.advanced_search_tool import AdvancedSearchTool

# 1. 零阈值（实际存储为 0，但立即触发节流）
tool = AdvancedSearchTool(max_consecutive_failures=0, reset_interval=10)
assert tool._max_consecutive_failures == 0
print("[OK] 零阈值=0（立即触发节流）")

# 2. 负恢复时间
tool = AdvancedSearchTool(max_consecutive_failures=3, reset_interval=-1)
print(f"[OK] 负恢复时间: reset_interval={tool._reset_interval}")

# 3. 连续失败到节流
tool = AdvancedSearchTool(max_consecutive_failures=2, reset_interval=60)
tool._record_failure()
tool._record_failure()
msg = tool._check_throttle()
assert msg is not None, "节流中应返回错误"
print("[OK] 连续失败→节流")

# 4. 成功重置
tool._record_success()
assert tool._consecutive_failures == 0
print("[OK] 成功重置计数器")

# 5. 自动恢复
tool = AdvancedSearchTool(max_consecutive_failures=2, reset_interval=0.1)
tool._record_failure()
tool._record_failure()
time.sleep(0.15)
msg = tool._check_throttle()
assert msg is None, f"恢复后应无错误, 实际: {msg}"
print("[OK] 自动恢复")

# 6. 大量节流/恢复循环
tool = AdvancedSearchTool(max_consecutive_failures=3, reset_interval=0.05)
for i in range(5):
    tool._record_failure()
    tool._record_failure()
    tool._record_failure()
    assert tool._check_throttle() is not None
    tool._record_success()
    assert tool._consecutive_failures == 0
print("[OK] 5次节流/恢复循环")

# 7. 终端错误信号
tool = AdvancedSearchTool(max_consecutive_failures=2, reset_interval=60)
tool._record_failure()
tool._record_failure()
r = tool.run({"query": "test"})
assert "【搜索不可用】" in r
assert "请勿重试" in r
print("[OK] 终端错误信号")

print("--- 节流机制极端场景完成 ---")
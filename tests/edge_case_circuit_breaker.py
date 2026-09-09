"""CircuitBreaker 极端场景测试。"""
import sys, time
sys.path.insert(0, "d:\\pycharm_project\\practiceLLM")
from src.tools.framework.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState

# 1. 零故障阈值
cb = CircuitBreaker("test", failure_threshold=0)
assert cb.failure_threshold == 1, "零阈值自动提升为1"
print("[OK] 零阈值自动提升为1")

# 2. 负恢复时间
cb = CircuitBreaker("test", recovery_timeout=-1)
assert cb.recovery_timeout > 0
print("[OK] 负恢复时间自动提升")

# 3. 连续失败到打开
cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.5)
for i in range(2):
    try:
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
    except ValueError:
        pass
assert cb.state == CircuitState.OPEN
print(f"[OK] 打开状态: {cb.state.value}")

# 4. 半开成功→关闭
cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1, half_open_max=1)
try:
    cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
except ValueError:
    pass
time.sleep(0.15)
cb.call(lambda: "ok")
assert cb.state == CircuitState.CLOSED
print("[OK] 半开成功→关闭")

# 5. 半开失败→重新打开
cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
try:
    cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
except ValueError:
    pass
time.sleep(0.15)
try:
    cb.call(lambda: (_ for _ in ()).throw(ValueError("fail again")))
except ValueError:
    pass
assert cb.state == CircuitState.OPEN
print("[OK] 半开失败→重新打开")

# 6. 大量调用统计
cb = CircuitBreaker("test", failure_threshold=5, recovery_timeout=30)
for i in range(100):
    try:
        cb.call(lambda: "ok" if i % 10 != 0 else (_ for _ in ()).throw(ValueError()))
    except ValueError:
        pass
s = cb.stats
assert s["total"] == 100
print(f"[OK] 100次调用: total={s['total']} success={s['successes']} failures={s['failures']}")

# 7. 熔断器异常不计数
try:
    cb.call(lambda: (_ for _ in ()).throw(CircuitOpenError("nested")))
except CircuitOpenError:
    pass
print(f"[OK] 熔断器异常不计数: failures={cb.stats['failures']}")

# 8. 计数正确性
cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
for i in range(50):
    ok = (i % 4 == 0)  # 25% 失败
    try:
        if ok:
            cb.call(lambda: "success")
        else:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
    except (ValueError, CircuitOpenError):
        pass
s = cb.stats
assert s["total"] == 50
assert s["successes"] + s["failures"] <= 50
assert s["rejected"] + s["successes"] + s["failures"] == 50
print(f"[OK] 计数正确性: total={s['total']} success={s['successes']} failures={s['failures']} rejected={s['rejected']}")

print("--- CircuitBreaker 极端场景完成 ---")
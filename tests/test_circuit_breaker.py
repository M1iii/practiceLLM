"""熔断器状态机测试。"""
import time
import pytest
from tools.framework.circuit_breaker import (
    CircuitBreaker, CircuitOpenError, CircuitState, circuit_breaker)


def test_initial_closed():
    cb = CircuitBreaker("t", failure_threshold=2, recovery_timeout=30)
    assert cb.state == CircuitState.CLOSED


def test_open_after_threshold_failures():
    cb = CircuitBreaker("t", failure_threshold=2, recovery_timeout=30)
    calls = []

    def flaky():
        calls.append(1)
        raise ValueError("boom")

    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(flaky)
    assert cb.state == CircuitState.OPEN
    # OPEN 快速失败，不再执行被保护函数
    with pytest.raises(CircuitOpenError):
        cb.call(flaky)
    assert len(calls) == 2           # 熔断后未再调用


def test_half_open_recovery():
    cb = CircuitBreaker("t", failure_threshold=2, recovery_timeout=0.2,
                        success_threshold=1)
    state = {"fail": 2}

    def flaky():
        if state["fail"] > 0:
            state["fail"] -= 1
            raise ValueError("temp")
        return "ok"

    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(flaky)
    assert cb.state == CircuitState.OPEN
    time.sleep(0.3)                  # 超过恢复时间 → 半开
    assert cb.call(flaky) == "ok"    # 半开试探成功 → 关闭
    assert cb.state == CircuitState.CLOSED


def test_half_open_failure_reopens():
    cb = CircuitBreaker("t", failure_threshold=2, recovery_timeout=0.1)
    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("x")))
    time.sleep(0.2)
    with pytest.raises(ValueError):   # 半开试探再次失败
        cb.call(lambda: (_ for _ in ()).throw(ValueError("x")))
    assert cb.state == CircuitState.OPEN


def test_explicit_check_and_record():
    """check + record_success/record_failure（ToolResponse 场景）。"""
    cb = CircuitBreaker("t", failure_threshold=3, recovery_timeout=30)
    for i in range(3):
        cb.check()
        cb.record_failure("tool_error")
    with pytest.raises(CircuitOpenError):
        cb.check()
    assert cb.status()["state"] == "open"


def test_record_success_resets_failures():
    cb = CircuitBreaker("t", failure_threshold=3)
    cb.check()
    cb.record_failure("e1")
    cb.check()
    cb.record_success()
    assert cb.status()["consecutive_failures"] == 0


def test_reset():
    cb = CircuitBreaker("t", failure_threshold=1)
    with pytest.raises(ValueError):
        cb.call(lambda: (_ for _ in ()).throw(ValueError("x")))
    assert cb.state == CircuitState.OPEN
    cb.reset()
    assert cb.state == CircuitState.CLOSED


def test_decorator():
    @circuit_breaker("deco", failure_threshold=1, recovery_timeout=30)
    def risky():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        risky()
    with pytest.raises(CircuitOpenError):
        risky()
    assert risky.circuit_breaker.status()["state"] == "open"

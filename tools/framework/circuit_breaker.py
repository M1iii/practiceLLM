"""CircuitBreaker：工具/服务调用的熔断器。

三态模型：
  CLOSED（关闭）    正常状态，调用放行；连续失败达到阈值 → OPEN
  OPEN（打开）      快速失败，直接拒绝调用（抛 CircuitOpenError）；
                    超过恢复时间后允许进入 HALF_OPEN 试探
  HALF_OPEN（半开） 放行少量试探请求；成功达到阈值 → CLOSED，
                    任一失败 → 立即回到 OPEN

用途：防止工具连续失败拖垮 Agent，给失败服务恢复时间，避免雪崩。

使用方式:
    breaker = CircuitBreaker("api", failure_threshold=3, recovery_timeout=30)
    result = breaker.call(risky_func, *args)     # 打开时抛 CircuitOpenError
    breaker.status()                              # 查看状态与统计
"""

import sys
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """熔断器打开时抛出的异常（快速失败）。"""


class CircuitBreaker:
    """三态熔断器。"""

    def __init__(self, name: str,
                 failure_threshold: int = 3,
                 recovery_timeout: float = 30.0,
                 success_threshold: int = 1,
                 half_open_max: int = 1):
        """
        Args:
            name: 熔断器名称（对应工具/服务）
            failure_threshold: 连续失败多少次后打开（默认 3）
            recovery_timeout: 打开后多少秒进入半开试探（默认 30s）
            success_threshold: 半开状态下连续成功多少次后关闭（默认 1）
            half_open_max: 半开状态同时放行的试探请求数（默认 1）
        """
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = max(0.1, float(recovery_timeout))
        self.success_threshold = max(1, int(success_threshold))
        self.half_open_max = max(1, int(half_open_max))

        self.state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._half_open_successes = 0
        self.last_error: Optional[str] = None
        # 统计
        self.stats = {"total": 0, "successes": 0, "failures": 0,
                      "rejected": 0, "opened": 0, "closed": 0}

    # ------------------------------------------------------------
    # 调用包装
    # ------------------------------------------------------------

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行被保护函数：打开时快速失败，半开时限制试探。

        Raises:
            CircuitOpenError: 熔断器打开（快速失败）
        """
        self.stats["total"] += 1
        self._maybe_transition_to_half_open()

        if self.state == CircuitState.OPEN:
            self.stats["rejected"] += 1
            raise CircuitOpenError(
                f"熔断器 '{self.name}' 已打开（连续 {self._consecutive_failures} 次失败），"
                f"请在 {self.recovery_timeout:g}s 后重试")

        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max:
                self.stats["rejected"] += 1
                raise CircuitOpenError(
                    f"熔断器 '{self.name}' 半开试探进行中，请稍后重试")
            self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
        except CircuitOpenError:
            raise
        except Exception as e:
            self._on_failure(e)
            raise
        else:
            self._on_success()
            return result

    # ------------------------------------------------------------
    # 状态转移
    # ------------------------------------------------------------

    def _maybe_transition_to_half_open(self):
        """OPEN 超过恢复时间 → HALF_OPEN。"""
        if (self.state == CircuitState.OPEN and self._opened_at is not None
                and time.time() - self._opened_at >= self.recovery_timeout):
            self.state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._half_open_successes = 0

    def _on_success(self):
        self.stats["successes"] += 1
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.success_threshold:
                self._close()
        elif self.state == CircuitState.CLOSED:
            self._consecutive_failures = 0

    def _on_failure(self, error: Exception):
        self.stats["failures"] += 1
        self.last_error = f"{type(error).__name__}: {error}"
        if self.state == CircuitState.HALF_OPEN:
            self._open()                     # 半开失败 → 立即回到打开
        elif self.state == CircuitState.CLOSED:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._open()

    def _open(self):
        self.state = CircuitState.OPEN
        self._opened_at = time.time()
        self.stats["opened"] += 1

    def _close(self):
        self.state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._opened_at = None
        self.stats["closed"] += 1

    # ------------------------------------------------------------
    # 查询与控制
    # ------------------------------------------------------------

    def check(self) -> None:
        """仅门控检查：OPEN 快速失败、OPEN 超时转 HALF_OPEN、半开限流。

        配合 record_success() / record_failure() 使用（适用于
        "调用不抛异常但返回失败结果"的场景，如 ToolResponse 协议）。

        Raises:
            CircuitOpenError: 熔断器打开（快速失败）
        """
        self._maybe_transition_to_half_open()
        if self.state == CircuitState.OPEN:
            self.stats["rejected"] += 1
            raise CircuitOpenError(
                f"熔断器 '{self.name}' 已打开（连续 {self._consecutive_failures} 次失败），"
                f"请在 {self.recovery_timeout:g}s 后重试")
        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max:
                self.stats["rejected"] += 1
                raise CircuitOpenError(
                    f"熔断器 '{self.name}' 半开试探进行中，请稍后重试")
            self._half_open_calls += 1

    def record_success(self) -> None:
        """记录一次成功（显式模式）。"""
        self.stats["successes"] += 1
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.success_threshold:
                self._close()
        elif self.state == CircuitState.CLOSED:
            self._consecutive_failures = 0

    def record_failure(self, error: str = "tool_error"):
        """将一次"结果性失败"计入熔断统计（用于异常被吞掉但返回失败结果的场景）。"""
        self.stats["failures"] += 1
        self.last_error = error
        if self.state == CircuitState.HALF_OPEN:
            self._open()                     # 半开失败 → 立即回到打开
        elif self.state == CircuitState.CLOSED:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._open()

    def status(self) -> Dict[str, Any]:
        """当前状态与统计。"""
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "opened_at": self._opened_at,
            "last_error": self.last_error,
            "stats": dict(self.stats),
        }

    def reset(self):
        """手动重置为关闭状态。"""
        self.state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open_calls = 0
        self._half_open_successes = 0
        self.last_error = None


def circuit_breaker(name: Optional[str] = None, **breaker_kwargs) -> Callable:
    """装饰器版熔断器。

    用法:
        @circuit_breaker("api", failure_threshold=2, recovery_timeout=5)
        def risky():
            ...
    """
    def decorator(func: Callable) -> Callable:
        cb = CircuitBreaker(name or func.__name__, **breaker_kwargs)

        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)

        wrapper.circuit_breaker = cb   # 可通过 wrapper.circuit_breaker.status() 查看
        return wrapper

    return decorator


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🛑 CircuitBreaker 熔断器演示")
    print("=" * 60)

    # 模拟不稳定服务：前 2 次失败（与阈值一致，之后恢复）
    class FlakyService:
        def __init__(self):
            self.calls = 0

        def invoke(self, value: int) -> int:
            self.calls += 1
            if self.calls <= 2:          # 前 2 次失败
                raise ConnectionError("上游服务暂时不可用")
            return value * 2

    svc = FlakyService()
    breaker = CircuitBreaker("flaky_api", failure_threshold=2,
                             recovery_timeout=1.0, success_threshold=1)

    def call_once(desc: str):
        try:
            result = breaker.call(svc.invoke, 21)
            print(f"   {desc:<34} → 成功 result={result}")
        except CircuitOpenError as e:
            print(f"   {desc:<34} → 🛑 快速失败: {str(e)[:40]}...")
        except ConnectionError as e:
            print(f"   {desc:<34} → 失败: {type(e).__name__}")

    print("\n--- 阶段1: 连续失败触发熔断 ---")
    call_once("调用1（失败1）")
    call_once("调用2（失败2）")
    print(f"   状态: {breaker.status()['state']}（连续失败 "
          f"{breaker.status()['consecutive_failures']} → 超过阈值2 打开）")
    call_once("调用3（熔断打开 → 快速失败）")
    call_once("调用4（熔断打开 → 快速失败）")

    print("\n--- 阶段2: 恢复时间后半开试探 ---")
    import time
    time.sleep(1.2)
    call_once("调用5（半开试探，服务已恢复 → 成功）")
    print(f"   状态: {breaker.status()['state']}（半开成功 → 关闭 ✅）")
    call_once("调用6（恢复正常）")

    print("\n--- 统计 ---")
    for k, v in breaker.status()["stats"].items():
        print(f"   {k}: {v}")
    print(f"   last_error: {breaker.status()['last_error']}")

    print("\n--- 装饰器用法 ---")
    @circuit_breaker("deco_api", failure_threshold=1, recovery_timeout=30)
    def risky():
        raise ValueError("deco 演示失败")

    try:
        risky()
    except ValueError as e:
        print(f"   首次调用抛 {e}（CLOSED 状态原样透传异常）")
    try:
        risky()
    except CircuitOpenError as e:
        print(f"   第二次调用 → {type(e).__name__}（已熔断 ✅）")
    print(f"   装饰器状态: {risky.circuit_breaker.status()['state']}")

    print("\n✅ CircuitBreaker 演示完成")

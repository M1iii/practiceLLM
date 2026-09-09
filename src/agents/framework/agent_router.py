"""AgentRouter 智能路由：根据用户请求自动选择最合适的 Agent 类型。

路由策略：
  1. LLM 分类路由（优先）：调用 LLM 对请求分类，返回对应 agent_type
  2. 特征收敛域路由（二级）：提取 4 维特征，匹配各 Agent 收敛域
  3. 兜底：匹配度低于阈值 → 默认类型（simple）

设计：
  - RouteDecision 记录路由决策（agent_type / method / confidence / reason）
  - LLM 不可用或分类失败时自动回退特征路由，保证路由始终可用
  - 特征路由由 TaskFeatureExtractor + FeatureBasedRouter 实现
  - 路由结果可被 run.py 的 --auto 模式使用（自动激活对应 Agent）

使用方式:
    router = AgentRouter(factory, llm_available=True)
    decision = router.route("帮我规划一下 RAG 系统的搭建步骤")
    print(decision.agent_type, decision.method, decision.confidence)
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple

from .feature_extractor import RouteDecision, TaskFeatureExtractor, FeatureBasedRouter


class AgentRouter:
    """智能路由：基于特征收敛域的路由。

    本地路由策略（无需 LLM）：
      1. TaskFeatureExtractor 提取 4 维特征
      2. FeatureBasedRouter 计算各 Agent 收敛域隶属度
      3. 匹配度低于 0.5 时回退 simple
    """

    def __init__(self, factory, llm_available: bool = False,
                 use_llm: bool = False,
                 default_type: str = "simple",
                 model: Optional[str] = None):
        """
        Args:
            factory: AgentFactory（用于校验路由目标类型是否可用）
            llm_available: 不再使用（强制不使用 LLM 分类）
            use_llm: 强制 False，只执行本地特征路由
            default_type: 兜底类型
            model: 不再使用
        """
        self.factory = factory
        self.llm_available = False
        self.use_llm = False  # 强制关闭 LLM 分类，只执行本地特征路由
        self.default_type = default_type
        self.model = None
        # 特征路由实例
        self._feature_router = FeatureBasedRouter(factory, None, default_type)

    # ------------------------------------------------------------
    # 路由主入口
    # ------------------------------------------------------------

    def route(self, query: str) -> RouteDecision:
        """路由：直接执行特征收敛域路由。"""
        query = (query or "").strip()
        if not query:
            return RouteDecision(self.default_type, "fallback", 0.0, "空请求")

        # 特征收敛域路由（匹配度低于 0.5 时自动回退 simple）
        return self._feature_router.route(query)

    # ------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        fstats = self._feature_router.stats()
        return {
            "llm_available": self.llm_available,
            "use_llm": self.use_llm,
            "default_type": self.default_type,
            "feature_router": fstats,
        }


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧭 AgentRouter 特征收敛域路由演示（纯本地）")
    print("=" * 60)

    from src.agents.framework.agent_framework import AgentFactory, BaseAgent

    factory = AgentFactory()
    for t in ("simple", "react", "reflection", "plan_and_solve",
              "function_call", "tree_of_thought", "echo", "reverse"):
        factory.register(t, type(t, (BaseAgent,), {"AGENT_TYPE": t,
                                                   "DESCRIPTION": t,
                                                   "_execute": lambda s, q, **k: q}))

    router = AgentRouter(factory)

    print("--- 特征收敛域路由 ---")
    samples = [
        "你好",
        "搜索一下今天北京天气",
        "反思一下刚才的代码哪里需要优化",
        "为什么会发生这种情况",
        "帮我规划一个电商系统的搭建步骤，输出表格",
        "帮我搜索一下最新的 Python 版本",
        "解决这个复杂的数学证明题",
        "请优化一下这段代码",
        "随便聊聊今天的新闻",
    ]
    for q in samples:
        d = router.route(q)
        print(f"   「{q[:28]:<28}」→ {d.agent_type:<16} [{d.method}] 置信度{d.confidence:.2f}")

    print("\n--- 空请求/兜底 ---")
    d = router.route("")
    print(f"   空请求 → {d.agent_type} [{d.method}]")
    print(f"   无特征请求 '12345' → {router.route('12345').agent_type} "
          f"[{router.route('12345').method}]")

    print(f"\n   Router 状态: {router.stats()}")
    print("\n✅ AgentRouter 演示完成")
"""AgentRouter 智能路由：根据用户请求自动选择最合适的 Agent 类型。

两级路由策略：
  1. LLM 分类路由（准确）：调用 LLM 对请求分类，返回对应 agent_type
  2. 关键词规则路由（快速，无 LLM）：按关键词表命中规则，带置信度
  3. 兜底：未命中任何规则 → 默认类型（simple）

设计：
  - RouteDecision 记录路由决策（agent_type / method / confidence / reason）
  - LLM 不可用或分类失败时自动回退关键词规则，保证路由始终可用
  - 路由结果可被 run.py 的 --auto 模式使用（自动激活对应 Agent）

使用方式:
    router = AgentRouter(factory, llm_available=True)
    decision = router.route("帮我规划一下 RAG 系统的搭建步骤")
    print(decision.agent_type, decision.method, decision.confidence)
"""

import sys
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple



@dataclass
class RouteDecision:
    """路由决策。"""
    agent_type: str                  # 选中的 Agent 类型
    method: str = "fallback"         # llm / keyword / fallback
    confidence: float = 0.0          # 置信度 0-1
    reason: str = ""                 # 决策说明


class AgentRouter:
    """智能路由：关键词规则 + LLM 分类两级路由。"""

    # 关键词规则（列表顺序即优先级，平局时靠前者胜出；
    # 强意图规则在前，simple 日常对话作为低优先兜底置后）
    RULES: List[Tuple[List[str], str, str]] = [
        (["规划", "步骤", "计划", "分步", "拆解", "方案", "流程",
          "怎么做", "如何完成", "task", "plan"], "plan_and_solve",
         "命中规划/步骤类关键词"),
        (["权衡", "多种可能", "可能性", "路径", "分支", "搜索空间",
          "备选", "决策树", "穷举"], "tree_of_thought",
         "命中多路径探索类关键词"),
        (["反思", "改进", "优化", "复盘", "检查", "评估", "审阅",
          "哪里不好", "怎么改"], "reflection",
         "命中反思/改进类关键词"),
        (["推理", "分析", "思考", "为什么", "如果", "推导", "论证",
          "推断", "逻辑"], "react",
         "命中推理分析类关键词"),
        (["搜索", "查询", "计算", "工具", "天气", "翻译", "换算",
          "查一下", "搜一下", "调用"], "function_call",
         "命中工具调用类关键词"),
        (["你好", "介绍", "闲聊", "聊天", "谢谢", "随便聊聊"], "simple",
         "命中日常对话类关键词"),
    ]

    # LLM 分类提示词模板
    CLASSIFY_PROMPT = """你是 Agent 路由分类器。根据用户请求，从下列 Agent 类型中选择最合适的一个，只输出类型名称（不带引号、不加说明）：
- simple: 日常对话、闲聊、一般性问答
- react: 需要推理、分析、推导的问题
- plan_and_solve: 需要规划步骤、分步执行的任务
- reflection: 需要反思、检查、改进的任务
- function_call: 需要调用工具（搜索/查询/计算/翻译等）
- tree_of_thought: 需要探索多种可能性、权衡路径的复杂问题

用户请求: {query}
"""

    def __init__(self, factory, llm_available: bool = False,
                 use_llm: bool = True,
                 default_type: str = "simple",
                 model: Optional[str] = None):
        """
        Args:
            factory: AgentFactory（用于校验路由目标类型是否可用）
            llm_available: LLM 是否可用（控制是否尝试 LLM 分类）
            use_llm: 是否启用 LLM 分类（True 时优先 LLM，失败回退关键词）
            default_type: 兜底类型
        """
        self.factory = factory
        self.llm_available = llm_available
        self.use_llm = use_llm and llm_available
        self.default_type = default_type
        self.model = model

    # ------------------------------------------------------------
    # 路由主入口
    # ------------------------------------------------------------

    def route(self, query: str) -> RouteDecision:
        """路由：LLM 分类 → 关键词规则 → 兜底。"""
        query = (query or "").strip()
        if not query:
            return RouteDecision(self.default_type, "fallback", 0.0, "空请求")

        # 1. LLM 分类路由（失败自动回退）
        if self.use_llm:
            decision = self._route_by_llm(query)
            if decision is not None:
                return decision

        # 2. 关键词规则路由
        decision = self._route_by_keyword(query)
        if decision is not None:
            return decision

        # 3. 兜底
        return RouteDecision(self.default_type, "fallback", 0.3,
                             f"未命中规则，兜底 {self.default_type}")

    # ------------------------------------------------------------
    # LLM 分类
    # ------------------------------------------------------------

    def _route_by_llm(self, query: str) -> Optional[RouteDecision]:
        """调用 LLM 分类请求。失败或解析失败返回 None（触发回退）。"""
        try:
            from src.core.llm import practiceLLM
            llm = practiceLLM(model=self.model)
            response = llm.invoke(
                [{"role": "user",
                  "content": self.CLASSIFY_PROMPT.format(query=query[:200])}],
                temperature=0)
            if not response:
                return None
            agent_type = self._normalize_type(response)
            if agent_type is None:
                return None
            return RouteDecision(agent_type, "llm", 0.9,
                                 f"LLM 分类为 {agent_type}")
        except Exception:
            return None

    def _normalize_type(self, response: str) -> Optional[str]:
        """从 LLM 响应中提取 agent_type（容错解析）。"""
        text = response.strip().lower()
        # 直接匹配
        available = list(self.factory.available_types())
        for t in available:
            if text == t or text.startswith(t) or text.endswith(t):
                return t
        # 从文本中查找任意可用类型名
        for t in available:
            if t in text:
                return t
        return None

    # ------------------------------------------------------------
    # 关键词规则
    # ------------------------------------------------------------

    def _route_by_keyword(self, query: str) -> Optional[RouteDecision]:
        """按关键词表匹配规则，返回带置信度的决策。"""
        low = query.lower()
        best: Optional[Tuple[int, int, str, str]] = None   # (命中数, 规则序号, 类型, 说明)
        for index, (keywords, agent_type, reason) in enumerate(self.RULES):
            hits = sum(1 for kw in keywords if kw.lower() in low)
            if hits == 0:
                continue
            if best is None or hits > best[0]:
                best = (hits, index, agent_type, reason)
        if best is None:
            return None
        hits, _index, agent_type, reason = best
        confidence = min(0.95, 0.6 + 0.15 * hits)     # 命中越多置信越高
        return RouteDecision(agent_type, "keyword", confidence,
                             f"{reason}（命中 {hits} 个关键词）")

    # ------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "llm_available": self.llm_available,
            "use_llm": self.use_llm,
            "default_type": self.default_type,
            "rules": len(self.RULES),
        }


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧭 AgentRouter 智能路由演示")
    print("=" * 60)

    from src.agents.framework.agent_framework import AgentFactory, BaseAgent

    factory = AgentFactory()
    for t in ("simple", "react", "reflection", "plan_and_solve",
              "function_call", "tree_of_thought", "echo", "reverse"):
        factory.register(t, type(t, (BaseAgent,), {"AGENT_TYPE": t,
                                                   "DESCRIPTION": t,
                                                   "_execute": lambda s, q, **k: q}))

    router = AgentRouter(factory, llm_available=False, use_llm=False)

    print("--- 1) 关键词规则路由 ---")
    samples = [
        "帮我规划一下 RAG 系统的搭建步骤",
        "搜索一下最新的 Python 版本",
        "反思一下刚才的方案哪里可以改进",
        "为什么大模型会有幻觉问题",
        "你好呀，今天天气不错",
        "这个复杂问题有哪些可能的解决路径",
        "随便聊聊",
    ]
    for q in samples:
        d = router.route(q)
        print(f"   「{q[:18]}...」→ {d.agent_type:<16} "
              f"[{d.method}] 置信度{d.confidence:.2f} | {d.reason}")

    print("\n--- 2) 空请求/兜底 ---")
    d = router.route("")
    print(f"   空请求 → {d.agent_type} [{d.method}]")
    print(f"   未命中规则请求 '12345' → {router.route('12345').agent_type} "
          f"[{router.route('12345').method}]")

    print("\n--- 3) LLM 分类路由（真实调用）---")
    from dotenv import load_dotenv
    load_dotenv()
    llm_ok = bool(os.getenv("LLM_MODEL_ID") and os.getenv("LLM_API_KEY")
                  and os.getenv("LLM_BASE_URL"))
    llm_router = AgentRouter(factory, llm_available=llm_ok, use_llm=llm_ok)
    print(f"   LLM 可用: {llm_ok} | use_llm: {llm_router.use_llm}")
    for q in ["请帮我规划一个三步骤的项目推进计划",
              "计算一下 128*37 等于多少"]:
        d = llm_router.route(q)
        print(f"   「{q}」→ {d.agent_type} [{d.method}] {d.reason}")

    print(f"\n   Router 状态: {router.stats()}")
    print("\n✅ AgentRouter 演示完成")
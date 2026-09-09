"""
TaskFeatureExtractor & FeatureBasedRouter
===========================================

任务特征提取器 + 基于收敛域的特征路由引擎。

流程：
  1. TaskFeatureExtractor 从用户请求中提取 4 维特征
  2. FeatureBasedRouter 计算各 Agent 收敛域隶属度，匹配最合适的范式

特征维度：
  - task_complexity (1~5) → 归一化 (0~1)
  - need_external_knowledge (0~1)
  - iteration_need (0~1)
  - output_structured (0~1)

收敛域匹配：
  每个 Agent 定义四维 [min, max] 收敛域 + priority。
  域隶属度用 SOFT_MARGIN=0.3 线性衰减，四维几何平均为整体匹配度。
  阈值 0.5 以下回退 SimpleAgent，平局按 priority 降序。

使用方式：
    extractor = TaskFeatureExtractor()
    feats = extractor.extract(query)          # -> dict
    router = FeatureBasedRouter(factory)
    decision = router.route(feats)            # -> RouteDecision
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# RouteDecision（路由决策数据结构）
# ============================================================

@dataclass
class RouteDecision:
    """路由决策。"""
    agent_type: str                  # 选中的 Agent 类型
    method: str = "fallback"         # llm / feature / fallback
    confidence: float = 0.0          # 置信度 0-1
    reason: str = ""                 # 决策说明


# ============================================================
# 四个特征维度的名称
# ============================================================

DIMENSION_NAMES = [
    "task_complexity",
    "need_external_knowledge",
    "iteration_need",
    "output_structured",
]

SOFT_MARGIN = 0.3
MATCH_THRESHOLD = 0.5


# ============================================================
# 六种思考范式的收敛域定义
# ============================================================
# 注意：键名必须与项目中实际注册的 Agent 类型名一致（都是小写）

PARADIGM_DOMAINS = {
    "simple": {
        "task_complexity":         (1, 2),
        "need_external_knowledge": (0, 0.3),
        "iteration_need":          (0, 0.2),
        "output_structured":       (0, 0.2),
        "priority": 1,
        "comment": "纯聊天/问答：低复杂度，无工具，无迭代，无格式",
    },
    "react": {
        "task_complexity":         (1, 3),
        "need_external_knowledge": (0.2, 1.0),
        "iteration_need":          (0, 0.3),
        "output_structured":       (0, 0.25),
        "priority": 3,
        "comment": "搜索/查询：need_tool≥0.2 + output_structured≤0.25",
    },
    "reflection": {
        "task_complexity":         (1, 3),
        "need_external_knowledge": (0, 0.3),
        "iteration_need":          (0.2, 1.0),
        "output_structured":       (0, 0.4),
        "priority": 4,
        "comment": "优化/反思：iteration≥0.2（1个迭代关键词命中）",
    },
    "plan_and_solve": {
        "task_complexity":         (2, 4),
        "need_external_knowledge": (0, 0.3),
        "iteration_need":          (0, 0.3),
        "output_structured":       (0.2, 1.0),
        "priority": 3,
        "comment": "规划/执行：complexity≥2 + structured≥0.2",
    },
    "function_call": {
        "task_complexity":         (1, 1),
        "need_external_knowledge": (0, 1.0),
        "iteration_need":          (0, 0.2),
        "output_structured":       (0.2, 0.4),
        "priority": 2,
        "comment": "精确计算/格式转换：structured∈[0.2,0.4] + 低复杂度 c=1",
    },
    "tree_of_thought": {
        "task_complexity":         (2, 5),
        "need_external_knowledge": (0, 0.3),
        "iteration_need":          (0, 0.3),
        "output_structured":       (0, 0.15),
        "priority": 4,
        "comment": "多路径探索：complexity≥2 + 结构化输出极低",
    },
}


# ============================================================
# TaskFeatureExtractor
# ============================================================

class TaskFeatureExtractor:
    """任务特征提取器：从用户请求中提取 4 维结构特征。"""

    # ---------- 复杂度关键词 ----------
    COMPLEXITY_INDICATORS = [
        "首先", "然后", "接着", "最后", "步骤", "规划", "方案", "系统",
        "拆解", "分步骤", "分步", "多步",
        "比较", "分析", "评估", "设计", "开发", "解决",
        "权衡", "取舍", "选择", "对比", "多种", "几种",
        "所有", "不同", "框架",
    ]

    # ---------- 外部知识/工具关键词 ----------
    TOOL_INDICATORS = [
        "搜索", "查询", "获取", "调用", "API",
        "查找", "价格",
        "查一下", "搜一下", "找一下", "汇率",
    ]

    # ---------- 迭代/反思关键词 ----------
    ITERATION_INDICATORS = [
        "优化", "润色", "改进", "审查", "验证",
        "检查", "迭代", "完善", "反思", "修复",
        "重写",
    ]

    # ---------- 结构化输出关键词 ----------
    STRUCTURE_INDICATORS = [
        "列表", "表格", "JSON", "格式", "结构化",
        "步骤", "模板", "框架", "代码", "计划",
        "规划", "设计",
        "计算", "翻译", "调用", "API", "生成",
    ]

    # ============================================================
    # 入口
    # ============================================================

    def extract(self, query: str) -> Dict[str, Any]:
        """提取 4 维特征，返回包含原始评分和归一化向量的字典。

        Returns:
            {
                "task_complexity": 3,               # 原始 1~5
                "need_external_knowledge": 0.0,     # 原始 0~1
                "iteration_need": 0.0,              # 原始 0~1
                "output_structured": 0.0,           # 原始 0~1
                "vector": [0.6, 0.0, 0.0, 0.0],    # 归一化向量（task_complexity/5）
                "reason": "...",                    # 诊断说明
            }
        """
        query = (query or "").strip()
        if not query:
            return self._make_result(1, 0, 0, 0, "空请求")

        task_complexity = self._score_complexity(query)
        need_external_knowledge = self._score_external(query)
        iteration_need = self._score_iteration(query)
        output_structured = self._score_structured(query)

        reason_parts = []
        reason_parts.append(f"复杂度={task_complexity}")
        if need_external_knowledge > 0:
            reason_parts.append(f"外部知识={need_external_knowledge:.2f}")
        if iteration_need > 0:
            reason_parts.append(f"迭代={iteration_need:.2f}")
        if output_structured > 0:
            reason_parts.append(f"结构化={output_structured:.2f}")

        return self._make_result(
            task_complexity,
            need_external_knowledge,
            iteration_need,
            output_structured,
            "; ".join(reason_parts) if reason_parts else "未命中任何规则",
        )

    # ============================================================
    # 各维度评分
    # ============================================================

    def _score_complexity(self, query: str) -> int:
        """任务复杂度评分 1~5。

        规则：
          - 基础分 1
          - 长度 > 30 字符 +1，> 80 字符 +1
          - 命中 1 个复杂度关键词 +1，命中 3 个 +1
          - 多个问句/句号（分割主题） +1
        """
        base = 1
        q = query.lower()

        # 长度加分（降低阈值，适配短查询）
        if len(q) > 30:
            base += 1
        if len(q) > 80:
            base += 1

        # 关键词加分
        hits = sum(1 for w in self.COMPLEXITY_INDICATORS if w in q)
        if hits >= 1:
            base += 1
        if hits >= 3:
            base += 1

        # 多问句/多主题检测：包含多个问号或句号表示多步骤
        sentence_ends = q.count("？") + q.count("?") + q.count("。")
        if sentence_ends >= 2:
            base += 1

        return max(1, min(5, base))

    def _score_external(self, query: str) -> float:
        """外部知识/工具调用需求评分 0~1。

        规则：min(hits × 0.3, 1.0)，1 个关键词命中 → 0.30。
        """
        hits = sum(1 for w in self.TOOL_INDICATORS if w in query.lower())
        return min(hits * 0.3, 1.0)

    def _score_iteration(self, query: str) -> float:
        """迭代/反思需求评分 0~1。

        规则：min(hits × 0.3, 1.0)，1 个关键词命中 → 0.30。
        """
        hits = sum(1 for w in self.ITERATION_INDICATORS if w in query.lower())
        return min(hits * 0.3, 1.0)

    def _score_structured(self, query: str) -> float:
        """结构化输出需求评分 0~1。

        规则：min(hits × 0.3, 1.0)，1 个关键词命中 → 0.30。
        """
        hits = sum(1 for w in self.STRUCTURE_INDICATORS if w in query.lower())
        return min(hits * 0.3, 1.0)

    # ============================================================
    # 辅助
    # ============================================================

    def _make_result(self, complexity: int, external: float,
                     iteration: float, structured: float,
                     reason: str) -> Dict[str, Any]:
        """构建特征结果字典。"""
        return {
            "task_complexity": complexity,
            "need_external_knowledge": round(external, 4),
            "iteration_need": round(iteration, 4),
            "output_structured": round(structured, 4),
            "vector": [
                complexity / 5.0,      # 归一化到 0~1
                round(external, 4),
                round(iteration, 4),
                round(structured, 4),
            ],
            "reason": reason,
        }


# ============================================================
# FeatureBasedRouter
# ============================================================

class FeatureBasedRouter:
    """基于收敛域的特征路由引擎。

    取代 AgentRouter 的关键词规则路由，作为 LLM 分类失败后的二级路由。
    """

    def __init__(self, factory, extractor: TaskFeatureExtractor = None,
                 default_type: str = "simple"):
        """
        Args:
            factory: AgentFactory（用于校验路由目标类型是否可用）
            extractor: TaskFeatureExtractor 实例
            default_type: 兜底类型
        """
        self.factory = factory
        self.extractor = extractor or TaskFeatureExtractor()
        self.default_type = default_type
        self._domains = dict(PARADIGM_DOMAINS)

    # ------------------------------------------------------------
    # 域管理
    # ------------------------------------------------------------

    def get_domains(self) -> Dict:
        """获取当前收敛域定义（只读副本）。"""
        return dict(self._domains)

    def update_domain(self, paradigm: str, dimension: str,
                      vmin: float, vmax: float) -> None:
        """更新单个收敛域范围。"""
        if paradigm in self._domains and dimension in self._domains[paradigm]:
            self._domains[paradigm][dimension] = (vmin, vmax)

    # ------------------------------------------------------------
    # 路由主入口
    # ------------------------------------------------------------

    def route(self, query_or_features) -> RouteDecision:
        """根据特征匹配合适的 Agent。

        Args:
            query_or_features: 用户请求字符串 或特征字典（含 vector）

        Returns:
            RouteDecision 路由决策
        """
        if isinstance(query_or_features, dict) and "vector" in query_or_features:
            features = query_or_features
        else:
            features = self.extractor.extract(query_or_features)

        vector = features["vector"]

        candidates = []
        for paradigm_name, domain in self._domains.items():
            score = self._match_score(vector, domain)
            candidates.append((paradigm_name, score, domain["priority"]))

        # 按 匹配度降序 → priority 降序 排序
        candidates.sort(key=lambda x: (-x[1], -x[2]))

        best_name, best_score, _ = candidates[0]

        if best_score >= MATCH_THRESHOLD:
            return RouteDecision(
                best_name, "feature", best_score,
                f"特征匹配: {best_name} (匹配度{best_score:.2f}) | {features.get('reason', '')}",
            )

        return RouteDecision(
            self.default_type, "fallback", best_score,
            f"无匹配(最高{best_score:.2f}<{MATCH_THRESHOLD})，兜底 {self.default_type}",
        )

    # ------------------------------------------------------------
    # 匹配计算
    # ------------------------------------------------------------

    @staticmethod
    def _domain_membership(value: float, vmin: float, vmax: float) -> float:
        """值在收敛域内的隶属度，越界线性衰减。"""
        if vmin <= value <= vmax:
            return 1.0
        if value < vmin:
            return max(0.0, 1.0 - (vmin - value) / SOFT_MARGIN)
        return max(0.0, 1.0 - (value - vmax) / SOFT_MARGIN)

    def _match_score(self, vector: List[float], domain: Dict) -> float:
        """计算特征向量与收敛域的 4 维几何平均匹配度。"""
        score = 1.0
        for i, dim in enumerate(DIMENSION_NAMES):
            vmin, vmax = domain[dim]
            value = vector[i]
            # task_complexity 已归一化，但域定义是 1~5，需转换
            if dim == "task_complexity":
                vmin_norm = vmin / 5.0
                vmax_norm = vmax / 5.0
                m = self._domain_membership(value, vmin_norm, vmax_norm)
            else:
                m = self._domain_membership(value, vmin, vmax)
            score *= m
        return score ** (1.0 / len(DIMENSION_NAMES))

    # ------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "domains_count": len(self._domains),
            "domains": list(self._domains.keys()),
            "default_type": self.default_type,
            "soft_margin": SOFT_MARGIN,
            "threshold": MATCH_THRESHOLD,
        }
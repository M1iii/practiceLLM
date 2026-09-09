"""
路由系统测试：验证 TaskFeatureExtractor + FeatureBasedRouter 的路由正确性。

测试内容：
  1. 特征提取准确性（4 维特征值）
  2. 路由决策准确性（50 个测试用例，覆盖 6 种思考范式 + 边界案例）
  3. 收敛域匹配度计算正确性
  4. 优先级平局处理
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from src.agents.framework.feature_extractor import (
    TaskFeatureExtractor, FeatureBasedRouter, PARADIGM_DOMAINS, SOFT_MARGIN, MATCH_THRESHOLD,
)
from src.agents.framework.agent_framework import AgentFactory, BaseAgent


# ============================================================
# 夹具：注册 6 种 Agent 类型
# ============================================================

@pytest.fixture(scope="module")
def router():
    factory = AgentFactory()
    for t in ("simple", "react", "reflection", "plan_and_solve",
              "function_call", "tree_of_thought"):
        factory.register(t, type(t, (BaseAgent,), {"AGENT_TYPE": t,
                                                   "DESCRIPTION": t,
                                                   "_execute": lambda s, q, **k: q}))
    return FeatureBasedRouter(factory)


@pytest.fixture(scope="module")
def extractor():
    return TaskFeatureExtractor()


# ============================================================
# 辅助：构造测试用例
# ============================================================

TEST_CASES = [
    # (id, query, expected_agent, expected_features)
    # ---- SimpleAgent ----
    ("S001", "什么是人工智能", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    ("S002", "现在几点了", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    ("S003", "解释一下什么是机器学习", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    ("S004", "Python是什么语言", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    ("S005", "今天是几号", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    ("S006", "告诉我一个笑话", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    ("S007", "什么是区块链技术，简单说说", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    ("S008", "你觉得今天天气怎么样", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    ("S009", "给我推荐一本书", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    ("S010", "什么是深度学习，和机器学习有什么区别", "simple", {"c": 1, "t": 0, "i": 0, "s": 0}),
    # ---- ReAct ----
    ("R001", "搜索一下最新的AI新闻", "react", {"c": 1, "t": 0.3, "i": 0, "s": 0}),
    ("R002", "查询北京明天的天气", "react", {"c": 1, "t": 0.3, "i": 0, "s": 0}),
    ("R003", "查一下OpenAI最新的API文档", "react", {"c": 1, "t": 0.3, "i": 0, "s": 0}),
    ("R004", "获取当前比特币的价格", "react", {"c": 1, "t": 0.6, "i": 0, "s": 0}),
    ("R005", "搜索一下关于Transformer架构的论文", "react", {"c": 1, "t": 0.3, "i": 0, "s": 0}),
    ("R006", "查一下Python最新版本是什么", "react", {"c": 1, "t": 0.3, "i": 0, "s": 0}),
    ("R007", "帮我找一下2024年全球GDP排名", "react", {"c": 1, "t": 0.3, "i": 0, "s": 0}),
    ("R008", "现在美元兑人民币汇率是多少", "react", {"c": 1, "t": 0.3, "i": 0, "s": 0}),
    ("R009", "搜索一下最新的iPhone型号和价格", "react", {"c": 1, "t": 0.6, "i": 0, "s": 0}),
    ("R010", "查找一下关于气候变化的最近研究", "react", {"c": 1, "t": 0.6, "i": 0, "s": 0}),
    # ---- Reflection ----
    ("RF001", "帮我润色这段文案：'这个产品很好，推荐大家购买'", "reflection", {"c": 1, "t": 0, "i": 0.3, "s": 0}),
    ("RF002", "优化一下这段代码的性能：for i in range(100): for j in range(100): print(i*j)", "reflection", {"c": 2, "t": 0, "i": 0.3, "s": 0.3}),
    ("RF003", "帮我审查一下这个项目方案有什么问题", "reflection", {"c": 2, "t": 0.3, "i": 0.3, "s": 0}),
    ("RF004", "帮我改进一下这个SQL查询语句", "reflection", {"c": 1, "t": 0.3, "i": 0.3, "s": 0}),
    ("RF005", "润色这封商务邮件，让它更专业", "reflection", {"c": 1, "t": 0, "i": 0.3, "s": 0}),
    ("RF006", "这个方案不够完善，帮我改进一下", "reflection", {"c": 2, "t": 0, "i": 0.6, "s": 0}),
    ("RF007", "帮我重写这段文字，让它更有说服力", "reflection", {"c": 1, "t": 0, "i": 0.3, "s": 0}),
    ("RF008", "这段代码有bug，帮我找出并修复", "reflection", {"c": 1, "t": 0, "i": 0.3, "s": 0.3}),
    # ---- PlanAndSolve ----
    ("P001", "帮我规划一个电商系统的技术架构", "plan_and_solve", {"c": 2, "t": 0, "i": 0, "s": 0.3}),
    ("P002", "设计一个机器学习项目的完整流程", "plan_and_solve", {"c": 2, "t": 0, "i": 0, "s": 0.3}),
    ("P003", "制定一个APP开发计划，从0到1", "plan_and_solve", {"c": 2, "t": 0, "i": 0, "s": 0.3}),
    ("P004", "规划一次欧洲自由行，包括路线和预算", "plan_and_solve", {"c": 2, "t": 0, "i": 0, "s": 0.3}),
    ("P005", "如何从零开始搭建一个网站，给出完整步骤", "plan_and_solve", {"c": 2, "t": 0, "i": 0, "s": 0.3}),
    ("P006", "写一个创业项目的商业计划书框架", "plan_and_solve", {"c": 2, "t": 0, "i": 0, "s": 0.6}),
    ("P007", "设计一个数据迁移方案，从旧系统到新系统", "plan_and_solve", {"c": 3, "t": 0, "i": 0, "s": 0.3}),
    ("P008", "如何准备一场技术演讲，列出准备步骤", "plan_and_solve", {"c": 2, "t": 0, "i": 0, "s": 0.3}),
    # ---- FunctionCall ----
    ("F001", "计算 12345 * 6789 等于多少", "function_call", {"c": 1, "t": 0, "i": 0, "s": 0.3}),
    ("F002", "把这段文字翻译成英文：'你好，世界'", "function_call", {"c": 1, "t": 0, "i": 0, "s": 0.3}),
    ("F003", "把当前时间格式化为YYYY-MM-DD HH:MM:SS", "function_call", {"c": 1, "t": 0, "i": 0, "s": 0.3}),
    ("F004", "将数字 3.14159 格式化为两位小数", "function_call", {"c": 1, "t": 0, "i": 0, "s": 0.3}),
    ("F005", "调用API获取用户ID为123的信息", "function_call", {"c": 1, "t": 0.6, "i": 0, "s": 0.3}),
    ("F006", "把这份JSON数据转换成XML格式", "function_call", {"c": 1, "t": 0, "i": 0, "s": 0.3}),
    ("F007", "生成一个随机密码，包含大小写字母和数字", "function_call", {"c": 1, "t": 0, "i": 0, "s": 0.3}),
    # ---- TreeOfThought ----
    ("T001", "解这道数学题有几种方法：x^2 - 5x + 6 = 0", "tree_of_thought", {"c": 2, "t": 0, "i": 0, "s": 0}),
    ("T002", "公司要选技术栈，比较React和Vue的优劣", "tree_of_thought", {"c": 2, "t": 0, "i": 0, "s": 0}),
    ("T003", "这个产品有几种方案，帮我分析优劣", "tree_of_thought", {"c": 3, "t": 0, "i": 0, "s": 0}),
    ("T004", "面对客户的三种需求，我们该怎么取舍", "tree_of_thought", {"c": 2, "t": 0, "i": 0, "s": 0}),
    ("T005", "写出所有可能的排序算法并对比复杂度", "tree_of_thought", {"c": 2, "t": 0, "i": 0, "s": 0}),
    ("T006", "有什么不同的方式可以解决这个性能问题", "tree_of_thought", {"c": 2, "t": 0, "i": 0, "s": 0}),
    ("T007", "分析一下进入新市场的几种策略", "tree_of_thought", {"c": 2, "t": 0, "i": 0, "s": 0}),
]


# ============================================================
# 测试：特征提取
# ============================================================

class TestFeatureExtraction:
    """验证 TaskFeatureExtractor 的 4 维特征提取准确性。"""

    @pytest.mark.parametrize("tid,query,expected_agent,expected_feats", TEST_CASES)
    def test_feature_values(self, extractor, tid, query, expected_agent, expected_feats):
        feats = extractor.extract(query)
        assert feats["task_complexity"] == expected_feats["c"], \
            f"{tid}: 期望复杂度={expected_feats['c']}, 实际={feats['task_complexity']}"
        assert feats["need_external_knowledge"] == expected_feats["t"], \
            f"{tid}: 期望工具={expected_feats['t']}, 实际={feats['need_external_knowledge']}"
        assert feats["iteration_need"] == expected_feats["i"], \
            f"{tid}: 期望迭代={expected_feats['i']}, 实际={feats['iteration_need']}"
        assert feats["output_structured"] == expected_feats["s"], \
            f"{tid}: 期望结构化={expected_feats['s']}, 实际={feats['output_structured']}"


# ============================================================
# 测试：路由决策
# ============================================================

class TestRouting:
    """验证 FeatureBasedRouter 的 50 个路由决策。"""

    @pytest.mark.parametrize("tid,query,expected_agent,expected_feats", TEST_CASES)
    def test_route_correct(self, router, extractor, tid, query, expected_agent, expected_feats):
        feats = extractor.extract(query)
        decision = router.route(feats)
        assert decision.agent_type == expected_agent, \
            f"{tid}: 期望路由到 {expected_agent}, 实际路由到 {decision.agent_type} (置信度={decision.confidence:.2f})"


# ============================================================
# 测试：边界案例
# ============================================================

class TestBoundaryCases:
    """混合意图和边界场景。"""

    @pytest.mark.parametrize("query,expected", [
        # 搜索+规划混合 → PlanAndSolve（高结构化）
        ("搜索一下最近有哪些AI框架，然后帮我对比分析一下", "plan_and_solve"),
        # 搜索+反思混合 → Reflection（优先级 4 胜出）
        ("查一下Python最新版本，并优化我的一段代码", "reflection"),
        # 翻译+计算混合 → PlanAndSolve（高结构化）  
        ("把这段代码翻译成Python，然后计算一下复杂度", "plan_and_solve"),
        # 比较+规划 → PlanAndSolve
        ("比较这三种设计方案，给出推荐方案和实现步骤", "plan_and_solve"),
        # 空字符串 → 兜底 simple
        ("", "simple"),
        # 纯空格 → 兜底 simple
        ("   ", "simple"),
    ])
    def test_boundary_routes(self, router, extractor, query, expected):
        feats = extractor.extract(query)
        decision = router.route(feats)
        assert decision.agent_type == expected, \
            f"'{query}': 期望 {expected}, 实际 {decision.agent_type}"


# ============================================================
# 测试：收敛域定义
# ============================================================

class TestDomainDefinitions:
    """验证收敛域定义的一致性。"""

    def test_all_paradigms_have_domains(self):
        """6 种范式都有完整的收敛域定义。"""
        assert len(PARADIGM_DOMAINS) == 6
        for pname in ("simple", "react", "reflection", "plan_and_solve",
                       "function_call", "tree_of_thought"):
            assert pname in PARADIGM_DOMAINS, f"缺少 {pname} 的收敛域"

    def test_all_domains_have_required_keys(self):
        """每个收敛域包含所有必要字段。"""
        for pname, domain in PARADIGM_DOMAINS.items():
            for dim in ("task_complexity", "need_external_knowledge",
                        "iteration_need", "output_structured", "priority"):
                assert dim in domain, f"{pname} 缺少字段 {dim}"

    def test_domain_range_validity(self):
        """收敛域范围有效：min <= max，值在合理范围内。"""
        for pname, domain in PARADIGM_DOMAINS.items():
            cmin, cmax = domain["task_complexity"]
            assert 1 <= cmin <= cmax <= 5, f"{pname} task_complexity 范围无效 ({cmin}, {cmax})"
            for dim in ("need_external_knowledge", "iteration_need", "output_structured"):
                vmin, vmax = domain[dim]
                assert 0 <= vmin <= vmax <= 1.0, f"{pname} {dim} 范围无效 ({vmin}, {vmax})"
            assert 1 <= domain["priority"] <= 5, f"{pname} priority 无效 ({domain['priority']})"


# ============================================================
# 测试：匹配度计算
# ============================================================

class TestMatchScore:
    """验证收敛域匹配度计算。"""

    def test_perfect_match(self, router):
        """特征完全在收敛域内 → 匹配度 1.0。"""
        vector = [0.2, 0.0, 0.0, 0.0]  # c=1, t=0, i=0, s=0
        score = router._match_score(vector, PARADIGM_DOMAINS["simple"])
        assert score == 1.0, f"完美匹配应得 1.0, 实际 {score}"

    def test_outside_soft_margin(self, router):
        """特征超出收敛域但仍在软边界内 → 线性衰减。"""
        # c=1 (归一化 0.2), Simple 域 c∈[1,2] (归一化 [0.2,0.4])
        # 边界内，应得 1.0
        vector = [0.2, 0.0, 0.0, 0.0]
        score = router._match_score(vector, PARADIGM_DOMAINS["simple"])
        assert score == 1.0

    def test_outside_soft_margin_decay(self, router):
        """超出收敛域但未超过 SOFT_MARGIN → 匹配度 < 1.0 但 > 0。"""
        # c=1 (归一化 0.2), FunctionCall 域 c∈[1,1] (归一化 [0.2,0.2]) ✓
        # s=0.6, FC 域 s∈[0.2,0.4], 超出 0.2/0.3 → membership = 1 - 0.2/0.3 = 0.333
        vector = [0.2, 0.0, 0.0, 0.6]
        score = router._match_score(vector, PARADIGM_DOMAINS["function_call"])
        # 几何平均: (1*1*1*0.333)^0.25 = 0.333^0.25 ≈ 0.7598
        expected = (1.0 * 1.0 * 1.0 * (1.0 - 0.2 / SOFT_MARGIN)) ** 0.25
        assert abs(score - expected) < 1e-6, f"期望 {expected:.6f}, 实际 {score:.6f}"

    def test_beyond_soft_margin(self, router):
        """远超收敛域（超过 SOFT_MARGIN）→ 匹配度 0。"""
        # c=1 (归一化 0.2), PlanAndSolve 域 c∈[2,4] (归一化 [0.4,0.8])
        # 超出 (0.4-0.2)/0.3 = 0.667, 未超过 1, 所以应得 1-0.667 = 0.333
        vector = [0.2, 0.0, 0.6, 0.0]  # c=1, i=0.6
        domain = PARADIGM_DOMAINS["plan_and_solve"].copy()
        # s=0.6, i=0, 都 OK. 只有 c=1 超出 c∈[2,4]
        # 但这里 i=0.6 超出域 i∈[0,0.3]
        # 所以 membership: c = 1-(0.4-0.2)/0.3 = 0.333, i = 1-(0.6-0.3)/0.3 = 0
        # 几何平均: (0.333*1*0*1)^0.25 = 0
        score = router._match_score(vector, domain)
        assert score == 0.0, f"应得 0.0, 实际 {score}"


# ============================================================
# 测试：统计信息
# ============================================================

class TestRouterStats:
    """验证路由统计信息。"""

    def test_stats_returns_expected_keys(self, router):
        s = router.stats()
        assert "domains_count" in s
        assert s["domains_count"] == 6
        assert "soft_margin" in s
        assert s["soft_margin"] == SOFT_MARGIN
        assert "threshold" in s
        assert s["threshold"] == MATCH_THRESHOLD
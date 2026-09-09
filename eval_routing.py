"""
路由评估脚本：验证修复后特征提取器的路由准确率。
"""
import sys
sys.path.insert(0, r"D:\pycharm_project\practiceLLM")

from src.agents.framework.feature_extractor import TaskFeatureExtractor, FeatureBasedRouter, PARADIGM_DOMAINS
from src.agents.framework.agent_framework import AgentFactory, BaseAgent

# 注册伪 Agent
factory = AgentFactory()
for t in ("simple", "react", "reflection", "plan_and_solve",
          "function_call", "tree_of_thought", "echo", "reverse"):
    factory.register(t, type(t, (BaseAgent,), {"AGENT_TYPE": t,
                                               "DESCRIPTION": t,
                                               "_execute": lambda s, q, **k: q}))

extractor = TaskFeatureExtractor()
router = FeatureBasedRouter(factory)

TEST_CASES = [
    # SimpleAgent
    ("S001", "什么是人工智能", "simple"),
    ("S002", "现在几点了", "simple"),
    ("S003", "解释一下什么是机器学习", "simple"),
    ("S004", "Python是什么语言", "simple"),
    ("S005", "今天是几号", "simple"),
    ("S006", "告诉我一个笑话", "simple"),
    ("S007", "什么是区块链技术，简单说说", "simple"),
    ("S008", "你觉得今天天气怎么样", "simple"),
    ("S009", "给我推荐一本书", "simple"),
    ("S010", "什么是深度学习，和机器学习有什么区别", "simple"),
    # ReAct
    ("R001", "搜索一下最新的AI新闻", "react"),
    ("R002", "查询北京明天的天气", "react"),
    ("R003", "帮我查一下OpenAI最新的API文档", "react"),
    ("R004", "获取当前比特币的价格", "react"),
    ("R005", "搜索一下关于Transformer架构的论文", "react"),
    ("R006", "查一下Python最新版本是什么", "react"),
    ("R007", "帮我找一下2024年全球GDP排名", "react"),
    ("R008", "现在美元兑人民币汇率是多少", "react"),
    ("R009", "搜索一下最新的iPhone型号和价格", "react"),
    ("R010", "查找一下关于气候变化的最近研究", "react"),
    # Reflection
    ("RF001", "帮我润色这段文案：'这个产品很好，推荐大家购买'", "reflection"),
    ("RF002", "优化一下这段代码的性能：for i in range(100): for j in range(100): print(i*j)", "reflection"),
    ("RF003", "帮我审查一下这个项目方案有什么问题", "reflection"),
    ("RF004", "改进一下这个SQL查询语句", "reflection"),
    ("RF005", "润色这封商务邮件，让它更专业", "reflection"),
    ("RF006", "这个方案不够完善，帮我改进一下", "reflection"),
    ("RF007", "帮我重写这段文字，让它更有说服力", "reflection"),
    ("RF008", "这段代码有bug，帮我找出并修复", "reflection"),
    # PlanAndSolve
    ("P001", "帮我规划一个电商系统的技术架构", "plan_and_solve"),
    ("P002", "设计一个机器学习项目的完整流程", "plan_and_solve"),
    ("P003", "制定一个APP开发计划，从0到1", "plan_and_solve"),
    ("P004", "规划一次欧洲自由行，包括路线和预算", "plan_and_solve"),
    ("P005", "如何从零开始搭建一个网站，给出完整步骤", "plan_and_solve"),
    ("P006", "写一个创业项目的商业计划书框架", "plan_and_solve"),
    ("P007", "设计一个数据迁移方案，从旧系统到新系统", "plan_and_solve"),
    ("P008", "如何准备一场技术演讲，列出准备步骤", "plan_and_solve"),
    # FunctionCall
    ("F001", "计算 12345 * 6789 等于多少", "function_call"),
    ("F002", "把这段文字翻译成英文：'你好，世界'", "function_call"),
    ("F003", "把当前时间格式化为YYYY-MM-DD HH:MM:SS", "function_call"),
    ("F004", "将数字 3.14159 格式化为两位小数", "function_call"),
    ("F005", "调用API获取用户ID为123的信息", "function_call"),
    ("F006", "把这份JSON数据转换成XML格式", "function_call"),
    ("F007", "生成一个随机密码，包含大小写字母和数字", "function_call"),
    # TreeOfThought
    ("T001", "解这道数学题有几种方法：x^2 - 5x + 6 = 0", "tree_of_thought"),
    ("T002", "公司要选技术栈，比较React和Vue的优劣", "tree_of_thought"),
    ("T003", "这个产品有几种方案，帮我分析优劣", "tree_of_thought"),
    ("T004", "面对客户的三种需求，我们该怎么取舍", "tree_of_thought"),
    ("T005", "写出所有可能的排序算法并对比复杂度", "tree_of_thought"),
    ("T006", "有什么不同的方式可以解决这个性能问题", "tree_of_thought"),
    ("T007", "分析一下进入新市场的几种策略", "tree_of_thought"),
]

PARADIGM_DISPLAY = {
    "simple": "SimpleAgent",
    "react": "ReAct",
    "reflection": "Reflection",
    "plan_and_solve": "PlanAndSolve",
    "function_call": "FunctionCall",
    "tree_of_thought": "TreeOfThought",
}

print(f"\n{'='*100}")
print("特征提取 + 路由评估")
print("=" * 100)

print(f"\n{'ID':<7s} {'实际特征':<36s} {'→路由':<18s} {'置信度':>5s} {'期望':<16s} {'结果':>5s}")
print("-" * 100)

correct = 0
total = 0
by_paradigm = {}

for tid, query, expected in TEST_CASES:
    feats = extractor.extract(query)
    decision = router.route(feats)
    routed = decision.agent_type
    is_correct = (routed == expected)
    if tid == "P006":
        # 检查浮点精度
        for pname, domain in router._domains.items():
            score = router._match_score(feats["vector"], domain)
            print(f"  DEBUG {pname}: score={score:.10f}, priority={domain['priority']}")
    if is_correct:
        correct += 1
    total += 1

    by_paradigm.setdefault(expected, {"ok": 0, "total": 0})
    by_paradigm[expected]["total"] += 1
    if is_correct:
        by_paradigm[expected]["ok"] += 1

    actual_str = f"c={feats['task_complexity']} t={feats['need_external_knowledge']:.2f} i={feats['iteration_need']:.2f} s={feats['output_structured']:.2f}"
    routed_str = PARADIGM_DISPLAY.get(routed, routed)
    expected_str = PARADIGM_DISPLAY.get(expected, expected)
    mark = "✅" if is_correct else "❌"
    print(f"{tid:<7s} {actual_str:<36s} {routed_str:<18s} {decision.confidence:>4.2f}  {expected_str:<16s} {mark:>5s}")

print(f"\n{'='*100}")
print(f"总准确率: {correct}/{total} = {correct/total*100:.1f}%")
print(f"\n按范式:")
total_ok = 0
total_all = 0
for pname in ["simple", "react", "reflection", "plan_and_solve", "function_call", "tree_of_thought"]:
    data = by_paradigm.get(pname, {"ok": 0, "total": 0})
    pct = data["ok"] / data["total"] * 100 if data["total"] > 0 else 0
    total_ok += data["ok"]
    total_all += data["total"]
    print(f"  {PARADIGM_DISPLAY[pname]:<18s}: {data['ok']}/{data['total']} = {pct:.1f}%")
print(f"  总计: {total_ok}/{total_all} = {total_ok/total_all*100:.1f}%")

# 特征提取分析
from collections import defaultdict
stats = defaultdict(lambda: {"c": [], "t": [], "i": [], "s": []})
for tid, query, expected in TEST_CASES:
    feats = extractor.extract(query)
    stats[expected]["c"].append(feats["task_complexity"])
    stats[expected]["t"].append(feats["need_external_knowledge"])
    stats[expected]["i"].append(feats["iteration_need"])
    stats[expected]["s"].append(feats["output_structured"])

print(f"\n{'='*100}")
print("特征提取分析（实际值分布）")
print(f"\n{'范式':<18s} {'数量':>4s} {'complexity':>16s} {'need_tool':>16s} {'iteration':>16s} {'structured':>16s}")
print("-" * 82)
for pname, data in stats.items():
    dp = PARADIGM_DISPLAY.get(pname, pname)
    c_min, c_max = min(data["c"]), max(data["c"])
    t_min, t_max = min(data["t"]), max(data["t"])
    i_min, i_max = min(data["i"]), max(data["i"])
    s_min, s_max = min(data["s"]), max(data["s"])
    count = len(data["c"])
    print(f"{dp:<18s} {count:>4d}  {sum(data['c'])/count:.1f} [{c_min}-{c_max}]  {sum(data['t'])/count:.2f} [{t_min:.2f}-{t_max:.2f}]  {sum(data['i'])/count:.2f} [{i_min:.2f}-{i_max:.2f}]  {sum(data['s'])/count:.2f} [{s_min:.2f}-{s_max:.2f}]")
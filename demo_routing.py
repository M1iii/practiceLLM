"""
路由系统观察演示脚本
=====================
模拟复杂对话场景，展示 TaskFeatureExtractor + FeatureBasedRouter 的路由决策过程。
不依赖 LLM，纯特征工程路由，所有场景均可直接运行。
"""
import sys
sys.path.insert(0, r"D:\pycharm_project\practiceLLM")

from src.agents.framework.feature_extractor import (
    TaskFeatureExtractor, FeatureBasedRouter, PARADIGM_DOMAINS, DIMENSION_NAMES, SOFT_MARGIN
)
from src.agents.framework.agent_framework import AgentFactory, BaseAgent

# ── 注册全部 6 种 Agent 类型（伪 Agent，无需 LLM）──
factory = AgentFactory()
for t in ("simple", "react", "reflection", "plan_and_solve",
          "function_call", "tree_of_thought"):
    factory.register(t, type(t, (BaseAgent,), {"AGENT_TYPE": t,
                                               "DESCRIPTION": t,
                                               "_execute": lambda s, q, **k: q}))

extractor = TaskFeatureExtractor()
router = FeatureBasedRouter(factory)

PARADIGM_ICON = {
    "simple":        "💬",
    "react":         "🔍",
    "reflection":    "🔧",
    "plan_and_solve":"📋",
    "function_call": "⚙️",
    "tree_of_thought":"🌲",
}

PARADIGM_CN = {
    "simple":        "SimpleAgent(聊天)",
    "react":         "ReAct(搜索)",
    "reflection":    "Reflection(反思)",
    "plan_and_solve":"PlanAndSolve(规划)",
    "function_call": "FunctionCall(函数调用)",
    "tree_of_thought":"TreeOfThought(多路径)",
}

def show_route(query: str, idx: int):
    """显示单条查询的路由全过程。"""
    feats = extractor.extract(query)
    decision = router.route(feats)
    icon = PARADIGM_ICON.get(decision.agent_type, "❓")
    name = PARADIGM_CN.get(decision.agent_type, decision.agent_type)

    # 特征向量
    c = feats["task_complexity"]
    t = feats["need_external_knowledge"]
    i = feats["iteration_need"]
    s = feats["output_structured"]

    # 各范式匹配度
    scores = []
    for pname in PARADIGM_CN:
        domain = PARADIGM_DOMAINS[pname]
        score = router._match_score(feats["vector"], domain)
        scores.append((pname, score, domain["priority"]))

    scores.sort(key=lambda x: (-x[1], -x[2]))

    print(f"\n{'─'*70}")
    print(f"  [{idx}] 用户: {query}")
    print(f"{'─'*70}")
    print(f"  📊 特征向量: 复杂度={c}  工具={t:.2f}  迭代={i:.2f}  结构化={s:.2f}")
    print(f"  🧭 路由结果: {icon} {name}  置信度={decision.confidence:.2f}")
    print(f"  📝 理由: {decision.reason}")
    print(f"  ── 匹配度排名 ──")
    for pname, score, pri in scores:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        cn = PARADIGM_CN.get(pname, pname)
        print(f"    {bar} {score:.2f}  {cn}  (优先级={pri})")


# ════════════════════════════════════════════════════════════════
# 场景一：日常聊天（SimpleAgent）
# ════════════════════════════════════════════════════════════════
print("\n" + "╔" + "═"*68 + "╗")
print("║  场景一：日常聊天 — 预期路由 → SimpleAgent 💬")
print("╚" + "═"*68 + "╝")

show_route("什么是人工智能？", 1)
show_route("今天天气怎么样？", 2)
show_route("给我讲个笑话吧", 3)
show_route("什么是深度学习，和机器学习有什么区别？", 4)

# ════════════════════════════════════════════════════════════════
# 场景二：搜索查询（ReAct）
# ════════════════════════════════════════════════════════════════
print("\n" + "╔" + "═"*68 + "╗")
print("║  场景二：搜索查询 — 预期路由 → ReAct 🔍")
print("╚" + "═"*68 + "╝")

show_route("搜索一下最新的AI新闻", 5)
show_route("查询北京明天的天气", 6)
show_route("查一下OpenAI最新的API文档", 7)
show_route("获取当前比特币的价格", 8)

# ════════════════════════════════════════════════════════════════
# 场景三：反思优化（Reflection）
# ════════════════════════════════════════════════════════════════
print("\n" + "╔" + "═"*68 + "╗")
print("║  场景三：反思优化 — 预期路由 → Reflection 🔧")
print("╚" + "═"*68 + "╝")

show_route("帮我润色这段文案：'这个产品很好，推荐大家购买'", 9)
show_route("优化一下这段代码的性能：for i in range(100): for j in range(100): print(i*j)", 10)
show_route("帮我审查一下这个项目方案有什么问题", 11)
show_route("帮我改进一下这个SQL查询语句", 12)

# ════════════════════════════════════════════════════════════════
# 场景四：规划执行（PlanAndSolve）
# ════════════════════════════════════════════════════════════════
print("\n" + "╔" + "═"*68 + "╗")
print("║  场景四：规划执行 — 预期路由 → PlanAndSolve 📋")
print("╚" + "═"*68 + "╝")

show_route("帮我规划一个电商系统的技术架构", 13)
show_route("设计一个机器学习项目的完整流程", 14)
show_route("制定一个APP开发计划，从0到1", 15)
show_route("写一个创业项目的商业计划书框架", 16)

# ════════════════════════════════════════════════════════════════
# 场景五：精确计算/格式转换（FunctionCall）
# ════════════════════════════════════════════════════════════════
print("\n" + "╔" + "═"*68 + "╗")
print("║  场景五：精确计算/格式转换 — 预期路由 → FunctionCall ⚙️")
print("╚" + "═"*68 + "╝")

show_route("计算 12345 * 6789 等于多少", 17)
show_route("把这段文字翻译成英文：'你好，世界'", 18)
show_route("把当前时间格式化为YYYY-MM-DD HH:MM:SS", 19)
show_route("调用API获取用户ID为123的信息", 20)

# ════════════════════════════════════════════════════════════════
# 场景六：多路径探索（TreeOfThought）
# ════════════════════════════════════════════════════════════════
print("\n" + "╔" + "═"*68 + "╗")
print("║  场景六：多路径探索 — 预期路由 → TreeOfThought 🌲")
print("╚" + "═"*68 + "╝")

show_route("解这道数学题有几种方法：x^2 - 5x + 6 = 0", 21)
show_route("公司要选技术栈，比较React和Vue的优劣", 22)
show_route("分析一下进入新市场的几种策略", 23)
show_route("有什么不同的方式可以解决这个性能问题", 24)

# ════════════════════════════════════════════════════════════════
# 场景七：复杂边界案例（考验区分度）
# ════════════════════════════════════════════════════════════════
print("\n" + "╔" + "═"*68 + "╗")
print("║  场景七：边界案例 — 考验路由区分度")
print("╚" + "═"*68 + "╝")

show_route("搜索一下最近有哪些AI框架，然后帮我对比分析一下", 25)
show_route("查一下Python最新版本，并优化我的一段代码", 26)
show_route("把这段代码翻译成Python，然后计算一下复杂度", 27)
show_route("比较这三种设计方案，给出推荐方案和实现步骤", 28)

# ════════════════════════════════════════════════════════════════
# 总结
# ════════════════════════════════════════════════════════════════
print("\n" + "╔" + "═"*68 + "╗")
print("║  路由系统总结")
print("╚" + "═"*68 + "╝")
print(f"""
  路由引擎: FeatureBasedRouter
  特征维度: {', '.join(DIMENSION_NAMES)}
  软边界:   SOFT_MARGIN = {SOFT_MARGIN}（超出收敛域时线性衰减）
  匹配阈值: MATCH_THRESHOLD = 0.5（低于此值回退 SimpleAgent）

  收敛域定义:
""")
for pname, domain in PARADIGM_DOMAINS.items():
    c = domain["task_complexity"]
    t = domain["need_external_knowledge"]
    i = domain["iteration_need"]
    s = domain["output_structured"]
    pri = domain["priority"]
    cn = PARADIGM_CN.get(pname, pname)
    print(f"    {PARADIGM_ICON.get(pname, ' ')} {cn:30s}  c∈{c}  t∈({t[0]:.1f},{t[1]:.1f})  i∈({i[0]:.1f},{i[1]:.1f})  s∈({s[0]:.1f},{s[1]:.1f})  pri={pri}")
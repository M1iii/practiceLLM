"""QueryClassifier：基于关键词规则的智能查询路由。"""

_COMPLEX_KEYWORDS = [
    "为什么", "如何", "怎么", "怎样", "请说明", "请解释",
    "详细", "比较", "区别", "关系", "联系", "差异",
    "原理", "机制", "流程", "步骤", "优缺点", "优势",
    "劣势", "影响", "作用", "应用场景", "最佳实践",
    "举例", "示例", "对比", "分析", "总结",
    "什么", "哪些", "哪几种", "是什么", "有哪些",
]

_REFERRAL_WORDS = [
    "它", "它们", "这个", "那个", "这些", "那些",
    "这种", "那种", "上述", "以上", "该",
]

_GREETINGS = [
    "你好", "您好", "hi", "hello", "hey",
    "在吗", "在不在", "你好吗",
]


class QueryClassifier:
    """基于关键词规则判断查询复杂度，决定是否启用 MQE/HyDE。

    返回 "fast"（快速通道，纯混合检索）或 "full"（全量通道，启用查询变换）。

    规则优先级：
      1. 问候语 → fast
      2. 含复杂关键词 → full
      3. 含指代词 → full
      4. 极短（< 8 字符）→ fast
      5. 长句（> 30 字符）→ full
      6. 其余 → fast
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def classify(self, query: str) -> str:
        query = query.strip()
        qlen = len(query)

        if query.lower() in _GREETINGS:
            self._log("问候语 → fast")
            return "fast"

        for kw in _COMPLEX_KEYWORDS:
            if kw in query:
                self._log(f"含复杂关键词「{kw}」→ full")
                return "full"

        for rw in _REFERRAL_WORDS:
            if rw in query:
                self._log(f"含指代词「{rw}」→ full")
                return "full"

        if qlen < 8:
            self._log(f"极短查询 ({qlen}字) → fast")
            return "fast"

        if qlen > 30:
            self._log(f"长句 ({qlen}字) → full")
            return "full"

        self._log("简单查询 → fast")
        return "fast"

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [QueryClassifier] {msg}")
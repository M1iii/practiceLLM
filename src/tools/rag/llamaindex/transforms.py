"""
查询变换模块：MQE（多查询扩展）、HyDE（假设文档嵌入）、QueryClassifier（智能路由）。

用法：
    from src.tools.rag.llamaindex.transforms import MultiQueryExpander, HyDEExpander, QueryClassifier

    expander = MultiQueryExpander()
    queries = expander.expand("什么是 RAG？")
    # -> ["RAG 是什么？", "检索增强生成的定义", ...]

    classifier = QueryClassifier()
    route = classifier.classify("你好")
    # -> "fast"（快速通道，跳过 MQE/HyDE）
"""

import re
from typing import List, Optional

from llama_index.core import PromptTemplate
from llama_index.core.retrievers import BaseRetriever

from src.tools.rag.llamaindex.adapters import QwenLLM


# ============================================================
# MQE：多查询扩展
# ============================================================

MQE_PROMPT = PromptTemplate(
    "你是一个 AI 助手，需要为用户的原始问题生成 {num_queries} 个语义等价的查询变体。\n"
    "这些变体应该从不同角度表述同一个问题，帮助检索系统找到更全面的信息。\n"
    "每个变体一行，直接输出问题，不要编号。\n\n"
    "原始问题: {query}\n\n"
    "查询变体:"
)


class MultiQueryExpander:
    """多查询扩展（MQE）：使用 LLM 生成多个语义等价查询变体。

    每个变体分别检索后合并结果，提升召回覆盖范围。
    """

    def __init__(
        self,
        llm: Optional[QwenLLM] = None,
        num_queries: int = 3,
        prompt: Optional[PromptTemplate] = None,
    ):
        self.llm = llm or QwenLLM()
        self.num_queries = num_queries
        self.prompt = prompt or MQE_PROMPT

    def expand(self, query: str) -> List[str]:
        """生成查询变体。"""
        response = self.llm.complete(
            self.prompt.format(query=query, num_queries=self.num_queries)
        )
        variants = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
        # 去重+去空
        seen = set()
        result = [query]  # 原始查询排第一
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                result.append(v)
        return result[: self.num_queries + 1]

    def retrieve_multi(
        self, query: str, retriever: BaseRetriever, top_k: int = 3
    ) -> list:
        """生成多查询后分别检索，合并排名结果。"""
        queries = self.expand(query)

        # 融合评分
        from collections import defaultdict

        score_map = defaultdict(float)
        seen_map = defaultdict(set)

        for q_idx, q in enumerate(queries):
            results = retriever.retrieve(q)
            for rank, result in enumerate(results):
                node_id = result.node.node_id
                # Reciprocal Rank Fusion: 1 / (rank + 60)
                score_map[node_id] += 1.0 / (rank + 60)
                if node_id not in seen_map:
                    seen_map[node_id] = result
                    seen_map[node_id].score = 0.0

        # 按融合评分排序
        sorted_nodes = sorted(score_map.items(), key=lambda x: -x[1])
        final_results = []
        for node_id, score in sorted_nodes[:top_k]:
            result = seen_map[node_id]
            result.score = score
            final_results.append(result)

        return final_results


# ============================================================
# HyDE：假设文档嵌入
# ============================================================

HYDE_PROMPT = PromptTemplate(
    "请根据以下问题，撰写一段假设性的答文本。\n"
    "这段文本不需要回答问题的具体细节，而是描述一个理想的回答中可能包含的内容。\n"
    "假设你正在回答这个问题，文本应该包含相关的概念、术语和解释。\n\n"
    "问题: {query}\n\n"
    "假设回答:"
)


class HyDEExpander:
    """假设文档嵌入（HyDE）：先用 LLM 生成一个"假设答案"，
    再用这个假设答案去检索，提升检索准确率。
    """

    def __init__(
        self,
        llm: Optional[QwenLLM] = None,
        prompt: Optional[PromptTemplate] = None,
    ):
        self.llm = llm or QwenLLM()
        self.prompt = prompt or HYDE_PROMPT

    def expand(self, query: str) -> str:
        """生成假设文档。"""
        response = self.llm.complete(
            self.prompt.format(query=query)
        )
        return response.text.strip()

    def retrieve_with_hypothetical(
        self, query: str, retriever: BaseRetriever, top_k: int = 3
    ) -> list:
        """生成假设文档后用其进行检索。"""
        hypothetical = self.expand(query)
        # 用假设文档检索
        results = retriever.retrieve(hypothetical)

        # 将原始查询的节点也加入，确保不丢失原始意图
        original_results = retriever.retrieve(query)

        # 融合
        seen_ids = set(r.node.node_id for r in results)
        for r in original_results:
            if r.node.node_id not in seen_ids:
                results.append(r)
                seen_ids.add(r.node.node_id)

        return results[:top_k]


# ============================================================
# 组合变换
# ============================================================

class QueryTransformPipeline:
    """组合多种查询变换：MQE + HyDE 串联使用。"""

    def __init__(
        self,
        mqe: Optional[MultiQueryExpander] = None,
        hyde: Optional[HyDEExpander] = None,
    ):
        self.mqe = mqe or MultiQueryExpander()
        self.hyde = hyde or HyDEExpander()

    def transform(self, query: str) -> List[str]:
        """生成用于检索的查询列表（原始查询 + 扩展 + 假设文档）。"""
        queries = [query]
        # MQE 扩展
        mqe_queries = self.mqe.expand(query)
        queries.extend(mqe_queries[1:])  # 不重复原始查询
        # HyDE 假设文档
        hyde_doc = self.hyde.expand(query)
        queries.append(hyde_doc)
        return queries


# ============================================================
# QueryClassifier：智能路由（快速通道 vs 全量通道）
# ============================================================

# 复杂问题关键词（触发全量通道）
_COMPLEX_KEYWORDS = [
    "为什么", "如何", "怎么", "怎样", "请说明", "请解释",
    "详细", "比较", "区别", "关系", "联系", "差异",
    "原理", "机制", "流程", "步骤", "优缺点", "优势",
    "劣势", "影响", "作用", "应用场景", "最佳实践",
    "举例", "示例", "对比", "分析", "总结",
    "什么", "哪些", "哪几种", "是什么", "有哪些",
]

# 指代词（可能为多轮追问）
_REFERRAL_WORDS = [
    "它", "它们", "这个", "那个", "这些", "那些",
    "这种", "那种", "上述", "以上", "该",
]

# 问候语（直接走快速通道）
_GREETINGS = [
    "你好", "您好", "hi", "hello", "hey",
    "在吗", "在不在", "你好吗",
]


class QueryClassifier:
    """智能查询路由分类器。

    基于关键词规则判断查询复杂度，决定是否启用 MQE/HyDE：
      - "fast"：快速通道，纯混合检索即可
      - "full"：全量通道，启用 MQE + HyDE 查询变换

    规则优先级：
      1. 问候/极短 → fast
      2. 含复杂关键词或指代词 → full
      3. 长句（>30 字符）→ full
      4. 其余 → fast
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def classify(self, query: str) -> str:
        """判断查询走 fast 还是 full 通道。"""
        query = query.strip()
        qlen = len(query)

        # 规则 1：问候语 → fast
        if query.lower() in _GREETINGS:
            if self.verbose:
                print(f"  [QueryClassifier] 问候语 → fast")
            return "fast"

        # 规则 2：含复杂关键词 → full（优先于短查询判断）
        for kw in _COMPLEX_KEYWORDS:
            if kw in query:
                if self.verbose:
                    print(f"  [QueryClassifier] 含复杂关键词「{kw}」→ full")
                return "full"

        # 规则 3：含指代词 → full（可能是多轮追问，优先于短查询判断）
        for rw in _REFERRAL_WORDS:
            if rw in query:
                if self.verbose:
                    print(f"  [QueryClassifier] 含指代词「{rw}」→ full")
                return "full"

        # 规则 4：极短（< 8 字符）→ fast
        if qlen < 8:
            if self.verbose:
                print(f"  [QueryClassifier] 极短查询 ({qlen}字) → fast")
            return "fast"

        # 规则 5：长句（> 30 字符）→ full
        if qlen > 30:
            if self.verbose:
                print(f"  [QueryClassifier] 长句 ({qlen}字) → full")
            return "full"

        # 默认 → fast
        if self.verbose:
            print(f"  [QueryClassifier] 简单查询 → fast")
        return "fast"
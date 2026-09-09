"""MQE：多查询扩展（Multi-Query Expansion），提升召回覆盖范围。"""

from collections import defaultdict
from typing import List, Optional

from llama_index.core import PromptTemplate
from llama_index.core.retrievers import BaseRetriever

from src.tools.rag.llamaindex.adapters import QwenLLM

MQE_PROMPT = PromptTemplate(
    "你是一个 AI 助手，需要为用户的原始问题生成 {num_queries} 个语义等价的查询变体。\n"
    "这些变体应该从不同角度表述同一个问题，帮助检索系统找到更全面的信息。\n"
    "每个变体一行，直接输出问题，不要编号。\n\n"
    "原始问题: {query}\n\n"
    "查询变体:"
)


class MultiQueryExpander:
    """使用 LLM 生成多个语义等价查询变体，分别检索后融合结果。"""

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
        response = self.llm.complete(
            self.prompt.format(query=query, num_queries=self.num_queries)
        )
        variants = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
        seen = set()
        result = [query]
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                result.append(v)
        return result[: self.num_queries + 1]

    def retrieve_multi(
        self, query: str, retriever: BaseRetriever, top_k: int = 3
    ) -> list:
        queries = self.expand(query)
        score_map = defaultdict(float)
        seen_map = {}

        for q_idx, q in enumerate(queries):
            results = retriever.retrieve(q)
            for rank, result in enumerate(results):
                node_id = result.node.node_id
                score_map[node_id] += 1.0 / (rank + 60)
                if node_id not in seen_map:
                    seen_map[node_id] = result
                    seen_map[node_id].score = 0.0

        sorted_nodes = sorted(score_map.items(), key=lambda x: -x[1])
        final_results = []
        for node_id, score in sorted_nodes[:top_k]:
            result = seen_map[node_id]
            result.score = score
            final_results.append(result)
        return final_results
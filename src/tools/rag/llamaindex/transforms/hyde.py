"""HyDE：假设文档嵌入（Hypothetical Document Embedding），提升检索准确率。"""

from typing import List, Optional

from llama_index.core import PromptTemplate
from llama_index.core.retrievers import BaseRetriever

from src.tools.rag.llamaindex.adapters import QwenLLM

HYDE_PROMPT = PromptTemplate(
    "请根据以下问题，撰写一段假设性的回答文本。\n"
    "这段文本不需要回答问题的具体细节，而是描述一个理想的回答中可能包含的内容。\n"
    "假设你正在回答这个问题，文本应该包含相关的概念、术语和解释。\n\n"
    "问题: {query}\n\n"
    "假设回答:"
)


class HyDEExpander:
    """HyDE 假设文档嵌入扩展。"""

    def __init__(
        self,
        llm: Optional[QwenLLM] = None,
        prompt: Optional[PromptTemplate] = None,
    ):
        self.llm = llm or QwenLLM()
        self.prompt = prompt or HYDE_PROMPT

    def expand(self, query: str) -> str:
        response = self.llm.complete(self.prompt.format(query=query))
        return response.text.strip()

    def retrieve_with_hypothetical(
        self, query: str, retriever: BaseRetriever, top_k: int = 3
    ) -> list:
        hypothetical = self.expand(query)
        results = retriever.retrieve(hypothetical)
        original_results = retriever.retrieve(query)

        seen_ids = set(r.node.node_id for r in results)
        for r in original_results:
            if r.node.node_id not in seen_ids:
                results.append(r)
                seen_ids.add(r.node.node_id)

        return results[:top_k]
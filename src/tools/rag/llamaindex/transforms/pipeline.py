"""组合变换管线：MQE + HyDE 串联使用。"""

from typing import List, Optional

from src.tools.rag.llamaindex.transforms.mqe import MultiQueryExpander
from src.tools.rag.llamaindex.transforms.hyde import HyDEExpander


class QueryTransformPipeline:
    """组合 MQE 多查询扩展与 HyDE 假设文档，生成用于检索的查询列表。"""

    def __init__(
        self,
        mqe: Optional[MultiQueryExpander] = None,
        hyde: Optional[HyDEExpander] = None,
    ):
        self.mqe = mqe or MultiQueryExpander()
        self.hyde = hyde or HyDEExpander()

    def transform(self, query: str) -> List[str]:
        queries = [query]
        mqe_queries = self.mqe.expand(query)
        queries.extend(mqe_queries[1:])
        hyde_doc = self.hyde.expand(query)
        queries.append(hyde_doc)
        return queries
"""LLM 重排序器：TF-IDF 粗筛 + LLM 精排 两阶段检索重排序。"""

import re
import json
import math
from typing import List, Dict, Any, Optional
from collections import Counter

from src.tools.memory.modules import MemoryEntry


class LLMReranker:
    """TF-IDF 粗筛 + LLM 精排 两阶段检索重排序器。"""

    RERANK_PROMPT = """你是一个记忆检索排序助手。请根据用户的查询，对以下候选记忆按相关性从高到低排序。

【查询】
{query}

【候选记忆列表】
{memory_list}

【任务】
分析每条记忆与查询的语义相关性，按相关度从高到低排序。只返回 JSON 数组，格式严格如下：
[{{"index": 原始编号, "relevance": 0.0-1.0的相关性评分, "reason": "10字以内的简短理由"}}, ...]

要求：
- relevance 精确到两位小数，1.0 表示完全匹配，0.0 表示完全不相关
- 只返回 JSON 数组，不要任何其他文字
- 按 relevance 从高到低排列"""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from src.core.llm import practiceLLM
            self._llm = practiceLLM()
        return self._llm

    def rerank(
        self,
        memories: list,
        query: str,
        coarse_limit: int = 15,
        rerank_limit: int = 5,
    ) -> list:
        if not memories:
            return []

        coarse_results = self._coarse_filter(memories, query, coarse_limit)

        if not coarse_results:
            return []

        llm_results = self._llm_rerank(coarse_results, query, rerank_limit)

        return llm_results

    def _coarse_filter(self, memories: list, query: str, top_k: int) -> list:
        try:
            from src.tools.memory.modules.working_memory import WorkingMemory as _WM

            query_tokens = _WM._tokenize_for_tfidf(query)
            if not query_tokens:
                return [(mem, 0.0) for mem in memories[:top_k]]

            doc_tokens_list = [_WM._tokenize_for_tfidf(m.content) for m in memories]

            df = Counter()
            for tokens in doc_tokens_list:
                for token in set(tokens):
                    df[token] += 1

            total_docs = len(memories)
            idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1
                   for token, freq in df.items()}

            query_tf = Counter(query_tokens)
            query_len = len(query_tokens)
            query_vector = {token: (count / query_len) * idf.get(token, 0.0)
                            for token, count in query_tf.items()}

            scores = []
            for mem, doc_tokens in zip(memories, doc_tokens_list):
                doc_len = len(doc_tokens) if doc_tokens else 1
                doc_tf = Counter(doc_tokens)
                doc_vector = {token: (count / doc_len) * idf.get(token, 0.0)
                              for token, count in doc_tf.items()}
                similarity = _WM._cosine_similarity(query_vector, doc_vector)
                scores.append((mem, similarity))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        except Exception:
            return [(mem, 0.0) for mem in memories[:top_k]]

    def _llm_rerank(self, coarse_results: list, query: str, top_k: int) -> list:
        memory_lines = []
        for i, (mem, tfidf_score) in enumerate(coarse_results, 1):
            memory_lines.append(
                f"[{i}] (TF-IDF:{tfidf_score:.3f}) {mem.content}"
            )
        memory_list_text = "\n".join(memory_lines)

        prompt = self.RERANK_PROMPT.format(
            query=query,
            memory_list=memory_list_text,
        )

        messages = [
            {"role": "system", "content": "你是一个精确的记忆检索排序助手。只返回 JSON 数组，不输出任何其他内容。"},
            {"role": "user", "content": prompt},
        ]

        try:
            llm = self._get_llm()
            response = llm.invoke(messages, temperature=0, stream=False)

            if not response:
                return self._fallback_rerank(coarse_results, top_k)

            ranked = self._parse_llm_response(response, coarse_results)
            return ranked[:top_k]

        except Exception:
            return self._fallback_rerank(coarse_results, top_k)

    def _parse_llm_response(self, response: str, coarse_results: list) -> list:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return self._fallback_rerank(coarse_results, len(coarse_results))

        try:
            rankings = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return self._fallback_rerank(coarse_results, len(coarse_results))

        if not isinstance(rankings, list):
            return self._fallback_rerank(coarse_results, len(coarse_results))

        results = []
        seen_indices = set()
        for item in rankings:
            idx = item.get("index", -1) - 1
            if 0 <= idx < len(coarse_results) and idx not in seen_indices:
                mem, _ = coarse_results[idx]
                relevance = max(0.0, min(1.0, float(item.get("relevance", 0.0))))
                reason = str(item.get("reason", ""))[:20]
                results.append((mem, relevance, reason))
                seen_indices.add(idx)

        return results

    def _fallback_rerank(self, coarse_results: list, top_k: int) -> list:
        return [(mem, score, "TF-IDF回退") for mem, score in coarse_results[:top_k]]

    @staticmethod
    def extract_keywords(text: str, top_k: int = 5) -> list:
        try:
            from src.tools.memory.modules.working_memory import WorkingMemory as _WM
            tokens = _WM._tokenize_for_tfidf(text)
            if not tokens:
                return []
            tf = Counter(tokens)
            total = len(tokens)
            return [word for word, _ in tf.most_common(top_k)]
        except Exception:
            return []
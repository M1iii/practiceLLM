"""
WorkingMemory：工作记忆模块
==========================

容量有限 + TTL自动清理 + TF-IDF + 关键词混合检索。
"""

import re
import math
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from collections import Counter

from .base import MemoryEntry, MemoryModule


class WorkingMemory(MemoryModule):
    """工作记忆模块 —— 容量有限 + TTL自动清理 + 混合检索。

    特点：
      - 容量有限（默认50条），超出时淘汰最不重要的
      - TTL自动清理：超过 max_age_minutes 分钟的记忆自动过期
      - 纯内存存储，访问速度极快
      - 混合检索：优先 TF-IDF 向量化语义检索，失败则回退关键词匹配
      - 评分公式：(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)
    """

    def __init__(self, max_capacity: int = 50, max_age_minutes: int = 60):
        super().__init__("working", max_capacity)
        self.max_age_minutes = max_age_minutes
        self._memories: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> str:
        """添加工作记忆：先过期清理，再容量管理，最后追加。"""
        self._expire_old_memories()

        evicted_msg = ""
        while len(self._memories) >= self.max_capacity:
            evicted = self._remove_lowest_priority_memory()
            if evicted and not evicted_msg:
                evicted_msg = (f"⚠️ 工作记忆已达容量上限({self.max_capacity})，"
                               f"已淘汰最不重要的记忆: {evicted.content[:40]}...")

        self._memories.append(entry)
        return evicted_msg

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[Tuple[float, float, MemoryEntry, Dict]]:
        """混合检索：TF-IDF向量化 + 关键词匹配。

        评分算法：
          base_relevance = tfidf_score * 0.7 + keyword_score * 0.3  (有TF-IDF时)
                         = keyword_score                             (TF-IDF失败时)
          time_decay = exp(-age_minutes / max_age_minutes)
          importance_weight = 0.8 + importance * 0.4
          final_score = (base_relevance * time_decay) * importance_weight

        Returns:
            [(final_score, base_relevance, memory, details_dict), ...]
            按分数降序排列，截断至 limit 条。
        """
        self._expire_old_memories()

        if not self._memories:
            return []

        vector_scores = self._try_tfidf_search(query)

        scored_memories = []
        for memory in self._memories:
            vector_score = vector_scores.get(memory.memory_id, 0.0)
            keyword_score = self._calculate_keyword_score(query, memory.content)

            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
                method = "hybrid"
            else:
                base_relevance = keyword_score
                method = "keyword"

            time_decay = self._calculate_time_decay(memory.timestamp)
            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * time_decay * importance_weight

            if final_score > 0:
                details = {
                    "method": method,
                    "tfidf_score": vector_score,
                    "keyword_score": keyword_score,
                    "time_decay": time_decay,
                    "importance_weight": importance_weight,
                    "base_relevance": base_relevance,
                    "final_score": final_score,
                }
                scored_memories.append((final_score, base_relevance, memory, details))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return scored_memories[:limit]

    def get_all(self) -> List[MemoryEntry]:
        return list(self._memories)

    def get_count(self) -> int:
        return len(self._memories)

    def clear(self):
        self._memories.clear()

    def set_memories(self, memories: List[MemoryEntry]):
        self._memories = list(memories)

    # ============================================================
    # 过期清理与容量管理
    # ============================================================

    def _expire_old_memories(self):
        """清理超过 TTL 的过期记忆。"""
        if self.max_age_minutes is None or self.max_age_minutes <= 0:
            return

        now = datetime.now()
        kept = []
        for mem in self._memories:
            try:
                mem_time = datetime.fromisoformat(mem.timestamp)
                age_minutes = (now - mem_time).total_seconds() / 60
                if age_minutes < self.max_age_minutes:
                    kept.append(mem)
            except (ValueError, TypeError):
                kept.append(mem)
        self._memories = kept

    def _remove_lowest_priority_memory(self) -> Optional[MemoryEntry]:
        """淘汰重要度最低的记忆（相同则淘汰最早的）。"""
        if not self._memories:
            return None
        self._memories.sort(key=lambda m: (m.importance, m.timestamp))
        return self._memories.pop(0)

    # ============================================================
    # TF-IDF 语义检索
    # ============================================================

    def _try_tfidf_search(self, query: str) -> Dict[str, float]:
        """尝试 TF-IDF 向量化语义检索，失败则返回空字典（触发关键词回退）。

        Returns:
            {memory_id: cosine_similarity} 映射，分数范围 0.0-1.0。
        """
        try:
            if len(self._memories) < 2:
                return {}

            query_tokens = self._tokenize_for_tfidf(query)
            if not query_tokens:
                return {}

            documents = [self._tokenize_for_tfidf(mem.content) for mem in self._memories]

            df = Counter()
            for doc_tokens in documents:
                for token in set(doc_tokens):
                    df[token] += 1

            total_docs = len(documents)
            idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1
                   for token, freq in df.items()}

            query_tf = Counter(query_tokens)
            query_len = len(query_tokens)
            query_vector = {token: (count / query_len) * idf.get(token, 0.0)
                            for token, count in query_tf.items()}

            if all(v == 0 for v in query_vector.values()):
                return {}

            scores = {}
            for mem, doc_tokens in zip(self._memories, documents):
                doc_len = len(doc_tokens) if doc_tokens else 1
                doc_tf = Counter(doc_tokens)
                doc_vector = {token: (count / doc_len) * idf.get(token, 0.0)
                              for token, count in doc_tf.items()}

                similarity = self._cosine_similarity(query_vector, doc_vector)
                if similarity > 0:
                    scores[mem.memory_id] = similarity

            return scores
        except Exception:
            return {}

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """计算两个稀疏向量的余弦相似度。"""
        if not vec_a or not vec_b:
            return 0.0

        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def _tokenize_for_tfidf(text: str) -> List[str]:
        """TF-IDF 专用分词：英文按单词，中文按整词 + 2-gram 展开。"""
        tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text.lower())
        expanded = []
        for token in tokens:
            if token and '\u4e00' <= token[0] <= '\u9fff' and len(token) >= 2:
                expanded.append(token)
                for i in range(len(token) - 1):
                    expanded.append(token[i:i + 2])
            elif token:
                expanded.append(token)
        return expanded

    # ============================================================
    # 关键词匹配
    # ============================================================

    @staticmethod
    def _calculate_keyword_score(query: str, content: str) -> float:
        """关键词匹配评分（0.0-1.0）：Jaccard相似度 + 子串匹配加成。"""
        query_tokens = set(re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', query.lower()))
        content_tokens = set(re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', content.lower()))

        if not query_tokens or not content_tokens:
            return 0.0

        def expand(tokens):
            expanded = set()
            for t in tokens:
                if t and '\u4e00' <= t[0] <= '\u9fff' and len(t) >= 2:
                    expanded.add(t)
                    for i in range(len(t) - 1):
                        expanded.add(t[i:i + 2])
                else:
                    expanded.add(t)
            return expanded

        q_expanded = expand(query_tokens)
        c_expanded = expand(content_tokens)

        intersection = q_expanded & c_expanded
        union = q_expanded | c_expanded
        jaccard = len(intersection) / len(union) if union else 0.0

        bonus = 0.15 if query.lower() in content.lower() else 0.0
        return min(1.0, jaccard + bonus)

    # ============================================================
    # 时间衰减
    # ============================================================

    def _calculate_time_decay(self, timestamp: str) -> float:
        """指数时间衰减：decay = exp(-age_minutes / max_age_minutes)。

        新记忆 ≈ 1.0，接近TTL的记忆 ≈ 1/e ≈ 0.368。
        """
        try:
            mem_time = datetime.fromisoformat(timestamp)
            age_minutes = (datetime.now() - mem_time).total_seconds() / 60
        except (ValueError, TypeError):
            return 0.5

        if self.max_age_minutes is None or self.max_age_minutes <= 0:
            return 1.0

        decay = math.exp(-age_minutes / self.max_age_minutes)
        return max(0.0, min(1.0, decay))

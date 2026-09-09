"""知识库索引 + LLM 客户端 + MQE/HyDE 扩展检索。"""

import os
import re
import math
import uuid
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

from src.tools.rag.rag_tool.models import RagChunk, RagDocument
from src.tools.rag.rag_tool.encoder import VectorEncoder
from src.tools.rag.rag_tool.splitter import _tokenize
from src.core.cache import SafeFullCache


class QwenChatClient:
    """阿里云百炼 Qwen 对话客户端（OpenAI 兼容接口）。"""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model: str = None):
        from openai import OpenAI
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("未配置 DASHSCOPE_API_KEY，无法使用 Qwen 问答")
        self.model = model or os.getenv("RAG_LLM_MODEL", "qwen-plus")
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL, timeout=60)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, stream=False,
        )
        return resp.choices[0].message.content or ""


class QueryExpander:
    """多查询扩展（Multi-Query Expansion，MQE）：提升检索召回。"""

    MQE_SYSTEM_PROMPT = "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"

    @staticmethod
    def build_prompt(query: str, n: int) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": QueryExpander.MQE_SYSTEM_PROMPT},
            {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"},
        ]

    def __init__(self, llm_client=None, threshold: int = 5, batch_size: int = 10,
                 cache: Optional[SafeFullCache] = None):
        self._llm = llm_client
        self._cache = cache
        self.threshold = max(1, threshold)
        self.batch_size = max(1, batch_size)

    def _get_llm(self) -> Optional[QwenChatClient]:
        if self._llm is None:
            try:
                self._llm = QwenChatClient()
            except ValueError:
                self._llm = None
        return self._llm

    def expand(self, query: str, n: int = 3) -> List[str]:
        queries = [query]
        cache_key = f"mqe:{query}:{n}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        llm = self._get_llm()
        if llm is None:
            return queries
        n = max(1, min(10, n))
        try:
            response = llm.chat(self.build_prompt(query, n), temperature=0.7)
        except Exception:
            return queries
        if not response:
            return queries
        seen = {query}
        for line in self._parse_response(response):
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            queries.append(line)
            if len(queries) - 1 >= n:
                break
        if self._cache is not None:
            self._cache.set(cache_key, queries)
        return queries

    @staticmethod
    def _parse_response(response: str) -> List[str]:
        lines = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'^[\d一二三四五六七八九十]+[.、)）:：]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            if line:
                lines.append(line)
        return lines

    def plan(self, query: str, n: int = 3) -> Tuple[List[str], str, int]:
        queries = self.expand(query, n=n)
        total = len(queries)
        mode = "batch" if total > self.threshold else "single"
        return queries, mode, total

    def iter_batches(self, queries: List[str]):
        for start in range(0, len(queries), self.batch_size):
            yield queries[start:start + self.batch_size]


class HydeGenerator:
    """假设文档嵌入（HyDE）：生成假设性答案文档用于向量检索。"""

    HYDE_SYSTEM_PROMPT = "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"

    @staticmethod
    def build_prompt(query: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": HydeGenerator.HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"},
        ]

    def __init__(self, llm_client=None, cache: Optional[SafeFullCache] = None):
        self._llm = llm_client
        self._cache = cache

    def _get_llm(self) -> Optional[QwenChatClient]:
        if self._llm is None:
            try:
                self._llm = QwenChatClient()
            except ValueError:
                self._llm = None
        return self._llm

    def generate(self, query: str) -> str:
        cache_key = f"hyde:{query}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        llm = self._get_llm()
        if llm is None:
            return query
        try:
            doc = llm.chat(self.build_prompt(query), temperature=0.7)
            doc = (doc or "").strip()
            if not doc:
                doc = query
        except Exception:
            doc = query
        if self._cache is not None:
            self._cache.set(cache_key, doc)
        return doc


class RagIndex:
    """知识库索引：稠密向量（嵌入）+ 稀疏 TF-IDF 双路召回。"""

    def __init__(self):
        self._encoder = VectorEncoder()
        self._embedding_client = self._encoder._embedding_client
        self._chunks: Dict[str, RagChunk] = {}
        self._dense: Dict[str, List[float]] = {}
        self._tfidf_vectors: Dict[str, Dict[str, float]] = {}
        self._idf: Dict[str, float] = {}
        self._docs: Dict[str, RagDocument] = {}
        self._doc_chunks: Dict[str, List[str]] = {}

    @property
    def embedding_available(self) -> bool:
        return self._encoder.embedding_available

    def add_chunk(self, chunk: RagChunk) -> bool:
        self._chunks[chunk.chunk_id] = chunk
        dense, sparse, _ = self._encoder.encode(chunk.content)
        if dense:
            self._dense[chunk.chunk_id] = dense
        if sparse:
            self._tfidf_vectors[chunk.chunk_id] = sparse
        return bool(dense)

    def index_chunks(self, chunks: List[RagChunk]) -> Tuple[List[RagChunk], int]:
        if not chunks:
            return chunks, 0
        texts = [c.content for c in chunks]
        embeddings = self._encoder.embed_batch(texts)
        embedded_count = 0
        for chunk, dense in zip(chunks, embeddings):
            self._chunks[chunk.chunk_id] = chunk
            sparse = self._encoder._tfidf_vectorize(chunk.content)
            if sparse:
                self._tfidf_vectors[chunk.chunk_id] = sparse
            if dense:
                self._dense[chunk.chunk_id] = dense
                chunk.metadata["vector_method"] = "embedding"
                embedded_count += 1
            else:
                chunk.metadata["vector_method"] = "tfidf"
            chunk.metadata["tokens"] = _approx_token_len(chunk.content)
        return chunks, embedded_count

    def add_document(self, doc: RagDocument, chunks: List[RagChunk]):
        self._docs[doc.doc_id] = doc
        self._doc_chunks[doc.doc_id] = [c.chunk_id for c in chunks]
        for chunk in chunks:
            tokens = _tokenize(chunk.content)
            for token in set(tokens):
                self._idf[token] = self._idf.get(token, 0.0) + 1.0 / max(1, len(chunks))

    def rebuild_idf(self):
        if not self._chunks:
            return
        total_docs = len(self._docs) or 1
        for term, doc_freq_weight in list(self._idf.items()):
            self._idf[term] = math.log((total_docs + 1) / (doc_freq_weight + 1)) + 1

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Tuple[float, RagChunk, str]]:
        if not self._chunks:
            return []
        dense_scores = self._dense_search(query)
        sparse_scores = self._sparse_search(query)
        results = []
        for chunk_id, chunk in self._chunks.items():
            d = dense_scores.get(chunk_id, 0.0)
            s = sparse_scores.get(chunk_id, 0.0)
            if d > 0 and s > 0:
                score = d * 0.7 + s * 0.3
                method = "hybrid"
            elif d > 0:
                score = d
                method = "embedding"
            elif s > 0:
                score = s
                method = "tfidf"
            else:
                continue
            if score < min_score:
                continue
            results.append((score, chunk, method))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def _dense_search(self, query: str) -> Dict[str, float]:
        if not self._dense or not self._embedding_client or not self._embedding_client.is_available:
            return {}
        query_vec = self._embedding_client.embed(query)
        if not query_vec:
            return {}
        from src.tools.memory.modules import EmbeddingClient as EC
        scores = {}
        for chunk_id, vec in self._dense.items():
            sim = EC.cosine_similarity_dense(query_vec, vec)
            if sim > 0:
                scores[chunk_id] = max(0.0, min(1.0, sim))
        return scores

    def _sparse_search(self, query: str) -> Dict[str, float]:
        if not self._tfidf_vectors:
            return {}
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}
        q_tf = Counter(query_tokens)
        q_len = len(query_tokens)
        q_vec = {t: (c / q_len) * self._idf.get(t, 0.0) for t, c in q_tf.items()}
        if not any(q_vec.values()):
            return {}
        scores = {}
        for chunk_id, doc_vec in self._tfidf_vectors.items():
            sim = self._cosine_sparse(q_vec, doc_vec)
            if sim > 0:
                scores[chunk_id] = sim
        return scores

    @staticmethod
    def _cosine_sparse(vec_a: Dict[str, float], vec_b_counter: Counter) -> float:
        if not vec_a or not vec_b_counter:
            return 0.0
        common = set(vec_a.keys()) & set(vec_b_counter.keys())
        if not common:
            return 0.0
        dot = sum(vec_a[k] * vec_b_counter[k] for k in common)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b_counter.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_docs(self) -> List[RagDocument]:
        return list(self._docs.values())

    def get_chunk_count(self) -> int:
        return len(self._chunks)

    def get_doc_count(self) -> int:
        return len(self._docs)

    def get_doc_chunks(self, doc_id: str) -> List[RagChunk]:
        return [self._chunks[cid] for cid in self._doc_chunks.get(doc_id, [])]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[RagChunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def delete_doc(self, doc_id: str) -> bool:
        if doc_id not in self._docs:
            return False
        for chunk_id in self._doc_chunks.get(doc_id, []):
            self._chunks.pop(chunk_id, None)
            self._dense.pop(chunk_id, None)
            self._tfidf_vectors.pop(chunk_id, None)
        self._doc_chunks.pop(doc_id, None)
        self._docs.pop(doc_id, None)
        return True

    def clear(self):
        self._chunks.clear()
        self._dense.clear()
        self._tfidf_vectors.clear()
        self._idf.clear()
        self._docs.clear()
        self._doc_chunks.clear()


from src.tools.rag.rag_tool.splitter import _approx_token_len
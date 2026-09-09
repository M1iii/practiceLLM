"""Embedding 编码器：百炼嵌入 API 优先，TF-IDF 稀疏向量兜底。"""

from typing import Dict, List, Optional, Tuple
from collections import Counter

from src.tools.rag.rag_tool.splitter import _tokenize

try:
    from src.tools.memory.modules import EmbeddingClient
except ImportError:
    EmbeddingClient = None


def _get_embedding_client():
    if EmbeddingClient is not None:
        return EmbeddingClient()
    return None


class VectorEncoder:
    """统一向量编码接口：百炼嵌入 API 优先，TF-IDF 稀疏向量兜底。"""

    def __init__(self, embedding_client=None):
        self._embedding_client = embedding_client or _get_embedding_client()

    @property
    def embedding_available(self) -> bool:
        return bool(self._embedding_client and self._embedding_client.is_available)

    def encode(self, text: str) -> Tuple[Optional[List[float]], Dict[str, float], str]:
        dense = None
        if self.embedding_available:
            try:
                dense = self._embedding_client.embed(text)
            except Exception:
                dense = None
        sparse = self._tfidf_vectorize(text)
        method = "embedding" if dense else "tfidf"
        return dense, sparse, method

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        if self.embedding_available:
            try:
                return self._embedding_client.embed_batch(texts)
            except Exception:
                pass
        return [None] * len(texts)

    @staticmethod
    def _tfidf_vectorize(text: str) -> Dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        return {term: count / total for term, count in tf.items()}


def index_chunks(chunks: List[Dict], embedding_client=None) -> List[Dict]:
    """统一分块索引：为每个 chunk 生成向量（百炼嵌入优先 + TF-IDF 兜底）。"""
    encoder = VectorEncoder(embedding_client)
    from src.tools.rag.rag_tool.splitter import _approx_token_len
    indexed: List[Dict] = []
    for chunk in chunks:
        content = chunk.get("content", "") or ""
        dense, sparse, method = encoder.encode(content)
        indexed.append({
            **chunk,
            "embedding": dense,
            "tfidf_vector": sparse,
            "vector_method": method,
            "tokens": chunk.get("tokens") or _approx_token_len(content),
        })
    return indexed
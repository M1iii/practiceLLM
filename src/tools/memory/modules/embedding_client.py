"""
嵌入模型客户端（阿里云百炼 DashScope）
=====================================

支持 qwen3-vl-embedding 多模态嵌入 / text-embedding-v3 纯文本嵌入。
"""

import os
import json
import base64
import hashlib
import math
import urllib.request
import urllib.error
from typing import List, Dict, Optional, Any


class EmbeddingClient:
    """阿里云百炼多模态嵌入 API 客户端，支持离线回退。

    特点：
      - 调用 DashScope qwen3-vl-embedding 多模态嵌入模型
      - 支持文本 + 图片多模态输入（独立嵌入 / 融合嵌入）
      - 纯文本调用向后兼容 text-embedding-v3 用法
      - MD5 内容缓存，避免重复 API 调用
      - 自动检测 API Key 可用性，不可用时返回 None 触发 fallback
      - 批量嵌入支持，减少网络请求次数
    """

    DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        self._available = self._check_available()
        self._cache: Dict[str, List[float]] = {}

    def _check_available(self) -> bool:
        """检查 API 是否可用（key 非空且非占位符）。"""
        if not self.api_key:
            return False
        placeholders = {"your_key", "your_api_key", "placeholder", "xxx", ""}
        if self.api_key.lower().strip() in placeholders:
            return False
        return True

    @property
    def is_available(self) -> bool:
        return self._available

    def embed(self, text: str = None, images: List[str] = None,
              enable_fusion: bool = False) -> Optional[List[float]]:
        """获取嵌入向量，支持多模态输入（文本 + 图片）。"""
        if not self._available:
            return None

        contents = []
        if text:
            contents.append({"text": text})
        if images:
            for img_path in images:
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(img_path)[1].lstrip(".") or "png"
                img_data_uri = f"data:image/{ext};base64,{img_b64}"
                contents.append({"image": img_data_uri})

        if not contents:
            return None

        cache_key = hashlib.md5(
            json.dumps(contents, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            embeddings = self._call_api(contents, enable_fusion=enable_fusion)
            if embeddings:
                result = embeddings[0]
                self._cache[cache_key] = result
                return result
        except Exception:
            pass

        return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量获取嵌入向量，减少 API 调用次数。"""
        if not self._available:
            return [None] * len(texts)

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_texts = []
        uncached_keys = []

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_keys.append((i, cache_key))

        if uncached_texts:
            try:
                embeddings = self._call_api(uncached_texts)
                for (idx, cache_key), emb in zip(uncached_keys, embeddings):
                    if emb:
                        results[idx] = emb
                        self._cache[cache_key] = emb
            except Exception:
                pass

        return results

    def _call_api(self, inputs: Any, embedding_type: str = "query",
                  enable_fusion: bool = False) -> List[List[float]]:
        """调用 DashScope 多模态嵌入 API。"""
        if inputs and isinstance(inputs[0], str):
            contents = [{"text": t} for t in inputs]
        else:
            contents = inputs

        params = {}
        if enable_fusion:
            params["enable_fusion"] = True

        data = json.dumps({
            "model": self.model,
            "input": {"contents": contents},
            "parameters": params,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.DASHSCOPE_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        embeddings = []
        for item in result.get("output", {}).get("embeddings", []):
            embeddings.append(item["embedding"])

        return embeddings

    @staticmethod
    def cosine_similarity_dense(vec_a: List[float], vec_b: List[float]) -> float:
        """计算两个稠密向量的余弦相似度。"""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
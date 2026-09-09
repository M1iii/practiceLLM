"""
LlamaIndex 适配器：将项目现有 EmbeddingClient 和 Qwen 对话 API
封装为 LlamaIndex 的 BaseEmbedding 和 LLM 接口。

用法：
    from src.tools.rag.llamaindex.adapters import DashScopeEmbedding, QwenLLM

    embed_model = DashScopeEmbedding()
    llm = QwenLLM()
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)
"""

import os
import sys
from typing import Any, List, Optional, Generator, Sequence


from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CustomLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field

from src.tools.memory.modules import EmbeddingClient


class DashScopeEmbedding(BaseEmbedding):
    """将项目现有的 EmbeddingClient 封装为 LlamaIndex BaseEmbedding。"""

    def __init__(self, client: Optional[EmbeddingClient] = None,
                 embed_batch_size: int = 10, **kwargs):
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        self._client = client or EmbeddingClient()

    @classmethod
    def class_name(cls) -> str:
        return "DashScopeEmbedding"

    def _get_text_embedding(self, text: str) -> List[float]:
        vec = self._client.embed(text)
        if vec is None:
            raise RuntimeError("Embedding API 返回空，请检查 DASHSCOPE_API_KEY 配置")
        return vec

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        results = self._client.embed_batch(texts)
        dim = len(results[0]) if results and results[0] is not None else 1024
        return [r if r is not None else [0.0] * dim for r in results]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)


# ============================================================
# LLM 适配器
# ============================================================

class QwenLLM(CustomLLM):
    """将项目现有 Qwen 对话 API 封装为 LlamaIndex CustomLLM。"""

    model: str = Field(default="qwen-plus", description="Qwen 模型名")
    temperature: float = Field(default=0.3, description="生成温度")
    max_tokens: int = Field(default=4096, description="最大生成 Token 数")

    def __init__(self, model: Optional[str] = None, temperature: float = 0.3,
                 max_tokens: Optional[int] = None, **kwargs):
        super().__init__(
            model=model or os.getenv("RAG_LLM_MODEL", "qwen-plus"),
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            **kwargs,
        )
        self._client: Optional[Any] = None

    @classmethod
    def class_name(cls) -> str:
        return "QwenLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            is_chat_model=True,
        )

    def _get_client(self):
        if self._client is None:
            from src.tools.rag.llamaindex.qwen_client import QwenChatClient
            self._client = QwenChatClient(model=self.model)
        return self._client

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        text = self._get_client().chat(
            [{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
        )
        return CompletionResponse(text=text)

    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        raw = [{"role": m.role.value, "content": m.content} for m in messages]
        text = self._get_client().chat(
            raw,
            temperature=kwargs.get("temperature", self.temperature),
        )
        return ChatResponse(
            message=ChatMessage(role="assistant", content=text),
        )

    def stream_complete(self, prompt: str, **kwargs) -> Generator[CompletionResponse, None, None]:
        full = ""
        for chunk in self._get_client().stream_chat(
            [{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
        ):
            full += chunk
            yield CompletionResponse(text=full, delta=chunk)


def get_default_embed_model() -> DashScopeEmbedding:
    return DashScopeEmbedding()


def get_default_llm() -> QwenLLM:
    return QwenLLM()
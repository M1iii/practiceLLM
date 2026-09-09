"""Qwen 对话客户端（OpenAI 兼容模式），供 LlamaIndex 适配层使用。"""

import os
from typing import Dict, Generator, List, Optional


class QwenChatClient:
    """DashScope Qwen 对话客户端（OpenAI 兼容接口）。"""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")
        self.model = model or os.getenv("RAG_LLM_MODEL", "qwen-plus")
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL, timeout=60)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    def stream_chat(
        self, messages: List[Dict[str, str]], temperature: float = 0.3
    ) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
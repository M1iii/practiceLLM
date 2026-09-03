"""轻量级 Qwen 对话客户端（OpenAI 兼容接口），供 LlamaIndex 适配器使用。"""

import os
from typing import List, Dict, Optional, Generator


class QwenChatClient:
    """阿里云百炼 Qwen 对话客户端（OpenAI 兼容接口）。"""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("未配置 DASHSCOPE_API_KEY，无法使用 Qwen 问答")
        self.model = model or os.getenv("RAG_LLM_MODEL", "qwen-plus")
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL, timeout=60)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """调用 Qwen 对话，返回完整回复文本。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    def stream_chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> Generator[str, None, None]:
        """流式调用 Qwen 对话，逐 chunk 生成文本片段。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
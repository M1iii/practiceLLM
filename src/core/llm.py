import os
import json
import hashlib
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Iterator, Optional, Protocol, runtime_checkable

from src.core.cache import SafeFullCache

# 加载 .env 文件中的环境变量
load_dotenv()


@runtime_checkable
class LLMClient(Protocol):
    """LLM 客户端统一接口。任何实现该接口的对象都可注入 Agent。"""

    model: str

    def invoke(self, messages: List[Dict[str, str]], temperature: float = 0,
               stream: bool = False, **kwargs) -> str: ...

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str: ...

    def stream_chunks(self, messages: List[Dict[str, str]],
                      temperature: float = 0, **kwargs) -> Iterator[str]: ...


class practiceLLM:
    """兼容 OpenAI 接口的 LLM 客户端，支持流式响应和缓存。"""
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None,
                 timeout: int = None,
                 cache_size: int = 50, cache_ttl: int = 3600):
        """初始化客户端，优先使用传入参数，未提供则从环境变量加载。"""
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

        self._cache = SafeFullCache(
            max_size=cache_size, default_ttl=cache_ttl) if cache_size > 0 else None

    def _cache_key(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        raw = json.dumps([self.model, messages, temperature], sort_keys=True,
                         ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _cache_hit(self, key: str) -> Optional[str]:
        return None if self._cache is None else self._cache.get(key)

    def _cache_set(self, key: str, value: str):
        if self._cache is not None:
            self._cache.set(key, value)

    def cache_stats(self) -> dict:
        if self._cache is None:
            return {"enabled": False, "size": 0, "hits": 0, "misses": 0}
        s = self._cache.stats()
        return {"enabled": True, "size": s["size"], "max_size": s["max_size"],
                "hits": s["hits"], "misses": s["misses"],
                "hit_rate": f"{s['hit_rate']:.1%}"}

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        print(f"[LLM] 正在调用 {self.model} 模型...")
        cache_key = self._cache_key(messages, temperature)
        cached = self._cache_hit(cache_key)
        if cached is not None:
            print("[OK] 缓存命中:")
            print(cached)
            return cached

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            print("[OK] 响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            result = "".join(collected_content)
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            print(f"[ERROR] 调用LLM API时发生错误: {e}")
            return None

    def invoke(self, messages: List[Dict[str, str]], temperature: float = 0,
           stream: bool = False, **kwargs) -> str:
        cache_key = self._cache_key(messages, temperature)
        cached = self._cache_hit(cache_key)
        if cached is not None:
            if stream:
                print(cached)
            return cached

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
            if stream:
                collected_content = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                    collected_content.append(content)
                print()
                result = "".join(collected_content)
                self._cache_set(cache_key, result)
                return result
            else:
                result = response.choices[0].message.content
                self._cache_set(cache_key, result)
                return result
        except Exception as e:
            print(f"[ERROR] 调用LLM API时发生错误: {e}")
            return None

    def stream_chunks(self, messages: List[Dict[str, str]],
                      temperature: float = 0, **kwargs) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content


# --- 客户端使用示例 ---
if __name__ == '__main__':
    llm = practiceLLM()
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "你好，请用一句话介绍你自己。"},
    ]
    llm.think(messages)

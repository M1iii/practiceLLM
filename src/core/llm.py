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
    """LLM 客户端统一接口（结构化鸭子类型）。

    任何实现该接口的对象（practiceLLM、mock、第三方包装）都可注入 Agent，
    实现依赖倒置：Agent 只依赖抽象接口，不依赖具体实现。

    必须成员：
      - model: 模型名
      - invoke(): 完整响应（stream 参数控制是否打印式流式）
      - think(): 思考式调用（打印式流式）
      - stream_chunks(): 增量文本块生成器（供 SSE 使用，不打印）
    """

    model: str

    def invoke(self, messages: List[Dict[str, str]], temperature: float = 0,
               stream: bool = False, **kwargs) -> str: ...

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str: ...

    def stream_chunks(self, messages: List[Dict[str, str]],
                      temperature: float = 0, **kwargs) -> Iterator[str]: ...


class practiceLLM:
    """
    为 "practice" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None,
                 timeout: int = None,
                 cache_size: int = 50, cache_ttl: int = 3600):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        :param cache_size: LLM 响应缓存容量（最近 N 次调用结果），0 表示不缓存
        :param cache_ttl: 缓存存活时间（秒），默认 3600s
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

        # LLM 响应缓存：相同 (model, messages, temperature) 直接返回历史结果
        self._cache = SafeFullCache(
            max_size=cache_size, default_ttl=cache_ttl) if cache_size > 0 else None

    def _cache_key(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """生成缓存键：model + messages(JSON稳定序列化) + temperature 的 MD5。"""
        raw = json.dumps([self.model, messages, temperature], sort_keys=True,
                         ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _cache_hit(self, key: str) -> Optional[str]:
        """尝试从缓存读取；无缓存时返回 None。"""
        if self._cache is None:
            return None
        return self._cache.get(key)

    def _cache_set(self, key: str, value: str):
        """写入缓存。"""
        if self._cache is not None:
            self._cache.set(key, value)

    def cache_stats(self) -> dict:
        """缓存统计。"""
        if self._cache is None:
            return {"enabled": False, "size": 0, "hits": 0, "misses": 0}
        s = self._cache.stats()
        return {"enabled": True, "size": s["size"], "max_size": s["max_size"],
                "hits": s["hits"], "misses": s["misses"],
                "hit_rate": f"{s['hit_rate']:.1%}"}

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        # 缓存命中检查
        cache_key = self._cache_key(messages, temperature)
        cached = self._cache_hit(cache_key)
        if cached is not None:
            print(f"🧠 正在调用 {self.model} 模型...")
            print("✅ 缓存命中（返回历史结果）:")
            print(cached)
            return cached

        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            result = "".join(collected_content)
            self._cache_set(cache_key, result)
            return result

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

    def invoke(self, messages: List[Dict[str, str]], temperature: float = 0, stream: bool = False, **kwargs) -> str:
        """
        调用大语言模型，返回完整响应文本。
        :param stream: 为 True 时流式打印响应内容，适用于对话场景；为 False 时直接返回，适用于工具调用等需要完整结果的场景。
        """
        # 缓存命中检查
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
                # 流式模式：实时打印内容，同时收集完整响应
                collected_content = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                    collected_content.append(content)
                print()  # 流式输出结束后换行
                result = "".join(collected_content)
                self._cache_set(cache_key, result)
                return result
            else:
                result = response.choices[0].message.content
                self._cache_set(cache_key, result)
                return result
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

    def stream_chunks(self, messages: List[Dict[str, str]],
                      temperature: float = 0, **kwargs) -> Iterator[str]:
        """
        流式返回文本块（生成器，不打印），供 SSE 等流式输出使用。
        每次 yield 一个增量文本块；异常时 yield 错误标记后结束。
        """
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

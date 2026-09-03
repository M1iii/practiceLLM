import os
import sys
import json
import time
import urllib.request
import urllib.error
from typing import List, Optional


from dotenv import load_dotenv
from src.tools.framework.tool_system import Tool, ToolParameter

load_dotenv()


class AdvancedSearchTool(Tool):
    """
    多源搜索工具，支持博查搜索和 Tavily 两种搜索引擎后端。
    智能选择策略：优先使用博查搜索，若博查不可用则自动回退到 Tavily。
    两个后端的搜索结果统一格式化输出。

    防重试机制：
      - 连续失败 max_consecutive_failures 次后返回终端错误信号
      - 终端错误以 "【搜索不可用】" 开头，明确告知 LLM 不要重试
      - 在 reset_interval 秒后自动重置计数器，允许后续重试
    """

    BOCHA_URL = "https://api.bochaai.com/v1/web-search"
    TAVILY_URL = "https://api.tavily.com/search"

    def __init__(self, max_consecutive_failures: int = 2, reset_interval: int = 60):
        self.bocha_api_key = os.getenv("BOCHA_API_KEY", "")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        # 防重试状态
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._max_consecutive_failures = max_consecutive_failures
        self._reset_interval = reset_interval

    @property
    def name(self) -> str:
        return "AdvancedSearch"

    @property
    def description(self) -> str:
        return (
            "多源搜索引擎工具，支持博查搜索和 Tavily 后端，优先博查、失败自动回退 Tavily，"
            "返回统一格式结果。"
            "适用：实时/最新网络信息、新闻与技术文档检索、按时间范围过滤的搜索。"
            "不适用：私有数据与需登录的查询、本地知识库检索（请用 RagTool）。"
            "注意：如果返回错误消息以【搜索不可用】开头，说明搜索后端已全部不可用，"
            "请勿重试，直接告知用户搜索功能暂不可用。"
        )

    def _check_throttle(self) -> Optional[str]:
        """检查是否触发节流：连续失败超限 → 返回终端错误信号。"""
        now = time.time()
        if self._consecutive_failures >= self._max_consecutive_failures:
            if now - self._last_failure_time > self._reset_interval:
                # 超过重置间隔，自动恢复
                self._consecutive_failures = 0
                return None
            return (
                "【搜索不可用】所有搜索后端均不可用，连续失败 "
                f"{self._consecutive_failures} 次。请勿重试。"
                "请检查网络连接或 API 密钥配置（BOCHA_API_KEY / TAVILY_API_KEY），"
                "确认后重新提问。"
            )
        return None

    def _record_failure(self):
        """记录一次失败。"""
        self._consecutive_failures += 1
        self._last_failure_time = time.time()

    def _record_success(self):
        """成功时重置计数器。"""
        self._consecutive_failures = 0

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询关键词",
                required=True,
            ),
            ToolParameter(
                name="count",
                type="number",
                description="返回结果数量",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="freshness",
                type="string",
                description="时间范围过滤",
                required=False,
                default="noLimit",
                enum=["noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"],
            ),
        ]

    # ============================================================
    # 核心运行逻辑
    # ============================================================

    def run(self, args: dict) -> str:
        query = args.get("query", "").strip()
        count = int(args.get("count", 5))
        freshness = args.get("freshness", "noLimit")

        if not query:
            return "错误：搜索关键词不能为空"

        # 节流检查：连续失败超限 → 终端错误信号
        throttle_msg = self._check_throttle()
        if throttle_msg:
            print(f"⛔ 搜索节流触发（连续失败 {self._consecutive_failures} 次）")
            return throttle_msg

        # 优先使用博查搜索
        if self._is_bocha_available():
            print("🔍 使用博查搜索...")
            result = self._search_bocha(query, count, freshness)
            if result:
                self._record_success()
                return result
            print("⚠️ 博查搜索失败，回退到 Tavily...")

        # 回退到 Tavily
        if self._is_tavily_available():
            print("🔍 使用 Tavily 搜索...")
            result = self._search_tavily(query, count)
            if result:
                self._record_success()
                return result
            print("⚠️ Tavily 搜索失败")

        # 所有后端均失败 → 记录失败计数
        self._record_failure()
        if self._consecutive_failures >= self._max_consecutive_failures:
            return (
                "【搜索不可用】所有搜索后端均不可用，连续失败 "
                f"{self._consecutive_failures} 次。请勿重试。"
                "请检查网络连接或 API 密钥配置（BOCHA_API_KEY / TAVILY_API_KEY），"
                "确认后重新提问。"
            )
        return "错误：所有搜索源均不可用。请在 .env 中配置有效的 BOCHA_API_KEY 或 TAVILY_API_KEY。"

    # ============================================================
    # 搜索源可用性检查
    # ============================================================

    def _is_bocha_available(self) -> bool:
        """检查博查搜索是否可用（API Key 已配置且非占位符）。"""
        return bool(self.bocha_api_key) and not self.bocha_api_key.startswith("sk-xxx")

    def _is_tavily_available(self) -> bool:
        """检查 Tavily 搜索是否可用。"""
        return bool(self.tavily_api_key) and not self.tavily_api_key.startswith("tvly-xxx")

    # ============================================================
    # 博查搜索后端
    # ============================================================

    def _search_bocha(self, query: str, count: int, freshness: str) -> Optional[str]:
        """调用博查 Web Search API。"""
        try:
            payload = json.dumps({
                "query": query,
                "count": count,
                "freshness": freshness,
                "summary": True,
            }).encode("utf-8")

            req = urllib.request.Request(
                self.BOCHA_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.bocha_api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # 业务码检查（code=200 表示成功）
            if data.get("code") != 200:
                print(f"  博查搜索业务错误: code={data.get('code')} msg={data.get('msg')}")
                return None

            # 解析博查响应：data.data.webPages.value[] 为网页结果数组
            inner = data.get("data") or data
            results = []
            web_pages = inner.get("webPages", {})
            for item in web_pages.get("value", []):
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("summary") or item.get("snippet", ""),
                    "source": item.get("siteName", ""),
                })

            if not results:
                return None

            return self._format_results(results, "博查搜索", query)

        except urllib.error.HTTPError as e:
            print(f"  博查搜索 HTTP 错误: {e.code} {e.reason}")
            return None
        except Exception as e:
            print(f"  博查搜索错误: {e}")
            return None

    # ============================================================
    # Tavily 搜索后端
    # ============================================================

    def _search_tavily(self, query: str, count: int) -> Optional[str]:
        """调用 Tavily Search API。"""
        try:
            payload = json.dumps({
                "query": query,
                "max_results": count,
                "search_depth": "basic",
                "include_answer": False,
            }).encode("utf-8")

            req = urllib.request.Request(
                self.TAVILY_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.tavily_api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # 解析 Tavily 响应：results[] 为结果数组
            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "source": "",
                })

            if not results:
                return None

            return self._format_results(results, "Tavily", query)

        except urllib.error.HTTPError as e:
            print(f"  Tavily 搜索 HTTP 错误: {e.code} {e.reason}")
            return None
        except Exception as e:
            print(f"  Tavily 搜索错误: {e}")
            return None

    # ============================================================
    # 统一格式化输出
    # ============================================================

    @staticmethod
    def _format_results(results: list, source: str, query: str) -> str:
        """将不同搜索源的结果统一格式化为文本。"""
        lines = [
            f"搜索来源: {source}",
            f"查询: {query}",
            f"共找到 {len(results)} 条结果",
            "",
        ]

        for i, item in enumerate(results, 1):
            lines.append(f"[{i}] {item['title']}")
            lines.append(f"    URL: {item['url']}")
            if item.get("source"):
                lines.append(f"    来源: {item['source']}")
            # 摘要截断，避免过长
            snippet = item.get("snippet", "")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            lines.append(f"    摘要: {snippet}")
            lines.append("")

        return "\n".join(lines)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    search_tool = AdvancedSearchTool()

    # 检查搜索源可用性
    print("=" * 60)
    print("📋 多源搜索工具演示")
    print("=" * 60)
    print(f"博查搜索可用: {search_tool._is_bocha_available()}")
    print(f"Tavily搜索可用: {search_tool._is_tavily_available()}")
    print()

    # 工具参数信息
    print("--- 工具参数 ---")
    for p in search_tool.get_parameters():
        tag = "必填" if p.required else "可选"
        print(f"  {p.name} ({p.type}, {tag}): {p.description}")
    print()

    # OpenAI function calling 格式
    print("--- OpenAI Function Calling 格式 ---")
    import json
    print(json.dumps(search_tool.to_openai_format(), ensure_ascii=False, indent=2))
    print()

    # 执行搜索（需要配置有效的 API Key）
    print("--- 执行搜索 ---")
    result = search_tool.run({"query": "Python AI Agent 开发", "count": 3})
    print(result)
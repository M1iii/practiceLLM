"""
StructuredDataTool: 结构化数据查询工具。

自然语言 → SQL → 执行 → 结果，纯结构化数据（用户表、配置表等）专用工具，
与 RAG 语义检索工具独立，Agent 根据查询意图自动选择调用。

支持：
  - 自动检测数据库表结构，无需手动配置
  - 自然语言查询自动转换为 SQL
  - SQL 执行结果自然语言格式化返回

用法：
    from src.tools.structured_data.structured_data_tool import StructuredDataTool

    tool = StructuredDataTool()
    result = tool.run({
        "action": "query",
        "question": "查找用户张三的电话号码",
    })
    print(result)
"""

import os
import logging
from typing import List

from src.tools.framework.tool_system import Tool, ToolParameter
from src.tools.framework.tool_response import ToolResponse
from src.tools.structured_data.sql_executor import SQLExecutor

logger = logging.getLogger(__name__)


class StructuredDataTool(Tool):
    """结构化数据查询工具（自然语言 → SQL → 结果）。

    支持 action:
      - query: 自然语言查询，返回自然语言结果
      - query_raw: 自然语言查询，返回 JSON 格式结果
      - list: 列出数据库中所有可用的用户表
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self._connection_string = config.get("connection_string")
        self._llm_model = config.get("llm_model")
        self._max_results = config.get("max_results", 20)
        self._executor: SQLExecutor | None = None

    @property
    def name(self) -> str:
        return "StructuredDataTool"

    @property
    def description(self) -> str:
        return (
            "结构化数据查询工具，支持自然语言查询 PostgreSQL 数据库中的表格数据，"
            "自动检测表结构，自动生成 SQL，自动返回结果。"
            "适用：用户信息表、配置表、业务数据表等纯结构化数据的精确查询，"
            "不适用：文本文档知识库检索（请用 LlamaIndexRAGTool）。"
            "注意：需要 PostgreSQL 数据库配置（复用 RAG 的 PG_DATABASE）。"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型",
                enum=["query", "query_raw", "list"],
                required=True,
            ),
            ToolParameter(
                name="question",
                type="string",
                description="自然语言查询问题（action=query/query_raw 需要）",
                required=False,
            ),
        ]

    def run(self, args: dict) -> ToolResponse:
        """工具执行入口。"""
        action = args.get("action", "query")
        question = args.get("question")

        # 惰性初始化执行器
        if self._executor is None:
            self._executor = SQLExecutor(
                llm_model=self._llm_model,
                max_results=self._max_results,
                connection_string=self._connection_string,
            )

        try:
            if action == "list":
                result = self._executor.list_tables()
                return ToolResponse(text=result, success=True)

            if action == "query":
                if not question:
                    return ToolResponse.error("query action 需要提供 question 参数")
                result = self._executor.query(question, format_result=True)
                return ToolResponse(text=result, success=True)

            if action == "query_raw":
                if not question:
                    return ToolResponse.error("query_raw action 需要提供 question 参数")
                result = self._executor.query(question, format_result=False)
                return ToolResponse(text=result, success=True)

            return ToolResponse.error(f"不支持的 action: {action}")

        except Exception as e:
            logger.exception("StructuredDataTool 执行出错")
            return ToolResponse.error(f"执行出错: {e}")

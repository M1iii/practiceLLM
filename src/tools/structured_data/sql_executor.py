"""
SQL 生成与执行器。

将自然语言查询 → LLM 生成 SQL → 执行 SQL → 返回格式化结果。

用法：
    from src.tools.structured_data.sql_executor import SQLExecutor

    executor = SQLExecutor()
    result = executor.query("查找张三的电话")
    print(result)  # "张三的电话是 13800138000"
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Row

from src.core.llm import practiceLLM
from src.tools.structured_data.db_schema import DatabaseInspector, TableInfo

logger = logging.getLogger(__name__)


class SQLExecutor:
    """SQL 生成与执行器。

    流程：
      1. 自动检测数据库表结构
      2. 根据自然语言查询，匹配相关表
      3. LLM 生成 SQL 语句
      4. 执行 SQL 并返回格式化结果
      5. 结果可再经 LLM 转为自然语言
    """

    SQL_GENERATION_PROMPT = """你是一个专业的 SQL 生成助手。根据给定的数据库表结构和用户问题，生成 SQL 查询语句。

{context}

用户问题：{question}

请生成 PostgreSQL 兼容的 SQL 查询语句。
要求：
1. 只返回纯 SQL 语句，不要包含 markdown 代码块标记（```sql 等）
2. 如果问题无法通过 SQL 查询回答，请返回 "-- 无法回答: 原因"
3. 表名和列名如果包含大写字母，请使用双引号包裹
4. 使用 LIMIT 限制结果数量，最多 {limit} 条
5. 对于模糊匹配，使用 ILIKE 而不是 LIKE 以支持中文
6. 除非必要，不要使用 JOIN 查询不相关的表

SQL:"""

    RESULT_FORMAT_PROMPT = """你是一个数据呈现助手。请将 SQL 查询结果用自然语言描述给用户。

查询结果（JSON 格式）：
{result}

用户原始问题：{question}

请用简洁的、口语化的中文回答用户的问题。如果结果为空，请告知用户没有找到匹配的数据。"""

    def __init__(
        self,
        llm_model: Optional[str] = None,
        max_results: int = 20,
        connection_string: Optional[str] = None,
    ):
        """
        初始化 SQL 执行器。

        Args:
            llm_model: 用于生成 SQL 的 LLM 模型名，默认使用环境变量 LLM_MODEL_ID。
            max_results: 查询结果最大行数。
            connection_string: 数据库连接字符串，默认从环境变量构建。
        """
        self._llm_model = llm_model or os.getenv("LLM_MODEL_ID", "qwen-plus")
        self._max_results = max_results
        self._connection_string = connection_string
        self._inspector = DatabaseInspector(connection_string=connection_string)
        self._engine: Optional[Engine] = None
        self._llm: Optional[practiceLLM] = None

    def _get_llm(self) -> practiceLLM:
        if self._llm is None:
            self._llm = practiceLLM()
        return self._llm

    def _get_engine(self) -> Engine:
        if self._engine is None:
            cs = self._connection_string or self._inspector._build_connection_string()
            self._engine = create_engine(cs)
        return self._engine

    def _extract_sql(self, llm_response: str) -> str:
        """从 LLM 响应中提取纯 SQL 语句。"""
        # 移除可能的 markdown 代码块标记
        text = llm_response.strip()
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # 检查是否是无法回答的标记
        if text.startswith("-- 无法回答"):
            return text

        # 确保以 SELECT / WITH 等 SQL 关键词开头
        sql_keywords = (
            "SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE",
        )
        if not text.upper().startswith(sql_keywords):
            # 尝试从响应中提取第一条 SQL 语句
            for kw in sql_keywords:
                idx = text.upper().find(kw)
                if idx >= 0:
                    text = text[idx:]
                    break
            else:
                return f"-- 无法解析: LLM 响应未包含有效 SQL\n{text}"

        # 确保有 LIMIT
        if "LIMIT" not in text.upper():
            text = text.rstrip(";") + f" LIMIT {self._max_results}"

        return text

    def _execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """执行 SQL 并返回结果列表。"""
        engine = self._get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
            return [dict(zip(columns, row)) for row in rows]

    def query(self, question: str, format_result: bool = True) -> str:
        """
        执行自然语言查询 → SQL → 结果 的完整链路。

        Args:
            question: 用户的自然语言查询。
            format_result: 是否将结果转为自然语言描述。

        Returns:
            查询结果字符串。
        """
        # Step 1: 检测数据库结构
        all_tables = self._inspector.get_all_tables()

        if not all_tables:
            return "数据库中没有找到可查询的用户表。请先创建数据表。"

        # Step 2: 匹配相关表
        related_tables = self._inspector.get_related_tables([question], top_n=3)
        if not related_tables:
            # 没有匹配到关键词，使用所有表
            related_tables = all_tables[:3]

        # Step 3: LLM 生成 SQL
        schema_context = self._inspector.format_schema_context(related_tables)
        sql_prompt = self.SQL_GENERATION_PROMPT.format(
            context=schema_context,
            question=question,
            limit=self._max_results,
        )

        llm = self._get_llm()
        messages = [
            {"role": "system", "content": "你是一个专业的 SQL 生成助手，只输出 SQL 语句。"},
            {"role": "user", "content": sql_prompt},
        ]
        llm_response = llm.invoke(messages, temperature=0.1, stream=False)
        sql = self._extract_sql(llm_response)

        # Step 4: 检查是否无法生成 SQL
        if sql.startswith("-- 无法回答"):
            return sql.replace("-- 无法回答: ", "")
        if sql.startswith("-- 无法解析"):
            return f"无法为查询生成 SQL: {question}"

        # Step 5: 执行 SQL
        try:
            rows = self._execute_sql(sql)
        except Exception as e:
            logger.warning("SQL 执行失败: %s\nSQL: %s", e, sql)
            return f"查询执行出错: {e}"

        # Step 6: 格式化结果
        if not rows:
            return "没有找到匹配的数据。"

        if not format_result:
            return json.dumps(rows, ensure_ascii=False, indent=2)

        # 自然语言格式化
        result_prompt = self.RESULT_FORMAT_PROMPT.format(
            result=json.dumps(rows, ensure_ascii=False, indent=2),
            question=question,
        )
        result_messages = [
            {"role": "system", "content": "你是一个数据呈现助手，用自然语言回答用户问题。"},
            {"role": "user", "content": result_prompt},
        ]
        final_answer = llm.invoke(result_messages, temperature=0.3, stream=False)
        return final_answer.strip()

    def query_raw(self, question: str) -> str:
        """
        执行自然语言查询，返回 JSON 格式结果（不经过自然语言格式化）。

        Args:
            question: 用户的自然语言查询。

        Returns:
            JSON 格式的查询结果。
        """
        return self.query(question, format_result=False)

    def list_tables(self) -> str:
        """列出数据库中所有可用的用户表。"""
        tables = self._inspector.get_all_tables()
        if not tables:
            return "数据库中没有可查询的用户表。"

        parts = ["数据库中的可用表："]
        for t in tables:
            cols = ", ".join(c.summary() for c in t.columns[:6])
            if len(t.columns) > 6:
                cols += f", ... 共 {len(t.columns)} 列"
            parts.append(f"  - {t.name}: {cols}")
        return "\n".join(parts)
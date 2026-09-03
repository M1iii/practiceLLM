"""
数据库元数据自动检测模块。

运行时自动连接 PostgreSQL，检测所有非系统表的结构信息，
包括列名、数据类型、可空性、主键、外键关系等。

用法：
    from src.tools.structured_data.db_schema import DatabaseInspector

    inspector = DatabaseInspector()
    tables = inspector.get_all_tables()
    for table in tables:
        print(table.summary())
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """数据库列信息。"""
    name: str
    type: str
    nullable: bool
    primary_key: bool
    default: Optional[str] = None
    comment: Optional[str] = None
    foreign_key: Optional[str] = None  # "ref_table.ref_column"

    def summary(self) -> str:
        parts = [f"{self.name} {self.type}"]
        if self.primary_key:
            parts.append("PK")
        if self.foreign_key:
            parts.append(f"FK->{self.foreign_key}")
        if self.nullable:
            parts.append("NULL")
        if self.comment:
            parts.append(f"({self.comment})")
        return " ".join(parts)


@dataclass
class TableInfo:
    """数据库表信息。"""
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None

    @property
    def primary_key_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.primary_key]

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def summary(self) -> str:
        cols = ", ".join(c.summary() for c in self.columns)
        count = f" (约 {self.row_count} 行)" if self.row_count is not None else ""
        return f"表 {self.name}{count}:\n  {cols}"

    def to_prompt_context(self) -> str:
        """生成供 LLM 使用的表结构描述文本。"""
        lines = [f"表名: {self.name}"]
        if self.row_count is not None:
            lines[0] += f" (约 {self.row_count} 行)"
        lines.append("列信息:")
        for col in self.columns:
            desc = f"  - {col.name} ({col.type})"
            if col.primary_key:
                desc += " 主键"
            if col.foreign_key:
                desc += f" 外键 -> {col.foreign_key}"
            if col.comment:
                desc += f" 注释: {col.comment}"
            lines.append(desc)
        return "\n".join(lines)


class DatabaseInspector:
    """数据库元数据检测器，自动获取表结构信息。"""

    EXCLUDED_TABLES = {
        "llamaindex_rag",        # RAG 索引表
        "llamaindex_rag_data",   # RAG 数据表
        "data_ingestion_log",    # 日志表
        "spatial_ref_sys",       # PostGIS 系统表
        "pg_stat_statements",    # 统计信息表
        "llamaindex",            # 可能的旧 RAG 表
    }

    def __init__(self, connection_string: Optional[str] = None):
        """
        初始化数据库检测器。
        
        Args:
            connection_string: PostgreSQL 连接字符串。
                              默认为 None，从环境变量 PG_* 构建。
        """
        self._connection_string = connection_string or self._build_connection_string()
        self._engine: Optional[Engine] = None
        self._cache: Optional[List[TableInfo]] = None

    def _build_connection_string(self) -> str:
        """从环境变量构建 PostgreSQL 连接字符串。"""
        host = os.getenv("PG_HOST", "127.0.0.1")
        port = os.getenv("PG_PORT", "5433")
        user = os.getenv("PG_USER", "postgres")
        password = os.getenv("PG_PASSWORD", "")
        database = os.getenv("PG_DATABASE", "practice_llm")
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def _get_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self._connection_string)
        return self._engine

    def get_all_tables(self, skip_excluded: bool = True) -> List[TableInfo]:
        """
        获取数据库中所有用户表的结构信息。
        
        Args:
            skip_excluded: 是否跳过 RAG 系统表等被排除的表。
        
        Returns:
            表信息列表。
        """
        if self._cache is not None:
            return self._cache

        engine = self._get_engine()
        insp = inspect(engine)

        # 获取所有表名
        raw_tables = insp.get_table_names()
        if skip_excluded:
            raw_tables = [t for t in raw_tables if t not in self.EXCLUDED_TABLES]

        tables = []
        for table_name in raw_tables:
            try:
                columns = []
                for col in insp.get_columns(table_name):
                    col_type = str(col.get("type", "unknown"))
                    # 获取外键信息
                    fk_str = None
                    fks = insp.get_foreign_keys(table_name)
                    for fk in fks:
                        if fk["constrained_columns"] and fk["constrained_columns"][0] == col["name"]:
                            ref_table = fk.get("referred_table", "?")
                            ref_cols = fk.get("referred_columns", ["?"])
                            fk_str = f"{ref_table}.{ref_cols[0]}"

                    columns.append(ColumnInfo(
                        name=col["name"],
                        type=col_type,
                        nullable=col.get("nullable", True),
                        primary_key=col.get("primary_key", False),
                        default=str(col.get("default", "")) if col.get("default") else None,
                        comment=col.get("comment"),
                        foreign_key=fk_str,
                    ))

                tables.append(TableInfo(name=table_name, columns=columns))
            except Exception as e:
                logger.warning("检测表 %s 结构时出错: %s", table_name, e)

        self._cache = tables
        return tables

    def get_table(self, table_name: str) -> Optional[TableInfo]:
        """获取指定表的结构信息。"""
        tables = self.get_all_tables()
        for t in tables:
            if t.name == table_name:
                return t
        return None

    # 中文分词辅助：常见查询动词/介词，用于从查询中切分关键词
    CHINESE_QUERY_SEPARATORS = {
        "查找", "查询", "搜索", "统计", "计算", "列出", "显示", "展示",
        "的", "中", "在", "和", "与", "或", "关于", "按照", "根据",
        "每个", "所有", "全部", "哪些", "什么", "谁", "哪个",
        "怎么", "如何", "多少", "几", "第",
    }

    def get_related_tables(self, queries: List[str], top_n: int = 3) -> List[TableInfo]:
        """
        根据查询关键词，找出最可能相关的表。
        
        简单关键词匹配：将查询拆分为 token，与表名、列名、注释匹配。
        
        Args:
            queries: 查询关键词列表。
            top_n: 返回最多的表数。
        
        Returns:
            TableInfo 列表，按相关性排序。
        """
        all_tables = self.get_all_tables()
        if not all_tables:
            return []

        keywords = set()
        for q in queries:
            # 1. 按逗号切分
            tokens = q.replace("，", ",").replace(" ", ",").split(",")
            # 2. 按中文查询动词/介词切分
            for sep in self.CHINESE_QUERY_SEPARATORS:
                q = q.replace(sep, ",")
            tokens.extend(q.split(","))
            # 3. 提取单字和双字词（中文查询的有效成分）
            for ch in q:
                if '\u4e00' <= ch <= '\u9fff' and len(ch.strip()) > 0:
                    tokens.append(ch)
            # 处理
            for token in tokens:
                token = token.strip().lower()
                if token and len(token) >= 1:
                    keywords.add(token)

        scored = []
        for table in all_tables:
            score = 0
            table_lower = table.name.lower()
            # 表名匹配
            for kw in keywords:
                if kw in table_lower:
                    score += 10
            # 列名匹配
            for col in table.columns:
                col_lower = col.name.lower()
                for kw in keywords:
                    if kw in col_lower:
                        score += 3
                    if col.comment and kw in col.comment.lower():
                        score += 2
            scored.append((score, table))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for score, t in scored[:top_n] if score > 0]

    def format_schema_context(self, tables: List[TableInfo]) -> str:
        """
        将表结构信息格式化为 LLM 可理解的上下文文本。
        
        Args:
            tables: 表信息列表。
        
        Returns:
            格式化的数据库 schema 描述。
        """
        parts = ["以下是数据库中可用的表结构：\n"]
        for table in tables:
            parts.append(table.to_prompt_context())
            parts.append("")
        parts.append("请根据用户问题，生成对应的 SQL 查询语句。")
        parts.append("注意：")
        parts.append("1. 只返回 SQL 语句，不要包含多余解释")
        parts.append("2. 如果问题无法通过 SQL 查询回答，说明原因")
        parts.append("3. 使用中文列名作为结果标签")
        return "\n".join(parts)
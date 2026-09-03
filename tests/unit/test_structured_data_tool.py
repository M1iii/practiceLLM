"""
StructuredDataTool 单元测试。

测试不依赖数据库连接，只测试纯逻辑部分：
- DatabaseInspector 关键词匹配相关表
- SQLExecutor SQL 提取（markdown 清理、LIMIT 补全）
- Metadata 数据类构造和格式化
- StructuredDataTool 参数验证
"""

import sys
import os

# 确保项目根目录在 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.tools.structured_data.db_schema import ColumnInfo, TableInfo, DatabaseInspector
from src.tools.structured_data.sql_executor import SQLExecutor
from src.tools.structured_data.structured_data_tool import StructuredDataTool


def test_ColumnInfo_basic_construction():
    """ColumnInfo 基本构造测试。"""
    col = ColumnInfo(
        name="phone",
        type="varchar(20)",
        nullable=True,
        primary_key=False,
    )
    assert col.name == "phone"
    assert col.type == "varchar(20)"
    assert col.nullable is True
    assert col.primary_key is False
    assert col.foreign_key is None
    assert col.comment is None


def test_ColumnInfo_summary_output():
    """ColumnInfo summary 输出测试。"""
    col_pk = ColumnInfo(
        name="id",
        type="integer",
        nullable=False,
        primary_key=True,
    )
    summary = col_pk.summary()
    assert "id" in summary
    assert "integer" in summary
    assert "PK" in summary

    col_fk = ColumnInfo(
        name="user_id",
        type="integer",
        nullable=False,
        primary_key=False,
        foreign_key="users.id",
    )
    summary_fk = col_fk.summary()
    assert "FK->users.id" in summary_fk

    col_comment = ColumnInfo(
        name="email",
        type="varchar",
        nullable=True,
        primary_key=False,
        comment="用户邮箱",
    )
    summary_comment = col_comment.summary()
    assert "(用户邮箱)" in summary_comment


def test_TableInfo_primary_key_extraction():
    """TableInfo primary_key_columns 测试。"""
    columns = [
        ColumnInfo(name="id", type="int", nullable=False, primary_key=True),
        ColumnInfo(name="name", type="varchar", nullable=False, primary_key=False),
    ]
    table = TableInfo(name="users", columns=columns)
    assert table.primary_key_columns == ["id"]


def test_TableInfo_column_names():
    """TableInfo column_names 测试。"""
    columns = [
        ColumnInfo(name="id", type="int", nullable=False, primary_key=True),
        ColumnInfo(name="name", type="varchar", nullable=False, primary_key=False),
        ColumnInfo(name="phone", type="varchar", nullable=True, primary_key=False),
    ]
    table = TableInfo(name="users", columns=columns)
    assert table.column_names == ["id", "name", "phone"]


def test_TableInfo_to_prompt_context():
    """TableInfo to_prompt_context 测试。"""
    columns = [
        ColumnInfo(name="id", type="integer", nullable=False, primary_key=True),
        ColumnInfo(name="name", type="varchar(100)", nullable=False, primary_key=False, comment="用户名"),
        ColumnInfo(name="phone", type="varchar(20)", nullable=True, primary_key=False),
    ]
    table = TableInfo(name="users", columns=columns, row_count=1000)
    ctx = table.to_prompt_context()
    assert "表名: users" in ctx
    assert "(约 1000 行)" in ctx
    assert "id" in ctx
    assert "主键" in ctx
    assert "name" in ctx
    assert "用户名" in ctx
    assert "phone" in ctx


def test_DatabaseInspector_excluded_tables_config():
    """DatabaseInspector 排除表配置测试。"""
    inspector = DatabaseInspector()
    assert "llamaindex_rag" in inspector.EXCLUDED_TABLES
    assert "llamaindex_rag_data" in inspector.EXCLUDED_TABLES


def test_DatabaseInspector_get_related_tables_keyword_match():
    """DatabaseInspector 关键词匹配相关表测试。"""
    inspector = DatabaseInspector()
    table1 = TableInfo(
        name="users",
        columns=[
            ColumnInfo(name="id", type="int", nullable=False, primary_key=True, comment="用户ID"),
            ColumnInfo(name="name", type="varchar", nullable=True, primary_key=False, comment="用户姓名"),
            ColumnInfo(name="phone", type="varchar", nullable=True, primary_key=False, comment="联系电话"),
        ],
    )
    table2 = TableInfo(
        name="orders",
        columns=[
            ColumnInfo(name="id", type="int", nullable=False, primary_key=True, comment="订单ID"),
            ColumnInfo(name="user_id", type="int", nullable=True, primary_key=False, comment="用户ID"),
            ColumnInfo(name="amount", type="numeric", nullable=True, primary_key=False, comment="订单金额"),
        ],
    )
    table3 = TableInfo(
        name="products",
        columns=[
            ColumnInfo(name="id", type="int", nullable=False, primary_key=True, comment="产品ID"),
            ColumnInfo(name="product_name", type="varchar", nullable=True, primary_key=False, comment="产品名称"),
            ColumnInfo(name="price", type="numeric", nullable=True, primary_key=False, comment="产品价格"),
        ],
    )
    inspector._cache = [table1, table2, table3]

    # 查询 "查找张三的电话" 应该匹配 users 表（phone 列注释含"电话"）
    related = inspector.get_related_tables(["查找张三的电话"], top_n=3)
    assert len(related) >= 1
    assert related[0].name == "users"

    # 查询 "订单金额统计" 应该匹配 orders
    related2 = inspector.get_related_tables(["订单金额统计"], top_n=3)
    assert len(related2) >= 1
    assert related2[0].name == "orders"

    # 查询 "产品价格" → 匹配 products
    related3 = inspector.get_related_tables(["产品价格"], top_n=3)
    assert len(related3) >= 1
    assert related3[0].name == "products"


def test_DatabaseInspector_format_schema_context():
    """DatabaseInspector format_schema_context 测试。"""
    inspector = DatabaseInspector()
    table1 = TableInfo(
        name="users",
        columns=[
            ColumnInfo(name="id", type="integer", nullable=False, primary_key=True),
            ColumnInfo(name="name", type="varchar", nullable=False, primary_key=False),
        ],
    )
    table2 = TableInfo(
        name="orders",
        columns=[
            ColumnInfo(name="id", type="integer", nullable=False, primary_key=True),
            ColumnInfo(name="user_id", type="integer", nullable=False, primary_key=False),
        ],
    )
    ctx = inspector.format_schema_context([table1, table2])
    assert "users" in ctx
    assert "orders" in ctx
    assert "id" in ctx
    assert "主键" in ctx


def test_SQLExecutor_extract_pure_sql():
    """SQLExecutor 纯 SQL 提取测试。"""
    ex = SQLExecutor()
    sql = ex._extract_sql("SELECT name, phone FROM users WHERE name = '张三'")
    assert sql.startswith("SELECT name")
    assert "LIMIT 20" in sql


def test_SQLExecutor_extract_from_markdown_code_block():
    """SQLExecutor 从 markdown 代码块提取测试。"""
    ex = SQLExecutor()
    input_text = """```sql
SELECT name, phone FROM users
WHERE name ILIKE '%张三%'
```"""
    sql = ex._extract_sql(input_text)
    assert sql.startswith("SELECT name")
    assert "ILIKE" in sql
    assert "```" not in sql


def test_SQLExecutor_extract_cannot_answer():
    """SQLExecutor 无法回答提取测试。"""
    ex = SQLExecutor()
    sql = ex._extract_sql("-- 无法回答: 数据库中没有相关的表")
    assert sql.startswith("-- 无法回答")


def test_SQLExecutor_extract_add_limit_when_missing():
    """SQLExecutor 缺失 LIMIT 自动补充测试。"""
    ex = SQLExecutor()
    sql = ex._extract_sql("SELECT * FROM users")
    assert "LIMIT 20" in sql


def test_SQLExecutor_extract_keep_existing_limit():
    """SQLExecutor 已有 LIMIT 不重复添加测试。"""
    ex = SQLExecutor()
    sql = ex._extract_sql("SELECT * FROM users LIMIT 10")
    assert sql.count("LIMIT") == 1
    assert "10" in sql


def test_SQLExecutor_extract_with_trailing_semicolon():
    """SQLExecutor 处理末尾分号测试。"""
    ex = SQLExecutor()
    sql = ex._extract_sql("SELECT name FROM users;")
    assert sql.endswith("LIMIT 20")


def test_StructuredDataTool_basic_properties():
    """StructuredDataTool 基本属性测试。"""
    tool = StructuredDataTool()
    assert tool.name == "StructuredDataTool"
    assert "结构化数据" in tool.description


def test_StructuredDataTool_get_parameters():
    """StructuredDataTool 参数声明测试。"""
    tool = StructuredDataTool()
    params = tool.get_parameters()
    assert len(params) == 2
    action_param = next(p for p in params if p.name == "action")
    assert action_param.required is True
    assert set(action_param.enum) == {"query", "query_raw", "list"}
    question_param = next(p for p in params if p.name == "question")
    assert question_param.required is False


def test_StructuredDataTool_lazy_initialization():
    """StructuredDataTool 惰性初始化测试。"""
    tool = StructuredDataTool()
    assert tool._executor is None


def test_StructuredDataTool_error_missing_question():
    """StructuredDataTool 缺少 question 参数测试。"""
    tool = StructuredDataTool()
    resp = tool.run({"action": "query"})
    assert resp.is_error
    assert "需要提供 question" in resp.output


def test_StructuredDataTool_error_unknown_action():
    """StructuredDataTool 未知 action 测试。"""
    tool = StructuredDataTool()
    resp = tool.run({"action": "unknown"})
    assert resp.is_error
    assert "不支持的 action" in resp.output
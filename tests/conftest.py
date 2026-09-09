"""pytest 公共配置：项目根目录加入 sys.path + PostgreSQL 可用性检查。"""
import sys
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def is_pg_available() -> bool:
    """检查 PostgreSQL 是否可连接。"""
    try:
        from src.core.storage import PostgreSQLBackend
        backend = PostgreSQLBackend()
        backend.close()
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(
    not is_pg_available(),
    reason="PostgreSQL 不可用，跳过依赖 PG 的测试",
)

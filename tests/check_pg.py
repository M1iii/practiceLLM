"""检查 PostgreSQL 和 pgvector 是否可用。"""
import psycopg2

try:
    conn = psycopg2.connect(
        host="127.0.0.1", port=5432,
        user="postgres", password="", dbname="postgres"
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT version()")
    print("PostgreSQL:", cur.fetchone()[0])
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    has_vector = cur.fetchone() is not None
    print("pgvector:", "已安装" if has_vector else "未安装")
    conn.close()
    print("连接成功 ✅")
except Exception as e:
    print("连接失败:", e)
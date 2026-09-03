# ============================================================
# 单阶段构建：python:3.12-slim + pip 安装 uv + /opt/venv
# ============================================================
FROM python:3.12-slim

# 安装 uv（使用默认 PyPI，清华镜像未同步 uv）
RUN pip install uv --no-cache-dir

# 创建独立虚拟环境（避免 bind mount 覆盖 /app 下的 .venv）
RUN uv venv /opt/venv

# 将虚拟环境加入 PATH
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 复制依赖文件，利用 Docker 层缓存加速构建
COPY pyproject.toml README.md ./

# 安装项目依赖到 /opt/venv（避免 bind mount 覆盖 /app/.venv）
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --no-dev

# 容器默认命令
CMD ["python", "run.py"]
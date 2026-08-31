FROM docker.1ms.run/library/postgres:18

RUN apt-get update && \
    apt-get install -y postgresql-18-pgvector && \
    rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD pg_isready -U postgres
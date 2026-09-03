"""LlamaIndex RAG 集成日志配置。"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """配置 LlamaIndex RAG 模块的日志。

    用法:
        from src.tools.rag.llamaindex.logging import setup_logging
        logger = setup_logging()
        logger.info("...")
    """
    logger = logging.getLogger("rag.llamaindex")

    # 避免重复添加
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    # 不向根日志传播
    logger.propagate = False

    return logger


def get_logger() -> logging.Logger:
    """获取 rag.llamaindex 日志器。"""
    return logging.getLogger("rag.llamaindex")
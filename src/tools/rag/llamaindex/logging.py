"""LlamaIndex RAG 集成日志配置。"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """配置 LlamaIndex RAG 模块日志，返回已初始化日志器。"""
    logger = logging.getLogger("rag.llamaindex")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """获取 rag.llamaindex 日志器（需先调用 setup_logging）。"""
    return logging.getLogger("rag.llamaindex")
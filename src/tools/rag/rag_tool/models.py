"""RAG 数据模型：文档、分块、扩展名映射。"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class RagChunk:
    """检索单元：文档切分后的一个文本块。"""
    chunk_id: str
    doc_id: str
    doc_name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RagDocument:
    """知识库中的一份文档。"""
    doc_id: str
    name: str
    source_path: Optional[str] = None
    doc_type: str = "text"
    chunk_count: int = 0
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


EXT_DOC_TYPE = {
    ".txt": "text", ".log": "text",
    ".md": "markdown", ".markdown": "markdown",
    ".pdf": "pdf",
    ".docx": "word", ".doc": "word",
    ".xlsx": "excel", ".xls": "excel", ".csv": "data",
    ".pptx": "ppt", ".ppt": "ppt",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".bmp": "image", ".webp": "image",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".aac": "audio", ".m4a": "audio",
    ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
    ".py": "code", ".js": "code", ".ts": "code", ".java": "code", ".go": "code",
    ".c": "code", ".cpp": "code", ".rs": "code", ".sql": "code",
    ".json": "data", ".xml": "data", ".yaml": "data", ".yml": "data",
}

TEXT_EXTENSIONS = {".txt", ".log", ".md", ".markdown", ".py", ".js", ".ts", ".java",
                   ".go", ".c", ".cpp", ".rs", ".sql", ".json", ".xml", ".yaml", ".yml", ".csv"}

METADATA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
                       ".mp3", ".wav", ".flac", ".aac", ".m4a",
                       ".mp4", ".avi", ".mov", ".mkv"}
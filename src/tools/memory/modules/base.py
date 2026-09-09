"""MemoryEntry 数据类 + MemoryModule 抽象基类 + 全局常量。"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

DEFAULT_IMPORTANCE = {
    "working": 0.6,
    "episodic": 0.8,
    "semantic": 0.9,
    "perceptual": 0.7,
}

FILE_MODALITY_MAP = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".bmp": "image", ".webp": "image",
    ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".aac": "audio",
    ".py": "code", ".js": "code", ".java": "code", ".cpp": "code",
    ".go": "code", ".rs": "code",
    ".txt": "text", ".md": "text",
    ".pdf": "document", ".doc": "document", ".docx": "document",
    ".csv": "data", ".json": "data", ".xml": "data",
}


@dataclass
class MemoryEntry:
    """单条记忆的完整数据结构。"""
    memory_id: str
    content: str
    memory_type: str                          # working/episodic/semantic/perceptual
    importance: float                         # 0.0-1.0
    timestamp: str                            # ISO 格式时间戳
    session_id: str                           # 会话归属
    file_path: Optional[str] = None           # 关联文件路径
    modality: Optional[str] = None            # 模态: text/image/audio/video/code/document/data
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryModule:
    """记忆模块抽象基类，定义统一接口契约。

    所有具体记忆模块（WorkingMemory/EpisodicMemory/SemanticMemory/PerceptualMemory）
    必须继承此类并实现所有抽象方法。
    """

    def __init__(self, memory_type: str, max_capacity: int = None):
        self.memory_type = memory_type
        self.max_capacity = max_capacity

    def add(self, entry: MemoryEntry) -> str:
        """添加一条记忆，返回 memory_id。"""
        raise NotImplementedError()

    def get_all(self) -> List[MemoryEntry]:
        """获取所有记忆条目。"""
        raise NotImplementedError()

    def get_count(self) -> int:
        """获取当前条目总数。"""
        raise NotImplementedError()

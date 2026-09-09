"""向后兼容数据模型（旧版 Episode / Entity / Relation）。"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any


@dataclass
class Episode:
    """情景记忆条目（旧版兼容，建议使用 MemoryEntry）。"""
    episode_id: str
    session_id: str
    timestamp: str
    content: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Entity:
    """知识图谱实体（旧版兼容，建议使用 SemanticMemory 内部 Entity）。"""
    entity_id: str
    name: str
    entity_type: str
    memory_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Relation:
    """知识图谱关系（旧版兼容，建议使用 SemanticMemory 内部 Relation）。"""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str
    memory_id: str
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
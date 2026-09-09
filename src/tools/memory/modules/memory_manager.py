"""
MemoryManager：记忆管理器
===========================

统一管理多个记忆模块，根据配置启用不同类型，提供统一的 add / retrieve / get_count 等操作接口。
"""

from typing import List, Dict, Any, Optional

from .base import MemoryEntry, MemoryModule
from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .perceptual_memory import PerceptualMemory


# 记忆类型 → 模块类的映射
MODULE_CLASSES = {
    "working": WorkingMemory,
    "episodic": EpisodicMemory,
    "semantic": SemanticMemory,
    "perceptual": PerceptualMemory,
}


class MemoryManager:
    """
    记忆管理器：作为 MemoryTool 的底层引擎，
    根据配置启用不同类型的记忆模块，统一调度 add / get / remove 等操作。
    """

    DEFAULT_CONFIG = {
        "enabled_types": ["working", "episodic", "semantic", "perceptual"],
        "working_capacity": 50,
        "working_memory_ttl": 60,
        "episodic_db_path": ":memory:",
        "semantic_db_path": None,   # 默认纯内存，传文件路径启用持久化
        "perceptual_db_path": None, # 默认纯内存，传文件路径启用持久化
    }

    def __init__(self, config: Dict[str, Any] = None):
        config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._config = config
        self._modules: Dict[str, MemoryModule] = {}

        enabled = config["enabled_types"]
        for mem_type in enabled:
            if mem_type not in MODULE_CLASSES:
                continue
            if mem_type == "working":
                self._modules[mem_type] = WorkingMemory(
                    max_capacity=config.get("working_capacity", 50),
                    max_age_minutes=config.get("working_memory_ttl", 60),
                )
            elif mem_type == "episodic":
                self._modules[mem_type] = EpisodicMemory(
                    db_path=config.get("episodic_db_path", ":memory:"),
                )
            elif mem_type == "semantic":
                self._modules[mem_type] = SemanticMemory(
                    db_path=config.get("semantic_db_path"),
                )
            elif mem_type == "perceptual":
                self._modules[mem_type] = PerceptualMemory(
                    db_path=config.get("perceptual_db_path"),
                )
            else:
                self._modules[mem_type] = MODULE_CLASSES[mem_type]()

        # 启动重建：从情景记忆恢复语义记忆和感知记忆
        self._rebuild_from_episodic()

    def _rebuild_from_episodic(self):
        """从 EpisodicMemory 启动重建，恢复语义记忆和感知记忆。

        重建条件：
          - 语义记忆：importance >= 0.8（与深层固化阈值一致）
          - 感知记忆：modality 不为 text/None/""（有非文本模态标记）
        """
        episodic = self._modules.get("episodic")
        if episodic is None:
            return

        semantic = self._modules.get("semantic")
        perceptual = self._modules.get("perceptual")

        all_entries = episodic.get_all()
        if not all_entries:
            return

        # 语义记忆重建：高重要性条目
        if semantic is not None:
            semantic_entries = [m for m in all_entries
                               if m.importance >= 0.8]
            if semantic_entries:
                semantic.set_memories(semantic_entries)

        # 感知记忆重建：非文本模态条目
        if perceptual is not None:
            perceptual_entries = [m for m in all_entries
                                 if m.modality and m.modality != "text"]
            if perceptual_entries:
                perceptual.set_memories(perceptual_entries)

    # --- 模块查询 ---

    def get_enabled_types(self) -> List[str]:
        return list(self._modules.keys())

    def is_enabled(self, memory_type: str) -> bool:
        return memory_type in self._modules

    def get_module(self, memory_type: str) -> Optional[MemoryModule]:
        return self._modules.get(memory_type)

    # --- 统一操作 ---

    def add(self, entry: MemoryEntry) -> str:
        """将记忆添加到对应类型的模块，返回模块的附加消息（如容量淘汰提示）。"""
        module = self._modules.get(entry.memory_type)
        if module is None:
            return (f"❌ 记忆类型 '{entry.memory_type}' 未启用，"
                    f"当前已启用: {list(self._modules.keys())}")
        return module.add(entry)

    def get_all(self) -> List[MemoryEntry]:
        """聚合所有模块的记忆，返回扁平列表。"""
        all_memories = []
        for module in self._modules.values():
            all_memories.extend(module.get_all())
        return all_memories

    def get_count(self) -> int:
        return sum(m.get_count() for m in self._modules.values())

    def get_by_type(self, memory_type: str) -> List[MemoryEntry]:
        module = self._modules.get(memory_type)
        return module.get_all() if module else []

    def replace_all(self, memories: List[MemoryEntry]):
        """清空所有模块，按类型重新分发记忆（用于 forget 操作后回写）。

        使用 set_memories() 而非直接操作 _memories，
        确保各模块（如 EpisodicMemory）能同步持久层和索引。
        """
        by_type: Dict[str, List[MemoryEntry]] = {}
        for mem in memories:
            by_type.setdefault(mem.memory_type, []).append(mem)

        for mem_type, module in self._modules.items():
            module.set_memories(by_type.get(mem_type, []))

    def remove_ids(self, ids: set) -> List[MemoryEntry]:
        """按 ID 集合删除记忆，返回被删除的列表。

        使用 set_memories() 回写，确保持久层同步。
        """
        removed = []
        for module in self._modules.values():
            kept = []
            for mem in module.get_all():
                if mem.memory_id in ids:
                    removed.append(mem)
                else:
                    kept.append(mem)
            module.set_memories(kept)
        return removed

    def get_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def close(self):
        """关闭所有模块的资源连接（如数据库）。"""
        for module in self._modules.values():
            if hasattr(module, 'close') and callable(module.close):
                module.close()

"""
SemanticMemory：语义记忆模块
============================

知识图谱 + 嵌入向量/TF-IDF检索 + 混合排序。
"""

import uuid
import re
import json
import math
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from .base import MemoryEntry, MemoryModule
from .embedding_client import EmbeddingClient
from .working_memory import WorkingMemory


@dataclass
class Entity:
    """知识图谱实体节点。"""
    entity_id: str
    name: str
    entity_type: str
    memory_ids: List[str]


@dataclass
class Relation:
    """知识图谱关系边。"""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str
    memory_id: str
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relation_type": self.relation_type,
            "memory_id": self.memory_id,
            "weight": self.weight,
        }


class SemanticMemory(MemoryModule):
    """语义记忆模块 —— 知识图谱 + 嵌入向量/TF-IDF检索 + 混合排序。

    特点：
      - 内存知识图谱：实体（Entity）+ 关系（Relation），支持图检索
      - SQLite 持久化选项：entities 和 relations 表，启动时自动加载
      - 嵌入向量检索（百炼 text-embedding-v3，API 不可用时自动回退 TF-IDF）
      - 轻量级实体/关系抽取（正则 + 关键词匹配，无外部 NLP 依赖）
      - 混合检索：向量检索（0.7权重） + 图检索（0.3权重）
      - 评分公式：base_relevance = vector_score × 0.7 + graph_score × 0.3
                        importance_weight = 0.8 + importance × 0.4
                        combined_score = base_relevance × importance_weight
    """

    _TECH_KEYWORDS = {
        "python": "language", "java": "language", "javascript": "language",
        "golang": "language", "go": "language", "rust": "language",
        "c++": "language", "c#": "language", "ruby": "language",
        "ai": "concept", "人工智能": "concept", "ml": "concept",
        "机器学习": "concept", "深度学习": "concept", "nlp": "concept",
        "自然语言处理": "concept", "llm": "concept", "大模型": "concept",
        "tensorflow": "tool", "pytorch": "tool", "numpy": "tool",
        "pandas": "tool", "scikit-learn": "tool", "docker": "tool",
        "kubernetes": "tool", "git": "tool",
        "react": "technology", "vue": "technology", "angular": "technology",
        "flask": "technology", "django": "technology", "fastapi": "technology",
        "redis": "technology", "mysql": "technology", "postgresql": "technology",
        "mongodb": "technology", "elasticsearch": "technology",
    }

    def __init__(self, db_path: str = None):
        super().__init__("semantic", None)
        self.entities: Dict[str, Entity] = {}       # name -> Entity
        self.relations: List[Relation] = []
        self._entity_index: Dict[str, set] = {}     # memory_id -> {entity_name, ...}
        self._embedding_client = EmbeddingClient()   # 百炼嵌入客户端
        self._embeddings: Dict[str, List[float]] = {}  # memory_id -> embedding vector
        self._memories: List[MemoryEntry] = []
        self._db_path = db_path

        if db_path:
            self._init_db()
            self._load_from_db()

    # ============================================================
    # SQLite 持久化
    # ============================================================

    def _init_db(self):
        """初始化 SQLite 数据库表（entities + relations + embeddings）。"""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_entities (
                entity_id   TEXT PRIMARY KEY,
                name        TEXT UNIQUE,
                entity_type TEXT,
                memory_ids  TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_relations (
                relation_id   TEXT PRIMARY KEY,
                source_entity TEXT,
                target_entity TEXT,
                relation_type TEXT,
                memory_id     TEXT,
                weight        REAL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_embeddings (
                memory_id TEXT PRIMARY KEY,
                embedding TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memories (
                memory_id   TEXT PRIMARY KEY,
                content     TEXT,
                memory_type TEXT,
                importance  REAL,
                timestamp   TEXT,
                session_id  TEXT,
                file_path   TEXT,
                modality    TEXT,
                metadata    TEXT
            )
        """)
        self._conn.commit()

    def _load_from_db(self):
        """从 SQLite 加载知识图谱和记忆到内存。"""
        # 加载实体
        cursor = self._conn.execute("SELECT * FROM semantic_entities")
        for row in cursor:
            ent = Entity(
                entity_id=row[0],
                name=row[1],
                entity_type=row[2],
                memory_ids=json.loads(row[3]) if row[3] else [],
            )
            self.entities[ent.name] = ent

        # 加载关系
        cursor = self._conn.execute("SELECT * FROM semantic_relations")
        for row in cursor:
            rel = Relation(
                relation_id=row[0],
                source_entity=row[1],
                target_entity=row[2],
                relation_type=row[3],
                memory_id=row[4],
                weight=row[5],
            )
            self.relations.append(rel)

        # 加载嵌入向量
        cursor = self._conn.execute("SELECT * FROM semantic_embeddings")
        for row in cursor:
            try:
                self._embeddings[row[0]] = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                pass

        # 加载记忆
        cursor = self._conn.execute("SELECT * FROM semantic_memories")
        for row in cursor:
            entry = MemoryEntry(
                memory_id=row[0],
                content=row[1],
                memory_type=row[2] or "semantic",
                importance=row[3] if row[3] is not None else 0.8,
                timestamp=row[4],
                session_id=row[5] or "default",
                file_path=row[6],
                modality=row[7],
                metadata=json.loads(row[8]) if row[8] else {},
            )
            self._memories.append(entry)

            # 重建 entity_index
            ent = self.entities.get(entry.content.split(":")[0].strip()
                                    if ":" in entry.content else "")
            if ent:
                self._entity_index.setdefault(entry.memory_id, set()).add(ent.name)

        self._conn.commit()

    def _persist_entity(self, entity: Entity):
        """持久化单条实体。"""
        self._conn.execute(
            "INSERT OR REPLACE INTO semantic_entities "
            "VALUES (?, ?, ?, ?)",
            (entity.entity_id, entity.name, entity.entity_type,
             json.dumps(entity.memory_ids, ensure_ascii=False))
        )
        self._conn.commit()

    def _persist_relation(self, relation: Relation):
        """持久化单条关系。"""
        self._conn.execute(
            "INSERT OR REPLACE INTO semantic_relations "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (relation.relation_id, relation.source_entity,
             relation.target_entity, relation.relation_type,
             relation.memory_id, relation.weight)
        )
        self._conn.commit()

    def _persist_embedding(self, memory_id: str, embedding: List[float]):
        """持久化嵌入向量。"""
        self._conn.execute(
            "INSERT OR REPLACE INTO semantic_embeddings VALUES (?, ?)",
            (memory_id, json.dumps(embedding))
        )

    def _persist_memory_entry(self, entry: MemoryEntry):
        """持久化记忆条目。"""
        self._conn.execute(
            "INSERT OR REPLACE INTO semantic_memories "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entry.memory_id, entry.content, entry.memory_type,
             entry.importance, entry.timestamp, entry.session_id,
             entry.file_path, entry.modality,
             json.dumps(entry.metadata, ensure_ascii=False))
        )

    def _flush_db(self):
        """清空所有表并按当前状态重新写入。"""
        for table in ["semantic_entities", "semantic_relations",
                       "semantic_embeddings", "semantic_memories"]:
            self._conn.execute(f"DELETE FROM {table}")

        for ent in self.entities.values():
            self._persist_entity(ent)
        for rel in self.relations:
            self._persist_relation(rel)
        for mid, emb in self._embeddings.items():
            self._persist_embedding(mid, emb)
        for mem in self._memories:
            self._persist_memory_entry(mem)
        self._conn.commit()

    # ============================================================
    # 增删操作（覆写基类）
    # ============================================================

    def add(self, entry: MemoryEntry) -> str:
        """添加语义记忆：存储 + 嵌入向量 + 抽取实体/关系 + 构建知识图谱 + 持久化。"""
        self._memories.append(entry)

        embedding = self._embedding_client.embed(entry.content)
        if embedding:
            self._embeddings[entry.memory_id] = embedding

        entity_names = self._extract_entities(entry.content)
        relations = self._extract_relations(entry.content, entity_names)

        matched_names = set()
        for name in entity_names:
            self._register_entity(name, entry.memory_id)
            matched_names.add(name)

        self._entity_index[entry.memory_id] = matched_names

        for rel in relations:
            self.relations.append(rel)

        entity_list = list(matched_names)
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                rel = Relation(
                    relation_id=str(uuid.uuid4()),
                    source_entity=entity_list[i],
                    target_entity=entity_list[j],
                    relation_type="co_occurs",
                    memory_id=entry.memory_id,
                    weight=0.5,
                )
                self.relations.append(rel)

        # 持久化
        if self._db_path:
            self._persist_memory_entry(entry)
            if embedding:
                self._persist_embedding(entry.memory_id, embedding)

        return ""

    def clear(self):
        """清空所有记忆和知识图谱。"""
        self._memories.clear()
        self.entities.clear()
        self._embeddings.clear()
        self.relations.clear()
        self._entity_index.clear()
        if self._db_path:
            self._flush_db()

    def set_memories(self, memories: List[MemoryEntry]):
        """直接替换记忆列表，重建知识图谱和嵌入向量。"""
        self._memories = list(memories)
        self.entities.clear()
        self.relations.clear()
        self._entity_index.clear()
        self._embeddings.clear()

        if memories and self._embedding_client.is_available:
            texts = [m.content for m in memories]
            embeddings = self._embedding_client.embed_batch(texts)
            for mem, emb in zip(memories, embeddings):
                if emb:
                    self._embeddings[mem.memory_id] = emb

        for entry in memories:
            entity_names = self._extract_entities(entry.content)
            relations = self._extract_relations(entry.content, entity_names)
            matched = set()
            for name in entity_names:
                self._register_entity(name, entry.memory_id)
                matched.add(name)
            self._entity_index[entry.memory_id] = matched
            self.relations.extend(relations)
            entity_list = list(matched)
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    self.relations.append(Relation(
                        relation_id=str(uuid.uuid4()),
                        source_entity=entity_list[i],
                        target_entity=entity_list[j],
                        relation_type="co_occurs",
                        memory_id=entry.memory_id,
                        weight=0.5,
                    ))

        if self._db_path:
            self._flush_db()

    def close(self):
        """关闭数据库连接。"""
        if self._db_path and hasattr(self, '_conn'):
            self._conn.close()

    def get_all(self) -> List[MemoryEntry]:
        return list(self._memories)

    def get_count(self) -> int:
        return len(self._memories)

    # ============================================================
    # 实体与关系抽取
    # ============================================================

    def _extract_entities(self, content: str) -> List[str]:
        """轻量级实体抽取：技术词汇 + 中文专有名词 + 英文专有名词。

        策略：
          1. 匹配已知技术关键词表
          2. 匹配英文大写开头的词（专有名词）
          3. 匹配中文特定模式（X是Y、X使用Y等结构中的X/Y）
          4. 匹配引号内的内容
        """
        entities = []
        content_lower = content.lower()

        for keyword, etype in self._TECH_KEYWORDS.items():
            if keyword in content_lower:
                entities.append(keyword)

        proper_nouns = re.findall(r'\b([A-Z][a-zA-Z]{2,})\b', content)
        for noun in proper_nouns:
            if noun.lower() not in [e.lower() for e in entities]:
                entities.append(noun)

        cn_patterns = [
            r'([\u4e00-\u9fff]{2,8})是(?:一种|一个|一类)?',
            r'([\u4e00-\u9fff]{2,8})使用',
            r'([\u4e00-\u9fff]{2,8})包含',
            r'([\u4e00-\u9fff]{2,8})属于',
            r'([\u4e00-\u9fff]{2,8})支持',
            r'([\u4e00-\u9fff]{2,8})提供了',
        ]
        for pattern in cn_patterns:
            matches = re.findall(pattern, content)
            for m in matches:
                if m not in entities and len(m) >= 2:
                    entities.append(m)

        is_target = re.findall(r'是(?:一种|一个|一类)?([\u4e00-\u9fff]{2,8})', content)
        for m in is_target:
            if m not in entities:
                entities.append(m)

        quoted = re.findall(r'[「"\'](.+?)[」"\']', content)
        for q in quoted:
            if q not in entities and len(q) >= 2:
                entities.append(q)

        return entities

    def _extract_relations(self, content: str, entity_names: List[str]) -> List[Relation]:
        """轻量级关系抽取：基于模式匹配提取三元组。

        支持的关系模式：
          - is_a:    X是Y / X是一种Y
          - uses:    X使用Y / X利用Y
          - part_of: X包含Y / X是Y的一部分
          - related_to: X和Y / X与Y
        """
        relations = []
        if len(entity_names) < 2:
            return relations

        entity_set = set(e.lower() for e in entity_names)
        content_lower = content.lower()

        for src in entity_names:
            pattern = re.escape(src) + r'是(?:一种|一个|一类)?(.{2,20})'
            match = re.search(pattern, content)
            if match:
                target_text = match.group(1).strip()
                for tgt in entity_names:
                    if tgt != src and (tgt in target_text or target_text.startswith(tgt)):
                        relations.append(Relation(
                            relation_id=str(uuid.uuid4()),
                            source_entity=src,
                            target_entity=tgt,
                            relation_type="is_a",
                            memory_id="",
                            weight=1.0,
                        ))

        for src in entity_names:
            for verb in ['使用', '利用', '用']:
                pattern = re.escape(src) + verb + r'(.{2,20})'
                match = re.search(pattern, content)
                if match:
                    target_text = match.group(1).strip()
                    for tgt in entity_names:
                        if tgt != src and (tgt in target_text or target_text.startswith(tgt)):
                            relations.append(Relation(
                                relation_id=str(uuid.uuid4()),
                                source_entity=src,
                                target_entity=tgt,
                                relation_type="uses",
                                memory_id="",
                                weight=1.0,
                            ))

        for i, src in enumerate(entity_names):
            for j, tgt in enumerate(entity_names):
                if i >= j:
                    continue
                pattern = re.escape(src) + r'[与和].{0,5}' + re.escape(tgt)
                if re.search(pattern, content):
                    relations.append(Relation(
                        relation_id=str(uuid.uuid4()),
                        source_entity=src,
                        target_entity=tgt,
                        relation_type="related_to",
                        memory_id="",
                        weight=0.8,
                    ))

        return relations

    def _register_entity(self, name: str, memory_id: str):
        """注册实体到知识图谱。"""
        key = name.lower()
        if key not in self.entities:
            etype = self._infer_entity_type(name)
            self.entities[key] = Entity(
                entity_id=str(uuid.uuid4()),
                name=name,
                entity_type=etype,
                memory_ids=[],
            )
        if memory_id not in self.entities[key].memory_ids:
            self.entities[key].memory_ids.append(memory_id)

    @classmethod
    def _infer_entity_type(cls, name: str) -> str:
        """推断实体类型。"""
        name_lower = name.lower()
        if name_lower in cls._TECH_KEYWORDS:
            return cls._TECH_KEYWORDS[name_lower]
        if re.match(r'^[A-Z][a-zA-Z]+$', name):
            return "person"
        return "concept"

    # ============================================================
    # 混合检索
    # ============================================================

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[Tuple[float, float, MemoryEntry, Dict]]:
        """混合检索：嵌入向量/TF-IDF向量检索 + 知识图谱检索。

        评分算法：
          vector_score  = 嵌入向量余弦相似度（API可用时）/ TF-IDF余弦相似度（回退）
          graph_score   = 实体重叠度 + 关系连接度
          base_relevance = vector_score × 0.7 + graph_score × 0.3
          importance_weight = 0.8 + importance × 0.4
          combined_score = base_relevance × importance_weight

        Returns:
            [(combined_score, base_relevance, memory, details_dict), ...]
        """
        if not self._memories:
            return []

        min_importance = kwargs.get("min_importance", 0.0)

        vector_scores, vector_method = self._vector_search(query)
        graph_scores, matched_entities = self._graph_search(query)

        scored = []
        for mem in self._memories:
            if mem.importance < min_importance:
                continue

            vec_score = vector_scores.get(mem.memory_id, 0.0)
            graph_score = graph_scores.get(mem.memory_id, 0.0)

            if vec_score == 0 and graph_score == 0:
                continue

            base_relevance = vec_score * 0.7 + graph_score * 0.3
            importance_weight = 0.8 + (mem.importance * 0.4)
            combined_score = base_relevance * importance_weight

            if combined_score > 0:
                if vec_score > 0 and graph_score > 0:
                    method = "hybrid"
                elif vec_score > 0:
                    method = "vector"
                else:
                    method = "graph"

                mem_entities = self._entity_index.get(mem.memory_id, set())
                mem_matched = [e for e in matched_entities if e in mem_entities]

                details = {
                    "method": method,
                    "vector_method": vector_method,
                    "vector_score": vec_score,
                    "graph_score": graph_score,
                    "base_relevance": base_relevance,
                    "importance_weight": importance_weight,
                    "combined_score": combined_score,
                    "matched_entities": mem_matched,
                    "entity_count": len(mem_entities),
                }
                scored.append((combined_score, base_relevance, mem, details))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    # ============================================================
    # 向量检索（嵌入优先 + TF-IDF 回退）
    # ============================================================

    def _vector_search(self, query: str) -> Tuple[Dict[str, float], str]:
        """向量语义检索：嵌入向量优先，TF-IDF 回退。

        Returns:
            ({memory_id: similarity}, vector_method)
            vector_method: "embedding" 或 "tfidf"
        """
        if self._embedding_client.is_available and self._embeddings:
            query_embedding = self._embedding_client.embed(query)
            if query_embedding:
                scores = {}
                for mem in self._memories:
                    mem_embedding = self._embeddings.get(mem.memory_id)
                    if mem_embedding:
                        similarity = EmbeddingClient.cosine_similarity_dense(
                            query_embedding, mem_embedding
                        )
                        if similarity > 0:
                            scores[mem.memory_id] = similarity
                if scores:
                    return scores, "embedding"

        tfidf_scores = self._tfidf_search(query)
        return tfidf_scores, "tfidf"

    def _tfidf_search(self, query: str) -> Dict[str, float]:
        """TF-IDF 向量语义检索（回退方案），返回 {memory_id: cosine_similarity}。"""
        try:
            if len(self._memories) < 1:
                return {}

            from collections import Counter
            import math

            query_tokens = WorkingMemory._tokenize_for_tfidf(query)
            if not query_tokens:
                return {}

            documents = [WorkingMemory._tokenize_for_tfidf(mem.content) for mem in self._memories]

            df = Counter()
            for doc_tokens in documents:
                for token in set(doc_tokens):
                    df[token] += 1

            total_docs = len(documents)
            idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1
                   for token, freq in df.items()}

            query_tf = Counter(query_tokens)
            query_len = len(query_tokens)
            query_vector = {token: (count / query_len) * idf.get(token, 0.0)
                            for token, count in query_tf.items()}

            if all(v == 0 for v in query_vector.values()):
                return {}

            scores = {}
            for mem, doc_tokens in zip(self._memories, documents):
                doc_len = len(doc_tokens) if doc_tokens else 1
                doc_tf = Counter(doc_tokens)
                doc_vector = {token: (count / doc_len) * idf.get(token, 0.0)
                              for token, count in doc_tf.items()}

                similarity = WorkingMemory._cosine_similarity(query_vector, doc_vector)
                if similarity > 0:
                    scores[mem.memory_id] = similarity

            return scores
        except Exception:
            return {}

    # ============================================================
    # 知识图谱检索
    # ============================================================

    def _graph_search(self, query: str) -> Tuple[Dict[str, float], List[str]]:
        """图检索：从查询中抽取实体，通过实体重叠和关系连接度评分。

        Returns:
            ({memory_id: graph_score}, [matched_entity_names])
        """
        query_entities = self._extract_entities(query)
        query_entity_keys = set(e.lower() for e in query_entities)

        if not query_entity_keys:
            return {}, []

        matched_entities = []
        for name in query_entities:
            if name.lower() in self.entities:
                matched_entities.append(name)

        scores: Dict[str, float] = {}
        for mem in self._memories:
            mem_entities = self._entity_index.get(mem.memory_id, set())
            if not mem_entities:
                continue

            mem_entity_keys = set(e.lower() for e in mem_entities)
            overlap = query_entity_keys & mem_entity_keys
            union = query_entity_keys | mem_entity_keys
            entity_overlap_score = len(overlap) / len(union) if union else 0.0

            relation_score = 0.0
            for rel in self.relations:
                if rel.memory_id == mem.memory_id:
                    src_in_query = rel.source_entity.lower() in query_entity_keys
                    tgt_in_mem = rel.target_entity.lower() in mem_entity_keys
                    tgt_in_query = rel.target_entity.lower() in query_entity_keys
                    src_in_mem = rel.source_entity.lower() in mem_entity_keys

                    if (src_in_query and tgt_in_mem) or (tgt_in_query and src_in_mem):
                        relation_score += rel.weight * 0.3
                else:
                    src_in_query = rel.source_entity.lower() in query_entity_keys
                    tgt_in_mem = rel.target_entity.lower() in mem_entity_keys
                    tgt_in_query = rel.target_entity.lower() in query_entity_keys
                    src_in_mem = rel.source_entity.lower() in mem_entity_keys

                    if (src_in_query and tgt_in_mem) or (tgt_in_query and src_in_mem):
                        relation_score += rel.weight * 0.15

            graph_score = entity_overlap_score * 0.6 + relation_score * 0.4
            if graph_score > 0:
                scores[mem.memory_id] = min(1.0, graph_score)

        return scores, matched_entities

    # ============================================================
    # 知识图谱查询辅助
    # ============================================================

    def get_entity_relations(self, entity_name: str) -> List[Dict[str, Any]]:
        """获取指定实体的所有关系。"""
        key = entity_name.lower()
        result = []
        for rel in self.relations:
            if rel.source_entity.lower() == key or rel.target_entity.lower() == key:
                result.append(rel.to_dict())
        return result

    def get_entity_count(self) -> int:
        return len(self.entities)

    def get_relation_count(self) -> int:
        return len(self.relations)

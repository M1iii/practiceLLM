"""
MemoryEngine：Agent 框架层的自动记忆引擎。

封装 MemoryTool，为 BaseAgent 提供两件事：
  1. retrieve(query, top_k, threshold) → 格式化记忆文本，供注入上下文
  2. store(question, answer) → 自动将本轮对话存入情景记忆

不依赖 MemoryTool 之外的任何工具，MemoryTool 内部异常时静默降级。
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Agent 自动记忆引擎。

    用法：
        engine = MemoryEngine()
        context = engine.retrieve("什么是 RAG？", top_k=3)
        engine.store("用户问了RAG", "回答了RAG的定义")
    """

    # 记忆上下文前缀模板
    MEMORY_TEMPLATE = """【相关记忆】
{memories}

【注意事项】
- 以上是历史对话中积累的相关记忆，供参考
- 如果记忆与当前问题无关，请忽略"""

    # 技术关键词（用于技术密度评分）
    TECH_KEYWORDS = [
        "RAG", "LLM", "API", "SQL", "JSON", "YAML", "XML", "HTTP",
        "向量", "嵌入", "索引", "分块", "检索", "生成", "模型", "训练",
        "推理", "部署", "缓存", "并发", "异步", "同步", "线程", "进程",
        "数据库", "存储", "查询", "过滤", "排序", "聚合", "事务", "锁",
        "Python", "Docker", "Git", "算法", "数据结构", "协议", "加密",
        "认证", "授权", "中间件", "微服务", "容器", "编排", "监控",
        "日志", "测试", "调试", "优化", "重构", "设计模式", "架构",
    ]

    # 重要性自适应权重
    IMPORTANCE_WEIGHTS = {
        "q_len": 0.25,      # 问题长度因子
        "a_len": 0.25,      # 回答长度因子
        "tech_density": 0.30,  # 技术密度因子
        "depth": 0.20,      # 对话深度因子
    }

    # 整合频率（store 次数）
    CONSOLIDATE_EVERY = 5       # 每 N 次 store 触发 working→episodic
    DEEP_CONSOLIDATE_EVERY = 20  # 每 M 次 store 触发 episodic→semantic
    FORGET_EVERY = 10            # 每 K 次 store 触发遗忘

    def __init__(self, enable: bool = True, top_k: int = 3,
                 similarity_threshold: float = 0.5,
                 memory_type: str = "episodic"):
        """
        :param enable: 总开关
        :param top_k: 每轮检索条数
        :param similarity_threshold: 余弦相似度阈值（低于此值丢弃）
        :param memory_type: 存储时使用的记忆类型
        """
        self._enable = enable
        self._top_k = top_k
        self._threshold = similarity_threshold
        self._memory_type = memory_type
        self._tool = None  # 惰性初始化
        # 记忆维护计数器
        self._store_count = 0

    def _lazy_init(self):
        """惰性初始化 MemoryTool，首次使用才创建。"""
        if self._tool is not None:
            return
        try:
            from src.tools.memory.memory_tool import MemoryTool
            self._tool = MemoryTool()
        except Exception as e:
            logger.warning("MemoryEngine: MemoryTool 初始化失败: %s", e)
            self._enable = False

    @property
    def enabled(self) -> bool:
        return self._enable

    def _calculate_importance(self, question: str, answer: str,
                               depth: int = 0) -> float:
        """根据对话特征动态计算重要性（0.5 ~ 1.0）。

        因子：
          - q_len: 问题长度（越长越重要）
          - a_len: 回答长度（越长越重要）
          - tech_density: 技术关键词密度
          - depth: 对话深度（连续追问加分）
        """
        scores = {}

        # 1. 问题长度因子（0~0.2）
        q_len = len(question)
        if q_len >= 30:
            scores["q_len"] = 0.2
        elif q_len >= 10:
            scores["q_len"] = 0.1
        else:
            scores["q_len"] = 0.0

        # 2. 回答长度因子（0~0.2）
        a_len = len(answer)
        if a_len >= 200:
            scores["a_len"] = 0.2
        elif a_len >= 50:
            scores["a_len"] = 0.1
        else:
            scores["a_len"] = 0.0

        # 3. 技术密度因子（0~0.3）
        combined = question + " " + answer
        tech_count = sum(
            1 for kw in self.TECH_KEYWORDS
            if re.search(re.escape(kw), combined, re.IGNORECASE)
        )
        if tech_count >= 5:
            scores["tech_density"] = 0.3
        elif tech_count >= 3:
            scores["tech_density"] = 0.2
        elif tech_count >= 1:
            scores["tech_density"] = 0.1
        else:
            scores["tech_density"] = 0.0

        # 4. 对话深度因子（0~0.2）
        if depth >= 3:
            scores["depth"] = 0.2
        elif depth >= 1:
            scores["depth"] = 0.1
        else:
            scores["depth"] = 0.0

        # 加权求和，映射到 0.5~1.0 区间
        raw = sum(
            scores[k] * self.IMPORTANCE_WEIGHTS[k]
            for k in self.IMPORTANCE_WEIGHTS
        )
        # 归一化（raw 范围 0~0.25，映射到 0.5~1.0）
        normalized = 0.5 + min(raw * 2.0, 0.5)
        return round(min(1.0, max(0.5, normalized)), 2)

    def _auto_consolidate(self, question: str = "", answer: str = ""):
        """自动整合管道：根据 store 次数触发记忆整合。

        每 CONSOLIDATE_EVERY 次 → working→episodic（重要度≥0.6）
        每 DEEP_CONSOLIDATE_EVERY 次 → episodic→semantic（重要度≥0.8）
        深层整合后自动触发知识抽取。
        """
        if self._tool is None:
            return

        count = self._store_count

        # 浅层整合：working → episodic
        if count > 0 and count % self.CONSOLIDATE_EVERY == 0:
            try:
                result = self._tool.run({
                    "action": "consolidate",
                    "from_type": "working",
                    "to_type": "episodic",
                    "importance_threshold": 0.6,
                })
                logger.info("MemoryEngine: 自动整合 working→episodic (%s)", result[:60])
            except Exception as e:
                logger.warning("MemoryEngine: 浅层整合失败: %s", e)

        # 深层整合：episodic → semantic + 知识抽取
        if count > 0 and count % self.DEEP_CONSOLIDATE_EVERY == 0:
            try:
                result = self._tool.run({
                    "action": "consolidate",
                    "from_type": "episodic",
                    "to_type": "semantic",
                    "importance_threshold": 0.8,
                })
                logger.info("MemoryEngine: 自动整合 episodic→semantic (%s)", result[:60])
            except Exception as e:
                logger.warning("MemoryEngine: 深层整合失败: %s", e)

            # 感知记忆固化：有非文本模态标记的条目 → perceptual
            try:
                result = self._tool.run({
                    "action": "consolidate",
                    "from_type": "episodic",
                    "to_type": "perceptual",
                    "modality_filter": "non_text",  # 仅非文本模态
                })
                logger.info("MemoryEngine: 自动整合 episodic→perceptual (%s)", result[:60])
            except Exception as e:
                logger.warning("MemoryEngine: 感知整合失败: %s", e)

            # 知识抽取：从当前对话提取实体/关系存入语义记忆
            if question or answer:
                self._extract_knowledge(question, answer)

    def _auto_forget(self):
        """定期遗忘调度：每 FORGET_EVERY 次 store 触发遗忘策略。

        使用 mixed 策略：删除低重要度（importance < 0.3）且超时（>7天）的记忆。
        """
        if self._tool is None:
            return

        count = self._store_count
        if count > 0 and count % self.FORGET_EVERY == 0:
            try:
                result = self._tool.run({
                    "action": "forget",
                    "strategy": "mixed",
                    "threshold": 0.3,
                    "max_age_days": 7,
                })
                logger.info("MemoryEngine: 自动遗忘 (%s)", result[:80])
            except Exception as e:
                logger.warning("MemoryEngine: 自动遗忘失败: %s", e)

    def _extract_knowledge(self, question: str, answer: str):
        """从对话内容中提取实体和关系，存入语义记忆。

        实体类型：人物、技术概念、术语
        关系类型：定义关系（XX是XX）、包含关系（XX包括XX）
        """
        if self._tool is None:
            return

        combined = f"{question} {answer}"
        extracted = []

        # 1. 提取定义关系：XX是XX
        for m in re.finditer(r'([\u4e00-\u9fff\w]{2,20})是([\u4e00-\u9fff\w]{2,50})', combined):
            subj, obj = m.group(1), m.group(2)
            if len(subj) >= 2 and len(obj) >= 2:
                extracted.append(f"实体: {subj} | 关系: 定义 | 对象: {obj}")

        # 2. 提取包含关系：XX包括XX、XX包含XX
        for m in re.finditer(r'([\u4e00-\u9fff\w]{2,20})(?:包括|包含|分为)([\u4e00-\u9fff\w]{2,50})', combined):
            subj, obj = m.group(1), m.group(2)
            if len(subj) >= 2 and len(obj) >= 2:
                extracted.append(f"实体: {subj} | 关系: 包含 | 对象: {obj}")

        # 3. 提取技术关键词作为实体
        for kw in self.TECH_KEYWORDS:
            if re.search(re.escape(kw), combined, re.IGNORECASE):
                extracted.append(f"实体: {kw} | 类型: 技术概念 | 来源: 对话")

        # 4. 去重并存储
        seen = set()
        for entry in extracted:
            if entry not in seen:
                seen.add(entry)
                try:
                    self._tool.run({
                        "action": "add",
                        "content": entry,
                        "memory_type": "semantic",
                        "importance": 0.75,
                    })
                except Exception as e:
                    logger.warning("MemoryEngine: 知识抽取存储失败: %s", e)

        if extracted:
            logger.info("MemoryEngine: 知识抽取完成，提取 %d 条知识", len(seen))

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 threshold: Optional[float] = None) -> str:
        """检索相关记忆，返回格式化文本（空字符串 = 无相关记忆）。"""
        if not self._enable:
            return ""
        self._lazy_init()
        if self._tool is None:
            return ""

        k = top_k or self._top_k
        t = threshold if threshold is not None else self._threshold

        try:
            result = self._tool.run({
                "action": "search",
                "query": query,
                "limit": k,
                "min_importance": t,
            })
        except Exception as e:
            logger.warning("MemoryEngine: 检索记忆失败: %s", e)
            return ""

        # 解析结果
        if not result or result.startswith("📭"):
            return ""

        # 提取记忆内容行（去掉格式标记）
        lines = []
        for line in result.split("\n"):
            line = line.strip()
            if not line or line.startswith("📌") or line.startswith("🔍"):
                continue
            lines.append(line)

        if not lines:
            return ""

        # 最多保留 top_k 条，避免污染上下文
        memory_text = "\n".join(lines[:k])
        return self.MEMORY_TEMPLATE.format(memories=memory_text)

    def store(self, question: str, answer: str,
              memory_type: Optional[str] = None,
              depth: int = 0) -> bool:
        """将本轮对话存入记忆，返回是否成功。

        :param depth: 对话深度（连续追问次数），用于自适应重要性
        """
        if not self._enable:
            return False
        self._lazy_init()
        if self._tool is None:
            return False

        mt = memory_type or self._memory_type
        try:
            # 动态计算重要性
            q_imp = self._calculate_importance(question, answer, depth=depth)
            a_imp = min(1.0, q_imp + 0.1)  # 回答比问题略重要

            # 存储用户问题
            self._tool.run({
                "action": "add",
                "content": question,
                "memory_type": mt,
                "importance": q_imp,
            })
            # 存储助手回答
            self._tool.run({
                "action": "add",
                "content": answer,
                "memory_type": mt,
                "importance": a_imp,
            })

            self._store_count += 1
            self._auto_consolidate(question=question, answer=answer)
            self._auto_forget()
            return True
        except Exception as e:
            logger.warning("MemoryEngine: 存储记忆失败: %s", e)
            return False

    def set_top_k(self, k: int):
        """动态调整检索条数。"""
        self._top_k = max(1, min(20, k))

    def set_threshold(self, t: float):
        """动态调整相似度阈值。"""
        self._threshold = max(0.0, min(1.0, t))
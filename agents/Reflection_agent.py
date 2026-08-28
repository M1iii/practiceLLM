import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import practiceLLM
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MemoryEntry:
    """记忆条目，记录智能体每一步的行动或反思。"""
    step: int
    type: str        # "initial" | "reflect" | "refine"
    content: str


class ShortTermMemory:
    """
    短期记忆模块，存储智能体的行动与反思轨迹。
    在一次任务执行中，按时间顺序记录每一步的输入输出，便于回溯和展示。
    """

    def __init__(self):
        self._entries: List[MemoryEntry] = []
        self._step = 0

    def add(self, entry_type: str, content: str) -> int:
        """添加一条记忆，返回当前步数。"""
        self._step += 1
        self._entries.append(MemoryEntry(step=self._step, type=entry_type, content=content))
        return self._step

    def get_all(self) -> List[MemoryEntry]:
        """获取所有记忆条目。"""
        return self._entries

    def get_by_type(self, entry_type: str) -> List[MemoryEntry]:
        """按类型筛选记忆条目。"""
        return [e for e in self._entries if e.type == entry_type]

    def get_last(self, entry_type: str = None) -> Optional[MemoryEntry]:
        """获取最后一条记忆，可按类型筛选。"""
        entries = self.get_by_type(entry_type) if entry_type else self._entries
        return entries[-1] if entries else None

    def clear(self):
        """清空所有记忆。"""
        self._entries = []
        self._step = 0

    def format(self) -> str:
        """格式化所有记忆条目，用于展示完整轨迹。"""
        if not self._entries:
            return "（暂无记忆）"

        type_labels = {
            "initial": "📝 初始回答",
            "reflect": "🔍 反思",
            "refine": "✨ 改进",
            "score": "📊 质量评分",
        }

        lines = []
        for entry in self._entries:
            label = type_labels.get(entry.type, entry.type)
            lines.append(f"--- 第{entry.step}步 [{label}] ---")
            lines.append(entry.content)
            lines.append("")
        return "\n".join(lines)


class ReflectionAgent:
    """
    反思智能体：通过"初始回答 → 反思 → 改进"的循环不断优化输出。
    每轮反思会检查当前回答的质量，如果发现问题则改进，直到回答无需改进或达到最大迭代次数。
    """

    # === 提示词模板 ===

    INITIAL_PROMPT = """请根据以下要求完成任务：
任务：{task}

请提供一个完整，准确的回答。"""

    REFLECT_PROMPT = """请仔细审查以下回答，并找出可能的问题或改进空间：
# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并给出具体的改进建议。
如果回答已经很好，请回答"无需改进"。"""

    REFINE_PROMPT = """请根据反馈意见改进你的回答：
# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}
请提供一个改进后的回答。"""

    SCORE_PROMPT = """请对以下回答的质量进行评分（0-100分）：

# 原始任务:
{task}

# 当前回答:
{content}

请从以下维度综合评分：
- 准确性：内容是否正确无误
- 完整性：是否完整回答了任务要求
- 清晰度：表达是否清晰易懂
- 实用性：是否具有实际参考价值

请仅输出一个0到100的整数分数，不要输出任何其他内容。"""

    def __init__(
        self,
        name: str = "ReflectionAgent",
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
        max_iterations: int = 3,
        score_threshold: int = 85,
    ):
        """
        初始化反思智能体。
        :param name: Agent 名称，用于输出展示
        :param max_iterations: 最大反思-改进轮次
        :param score_threshold: 质量评分阈值，达到或超过则提前终止（0-100）
        """
        self.name = name
        self.llm = practiceLLM(model=model, apiKey=apiKey, baseUrl=baseUrl, timeout=timeout)
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.memory = ShortTermMemory()

    def _build_prompt(self, template: str, **kwargs) -> str:
        """填充提示词模板中的占位符。"""
        prompt = template
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        return prompt

    def _initial(self, task: str, stream: bool = False) -> str:
        """阶段一：生成初始回答。"""
        prompt = self._build_prompt(self.INITIAL_PROMPT, task=task)
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, stream=stream)

    def _reflect(self, task: str, content: str, stream: bool = False) -> str:
        """阶段二：反思当前回答，返回反馈意见。"""
        prompt = self._build_prompt(self.REFLECT_PROMPT, task=task, content=content)
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, stream=stream)

    def _refine(self, task: str, last_attempt: str, feedback: str, stream: bool = False) -> str:
        """阶段三：根据反馈改进回答。"""
        prompt = self._build_prompt(self.REFINE_PROMPT, task=task, last_attempt=last_attempt, feedback=feedback)
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, stream=stream)

    def _score(self, task: str, content: str) -> int:
        """
        阶段四：对当前回答进行质量评分。
        使用非流式调用，确保获取完整响应后解析分数。
        :return: 0-100 的整数分数，解析失败返回 0
        """
        prompt = self._build_prompt(self.SCORE_PROMPT, task=task, content=content)
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages, temperature=0)

        if not response:
            return 0

        # 从响应中提取数字
        response = response.strip()
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            score = int(numbers[0])
            return min(max(score, 0), 100)
        return 0

    def run(self, task: str, stream: bool = False) -> str:
        """
        运行反思循环：初始回答 → 反思 → 评分 → 改进 → 反思 → ... 
        当评分达到阈值、反思认为无需改进、或达到最大轮次时终止。
        :param task: 任务描述
        :param stream: 是否流式打印每步的 LLM 响应
        :return: 最终回答
        """
        print("=" * 60)
        print(f"🎯 任务: {task}")
        print(f"📊 质量阈值: {self.score_threshold} 分")
        print("=" * 60)

        # 1. 生成初始回答
        print("\n📝 生成初始回答...")
        answer = self._initial(task, stream=stream)
        if not answer:
            print("❌ 初始回答生成失败")
            return ""
        self.memory.add("initial", answer)
        if not stream:
            print(answer)

        # 2. 反思-评分-改进循环
        for i in range(self.max_iterations):
            iteration = i + 1
            print(f"\n{'─' * 60}")
            print(f"📍 反思-改进 第 {iteration}/{self.max_iterations} 轮")
            print("─" * 60)

            # 2a. 反思当前回答
            print("\n🔍 反思当前回答...")
            feedback = self._reflect(task, answer, stream=stream)
            if not feedback:
                print("❌ 反思失败，终止循环")
                break
            self.memory.add("reflect", feedback)
            if not stream:
                print(feedback)

            # 2b. 质量评分
            print("\n📊 评估当前回答质量...")
            score = self._score(task, answer)
            self.memory.add("score", f"得分: {score}/100")
            print(f"   当前评分: {score}/100（阈值: {self.score_threshold}）")

            # 2c. 评分达到阈值，提前终止
            if score >= self.score_threshold:
                print(f"\n✅ 评分 {score} 已达到阈值 {self.score_threshold}，提前终止优化。")
                break

            # 2d. 反思认为无需改进
            if "无需改进" in feedback:
                print("\n✅ 反思认为回答已经很好，无需进一步改进。")
                break

            # 2e. 根据反馈改进回答
            print("\n✨ 根据反馈改进回答...")
            answer = self._refine(task, answer, feedback, stream=stream)
            if not answer:
                print("❌ 改进失败，终止循环")
                break
            self.memory.add("refine", answer)
            if not stream:
                print(answer)
        else:
            # for 循环正常结束（未 break），说明达到最大轮次
            print(f"\n⚠️ 已达到最大反思轮次 ({self.max_iterations})，停止改进。")

        # 3. 输出最终结果
        print("\n" + "=" * 60)
        print("📋 最终回答:")
        print("=" * 60)
        print(answer)

        return answer

    def show_memory(self):
        """展示短期记忆中的完整行动与反思轨迹。"""
        print("\n" + "=" * 60)
        print("🧠 短期记忆轨迹")
        print("=" * 60)
        print(self.memory.format())

    def reset(self):
        """清空短期记忆，重新开始。"""
        self.memory.clear()
        print("🔄 记忆已清空。")


if __name__ == "__main__":
    # 创建反思智能体，设置质量评分阈值为 85 分
    agent = ReflectionAgent(name="反思助手", max_iterations=3, score_threshold=85)

    test_task = "请解释什么是闭包(Closure)，并给出一个Python示例。"
    agent.run(test_task)

    # 展示完整的反思轨迹（含评分记录）
    agent.show_memory()

"""
Tree-of-Thought (ToT) 智能体
=============================
通过"生成多个思路 → 评估思路价值 → 搜索与剪枝"的循环，
在推理树中进行 beam search，寻找最优解题路径。

核心三步:
  1. Propose: 在当前节点生成 k 个不同的推理思路
  2. Evaluate: 对每个思路评分（0-10），判断引导至正确答案的可能性
  3. Search & Prune: 保留评分最高的 b 条路径（剪枝），继续向下探索
"""

import re
import json
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional


from src.core.llm import practiceLLM


# ============================================================
# 数据结构：推理树节点
# ============================================================

@dataclass
class ThoughtNode:
    """推理树的一个节点，代表一个思路步骤。"""
    thought: str                          # 当前思路内容
    score: float = 0.0                    # 评估分数（0-10）
    depth: int = 0                        # 在树中的深度
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = field(default_factory=list)

    def get_path(self) -> List[str]:
        """从根到当前节点的完整思路路径。"""
        path = []
        node = self
        while node is not None:
            path.append(node.thought)
            node = node.parent
        return list(reversed(path))

    def path_summary(self) -> str:
        """返回路径的格式化摘要。"""
        path = self.get_path()
        return "\n".join(f"  步骤{i+1}: {p}" for i, p in enumerate(path))


# ============================================================
# Tree-of-Thought 智能体
# ============================================================

class TreeOfThoughtAgent:
    """
    Tree-of-Thought 智能体。
    使用 beam search 在推理树中搜索最优解题路径。

    参数:
      - n_branches: 每个节点生成的思路数量（Propose 的 k）
      - beam_width: 保留的最优路径数（Prune 的 b）
      - max_depth: 最大搜索深度
      - value_threshold: 低于此分数的思路直接剪枝
    """

    # === 提示词模板 ===

    PROPOSE_PROMPT = """你是一个问题解决专家。请针对以下问题，生成 {k} 个不同的、可能的下一步推理思路。

# 问题
{problem}

# 已有推理路径
{current_path}

请生成 {k} 个不同的思路，每个思路应该是解题过程中的一个独立推理步骤。
请严格按照以下 JSON 格式输出，不要输出其他内容：
```json
[
    "思路1的内容",
    "思路2的内容",
    "思路3的内容"
]
```"""

    EVALUATE_PROMPT = """请评估以下推理思路对于解决给定问题的价值。

# 问题
{problem}

# 完整推理路径（含当前思路）
{full_path}

请评估当前思路及路径是否能有效引导至正确答案。
考虑：逻辑正确性、方向合理性、信息增益。
请仅输出一个 0 到 10 的数字（可含小数），10 表示非常有价值，0 表示毫无价值。
不要输出任何其他内容。"""

    ANSWER_PROMPT = """基于以下推理路径，给出问题的最终答案。

# 问题
{problem}

# 最优推理路径
{best_path}

请综合以上推理，给出一个清晰、完整的最终答案。"""

    def __init__(
        self,
        name: str = "ToTAgent",
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
        n_branches: int = 3,
        beam_width: int = 2,
        max_depth: int = 3,
        value_threshold: float = 3.0,
    ):
        self.name = name
        self.llm = practiceLLM(model=model, apiKey=apiKey, baseUrl=baseUrl, timeout=timeout)
        self.n_branches = n_branches
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.value_threshold = value_threshold

    # ============================================================
    # 核心步骤 1: Propose — 生成多个思路
    # ============================================================

    def _propose(self, problem: str, current_path: str) -> List[str]:
        """在当前节点生成 k 个不同的推理思路。"""
        prompt = self.PROPOSE_PROMPT.format(
            k=self.n_branches,
            problem=problem,
            current_path=current_path or "（起始节点，尚无推理路径）",
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages, temperature=0.8)

        if not response:
            return []

        # 解析 JSON 格式的思路列表
        return self._parse_thoughts(response)

    @staticmethod
    def _parse_thoughts(response: str) -> List[str]:
        """从 LLM 响应中解析思路列表。"""
        # 尝试提取 JSON 代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                thoughts = json.loads(json_match.group(1))
                if isinstance(thoughts, list):
                    return [str(t) for t in thoughts]
            except json.JSONDecodeError:
                pass

        # 尝试直接解析整个响应为 JSON
        try:
            thoughts = json.loads(response.strip())
            if isinstance(thoughts, list):
                return [str(t) for t in thoughts]
        except json.JSONDecodeError:
            pass

        # 降级：按行分割
        lines = [l.strip().lstrip('0123456789.-) ').strip() for l in response.strip().split("\n")]
        return [l for l in lines if l][:5]

    # ============================================================
    # 核心步骤 2: Evaluate — 评估思路价值
    # ============================================================

    def _evaluate(self, problem: str, full_path: str) -> float:
        """对当前推理路径进行评分（0-10）。"""
        prompt = self.EVALUATE_PROMPT.format(
            problem=problem,
            full_path=full_path,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages, temperature=0)

        if not response:
            return 0.0

        # 提取数字
        numbers = re.findall(r'(\d+\.?\d*)', response.strip())
        if numbers:
            score = float(numbers[0])
            return min(max(score, 0.0), 10.0)
        return 0.0

    # ============================================================
    # 核心步骤 3: Search & Prune — 搜索与剪枝（Beam Search）
    # ============================================================

    def run(self, problem: str) -> str:
        """
        运行 Tree-of-Thought 搜索。
        使用 beam search 策略：每层保留 beam_width 条最优路径。
        """
        print("=" * 60)
        print(f"🌳 Tree-of-Thought 搜索启动")
        print(f"   问题: {problem}")
        print(f"   参数: branches={self.n_branches}, beam={self.beam_width}, "
              f"max_depth={self.max_depth}, threshold={self.value_threshold}")
        print("=" * 60)

        # 初始化：根节点为空思路
        root = ThoughtNode(thought="（起始）", score=10.0, depth=0)
        frontier = [root]  # 当前待扩展的节点（beam）

        all_leaves: List[ThoughtNode] = []

        for depth in range(1, self.max_depth + 1):
            print(f"\n{'━' * 60}")
            print(f"📂 深度 {depth}/{self.max_depth}")
            print("━" * 60)

            candidates: List[ThoughtNode] = []

            for node in frontier:
                path_str = node.path_summary()
                print(f"\n  🔍 从路径扩展 (当前分数: {node.score:.1f}):")
                print(f"     {path_str}")

                # 步骤 1: Propose — 生成 k 个思路
                print(f"\n  💡 生成 {self.n_branches} 个候选思路...")
                thoughts = self._propose(problem, path_str)

                if not thoughts:
                    print("     ⚠️ 未生成有效思路")
                    continue

                # 步骤 2: Evaluate — 评估每个思路
                for thought in thoughts:
                    child = ThoughtNode(
                        thought=thought,
                        depth=depth,
                        parent=node,
                    )

                    full_path = child.path_summary()
                    print(f"\n  📊 评估思路: {thought[:80]}...")
                    score = self._evaluate(problem, full_path)
                    child.score = score
                    print(f"     得分: {score:.1f}/10")

                    # 剪枝：低于阈值的思路直接丢弃
                    if score >= self.value_threshold:
                        candidates.append(child)
                        node.children.append(child)
                        print(f"     ✅ 保留")
                    else:
                        print(f"     ✂️ 剪枝（低于阈值 {self.value_threshold}）")

            if not candidates:
                print(f"\n⚠️ 深度 {depth} 无有效候选，停止搜索")
                all_leaves = frontier
                break

            # 步骤 3: Prune — 按分数排序，保留 beam_width 条最优路径
            candidates.sort(key=lambda n: n.score, reverse=True)
            frontier = candidates[:self.beam_width]

            print(f"\n  🌿 深度 {depth} 剪枝结果（保留 {len(frontier)} 条）:")
            for i, node in enumerate(frontier, 1):
                print(f"     #{i} 分数={node.score:.1f}: {node.thought[:60]}...")

            all_leaves = frontier

        # 选择全局最优路径
        best_node = max(all_leaves, key=lambda n: n.score)
        best_path = best_node.path_summary()

        print(f"\n{'━' * 60}")
        print(f"🏆 最优路径（分数: {best_node.score:.1f}）:")
        print("━" * 60)
        print(best_path)

        # 基于最优路径生成最终答案
        print(f"\n📝 生成最终答案...")
        prompt = self.ANSWER_PROMPT.format(
            problem=problem,
            best_path=best_path,
        )
        messages = [{"role": "user", "content": prompt}]
        final_answer = self.llm.invoke(messages, temperature=0.3)

        print(f"\n{'=' * 60}")
        print("📋 最终答案:")
        print("=" * 60)
        print(final_answer)

        return final_answer


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    agent = TreeOfThoughtAgent(
        name="ToT助手",
        n_branches=3,
        beam_width=2,
        max_depth=3,
        value_threshold=3.0,
    )

    # 经典推理问题
    problem = "一个水池有两个进水管和一个出水管。进水管A单独注满水池需要6小时，进水管B单独注满需要8小时，出水管C单独排空需要12小时。如果三管同时打开，多少小时能注满水池？"

    agent.run(problem)
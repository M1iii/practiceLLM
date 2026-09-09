import re
import ast
import json
import sys
import os


from src.core.llm import practiceLLM
from src.tools.framework.tool_system import Tool, ToolRegistry
from typing import List, Dict, Optional


class Planner:
    """
    规划器：将复杂问题分解为有序的子任务列表。
    调用 LLM 生成计划，并从响应中解析出 Python 列表。
    """

    DEFAULT_PROMPT = """你是一个顶级的AI规划专家，你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划，请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列，你的输出必须是一个python列表，其中每个元素都是一个描述子任务的字符串。

问题：{question}

请严格按照以下格式输出你的计划：
```python
["步骤1", "步骤2", "步骤3", ...]
```"""

    def __init__(self, llm: practiceLLM, prompt_template: str = None):
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def plan(self, question: str) -> List[str]:
        """调用 LLM 生成计划，返回子任务列表。"""
        # 填充提示词模板
        prompt = self.prompt_template.replace("{question}", question)
        messages = [{"role": "user", "content": prompt}]

        # 规划阶段使用非流式调用，因为需要完整响应来解析列表
        response = self.llm.invoke(messages)
        if not response:
            print("⚠️ 规划器未收到响应")
            return []

        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> List[str]:
        """从 LLM 响应中解析 Python 列表。"""
        # 优先尝试从 ```python ... ``` 代码块中提取
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            list_str = code_match.group(1)
        else:
            # 回退：尝试直接匹配列表模式 [ ... ]
            list_match = re.search(r'\[.*\]', response, re.DOTALL)
            if list_match:
                list_str = list_match.group(0)
            else:
                print("⚠️ 无法从响应中提取计划列表")
                return []

        # 使用 ast.literal_eval 安全解析 Python 字面量
        try:
            plan = ast.literal_eval(list_str)
            if isinstance(plan, list):
                return [str(item) for item in plan]
        except (ValueError, SyntaxError) as e:
            print(f"⚠️ 计划解析失败: {e}")

        return []


class Executor:
    """
    执行器：按计划逐步执行子任务，维护历史记录。
    每一步都会将原始问题、完整计划、历史结果和当前步骤传入提示词，确保 LLM 有充分上下文。
    当工具可用时，使用 Function Calling 让 LLM 自主调用工具获取信息。
    """

    DEFAULT_PROMPT = """你是一位顶级的AI执行专家，你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及直到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释和对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:"""

    def __init__(self, llm: practiceLLM, prompt_template: str = None,
                 tools: List[Tool] = None):
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.history: List[Dict[str, str]] = []
        self.tools = tools or []
        self.tool_registry = ToolRegistry(self.tools) if self.tools else None

    def execute(self, question: str, plan: List[str], stream: bool = False) -> str:
        """遍历计划中的每个步骤，逐步执行并记录结果。"""
        self.history = []

        for i, step in enumerate(plan, 1):
            print(f"\n📍 步骤 {i}/{len(plan)}: {step}")

            # 执行当前步骤
            result = self._execute_step(question, plan, step, stream=stream)
            if result is None:
                result = "（执行失败）"

            # 记录到历史
            self.history.append({"step": step, "result": result})

            # 非流式模式下打印结果（流式模式下已由 invoke 实时打印）
            if not stream:
                print(f"→ {result}")

        # 返回最后一步的结果作为最终答案
        return self.history[-1]["result"] if self.history else ""

    def _execute_step(self, question: str, plan: List[str], current_step: str, stream: bool = False) -> str:
        """构建提示词并执行单个步骤（工具可用时自动调用工具获取信息）。"""
        prompt = self.prompt_template
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{plan}", self._format_plan(plan))
        prompt = prompt.replace("{history}", self._format_history())
        prompt = prompt.replace("{current_step}", current_step)

        if self.tool_registry and not stream:
            return self._invoke_with_tools(prompt)
        return self.llm.invoke([{"role": "user", "content": prompt}], stream=stream)

    def _invoke_with_tools(self, prompt: str, max_rounds: int = 8) -> str:
        """使用 Function Calling 执行 LLM 调用，支持工具循环。

        当工具可用时，LLM 可自主决定调用工具获取信息，然后再生成文本回答。
        """
        messages = [
            {"role": "system",
             "content": "你是一个可以调用工具的AI助手。"
                        "调用工具收集到足够信息后，请直接给出文本回答，不要继续调用工具。"},
            {"role": "user", "content": prompt},
        ]
        openai_tools = [t.to_openai_format() for t in self.tool_registry.get_tools()]

        for _ in range(max_rounds):
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                result = self.tool_registry.execute_structured(fn_name, fn_args)
                output = result.output if hasattr(result, "output") else str(result)
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(output)})
                print(f"🔧 {fn_name}({fn_args}) → {str(output)[:80]}")

        # 超限后，使用已有工具结果请求合成，不丢弃上下文
        messages.append({"role": "user",
                         "content": "请基于以上所有工具结果，给出最终回答。"})
        final = self.llm.client.chat.completions.create(
            model=self.llm.model, messages=messages, temperature=0,
        )
        return final.choices[0].message.content or ""

    def _format_plan(self, plan: List[str]) -> str:
        """将计划格式化为可读文本，供执行器提示词使用。"""
        lines = []
        for i, step in enumerate(plan, 1):
            lines.append(f"{i}. {step}")
        return "\n".join(lines)

    def _format_history(self) -> str:
        """格式化历史记录，供执行器提示词使用。"""
        if not self.history:
            return "（暂无历史步骤）"

        lines = []
        for i, entry in enumerate(self.history, 1):
            lines.append(f"步骤{i}: {entry['step']}")
            lines.append(f"结果: {entry['result']}")
            lines.append("")
        return "\n".join(lines)

    def get_history(self) -> List[Dict[str, str]]:
        """获取执行历史记录。"""
        return self.history

    def reset(self):
        """清空历史记录。"""
        self.history = []


class PlanAndSolveAgent:
    """
    计划与解决智能体：先由规划器将复杂问题分解为子任务列表，
    再由执行器逐步执行每个子任务，最终得到完整答案。

    工作流程: 问题 → Planner 生成计划 → Executor 逐步执行 → 最终答案
    """

    def __init__(
        self,
        llm: practiceLLM,
        name: str = "PlanAndSolveAgent",
        planner_prompt: str = None,
        executor_prompt: str = None,
        tools: List[Tool] = None,
    ):
        """
        初始化智能体。
        :param llm: LLM 客户端，由规划器和执行器共享
        :param name: Agent 名称
        :param planner_prompt: 自定义规划器提示词
        :param executor_prompt: 自定义执行器提示词
        :param tools: 可用工具列表（执行阶段使用）
        """
        self.name = name
        self.llm = llm
        self.planner = Planner(llm, planner_prompt)
        self.executor = Executor(llm, executor_prompt, tools=tools)

    def run(self, question: str, stream: bool = False) -> str:
        """
        启动完整流程：规划 → 逐步执行 → 返回最终答案。
        :param question: 用户的问题
        :param stream: 是否流式打印执行步骤的响应
        :return: 最终答案
        """
        print("=" * 60)
        print(f"🎯 问题: {question}")
        print("=" * 60)

        # 1. 规划阶段：将问题分解为子任务
        print("\n📋 [规划阶段] 正在分解任务...")
        plan = self.planner.plan(question)

        if not plan:
            print("❌ 规划失败，无法生成有效计划")
            return ""

        print(f"✅ 共分解为 {len(plan)} 个步骤:")
        for i, step in enumerate(plan, 1):
            print(f"   {i}. {step}")

        # 2. 执行阶段：逐步执行每个子任务
        print(f"\n🚀 [执行阶段] 开始逐步执行...")
        final_answer = self.executor.execute(question, plan, stream=stream)

        # 3. 输出最终结果
        print("\n" + "=" * 60)
        print("📋 最终答案:")
        print("=" * 60)
        print(final_answer)

        return final_answer

    def show_history(self):
        """展示执行历史记录。"""
        history = self.executor.get_history()
        if not history:
            print("\n📜 暂无执行历史")
            return

        print("\n" + "=" * 60)
        print("📜 执行历史")
        print("=" * 60)
        for i, entry in enumerate(history, 1):
            print(f"\n--- 步骤 {i}: {entry['step']} ---")
            print(f"结果: {entry['result']}")

    def reset(self):
        """重置执行器历史。"""
        self.executor.reset()
        print("🔄 已重置执行历史。")


if __name__ == "__main__":
    # 创建 LLM 客户端并传入 PlanAndSolveAgent
    llm = practiceLLM()
    agent = PlanAndSolveAgent(llm=llm, name="规划执行助手")

    test_question = "一个水池有一个进水管和一个出水管，进水管每小时注水5吨，出水管每小时放水3吨，水池容量为20吨。如果水池初始为空，同时打开进水管和出水管，多少小时后水池能注满？"
    agent.run(test_question)
    agent.show_history()
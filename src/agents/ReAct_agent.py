import re
import sys
import os


from src.core.llm import practiceLLM, LLMClient    # LLM 客户端 + 统一接口协议
from src.tools.framework.tool_system import Tool, ToolRegistry, CalculatorTool, TimeTool
from typing import List, Dict, Optional


class ReActAgent:
    """
    基于 ReAct (Reasoning + Acting) 模式的 Agent。
    通过 Thought → Action → Observation 的循环，逐步推理并调用工具，直到得出最终答案。
    """

    # ReAct 提示词模板
    PROMPT_TEMPLATE = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tool}

## 工作流程
请严格按照以下格式进行回应，每次只执行一个步骤：

Thought：分析当前问题，思考需要所需信息或采取什么行动。
Action：选择一个行动，格式必须是以下之一：
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1.每次回应必须包含Thought和Action两部分
2.工具调用的格式必须严格遵守：工具名[参数]
3.只有当你确定有足够信息回答问题时，才使用Finish
4.如果工具返回的信息不够，继续使用其他或相同工具的不同参数

## 当前任务
**Question：**{question}

##执行历史
{history}

现在开始你的推理和行动："""

    def __init__(
        self,
        tools: List[Tool] = None,
        llm: Optional[LLMClient] = None,      # 注入点：外部传入 LLM（mock/真实/包装）
        model: str = None,                    # 以下仅内部构造回退路径使用（向后兼容）
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
        max_steps: int = 5,
    ):
        """
        初始化 ReAct Agent。
        :param tools: 可用工具列表，默认使用 Calculator 和 Time
        :param llm: LLM 客户端（注入）；None 时内部构造 practiceLLM
        :param max_steps: 最大推理步数，防止无限循环
        """
        self.tools = tools if tools is not None else [CalculatorTool(), TimeTool()]

        # 依赖注入优先；未注入时回退内部构造（兼容旧调用方）
        self.llm: LLMClient = llm or practiceLLM(
            model=model, apiKey=apiKey, baseUrl=baseUrl, timeout=timeout)
        self.max_steps = max_steps

        # 统一工具注册表（find/execute_structured + 熔断保护）
        self.tool_registry = ToolRegistry(self.tools)

    def _format_tools(self) -> str:
        """将所有工具的名称和描述格式化为提示词中的工具说明文本。"""
        return "\n".join(str(tool) for tool in self.tools)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """将执行历史格式化为提示词中的历史记录文本。"""
        if not history:
            return "（暂无执行历史）"

        lines = []
        for i, step in enumerate(history, 1):
            lines.append(f"--- 第{i}步 ---")
            lines.append(f"Thought：{step['thought']}")
            lines.append(f"Action：{step['action']}")
            lines.append(f"Observation：{step['observation']}")
            lines.append("")
        return "\n".join(lines)

    def _build_prompt(self, question: str, history: List[Dict[str, str]]) -> str:
        """组装完整的提示词，填入工具、问题和执行历史。"""
        prompt = self.PROMPT_TEMPLATE
        prompt = prompt.replace("{tool}", self._format_tools())
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{history}", self._format_history(history))
        return prompt

    def _parse_response(self, response: str) -> Optional[Dict[str, str]]:
        """
        从 LLM 的回复中解析 Thought 和 Action。
        支持中英文冒号，若解析失败则返回 None。
        """
        result = {"thought": "", "action": ""}

        # 提取 Thought 部分（匹配 "Thought：" 或 "Thought:" 之后的内容）
        thought_match = re.search(r"Thought[：:]\s*(.*?)(?=\n\s*Action[：:]|$)", response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 提取 Action 部分（匹配到下一个 Thought/Observation 或末尾为止）
        action_match = re.search(r"Action[：:]\s*(.*?)(?=\n\s*(?:Thought|Observation)[：:]|\Z)", response, re.DOTALL)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # 如果未提取到任何内容，返回 None 表示解析失败
        if not result["thought"] and not result["action"]:
            return None

        return result

    def _parse_action(self, action_str: str) -> Dict[str, str]:
        """
        解析 Action 字符串，提取工具名和参数。
        返回 {"tool_name": ..., "tool_input": ...} 或 {"finish": ...}。
        """
        action_str = action_str.strip()

        # 尝试匹配 Finish[最终答案] 格式
        finish_match = re.match(r"Finish\[(.*)\]$", action_str, re.DOTALL)
        if finish_match:
            return {"finish": finish_match.group(1).strip()}

        # 尝试匹配 tool_name[tool_input] 格式（允许空参数）
        tool_match = re.match(r"(\w+)\[(.*)\]$", action_str, re.DOTALL)
        if tool_match:
            return {"tool_name": tool_match.group(1), "tool_input": tool_match.group(2).strip()}

        # 格式不匹配，返回原始文本供调用方处理
        return {"raw": action_str}

    def _get_tool(self, tool_name: str) -> Optional[Tool]:
        """按名称获取工具对象（大小写不敏感，兼容 LLM 输出差异）。"""
        return self.tool_registry.find_tool(tool_name)

    def _parse_input(self, tool: Tool, tool_input: str) -> dict:
        """将 Action 的字符串参数转换为工具参数字典（新框架 Tool.run 接受 dict）。"""
        params = tool.get_parameters()
        if not params or not tool_input:
            return {}
        # 计算器：表达式直接映射到 expression 参数
        if tool.name.lower() == "calculator":
            return {"expression": tool_input}
        # 其他工具：按第一个参数名推断
        return {params[0].name: tool_input}

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """根据工具名查找并执行对应工具，返回执行结果（统一走 ToolRegistry 熔断保护）。"""
        tool = self._get_tool(tool_name)
        if tool is None:
            available = ", ".join(self.tool_registry.list_tools())
            return f"错误：未找到工具 '{tool_name}'，可用工具：{available}"
        try:
            resp = self.tool_registry.execute_structured(
                tool.name, self._parse_input(tool, tool_input))
            if resp.is_error:
                return f"工具执行失败: {resp.error}"
            return resp.output
        except Exception as e:
            return f"工具执行错误: {e}"

    def run(self, question: str) -> str:
        """
        ReAct 主循环：不断推理 → 行动 → 观察，直到得出最终答案或达到最大步数。
        """
        print("=" * 60)
        print(f"🎯 问题: {question}")
        print("=" * 60)

        # 执行历史，记录每一步的 Thought、Action 和 Observation
        history: List[Dict[str, str]] = []

        for step in range(1, self.max_steps + 1):
            print(f"\n{'─' * 60}")
            print(f"📍 第 {step} 步")
            print("─" * 60)

            # 1. 组装提示词并调用 LLM
            prompt = self._build_prompt(question, history)
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.think(messages, temperature=0)

            if not response:
                print("❌ LLM 返回空响应，终止推理。")
                return "推理失败：LLM 无响应。"

            # 2. 解析 LLM 回复中的 Thought 和 Action
            parsed = self._parse_response(response)
            if parsed is None:
                print("❌ 无法解析 LLM 回复格式，终止推理。")
                print(f"原始回复:\n{response}")
                return "推理失败：回复格式无法解析。"

            thought = parsed["thought"]
            action_str = parsed["action"]

            print(f"\n💭 Thought: {thought}")
            print(f"⚡ Action: {action_str}")

            # 3. 解析 Action 并执行
            action = self._parse_action(action_str)

            # 情况 A：Action 是 Finish，返回最终答案
            if "finish" in action:
                final_answer = action["finish"]
                print(f"\n✅ 最终答案: {final_answer}")
                print("=" * 60)
                return final_answer

            # 情况 B：Action 是工具调用
            elif "tool_name" in action:
                tool_name = action["tool_name"]
                tool_input = action["tool_input"]

                # 执行工具，获取观察结果
                observation = self._execute_tool(tool_name, tool_input)
                print(f"🔍 Observation: {observation}")

                # 将本步结果记录到历史中
                history.append({
                    "thought": thought,
                    "action": action_str,
                    "observation": observation,
                })

            # 情况 C：Action 格式无法识别
            else:
                observation = f"错误：无法识别的 Action 格式: {action_str}"
                print(f"🔍 Observation: {observation}")
                history.append({
                    "thought": thought,
                    "action": action_str,
                    "observation": observation,
                })

        # 达到最大步数仍未得出答案
        print(f"\n⚠️ 已达到最大推理步数 ({self.max_steps})，强制终止。")
        return "推理失败：达到最大步数限制。"


if __name__ == "__main__":
    # 创建 ReAct Agent 并运行示例问题
    agent = ReActAgent()

    test_question = "现在几点了？另外帮我算一下 (15 + 27) * 3 等于多少？"
    agent.run(test_question)
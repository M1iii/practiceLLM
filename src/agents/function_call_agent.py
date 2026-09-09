import json
import sys
import os
import time
from typing import List, Dict, Optional, Union, Any, Set


from src.core.llm import practiceLLM
from src.tools.framework.tool_system import Tool, ToolRegistry, CalculatorTool, TimeTool    # 新框架工具


class FunctionCallAgent:
    """
    基于 OpenAI 原生 Function Calling 的智能体。
    与文本解析式工具调用不同，本 Agent 利用 LLM 原生的 function calling 能力，
    由模型直接返回结构化的工具调用请求，无需解析文本标记，更可靠也更精确。

    工作流程:
    1. 将工具转换为 OpenAI function schema，随消息一起发送给 LLM
    2. LLM 决定是否调用工具，返回 tool_calls 或最终回答
    3. 若有 tool_calls：执行对应工具，将结果以 role="tool" 回传给 LLM，回到步骤 2
    4. 若无 tool_calls：当前回答即为最终答案

    防重试机制:
    - 同工具连续失败 max_tool_failures 次后自动禁用该工具
    - 禁用持续 tool_reset_interval 秒后自动恢复
    - 返回结果包含"【搜索不可用】"等终端错误信号时立即禁用
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是一个有用的AI助手，可以借助工具来帮助用户回答问题。\n"
        "当用户提到图片或提供图片路径时，使用 VLMediaTool 分析图片内容。\n"
        "调用工具收集到足够信息后，请直接给出文本回答，不要继续调用工具。"
    )

    # 终端错误信号前缀：匹配到这些前缀的工具结果将立即禁用该工具
    TERMINAL_ERROR_PREFIXES = ("【搜索不可用】",)

    def __init__(
        self,
        llm: practiceLLM,
        name: str = "FunctionCallAgent",
        tools: List[Tool] = None,
        system_prompt: str = None,
        max_iterations: int = 10,
        max_tool_failures: int = 3,
        tool_reset_interval: int = 120,
    ):
        """
        初始化 Function Call Agent。
        :param llm: LLM 客户端，需包含 .client (OpenAI 实例) 和 .model 属性
        :param name: Agent 名称
        :param tools: 可用工具列表
        :param system_prompt: 系统提示词
        :param max_iterations: 最大工具调用轮次
        :param max_tool_failures: 同工具连续失败次数上限（达到后禁用）
        :param tool_reset_interval: 工具禁用后自动恢复的秒数
        """
        self.name = name
        self.llm = llm
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # 统一工具注册表（register/remove/find/execute_structured + 熔断保护）
        self.tool_registry = ToolRegistry()
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

        # 防重试状态
        self._tool_failures: Dict[str, int] = {}          # 工具名 → 连续失败次数
        self._disabled_tools: Dict[str, float] = {}       # 工具名 → 禁用时间戳
        self._max_tool_failures = max_tool_failures
        self._tool_reset_interval = tool_reset_interval

        # 断点续聊：多轮对话历史（role/content 字典列表）
        self._history: List[Dict[str, str]] = []
        self._max_history_turns: int = 20  # 保留最近 20 轮（40 条消息）

    def _convert_tools_to_openai_format(self) -> List[Dict[str, Any]]:
        """将 Tool 对象列表转换为 OpenAI function calling 所需的 schema 格式。

        跳过已禁用的工具（防重试）。
        """
        self._recover_disabled_tools()
        openai_tools = []
        for tool in self.tool_registry.get_tools():
            if tool.name in self._disabled_tools:
                continue
            openai_tools.append(tool.to_openai_format())
        return openai_tools

    def _invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, dict] = "auto",
        temperature: float = 0,
    ):
        """
        调用底层 OpenAI 客户端执行带工具的对话。
        直接访问 practiceLLM 内部的 client 和 model，以使用原生 function calling。
        """
        return self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )

    # ============================================================
    # 防重试：工具禁用/恢复
    # ============================================================

    def _recover_disabled_tools(self):
        """检查并恢复已过期的禁用工具。"""
        now = time.time()
        expired = [name for name, ts in self._disabled_tools.items()
                   if now - ts > self._tool_reset_interval]
        for name in expired:
            del self._disabled_tools[name]
            self._tool_failures.pop(name, None)
            print(f"🔁 工具 {name} 已自动恢复（禁用期结束）")

    def _is_terminal_error(self, result: str) -> bool:
        """判断工具返回结果是否包含终端错误信号。"""
        return any(result.startswith(prefix) for prefix in self.TERMINAL_ERROR_PREFIXES)

    def _track_tool_result(self, function_name: str, result: str):
        """跟踪工具执行结果，更新连续失败计数并决定是否禁用。"""
        # 判断是否失败
        is_error = result.startswith("错误：") or result.startswith("工具执行失败:") or result.startswith("【搜索不可用】")
        is_terminal = self._is_terminal_error(result)

        if is_error:
            self._tool_failures[function_name] = self._tool_failures.get(function_name, 0) + 1
            failures = self._tool_failures[function_name]

            if is_terminal or failures >= self._max_tool_failures:
                # 终端错误或连续失败超限 → 禁用工具
                self._disabled_tools[function_name] = time.time()
                reason = "终端错误信号" if is_terminal else f"连续失败 {failures} 次"
                print(f"⛔ 工具 {function_name} 已禁用（{reason}）")
        else:
            # 成功 → 重置连续失败计数
            self._tool_failures.pop(function_name, None)

    def _execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """根据函数名和参数执行对应的工具，返回结果字符串。

        统一走 ToolRegistry.execute_structured（熔断器 + 参数校验 + ToolResponse 协议）。
        """
        resp = self.tool_registry.execute_structured(function_name, arguments or {})
        if resp.is_error:
            return f"工具执行失败: {resp.error}"
        return resp.output

    def run(self, question: str, temperature: float = 0) -> str:
        """
        启动 Function Calling 循环。
        :param question: 用户问题
        :param temperature: LLM 温度参数
        :return: 最终回答
        """
        print("=" * 60)
        print(f"🎯 问题: {question}")
        print("=" * 60)

        # 1. 初始化消息列表（含断点续聊历史）
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if self._history:
            for msg in self._history:
                role = msg.get("role", "")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": msg.get("content", "")})
            print(f"💬 已注入 {len(self._history)} 条历史消息")

        # 当前用户问题
        messages.append({"role": "user", "content": question})

        # 2. 转换工具为 OpenAI 格式
        openai_tools = self._convert_tools_to_openai_format()

        # 3. 工具调用循环
        for iteration in range(self.max_iterations):
            print(f"\n{'─' * 60}")
            print(f"📍 第 {iteration + 1}/{self.max_iterations} 轮")
            print("─" * 60)

            # 3a. 刷新工具列表（移除已禁用的工具）
            active_tool_names = {t["function"]["name"] for t in openai_tools} - set(self._disabled_tools.keys())
            available_tools = [t for t in openai_tools if t["function"]["name"] in active_tool_names]

            # 3b. 调用 LLM（带工具）
            response = self._invoke_with_tools(messages, available_tools, tool_choice="auto", temperature=temperature)
            message = response.choices[0].message

            # 3b. 检查是否有工具调用
            if message.tool_calls:
                # 将助手消息（含 tool_calls）加入对话历史
                messages.append(message)

                # 逐个执行工具调用
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name

                    # 解析 JSON 格式的参数字符串
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    print(f"\n🔧 调用工具: {function_name}({arguments})")

                    # 执行工具
                    result = self._execute_function(function_name, arguments)
                    print(f"🔍 结果: {result}")

                    # 跟踪工具执行结果（防重试）
                    self._track_tool_result(function_name, result)

                    # 将工具执行结果加入对话历史（role="tool"）
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })

                # 有工具调用，继续下一轮让 LLM 处理结果
                continue

            # 3c. 没有工具调用，当前回答即为最终答案
            final_answer = message.content or ""
            print(f"\n✅ 最终回答:")
            print(final_answer)
            print("=" * 60)

            # 保存本轮对话到历史（断点续聊）
            self._append_turn(question, final_answer)
            return final_answer

        # 4. 达到最大迭代次数，用已有工具结果请求合成
        messages.append({"role": "user",
                         "content": "请基于以上所有工具结果，给出最终回答。"})
        fallback = self._invoke_with_tools(messages, openai_tools, tool_choice="none", temperature=0)
        final_answer = fallback.choices[0].message.content or "抱歉，处理超时。"
        print(f"\n⚠️ 已达到最大工具调用轮次 ({self.max_iterations})，强制合成。")
        print(f"✅ 最终回答:\n{final_answer}")
        print("=" * 60)
        self._append_turn(question, final_answer)
        return final_answer

    def _append_turn(self, question: str, answer: str):
        """保存一轮对话到历史，并裁剪超限轮次。"""
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})
        # 裁剪：保留最近 max_history_turns 轮
        max_msgs = self._max_history_turns * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def add_tool(self, tool: Tool):
        """添加一个工具。"""
        self.tool_registry.register(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """移除一个工具，返回是否成功。"""
        return self.tool_registry.remove(tool_name)

    def list_tools(self) -> List[str]:
        """列出所有可用工具的名称。"""
        return self.tool_registry.list_tools()

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """按名称获取工具对象（大小写不敏感）。"""
        return self.tool_registry.find_tool(tool_name)


if __name__ == "__main__":
    # 创建 LLM 客户端并传入 FunctionCallAgent
    llm = practiceLLM()
    agent = FunctionCallAgent(
        llm=llm,
        name="函数调用助手",
        tools=[CalculatorTool(), TimeTool()],
    )

    test_question = "现在几点了？另外帮我算一下 (15 + 27) * 3 等于多少？"
    agent.run(test_question)
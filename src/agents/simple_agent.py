import re
import sys
import os


from src.core.llm import practiceLLM, LLMClient    # LLM 客户端 + 统一接口协议
from src.agents.framework.agent_framework import Message    # 统一消息模型（agent_framework.py）
from src.tools.framework.tool_system import Tool, ToolRegistry    # 新框架工具基类 + 统一注册表
from typing import List, Optional


class SimpleAgent:
    """
    一个负责日常聊天的简单 Agent。
    支持普通对话和可选的工具调用，基于 LLM 客户端维护多轮对话历史。

    依赖注入：LLM 由外部传入（llm 参数），未注入时回退内部构造 practiceLLM
    （兼容旧调用方）。工具执行统一走 ToolRegistry.execute_structured（熔断 + 校验）。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是一个友好的日常聊天助手。请用自然、亲切的语气回答用户的问题，"
        "保持对话的连贯性和趣味性。"
    )

    def __init__(
        self,
        name: str = "SimpleAgent",
        llm: Optional[LLMClient] = None,      # 注入点：外部传入 LLM（mock/真实/包装）
        system_prompt: str = None,
        tools: List[Tool] = None,
        enable_tool_calling: bool = False,
        model: str = None,                    # 以下仅内部构造回退路径使用（向后兼容）
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
    ):
        """
        初始化 Agent。
        :param name: Agent 名称，用于输出展示
        :param llm: LLM 客户端（注入）；None 时内部构造 practiceLLM
        :param system_prompt: 系统提示词
        :param tools: 可用工具列表
        :param enable_tool_calling: 是否启用工具调用
        """
        self.name = name
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # 依赖注入优先；未注入时回退内部构造（兼容旧调用方）
        self.llm: LLMClient = llm or practiceLLM(
            model=model, apiKey=apiKey, baseUrl=baseUrl, timeout=timeout)

        self._history: List[Message] = []
        self.enable_tool_calling = enable_tool_calling
        self.tool_registry = None
        if tools:
            for tool in tools:
                self.add_tool(tool)

    def add_message(self, message: Message):
        """将一条消息添加到对话历史。"""
        self._history.append(message)

    def add_tool(self, tool: Tool):
        """添加一个工具到注册表。"""
        if not self.tool_registry:
            self.tool_registry = ToolRegistry()
        self.tool_registry.register(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """移除一个工具，返回是否成功。"""
        if not self.tool_registry:
            return False
        return self.tool_registry.remove(tool_name)

    def has_tools(self) -> bool:
        """检查是否有可用工具。"""
        return self.tool_registry is not None and self.tool_registry.has_tools()

    def list_tools(self) -> List[str]:
        """列出所有可用工具的名称。"""
        if not self.tool_registry:
            return []
        return self.tool_registry.list_tools()

    def _get_enhanced_system_prompt(self) -> str:
        """动态构建增强的系统提示词，能够获取实时工具信息。"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        # 未启用工具调用或无工具注册时，返回基础提示词
        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        # 拼接工具说明部分
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section

    def run(self, input_text: str, max_tool_iterations: int = 3, stream: bool = False, **kwargs) -> str:
        """
        运行方法 - 实现简单对话逻辑，支持可选工具调用。
        :param input_text: 用户输入
        :param max_tool_iterations: 最大工具调用轮次
        :param stream: 是否流式打印响应内容
        :return: Agent 的响应文本
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")

        # 构建消息列表
        messages = []

        # 添加系统消息（可能包含工具信息）
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，使用简单对话逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, stream=stream, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"✅ {self.name} 响应完成")
            if not stream:
                print(f"💬 {self.name}: {response}")
            return response

        # 支持多轮工具调用的逻辑
        return self._run_with_tools(messages, input_text, max_tool_iterations, stream=stream, **kwargs)

    def _run_with_tools(self, messages: list, input_text: str, max_tool_iterations: int, stream: bool = False, **kwargs) -> str:
        """支持工具调用的运行逻辑"""
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            # 1. 调用 LLM 获取响应
            response = self.llm.invoke(messages, stream=stream, **kwargs)
            if not response:
                print(f"❌ {self.name} LLM 无响应")
                return "抱歉，我无法处理您的请求。"

            # 2. 解析响应中的工具调用（解析器：LLM 响应后）
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用")
                # 执行所有工具调用并收集结果
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    # 3. 执行工具调用（执行器：真正调用工具）
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # 从响应中移除工具调用标记，保留纯文本部分
                    clean_response = clean_response.replace(call['original'], "")

                # 将清理后的助手响应加入消息
                messages.append({"role": "assistant", "content": clean_response.strip()})

                # 添加工具结果，引导 LLM 基于结果继续回答
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue

            # 4. 没有工具调用，当前响应即为最终答案
            final_response = response
            break

        # 5. 如果超过最大迭代次数仍未得到最终答案，再调用一次 LLM
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, stream=stream, **kwargs)

        # 6. 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"✅ {self.name} 响应完成")
        if not stream:
            print(f"💬 {self.name}: {final_response}")

        return final_response

    def _parse_tool_calls(self, text: str) -> list:
        """
        解析器：从 LLM 响应文本中提取工具调用标记。
        格式: [TOOL_CALL:tool_name:parameters]
        """
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]*)\]'
        tool_calls = []

        for match in re.finditer(pattern, text):
            tool_name = match.group(1).strip()
            parameters = match.group(2).strip()
            original = match.group(0)  # 完整匹配文本，用于后续从响应中移除

            # 清理参数：LLM 可能在参数外多包一层方括号
            if parameters.startswith('[') and parameters.endswith(']'):
                parameters = parameters[1:-1]
            elif parameters.startswith('['):
                parameters = parameters[1:]

            tool_calls.append({
                'tool_name': tool_name,
                'parameters': parameters,
                'original': original
            })

        return tool_calls

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """
        参数解析器：将工具参数字符串智能解析为字典。
        支持 key=value 格式和基于工具参数 schema 的自动推断。
        """
        param_dict = {}

        if '=' in parameters:
            # 格式: key=value 或 key1=val1,key2=val2
            if ',' in parameters:
                # 多个参数: action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数: key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()
        else:
            # 无 '=' 时按工具类型/参数 schema 智能推断
            if tool_name.lower() == 'memory':
                param_dict = {'action': 'search', 'query': parameters}
            else:
                # 优先按工具第一个参数名映射（适配新框架参数 schema）
                param_name = self._first_param_name(tool_name)
                param_dict = {param_name: parameters} if param_name else {'input': parameters}

        return param_dict

    def _get_tool(self, tool_name: str):
        """按名称获取工具对象（大小写不敏感，兼容 LLM 输出的大小写差异）。"""
        return self.tool_registry.find_tool(tool_name)

    def _first_param_name(self, tool_name: str) -> Optional[str]:
        """获取工具的第一个参数名（用于无 key=value 时的参数推断）。"""
        tool = self._get_tool(tool_name)
        if not tool:
            return None
        params = tool.get_parameters()
        return params[0].name if params else None

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """
        执行器：执行单个工具调用。
        统一走新框架 ToolRegistry 结构化执行（熔断器 + 参数校验 + ToolResponse 协议）。
        """
        if not self.tool_registry:
            return "❌ 错误:未配置工具注册表"

        try:
            # 获取工具对象（大小写不敏感）
            tool = self._get_tool(tool_name)
            if not tool:
                available = ", ".join(self.tool_registry.list_tools())
                return f"❌ 错误:未找到工具 '{tool_name}'，可用工具：{available}"

            # 智能参数解析：计算器类工具直接传入表达式，其他工具按 schema 推断
            if tool_name.lower() == 'calculator':
                param_dict = {'expression': parameters} if parameters else {}
            else:
                param_dict = self._parse_tool_parameters(tool_name, parameters)

            # 统一走新框架结构化执行（含熔断保护）
            resp = self.tool_registry.execute_structured(tool.name, param_dict)
            if resp.is_error:
                return f"❌ 工具 {tool_name} 执行失败: {resp.error}"

            return f"🔧 工具 {tool_name} 执行结果:\n{resp.output}"

        except Exception as e:
            return f"❌ 工具调用失败:{str(e)}"

    def chat(self, user_input: str, **kwargs) -> str:
        """快捷方法，等价于调用 run。"""
        return self.run(user_input, **kwargs)

    def reset(self):
        """清空对话历史，重新开始聊天。"""
        self._history = []
        print("🔄 对话已重置。")

    def show_history(self):
        """打印当前对话历史。"""
        print("\n📜 对话历史:")
        for msg in self._history:
            label = {"user": "🧑 你", "assistant": "🤖 助手"}.get(msg.role, msg.role)
            print(f"  {label}: {msg.content}")


if __name__ == "__main__":
    from src.tools.framework.tool_system import CalculatorTool, TimeTool

    # 创建带工具的 Agent 演示
    agent = SimpleAgent(
        name="小助手",
        tools=[CalculatorTool(), TimeTool()],
        enable_tool_calling=True,
    )

    print("=" * 50)
    print(f"  欢迎使用 {agent.name} 聊天助手（工具已启用）")
    print("  输入 'quit' 或 'exit' 退出")
    print("  输入 'reset' 重置对话")
    print("  输入 'history' 查看历史")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("👋 再见！")
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue
        if user_input.lower() == "history":
            agent.show_history()
            continue

        agent.run(user_input)
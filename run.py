"""run.py — Agent 启动器（运行文件）

激活并运行项目中的 Agent（基于统一框架 BaseAgent + AgentFactory + AgentRegistry）。

用法:
    python run.py                     # 交互式会话（智能路由默认开启）
    python run.py "你的问题"          # 单次执行（智能路由自动选择 Agent）
    python run.py --type react "问题" # 指定 Agent 类型执行（关闭智能路由）
    python run.py --stream "问题"     # SSE 流式输出（token 级实时）
    python run.py --session <id>      # 会话持久化（断点续聊）
    python run.py --list              # 列出可用 Agent 类型
    python run.py --demo              # 演示模式（无需 LLM 配置）

交互会话命令:
    /help        显示帮助
    /agents      列出已注册 Agent
    /type <名称> 切换当前 Agent（关闭智能路由）
    /auto        切换智能路由模式开关
    /status      查看当前 Agent 元信息
    /image <路径> 分析图片（支持 jpg/png/gif/bmp/webp）
    /img <路径>   /image 的别名
    /file <路径>  读取文档（PDF/Word/Excel/PPT/文本/代码等）
    /exit        退出

说明: 真实 Agent 需要 .env 中的 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL；
未配置时自动降级为演示模式（echo 等无 LLM Agent），保证运行文件始终可用。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.framework.agent_framework import (
    AgentFactory, AgentRegistry, BaseAgent, AgentAdapter, AgentMeta, AgentResult,
)
from src.agents.framework.agent_router import AgentRouter
from src.agents.framework.session_store import SessionStore

# ============================================================
# Agent 类型注册
# ============================================================

# --- 演示 Agent（无 LLM，始终可用）---

class EchoAgent(BaseAgent):
    """回声 Agent：原样返回输入（演示模式用）。"""
    AGENT_TYPE = "echo"
    DESCRIPTION = "回声 Agent：原样返回输入文本（无需 LLM）"
    VERSION = "1.0.0"

    def _execute(self, input_text: str, **kwargs) -> str:
        return f"[回声] {input_text}"

    def capabilities(self) -> list:
        return ["演示", "无 LLM"]


class ReverseAgent(BaseAgent):
    """反转 Agent：反转输入（演示模式用）。"""
    AGENT_TYPE = "reverse"
    DESCRIPTION = "反转 Agent：将输入文本反转（无需 LLM）"
    VERSION = "1.0.0"

    def _execute(self, input_text: str, **kwargs) -> str:
        return input_text[::-1]

    def capabilities(self) -> list:
        return ["演示", "无 LLM"]


class StreamAgent(BaseAgent):
    """SSE 流式 Agent：token 级实时输出（需 LLM）。"""
    AGENT_TYPE = "stream"
    DESCRIPTION = "SSE 流式 Agent（token 级实时输出）"
    VERSION = "1.0.0"

    def _execute(self, input_text: str, **kwargs) -> str:
        return "".join(self._stream(input_text, **kwargs))

    def _stream(self, input_text: str, **kwargs):
        from src.core.llm import practiceLLM
        llm = practiceLLM()
        yield from llm.stream_chunks(
            [{"role": "user", "content": input_text}], temperature=0)

    def capabilities(self) -> list:
        return ["流式输出"]


# --- 真实 Agent 包装（懒加载：构造时才导入并创建，避免启动即失败）---

class _RealAgentAdapter(AgentAdapter):
    """真实 Agent 的适配器基类（子类在 _build_agent() 中各自构造）。"""

    def __init__(self, name=None, session_store=None, session_id=None,
                 **kwargs):
        agent = self._build_agent(name=name or self.AGENT_TYPE, **kwargs)
        super().__init__(agent, name=name,
                         agent_type=self.AGENT_TYPE,
                         description=self.DESCRIPTION,
                         version=self.VERSION,
                         session_store=session_store,
                         session_id=session_id)

    def _build_agent(self, name, **kwargs):  # pragma: no cover - 子类实现
        raise NotImplementedError


class SimpleAgentWrapper(_RealAgentAdapter):
    AGENT_TYPE = "simple"
    DESCRIPTION = "简单聊天 Agent（日常对话 + 基础工具调用）"
    VERSION = "1.0.0"

    def _build_agent(self, name, **kwargs):
        from src.agents.simple_agent import SimpleAgent
        from src.core.llm import practiceLLM
        from src.tools.framework.registry_factory import build_all_tools_registry
        bundle = build_all_tools_registry(include=["basic"])
        return SimpleAgent(name=name, llm=practiceLLM(),
                           tools=bundle.registry.get_tools(),
                           enable_tool_calling=True, **kwargs)


class ReActAgentWrapper(_RealAgentAdapter):
    AGENT_TYPE = "react"
    DESCRIPTION = "ReAct 推理行动 Agent（搜索/终端/VL 多工具）"
    VERSION = "1.0.0"

    def _build_agent(self, name, **kwargs):
        from src.agents.ReAct_agent import ReActAgent
        from src.core.llm import practiceLLM
        from src.tools.framework.registry_factory import build_all_tools_registry
        bundle = build_all_tools_registry(include=["basic", "search", "terminal",
                                                     "vl_media"])
        return ReActAgent(llm=practiceLLM(),
                          tools=bundle.registry.get_tools(),
                          max_steps=8, **kwargs)


class ReflectionAgentWrapper(_RealAgentAdapter):
    AGENT_TYPE = "reflection"
    DESCRIPTION = "反思改进 Agent（搜索/记忆/RAG 事实核查）"
    VERSION = "1.0.0"

    def _build_agent(self, name, **kwargs):
        from src.agents.Reflection_agent import ReflectionAgent
        from src.tools.framework.registry_factory import build_all_tools_registry
        bundle = build_all_tools_registry(include=["search", "rag", "memory", "note"])
        return ReflectionAgent(name=name, tools=bundle.registry.get_tools(), **kwargs)


class PlanAndSolveAgentWrapper(_RealAgentAdapter):
    AGENT_TYPE = "plan_and_solve"
    DESCRIPTION = "规划执行 Agent（搜索/RAG/结构化数据/子代理）"
    VERSION = "1.0.0"

    def _build_agent(self, name, **kwargs):
        from src.agents.plan_and_solve_agent import PlanAndSolveAgent
        from src.core.llm import practiceLLM
        from src.tools.framework.registry_factory import build_all_tools_registry
        bundle = build_all_tools_registry(include=["search", "rag", "structured_data",
                                                     "subagent"])
        return PlanAndSolveAgent(llm=practiceLLM(), name=name,
                                 tools=bundle.registry.get_tools(), **kwargs)


class FunctionCallAgentWrapper(_RealAgentAdapter):
    AGENT_TYPE = "function_call"
    DESCRIPTION = "原生函数调用 Agent（全量工具 + 角色过滤）"
    VERSION = "1.0.0"

    def _build_agent(self, name, **kwargs):
        from src.agents.function_call_agent import FunctionCallAgent
        from src.core.llm import practiceLLM
        from src.tools.framework.registry_factory import (
            build_all_tools_registry, build_role_tool_filter)
        # 全量装配（analyst 角色）+ 角色过滤视图 → 传给 Agent 的工具列表
        bundle = build_all_tools_registry(tool_filter=build_role_tool_filter(),
                                          owner="analyst")
        tools = bundle.registry.get_tools(owner="analyst")
        if bundle.failures:
            print(f"⚠️ FunctionCallAgent 工具装配部分失败: {bundle.failures}")
        return FunctionCallAgent(llm=practiceLLM(), name=name,
                                 tools=tools, **kwargs)


class TreeOfThoughtAgentWrapper(_RealAgentAdapter):
    AGENT_TYPE = "tree_of_thought"
    DESCRIPTION = "Tree-of-Thought 搜索 Agent"
    VERSION = "1.0.0"

    def _build_agent(self, name, **kwargs):
        from src.agents.TreeOfThought_agent import TreeOfThoughtAgent
        return TreeOfThoughtAgent(name=name, **kwargs)


# ============================================================
# 启动器
# ============================================================

def _llm_available() -> bool:
    """检查 .env 是否配置了 LLM。"""
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.getenv("LLM_MODEL_ID") and os.getenv("LLM_API_KEY")
                and os.getenv("LLM_BASE_URL"))


def build_agents(demo_only: bool = False) -> tuple:
    """构建 AgentFactory 与 AgentRegistry。

    Returns:
        (factory, registry, default_type)
    """
    factory = AgentFactory()
    registry = AgentRegistry()

    # 演示 Agent 始终注册
    factory.register("echo", EchoAgent, "回声（无 LLM）")
    factory.register("reverse", ReverseAgent, "反转（无 LLM）")

    if demo_only or not _llm_available():
        default = "echo"
        if not demo_only:
            print("⚠️  未检测到 LLM 配置（LLM_MODEL_ID/LLM_API_KEY/LLM_BASE_URL），"
                  "已降级为演示模式；请配置 .env 后使用真实 Agent。")
        else:
            print("💡 演示模式（--demo），仅注册无 LLM 的 Agent。")
    else:
        factory.register("simple", SimpleAgentWrapper, "简单聊天 Agent")
        factory.register("react", ReActAgentWrapper, "ReAct 推理行动 Agent")
        factory.register("reflection", ReflectionAgentWrapper, "反思改进 Agent")
        factory.register("plan_and_solve", PlanAndSolveAgentWrapper, "规划执行 Agent")
        factory.register("function_call", FunctionCallAgentWrapper, "原生函数调用 Agent")
        factory.register("tree_of_thought", TreeOfThoughtAgentWrapper, "ToT 搜索 Agent")
        factory.register("stream", StreamAgent, "SSE 流式 Agent（token 级）")
        default = "simple"

    return factory, registry, default


def activate(factory: AgentFactory, registry: AgentRegistry,
             agent_type: str, name: str = None,
             session_store: SessionStore = None,
             session_id: str = None) -> BaseAgent:
    """激活（创建并注册）一个 Agent（可选会话持久化）。"""
    agent = factory.create(agent_type, name=name or f"{agent_type}_agent",
                           session_store=session_store, session_id=session_id)
    key = registry.register(agent, agent_type)
    print(f"✅ 已激活 Agent: {agent}（注册名: {key}）")
    return agent


def route_and_execute(registry: AgentRegistry, factory: AgentFactory,
                      router: AgentRouter, question: str, show_route: bool = True,
                      session_store: SessionStore = None,
                      session_id: str = None) -> str:
    """智能路由并执行：返回路由到的 agent_type。"""
    decision = router.route(question)
    if show_route:
        print(f"🧭 路由决策: {decision.agent_type} "
              f"[{decision.method}] 置信度{decision.confidence:.2f} | {decision.reason}")
    agent = registry.get(decision.agent_type)
    if agent is None:
        try:
            activate(factory, registry, decision.agent_type,
                     session_store=session_store, session_id=session_id)
            agent = registry.get(decision.agent_type)
        except KeyError:
            # 路由目标不可用（如演示模式）→ 回退默认类型
            print(f"⚠️ 路由目标 {decision.agent_type} 未注册，回退 {router.default_type}")
            decision.agent_type = router.default_type
            agent = registry.get(decision.agent_type)
            if agent is None:
                activate(factory, registry, decision.agent_type,
                         session_store=session_store, session_id=session_id)
                agent = registry.get(decision.agent_type)
    result = agent.execute(question)
    if result.status == "success":
        print(f"\n🤖 {decision.agent_type} 回答:\n{result.output}")
    else:
        print(f"❌ 执行失败: {result.error}")
    return decision.agent_type


def run_once(registry: AgentRegistry, agent_type: str, question: str,
             factory: AgentFactory = None, router: AgentRouter = None,
             session_store: SessionStore = None, session_id: str = None):
    """单次执行模式（router 提供时走智能路由）。"""
    if router is not None and factory is not None:
        route_and_execute(registry, factory, router, question,
                          session_store=session_store, session_id=session_id)
        return
    agent = registry.get(agent_type)
    if agent is None:
        print(f"❌ Agent 未激活: {agent_type}；可用: {registry.list_agents()}")
        return
    print(f"🤖 当前 Agent: {agent.name} [{agent_type}]")
    result = agent.execute(question)
    if result.status == "success":
        print(f"\n📝 回答:\n{result.output}")
    else:
        print(f"❌ 执行失败: {result.error}")


def stream_once(registry: AgentRegistry, agent_type: str, question: str):
    """单次流式执行模式（SSE：逐块打印 delta）。"""
    agent = registry.get(agent_type)
    if agent is None:
        print(f"❌ Agent 未激活: {agent_type}；可用: {registry.list_agents()}")
        return
    print(f"🤖 {agent.name} [{agent_type}] 流式回答:\n", flush=True)
    for ev in agent.stream(question):
        if ev.event == "delta":
            print(ev.data, end="", flush=True)
        elif ev.event == "result":
            print(ev.data, end="", flush=True)
        elif ev.event == "error":
            print(f"\n❌ 流式执行失败: {ev.data}")
        elif ev.event == "done":
            print(f"\n\n✅ 完成（耗时 {ev.data.get('duration_ms', 0)}ms）")
    print()


# ============================================================
# 终端图片处理（/image 命令）
# ============================================================

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def _handle_image_cmd(path: str) -> str:
    """分析图片文件，返回文本描述。

    Args:
        path: 图片文件路径

    Returns:
        分析结果文本（成功 = 描述内容，失败 = 错误信息）
    """
    # 1. 展开路径中的引号并检查是否存在
    path = path.strip("\"' ")
    if not os.path.exists(path):
        return f"❌ 文件不存在: {path}"
    if not os.path.isfile(path):
        return f"❌ 路径不是文件: {path}"

    # 2. 检查文件扩展名
    ext = os.path.splitext(path)[1].lower()
    if ext not in _IMAGE_EXTENSIONS:
        return (f"❌ 不支持的文件格式: {ext}，"
                f"支持: {', '.join(sorted(_IMAGE_EXTENSIONS))}")

    # 3. 检查文件大小（限制 10MB）
    fsize = os.path.getsize(path)
    if fsize > 10 * 1024 * 1024:
        return f"❌ 文件过大 ({fsize / 1024 / 1024:.1f}MB)，限制 10MB"

    # 4. 初始化 VL 客户端并分析
    try:
        from src.tools.vl_media.vl_media_tool import VLMediaTool
        tool = VLMediaTool()
        result = tool.run({
            "action": "analyze_image",
            "file_path": path,
            "prompt": "请详细描述这张图片的内容，包括主体、背景、颜色、文字等所有可见信息。",
        })
        return result
    except Exception as e:
        return f"❌ 图片分析失败: {e}"


# ============================================================
# 终端文件读取（/file 命令）
# ============================================================

_SUPPORTED_DOC_EXTENSIONS = {
    ".txt", ".log", ".md", ".markdown",
    ".pdf",
    ".doc", ".docx",
    ".xls", ".xlsx", ".csv",
    ".ppt", ".pptx",
    ".json", ".xml", ".yaml", ".yml",
    ".py", ".js", ".ts", ".java", ".go", ".c", ".cpp", ".rs", ".sql",
}

_MAX_FILE_OUTPUT_CHARS = 5000  # 终端输出截断长度


def _handle_file_cmd(path: str) -> str:
    """读取文档文件，返回文本内容。

    Args:
        path: 文件路径（PDF/Word/Excel/PPT/文本/代码等）

    Returns:
        提取的文本内容（超长自动截断）
    """
    path = path.strip("\"' ")
    if not os.path.exists(path):
        return f"❌ 文件不存在: {path}"
    if not os.path.isfile(path):
        return f"❌ 路径不是文件: {path}"

    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_DOC_EXTENSIONS:
        return (f"❌ 不支持的文件格式: {ext}，"
                f"支持: {', '.join(sorted(_SUPPORTED_DOC_EXTENSIONS))}")

    fsize = os.path.getsize(path)
    if fsize > 50 * 1024 * 1024:
        return f"❌ 文件过大 ({fsize / 1024 / 1024:.1f}MB)，限制 50MB"

    try:
        from src.tools.rag.rag_tool import DocumentLoader
        loader = DocumentLoader()
        doc_meta, blocks = loader.load(path)
    except Exception as e:
        return f"❌ 文件读取失败: {e}"

    # 组装输出
    lines = [f"📄 {doc_meta['name']}"]
    lines.append(f"   类型: {doc_meta['doc_type']}  |  大小: "
                 f"{doc_meta['size'] / 1024:.1f} KB")
    lines.append(f"   转换器: {doc_meta.get('converter', 'unknown')}")
    lines.append(f"   分块数: {len(blocks)}")
    lines.append("")

    total_chars = 0
    for i, block in enumerate(blocks, 1):
        content = block["content"].strip()
        if not content:
            continue
        section = block["metadata"].get("section", "")
        header = f"--- 块 {i}" + (f" [{section}]" if section else "") + " ---"
        lines.append(header)
        lines.append(content)

        total_chars += len(content)
        if total_chars > _MAX_FILE_OUTPUT_CHARS:
            lines.append("")
            lines.append(f"... (已截断，仅显示前 {_MAX_FILE_OUTPUT_CHARS} 字符)")
            break

    return "\n".join(lines)


def repl(registry: AgentRegistry, default_type: str, factory: AgentFactory,
         auto_mode: bool = True, router: AgentRouter = None,
         session_store: SessionStore = None, session_id: str = None):
    """交互式会话（auto_mode 时启用智能路由；session_id 时持久化）。"""
    active = default_type
    auto = auto_mode
    print("\n" + "=" * 60)
    print("🧠 Agent 交互会话已启动（输入 /help 查看命令，/exit 退出）")
    print(f"   智能路由: {'✅ 默认开启' if auto else '❌ 已关闭'}（/auto 切换）"
          + (f" | 会话: {session_id}" if session_id else ""))
    print("=" * 60)
    while True:
        try:
            text = input(f"\n[{active}{'|auto' if auto else ''}] 你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        if not text:
            continue

        if text == "/exit" or text == "/quit":
            print("👋 再见！")
            break
        if text == "/help":
            print("命令: /help /agents /type <名称> /auto /status /exit")
            print("      /image <路径> 或 /img <路径>  分析图片")
            print("      /file <路径>                 读取文档")
            continue
        if text == "/agents":
            print(f"已注册 Agent: {registry.list_agents()}")
            print(f"可用类型: {list(factory.available_types())}")
            continue
        if text == "/sessions":
            if session_store is None:
                print("⚠️ 会话存储未初始化，无法列出会话")
                continue
            sessions = session_store.list_sessions(limit=10)
            if not sessions:
                print("（暂无会话记录）")
            for s in sessions:
                print(f"   {s['session_id']} | {s['agent_type']} | "
                      f"{s['message_count']} 条消息 | {s['updated_at']}")
            continue
        if text == "/auto":
            auto = not auto
            print(f"✅ 智能路由已{'开启' if auto else '关闭'}"
                  f"（当前 Agent: {active}）")
            continue
        if text == "/status":
            agent = registry.get(active)
            if agent:
                m = agent.metadata()
                print(f"当前 Agent: {m.name} | 类型: {m.agent_type} | "
                      f"版本: {m.version}\n描述: {m.description}\n"
                      f"能力: {m.capabilities or '(无)'}")
            continue
        if text.startswith("/type "):
            t = text.split(" ", 1)[1].strip()
            if t not in factory.available_types():
                print(f"❌ 未知类型: {t}；可用: {list(factory.available_types())}")
                continue
            if registry.get(t) is None:
                activate(factory, registry, t)
            active = t
            auto = False
            print(f"✅ 已切换到 Agent 类型: {t}（智能路由已关闭）")
            continue
        if text.startswith("/image ") or text.startswith("/img "):
            path = text.split(" ", 1)[1].strip()
            print(f"🖼️ 正在分析图片: {path}")
            result = _handle_image_cmd(path)
            print(f"\n📝 分析结果:\n{result}")
            continue
        if text.startswith("/file "):
            path = text.split(" ", 1)[1].strip()
            print(f"📄 正在读取文件: {path}")
            result = _handle_file_cmd(path)
            print(f"\n{result}")
            continue
        if text.startswith("/"):
            print(f"❌ 未知命令: {text}（输入 /help 查看命令）")
            continue

        # 智能路由模式：自动选择 Agent
        if auto and router is not None:
            active = route_and_execute(registry, factory, router, text)
            continue

        # 固定 Agent 执行（统一接口）
        agent = registry.get(active)
        result = agent.execute(text) if agent else None
        if result is None:
            print(f"❌ Agent 未激活: {active}")
        elif result.status == "success":
            print(f"\n🤖 {active} 回答:\n{result.output}")
        else:
            print(f"❌ 执行失败: {result.error}")


def main(argv: list) -> int:
    """启动器主入口。"""
    # 参数解析（轻量）
    args = [a for a in argv if not a.startswith("--")]
    flags = set(a for a in argv if a.startswith("--"))
    agent_type = None
    if "--type" in argv:
        idx = argv.index("--type")
        if idx + 1 < len(argv):
            agent_type = argv[idx + 1]
            flags.discard("--type")
            args = [a for a in args if a != agent_type]

    demo_only = "--demo" in flags
    stream_mode = "--stream" in flags
    # 会话持久化：--session <id> 指定（或复用）会话；未指定时无状态
    session_id = None
    if "--session" in argv:
        idx = argv.index("--session")
        if idx + 1 < len(argv):
            session_id = argv[idx + 1]
            flags.discard("--session")
            args = [a for a in args if a != session_id]
    session_store = SessionStore() if session_id else None
    if session_id:
        exists = session_store.get_session(session_id)
        print(f"💾 会话持久化: {session_id}"
              f"{'（恢复历史 ' + str(exists['message_count']) + ' 条消息）' if exists else '（新建）'}")
    else:
        print("💡 提示: 加 --session <id> 可启用断点续聊" if not demo_only else "")

    factory, registry, default = build_agents(demo_only=demo_only)
    # 智能路由：默认开启，--type 可关闭
    router = AgentRouter(factory, default_type=default) if not agent_type else None
    if router:
        print(f"🧭 智能路由已启用（特征收敛域路由，兜底 {default}）")
    else:
        print(f"🧭 智能路由已关闭（指定 --type {agent_type}）")

    if "--list" in flags:
        print("📋 可用 Agent 类型:")
        for t, desc in factory.available_types().items():
            mark = " ✅" if t == default else ""
            print(f"   {t:<16} {desc}{mark}")
        if router:
            print(f"\n🧭 智能路由状态: {router.stats()}")
        return 0

    # 单次执行：流式模式优先
    if args and stream_mode:
        target = agent_type or default
        activate(factory, registry, target,
                 session_store=session_store, session_id=session_id)
        stream_once(registry, target, args[0])
        return 0
    # 单次执行
    if args:
        if router is not None:
            run_once(registry, default, args[0], factory=factory, router=router,
                     session_store=session_store, session_id=session_id)
        else:
            target = agent_type or default
            activate(factory, registry, target,
                     session_store=session_store, session_id=session_id)
            run_once(registry, target, args[0])
        return 0

    target = agent_type or default
    activate(factory, registry, target,
             session_store=session_store, session_id=session_id)
    repl(registry, target, factory, router=router,
         auto_mode=(router is not None),
         session_store=session_store, session_id=session_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# 架构说明

## 分层架构

```
┌─────────────────────────────────────────────────┐
│                  入口层 run.py                   │
│   交互会话 / 单次执行 / 路由 / 流式 / 会话持久化    │
├─────────────────────────────────────────────────┤
│                 Agent 层 agents/                 │
│  统一框架(B/A/Factory/Registry/Adapter) 智能路由   │
│  执行轨迹(TraceLogger) 会话存储(SessionStore)      │
│  simple react reflection plan_and_solve          │
│  function_call tree_of_thought stream           │
├─────────────────────────────────────────────────┤
│                工具层 tools/                      │
│  Tool/ToolParameter/ToolRegistry                │
│  ToolResponse 协议  CircuitBreaker 熔断器         │
│  ToolFilter 过滤   SubAgentTool 子代理委派         │
│  记忆(4类) 终端(只读) 搜索(聚合) RAG(全流程)       │
├─────────────────────────────────────────────────┤
│                知识层 skills/                     │
│  SkillLoader（front-matter + 上下文注入）          │
├─────────────────────────────────────────────────┤
│                基础层 core/                       │
│  practiceLLM（OpenAI 兼容客户端）                  │
│  SSE 流式（SSEEvent / stream_agent）              │
└─────────────────────────────────────────────────┘
```

## 核心数据流

### 一次 Agent 执行（含全部横切能力）

```
用户输入
  → BaseAgent.execute()
      ├─ TraceLogger.start()          # 轨迹开启
      ├─ SessionStore 恢复历史        # 断点续聊（构造时注入 _history）
      ├─ with logger.active(trace)    # 活动轨迹（LLM Hook / 工具 Observer 记录）
      │    └─ _execute()
      │         ├─ practiceLLM（LLM Hook 记录调用）
      │         └─ ToolRegistry.execute_structured()
      │              ├─ ToolFilter 校验（角色权限）
      │              ├─ CircuitBreaker 门控（熔断保护）
      │              ├─ Tool.execute() → ToolResponse
      │              └─ observer 记录工具事件
      ├─ SessionStore 保存 user/assistant
      └─ TraceLogger.finish()         # 轨迹结束 + JSONL 落盘
```

### 智能路由（--auto）

```
请求 → AgentRouter.route()
      ├─ LLM 分类（优先，失败自动回退）
      ├─ 关键词规则（6 类，置信度 0.6-0.95）
      └─ 兜底默认类型
    → 激活目标 Agent → 执行
```

### SSE 流式（--stream）

```
请求 → agent.stream()
      ├─ meta 事件（Agent 元信息）
      ├─ delta×N（token 级，_stream 生成器）或 result（非流式回退）
      └─ done/error 事件
    → SSEClientStream 格式化 → HTTP text/event-stream
```

## 关键设计决策

1. **ToolResponse 兼容旧接口**：`str(ToolResponse) == output`，历史 `Tool.run()` 字符串返回不受影响；结构化数据经 `data` 字段传递。
2. **熔断器双模式**：`call()`（异常式）与 `check() + record_success/failure`（显式门控，适配错误响应场景）。
3. **TraceLogger 挂接**：`install_llm_hook()` 包装 practiceLLM；`tool_observer()` 挂 ToolRegistry；BaseAgent.execute 自动 start/finish 并维持活动轨迹上下文。
4. **会话持久化三层接入**：BaseAgent 自动保存每轮；AgentAdapter 恢复被包装 Agent 的内部历史；run.py `--session` 复用会话。
5. **子代理深度保护**：threading.local 嵌套计数，超过 3 层拒绝委派，防止无限递归。
6. **Skills 知识外化**：SKILL.md（front-matter + 正文）→ `inject()` 注入系统提示词 → 运行时经 SkillTool 取用。

## 可观测性闭环

| 维度 | 组件 | 输出 |
|---|---|---|
| 执行轨迹 | TraceLogger | 事件链（LLM/工具/步骤）+ 失败链 + JSONL |
| 上下文统计 | ContextMonitor | 构建耗时/体积/截断/缓存命中（JSONL） |
| 会话状态 | SessionStore | SQLite 会话与消息 |

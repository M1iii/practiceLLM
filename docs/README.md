# practice — 多 Agent 框架项目

一个从零构建的多 Agent 实践项目：统一 Agent 框架 + 工具系统 + 记忆/上下文 + 可观测性 + 工程化。

## 快速开始

```bash
# 安装依赖（venv 已配置时跳过）
uv sync                 # 或 pip install -r 依赖

# 配置 LLM（.env）
# LLM_MODEL_ID=deepseek-v4-flash
# LLM_API_KEY=sk-xxx
# LLM_BASE_URL=https://api.deepseek.com

# 交互会话（默认激活 simple Agent）
python run.py

# 常用模式
python run.py "你的问题"            # 单次执行
python run.py --type react "问题"   # 指定 Agent 类型
python run.py --auto "问题"         # 智能路由（自动选择 Agent）
python run.py --stream "问题"       # SSE 流式输出
python run.py --session s1          # 会话持久化（断点续聊）
python run.py --demo                # 演示模式（无需 LLM）
```

## 目录结构

```
practice/
├── core/        # 基础设施：practiceLLM + SSE 流式输出
├── agents/      # Agent 层：统一框架 / 路由 / 轨迹 / 会话 / 6 种 Agent
├── tools/       # 工具系统：Tool/Registry/Response/熔断/过滤/子代理 + 记忆/终端/搜索/RAG
├── skills/      # Skills 知识外化（SkillLoader + 内置技能）
├── tests/       # pytest 单元测试（51 个用例，无网络依赖）
├── examples/    # 示例代码
├── docs/        # 文档
├── run.py       # 运行文件（交互/单次/路由/流式/会话）
└── CHANGELOG.txt
```

## 核心能力

| 模块 | 说明 |
|---|---|
| Agent 统一框架 | BaseAgent 接口 + AgentFactory + AgentRegistry + AgentAdapter |
| 智能路由 | LLM 分类 + 关键词规则两级路由（--auto） |
| 工具系统 | ToolResponse 协议 + 熔断器 + 工具过滤 + 子代理委派 |
| 记忆与上下文 | 四类记忆（事实/事件/想法/工具）、NoteTool、GSSC 上下文流水线 |
| 可观测性 | TraceLogger（执行轨迹）+ ContextMonitor（上下文统计） |
| 会话持久化 | SQLite 断点续聊（--session） |
| SSE 流式 | token 级实时输出（--stream） |
| Skills | 知识外化，按需注入 Agent 上下文 |

## 文档

- [架构说明](architecture.md)
- [运行文件用法](usage.md)

## 测试

```bash
python -m pytest tests/ -q     # 51 passed
```

## 详细演进

参见 [CHANGELOG.txt](../CHANGELOG.txt)（60 条迭代记录）。

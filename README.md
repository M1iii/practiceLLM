# practiceLLM — 多 Agent 框架实践项目

一个从零构建的多 Agent 实践项目，涵盖 **6 种思考范式**、**14 个工具**、**四层记忆系统**、**RAG 检索增强**、**多模态分析**、**智能路由** 和完整的工程化（可观测性 / 会话持久化 / 流式输出 / Docker 容器化 / LlamaIndex 集成）。

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                    入口层 run.py                          │
│   交互会话 / 单次执行 / 智能路由 / 流式 / 会话持久化       │
├──────────────────────────────────────────────────────────┤
│                Agent 层 src/agents/                       │
│   统一框架 (BaseAgent/Factory/Registry/Adapter)           │
│   智能路由 (收敛域分类 + 特征提取器)                       │
│   执行轨迹 (TraceLogger) / 会话存储 (SessionStore)         │
│   自动记忆引擎 (MemoryEngine)                             │
│   simple / react / reflection / plan_and_solve            │
│   function_call / tree_of_thought                        │
├──────────────────────────────────────────────────────────┤
│               工具层 src/tools/                            │
│   框架: Tool/ToolResponse/熔断器/过滤/子代理/一键装配       │
│   上下文: ContextBuilder / GSSC 流水线 / Monitor           │
│   记忆: 四类记忆 (Working/Episodic/Semantic/Perceptual)    │
│   终端: 安全终端 / 协同终端 (SmartTerminal)                │
│   搜索: 博查 + Tavily 双后端聚合                          │
│   RAG: 自研 RAG 工具 + LlamaIndex 集成管线                │
│   结构化数据: 自然语言→SQL 查询                            │
│   多模态 VL: 图像/视频分析 (qwen-vl-plus)                 │
├──────────────────────────────────────────────────────────┤
│               知识层 src/skills/                           │
│   SkillLoader (front-matter + 上下文注入)                  │
│   内置技能: code_review / rag_practice                    │
├──────────────────────────────────────────────────────────┤
│               基础层 src/core/                             │
│   practiceLLM (OpenAI 兼容客户端)                          │
│   SafeFullCache 缓存层 / SSE 流式输出                      │
│   StorageBackend (MySQL/SQLite/PostgreSQL+pgvector)       │
└──────────────────────────────────────────────────────────┘
```

## 核心特性

### 6 种 Agent 思考范式

| 范式 | 说明 | 适用场景 |
|---|---|---|
| **Simple** | 简单聊天 + 可选工具调用 | 日常对话、轻量问答 |
| **ReAct** | Thought-Action 推理循环 | 搜索验证、多步骤推理 |
| **Reflection** | 生成 → 评分 → 优化循环 | 写作优化、质量审查 |
| **PlanAndSolve** | 先分解计划再逐步执行 | 复杂任务拆解、项目管理 |
| **FunctionCall** | 原生 OpenAI Function Calling | 工具密集型任务、全量工具 |
| **TreeOfThought** | 多路径 Beam Search 推理 | 创造性问题、多方案探索 |

### 智能路由

基于 **收敛域分类** 的自动路由引擎，从 4 个维度（任务复杂度、外部知识需求、迭代需求、输出结构化程度）提取特征向量，通过收敛域匹配自动选择最合适的 Agent 范式。

- 特征提取器：基于关键词 + 长度 + 多问句检测的轻量级分类
- 收敛域映射：6 个范式各自定义在 4 维空间中的收敛区间
- 软边界：SOFT_MARGIN=0.3，边界附近的查询按优先级和置信度裁决
- 默认开启，无需手动指定 `--auto`

### 14 个工具

| 工具 | 说明 |
|---|---|
| Calculator | 数学计算 |
| Time | 当前时间查询 |
| Weekday | 星期/日期查询 |
| Search / AdvancedSearch | 博查 + Tavily 双搜索引擎 |
| TerminalTool | 安全终端（只读白名单 + 沙箱隔离） |
| MemoryTool | 四类记忆（Working/Episodic/Semantic/Perceptual） |
| NoteTool | 结构化笔记（Markdown + YAML Front Matter） |
| RagTool | 自研 RAG 工具（文档加载/分块/混合检索/Qwen QA） |
| LlamaIndexRAGTool | LlamaIndex 集成（PGVectorStore + BM25 + MQE + HyDE） |
| StructuredDataTool | 自然语言 → SQL，查询结构化数据 |
| VLMediaTool | 图像/视频/混合多模态分析（qwen-vl-plus） |
| Skill | 技能知识调用 |
| SubAgent | 子代理委派 |

### 四层记忆系统

- **WorkingMemory**：短期工作记忆，容量限制 + TTL 过期 + TF-IDF 混合检索
- **EpisodicMemory**：情景记忆，SQLite 或 PostgreSQL 持久化 + 会话级索引
- **SemanticMemory**：语义记忆，知识图谱 + 嵌入向量检索
- **PerceptualMemory**：感知记忆，多模态存储（text/image/audio/video/code/document/data）+ 跨模态检索

### 上下文构建 (GSSC 流水线)

四阶段上下文构建管道：**Gather**（多源汇集）→ **Select**（评分选择）→ **Structure**（结构化模板）→ **Compress**（超限压缩），支持全程缓存 + 可观测性日志。

### RAG 检索增强

支持两款 RAG 实现：
- **自研 RagTool**：文档加载（MarkItDown 转换引擎）→ Token 级智能分块 → 混合检索（稠密+稀疏）→ MQE 多查询扩展 → HyDE 假设文档嵌入 → Qwen 增强问答
- **LlamaIndexRAGTool**：PGVectorStore + BM25 混合检索 + QueryFusionRetriever + 结构化索引（元数据抽取 + GIN 过滤 + 先筛后查）+ 智能查询路由

### 多模态媒体分析

- 图像分析：单张/多张图片分析（qwen-vl-plus）
- 视频分析：ffmpeg/OpenCV 帧提取 + 均匀采样 + 逐帧分析
- 混合分析：文本 + 图片联合输入
- 终端 `/image` 命令直接发送图片路径

### 工程化能力

- **可观测性**：TraceLogger（执行轨迹 JSONL）、ContextMonitor（上下文构建统计）
- **会话持久化**：SQLite 断点续聊（`--session`）
- **SSE 流式输出**：token 级实时输出（`--stream`）
- **熔断器**：三态状态机（CLOSED/OPEN/HALF-OPEN），工具级熔断保护
- **工具过滤**：基于角色的白名单/黑名单策略
- **子代理委派**：嵌套深度保护，防止无限递归
- **Skills 知识外化**：SKILL.md（front-matter + 正文）按需注入
- **自动记忆引擎**：重要性自适应 + 自动整合 + 知识抽取 + 定期遗忘
- **Docker 容器化**：一键部署，含 PostgreSQL pgvector 支持

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/M1iii/practiceLLM.git
cd practiceLLM

# 安装依赖（推荐使用 uv）
pip install uv
uv sync

# 或使用 pip
pip install -e .
```

### 配置 LLM

复制 `.env` 文件并配置至少一个 LLM 后端：

```bash
# 必须配置（至少一个）
# DeepSeek
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL_ID=deepseek-v4-flash

# 或阿里云百炼（用于 RAG、嵌入、VL 多模态）
DASHSCOPE_API_KEY=sk-xxx
EMBEDDING_MODEL=qwen3-vl-embedding
RAG_LLM_MODEL=qwen-plus
```

未配置 LLM 时自动降级为演示模式，运行文件始终可用。

### 运行

```bash
# 交互式会话（智能路由默认开启）
python run.py

# 单次执行
python run.py "介绍一下 RAG 技术"

# 指定 Agent 类型
python run.py --type react "搜索最近的 AI 新闻"

# SSE 流式输出
python run.py --stream "用一句话介绍机器学习"

# 会话持久化（断点续聊）
python run.py --session my_session

# 演示模式（无需 LLM）
python run.py --demo
```

### 交互式命令

```
/help         显示帮助
/agents       列出已注册 Agent
/type <名称>  切换当前 Agent
/status       查看当前 Agent 元信息
/auto         切换智能路由模式开关
/image <路径> 分析图片（支持 jpg/png/gif/bmp/webp）
/img <路径>   同上
/file <路径>  读取文档（PDF/Word/Excel/PPT/代码等）
/sessions     列出历史会话
/exit         退出
```

## 项目结构

```
practice/
├── src/                          # 主源代码
│   ├── agents/                   # Agent 实现
│   │   ├── framework/            # Agent 基础设施层
│   │   │   ├── agent_framework.py    # BaseAgent / AgentFactory / AgentRegistry
│   │   │   ├── agent_router.py       # 智能路由（收敛域分类）
│   │   │   ├── agent_trace.py        # 执行轨迹可观测性
│   │   │   ├── feature_extractor.py  # 路由特征提取器
│   │   │   ├── memory_engine.py      # 自动记忆引擎
│   │   │   └── session_store.py      # 会话持久化 (SQLite)
│   │   ├── simple_agent.py
│   │   ├── ReAct_agent.py
│   │   ├── Reflection_agent.py
│   │   ├── plan_and_solve_agent.py
│   │   ├── function_call_agent.py
│   │   └── TreeOfThought_agent.py
│   ├── core/                     # 核心基础设施
│   │   ├── llm.py                # LLM 客户端 (practiceLLM)
│   │   ├── cache.py              # SafeFullCache 缓存层
│   │   ├── storage.py            # 存储后端 (MySQL/SQLite/PostgreSQL)
│   │   └── streaming.py          # SSE 流式输出
│   ├── tools/                    # 工具系统
│   │   ├── framework/            # 工具框架
│   │   │   ├── tool_system.py         # 统一工具系统
│   │   │   ├── tool_response.py       # 结构化返回协议
│   │   │   ├── circuit_breaker.py     # 熔断器
│   │   │   ├── tool_filter.py         # 工具过滤
│   │   │   ├── sub_agent_tool.py      # 子代理委派
│   │   │   ├── registry_factory.py    # 一键装配工厂
│   │   │   ├── tool_chain_manager.py  # 工具链管理器
│   │   │   └── async_tool_executor.py # 异步工具执行器
│   │   ├── context/              # 上下文构建
│   │   │   ├── context_builder.py
│   │   │   ├── context_monitor.py
│   │   │   └── gs_sc_pipeline.py
│   │   ├── memory/               # 记忆系统
│   │   │   ├── memory_tool.py
│   │   │   ├── modules/          # 模块化记忆包
│   │   │   │   ├── base.py
│   │   │   │   ├── embedding_client.py
│   │   │   │   ├── working_memory.py
│   │   │   │   ├── episodic_memory.py
│   │   │   │   ├── semantic_memory.py
│   │   │   │   ├── perceptual_memory.py
│   │   │   │   └── memory_manager.py
│   │   │   ├── note_tool.py
│   │   │   └── note_context_bridge.py
│   │   ├── terminal/             # 终端工具
│   │   │   ├── terminal_tool.py
│   │   │   └── smart_terminal.py
│   │   ├── search/               # 搜索工具
│   │   │   ├── advanced_search_tool.py
│   │   │   └── search_registry.py
│   │   ├── rag/                  # RAG 检索增强
│   │   │   ├── rag_tool.py
│   │   │   ├── llamaindex_tool.py
│   │   │   └── llamaindex/       # LlamaIndex 集成
│   │   │       ├── adapters.py
│   │   │       ├── retrievers.py
│   │   │       ├── transforms.py
│   │   │       ├── structured_index.py
│   │   │       ├── query_parser.py
│   │   │       └── ...
│   │   ├── structured_data/      # 结构化数据查询
│   │   │   ├── db_schema.py
│   │   │   ├── sql_executor.py
│   │   │   └── structured_data_tool.py
│   │   └── vl_media/             # 多模态媒体分析
│   │       ├── vl_client.py
│   │       └── vl_media_tool.py
│   └── skills/                   # 知识外化系统
│       ├── loader.py
│       ├── skill_tool.py
│       └── builtin/
├── config/                       # 基础设施配置
│   ├── Dockerfile
│   └── docker-compose.yml
├── docker/
│   └── postgres-pgvector/        # PostgreSQL + pgvector 镜像
├── docs/                         # 文档
├── examples/                     # 示例代码（5 个可运行示例）
├── tests/                        # 测试
│   ├── unit/                     # 单元测试
│   ├── conftest.py
│   └── test_*.py                 # 模块测试
├── .env                          # 环境变量配置
├── run.py                        # Agent 启动器
├── pyproject.toml                # 项目依赖
└── CHANGELOG.txt                 # 102 次迭代变更记录
```

## 配置参考

核心环境变量（`.env`）：

| 变量 | 说明 | 必需 |
|---|---|---|
| `LLM_API_KEY` | LLM API 密钥 | 是（至少一个 LLM） |
| `LLM_BASE_URL` | LLM 接口地址 | 是 |
| `LLM_MODEL_ID` | 模型名 | 是 |
| `DASHSCOPE_API_KEY` | 阿里云百炼密钥（RAG/嵌入/VL） | 推荐 |
| `EMBEDDING_MODEL` | 嵌入模型（默认 `qwen3-vl-embedding`） | 否 |
| `RAG_LLM_MODEL` | RAG 问答模型（默认 `qwen-plus`） | 否 |
| `BOCHA_API_KEY` | 博查搜索密钥 | 否 |
| `TAVILY_API_KEY` | Tavily 搜索密钥 | 否 |
| `PG_HOST/PG_PORT/PG_USER/PG_PASSWORD/PG_DATABASE` | PostgreSQL 连接配置 | 否 |
| `MYSQL_HOST/MYSQL_PORT/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DATABASE` | MySQL 连接配置 | 否 |
| `STORAGE_BACKEND` | 存储后端类型（`sqlite`/`mysql`） | 否 |

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -q

# 运行特定测试
python -m pytest tests/test_routing.py -q
python -m pytest tests/unit/test_memory_engine.py -q
python -m pytest tests/unit/test_structured_data_tool.py -q
```

## Docker 部署

```bash
# 构建并启动（含 PostgreSQL pgvector）
docker compose -f config/docker-compose.yml up -d

# 交互式运行
docker compose -f config/docker-compose.yml run --rm practice-llm

# 单独构建 pgvector 镜像
docker build -f docker/postgres-pgvector/Dockerfile -t postgres-pgvector .
```

## 文档

- [架构说明](docs/architecture.md) — 分层架构、数据流、设计决策
- [运行文件用法](docs/usage.md) — run.py 全部模式详解
- [RAG 技术测试](docs/rag_test.md) — RAG 概念、分块策略、高级 RAG 技术
- [CHANGELOG](CHANGELOG.txt) — 102 次迭代变更记录
- [示例代码](examples/README.md) — 5 个可运行示例

## 示例

```bash
# Agent 统一框架
python examples/01_agent_framework.py

# 工具系统（ToolResponse + 熔断 + 过滤 + 子代理）
python examples/02_tools.py

# 会话持久化 + 执行轨迹
python examples/03_session_and_trace.py

# Skills 知识外化
python examples/04_skills.py

# SSE 流式输出（HTTP）
python examples/05_streaming.py
```

## 技术栈

- **Python** >= 3.12
- **LLM 后端**：DeepSeek API / 阿里云百炼 Qwen
- **嵌入模型**：qwen3-vl-embedding（多模态，2560 维）
- **向量数据库**：PostgreSQL + pgvector（JSONB 元数据）
- **RAG 框架**：LlamaIndex（PGVectorStore / BM25 / QueryFusionRetriever）
- **文档转换**：MarkItDown（24 种格式 → Markdown）
- **多模态 VL**：qwen-vl-plus（图像 / 视频分析）
- **搜索后端**：博查搜索 / Tavily
- **存储后端**：SQLite / MySQL / PostgreSQL
- **容器化**：Docker + docker-compose
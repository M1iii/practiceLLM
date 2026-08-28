# run.py 用法

## 模式一览

| 命令 | 说明 |
|---|---|
| `python run.py` | 交互式会话（激活默认 Agent：有 LLM 时 simple，否则 echo） |
| `python run.py "问题"` | 单次执行（默认 Agent） |
| `python run.py --type react "问题"` | 指定 Agent 类型执行 |
| `python run.py --auto "问题"` | 智能路由：自动选择最合适的 Agent |
| `python run.py --stream "问题"` | SSE 流式输出（token 级实时） |
| `python run.py --session <id> "问题"` | 会话持久化（指定/复用会话） |
| `python run.py --list` | 列出可用 Agent 类型 |
| `python run.py --demo` | 演示模式（无需 LLM 配置） |

## Agent 类型

| 类型 | 说明 | 需要 LLM |
|---|---|---|
| `simple` | 简单聊天（日常对话 + 可选工具） | 是 |
| `react` | ReAct 推理行动 | 是 |
| `reflection` | 反思改进（生成→评分→优化循环） | 是 |
| `plan_and_solve` | 规划执行（先分解再逐步执行） | 是 |
| `function_call` | 原生函数调用 | 是 |
| `tree_of_thought` | ToT 多路径搜索 | 是 |
| `stream` | SSE 流式（token 级） | 是 |
| `echo` / `reverse` | 演示（无 LLM） | 否 |

## 交互会话命令

```
/help         显示帮助
/agents       列出已注册 Agent
/type <名称>  切换当前 Agent
/auto         切换智能路由模式
/sessions     列出历史会话（需 --session 启用存储）
/status       查看当前 Agent 元信息
/exit         退出
```

## 示例

```bash
# 智能路由：写作/规划/推理请求自动分派到对应 Agent
python run.py --auto "帮我规划一个三步骤的发布计划"

# 断点续聊：两次运行共享同一会话（第 2 次恢复历史）
python run.py --session my_s1 "你好，介绍一下你自己"
python run.py --session my_s1 "你刚才介绍过了吗"

# 流式输出
python run.py --stream --type stream "用一句话介绍 RAG"
```

## 环境变量（.env）

| 变量 | 说明 | 示例 |
|---|---|---|
| `LLM_MODEL_ID` | 模型名 | `deepseek-v4-flash` |
| `LLM_API_KEY` | API 密钥 | `sk-xxx` |
| `LLM_BASE_URL` | 接口地址 | `https://api.deepseek.com` |

未配置 LLM 时自动降级为演示模式，运行文件始终可用。

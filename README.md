# LangGraph 多智能体客服系统

这是一个基于 LangGraph 的多智能体客服 MVP 项目。系统通过 `supervisor` 主管节点识别用户意图，再把任务路由给不同专业 Agent 处理，当前已完成意图识别、任务路由、工具调用、RAG 检索增强和 Gradio 页面展示。

项目优先完成多智能体客服的核心闭环：

```text
用户输入 -> supervisor 分类 -> 条件路由 -> 专业 Agent 执行 -> 结果回传 -> 页面展示
```

### 当前版本 2.0

当前版本已经完成多智能体客服闭环、真实大模型接入、外部工具调用、RAG 检索增强和页面可观测展示。

1. 用户可以在 Gradio 页面输入自然语言问题。
2. 系统会先由 `supervisor` 识别意图，并把问题路由到对应专业 Agent。
3. 旅游规划 Agent 已接入高德 MCP，可以把外部地图工具纳入 Agent 执行流程。
4. 对联生成 Agent 已接入本地样本库和 Embedding 相似检索，可以参考 Top 3 相似对联再生成回答。
5. Redis 不可用时，系统会自动降级为本地 NumPy 相似度计算，保证 MVP 主链路不依赖完整 Redis Stack 环境。
6. 页面已经展示“意图识别结果、RAG Top 3 参考内容、最终回答”，方便观察多智能体执行过程。
7. 项目已经按 `agents/`、`graph/`、`services/`、`config/` 拆分模块，后续扩展新 Agent 更清晰。

💗 亮点：当前版本已经跑通 **用户输入 -> 意图识别 -> 条件路由 -> MCP 工具 / RAG 检索 -> 专业 Agent 生成 -> 页面可观测展示** 的完整路径，项目中用到 LangGraph、多智能体协作、工具调用和检索增强等，有不错的完整度。


——————————————————————————————————————————————————

## 核心功能

### 1. Supervisor 意图识别

`supervisor_node` 会把用户输入分类为：
- `travel`：旅游规划问题。
- `couplet`：对联生成问题。
- `joke`：笑话生成问题。
- `other`：暂不支持的问题。

分类结果会进入 LangGraph 条件路由，决定下一步由哪个 Agent 处理。

### 2. 旅游规划 Agent

`travel_node` 负责旅游路线规划，并通过 `MultiServerMCPClient` 接入高德地图 MCP；节点内部使用 ReAct Agent，让模型可以在回答前调用外部地图工具。

### 3. 对联生成 Agent

`couplet_node` 负责根据用户输入的上联生成下联。
- 从 `resource/couplettest.csv` 读取对联样本。
- 优先使用 Redis VectorStore 做相似对联检索。
- Redis 不可用或缺少 RediSearch 能力时，自动降级为本地内存相似度检索。
- 将相似样本放进提示词，让模型结合参考内容生成下联。

### 4. 笑话 Agent

`joke_node` 负责处理讲笑话类问题。它使用独立提示词约束角色和回答长度。

### 5. Gradio 页面

`app.py` 提供本地 Gradio 页面，展示：
- 意图识别结果。
- RAG Top 3 参考内容。
- 智能体最终回复。


——————————————————————————————————————————————————

## 版本说明
### V1：多智能体路由基础版 ✅

目的：先跑通 LangGraph 多智能体任务分发闭环，让系统不再把所有问题都交给同一个提示词处理。

1. 使用 `StateGraph` 构建多智能体执行图。
2. 增加 `supervisor_node`，负责识别用户意图。
3. 根据意图把问题分发给 `travel_node`、`couplet_node`、`joke_node` 或 `other_node`。
4. 各专业 Agent 处理完成后回到 supervisor 收束流程。
5. 使用 Gradio 提供本地页面入口，方便直接体验。

💗 亮点：已经验证一条完整路径：**用户输入 -> 意图识别 -> 条件路由 -> 专业 Agent 执行 -> 最终回答**。

### V2：工具调用与RAG增强版 ✅

目的：让 Agent 不只依赖大模型直接生成，而是能结合外部工具和本地数据完成更具体的任务。

1. 旅游规划 Agent 接入高德 MCP 工具，让模型可以使用外部地图能力。
2. 对联生成 Agent 接入本地对联样本库，先检索相似参考，再生成下联。
3. 支持 Redis VectorStore 做向量检索。
4. Redis 不可用或缺少 RediSearch 时，自动降级为本地 NumPy 余弦相似度计算。
5. 页面展示 supervisor 意图识别结果和 RAG Top 3 参考内容。
6. 增加资源路径、模型配置和 trace 展示相关测试。

💗 亮点：项目形成了 **LangGraph + LLM + MCP + RAG + 页面可观测** 的完整 MVP 闭环。

### V3：工程化与稳定性优化（待完成）

目的：让项目从可运行 MVP 继续升级为更容易复现、更容易维护、更适合展示的工程化项目。

1. 补 `requirements.txt`，固定依赖版本，让环境可以稳定复现。
2. 把真实 Key 读取升级为“环境变量优先，本地配置文件兜底”。
3. 给 supervisor 分类增加结构化输出或枚举校验，降低模型返回脏文本导致路由失败的概率。
4. 给 MCP 工具调用增加更明确的异常提示，避免外部服务失败时页面只显示底层报错。
5. 增加不依赖真实大模型的 Mock 测试，覆盖路由函数、图构建和 Redis 降级逻辑。
6. 整理一张架构图，放到 `docs/architecture.md`，说明 supervisor、Agent、MCP 和 RAG 的关系。

💗 亮点：V3 的重点不是继续堆 Agent，而是把已经跑通的多智能体闭环打磨成更稳定、更清晰、更容易讲解的项目。



——————————————————————————————————————————————————
## 技术架构

- 编排框架：LangGraph
- Agent 执行：LangGraph Prebuilt ReAct Agent
- 大模型：阿里云百炼通义千问，模型名由 `config/model_config.py` 统一管理
- Embedding：DashScope `text-embedding-v1`
- 工具协议：MCP
- 地图工具：高德地图 MCP
- 向量检索：Redis VectorStore，可降级为 NumPy 内存相似度计算
- 本地页面：Gradio
- 测试框架：unittest

## 目录结构

| 路径 | 作用 |
| --- | --- |
| `app.py` | Gradio 本地页面入口 |
| `agents/` | 各类专业 Agent 节点 |
| `graph/` | LangGraph 状态定义和图构建 |
| `services/` | 模型、MCP 工具、对联检索和页面 trace 处理 |
| `scripts/` | 辅助运行脚本 |
| `config/` | Key 读取、模型配置和资源路径 |
| `resource/` | 本地数据资源 |
| `tests/` | 单元测试 |
| `docs/constitution.md` | 项目开发和维护规则 |

## 本地启动

### 1. 准备环境

建议使用：

- Python 3.11+
- 阿里云百炼 API Key
- 高德地图 MCP API Key
- Redis 或 Redis Stack，可选

### 2. 安装依赖

当前项目还没有固定 `requirements.txt`，可以按实际环境安装核心依赖：

```bash
pip install langgraph langchain-core langchain-community langchain-mcp-adapters langchain-redis redis numpy gradio dashscope
```

### 3. 配置 Key

复制配置模板：

```bash
cp config/Keys.example.json config/Keys.json
```

填入：

```json
{
  "BAILIAN_API_KEY": "你的百炼 API Key",
  "AMAP_API_KEY": "你的高德 MCP API Key"
}
```

真实 `config/Keys.json` 不应该提交到仓库。

### 4. 启动页面

```bash
python3 app.py
```

访问：

```bash
http://127.0.0.1:7863
```

### 5. 常用命令

```bash
# 运行全部测试
python3 -m unittest discover tests

# 检查 Python 语法
python3 -m compileall .
```

加载对联样本到 Redis：

```bash
python3 -m scripts.couplet_loader
```

单独测试对联检索与生成：

```bash
python3 -m scripts.couplet_retrieval
```

## 开发约定

- 新增 Agent 时，必须补 supervisor 分类规则、节点函数和条件路由。
- 新增外部工具时，要说明失败时的兜底策略。
- 新增本地资源时，优先通过 `config/paths.py` 解析路径。
- 真实密钥只能放在本地私密配置或环境变量中。
- 修改核心流程后至少运行 `python3 -m unittest discover tests`。

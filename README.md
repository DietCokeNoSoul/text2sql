# Text2SQL Agent

基于 LangGraph 的智能 SQL 查询代理，支持自然语言到 SQL 的转换，采用 **Skill 模块化架构**自动路由不同复杂度的查询。

## 架构概览

```
主图 (Main Graph)
  └─ query_router ──→ 判断查询类型
        │
        ├─ simple  ──→ Simple Query Skill
        │               list_tables → get_schema → generate_query
        │               → execute → [error?] → fix_query (最多3次重试)
        │
        ├─ complex ──→ Complex Query Skill  
        │               list_tables → get_schema → plan
        │               → execute_step_1 → execute_step_2 → ... → aggregate
        │
        └─ analysis ─→ Data Analysis Skill
                        understand_goal → explore_data → plan_analysis
                        → generate_queries → execute_queries
                        → analyze_results → visualize → generate_report
```

## 功能特性

- **智能路由**：自动识别查询类型（简单 / 复杂 / 分析），100% 准确率
- **Simple Query**：单步 SQL 查询，内置错误自动修复循环（最多 3 次重试）
- **Complex Query**：多步骤规划执行，双塔检索（Milvus + NetworkX Steiner Tree）剪枝 Schema
- **Data Analysis**：8 步分析流水线，生成中文 Markdown 报告 + SVG 图表 + CSV/Excel 导出
- **流式输出**：节点级流式（`run_query`）和 token 级流式（`run_query_streaming`）双模式
- **LLM 响应缓存**：SQLite 本地缓存，相同问题命中缓存时延迟 ~0ms / token 消耗 = 0
- **LLM 自动重试**：网络抖动时指数退避重试（最多 3 次）
- **Schema 缓存**：TTL 缓存（默认 5 分钟），减少重复数据库 Schema 查询
- **列名模糊匹配**：SQL 报错时自动建议相似列名，提升错误修复成功率
- **SQL 安全护栏**：四层防御拦截危险 SQL，支持表访问控制、LIMIT 强制、敏感列检测
- **会话计划追踪**：每个 ComplexQuery / DataAnalysis 会话生成独立 plan.md，记录任务链
- **多轮对话**：通过 `thread_id` 维持会话记忆

## 快速开始

### 1. 配置环境

```bash
# 安装依赖（推荐使用 uv）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

复制环境变量模板并填入真实值：

```bash
cp .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY 等必填项
```

最小配置示例（`.env`）：

```env
DB_URI=sqlite:///Chinook.db
DASHSCOPE_API_KEY=your_api_key_here
LLM_MODEL=qwen-plus
LLM_PROVIDER=tongyi
```

### 2. 运行代理

```bash
# 交互式命令行（默认节点级流式输出）
python main.py

# 启动后输入 'stream' 切换到 token 级实时流式模式
```

### 3. 代码集成

```python
from agent.graph import run_query, run_query_streaming

# 节点级流式（同步，每个节点完成后打印）
result = run_query("查询评分最高的前10家商店", thread_id="session-1")
print(result["final_message"])    # 最终回答
print(result["nodes_visited"])    # 执行的节点列表
print(result["export_files"])     # 导出的 CSV/Excel 文件（DataAnalysis 才有）

# token 级流式（实时打印 LLM token）
result = run_query_streaming("分析各类型音乐的销售趋势", thread_id="session-1")
```

## 项目结构

```
text2sql/
├── main.py                    # 入口（交互式 CLI）
├── .env.example               # 环境变量模板（复制为 .env 并填写）
├── requirements.txt
│
├── agent/                     # 核心组件
│   ├── config.py              # 配置管理（支持 .env，含 CacheConfig）
│   ├── database.py            # 数据库连接 + SchemaCache + SecurityGuard 集成
│   ├── security.py            # SQL 安全护栏（四层防御）
│   ├── tools.py               # SQLToolManager + CachedSchemaTool
│   ├── graph.py               # 图初始化 / run_query() / run_query_streaming()
│   ├── column_index.py        # Milvus 列向量索引（模块级模型单例）
│   ├── schema_graph.py        # NetworkX Schema 图 + Steiner Tree
│   ├── retrieval.py           # 双塔检索协调器
│   ├── session_plan.py        # 会话任务计划追踪（plan.md + plan.json）
│   ├── skill_graph_builder.py # Skill 主图构建器
│   ├── nodes/
│   │   └── common.py          # 可复用节点工厂
│   └── skills/
│       ├── base.py            # BaseSkill 抽象类
│       ├── states.py          # State 定义
│       └── registry.py        # Skill 注册中心
│
├── skills/                    # Skill 实现
│   ├── simple_query/skill.py  # Simple Query（错误修复循环）
│   ├── complex_query/skill.py # Complex Query（双塔检索 + 多步规划）
│   └── data_analysis/
│       ├── skill.py           # Data Analysis（8步，含导出节点）
│       └── chart_generator.py # SVG 图表生成
│
├── report/                    # 输出目录（报告 / 图表 / CSV / Excel）
└── tests/                     # 测试套件
```

## 新功能说明

### 流式输出

支持两种流式模式，可在运行时切换：

| 模式 | 函数 | 说明 |
|------|------|------|
| 节点级流式 | `run_query()` | 每个节点执行完后推送摘要，延迟低 |
| Token 级流式 | `run_query_streaming()` | 实时推送 LLM token，体验接近 ChatGPT |

交互式 CLI 中输入 `stream` 即可切换到 token 级流式模式。

### LLM 响应缓存

相同的问题命中缓存时，响应延迟 ≈ 0ms，token 消耗 = 0。

```env
LLM_CACHE_ENABLED=true           # 开启缓存（默认 true）
LLM_CACHE_BACKEND=sqlite          # 支持 sqlite（未来可扩展 redis）
LLM_CACHE_SQLITE_PATH=.cache/llm_cache.db  # SQLite 缓存文件路径
```

### 查询结果导出

DataAnalysis 技能每次执行后，在 `report/` 目录下自动生成：
- `report_{timestamp}.csv` — 原始查询结果
- `report_{timestamp}.xlsx` — 格式化 Excel 文件（需 openpyxl）
- `report_{timestamp}.md` — 中文 Markdown 分析报告
- `report/charts/` — SVG 可视化图表

### 双塔检索（复杂查询优化）

Complex Query 技能使用**双塔检索**定位相关表结构，大幅减少传入 LLM 的 Schema 体积：

| 检索通道 | 技术 | 说明 |
|----------|------|------|
| 向量塔 | Milvus + SentenceTransformer | 对问题语义进行向量检索，匹配相似列 |
| 图塔 | NetworkX + Steiner Tree | 在 Schema 图中规划跨表连接路径 |

### 会话计划追踪

每个 ComplexQuery 和 DataAnalysis 会话都会在 `report/` 目录生成独立的 `plan_{session_id}.md`，结构示例：

```markdown
# 任务计划 - 2024-01-15 10:30:00

## 任务分解
- [x] 步骤1: 检索相关表结构
- [x] 步骤2: 生成子查询 SQL
- [ ] 步骤3: 汇总结果

## 执行日志
- 10:30:01 — 路由判断: complex_query
- 10:30:02 — 双塔检索命中: tracks, albums, genres
```



## 测试

```bash
# 无需 API Key 的单元测试
pytest tests/test_schema_cache.py
pytest tests/test_column_fuzzy_match.py
pytest tests/test_chart_generation.py
pytest tests/test_report_saving.py
pytest tests/test_security.py
pytest tests/test_cache_and_export.py   # LLM 缓存 + 结果导出（18 个测试）

# 运行全部单元测试
pytest tests/ -k "not integration"

# 需要 API Key 的集成测试
pytest tests/test_router_accuracy.py    # 路由准确率（15 用例，100%）
pytest tests/test_simple_skill.py       # Simple Query 端到端
pytest tests/test_complex_detailed.py   # Complex Query 端到端
pytest tests/test_analysis_detailed.py  # Data Analysis 8步流程
pytest tests/test_main_graph.py         # 主图集成
```

## SQL 安全护栏

代理在执行每条 SQL 前自动经过四层安全检查：

```
Layer 1 — 语句类型控制   只允许 SELECT，拒绝 DROP/DELETE/UPDATE/INSERT 等
Layer 2 — 表访问控制     支持 denylist（黑名单）和 allowlist（白名单）
Layer 3 — 复杂度限制     自动注入/降低 LIMIT，限制 SQL 字符长度
Layer 4 — 结果脱敏       检测 password/token/phone 等敏感列并发出警告
```

通过 `.env` 配置护栏行为：

```env
SECURITY_MAX_ROWS=1000
SECURITY_TABLE_DENYLIST=users,auth_tokens,secrets
SECURITY_TABLE_ALLOWLIST=                    # 留空表示允许所有表
SECURITY_AUDIT_LOG=true
SECURITY_AUDIT_LOG_FILE=logs/sql_audit.jsonl
```

## 技术栈

| 组件 | 版本 | 说明 |
|------|------|------|
| LangGraph | ≥ 1.0.0 | 多 Skill 子图编排 |
| LangChain | ≥ 1.0.0 | LLM 抽象层，with_retry 自动重试 |
| LLM | 通义千问 qwen-plus (ChatTongyi) | 默认模型，可通过 LLM_PROVIDER 切换 |
| Milvus | 2.4.x | 列向量索引（双塔检索向量端） |
| NetworkX | ≥ 3.x | Schema 关系图 + Steiner Tree 路径规划 |
| SentenceTransformer | ≥ 2.x | 列语义向量编码（模块级单例缓存） |
| sqlglot | ≥ 23.x | SQL AST 解析（安全护栏） |
| openpyxl | ≥ 3.1 | Excel 导出支持 |
| 数据库 | MySQL 8.0 / SQLite | SQLAlchemy URI 配置 |
| Python | ≥ 3.10 | |

## 配置项

> 完整变量列表见 [.env.example](.env.example)

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DB_URI` | `sqlite:///Chinook.db` | 数据库连接字符串 |
| `DASHSCOPE_API_KEY` | — | 通义千问 API Key（必填） |
| `LLM_MODEL` | `qwen-plus` | 模型名称 |
| `LLM_PROVIDER` | `tongyi` | LLM 提供商 |
| `LLM_TEMPERATURE` | `0.0` | 生成温度 |
| `LLM_MAX_TOKENS` | — | 最大输出 token 数 |
| `LLM_CACHE_ENABLED` | `true` | 启用 LLM 响应缓存 |
| `LLM_CACHE_BACKEND` | `sqlite` | 缓存后端类型 |
| `LLM_CACHE_SQLITE_PATH` | `.cache/llm_cache.db` | SQLite 缓存文件路径 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `REPORT_DIR` | `report` | 分析报告保存目录 |
| `CHART_DIR` | `report/charts` | 图表保存目录 |
| `MILVUS_URI` | `http://127.0.0.1:19530` | Milvus 服务地址 |
| `MILVUS_TOKEN` | `root:Milvus` | Milvus 认证 token |
| `SECURITY_MAX_ROWS` | `1000` | 最大返回行数（自动注入 LIMIT） |
| `SECURITY_TABLE_DENYLIST` | — | 禁止访问的表（逗号分隔） |
| `SECURITY_TABLE_ALLOWLIST` | — | 允许访问的表（留空=允许全部） |
| `SECURITY_AUDIT_LOG` | `true` | 是否启用审计日志 |
| `SECURITY_AUDIT_LOG_FILE` | — | 审计日志文件路径（JSONL 格式） |

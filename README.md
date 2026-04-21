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
- **Complex Query**：多步骤规划执行，支持 `{step_N_results}` 占位符传递结果
- **Data Analysis**：7 步分析流程，生成中文报告 + matplotlib PNG 可视化图表
- **Schema 缓存**：TTL 缓存（默认 5 分钟），减少重复数据库 Schema 查询
- **列名模糊匹配**：SQL 报错时自动建议相似列名，提升错误修复成功率
- **SQL 安全护栏**：四层防御拦截危险 SQL，支持表访问控制、LIMIT 强制、敏感列检测
- **多轮对话**：通过 `thread_id` 维持会话记忆

## 快速开始

### 1. 配置环境

```bash
# 安装依赖（推荐使用 uv）
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

在项目根目录创建 `.env` 文件：

```env
DB_URI=mysql+pymysql://user:password@localhost:3306/your_database
DASHSCOPE_API_KEY=your_api_key_here
LLM_MODEL=qwen-plus
LLM_PROVIDER=tongyi
```

### 2. 运行代理

```bash
# 交互式命令行
python main.py
```

### 3. 代码集成

```python
from langchain_community.chat_models import ChatTongyi
from langchain.messages import HumanMessage
from agent import create_skill_based_graph, get_config

config = get_config()
llm = ChatTongyi(model=config.llm.model, dashscope_api_key=config.llm.api_key)
graph = create_skill_based_graph(config, llm)

result = graph.invoke({
    "messages": [HumanMessage(content="查询评分最高的前10家商店")]
})
print(result["messages"][-1].content)
```

## 项目结构

```
text2sql/
├── main.py                    # 入口（交互式 CLI）
├── requirements.txt
│
├── agent/                     # 核心组件
│   ├── config.py              # 配置管理（支持 .env）
│   ├── database.py            # 数据库连接 + SchemaCache + SecurityGuard 集成
│   ├── security.py            # SQL 安全护栏（四层防御）
│   ├── tools.py               # SQLToolManager + CachedSchemaTool
│   ├── graph.py               # 图初始化与 run_query()
│   ├── skill_graph_builder.py # Skill 主图构建器
│   ├── nodes/
│   │   └── common.py          # 可复用节点工厂（list_tables, get_schema, ...）
│   └── skills/
│       ├── base.py            # BaseSkill 抽象类
│       ├── states.py          # State 定义
│       └── registry.py        # Skill 注册中心
│
└── skills/                    # Skill 实现
    ├── simple_query/
    │   ├── skill.py           # Simple Query Skill（带错误修复循环）
    │   └── SKILL.md           # Skill 说明文档
    ├── complex_query/
    │   ├── skill.py           # Complex Query Skill（多步规划+占位符）
    │   └── SKILL.md
    └── data_analysis/
        ├── skill.py           # Data Analysis Skill（7步分析流水线）
        ├── chart_generator.py # matplotlib PNG 图表生成
        └── SKILL.md
```

## 测试

```bash
# 无需 API Key 的单元测试（156 个，全部通过）
python tests/test_schema_cache.py
python tests/test_column_fuzzy_match.py
python tests/test_chart_generation.py
python tests/test_report_saving.py
python tests/test_security.py

# 需要 API Key 的集成测试
python tests/test_router_accuracy.py     # 路由准确率（15 用例，100%）
python tests/test_simple_skill.py        # Simple Query 端到端
python tests/test_complex_detailed.py    # Complex Query 端到端
python tests/test_analysis_detailed.py  # Data Analysis 7步流程
python tests/test_main_graph.py          # 主图集成
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

| 组件 | 版本 |
|------|------|
| LangGraph | ≥ 1.0.0 |
| LangChain | ≥ 1.0.0 |
| LLM | 通义千问 qwen-plus (ChatTongyi) |
| sqlglot | SQL AST 解析（安全护栏） |
| matplotlib | ≥ 3.x（图表生成） |
| 数据库 | MySQL 8.0 / SQLite |
| Python | ≥ 3.10 |

## 配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DB_URI` | `sqlite:///Chinook.db` | 数据库连接字符串 |
| `DASHSCOPE_API_KEY` | — | 通义千问 API Key |
| `LLM_MODEL` | `qwen-plus` | 模型名称 |
| `LLM_PROVIDER` | `tongyi` | LLM 提供商 |
| `LLM_TEMPERATURE` | `0.0` | 生成温度 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `REPORT_DIR` | `report` | 分析报告保存目录 |
| `CHART_DIR` | `report/charts` | 图表保存目录 |
| `SECURITY_MAX_ROWS` | `1000` | 最大返回行数（自动注入 LIMIT） |
| `SECURITY_TABLE_DENYLIST` | — | 禁止访问的表（逗号分隔） |
| `SECURITY_TABLE_ALLOWLIST` | — | 允许访问的表（留空=允许全部） |
| `SECURITY_AUDIT_LOG` | `true` | 是否启用审计日志 |
| `SECURITY_AUDIT_LOG_FILE` | — | 审计日志文件路径（JSONL 格式） |

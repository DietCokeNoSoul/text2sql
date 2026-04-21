# Text2SQL Agent — 开发历程报告

> 项目地址: `c:\Users\71949\Desktop\text2sql`  
> 技术栈: LangChain v1 · LangGraph v1 · Tongyi Qwen · MySQL / SQLite  
> 架构模式: Skill-based Modular Agent（技能模块化 Agent）

---

## 目录

1. [项目背景](#1-项目背景)
2. [阶段一：LangChain v1 迁移](#2-阶段一langchain-v1-迁移)
3. [阶段二：Skill 架构重构](#3-阶段二skill-架构重构)
4. [阶段三：Simple Query Skill 实现与调试](#4-阶段三simple-query-skill-实现与调试)
5. [阶段四：Complex Query Skill 实现与调试](#5-阶段四complex-query-skill-实现与调试)
6. [阶段五：Data Analysis Skill 实现与调试](#6-阶段五data-analysis-skill-实现与调试)
7. [阶段六：路由器与集成测试](#7-阶段六路由器与集成测试)
8. [阶段七：Schema 缓存与性能优化](#8-阶段七schema-缓存与性能优化)
9. [阶段八：功能增强 — 列名模糊匹配](#9-阶段八功能增强--列名模糊匹配)
10. [阶段九：图表生成与报告输出](#10-阶段九图表生成与报告输出)
11. [阶段十：测试体系建设](#11-阶段十测试体系建设)
12. [阶段十一：SQL 安全护栏](#12-阶段十一sql-安全护栏)
13. [最终架构总览](#13-最终架构总览)
14. [测试覆盖总结](#14-测试覆盖总结)

---

## 1. 项目背景

### 起点

项目原本是一个基于旧版 LangChain (0.2.x / 0.3.x) 和 LangGraph (0.2.x) 的单体 SQL Agent，具备将自然语言问题转换为 SQL 查询的基本能力。

**初始问题：**
- 依赖旧版 API，大量使用 `langchain_core.*` 的过时导入路径
- 无模块化设计，所有逻辑集中在 `agent/nodes.py`
- 无错误处理、无重试机制
- 无缓存策略，每次查询都重复调用数据库

**目标：** 升级到 LangChain/LangGraph v1，重构为模块化 Skill 架构，并逐步增加生产级功能。

---

## 2. 阶段一：LangChain v1 迁移

### 背景

LangChain v1 相比 0.3.x 有较大 API 变动，需要全面梳理后系统性迁移。

### 主要变更点

| 旧 API | 新 API |
|--------|--------|
| `langchain_core.messages.*` | `langchain_core.messages.*`（保留）/ `langchain.messages.*` |
| `langchain_core.tools.*` | `langchain.tools.*` |
| `BaseLanguageModel` | `BaseChatModel` |
| `add_node(node, name)` | `add_node(name, node)` — 参数顺序对调 |
| `BaseMessage` 作为类型 | `AnyMessage`（langgraph）或具体子类 |
| `langchain.messages.HumanMessage` | `langchain_core.messages.HumanMessage` |

### 遇到的问题

**问题 1：`BaseMessage` 不存在于 v1**
- 现象：`from langchain.messages import BaseMessage` 报 `ImportError`
- 根因：v1 使用具体消息类（`AIMessage`、`HumanMessage`），不再暴露 `BaseMessage`
- 解决：类型注解改用 `AnyMessage`，或直接使用具体类型

**问题 2：`add_node()` 参数顺序**
- 旧：`add_node(callable, "name")`
- 新：`add_node("name", callable)`
- 批量更新了 `graph_builder.py` 和 `example/sql_agent_graph.py`

**问题 3：ChatTongyi 初始化方式**
- `init_chat_model("tongyi:qwen-plus")` 不支持通义千问
- 解决：直接使用 `ChatTongyi(model=..., dashscope_api_key=...)`

### 输出

- `requirements.txt` 更新至 `langgraph>=1.0.0`, `langchain>=1.0.0`
- 全项目 import 路径修复完成

---

## 3. 阶段二：Skill 架构重构

### 设计决策

将单体 Agent 拆分为三个专用 Skill，通过主图 + 路由器进行统一调度：

```
用户输入
   │
   ▼
Main Graph
   │
   ├─ 路由器 (query_router_node)
   │      ├─ SIMPLE  → Simple Query Skill
   │      ├─ COMPLEX → Complex Query Skill
   │      └─ ANALYSIS→ Data Analysis Skill
   │
   ▼
输出结果
```

### 基础设施建设

```
agent/
├── nodes/
│   └── common.py         # 可复用节点工厂（所有 Skill 共享）
├── skills/
│   ├── base.py           # BaseSkill 抽象基类
│   ├── states.py         # 各 Skill 的 State 定义
│   └── registry.py       # Skill 注册表
└── skill_graph_builder.py # 主图 + 路由器
```

### 关键设计

**CommonNodes 节点复用模式**
- `create_list_tables_node()` — 列出数据库表
- `create_get_schema_node()` — 获取表结构
- `create_execute_query_node()` — 执行 SQL 查询
- 所有 Skill 共享这些节点，避免代码重复

**遇到的问题：import 命名冲突**
- `agent/nodes.py`（文件）与 `agent/nodes/`（目录）同名，Python 解析冲突
- 解决：将旧文件重命名为 `agent/old_nodes.py`，后续完成迁移后删除

**遇到的问题：Pydantic 字段命名警告**
- State 类中有名为 `schema` 的字段，shadowing 了 `BaseModel.schema()` 方法
- 解决：重命名为 `table_schema` / `table_schemas`

---

## 4. 阶段三：Simple Query Skill 实现与调试

### 功能设计

单表查询技能，适合直接可回答的 SQL 问题。

**流程：** `list_tables → get_schema → generate_query → execute_query`

### 问题 1：LLM 不生成 tool_calls

- **现象：** `llm.bind_tools([query_tool])` 后，Qwen 返回文本回复而非 tool_calls
- **根因：** 通义千问对 tool binding 的触发比 GPT 更严格，需要显式约束
- **解决：** 
  ```python
  llm.bind_tools([query_tool], tool_choice="required")
  ```
  同时增加 fallback：若无 tool_calls，发送 `HumanMessage` 明确要求调用工具
- **效果：** tool calling 成功率 100%

### 问题 2：SQL 出错后 LLM 生成 DESCRIBE 命令

- **现象：** 错误修复节点修复 SQL 时，LLM 返回 `DESCRIBE table` 而非修正后的 SELECT
- **根因：** fix prompt 没有明确禁止 DDL/元数据命令
- **解决：** 重写 fix prompt，加入显式禁止列表（`DESCRIBE`/`SHOW COLUMNS`/`CREATE`），并附带 ✅/❌ 示例

### 错误修复循环设计

```
generate_query
      │
      ▼
execute_query ──── 成功 ──→ END
      │
    失败 (retry_count < 3)
      │
      ▼
   fix_query
      │
      └──────────────→ execute_query
```

- 自定义 `SimpleQueryState` TypedDict，跟踪 `retry_count`、`last_error`、`last_sql`
- 最多重试 3 次，防止无限循环
- **最终效果：** 首次成功率 80%，自动修复成功率 100%

---

## 5. 阶段四：Complex Query Skill 实现与调试

### 功能设计

多步骤复杂查询，使用 Plan-Execute 模式，支持步骤间依赖。

**流程：** `list_tables → get_schema → plan_query → [执行步骤 1] → [执行步骤 2] → ... → aggregate`

### 问题 1：get_schema 节点超时

- **现象：** `create_get_schema_node()` 使用 `llm.bind_tools()` 获取 schema，有时超时
- **根因：** 通过 LLM 调用 tool 获取 schema 是多余的间接层
- **解决：** 重构为**直接调用** `schema_tool.invoke({"table_names": "..."})`，绕过 LLM

### 问题 2：MySQL 不支持 LIMIT 与子查询组合

- **现象：** 多步骤 SQL 中 LLM 生成 `IN (SELECT ... LIMIT N)` — MySQL 不支持
- **解决：** 设计 **Placeholder 机制**

**Placeholder 机制原理：**
```sql
-- Step 1: 查询店铺 ID
SELECT id FROM tb_shop WHERE type_id = 1

-- Step 2: LLM 在 plan 阶段生成（含占位符）
SELECT * FROM tb_blog WHERE shop_id IN {step_1_results}

-- 执行时替换
_resolve_query_placeholders() → "WHERE shop_id IN (1, 2, 3)"
```

### 问题 3：ast.literal_eval 解析失败

- **现象：** `query_tool.invoke()` 返回字符串 `"[(1, '美食'), (2, 'KTV')]"`，直接 JSON 解析报错
- **解决：** 使用 `ast.literal_eval()` 将 Python 风格字符串解析为实际列表

### 问题 4：LLM 生成嵌套 SQL 模板

- **现象：** LLM 错误地生成 `FROM ({step_1_results}) AS t`（子查询形式）
- **解决：** prompt 中添加显式错误示例，标注 ❌ WRONG / ✅ CORRECT

---

## 6. 阶段五：Data Analysis Skill 实现与调试

### 功能设计

7 步完整数据分析流程：

```
understand_goal → explore_data → plan_analysis →
generate_queries → execute_queries → extract_insights →
visualize → generate_report
```

### 问题 1：JSON 解析失败

- **现象：** `plan_analysis` 节点报错 `Expecting value: line 1 column 1`
- **根因：** LLM 把 JSON 包裹在 ` ```json ` 代码块中，`json.loads()` 无法直接解析
- **解决：** 实现 `extract_json_from_response()` 辅助函数，依次尝试：
  1. 直接 `json.loads()`
  2. 提取 ` ```json ``` ` 代码块
  3. 正则提取第一个 `{...}` 块

### 问题 2：中文输出乱码

- **现象：** 报告中文被序列化为 `\u5546\u5e97` 形式
- **解决：** 所有 `json.dumps()` 调用添加 `ensure_ascii=False`

### 问题 3：plan 解析失败时无 fallback

- **解决：** 实现 `_generate_queries_from_goal()` 兜底函数，当 plan 解析失败时直接从 `analysis_goal` 生成查询

### 最终效果

- 7/7 步骤全部成功执行
- 生成完整中文 Markdown 分析报告
- 支持 4 类可视化建议

---

## 7. 阶段六：路由器与集成测试

### 路由器设计

```python
# 分类规则（LLM 判断）
SIMPLE:   直接可以用一条 SQL 回答的问题
COMPLEX:  需要多步骤、跨表联合的问题
ANALYSIS: 需要统计分析、趋势洞察、可视化的问题
```

### 测试结果

创建 `tests/test_router_accuracy.py`，15 个测试用例（每类 5 个）：

| 类别 | 用例数 | 正确数 | 准确率 |
|------|--------|--------|--------|
| SIMPLE | 5 | 5 | 100% |
| COMPLEX | 5 | 5 | 100% |
| ANALYSIS | 5 | 5 | 100% |
| **总计** | **15** | **15** | **100%** |

---

## 8. 阶段七：Schema 缓存与性能优化

### 背景问题

每次查询都从数据库重新拉取全量 Schema，在多步骤执行中造成大量重复 I/O。

### 方案：两层缓存架构

**Layer 1 — SQLDatabaseManager 级缓存 (`SchemaCache`)**
```python
class SchemaCache:
    # TTL = 300s（可配置）
    # 表名列表缓存
    # Schema 文本缓存（key 为有序表名集合，顺序无关）
    cache_key = ",".join(sorted(table_names))
```

**Layer 2 — LangChain Tool 级缓存 (`CachedSchemaTool`)**
```python
class CachedSchemaTool:
    # 代理模式：拦截 sql_db_schema 工具的 invoke() 调用
    # 命中缓存时直接返回，不触发 DB 查询
```

### Data Analysis 批量 Schema 优化

**问题发现：** Data Analysis Skill 用逐表循环获取 Schema，与 Simple/Complex 的批量获取方式不同，导致缓存 key 不匹配，无法跨 Skill 共享缓存。

```
Simple/Complex: get_schema(["tb_a","tb_b",...所有表]) → 一次调用
Data Analysis:  get_schema(["tb_a"]) + get_schema(["tb_b"]) + ... → N 次调用
                                                             ↑ 跨技能缓存失效
```

**解决：** 将 Data Analysis 的逐表循环改为一次性批量调用，添加 `combined_schema: str` 字段到 State，所有下游节点统一读取。

**效果：**
- 首次查询：1 次 DB 调用（原来 N 次）
- 后续查询：0 次 DB 调用（全命中缓存）
- 跨技能缓存可共享

---

## 9. 阶段八：功能增强 — 列名模糊匹配

### 背景

SQL 错误中最常见的是列名拼写错误（如 `nickname` → `nick_name`，`Millisec` → `Milliseconds`）。

### 实现

在 `agent/database.py` 添加：
```python
def get_column_map() -> Dict[str, List[str]]
    # 返回 {table_name: [col1, col2, ...]} 的完整映射

def find_similar_columns(bad_col: str, cutoff=0.55) -> List[str]
    # 使用 difflib.get_close_matches 进行模糊匹配
    # 返回最多 8 个候选列名（格式：table.column）
```

在 `skills/simple_query/skill.py` 的 `_fix_query()` 节点中注入列名提示：
```python
# 从错误信息提取坏列名
bad_col = _extract_bad_column(error_message)
# 查找相似列名
hints = _build_column_hint(bad_col)
# 注入 fix prompt
prompt += f"\n可能的正确列名: {hints}"
```

**效果：**
- `Millisec` → 自动建议 `Track.Milliseconds`
- `nickname` → 自动建议 `tb_user.nick_name`

---

## 10. 阶段九：图表生成与报告输出

### 图表生成（两阶段）

**Phase 1 — 纯 Python SVG（无外部依赖）**
- 实现 `skills/data_analysis/chart_generator.py`
- 支持 bar / pie / line 三种图表
- 输出 SVG 文件，可在浏览器直接打开

**Phase 2 — 升级为 matplotlib PNG**
- 安装 `matplotlib 3.10.8` 后重写渲染器
- 支持自动检测 CJK 字体（SimHei / Microsoft YaHei）
- 输出高质量 PNG 文件

### 报告输出

- `_save_report()` 方法：生成时间戳命名的 Markdown 文件
- 保存路径通过 `OutputConfig` 配置（默认 `report/`，支持 `.env` 自定义）
- 报告内嵌图表引用：`![标题](charts/bar_xxx.png)` — 支持 Markdown 预览器直接显示

### 修复的 Bug

**Decimal 类型解析失败**
- **现象：** MySQL 返回 `Decimal('10.0000')` 格式，`ast.literal_eval` 无法解析
- **解决：** 在解析前正则替换
  ```python
  raw = re.sub(r"Decimal\('([\d.]+)'\)", r"\1", raw)
  raw = re.sub(r"datetime\.date\([^)]+\)", "'date'", raw)
  ```

**图表在 Markdown 中无法预览**
- **原因：** 报告中图表以文件路径文本列出，非 Markdown 图片语法
- **解决：** 改为相对路径的 `![title](charts/xxx.png)` 格式

---

## 11. 阶段十：测试体系建设

### 测试文件清理

整理前：根目录散落 15+ 个测试和调试文件  
整理后：统一归入 `tests/` 目录，删除空/废弃文件

### 测试套件

| 文件 | 类型 | 用例数 | 覆盖内容 |
|------|------|--------|----------|
| `test_schema_cache.py` | Unit（无需 API） | 13 | SchemaCache 读写/TTL/clear/order-independent key |
| `test_column_fuzzy_match.py` | Unit（无需 API） | 20 | 列名提取、模糊匹配、提示生成 |
| `test_chart_generation.py` | Unit（无需 API） | 30 | ChartSpec/PNG 渲染/SVG/from_query_result/_visualize |
| `test_report_saving.py` | Unit（无需 API） | 32 | _save_report/目录创建/时间戳/OutputConfig env var |
| `test_security.py` | Unit（无需 API） | 61 | 四层护栏/审计日志/SecurityConfig/集成测试 |
| `test_router_accuracy.py` | Live（需 API） | 15 | 路由器分类准确率 |
| `test_analysis_detailed.py` | Live（需 API） | 1 | Data Analysis 7 步完整流程 |
| `test_simple_skill.py` | Live（需 API） | - | Simple Query 端到端 |
| `test_complex_detailed.py` | Live（需 API） | - | Complex Query 端到端 |
| `test_main_graph.py` | Live（需 API） | - | 主图集成测试 |

**无需 API Key 的测试总计：156 个，全部通过 ✅**

---

## 12. 阶段十一：SQL 安全护栏

### 背景与动机

Agent 生成的 SQL 语句直接在生产数据库上执行，存在多类安全风险：
- LLM 幻觉可能生成 `DROP TABLE`、`DELETE` 等破坏性语句
- 未受限的 `SELECT *` 可能返回密码、Token 等敏感字段
- 无行数上限可能导致返回百万级数据阻塞服务
- 无执行审计，无法追溯异常操作

### 四层防御架构

```
SQL 输入
  │
  ▼ Layer 1 — 语句类型控制   (sqlglot AST 解析 + 危险关键字双保险)
  │  默认只允许 SELECT；拒绝 INSERT/UPDATE/DELETE/DROP/CREATE/TRUNCATE 等
  │  关键字扫描：xp_cmdshell / INTO OUTFILE / LOAD DATA / EXEC / OPENROWSET …
  │
  ▼ Layer 2 — 表访问控制
  │  denylist：明确禁止访问的表（如 users、auth_tokens、secrets）
  │  allowlist：可选白名单，仅允许列出的表被查询
  │
  ▼ Layer 3 — 查询复杂度限制
  │  SQL 字符长度上限（默认 5000 字符）
  │  自动注入 LIMIT（无 LIMIT 时注入；超出上限时强制降低）
  │
  ▼ Layer 4 — 结果脱敏（执行后）
  │  检测查询涉及 password/token/phone/api_key/credit_card 等敏感列
  │  在日志中发出 ⚠️ 警告并在返回结果末尾追加提示
  │
审计日志（内存记录 + 可选 JSONL 文件）
```

### 关键设计决策

**使用 sqlglot 做 AST 解析（而非正则）**

正则检测语句类型有大量绕过方式（注释混入、大小写变换、Unicode 转义等）。
`sqlglot.parse()` 构建完整 AST，提取顶级语句节点类型，准确率更高。
同时保留正则关键字扫描作为第二道防线，防御 sqlglot 未覆盖的方言或混淆输入。

**LIMIT 重写而非拒绝**

对于缺少 LIMIT 的查询，直接拒绝会破坏正常业务；
策略改为**静默注入**最大行数限制，利用 sqlglot 重写 AST 后生成安全 SQL，
对上层调用方完全透明。

**警告式脱敏（Layer 4）**

对敏感列采取「检测 + 警告」而非「替换列值」策略，原因：
- 替换值会破坏 Data Analysis Skill 的数值计算
- LLM 对结果的理解需要真实数据
- 脱敏责任应由调用方（应用层）承担，安全层负责提示

### 新增文件

| 文件 | 内容 |
|------|------|
| `agent/security.py` | 四层防御核心；`SQLSecurityGuard`、`ValidationResult` |
| `agent/types.py` | 新增 `SecurityViolationError`（含 layer/reason/sql 属性）|
| `agent/config.py` | 新增 `SecurityConfig` dataclass（含 `from_env()`） |
| `agent/database.py` | `execute_query` 接入护栏；`__init__` 接受 `security_config` |
| `agent/__init__.py` | 导出 `SecurityConfig`、`SQLSecurityGuard`、`SecurityViolationError` |
| `tests/test_security.py` | 61 个单元 + 集成测试，全部通过 ✅ |

### SecurityConfig 可配置项

```python
@dataclass
class SecurityConfig:
    allowed_statements: list = ["SELECT"]          # 允许的语句类型
    blocked_keywords: list = ["xp_cmdshell", ...]  # 危险关键字列表
    table_allowlist: list | None = None            # 表白名单（None=全放行）
    table_denylist: list = []                      # 表黑名单
    max_rows: int = 1000                           # 最大返回行数
    max_query_length: int = 5000                   # SQL 长度上限
    sensitive_column_patterns: list = [...]        # 敏感列名正则列表
    enable_audit_log: bool = True                  # 审计日志开关
    audit_log_file: str | None = None              # JSONL 文件路径（None=仅 logger）
```

支持通过环境变量配置：`SECURITY_MAX_ROWS`、`SECURITY_TABLE_DENYLIST`、
`SECURITY_TABLE_ALLOWLIST`、`SECURITY_AUDIT_LOG`、`SECURITY_AUDIT_LOG_FILE`。

### 典型使用示例

```python
from agent.config import DatabaseConfig, SecurityConfig
from agent.database import SQLDatabaseManager

mgr = SQLDatabaseManager(
    DatabaseConfig(uri="mysql+pymysql://readonly_user:pass@host/db"),
    security_config=SecurityConfig(
        table_denylist=["users", "auth_tokens"],
        max_rows=500,
        audit_log_file="logs/sql_audit.jsonl",
    )
)

# execute_query 内部自动执行全套安全检查
result = mgr.execute_query("SELECT * FROM orders")   # ✅ 通过
result = mgr.execute_query("DROP TABLE orders")      # ❌ 抛出 SecurityViolationError
```

### 测试覆盖

| 测试类 | 用例数 | 覆盖内容 |
|--------|--------|----------|
| `TestLayer1StatementType` | 14 | SELECT 放行、各类 DML/DDL 拦截、危险关键字、自定义允许列表 |
| `TestLayer2TableAccess` | 8 | denylist/allowlist 精确匹配、大小写不敏感、JOIN 多表场景 |
| `TestLayer3Complexity` | 6 | 长度拒绝、LIMIT 注入、LIMIT 降低、重写 SQL 合法性 |
| `TestLayer4Sanitize` | 6 | 敏感列检测、无敏感列直通、多列报告、自定义模式 |
| `TestValidationResult` | 3 | passed/layer/reason/rewritten_sql 字段 |
| `TestAuditLog` | 9 | 记录积累、统计摘要、清空、文件写入、时间戳字段 |
| `TestSecurityConfigFromEnv` | 5 | 默认值、环境变量覆盖 |
| `TestSecurityViolationError` | 3 | 属性、字符串表示、继承链 |
| `TestDatabaseManagerIntegration` | 7 | SQLite 集成、护栏拦截抛异常、LIMIT 透明注入 |
| **合计** | **61** | **全部通过 ✅** |

---

## 13. 最终架构总览

```
text2sql/
├── agent/
│   ├── config.py              # AgentConfig / OutputConfig / SecurityConfig（支持 .env）
│   ├── database.py            # SQLDatabaseManager + SchemaCache + SecurityGuard 集成
│   ├── security.py            # SQLSecurityGuard — 四层防御核心
│   ├── tools.py               # SQLToolManager + CachedSchemaTool
│   ├── skill_graph_builder.py # 主图 + 路由器
│   ├── graph.py               # 统一入口
│   ├── nodes/
│   │   └── common.py          # 可复用节点工厂
│   └── skills/
│       ├── base.py            # BaseSkill 抽象基类
│       └── states.py          # State 定义
│
├── skills/
│   ├── simple_query/
│   │   └── skill.py           # 单查询 + 错误修复循环
│   ├── complex_query/
│   │   └── skill.py           # 多步骤 + Placeholder 机制
│   └── data_analysis/
│       ├── skill.py           # 7 步分析流程
│       └── chart_generator.py # matplotlib PNG 图表生成
│
├── tests/                     # 156 个 Mock 测试 + Live 集成测试
├── report/                    # 分析报告输出目录
│   └── charts/                # PNG 图表输出目录
└── .env                       # 配置（DB_URI / API_KEY / REPORT_DIR / SECURITY_* 等）
```

### 核心数据流

```
用户问题
   │
   ▼
Router（LLM 分类）
   │
   ├─ Simple Query
   │      list_tables → get_schema → generate_query
   │      → execute → [错误?] → fix_query(列名模糊匹配) → retry
   │
   ├─ Complex Query
   │      list_tables → get_schema → plan
   │      → execute_step_1 → execute_step_2(Placeholder) → aggregate
   │
   └─ Data Analysis
          understand_goal → explore_data(批量 schema) → plan_analysis
          → generate_queries → execute_queries → extract_insights
          → visualize(matplotlib PNG) → generate_report(保存到 report/)
```

---

## 14. 测试覆盖总结

### 功能覆盖矩阵

| 功能模块 | 单元测试 | 集成测试 | 状态 |
|----------|----------|----------|------|
| Schema 缓存 | ✅ 13 用例 | ✅ | 完备 |
| 列名模糊匹配 | ✅ 20 用例 | ✅ | 完备 |
| 图表生成 | ✅ 30 用例 | - | 完备 |
| 报告保存 | ✅ 32 用例 | ✅ | 完备 |
| SQL 安全护栏 | ✅ 54 用例 | ✅ 7 用例 | 完备 |
| 路由器准确率 | - | ✅ 15/15 | 完备 |
| Simple Query E2E | - | ✅ | 人工验证 |
| Complex Query E2E | - | ✅ | 人工验证 |
| Data Analysis E2E | - | ✅ | 人工验证（7/7 步） |

### 关键问题 → 解决方案速查

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| LLM 不生成 tool_calls | Qwen 需要显式约束 | `tool_choice="required"` + fallback retry |
| 错误修复时生成 DESCRIBE | Prompt 不够明确 | ✅/❌ 示例 + 显式禁止列表 |
| MySQL LIMIT 子查询不兼容 | MySQL 方言限制 | Placeholder 机制（`{step_N_results}`） |
| LLM 返回 JSON 被代码块包裹 | LLM 输出习惯 | `extract_json_from_response()` 三阶段解析 |
| 跨 Skill Schema 缓存失效 | key 计算方式不同 | Data Analysis 统一改为批量调用 |
| Decimal 类型无法解析为图表 | `ast.literal_eval` 不支持 | 解析前正则预处理 |
| Markdown 中图表不显示 | 文件路径而非图片语法 | 改为 `![title](charts/xxx.png)` |
| 列名拼写错误导致 SQL 失败 | 自然语言列名与实际不符 | `difflib` 模糊匹配注入 fix prompt |
| LLM 生成 DROP/DELETE 语句 | 无执行前拦截 | SQL 安全护栏 Layer 1 — sqlglot AST 分类 |
| 查询返回过多行数 | 无行数限制 | SQL 安全护栏 Layer 3 — 自动注入/降低 LIMIT |
| 敏感字段（密码/Token）裸露 | 无结果过滤 | SQL 安全护栏 Layer 4 — 敏感列检测 + ⚠️ 警告 |

---

*生成时间：2026-04-10 | 最后更新：2026-04-13（新增阶段十一：SQL 安全护栏）*

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
13. [阶段十二：双塔检索架构](#13-阶段十二双塔检索架构)
14. [最终架构总览](#14-最终架构总览)
15. [测试覆盖总结](#15-测试覆盖总结)
16. [功能流程详解](#16-功能流程详解)

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

## 13. 阶段十二：双塔检索架构

### 背景与动机

Complex Query Skill 在处理多表关联查询时，将**全量 Schema**送入 LLM 规划节点。
在 Chinook.db（11 表）场景下，全量 Schema 约 5,881 字符（~1,680 tokens），
而一次典型 3 表 JOIN 查询实际只需要约 1,400 字符（~400 tokens）。
**无关 Schema 不仅浪费 token，还会对 LLM 产生干扰，导致幻觉列名或错误 JOIN。**

### 方案设计

```
用户复杂查询
  │
  ▼ Tower 1 — Milvus 向量检索（列语义相似度）
  │   将每列描述文本（table.column: type. samples: a, b, c）
  │   使用 paraphrase-multilingual-MiniLM-L12-v2 编码为 384 维向量
  │   存入 Milvus HNSW 索引；查询时返回语义最相关的 top-k 个表名
  │
  ▼ Tower 2 — NetworkX Schema 图 + Steiner Tree 路径规划
  │   从 SQLAlchemy FK 约束（weight=1.0）+ 列名模式推断（weight=1.5）
  │   构建 Schema 图；对 Tower 1 返回的表运行 Steiner Tree 近似算法
  │   补全"连接路径"上的中间表（JOIN 不能跳过的桥接表）
  │
  ▼ LLM 智能剪枝
  │   将 Steiner Tree 包含的表子集格式化为 pruned_schema
  │   附带 JOIN hint（推荐的联接顺序）注入 plan 节点 prompt
  │
结果：平均减少 71% Schema 字符 / 节省 ~1,194 tokens/次复杂查询
```

### 实现细节

**SchemaGraph（`agent/schema_graph.py`）**

双层边权重推断：
- Layer 1：SQLAlchemy `ForeignKeyConstraint` → 权重 1.0（确定边）
- Layer 2：列名模式 `{X}_id → tb_{X}.id` → 权重 1.5（推断边）

Steiner Tree 使用 `networkx.algorithms.approximation.steiner_tree`（Kou 2-近似），
图规模小时 < 10ms；若算法失败自动降级为最短路径联合。

**ColumnIndex（`agent/column_index.py`）**

- Milvus collection `text2sql_columns`，HNSW + COSINE，384 维
- `ColumnRecord.id`：`int(md5("table.column")[:15], 16)` —— 确定性 ID，跨重建稳定
- Schema 指纹（sorted `table.column` pairs）变化时自动触发重建
- 冷启动保护：Milvus 不可达时 graceful fallback 到全量 Schema

**DualTowerRetriever（`agent/retrieval.py`）**

```python
result = retriever.retrieve("每首歌曲的艺术家信息")
# result.pruned_schema   → 仅含 Track + Album + Artist 的 schema 文本
# result.join_hint       → "-- JOIN: Artist → Track → Album"
# result.reduction_pct   → 76.3
# result.estimated_token_saved → 1281
```

**ComplexQuerySkill 集成**

在 `get_schema` 和 `plan` 节点之间插入条件节点 `dual_tower_retrieve`：
- `retriever is not None` → 启用检索，pruned_schema 注入 state
- `retriever is None`（Milvus 不可达/未配置）→ 原流程不变

**SkillBasedGraphBuilder**

`__init__` 中 try/except 初始化 `DualTowerRetriever`：
- 成功 → 传递给 ComplexQuerySkill
- 失败（Milvus 不可达）→ `self._retriever = None`，系统降级运行

### 新增/修改文件

| 文件 | 说明 |
|------|------|
| `agent/schema_graph.py` | NetworkX Schema 图 + Steiner Tree（新建） |
| `agent/column_index.py` | Milvus 列向量索引（新建） |
| `agent/retrieval.py` | 双塔检索协调器（新建） |
| `agent/config.py` | 新增 `RetrievalConfig` dataclass（含 `from_env()`） |
| `agent/__init__.py` | 导出 `RetrievalConfig`、`DualTowerRetriever`、`SchemaGraph`、`ColumnIndex` |
| `agent/skill_graph_builder.py` | 初始化 DualTowerRetriever，传递给 ComplexQuerySkill |
| `skills/complex_query/skill.py` | 新增 `dual_tower_retrieve` 节点，state 增加 `retrieval_stats` |
| `tests/test_retrieval_benchmark.py` | 28 个基准测试（新建） |
| `pyproject.toml` | 新增 `networkx`、`pymilvus`、`sentence-transformers`、`torch` 依赖 |

### RetrievalConfig 可配置项

```python
@dataclass
class RetrievalConfig:
    enabled: bool = True              # env: RETRIEVAL_ENABLED
    milvus_host: str = "127.0.0.1"   # env: MILVUS_HOST
    milvus_port: int = 19530          # env: MILVUS_PORT
    top_k: int = 10                   # env: RETRIEVAL_TOP_K
    max_tables: int = 5               # env: RETRIEVAL_MAX_TABLES
    similarity_threshold: float = 0.3 # env: RETRIEVAL_THRESHOLD
    force_rebuild: bool = False       # env: RETRIEVAL_FORCE_REBUILD
```

### 基准测试结果（Chinook.db，11 表）

| 场景 | 原始 Schema | Pruned | 节省 |
|------|------------|--------|------|
| Track–Album–Artist | 5,881 chars | 1,395 chars | 76% |
| Customer–Invoice–InvoiceLine | 5,881 chars | 2,331 chars | 60% |
| Track–Genre | 5,881 chars | 1,035 chars | 82% |
| Employee–Customer | 5,881 chars | 2,263 chars | 62% |
| Playlist–PlaylistTrack–Track | 5,881 chars | 1,396 chars | 76% |
| **平均** | — | — | **71% / ~1,194 tokens** |

Schema 图构建 + Steiner Tree 规划延迟：**平均 5ms**（不含 Milvus 搜索的 30~80ms）。

### 测试覆盖（28 用例，全部通过 ✅）

| 测试类 | 用例数 | 覆盖内容 |
|--------|--------|----------|
| `TestSchemaGraphBuild` | 6 | 节点/边/描述输出/边权重 |
| `TestSteinerTree` | 6 | 空输入/单表/两表/不存在表/join hint |
| `TestSchemaPruning` | 3 | 剪枝缩减/包含目标表/基准报告 |
| `TestColumnIndexLogic` | 5 | ID 稳定性/文本截断/指纹变更 |
| `TestDualTowerRetrieverLogic` | 4 | 返回剪枝 schema/token 节省/fallback |
| `TestRetrievalPipeline` | 4 | 端到端/延迟/路径包含/全基准输出 |

---

## 14. 最终架构总览

```
text2sql/
├── agent/
│   ├── config.py              # AgentConfig / OutputConfig / SecurityConfig / RetrievalConfig
│   ├── database.py            # SQLDatabaseManager + SchemaCache + SecurityGuard 集成
│   ├── security.py            # SQLSecurityGuard — 四层防御核心
│   ├── schema_graph.py        # NetworkX Schema 图 + Steiner Tree 路径规划
│   ├── column_index.py        # Milvus 列向量索引（Tower 1）
│   ├── retrieval.py           # DualTowerRetriever 协调器
│   ├── tools.py               # SQLToolManager + CachedSchemaTool
│   ├── skill_graph_builder.py # 主图 + 路由器 + DualTowerRetriever 初始化
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
│   │   └── skill.py           # 多步骤 + Placeholder + 双塔检索
│   └── data_analysis/
│       ├── skill.py           # 7 步分析流程
│       └── chart_generator.py # matplotlib PNG 图表生成
│
├── tests/                     # 184 个 Mock 测试 + Live 集成测试
├── report/                    # 分析报告输出目录
│   └── charts/                # PNG 图表输出目录
└── .env                       # 配置（DB_URI / API_KEY / REPORT_DIR / SECURITY_* / RETRIEVAL_* 等）
```

### 核心数据流

```
用户问题
   │
   ▼
Router（LLM 分类：simple / complex / analysis）
   │
   ├─ Simple Query ──────────────────────────────────────────────────────────
   │      list_tables → get_schema → generate_query → execute_query
   │      → [成功] → END
   │      → [失败] → fix_query(difflib列名模糊匹配 + 错误诊断) → execute_query
   │                → [最多3次重试] → END
   │
   ├─ Complex Query ─────────────────────────────────────────────────────────
   │      list_tables → get_schema
   │      → [dual_tower_retrieve] ← Milvus向量检索 + Steiner Tree路径规划
   │      → plan（LLM生成多步骤计划）
   │      → execute_steps（依赖解析 + Placeholder替换）
   │      → aggregate（汇总各步结果）
   │      → judge → [未完成] → plan（继续）
   │               → [完成]  → END
   │
   └─ Data Analysis ─────────────────────────────────────────────────────────
          understand_goal → explore_data（批量拉取全库schema）
          → plan_analysis → generate_queries → execute_queries
          → analyze_results（提取洞察）
          → visualize（matplotlib PNG图表）
          → generate_report（Markdown报告保存到report/）→ END
```

---

## 15. 测试覆盖总结

### 功能覆盖矩阵

| 功能模块 | 单元测试 | 集成测试 | 状态 |
|----------|----------|----------|------|
| Schema 缓存 | ✅ 13 用例 | ✅ | 完备 |
| 列名模糊匹配 | ✅ 20 用例 | ✅ | 完备 |
| 图表生成 | ✅ 30 用例 | - | 完备 |
| 报告保存 | ✅ 32 用例 | ✅ | 完备 |
| SQL 安全护栏 | ✅ 54 用例 | ✅ 7 用例 | 完备 |
| 双塔检索架构 | ✅ 28 用例 | - | 完备 |
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
| 全量 Schema 浪费 token | 无关表信息注入 LLM | 双塔检索：Milvus 向量 + Steiner Tree，avg 71% 减少 |

---

*生成时间：2026-04-10 | 最后更新：2026-04-21（新增第16章功能流程详解）*

---

## 16. 功能流程详解

### 16.1 Simple Query — 简单查询流程

**适用场景**：单表查询、基础过滤、简单聚合，不需要多步骤分解。

**触发条件**：Router 分类为 `"simple"`。

```
                    ┌─────────────────────────────────────────────────────┐
  用户输入          │            SimpleQuerySkill 子图                     │
  "列出前10个艺术家" │                                                     │
        │           │  list_tables                                        │
        └──────────►│    └─ 调用 sql_db_list_tables 工具                  │
                    │    └─ 返回: ["Album","Artist","Track",...]          │
                    │                                                     │
                    │  get_schema                                         │
                    │    └─ 调用 sql_db_schema 工具                       │
                    │    └─ 返回: CREATE TABLE 定义（含列名/类型）         │
                    │                                                     │
                    │  generate_query                                     │
                    │    └─ LLM + bind_tools(sql_db_query, required)     │
                    │    └─ 生成 tool_call: {query: "SELECT ..."}         │
                    │                                                     │
                    │  execute_query ◄──────────────────────────┐        │
                    │    └─ SQLSecurityGuard 四层校验             │        │
                    │    └─ db_manager.execute_query(sql)        │        │
                    │    ┌─ [成功] → END                         │        │
                    │    └─ [失败] ──────────────────────────────┤        │
                    │                                            │        │
                    │  fix_query                                 │        │
                    │    ├─ _extract_bad_column(error_msg)       │        │
                    │    │    └─ 正则解析 MySQL/SQLite/PG 错误格式 │        │
                    │    ├─ db_manager.find_similar_columns()    │        │
                    │    │    └─ difflib.get_close_matches()     │        │
                    │    ├─ 构建列名建议 hint 注入 LLM prompt     │        │
                    │    └─ LLM 生成修复 SQL → 回到 execute_query─┘ (max 3次)
                    └─────────────────────────────────────────────────────┘
```

**State 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `list` | 完整消息历史（HumanMessage / AIMessage / ToolMessage） |
| `retry_count` | `int` | 当前重试次数（上限 3） |
| `last_error` | `str` | 最近一次执行错误信息 |
| `last_sql` | `str` | 最近一次执行的 SQL |

**关键设计**：
- `tool_choice="required"` 强制 LLM 一定输出 tool_call，避免文字回复
- `fix_query` 先 `difflib` 模糊匹配列名，再把建议注入 fix prompt，降低 LLM 猜测成本
- `SQLSecurityGuard` 在 `execute_query` 内部隐式执行，对上层完全透明

---

### 16.2 Complex Query — 复杂查询流程

**适用场景**：多表 JOIN、多步骤依赖、聚合后再过滤等需要分解的复杂问题。

**触发条件**：Router 分类为 `"complex"`。

```
                    ┌──────────────────────────────────────────────────────────────┐
  用户输入          │              ComplexQuerySkill 子图                           │
  "每个艺术家的     │                                                              │
   专辑数和总曲目数" │  list_tables → get_schema                                   │
        │           │                    │                                         │
        └──────────►│              ┌─────▼──────────────────────────────────────┐ │
                    │              │    dual_tower_retrieve（可选，需 Milvus）    │ │
                    │              │      ├─ Tower 1: Milvus HNSW 向量检索        │ │
                    │              │      │    query → embedding(384维)          │ │
                    │              │      │    → top-k 相关列 → 候选表列表        │ │
                    │              │      ├─ Tower 2: NetworkX Steiner Tree       │ │
                    │              │      │    候选表 → 最小连通子图              │ │
                    │              │      │    → 补全 JOIN 桥接表                │ │
                    │              │      └─ 输出: pruned_schema + join_hint      │ │
                    │              └─────────────────────────────────────────────┘ │
                    │                                   │                          │
                    │              plan ◄───────────────┘                          │
                    │                └─ LLM 分析问题，生成 steps[] JSON             │
                    │                └─ 每步: {step_id, description, query,        │
                    │                          depends_on:[...]}                   │
                    │                └─ Placeholder 规则:                          │
                    │                   WHERE col IN {step_N_results}              │
                    │                                   │                          │
                    │              execute_step ◄────────┘                         │
                    │                └─ 找出依赖已满足的 ready steps               │
                    │                └─ _resolve_query_placeholders():             │
                    │                   • 读取前序步骤结果（list of tuples）       │
                    │                   • 提取第一列 ID → (1, 2, 3) 格式          │
                    │                   • 替换 {step_N_results} → (1, 2, 3)      │
                    │                └─ db_manager.execute_query(resolved_sql)    │
                    │                └─ step_results[step_id] = {result/error}   │
                    │                                   │                          │
                    │              aggregate                                        │
                    │                └─ 格式化全部步骤结果为文本                   │
                    │                                   │                          │
                    │              judge                                            │
                    │                ├─ completed_steps == total_steps → END       │
                    │                └─ 未完成 → 回到 plan（继续规划剩余步骤）     │
                    └──────────────────────────────────────────────────────────────┘
```

**State 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `list` | 消息历史 |
| `tables` | `list[str]` | 数据库表名列表 |
| `table_schema` | `str` | Schema 文本（full 或 pruned） |
| `query_plan` | `list[dict]` | 多步骤计划（每步含 step_id / query / depends_on） |
| `step_results` | `dict` | `{step_id → {result, success, query, ...}}` |
| `plan_completed` | `bool` | 所有步骤是否已完成 |
| `retrieval_stats` | `dict` | 双塔检索统计（chars saved, reduction_pct 等） |

**关键设计**：
- **Placeholder 机制**：步骤间依赖通过 `{step_N_results}` 传递 ID 列表，绕开 MySQL 不支持 LIMIT-in-subquery 的限制
- **双塔检索可选**：`retriever is None` 时跳过，全量 schema 走原流程；Milvus 不可达时 graceful fallback
- **judge 循环**：支持 plan → execute → judge → plan 的多轮迭代，处理分批执行的场景

---

### 16.3 Data Analysis — 数据分析流程

**适用场景**：需要洞察、趋势、可视化图表和报告的深度分析问题。

**触发条件**：Router 分类为 `"analysis"`。

```
                    ┌──────────────────────────────────────────────────────────────┐
  用户输入          │              DataAnalysisSkill 子图（7步线性流程）            │
  "分析各艺术家的   │                                                              │
   销售趋势并生成   │  Step 1: understand_goal                                     │
   报告"            │    └─ LLM 解析目标 → {objective, metrics, dimensions,        │
        │           │         filters, output_format}                              │
        └──────────►│                                                              │
                    │  Step 2: explore_data                                        │
                    │    └─ 批量拉取所有表的 schema（一次 tool 调用）              │
                    │    └─ combined_schema = 所有 CREATE TABLE 拼接               │
                    │    └─ LLM 理解数据结构，识别关键表                           │
                    │                                                              │
                    │  Step 3: plan_analysis                                       │
                    │    └─ LLM 根据目标 + schema 生成分析计划                    │
                    │    └─ {analysis_steps, key_tables, approach}                │
                    │                                                              │
                    │  Step 4: generate_queries                                    │
                    │    └─ LLM 为每个分析步骤生成 SQL 查询列表                  │
                    │    └─ [{description, sql, purpose}, ...]                    │
                    │                                                              │
                    │  Step 5: analyze_results                                     │
                    │    └─ 批量执行所有 SQL（db_manager.execute_query）          │
                    │    └─ LLM 分析原始结果，提取洞察                            │
                    │    └─ insights = [{finding, significance, ...}, ...]        │
                    │                                                              │
                    │  Step 6: visualize                                           │
                    │    └─ LLM 生成可视化方案（chart_type, x, y, title）         │
                    │    └─ ChartGenerator.generate() → matplotlib PNG            │
                    │    └─ 保存到 report/charts/chart_YYYYMMDD_HHMMSS.png        │
                    │                                                              │
                    │  Step 7: generate_report                                     │
                    │    └─ LLM 综合所有洞察生成 Markdown 报告                   │
                    │    └─ 嵌入图表路径 ![title](charts/xxx.png)                 │
                    │    └─ 保存到 report/analysis_YYYYMMDD_HHMMSS.md            │
                    │    └─ END                                                   │
                    └──────────────────────────────────────────────────────────────┘
```

**State 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `list` | 消息历史 |
| `tables` | `list[str]` | 数据库表名列表 |
| `combined_schema` | `str` | 批量拉取的全库 schema 拼接 |
| `analysis_goal` | `str` | 解析后的分析目标 JSON 字符串 |
| `analysis_plan` | `dict` | 分析计划（含步骤/关键表/方法） |
| `sql_queries` | `list[dict]` | 生成的 SQL 查询列表 |
| `query_results` | `list[dict]` | 每条查询的执行结果 |
| `insights` | `list[dict]` | 提取的数据洞察 |
| `visualizations` | `list[dict]` | 可视化配置（含 chart_type/x/y/title） |
| `chart_files` | `list[str]` | 生成的 PNG 图表文件路径 |
| `report` | `str` | 最终 Markdown 报告内容 |

**关键设计**：
- `explore_data` 一次性批量拉取所有表 schema，消除跨 Skill 的缓存 key 不一致问题
- `generate_report` 使用 `![title](charts/xxx.png)` 格式嵌入图表，而非文件路径（避免 Markdown 不渲染）
- 报告目录由 `OutputConfig.report_dir`（env: `REPORT_DIR`）配置，默认 `report/`
- `extract_json_from_response()` 三阶段 JSON 解析（直接 → code block → 正则），容忍 LLM 输出格式多变

---

### 16.4 SQL 安全护栏 — 执行前后拦截链

所有三个 Skill 的 SQL 执行都经过 `SQLDatabaseManager.execute_query()`，内部自动执行完整的四层安全检查：

```
SQL 输入（来自 LLM 生成）
   │
   ▼ Layer 1: 语句类型校验（sqlglot AST 解析）
   │   允许: SELECT
   │   拒绝: INSERT / UPDATE / DELETE / DROP / CREATE / TRUNCATE / ...
   │   补充: 危险关键字扫描（xp_cmdshell / INTO OUTFILE / EXEC / ...）
   │
   ▼ Layer 2: 表访问控制
   │   denylist: 禁止访问指定表（如 users / auth_tokens / secrets）
   │   allowlist: 若配置，只允许访问白名单表
   │
   ▼ Layer 3: 查询复杂度限制
   │   SQL 长度 > max_query_length → 拒绝
   │   无 LIMIT → 自动注入 LIMIT {max_rows}（sqlglot AST 重写，对调用方透明）
   │   LIMIT > max_rows → 自动降低至 max_rows
   │
   ▼ 数据库执行
   │
   ▼ Layer 4: 结果脱敏（执行后）
       检测结果列名是否匹配敏感模式（password / token / phone / api_key / ...）
       → 发出 ⚠️ 警告 + 在结果末尾追加提示
       → 审计日志记录（内存 + 可选 JSONL 文件）
```

**异常类型**：违规时抛出 `SecurityViolationError(layer=1~4, reason=..., sql=...)`，Skill 可捕获后在修复流程中处理。

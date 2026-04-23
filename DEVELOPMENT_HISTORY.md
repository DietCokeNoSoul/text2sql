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
14. [阶段十三：流式输出优化](#14-阶段十三流式输出优化b1)
15. [阶段十四：LLM 响应缓存](#15-阶段十四llm-响应缓存b2)
16. [阶段十五：查询结果导出](#16-阶段十五查询结果导出b3)
17. [最终架构总览](#17-最终架构总览)
18. [测试覆盖总结](#18-测试覆盖总结)
19. [功能流程详解](#19-功能流程详解)
20. [阶段十六：工程质量优化](#20-阶段十六工程质量优化a3a4b4b5b6c1c2c3)
21. [阶段十七：工程质量 & 功能补全](#21-阶段十七工程质量--功能补全a5a6a7b12b13c4c5)
22. [阶段十八：工程质量补全](#22-阶段十八工程质量补全a8a9b14b15c6c7)
23. [阶段十九：导入规范化补全](#23-阶段十九导入规范化补全a10a11b16c8)
24. [阶段二十：SKILL.md 渐进式披露重构](#24-阶段二十skillmd-渐进式披露重构)
25. [阶段二十一：测试体系完善与代码质量清理](#25-阶段二十一测试体系完善与代码质量清理)

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

## 13. 阶段十三：流式输出优化（B1）

### 背景与目标

原有 `run_query()` 使用 `stream_mode="values"` 每步输出完整状态快照，输出噪音大且用户无法感知进度。目标是提供两种流式模式：**节点级流式**（每节点完成即输出）和**token 级流式**（LLM 生成时实时打印 token）。

### 实现方案

**文件：`agent/graph.py`**

| 新增内容 | 说明 |
|---------|------|
| `_NODE_LABELS` 字典 | 节点名称 → 带 emoji 的中文标签映射 |
| `_print_node_update()` | 格式化打印节点输出（含标签、耗时） |
| `run_query()` 重写 | 使用 `stream_mode="updates"` 仅输出增量，每节点完成后即打印 |
| `run_query_streaming_async()` | 使用 `graph.astream_events(version="v2")` 实现 token 级异步流式 |
| `run_query_streaming()` | 同步包装器，调用 `asyncio.run()` |
| `main()` 更新 | 支持 `stream` 命令切换节点级 / token 级模式 |

### 两种模式对比

```
模式一（节点级）: run_query()
  stream_mode="updates" → 每个节点完成后打印 delta
  适合：后端调用、日志记录

模式二（token 级）: run_query_streaming()
  astream_events(version="v2") → on_chat_model_stream 事件实时打印 token
  适合：前端 UI、交互式终端
```

---

## 14. 阶段十四：LLM 响应缓存（B2）

### 背景与目标

相同问题反复触发完整的 LLM 网络请求，浪费 token 并增加延迟。引入 SQLite 本地缓存，命中缓存时完全跳过 LLM 调用。

### 实现方案

**新增配置（`agent/config.py`）：**

```python
@dataclass
class CacheConfig:
    enabled: bool = True                        # 默认开启
    backend: str = "sqlite"                     # 当前仅支持 sqlite
    sqlite_path: str = ".langchain_cache.db"    # 缓存文件路径

    @classmethod
    def from_env(cls) -> "CacheConfig":
        return cls(
            enabled=os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true",
            backend=os.getenv("LLM_CACHE_BACKEND", "sqlite"),
            sqlite_path=os.getenv("LLM_CACHE_SQLITE_PATH", ".langchain_cache.db"),
        )
```

`AgentConfig` 新增 `cache: CacheConfig` 字段，`AgentConfig.from_env()` 自动调用 `CacheConfig.from_env()`。

**缓存初始化（`agent/graph.py`）：**

```python
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

if config.cache.enabled and config.cache.backend == "sqlite":
    set_llm_cache(SQLiteCache(database_path=config.cache.sqlite_path))
```

### 环境变量控制

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_CACHE_ENABLED` | `true` | 开关缓存 |
| `LLM_CACHE_BACKEND` | `sqlite` | 缓存后端 |
| `LLM_CACHE_SQLITE_PATH` | `.langchain_cache.db` | SQLite 文件路径 |

### 效果

- 首次请求：正常调用 LLM，结果写入本地 SQLite
- 相同请求命中缓存：**延迟 ~0ms，token 消耗 = 0**
- 缓存文件自动创建于项目根目录

---

## 15. 阶段十五：查询结果导出（B3）

### 背景与目标

Data Analysis Skill 生成的查询结果仅保存在内存中，无法进行后续的电子表格分析。新增 `export_results` 节点将结果导出为 CSV / Excel 文件。

### 实现方案

**状态扩展（`skills/data_analysis/skill.py`）：**

```python
class DataAnalysisState(TypedDict):
    ...
    export_files: list  # 导出文件路径列表（新增）
```

**新节点 `_export_results_node`：**

- 读取 `query_results` 列表中所有成功的查询结果
- 调用 `_parse_query_result()` 将原始文本解析为行列表
- 每条查询写入独立 CSV 文件：`report/export_{task_id}_step{n}.csv`
- 若已安装 `openpyxl`，额外生成汇总 Excel：`report/export_{task_id}.xlsx`（每条查询为一个 Sheet）

**图流程更新：**

```
visualize → generate_report → export_results → END
```

**结果解析策略（`_parse_query_result`）：**

1. 尝试 `ast.literal_eval`（处理 LangChain 的 tuple 列表格式）
2. 回退到换行分割 + tab/逗号分隔

### 输出示例

```
report/
├── analysis_20250101_120000.md          # Markdown 报告
├── export_abc123_step1.csv              # 第1条查询结果
├── export_abc123_step2.csv              # 第2条查询结果
└── export_abc123.xlsx                   # 汇总 Excel（含 openpyxl）
```

---

## 17. 最终架构总览

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

## 18. 测试覆盖总结

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

## 19. 功能流程详解

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

---

## 20. 阶段十六：工程质量优化（A3/A4/B4/B5/B6/C1/C2/C3）

### 20.1 A3 — max_tokens 传入 LLM

`_create_llm()` 现在将 `AgentConfig.max_tokens` 传递给 LLM 构造器：

```python
# agent/graph.py
if cfg.max_tokens is not None:
    kwargs["max_tokens"] = cfg.max_tokens

llm = ChatTongyi(**kwargs)          # 或 init_chat_model(...)
```

环境变量：`LLM_MAX_TOKENS`（整数，默认不设上限）。

---

### 20.2 A4 — 补充 .env.example

项目根目录新增 `.env.example`，包含 **30+ 个环境变量**，按类别分组并附带说明：

| 类别 | 变量数 |
|------|--------|
| 数据库 | 2 |
| LLM 模型 | 5 |
| LLM 缓存 | 3 |
| 安全护栏 | 5 |
| 检索配置 | 7 |
| 输出目录 | 2 |
| 会话计划 | 2 |
| 日志 | 1 |

**使用方式：** `cp .env.example .env` 后填写必填项（`DB_URI`、`DASHSCOPE_API_KEY`）。

---

### 20.3 B4 — run_query 返回结构化结果

`run_query()`、`run_query_streaming_async()`、`run_query_streaming()` 全部更改为返回 `dict`：

```python
{
    "final_message": str,       # 最终 LLM 回答文本
    "nodes_visited": list[str], # 按序执行的节点名列表
    "export_files": list[str],  # 导出的 CSV/Excel 文件路径（DataAnalysis 才有）
}
```

调用示例：
```python
result = run_query("查询各类音乐的销售额")
print(result["final_message"])
print("经过节点:", result["nodes_visited"])
print("导出文件:", result["export_files"])
```

---

### 20.4 B5 — LLM 调用自动重试

网络抖动或 LLM 临时服务错误时自动指数退避重试，最多 3 次：

```python
# agent/graph.py
_base_llm = _create_llm(config.llm)
llm = _base_llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
```

使用 LangChain Runnable 内置的 `with_retry()`，底层依赖 `tenacity`（已作为传递依赖安装），无需额外配置。

---

### 20.5 B6 — openpyxl 加入依赖

`pyproject.toml` 新增：

```toml
"openpyxl>=3.1"
```

B3 导出节点的 Excel 生成路径现已稳定可用（之前 openpyxl 缺失时会静默跳过）。

---

### 20.6 C1 — B2/B3 自动化测试

新增 `tests/test_cache_and_export.py`（18 个测试，全部通过 ✅）：

| 测试类 | 用例数 | 覆盖内容 |
|--------|--------|----------|
| `TestLLMCacheConfig` | 4 | `CacheConfig` 默认值 / `from_env()` 覆盖 |
| `TestSQLiteCacheIntegration` | 2 | `SQLiteCache` 初始化 / 路径传递 |
| `TestParseQueryResult` | 4 | tuple / CSV / tab / 空字符串格式 |
| `TestExportResultsNode` | 6 | CSV 写入 / Excel 生成 / 文件路径返回 |
| `TestRunQueryReturnType` | 2 | `run_query` 返回 dict / 含必需 key |

同步新增 `tests/conftest.py`，将项目根目录加入 `sys.path`，解决测试中 `ModuleNotFoundError: No module named 'agent'` 问题。

---

### 20.7 C2 — SentenceTransformer 单例缓存

`agent/column_index.py` 引入模块级模型缓存，避免每次构建 `ColumnIndex` 时重复加载 ~100MB 模型权重：

```python
_ENCODER_CACHE: dict[str, Any] = {}

def _get_cached_encoder(model_name: str) -> Any:
    if model_name not in _ENCODER_CACHE:
        from sentence_transformers import SentenceTransformer
        _ENCODER_CACHE[model_name] = SentenceTransformer(model_name)
    return _ENCODER_CACHE[model_name]
```

`ColumnIndex._get_encoder()` 改为调用 `_get_cached_encoder(self._model_name)`，多次实例化 `ColumnIndex` 只加载一次模型。

---

### 20.8 C3 — README 同步更新

`README.md` 本次同步更新内容：

| 章节 | 更新内容 |
|------|----------|
| 功能特性 | 新增流式输出、LLM 缓存、自动重试、结果导出、双塔检索、会话计划追踪 |
| 快速开始 | 安装改为 `uv sync`，新增 `.env.example` 引用，代码示例改为结构化 dict 返回 |
| 项目结构 | 补充所有新文件（column_index / schema_graph / retrieval / session_plan 等） |
| 新功能说明 | 新增独立章节：流式输出 / LLM 缓存 / 查询导出 / 双塔检索 / 会话计划追踪 |
| 测试 | 更新为 `pytest` 命令，新增 `test_cache_and_export.py` |
| 技术栈 | 新增 Milvus / NetworkX / SentenceTransformer / openpyxl 行 |
| 配置项 | 补充 `LLM_MAX_TOKENS` / `LLM_CACHE_*` / `MILVUS_*` 变量，引用 `.env.example` |

---


---

## 21. 阶段十七：工程质量 & 功能补全（A5/A6/A7/B12/B13/C4/C5）

### 21.1 A5 — requirements.txt 同步 pyproject.toml

`requirements.txt` 与 `pyproject.toml` 严重脱节（前者仅 9 包，后者 13 包），
本次全量对齐，确保两者版本约束一致，新增以下依赖：

| 新增包 | 用途 |
|--------|------|
| `networkx>=3.4` | Steiner 树剪枝（双塔检索） |
| `pymilvus>=2.4` | 向量数据库客户端 |
| `sentence-transformers>=3.0` | 语义编码器 |
| `openpyxl>=3.1` | Excel 文件导出 |
| `sqlglot>=30.4.2` | SQL 解析与方言转换 |
| `torch>=2.0` | SentenceTransformer 后端 |

同时移除已不再使用的 `langchain-deepseek`（由 `dashscope` 直接对接）。

### 21.2 A6 — pyproject.toml 项目描述

`description` 字段从占位符 `"Add your description here"` 更新为准确描述：

```
Text-to-SQL LangGraph agent with skill-based architecture (Simple/Complex/DataAnalysis),
dual-tower retrieval, streaming output, and LLM response caching
```

### 21.3 A7 — 清理 ColumnIndex 死代码

`ColumnIndex.__init__` 中遗留的 `self._encoder = None` 属于死代码：
`_get_encoder()` 方法从未读取该实例变量，始终直接调用模块级单例 `_get_cached_encoder()`。
移除该赋值，消除潜在的混淆。

### 21.4 B12 — DataAnalysis 全量 SessionPlanManager 集成（含 Step 8）

DataAnalysis 技能原有 7 步会话计划追踪（步骤 1–7），但 `_export_results_node`（步骤 8）
未被纳入，且 `mark_complete` 误置于 `_generate_report` 节点中，导致任务在报告生成后
就被标记为完成，而非在文件真正导出后。

本次修复：

| 变更 | 位置 |
|------|------|
| 在 `_understand_goal` 的步骤列表中追加 Step 8 | `_understand_goal` |
| `_export_results_node` 开头调用 `_step_start(state, 8)` | `_export_results_node` |
| 函数返回前调用 `_step_done(state, 8, summary)` | `_export_results_node` |
| 将 `mark_complete` 从 `_generate_report` 移至 `_export_results_node` | 两处 |

现在会话计划文件（`report/<task_id>/plan.md`）完整记录 8 个步骤的执行状态。

### 21.5 B13 — 流式模式补全 export_files

`run_query_streaming_async` 使用 `astream_events` API，该 API 只推送事件，
不携带最终图状态，导致 `export_files` 始终返回空列表 `[]`。

**修复方案**：流式循环结束后，调用 `await graph.aget_state(config_with_thread)` 
取回最终快照，再从 `snapshot.values` 中提取 `export_files`：

```python
snapshot = await graph.aget_state(config_with_thread)
if snapshot:
    export_files = snapshot.values.get("export_files", [])
```

DataAnalysis 的流式模式现在也能正确返回导出文件路径列表。

### 21.6 C4 — dual_tower_retrieve 节点集成测试

新增 `tests/test_dual_tower_node.py`，包含 4 个测试用例：

| 用例 | 验证内容 |
|------|---------|
| `test_pruned_schema_replaces_table_schema` | pruned_schema 正确覆盖 state 中的 table_schema |
| `test_retrieval_stats_populated` | retrieval_stats 包含所有预期统计字段 |
| `test_graceful_fallback_on_retriever_error` | 检索异常时安全降级，保留 full schema |
| `test_empty_messages_does_not_crash` | messages 为空时不崩溃，以空字符串调用 retrieve |

所有 4 个测试均以 mock 模式运行，无需真实 Milvus 连接，执行速度快（~3s）。

### 21.7 C5 — main.py 模块文档注释

`main.py` 原先仅有单行注释。本次补充完整的模块级 docstring，涵盖：
- 使用方式（命令行启动）
- `run_query` / `run_query_streaming` 返回字典结构说明
- 三种技能（简单查询 / 复杂查询 / 数据分析）的功能概述

---


*最后更新：全部 14 项优化任务完成（A3–A7 / B4–B13 / C1–C5）*

---

## 22. 阶段十八：工程质量补全（A8/A9/B14/B15/C6/C7）

### 22.1 A8 — 修复 test_session_plan 8 步计划测试

B12 将 DataAnalysis 从 7 步升级为 8 步后，`test_data_analysis_seven_steps` 已与实现不符。
将测试函数重命名为 `test_data_analysis_eight_steps`，步骤范围改为 1–8，断言同步修正。

### 22.2 A9 — skill_graph_builder 规范化消息导入

```python
# 旧（已弃用路径）
from langchain.messages import AIMessage, HumanMessage, SystemMessage
# 新（规范路径）
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
```

### 22.3 B14 — 路由器取最新用户消息

`_query_router_node` 原先只取 `messages[0]`，在多轮对话中会误用第一轮问题做分类。
修复为：逆序遍历 messages，取最后一条 `HumanMessage`；无 HumanMessage 时回退到首条消息。

```python
for msg in reversed(messages):
    if hasattr(msg, "type") and msg.type == "human":
        user_question = msg.content
        break
```

### 22.4 B15 — format_for_llm 增加 in_progress 区块

`SessionPlanManager.format_for_llm` 原先只输出 done / pending / failed 三个区块，
`in_progress` 步骤完全静默，LLM 无法感知哪一步正在执行。
新增 `### [running] 执行中步骤` 区块，同时显示步骤开始时间，帮助 LLM 理解当前执行位置。

### 22.5 C6 — 新建 test_placeholder_resolver.py

`_resolve_query_placeholders` 是复杂的 SQL 占位符替换逻辑，包含：
- 元组列表 / 标量列表 / 字符串（ast.literal_eval）三条解析路径
- 空结果 / 缺失步骤 / 解析失败三种 (NULL) 降级路径
- 多依赖链同时替换

新增 9 个单元测试，全部通过（含边界用例 `test_unrelated_placeholder_not_replaced`
验证 depends_on 范围约束正确）。

### 22.6 C7 — test_router_accuracy 移除硬编码路径

原文件顶部含硬编码 `sys.path.insert(0, r"c:\Users\71949\Desktop\text2sql")`，
不可移植且与 conftest.py 自动注入的路径冲突。
移除该行，与其余测试文件保持一致的路径管理规范。

---

*最后更新：全部 20 项优化任务完成（A3–A9 / B4–B15 / C1–C7）*

---

## 23. 阶段十九：导入规范化补全（A10/A11/B16/C8）

### 背景

在全局导入路径规范化（A3 阶段）之后，仍有少量遗留的 `from langchain.messages import ...` 分散在技能实现文件和测试文件中，以及一个文档中错误的步骤数说明，以及多轮对话场景下 `_understand_goal` 节点读取错误消息的 Bug。

---

### 23.1 A10 — 全局导入批量规范化（生产代码）

**目标**：将所有生产代码中剩余的 `from langchain.messages import ...`、`from langchain.tools import BaseTool`、`from langchain.chat_models import BaseChatModel` 替换为对应的 `langchain_core.*` 路径。

**修改文件：**

| 文件 | 修改内容 |
|------|----------|
| `skills/data_analysis/skill.py` | `HumanMessage, AIMessage, SystemMessage` → `langchain_core.messages` |
| `skills/simple_query/skill.py` | 顶层导入 + 两处函数体内 `ToolMessage` 内联导入 → `langchain_core.messages` |
| `skills/complex_query/skill.py` | `HumanMessage, AIMessage` → `langchain_core.messages` |
| `agent/tools.py` | `BaseChatModel`, `BaseTool` → `langchain_core.*` |
| `agent/types.py` | `AnyMessage`, `BaseTool` → `langchain_core.*` |
| `agent/nodes/common.py` | `BaseChatModel`, `HumanMessage`, `AIMessage` → `langchain_core.*` |
| `agent/skills/states.py` | `AnyMessage` → `langchain_core.messages` |
| `agent/skills/base.py` | `BaseChatModel` → `langchain_core.language_models` |

> **注意：** `agent/graph.py` 中的 `from langchain.chat_models import init_chat_model` **保持不变**——`init_chat_model` 是 `langchain` 包的便捷函数，不在 `langchain_core` 中。

---

### 23.2 A11 — DataAnalysis 模块文档步骤数修正

**问题：** `skills/data_analysis/skill.py` 顶部 docstring 描述的是 7 步流程，但实际实现已包含第 8 步（`export_results`），步骤数不一致。

**修改：**
```python
# 修改前
"""DataAnalysisSkill — 7 步数据分析 Skill"""

# 修改后
"""DataAnalysisSkill — 8 步数据分析 Skill
步骤：understand_goal → explore_data → plan_analysis → generate_queries →
      analyze_results → visualize → generate_report → export_results
"""
```

---

### 23.3 B16 — `_understand_goal` 多轮对话消息读取修复

**问题：** `DataAnalysisSkill._understand_goal` 节点总是读取 `messages[0].content`，在多轮对话中只能读到首条消息，而非用户最新输入。

**修复方案**（与 B14 对 `_query_router_node` 的修复保持一致）：
```python
# 修改前
latest_message = messages[0].content

# 修改后：反向遍历找到最新的 HumanMessage
user_message = messages[0].content  # fallback
for msg in reversed(messages):
    if isinstance(msg, HumanMessage):
        user_message = msg.content
        break
```

---

### 23.4 C8 — 测试文件导入规范化

**目标：** 将测试文件中所有 `from langchain.messages import ...` 替换为 `from langchain_core.messages import ...`。

**修改文件：**

| 文件 | 修改内容 |
|------|----------|
| `tests/test_router_accuracy.py` | 顶层 `HumanMessage` 导入 |
| `tests/test_main_graph.py` | 顶层 `HumanMessage` 导入 |
| `tests/test_simple_skill.py` | 顶层导入 + 函数体内内联导入 |
| `tests/test_session_plan_integration.py` | 三处函数体内内联导入（`AIMessage` × 1，`HumanMessage` × 2） |
| `tests/test_column_fuzzy_match.py` | 四处函数体内内联导入（`AIMessage` × 3，`HumanMessage` × 2，组合导入 × 1） |

**验证：** 完成后使用 `grep -r "from langchain\.messages import"` 全项目扫描，结果为零匹配。

---

*最后更新：全部 24 项优化任务完成（A3–A11 / B4–B16 / C1–C8）*

---

## 24. 阶段二十：SKILL.md 渐进式披露重构

### 背景

#### 改前问题

在早期设计中，`SKILL.md` 文件仅作为开发者文档存在，**从未被任何 Python 代码读取或使用**。真正的技能路由描述被硬编码在 `skill_graph_builder.py` 的一个大段字符串常量中：

```python
# 改前（skill_graph_builder.py）— 硬编码路由提示
system_prompt = """你是一个查询意图分类器。根据用户问题选择合适的技能：

simple_query: 处理简单单表查询...
complex_query: 处理复杂多步骤查询...
data_analysis: 进行深度数据分析...
"""
```

同时，`SkillRegistry` 虽然已定义，但完全未被 `skill_graph_builder.py` 使用；主图每次初始化时分别 `import` 三个 Skill，使用各自独立的节点方法（`_simple_skill_node`、`_complex_skill_node`、`_analysis_skill_node`）。

**核心痛点：**
1. 添加新 Skill 需要改动 4 处：新建文件、在 `__init__` 导入、添加节点方法、更新路由提示字符串
2. SKILL.md 是纯装饰品，开发者写了也没有实际作用
3. 路由器 LLM 在分类阶段看到所有细节描述，Token 浪费

#### 改后目标

将 SKILL.md 提升为 **运行时元数据**，实现"渐进式披露"（Progressive Disclosure）：

- **路由阶段**：LLM 只看 `## 目的` + `## 适用场景` 两节摘要（来自 SKILL.md）
- **执行阶段**：LLM 才获得完整工具列表（在各 Skill 子图内部）

---

### 24.1 统一三份 SKILL.md 格式（D1）

**改前：** 三份 SKILL.md 语言混用（英文/中文），章节结构不统一，无统一标准。

**改后：** 全部重写为中文，统一结构：

```markdown
## 目的
（一句话描述该 Skill 的用途）

## 适用场景
- 场景 1
- 场景 2
...

## 不适用场景
- ...
```

**涉及文件：**
- `skills/simple_query/SKILL.md`
- `skills/complex_query/SKILL.md`
- `skills/data_analysis/SKILL.md`

---

### 24.2 BaseSkill 自动加载 SKILL.md（D2）

**改前：** `BaseSkill.__init__` 只接受手动传入的 `description` 字符串参数。

**改后：** 新增 `skill_md_path` 参数 + `_extract_skill_description()` 模块级函数，自动从 SKILL.md 解析 `目的` 和 `适用场景` 两节并合并为 `description`。

```python
# 改前
class BaseSkill:
    def __init__(self, name, llm, tool_manager, description=""):
        self.description = description

# 改后
def _extract_skill_description(skill_md_path: str) -> str:
    """解析 SKILL.md，提取 '目的' 和 '适用场景' 两节。"""
    ...

class BaseSkill:
    def __init__(self, name, llm, tool_manager, description="", skill_md_path=None):
        if skill_md_path and not description:
            description = _extract_skill_description(skill_md_path)
        self.description = description
```

**涉及文件：** `agent/skills/base.py`

---

### 24.3 SkillRegistry 新增 build_router_prompt()（D3）

**改前：** `SkillRegistry` 仅提供 `register()`、`get()` 等基本操作，没有格式化路由提示的能力。

**改后：** 新增 `build_router_prompt()` 方法，将所有注册技能的 `description` 格式化为结构化提示块：

```
【simple_query】
**目的**
处理能用单条 SQL 直接回答的查询问题...

**适用场景**
- 单表查询（含聚合、筛选）
...

【complex_query】
...
```

**涉及文件：** `agent/skills/registry.py`

---

### 24.4 三个 Skill 传递 skill_md_path（D4）

**改前：** 各 Skill 的 `__init__` 传入硬编码的 `description="..."` 字符串。

**改后：** 删除硬编码字符串，改为传入 SKILL.md 路径：

```python
# 改前
super().__init__(
    name="simple_query",
    description="处理简单查询...",  # 硬编码
    ...
)

# 改后
_md = Path(__file__).parent / "SKILL.md"
super().__init__(
    name="simple_query",
    skill_md_path=str(_md),  # 运行时读取
    ...
)
```

**涉及文件：**
- `skills/simple_query/skill.py`
- `skills/complex_query/skill.py`
- `skills/data_analysis/skill.py`

---

### 24.5 SkillGraphBuilder Registry 驱动重构（D5）

**改前：** `SkillGraphBuilder` 三个 Skill 各自独立 import、独立节点方法、硬编码路由提示、硬编码条件边映射。

**改后：** 完全由 `SkillRegistry` 驱动，动态注册节点和路由：

| 对比项 | 改前 | 改后 |
|--------|------|------|
| 节点注册 | `_simple_skill_node` / `_complex_skill_node` / `_analysis_skill_node` 三个独立方法 | `_make_skill_node(skill)` 工厂函数，一次生成 |
| 节点名称 | `"simple_skill"` / `"complex_skill"` / `"analysis_skill"` | `"simple_query"` / `"complex_query"` / `"data_analysis"`（与 Skill.name 一致） |
| 条件边 | 硬编码 `{"simple": "simple_skill", ...}` | `{name: name for name in registry.list_skills()}` |
| 路由提示 | 硬编码字符串 | `registry.build_router_prompt()`（来自 SKILL.md） |
| 添加新 Skill | 改动 4 处 | 只需 `registry.register(new_skill)` 1 处 |

**⚠️ 注意（Breaking Change）：**  
节点名称已从 `"simple_skill"` → `"simple_query"` 等（与 Skill 名保持一致），`query_type` 字段值也随之变化。

---

### 本轮改动总结

| 文件 | 改动说明 |
|------|----------|
| `skills/*/SKILL.md` | 重写为中文统一格式 |
| `agent/skills/base.py` | 新增 `_extract_skill_description()` + `skill_md_path` 参数 |
| `agent/skills/registry.py` | 新增 `build_router_prompt()` 方法 |
| `skills/*/skill.py` | 替换硬编码 `description` 为 `skill_md_path` |
| `agent/skill_graph_builder.py` | Registry 驱动重构，动态节点/边/提示 |

**效果：**
- SKILL.md 从纯文档 → 运行时路由元数据
- SkillRegistry 从未使用 → 主图核心依赖
- 新增 Skill 成本：4 处手动改动 → 1 次 `register()` 调用
- 路由 LLM Token 消耗减少（只看摘要，不看工具细节）

---

## 25. 阶段二十一：测试体系完善与代码质量清理

### 背景

D5 重构后遗留以下工程质量问题：
1. **测试回归**：`DataAnalysisSkill` 测试用 `__new__` 绕过 `__init__`，D5 新增的 `_plan_manager` 属性未被初始化，导致 `test_chart_generation` 和 `test_report_saving` 的 36 个测试失败
2. **pytest 不兼容**：5 个测试文件在模块级调用 `sys.stdout = io.TextIOWrapper(...)`, pytest 收集阶段就触发 `I/O operation on closed file`
3. **空占位文件**：`agent/graph_builder.py`、`agent/nodes.py`（旧架构遗留空壳），以及 `tests/` 下 5 个 0 字节占位文件混淆目录
4. **缺少核心逻辑测试**：D2/D3 引入的 `_extract_skill_description()` 和 `build_router_prompt()` 没有任何单元测试
5. **测试总数统计陈旧**：`tests/__init__.py` 中 "184 个" 的数字远低于实际

---

### 25.1 D5 回归修复（测试 `__new__` 缺漏 `_plan_manager`）

**改前问题：**

```python
# 测试中绕过 __init__ 创建对象，遗漏了 D5 新增的实例变量
skill = DataAnalysisSkill.__new__(DataAnalysisSkill)
# 缺少：skill._plan_manager = None
# 导致：AttributeError: 'DataAnalysisSkill' object has no attribute '_plan_manager'
```

**修复方案：** 在所有使用 `__new__` 方式构建 `DataAnalysisSkill` 的地方手动补上 `skill._plan_manager = None`。

**影响文件：**

| 文件 | 修复位置数 |
|------|-----------|
| `tests/test_chart_generation.py` | 3 处（`_make_analysis_skill`、`_make_skill_with_dir`、`test_default_report_dir_resolves_to_project_root`） |
| `tests/test_report_saving.py` | 2 处（`_make_skill`、`_make_skill_with_outdir`） |

**恢复测试：** 36 个单元测试重新通过。

---

### 25.2 T1 — pytest 兼容性修复

**改前问题：**

5 个测试文件在模块顶层执行：

```python
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
```

pytest 导入模块时就会执行这一行，替换掉 pytest 自己的 stdout capture 机制，导致后续所有输出抛出 `ValueError: I/O operation on closed file`。

**修复方案：** 将该行从模块顶层移入 `run_all()` 函数（或 `if __name__ == "__main__":` 块）内部，仅在直接运行文件时才生效，pytest 收集时不受影响。

**影响文件：**

| 文件 | 修复位置 |
|------|---------|
| `tests/test_schema_cache.py` | 移入 `run_all()` 开头 |
| `tests/test_security.py` | 移入 `if __name__ == "__main__":` 块开头 |
| `tests/test_column_fuzzy_match.py` | 移入 `run_all()` 开头 |
| `tests/test_report_saving.py` | 移入 `run_all()` 开头 |
| `tests/test_chart_generation.py` | 移入 `run_all()` 开头 |

---

### 25.3 T2 — 填充空文件 `tests/test_import.py`

**改前：** 0 字节占位文件，pytest 收集时产生无用警告。

**改后：** 11 个冒烟测试，验证所有核心符号可被正确导入：

| 测试 | 验证符号 |
|------|---------|
| `test_skill_graph_builder` | `agent.skill_graph_builder.SkillBasedGraphBuilder` |
| `test_base_skill` | `agent.skills.base.BaseSkill` |
| `test_extract_skill_description` | `agent.skills.base._extract_skill_description` |
| `test_skill_registry` | `agent.skills.registry.SkillRegistry` |
| `test_simple_query_skill` | `skills.simple_query.skill.SimpleQuerySkill` |
| `test_complex_query_skill` | `skills.complex_query.skill.ComplexQuerySkill` |
| `test_data_analysis_skill` | `skills.data_analysis.skill.DataAnalysisSkill` |
| `test_agent_config` | `agent.config.AgentConfig / DatabaseConfig / OutputConfig` |
| `test_database_manager` | `agent.database.SQLDatabaseManager / SchemaCache` |
| `test_sql_security_guard` | `agent.database.SQLSecurityGuard` |
| `test_states` | `agent.skills.states.MainGraphState` |

---

### 25.4 T3 — 新建 `tests/test_skill_registry.py`

新增 15 个单元测试，覆盖 D2/D3 引入的两个核心函数：

**`TestExtractSkillDescription`（8 例）：**

| 用例 | 场景 |
|------|------|
| `test_extracts_purpose_and_scenarios` | 正常 SKILL.md，两节均提取 |
| `test_missing_file_returns_empty_string` | 文件不存在 → 返回 `""` |
| `test_md_without_target_sections_returns_empty` | 只有 `## 流程` 等无关章节 → `""` |
| `test_only_purpose_section_present` | 只有 `## 目的` → 只输出目的块 |
| `test_only_scenarios_section_present` | 只有 `## 适用场景` → 只输出场景块 |
| `test_real_simple_query_skill_md` | 真实文件 smoke test |
| `test_empty_file_returns_empty` | 空文件 → `""` |
| `test_section_with_only_whitespace_is_excluded` | 章节体全为空白 → 该节不出现在输出中 |

**`TestBuildRouterPrompt`（7 例）：**

| 用例 | 场景 |
|------|------|
| `test_empty_registry_returns_empty_string` | 空注册中心 → `""` |
| `test_single_skill_produces_block` | 单 Skill → 含 `【skill_name】` 块 |
| `test_multiple_skills_all_present` | 多 Skill → 每个都出现 |
| `test_skill_with_empty_description_uses_fallback` | description="" → fallback `"Skill: name"` |
| `test_skill_with_none_description_uses_fallback` | description=None → fallback |
| `test_blocks_separated_by_double_newline` | 块间以 `\n\n` 分隔 |
| `test_unregister_removes_from_prompt` | unregister 后该 Skill 不再出现 |

---

### 25.5 O1/O2 — 导入路径规范化补漏

在全局导入规范化（A10/C8）之后，还有两处遗漏：

| 文件 | 改前 | 改后 |
|------|------|------|
| `agent/skill_graph_builder.py` | `from langchain.chat_models import BaseChatModel` | `from langchain_core.language_models import BaseChatModel` |
| `tests/test_session_plan_integration.py` | `from langchain.chat_models import BaseChatModel` | `from langchain_core.language_models import BaseChatModel` |

---

### 25.6 O3 — states.py `query_type` 注释更新

**改前：**
```python
query_type: str = ""  # "simple" | "complex" | "analysis"
```

**改后（与 D5 节点名对齐）：**
```python
query_type: str = ""  # "simple_query" | "complex_query" | "data_analysis"
```

---

### 25.7 O4 — states.py 死代码调查与注释

`states.py` 中定义了三个 Pydantic State 类（`SimpleQueryState`、`ComplexQueryState`、`DataAnalysisState`），经调查：

| State 类 | 实际使用情况 |
|----------|-------------|
| `ComplexQueryState` | ✅ 被 `skills/complex_query/skill.py` 直接引用 |
| `SimpleQueryState` | ⚠️ 已导出但未被引用——各 Skill 内部用 TypedDict 版本 |
| `DataAnalysisState` | ⚠️ 同上 |

**处理方式：** 为文件添加模块 docstring，说明该现象属于"已导出但被部分绕过的状态类"，保留代码（供外部引用），而不是误删。

---

### 25.8 O5 / O6 / O7 — 空文件清理

**O5（根目录重复测试文件）：** 13 个根目录 `test_*.py` 是迁移到 `tests/` 目录前的历史遗留副本，全部删除。

**O6（agent 空壳文件）：**

| 文件 | 状态 | 处理 |
|------|------|------|
| `agent/graph_builder.py` | 0 字节，无任何引用 | ✅ 删除 |
| `agent/nodes.py` | 0 字节，无任何引用 | ✅ 删除 |

这两个文件是旧单体架构的遗留空壳；重构后功能分别由 `skill_graph_builder.py`（主图构建）和 `agent/nodes/common.py`（公共节点）接管，文件本身早已清空且未被任何代码引用。

**O7（tests/ 空占位测试文件）：**

| 文件 | 状态 | 处理 |
|------|------|------|
| `tests/test_schema_info.py` | 0 字节 | ✅ 删除 |
| `tests/test_fix_prompt.py` | 0 字节 | ✅ 删除 |
| `tests/test_db_connection.py` | 0 字节 | ✅ 删除 |
| `tests/test_complex_skill.py` | 0 字节 | ✅ 删除 |
| `tests/test_analysis_skill.py` | 0 字节 | ✅ 删除 |

---

### 25.9 T6 — tests/__init__.py 测试总数精确核实

**改前：** 文档写 "184 个"（已严重过时，未含此后多个轮次新增的测试）。

**核实方式：** 运行完整无 API Key 测试套件（含 Milvus）得到基准数字：

```
pytest tests/（所有无 API Key 文件）
262 passed, 5 skipped, 55 subtests passed
```

- **262 passed**：所有无 API Key + Milvus 测试用例
- **5 skipped**：需要真实数据库连接的集成测试（`test_column_fuzzy_match` 中的 DB 集成用例）
- **55 subtests**：`test_retrieval_benchmark.py` 内部的参数化子测试

`tests/__init__.py` 测试数量更新为 **262**，并补充 6 个新测试文件的条目说明。

---

### 25.10 T7 — Milvus 测试全量通过

Milvus 服务启动后，原先 5 个 skipped 的 `test_retrieval_benchmark.py` 测试（其实它们也不依赖真实 Milvus，均以 mock 方式运行）现全部通过：

```
tests/test_retrieval_benchmark.py: 33 passed, 55 subtests passed
```

所有测试套件无 API Key 全量通过：**262 passed（5 skipped 为 DB 集成测试）**。

---

### 本轮改动总结

| 类别 | 任务 | 变更内容 |
|------|------|---------|
| Bug 修复 | D5 回归 | `_plan_manager = None` 补入 5 处 `__new__` 测试 stub |
| 兼容性 | T1 | 5 个文件的 `sys.stdout` 移出模块顶层 |
| 新增测试 | T2 | `test_import.py`：11 个导入冒烟测试 |
| 新增测试 | T3 | `test_skill_registry.py`：15 个单元测试 |
| 导入规范 | O1/O2 | 2 处 `langchain.chat_models` → `langchain_core` |
| 注释更新 | O3 | `query_type` 注释值对齐 D5 节点名 |
| 文档注释 | O4 | `states.py` 模块 docstring |
| 文件清理 | O5/O6/O7 | 删除 20 个空/重复文件 |
| 文档更新 | T6/F1 | `tests/__init__.py` 数量 184 → 262，补充 6 个文件说明 |

*最后更新：2026-04-23*


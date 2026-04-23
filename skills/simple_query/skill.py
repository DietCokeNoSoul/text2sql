"""Simple Query Skill - 处理单表或简单多表查询。

此 Skill 复用现有的 generate_query, check_query, run_query 节点逻辑。
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from agent.skills.base import BaseSkill
from agent.tools import SQLToolManager
from agent.database import SQLDatabaseManager

logger = logging.getLogger(__name__)


class SimpleQuerySkill(BaseSkill):
    """简单查询 Skill。
    
    工作流: 
      list_tables → get_schema → generate_query → check_query → run_query
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tool_manager: SQLToolManager,
        db_manager: SQLDatabaseManager
    ):
        """初始化 Simple Query Skill。
        
        参数:
            llm: 语言模型
            tool_manager: 工具管理器
            db_manager: 数据库管理器
        """
        self.db_manager = db_manager
        
        _md = Path(__file__).parent / "SKILL.md"
        super().__init__(
            name="simple_query",
            llm=llm,
            tool_manager=tool_manager,
            skill_md_path=str(_md),
        )
    
    def _build_graph(self) -> StateGraph:
        """构建简单查询子图。
        
        流程: list_tables → get_schema → generate_query → execute_query 
              → [成功→END, 失败→fix_query→execute_query (最多3次)]
        """
        from typing_extensions import TypedDict
        from typing import Annotated
        from langgraph.graph import add_messages
        
        # 定义自定义 State（包含重试计数）
        class SimpleQueryState(TypedDict):
            messages: Annotated[list, add_messages]
            retry_count: int
            last_error: str
            last_sql: str
        
        builder = StateGraph(SimpleQueryState)
        
        # 添加节点
        builder.add_node("list_tables", self.common.create_list_tables_node())
        builder.add_node("get_schema", self.common.create_get_schema_node())
        builder.add_node("generate_query", self._generate_query)
        builder.add_node("execute_query", self._execute_with_error_capture)
        builder.add_node("fix_query", self._fix_query)
        
        # 定义流程
        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "get_schema")
        builder.add_edge("get_schema", "generate_query")
        builder.add_edge("generate_query", "execute_query")
        
        # 条件路由：根据执行结果决定是结束还是修复
        builder.add_conditional_edges(
            "execute_query",
            self._should_retry,
            {
                "fix": "fix_query",
                "end": END
            }
        )
        
        builder.add_edge("fix_query", "execute_query")
        
        
        return builder.compile()
    
    # ========== 特有节点实现 ==========
    
    def _get_generate_system_prompt(self) -> str:
        """获取生成查询的系统提示。"""
        dialect = self.db_manager.get_dialect()
        max_results = self.db_manager.config.max_query_results
        
        return f"""
您是一个专用于与SQL数据库交互的智能体。

任务：根据用户问题生成并执行 {dialect.value} SQL 查询。

要求：
1. 生成语法正确的 {dialect.value} SQL 查询
2. 除非用户明确指定，否则限制结果在 {max_results} 条以内
3. 只查询必要的列，避免 SELECT *
4. 禁止使用 INSERT、UPDATE、DELETE、DROP 等修改数据的语句

**重要**: 您必须调用 sql_db_query 工具来执行查询。

调用示例：
- 查询用户：使用 sql_db_query 工具，参数为 {{"query": "SELECT id, nick_name FROM tb_user LIMIT 5"}}
- 查询商店：使用 sql_db_query 工具，参数为 {{"query": "SELECT * FROM tb_shop WHERE area = '静安区' LIMIT 10"}}

请立即使用 sql_db_query 工具执行查询。
        """.strip()
    
    def _generate_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点: 生成并执行 SQL 查询。"""
        try:
            logger.info("[SimpleQuery] Generating and executing SQL query")
            
            system_message = {
                "role": "system",
                "content": self._get_generate_system_prompt(),
            }
            
            # 绑定查询工具
            query_tool = self.tool_manager.get_query_tool()
            llm_with_tools = self.llm.bind_tools([query_tool], tool_choice="required")
            
            messages = state.get("messages", [])
            
            # 调用 LLM 生成工具调用
            response = llm_with_tools.invoke([system_message] + messages)
            
            # 检查是否有工具调用
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                logger.warning("[SimpleQuery] No tool calls generated, trying again with explicit instruction")
                # 如果没有工具调用，添加明确指令
                explicit_message = HumanMessage(
                    content="请使用 sql_db_query 工具来执行 SQL 查询。"
                )
                response = llm_with_tools.invoke([system_message] + messages + [explicit_message])
            
            logger.info("[SimpleQuery] Query generated successfully")
            
            messages.append(response)
            
            # 初始化重试计数
            return {
                "messages": messages,
                "retry_count": 0,
                "last_error": "",
                "last_sql": ""
            }
            
        except Exception as e:
            logger.error(f"[SimpleQuery] Error generating query: {e}")
            error_message = AIMessage(content=f"Error generating query: {str(e)}")
            messages = state.get("messages", [])
            messages.append(error_message)
            return {
                "messages": messages,
                "retry_count": 0,
                "last_error": str(e),
                "last_sql": ""
            }
    
    def _execute_with_error_capture(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点: 执行查询并捕获错误信息。"""
        try:
            logger.info("[SimpleQuery] Executing query with error capture")
            
            messages = state.get("messages", [])
            if not messages:
                return state
            
            last_message = messages[-1]
            
            # 提取 SQL 查询
            sql_query = ""
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_call = last_message.tool_calls[0]
                sql_query = tool_call.get("args", {}).get("query", "")
            
            # 执行查询
            try:
                result = self.db_manager.execute_query(sql_query)
                
                # 成功执行
                logger.info("[SimpleQuery] Query executed successfully")
                
                from langchain_core.messages import ToolMessage
                tool_message = ToolMessage(
                    content=result,
                    tool_call_id=last_message.tool_calls[0].get("id", ""),
                    name="sql_db_query"
                )
                
                messages.append(tool_message)
                
                return {
                    "messages": messages,
                    "retry_count": state.get("retry_count", 0),
                    "last_error": "",  # 清空错误
                    "last_sql": sql_query
                }
                
            except Exception as exec_error:
                # 执行失败，捕获错误
                error_msg = str(exec_error)
                logger.warning(f"[SimpleQuery] Query execution failed: {error_msg}")
                
                from langchain_core.messages import ToolMessage
                tool_message = ToolMessage(
                    content=f"Error: {error_msg}",
                    tool_call_id=last_message.tool_calls[0].get("id", ""),
                    name="sql_db_query"
                )
                
                messages.append(tool_message)
                
                return {
                    "messages": messages,
                    "retry_count": state.get("retry_count", 0),
                    "last_error": error_msg,
                    "last_sql": sql_query
                }
                
        except Exception as e:
            logger.error(f"[SimpleQuery] Error in execute_with_error_capture: {e}")
            return state
    
    def _should_retry(self, state: Dict[str, Any]) -> str:
        """条件边: 判断是否需要重试。"""
        retry_count = state.get("retry_count", 0)
        last_error = state.get("last_error", "")
        
        # 如果没有错误，结束
        if not last_error:
            logger.info("[SimpleQuery] Query succeeded, ending")
            return "end"
        
        # 如果重试次数超过3次，结束
        if retry_count >= 3:
            logger.warning(f"[SimpleQuery] Max retries ({retry_count}) reached, ending")
            return "end"
        
        # 需要修复
        logger.info(f"[SimpleQuery] Error detected, will retry (attempt {retry_count + 1}/3)")
        return "fix"
    
    def _get_fix_system_prompt(self) -> str:
        """获取修复查询的系统提示。"""
        dialect = self.db_manager.get_dialect()
        
        return f"""
您是一位专业的 SQL 错误诊断和修复专家。

**任务**: 直接生成修复后的 SQL 查询并执行，不要使用任何检查命令！

⚠️ **严格禁止**:
- ❌ 不要使用 DESCRIBE
- ❌ 不要使用 SHOW TABLES
- ❌ 不要使用 SHOW COLUMNS
- ❌ 不要使用任何 DDL 命令

✅ **正确做法**:
- ✅ 直接生成修复后的查询
- ✅ 参考之前消息中已有的 schema 信息
- ✅ 保留原查询的业务逻辑（WHERE、ORDER BY、LIMIT等）
- ✅ 使用 sql_db_query 工具执行修复后的查询

---

**修复流程**:

1. **分析错误类型**:
   - "Unknown column 'X'" → 列名错误，在 schema 中找正确列名
   - "Unknown table 'X'" → 表名错误，在表列表中找正确表名
   - "Syntax error" → SQL 语法错误，修正语法

2. **查找正确名称**:
   - 查看之前消息中的 schema 信息
   - 找到实际存在的列名/表名
   - 常见映射: shop_name → name, nickname → nick_name

3. **生成修复查询**:
   - 复制原查询
   - 只替换错误的列名/表名
   - 保留所有其他部分（WHERE、ORDER BY、LIMIT、JOIN等）

---

**修复示例**:

示例 1: 列名错误
```
原查询: SELECT shop_name, score FROM tb_shop LIMIT 3
错误: Unknown column 'shop_name'
Schema: tb_shop (id, name, score, area, ...)

分析: shop_name 不存在，应该是 name
修复: SELECT name, score FROM tb_shop LIMIT 3
      ^^^^^^ 只改这里
```

示例 2: 多个列名错误
```
原查询: SELECT nickname, user_name FROM tb_user WHERE id < 10
错误: Unknown column 'nickname'
Schema: tb_user (id, nick_name, phone, ...)

分析: nickname 应该是 nick_name, user_name 可能也不存在
修复: SELECT nick_name, phone FROM tb_user WHERE id < 10
      ^^^^^^^^^^^^
```

示例 3: 保留业务逻辑
```
原查询: SELECT shop_name FROM tb_shop WHERE area = '静安区' ORDER BY score DESC LIMIT 5
错误: Unknown column 'shop_name'
Schema: tb_shop (id, name, score, area)

修复: SELECT name FROM tb_shop WHERE area = '静安区' ORDER BY score DESC LIMIT 5
      ^^^^                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      只改列名                  保留所有过滤和排序逻辑
```

---

**关键提醒**:
1. Schema 信息已经在之前的消息中，不需要重新查询
2. 直接使用 sql_db_query 工具执行修复后的查询
3. 不要改变查询的业务含义，只修复错误
4. 优先使用最相似的列名（如 shop_name → name）

现在请基于错误信息生成修复后的查询并执行！
        """.strip()
    
    def _extract_bad_column(self, error_msg: str) -> Optional[str]:
        """从错误信息中提取出错的列名。
        
        支持格式:
        - MySQL:    Unknown column 'xxx' in 'field list'
        - SQLite:   table tb_xxx has no column named xxx
        - General:  column "xxx" does not exist
        """
        patterns = [
            r"Unknown column '([^']+)'",          # MySQL
            r"has no column named (\w+)",           # SQLite
            r'column "([^"]+)" does not exist',     # PostgreSQL
            r"Invalid column name '([^']+)'",       # MSSQL
        ]
        for pattern in patterns:
            m = re.search(pattern, error_msg, re.IGNORECASE)
            if m:
                # 如果提取到的是 table.column 格式，只取列名部分
                col = m.group(1)
                return col.split(".")[-1] if "." in col else col
        return None

    def _build_column_hint(self, error_msg: str) -> str:
        """解析错误并构建列名建议提示，供 _fix_query 注入。"""
        bad_col = self._extract_bad_column(error_msg)
        if not bad_col:
            return ""
        
        suggestions = self.db_manager.find_similar_columns(bad_col)
        if not suggestions:
            # 没有模糊匹配，给出所有表的列名供 LLM 参考
            column_map = self.db_manager.get_column_map()
            lines = [f"  - {tbl}: {', '.join(cols)}" for tbl, cols in column_map.items() if cols]
            cols_text = "\n".join(lines) if lines else "（无法获取列信息）"
            return f"""
**列名纠错提示**:
- 错误列名: `{bad_col}`
- 无法找到相似列名，以下是所有可用列，请从中选择最合适的：
{cols_text}
""".strip()
        
        suggestions_text = "\n".join(f"  - `{s}`" for s in suggestions)
        return f"""
**列名纠错提示**:
- 错误列名: `{bad_col}`
- 最相似的实际列名（优先选择这些）:
{suggestions_text}
""".strip()

    def _fix_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点: 引导式错误纠正（含列名模糊匹配建议）。"""
        try:
            retry_count = state.get("retry_count", 0)
            last_error = state.get("last_error", "")
            last_sql = state.get("last_sql", "")
            
            logger.info(f"[SimpleQuery] Fixing query (attempt {retry_count + 1}/3)")
            logger.info(f"[SimpleQuery] Last error: {last_error}")
            logger.info(f"[SimpleQuery] Last SQL: {last_sql}")
            
            # 构建列名模糊匹配建议
            column_hint = self._build_column_hint(last_error)
            if column_hint:
                logger.info(f"[SimpleQuery] Column hint generated: {column_hint[:200]}")
            
            system_message = {
                "role": "system",
                "content": self._get_fix_system_prompt(),
            }
            
            # 构建修复提示（含列名建议）
            column_section = f"\n\n{column_hint}" if column_hint else "\n\n（请查看之前消息中的 schema 信息）"
            fix_prompt = f"""
请修复以下 SQL 查询：

**原始查询**：
```sql
{last_sql}
```

**错误信息**：
{last_error}
{column_section}

请分析错误原因，并生成修复后的正确 SQL 查询。
使用 sql_db_query 工具执行修复后的查询。
            """.strip()
            
            fix_message = HumanMessage(content=fix_prompt)
            
            # 绑定查询工具
            query_tool = self.tool_manager.get_query_tool()
            llm_with_tools = self.llm.bind_tools([query_tool], tool_choice="required")
            
            messages = state.get("messages", [])
            response = llm_with_tools.invoke([system_message, fix_message])
            
            logger.info("[SimpleQuery] Fix query generated")
            
            messages.append(fix_message)
            messages.append(response)
            
            return {
                "messages": messages,
                "retry_count": retry_count + 1,
                "last_error": last_error,
                "last_sql": last_sql
            }
            
        except Exception as e:
            logger.error(f"[SimpleQuery] Error fixing query: {e}")
            messages = state.get("messages", [])
            error_message = AIMessage(content=f"Error fixing query: {str(e)}")
            messages.append(error_message)
            return {
                "messages": messages,
                "retry_count": state.get("retry_count", 0) + 1,
                "last_error": state.get("last_error", ""),
                "last_sql": state.get("last_sql", "")
            }

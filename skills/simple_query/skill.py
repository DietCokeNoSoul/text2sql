"""Simple Query Skill - 处理单表或简单多表查询。

此 Skill 复用现有的 generate_query, check_query, run_query 节点逻辑。
"""

import logging
from typing import Dict, List, Any

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, AnyMessage
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
        
        super().__init__(
            name="simple_query",
            llm=llm,
            tool_manager=tool_manager,
            description="处理单表或简单多表SQL查询"
        )
    
    def _build_graph(self) -> StateGraph:
        """构建简单查询子图。"""
        from langgraph.graph import MessagesState
        
        builder = StateGraph(MessagesState)
        
        # 添加节点（复用公共节点 + 特有节点）
        builder.add_node("list_tables", self.common.create_list_tables_node())
        builder.add_node("get_schema", self.common.create_get_schema_node())
        builder.add_node("generate_query", self._generate_query)
        builder.add_node("check_query", self._check_query)
        builder.add_node("run_query", self.common.create_execute_query_node())
        
        # 定义流程
        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "get_schema")
        builder.add_edge("get_schema", "generate_query")
        builder.add_edge("generate_query", "check_query")
        builder.add_edge("check_query", "run_query")
        builder.add_edge("run_query", END)
        
        return builder.compile()
    
    # ========== 特有节点实现 ==========
    
    def _get_generate_system_prompt(self) -> str:
        """获取生成查询的系统提示。"""
        dialect = self.db_manager.get_dialect()
        max_results = self.db_manager.config.max_query_results
        
        return f"""
您是一个专用于与SQL数据库交互的智能体。
根据输入的问题，请生成语法正确的 {dialect.value} 查询语句，
随后查看查询结果并返回答案。除非用户明确指定要获取的示例数量，
否则请始终将查询结果限制在最多 {max_results} 条。

您可以通过相关列对结果进行排序，以返回数据库中最有价值的示例。
切勿查询特定表的所有列，仅获取问题相关的必要列。

禁止对数据库执行任何数据操作语言语句（INSERT、UPDATE、DELETE、DROP等）。
        """.strip()
    
    def _generate_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点: 生成 SQL 查询。"""
        try:
            logger.info("[SimpleQuery] Generating SQL query")
            
            system_message = {
                "role": "system",
                "content": self._get_generate_system_prompt(),
            }
            
            # 绑定查询工具
            query_tool = self.tool_manager.get_query_tool()
            llm_with_tools = self.llm.bind_tools([query_tool])
            
            messages = state.get("messages", [])
            response = llm_with_tools.invoke([system_message] + messages)
            
            logger.info("[SimpleQuery] Query generated successfully")
            
            messages.append(response)
            return {"messages": messages}
            
        except Exception as e:
            logger.error(f"[SimpleQuery] Error generating query: {e}")
            error_message = AIMessage(f"Error generating query: {str(e)}")
            messages = state.get("messages", [])
            messages.append(error_message)
            return {"messages": messages}
    
    def _get_check_system_prompt(self) -> str:
        """获取检查查询的系统提示。"""
        dialect = self.db_manager.get_dialect()
        
        return f"""
您是一位注重细节的SQL专家。  
请仔细检查 {dialect.value} 查询中的常见错误，包括：  
- 在NOT IN子句中使用NULL值  
- 应当使用UNION ALL时却使用了UNION  
- 使用BETWEEN处理不包含边界的情况  
- 谓词中的数据类型不匹配  
- 正确引用标识符  
- 为函数使用正确数量的参数  
- 转换为正确的数据类型  
- 使用合适的列进行连接  

如果存在上述任何错误，请重写查询。如果没有错误，请直接返回原始查询。  

完成检查后，您将调用相应的工具来执行查询。
        """.strip()
    
    def _check_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """节点: 检查并验证 SQL 查询。"""
        try:
            logger.info("[SimpleQuery] Checking SQL query")
            
            # 从最后一条消息中获取查询
            messages = state.get("messages", [])
            if not messages:
                logger.error("[SimpleQuery] No messages found")
                error_message = AIMessage("Error: No messages to check")
                return {"messages": [error_message]}
            
            last_message = messages[-1]
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                logger.error("[SimpleQuery] No tool calls found")
                error_message = AIMessage("Error: No query found to validate")
                messages.append(error_message)
                return {"messages": messages}
            
            tool_call = last_message.tool_calls[0]
            query = tool_call["args"]["query"]
            
            system_message = {
                "role": "system",
                "content": self._get_check_system_prompt(),
            }
            
            user_message = {"role": "user", "content": query}
            
            # 绑定查询工具
            query_tool = self.tool_manager.get_query_tool()
            llm_with_tools = self.llm.bind_tools([query_tool])
            response = llm_with_tools.invoke([system_message, user_message])
            
            logger.info("[SimpleQuery] Query checked successfully")
            
            messages.append(response)
            return {"messages": messages}
            
        except Exception as e:
            logger.error(f"[SimpleQuery] Error checking query: {e}")
            error_message = AIMessage(f"Error checking query: {str(e)}")
            messages = state.get("messages", [])
            messages.append(error_message)
            return {"messages": messages}

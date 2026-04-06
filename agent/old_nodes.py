"""SQL Agent 图的节点实现。

此模块包含 SQL Agent 工作流中使用的所有图节点的实现。
"""

import logging
from typing import Dict, List, Literal

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, AnyMessage
from langgraph.graph import END, MessagesState
from langgraph.prebuilt import tools_condition

from .database import SQLDatabaseManager
from .tools import SQLToolManager
from .types import BaseNode


logger = logging.getLogger(__name__)


class ListTablesNode(BaseNode):
    """用于列出可用数据库表的节点。"""
    
    def __init__(self, tool_manager: SQLToolManager) -> None:
        """初始化列表表节点。
        
        参数:
            tool_manager: SQL 工具管理器实例
        """
        super().__init__("list_tables")
        self.tool_manager = tool_manager
    
    def execute(self, state: MessagesState) -> Dict[str, List[AnyMessage]]:
        """执行列表表操作。
        
        参数:
            state: 当前对话状态
            
        返回:
            包含表列表的更新状态
        """
        try:
            logger.info("Executing list tables node")
            
            # 创建一个预定的工具调用
            tool_call = {
                "name": "sql_db_list_tables",
                "args": {},
                "id": "list_tables_call",
                "type": "tool_call",
            }
            tool_call_message = AIMessage(content="", tool_calls=[tool_call])
            
            # 获取工具并执行
            list_tables_tool = self.tool_manager.get_list_tables_tool()
            tool_message = list_tables_tool.invoke(tool_call)
            
            # 创建响应消息
            response = AIMessage(f"Available tables: {tool_message.content}")
            
            logger.info(f"Listed tables successfully: {tool_message.content}")
            return {"messages": [tool_call_message, tool_message, response]}
            
        except Exception as e:
            logger.error(f"Error in list tables node: {e}")
            error_message = AIMessage(f"Error listing tables: {str(e)}")
            return {"messages": [error_message]}


class GetSchemaNode(BaseNode):
    """用于检索数据库架构信息的节点。"""
    
    def __init__(self, tool_manager: SQLToolManager, llm: BaseChatModel) -> None:
        """初始化获取架构节点。
        
        参数:
            tool_manager: SQL 工具管理器实例
            llm: 用于工具绑定的语言模型
        """
        super().__init__("get_relative_schema")
        self.tool_manager = tool_manager
        self.llm = llm
    
    def execute(self, state: MessagesState) -> Dict[str, List[AnyMessage]]:
        """执行架构检索操作。
        
        参数:
            state: 当前对话状态
            
        返回:
            包含架构信息的更新状态
        """
        try:
            logger.info("Executing get schema node")
            
            schema_tool = self.tool_manager.get_schema_tool() # 获取模式工具
            llm_with_tools = self.llm.bind_tools([schema_tool], tool_choice="auto") # 绑定工具但不强制使用
            response = llm_with_tools.invoke(state["messages"]) # 传递整个消息状态以提供上下文
            
            logger.info("Schema retrieval completed successfully")
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in get schema node: {e}")
            error_message = AIMessage(f"Error retrieving schema: {str(e)}")
            return {"messages": [error_message]}


class GenerateQueryNode(BaseNode):
    """用于基于用户问题生成 SQL 查询的节点。"""
    
    def __init__(
        self,
        tool_manager: SQLToolManager,
        llm: BaseChatModel,
        db_manager: SQLDatabaseManager
    ) -> None:
        """初始化生成查询节点。
        
        参数:
            tool_manager: SQL 工具管理器实例
            llm: 用于查询生成的语言模型
            db_manager: 用于方言信息的数据库管理器
        """
        super().__init__("generate_query")
        self.tool_manager = tool_manager
        self.llm = llm
        self.db_manager = db_manager
    
    def _get_system_prompt(self) -> str:
        """获取用于查询生成的系统提示。"""
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
    
    def execute(self, state: MessagesState) -> Dict[str, List[AnyMessage]]:
        """执行查询生成操作。
        
        参数:
            state: 当前对话状态
            
        返回:
            包含生成查询的更新状态
        """
        try:
            logger.info("Executing generate query node")
            
            system_message = {
                "role": "system",
                "content": self._get_system_prompt(),
            }
            
            # 绑定查询工具但不强制使用
            query_tool = self.tool_manager.get_query_tool()
            llm_with_tools = self.llm.bind_tools([query_tool])
            response = llm_with_tools.invoke([system_message] + state["messages"])
            
            logger.info("Query generation completed successfully")
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in generate query node: {e}")
            error_message = AIMessage(f"Error generating query: {str(e)}")
            return {"messages": [error_message]}


class CheckQueryNode(BaseNode):
    """用于验证和检查 SQL 查询的节点。"""
    
    def __init__(
        self,
        tool_manager: SQLToolManager,
        llm: BaseChatModel,
        db_manager: SQLDatabaseManager
    ) -> None:
        """初始化检查查询节点。
        
        参数:
            tool_manager: SQL 工具管理器实例
            llm: 用于查询验证的语言模型
            db_manager: 用于方言信息的数据库管理器
        """
        super().__init__("check_query")
        self.tool_manager = tool_manager
        self.llm = llm
        self.db_manager = db_manager
    
    def _get_system_prompt(self) -> str:
        """获取用于查询验证的系统提示。"""
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
    
    def execute(self, state: MessagesState) -> Dict[str, List[AnyMessage]]:
        """执行查询验证操作。
        
        参数:
            state: 当前对话状态
            
        返回:
            包含已验证查询的更新状态
        """
        try:
            logger.info("Executing check query node")
            
            # 从最后一条消息的工具调用中获取查询
            last_message = state["messages"][-1]
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                logger.error("No tool calls found in last message")
                error_message = AIMessage("Error: No query found to validate")
                return {"messages": [error_message]}
            
            tool_call = last_message.tool_calls[0]
            query = tool_call["args"]["query"]
            
            system_message = {
                "role": "system",
                "content": self._get_system_prompt(),
            }
            
            user_message = {"role": "user", "content": query}
            
            query_tool = self.tool_manager.get_query_tool()
            llm_with_tools = self.llm.bind_tools([query_tool], tool_choice="auto")
            response = llm_with_tools.invoke([system_message, user_message])
            
            # 保留原始消息 ID
            response.id = last_message.id
            
            logger.info("Query validation completed successfully")
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in check query node: {e}")
            error_message = AIMessage(f"Error validating query: {str(e)}")
            return {"messages": [error_message]}


def should_continue(state: MessagesState) -> Literal["check_query", "__end__"]:
    """确定是否继续进行查询验证或结束。

    参数:
        state: 当前对话状态

    返回:
        下一个节点名称或 END
    """
    try:
        messages = state["messages"]
        last_message = messages[-1]

        # 检查最后一条消息是否有工具调用
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.debug("Tool calls found, continuing to check_query")
            return "check_query"
        else:
            logger.debug("No tool calls found, ending conversation")
            return END

    except Exception as e:
        logger.error(f"Error in should_continue: {e}")
        return END

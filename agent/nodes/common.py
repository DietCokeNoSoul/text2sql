"""公共节点库 - 可被所有 Skill 复用的节点工厂。

此模块提供可复用的节点创建函数，避免在多个 Skill 中重复代码。
"""

import logging
from typing import Dict, List, Any, Callable

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, AnyMessage
from langgraph.graph import MessagesState

from ..tools import SQLToolManager

logger = logging.getLogger(__name__)


class CommonNodes:
    """公共节点库，提供可复用的节点工厂函数。
    
    所有 Skill 都可以通过这个类创建常用的节点，
    保证一致性并避免代码重复。
    """
    
    def __init__(self, tool_manager: SQLToolManager, llm: BaseChatModel):
        """初始化公共节点库。
        
        参数:
            tool_manager: SQL 工具管理器
            llm: 语言模型实例
        """
        self.tool_manager = tool_manager
        self.llm = llm
    
    def create_list_tables_node(self) -> Callable:
        """创建"列出表"节点。
        
        返回:
            节点函数，列出数据库中所有表名
        """
        def list_tables(state: Dict[str, Any]) -> Dict[str, Any]:
            """执行列表表操作。"""
            try:
                logger.info("Executing list tables node (common)")
                
                # 创建工具调用
                tool_call = {
                    "name": "sql_db_list_tables",
                    "args": {},
                    "id": "list_tables_call",
                    "type": "tool_call",
                }
                tool_call_message = AIMessage(content="", tool_calls=[tool_call])
                
                # 执行工具
                list_tables_tool = self.tool_manager.get_list_tables_tool()
                tool_message = list_tables_tool.invoke(tool_call)
                
                # 创建响应
                response = AIMessage(f"Available tables: {tool_message.content}")
                
                logger.info(f"Listed tables: {tool_message.content}")
                
                # 返回消息更新
                messages = state.get("messages", [])
                messages.extend([tool_call_message, tool_message, response])
                
                return {"messages": messages}
                
            except Exception as e:
                logger.error(f"Error in list tables node: {e}")
                error_message = AIMessage(f"Error listing tables: {str(e)}")
                messages = state.get("messages", [])
                messages.append(error_message)
                return {"messages": messages}
        
        return list_tables
    
    def create_get_schema_node(self) -> Callable:
        """创建"获取结构"节点。
        
        返回:
            节点函数，获取相关表的结构信息
        """
        def get_schema(state: Dict[str, Any]) -> Dict[str, Any]:
            """执行架构检索操作。"""
            try:
                logger.info("Executing get schema node (common)")
                
                schema_tool = self.tool_manager.get_schema_tool()
                llm_with_tools = self.llm.bind_tools([schema_tool], tool_choice="auto")
                
                messages = state.get("messages", [])
                response = llm_with_tools.invoke(messages)
                
                logger.info(f"Schema retrieval completed")
                
                messages.append(response)
                return {"messages": messages}
                
            except Exception as e:
                logger.error(f"Error in get schema node: {e}")
                error_message = AIMessage(f"Error retrieving schema: {str(e)}")
                messages = state.get("messages", [])
                messages.append(error_message)
                return {"messages": messages}
        
        return get_schema
    
    def create_execute_query_node(self) -> Callable:
        """创建"执行查询"节点（单个查询）。
        
        返回:
            节点函数，执行单个 SQL 查询
        """
        def execute_query(state: Dict[str, Any]) -> Dict[str, Any]:
            """执行单个 SQL 查询。"""
            try:
                logger.info("Executing query node (common)")
                
                # 从最后一条消息中获取 tool_calls
                messages = state.get("messages", [])
                if not messages:
                    logger.warning("No messages in state")
                    return {"messages": messages}
                
                last_message = messages[-1]
                if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                    logger.warning("No tool calls found in last message")
                    return {"messages": messages}
                
                # 执行查询工具
                query_tool_node = self.tool_manager.get_query_node()
                result = query_tool_node.invoke(state)
                
                logger.info("Query executed successfully")
                return result
                
            except Exception as e:
                logger.error(f"Error in execute query node: {e}")
                error_message = AIMessage(f"Error executing query: {str(e)}")
                messages = state.get("messages", [])
                messages.append(error_message)
                return {"messages": messages}
        
        return execute_query
    
    def create_batch_execute_node(self) -> Callable:
        """创建"批量执行查询"节点（多个查询）。
        
        返回:
            节点函数，批量执行多个 SQL 查询
        """
        def batch_execute_queries(state: Dict[str, Any]) -> Dict[str, Any]:
            """批量执行多个 SQL 查询。
            
            从 state["sql_queries"] 读取查询列表，
            返回 state["query_results"] 结果列表。
            """
            try:
                logger.info("Executing batch queries node (common)")
                
                query_tool = self.tool_manager.get_query_tool()
                results = []
                
                sql_queries = state.get("sql_queries", [])
                logger.info(f"Executing {len(sql_queries)} queries")
                
                for i, query_info in enumerate(sql_queries):
                    try:
                        sql = query_info.get("sql", "")
                        if not sql:
                            continue
                        
                        # 执行查询
                        result = query_tool.invoke({"query": sql})
                        
                        results.append({
                            "purpose": query_info.get("purpose", f"Query {i+1}"),
                            "sql": sql,
                            "data": result,
                            "status": "success"
                        })
                        logger.info(f"Query {i+1}/{len(sql_queries)} executed successfully")
                        
                    except Exception as e:
                        logger.error(f"Query {i+1} failed: {e}")
                        results.append({
                            "purpose": query_info.get("purpose", f"Query {i+1}"),
                            "sql": query_info.get("sql", ""),
                            "error": str(e),
                            "status": "failed"
                        })
                
                logger.info(f"Batch execution completed: {len(results)} queries")
                
                # 更新消息
                messages = state.get("messages", [])
                summary = f"✅ 执行了 {len(results)} 个查询，成功 {sum(1 for r in results if r['status'] == 'success')} 个"
                messages.append(AIMessage(content=summary))
                
                return {
                    "query_results": results,
                    "messages": messages
                }
                
            except Exception as e:
                logger.error(f"Error in batch execute node: {e}")
                error_message = AIMessage(f"Error in batch execution: {str(e)}")
                messages = state.get("messages", [])
                messages.append(error_message)
                return {"messages": messages, "query_results": []}
        
        return batch_execute_queries

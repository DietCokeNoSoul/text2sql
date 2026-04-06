"""SQL Agent 的工具管理。

本模块提供 SQL 数据库工具和代理使用的其他实用程序的管理。
"""

import logging
from typing import List, Optional

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langgraph.prebuilt import ToolNode

from .database import SQLDatabaseManager
from .types import ToolNotFoundError


logger = logging.getLogger(__name__)


class SQLToolManager:
    """管理 SQL 数据库工具和工具节点。"""
    
    def __init__(self, db_manager: SQLDatabaseManager, llm: BaseChatModel) -> None:
        """初始化工具管理器。
        
        参数:
            db_manager: 数据库管理器实例
            llm: 用于工具操作的语言模型
        """
        self.db_manager = db_manager
        self.llm = llm
        self._toolkit: Optional[SQLDatabaseToolkit] = None
        self._tools: Optional[List[BaseTool]] = None
        self._tool_nodes: dict[str, ToolNode] = {}
    
    @property
    def toolkit(self) -> SQLDatabaseToolkit:
        """获取 SQL 数据库工具集，如有必要则创建它。"""
        if self._toolkit is None:
            self._create_toolkit()
        return self._toolkit
    
    def _create_toolkit(self) -> None:
        """创建 SQL 数据库工具集。"""
        try:
            logger.info("Creating SQL database toolkit")
            self._toolkit = SQLDatabaseToolkit(
                db=self.db_manager.db,
                llm=self.llm
            )
            self._tools = self._toolkit.get_tools()
            logger.info(f"Created toolkit with {len(self._tools)} tools")
        except Exception as e:
            logger.error(f"Failed to create SQL toolkit: {e}")
            raise ToolNotFoundError(f"Failed to create SQL toolkit: {e}") from e
    
    def get_all_tools(self) -> List[BaseTool]:
        """获取所有可用的工具。"""
        if self._tools is None:
            self._create_toolkit()
        return self._tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """根据名称获取特定的工具。
        
        参数:
            name: 要检索的工具名称
            
        返回:
            如果找到则返回工具，否则返回 None
        """
        tools = self.get_all_tools()
        for tool in tools:
            if tool.name == name:
                logger.debug(f"Found tool: {name}")
                return tool
        
        logger.warning(f"Tool not found: {name}")
        return None
    
    def get_required_tool(self, name: str) -> BaseTool:
        """根据名称获取必需的工具，如果未找到则引发错误。
        
        参数:
            name: 要检索的工具名称
            
        返回:
            工具
            
        异常:
            ToolNotFoundError: 如果未找到工具
        """
        tool = self.get_tool_by_name(name)
        if tool is None:
            available_tools = [t.name for t in self.get_all_tools()]
            raise ToolNotFoundError(
                f"Required tool '{name}' not found. Available tools: {available_tools}"
            )
        return tool
    
    def get_schema_tool(self) -> BaseTool:
        """获取模式检索工具。"""
        return self.get_required_tool("sql_db_schema")
    
    def get_query_tool(self) -> BaseTool:
        """获取查询执行工具。"""
        return self.get_required_tool("sql_db_query")
    
    def get_list_tables_tool(self) -> BaseTool:
        """获取表列表工具。"""
        return self.get_required_tool("sql_db_list_tables")
    
    def get_tool_node(self, tool_name: str, node_name: Optional[str] = None) -> ToolNode:
        """获取特定工具的工具节点。
        
        参数:
            tool_name: 工具名称
            node_name: 节点的可选名称（默认为 tool_name）
            
        返回:
            ToolNode 实例
        """
        if node_name is None:
            node_name = tool_name
        
        if node_name not in self._tool_nodes:
            tool = self.get_required_tool(tool_name)
            self._tool_nodes[node_name] = ToolNode([tool], name=node_name)
            logger.debug(f"Created tool node: {node_name}")
        
        return self._tool_nodes[node_name]
    
    def get_schema_node(self) -> ToolNode:
        """获取模式工具节点。"""
        return self.get_tool_node("sql_db_schema", "get_schema")
    
    def get_query_node(self) -> ToolNode:
        """获取查询工具节点。"""
        return self.get_tool_node("sql_db_query", "run_query")
    
    def list_available_tools(self) -> List[str]:
        """获取可用工具名称列表。
        
        返回:
            工具名称列表
        """
        tools = self.get_all_tools()
        tool_names = [tool.name for tool in tools]
        logger.debug(f"Available tools: {tool_names}")
        return tool_names
    
    def validate_tools(self) -> bool:
        """验证所有必需的工具是否可用。
        
        返回:
            如果所有必需的工具都可用则返回 True，否则返回 False
        """
        required_tools = ["sql_db_schema", "sql_db_query", "sql_db_list_tables"]
        available_tools = self.list_available_tools()
        
        missing_tools = [tool for tool in required_tools if tool not in available_tools]
        
        if missing_tools:
            logger.error(f"Missing required tools: {missing_tools}")
            return False
        
        logger.info("All required tools are available")
        return True

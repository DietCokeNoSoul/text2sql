"""SQL Agent 的工具管理。

本模块提供 SQL 数据库工具和代理使用的其他实用程序的管理。
支持 Schema 缓存包装器以减少重复数据库调用。
"""

import logging
from typing import List, Optional

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from .database import SQLDatabaseManager
from .types import ToolNotFoundError


logger = logging.getLogger(__name__)


class CachedSchemaTool:
    """Schema 工具的缓存包装器。
    
    拦截 sql_db_schema 工具调用，优先从 SQLDatabaseManager 的缓存中获取，
    避免每次都访问数据库。
    """
    
    def __init__(self, original_tool: BaseTool, db_manager: SQLDatabaseManager):
        self._original = original_tool
        self._db_manager = db_manager
        self.name = original_tool.name
        self.description = original_tool.description
        self.args_schema = original_tool.args_schema
    
    def invoke(self, input_data, config=None, **kwargs):
        """通过缓存层调用 schema 工具。"""
        # 提取表名参数
        if isinstance(input_data, dict):
            table_names_str = input_data.get("table_names", "")
        else:
            table_names_str = str(input_data)
        
        if table_names_str and self._db_manager.schema_cache:
            table_list = [t.strip() for t in table_names_str.split(",")]
            cached = self._db_manager.schema_cache.get_schema(table_list)
            if cached is not None:
                logger.debug(f"[CachedSchemaTool] Cache HIT for: {table_names_str}")
                return cached
        
        # 缓存未命中，调用原始工具
        result = self._original.invoke(input_data, config=config, **kwargs)
        
        # 写入缓存
        if table_names_str and self._db_manager.schema_cache:
            table_list = [t.strip() for t in table_names_str.split(",")]
            self._db_manager.schema_cache.set_schema(table_list, result)
            logger.debug(f"[CachedSchemaTool] Cached schema for: {table_names_str}")
        
        return result
    
    def __getattr__(self, name):
        """将其他属性代理到原始工具。"""
        return getattr(self._original, name)


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
            # SQLDatabaseToolkit (Pydantic v2) requires a BaseChatModel, not a
            # RunnableRetry wrapper. Unwrap if necessary.
            llm_for_toolkit = self.llm
            if not isinstance(llm_for_toolkit, BaseChatModel) and hasattr(llm_for_toolkit, "bound"):
                llm_for_toolkit = llm_for_toolkit.bound
            self._toolkit = SQLDatabaseToolkit(
                db=self.db_manager.db,
                llm=llm_for_toolkit
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
        """获取模式检索工具（带缓存包装）。"""
        original = self.get_required_tool("sql_db_schema")
        if self.db_manager.schema_cache:
            return CachedSchemaTool(original, self.db_manager)
        return original
    
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

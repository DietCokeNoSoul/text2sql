"""SQL Agent 的图构建器。

本模块提供主要的图构建器类，用于构建
SQL Agent 工作流的 LangGraph 状态图。
"""

import logging
from typing import Any, Optional

from langchain.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph

from .config import AgentConfig
from .database import SQLDatabaseManager
from .old_nodes import CheckQueryNode, GenerateQueryNode, GetSchemaNode, ListTablesNode, should_continue
from .tools import SQLToolManager

logger = logging.getLogger(__name__)


class SQLAgentGraphBuilder:
    """构建 SQL Agent 状态图。"""
    
    def __init__(
        self,
        config: AgentConfig,
        llm: BaseChatModel,
        db_manager: SQLDatabaseManager,
        tool_manager: SQLToolManager
    ) -> None:
        """初始化图构建器。
        
        参数：
            config: Agent 配置
            llm: 语言模型实例
            db_manager: 数据库管理器实例
            tool_manager: 工具管理器实例
        """
        self.config = config
        self.llm = llm
        self.db_manager = db_manager
        self.tool_manager = tool_manager
        self.builder = StateGraph(MessagesState)
        
        # 初始化节点
        self.list_tables_node = ListTablesNode(tool_manager)
        self.get_schema_node = GetSchemaNode(tool_manager, llm)
        self.generate_query_node = GenerateQueryNode(tool_manager, llm, db_manager)
        self.check_query_node = CheckQueryNode(tool_manager, llm, db_manager)
        
        # 获取工具节点
        self.schema_tool_node = tool_manager.get_schema_node()
        self.query_tool_node = tool_manager.get_query_node()
    
    def add_nodes(self) -> None:
        """添加所有节点到图中。"""
        logger.info("Adding nodes to graph")
        
        # 添加自定义节点
        self.builder.add_node(self.list_tables_node.name, self.list_tables_node)
        self.builder.add_node(self.get_schema_node.name, self.get_schema_node)
        self.builder.add_node(self.generate_query_node.name, self.generate_query_node)
        self.builder.add_node(self.check_query_node.name, self.check_query_node)
        
        # 添加工具节点（LangGraph v1: name 在前，node 在后）
        self.builder.add_node("get_schema", self.schema_tool_node)
        self.builder.add_node("run_query", self.query_tool_node)
        
        logger.info("All nodes added successfully")
    
    def add_edges(self) -> None:
        """添加所有边到图中。"""
        logger.info("Adding edges to graph")
        
        # 定义工作流
        self.builder.add_edge(START, "list_tables")
        self.builder.add_edge("list_tables", "get_relative_schema")
        self.builder.add_edge("get_relative_schema", "get_schema")
        self.builder.add_edge("get_schema", "generate_query")
        
        # 基于查询是否需要验证的条件边
        self.builder.add_conditional_edges(
            "generate_query",
            should_continue,
        )
        
        self.builder.add_edge("check_query", "run_query")
        self.builder.add_edge("run_query", "generate_query")
        
        logger.info("All edges added successfully")
    
    def build_graph(self, checkpointer: Optional[BaseCheckpointSaver] = None) -> Any:
        """构建并返回编译后的图。
        
        参数：
            checkpointer: 可选的检查点保存器，用于持久化对话状态
        
        返回值：
            编译后的 LangGraph 状态图
        """
        logger.info("Building SQL Agent graph")
        
        try:
            # 在构建之前验证工具
            if not self.tool_manager.validate_tools():
                raise RuntimeError("Required tools are not available")
            
            # 添加节点和边
            self.add_nodes()
            self.add_edges()
            
            # 编译图（带或不带 checkpointer）
            if checkpointer:
                graph = self.builder.compile(checkpointer=checkpointer)
                logger.info("Graph compiled with checkpointer (memory enabled)")
            else:
                graph = self.builder.compile()
            
            logger.info("Graph built successfully")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            raise RuntimeError(f"Failed to build graph: {e}") from e
    

'''
    SQL Agent 图结构：
    ===================

    START（开始）
    ↓
    list_tables（列出可用的数据库表）
    ↓
    get_relative_schema（获取相关表的结构）
    ↓
    get_schema（工具：检索表结构）
    ↓
    generate_query（根据用户问题生成 SQL 查询）
    ↓
    [条件判断：是否有工具调用？]
    ├─ 是 → check_query（验证并优化查询）
    │         ↓
    │       run_query（工具：执行 SQL 查询）
    │         ↓
    │       generate_query（继续对话）
    │
    └─ 否 → END（结束，无查询需执行）

    说明：
    - 该图处理用户关于数据库的问题
    - 自动列出表并检索结构信息
    - 生成合适的 SQL 查询
    - 执行前验证查询
    - 可处理同一对话中的后续问题
'''


def create_sql_agent_graph(
    config: AgentConfig,
    llm: BaseChatModel,
    checkpointer: Optional[BaseCheckpointSaver] = None
) -> Any:
    """使用给定的配置创建 SQL Agent 图。
    
    参数：
        config: Agent 配置
        llm: 语言模型实例
        checkpointer: 可选的检查点保存器，用于启用对话记忆
        
    返回值：
        编译后的 LangGraph 状态图
    """
    logger.info("Creating SQL Agent graph")
    
    # 创建管理器
    db_manager = SQLDatabaseManager(config.database)
    tool_manager = SQLToolManager(db_manager, llm)
    
    # 创建并构建图
    builder = SQLAgentGraphBuilder(config, llm, db_manager, tool_manager)
    graph = builder.build_graph(checkpointer=checkpointer)
    
    logger.info("SQL Agent graph created successfully")
    return graph

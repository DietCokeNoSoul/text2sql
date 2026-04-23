"""SQL Agent 的类型定义和协议。

本模块定义了整个 SQL Agent 应用中使用的核心类型、协议和接口。
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState


class DatabaseDialect(Enum):
    """支持的数据库方言。"""
    
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MSSQL = "mssql"
    ORACLE = "oracle"


class BaseNode(ABC):
    """图节点的抽象基类。"""
    
    def __init__(self, name: str) -> None:
        """初始化节点。
        
        参数:
            name: 节点名称
        """
        self.name = name
    
    @abstractmethod
    def execute(self, state: MessagesState) -> Dict[str, List[AnyMessage]]:
        """执行节点逻辑。
        
        参数:
            state: 当前对话状态
            
        返回:
            包含新消息的更新状态
        """
        ...
    
    def __call__(self, state: MessagesState) -> Dict[str, List[AnyMessage]]:
        """使节点可调用。"""
        return self.execute(state)
    

# ==============================================================================
# 自定义异常类
# ==============================================================================

class SQLAgentError(Exception):
    """SQL Agent 错误的基础异常类。"""
    pass


class DatabaseConnectionError(SQLAgentError):
    """数据库连接失败时抛出。"""
    pass


class QueryExecutionError(SQLAgentError):
    """查询执行失败时抛出。"""
    pass


class ToolNotFoundError(SQLAgentError):
    """找不到所需工具时抛出。"""
    pass


class SecurityViolationError(SQLAgentError):
    """SQL 安全护栏拦截时抛出。
    
    属性:
        layer: 触发拦截的防御层（如 "Layer1_StatementType"）
        reason: 拦截原因描述
        sql: 被拒绝的 SQL 语句
    """
    
    def __init__(self, reason: str, layer: str = "", sql: str = "") -> None:
        super().__init__(reason)
        self.layer = layer
        self.reason = reason
        self.sql = sql
    
    def __str__(self) -> str:
        return f"[{self.layer}] {self.reason}"
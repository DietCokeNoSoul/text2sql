"""SQL Agent 的数据库管理模块。

本模块提供数据库连接管理和 SQL 操作，
具备完善的错误处理和验证功能。
"""

import logging
from typing import List, Optional

from langchain_community.utilities import SQLDatabase

from .config import DatabaseConfig
from .types import DatabaseConnectionError, DatabaseDialect, QueryExecutionError


logger = logging.getLogger(__name__)


class SQLDatabaseManager:
    """管理 SQL 数据库连接和操作。"""
    
    def __init__(self, config: DatabaseConfig) -> None:
        """初始化数据库管理器。
        
        参数:
            config: 数据库配置
        """
        self.config = config
        self._db: Optional[SQLDatabase] = None
        self._dialect: Optional[DatabaseDialect] = None
    
    @property
    def db(self) -> SQLDatabase:
        """获取数据库连接，如有必要则创建连接。"""
        if self._db is None:
            self._connect()
        return self._db
    
    def _connect(self) -> None:
        """建立数据库连接。"""
        try:
            logger.info(f"Connecting to database: {self.config.uri}")
            self._db = SQLDatabase.from_uri(
                self.config.uri,
                max_string_length=10000,
                include_tables=None,
                sample_rows_in_table_info=3
            )
            self._dialect = self._detect_dialect()
            logger.info(f"Successfully connected to {self._dialect.value} database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseConnectionError(f"Failed to connect to database: {e}") from e
    
    def _detect_dialect(self) -> DatabaseDialect:
        """从连接中检测数据库方言。"""
        if self._db is None:
            raise DatabaseConnectionError("Database not connected")
        
        dialect_str = self._db.dialect.lower()
        
        if "sqlite" in dialect_str:
            return DatabaseDialect.SQLITE
        elif "postgresql" in dialect_str or "postgres" in dialect_str:
            return DatabaseDialect.POSTGRESQL
        elif "mysql" in dialect_str:
            return DatabaseDialect.MYSQL
        elif "mssql" in dialect_str or "sqlserver" in dialect_str:
            return DatabaseDialect.MSSQL
        elif "oracle" in dialect_str:
            return DatabaseDialect.ORACLE
        else:
            logger.warning(f"Unknown dialect: {dialect_str}, defaulting to SQLite")
            return DatabaseDialect.SQLITE
    
    def get_connection_string(self) -> str:
        """获取数据库连接字符串。"""
        return self.config.uri
    
    def get_dialect(self) -> DatabaseDialect:
        """获取数据库方言。"""
        if self._dialect is None:
            self._connect()
        return self._dialect
    
    def get_table_names(self) -> List[str]:
        """获取可用的表名列表。"""
        try:
            tables = self.db.get_usable_table_names()
            logger.debug(f"Found {len(tables)} tables: {tables}")
            return tables
        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            raise QueryExecutionError(f"Failed to get table names: {e}") from e
    
    def execute_query(self, query: str) -> str:
        """执行 SQL 查询并返回结果。
        
        参数:
            query: 要执行的 SQL 查询
            
        返回:
            查询结果字符串
        """
        try:
            logger.debug(f"Executing query: {query}")
            result = self.db.run(query)
            logger.debug(f"Query executed successfully, result length: {len(str(result))}")
            return result
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise QueryExecutionError(f"Failed to execute query: {e}") from e
    
    def get_table_schema(self, table_names: Optional[List[str]] = None) -> str:
        """获取指定表的模式信息。
        
        参数:
            table_names: 要获取模式的表名列表。
                        如果为 None，则获取所有表的模式。
        
        返回:
            模式信息字符串
        """
        try:
            if table_names is None:
                table_names = self.get_table_names()
            
            logger.debug(f"Getting schema for tables: {table_names}")
            schema = self.db.get_table_info(table_names=table_names)
            logger.debug(f"Schema retrieved successfully, length: {len(schema)}")
            return schema
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            raise QueryExecutionError(f"Failed to get table schema: {e}") from e
        
    
    def close(self) -> None:
        """关闭数据库连接。"""
        if self._db is not None:
            try:
                # SQLDatabase 没有明确的 close 方法
                # 可以重置引用
                self._db = None
                self._dialect = None
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
    
    def __enter__(self) -> "SQLDatabaseManager":
        """上下文管理器入口。"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口。"""
        self.close()

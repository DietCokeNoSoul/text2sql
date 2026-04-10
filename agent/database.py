"""SQL Agent 的数据库管理模块。

本模块提供数据库连接管理和 SQL 操作，
具备完善的错误处理、验证功能和 Schema 缓存。
"""

import difflib
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from langchain_community.utilities import SQLDatabase

from .config import DatabaseConfig
from .types import DatabaseConnectionError, DatabaseDialect, QueryExecutionError


logger = logging.getLogger(__name__)


class SchemaCache:
    """Schema 缓存，通过 TTL 机制减少重复的数据库 schema 查询。
    
    缓存策略:
    - 表名列表: 单独缓存
    - 表结构信息: 按表名集合缓存（同一组表只查一次）
    - 所有缓存条目带 TTL 自动过期
    """
    
    def __init__(self, ttl_seconds: int = 300):
        """初始化 Schema 缓存。
        
        参数:
            ttl_seconds: 缓存过期时间（秒），默认 300 秒（5 分钟）
        """
        self.ttl_seconds = ttl_seconds
        self._table_names_cache: Optional[Tuple[List[str], float]] = None
        self._schema_cache: Dict[str, Tuple[str, float]] = {}
        self._stats = {"hits": 0, "misses": 0}
    
    def _is_expired(self, timestamp: float) -> bool:
        """检查缓存条目是否已过期。"""
        return (time.time() - timestamp) > self.ttl_seconds
    
    def get_table_names(self) -> Optional[List[str]]:
        """获取缓存的表名列表。"""
        if self._table_names_cache is None:
            self._stats["misses"] += 1
            return None
        names, ts = self._table_names_cache
        if self._is_expired(ts):
            self._table_names_cache = None
            self._stats["misses"] += 1
            return None
        self._stats["hits"] += 1
        return names
    
    def set_table_names(self, names: List[str]) -> None:
        """缓存表名列表。"""
        self._table_names_cache = (names, time.time())
    
    def get_schema(self, table_names: List[str]) -> Optional[str]:
        """获取缓存的 schema 信息。
        
        参数:
            table_names: 表名列表
            
        返回:
            缓存的 schema 字符串，未命中时返回 None
        """
        key = self._make_key(table_names)
        if key not in self._schema_cache:
            self._stats["misses"] += 1
            return None
        schema, ts = self._schema_cache[key]
        if self._is_expired(ts):
            del self._schema_cache[key]
            self._stats["misses"] += 1
            return None
        self._stats["hits"] += 1
        return schema
    
    def set_schema(self, table_names: List[str], schema: str) -> None:
        """缓存 schema 信息。"""
        key = self._make_key(table_names)
        self._schema_cache[key] = (schema, time.time())
    
    def _make_key(self, table_names: List[str]) -> str:
        """生成缓存 key（排序后的表名，保证顺序无关）。"""
        return ",".join(sorted(t.strip() for t in table_names))
    
    def clear(self) -> None:
        """清空所有缓存。"""
        self._table_names_cache = None
        self._schema_cache.clear()
        logger.info("[SchemaCache] Cache cleared")
    
    @property
    def stats(self) -> Dict[str, int]:
        """获取缓存统计信息。"""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "total": total,
            "hit_rate": f"{self._stats['hits']/total*100:.1f}%" if total > 0 else "N/A",
            "schema_entries": len(self._schema_cache),
        }
    
    def __repr__(self) -> str:
        stats = self.stats
        return f"SchemaCache(ttl={self.ttl_seconds}s, entries={stats['schema_entries']}, hit_rate={stats['hit_rate']})"


class SQLDatabaseManager:
    """管理 SQL 数据库连接和操作，内置 Schema 缓存。"""
    
    def __init__(self, config: DatabaseConfig, cache_ttl: int = 300) -> None:
        """初始化数据库管理器。
        
        参数:
            config: 数据库配置
            cache_ttl: Schema 缓存 TTL（秒），默认 300 秒。设为 0 禁用缓存。
        """
        self.config = config
        self._db: Optional[SQLDatabase] = None
        self._dialect: Optional[DatabaseDialect] = None
        self.schema_cache = SchemaCache(ttl_seconds=cache_ttl) if cache_ttl > 0 else None
    
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
        """获取可用的表名列表（带缓存）。"""
        # 尝试从缓存获取
        if self.schema_cache:
            cached = self.schema_cache.get_table_names()
            if cached is not None:
                logger.debug(f"[SchemaCache] Table names cache HIT ({len(cached)} tables)")
                return cached
        
        try:
            tables = self.db.get_usable_table_names()
            logger.debug(f"Found {len(tables)} tables: {tables}")
            
            # 写入缓存
            if self.schema_cache:
                self.schema_cache.set_table_names(tables)
                logger.debug("[SchemaCache] Table names cached")
            
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
        """获取指定表的模式信息（带缓存）。
        
        参数:
            table_names: 要获取模式的表名列表。
                        如果为 None，则获取所有表的模式。
        
        返回:
            模式信息字符串
        """
        try:
            if table_names is None:
                table_names = self.get_table_names()
            
            # 尝试从缓存获取
            if self.schema_cache:
                cached = self.schema_cache.get_schema(table_names)
                if cached is not None:
                    logger.debug(f"[SchemaCache] Schema cache HIT for {table_names}")
                    return cached
            
            logger.debug(f"Getting schema for tables: {table_names}")
            schema = self.db.get_table_info(table_names=table_names)
            logger.debug(f"Schema retrieved successfully, length: {len(schema)}")
            
            # 写入缓存
            if self.schema_cache:
                self.schema_cache.set_schema(table_names, schema)
                logger.debug(f"[SchemaCache] Schema cached for {table_names}")
            
            return schema
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            raise QueryExecutionError(f"Failed to get table schema: {e}") from e
        
    
    def get_column_map(self) -> Dict[str, List[str]]:
        """获取所有表的列名映射 {表名: [列名, ...]}。
        
        使用 SQLAlchemy inspect 直接从数据库获取列名，
        结果从 schema 缓存中复用（不额外存储，因为列名本身从已缓存的 schema 解析）。
        
        返回:
            字典 {表名: [列名列表]}
        """
        try:
            from sqlalchemy import inspect as sa_inspect
            inspector = sa_inspect(self.db._engine)
            column_map: Dict[str, List[str]] = {}
            for table_name in self.get_table_names():
                try:
                    cols = [col["name"] for col in inspector.get_columns(table_name)]
                    column_map[table_name] = cols
                except Exception:
                    column_map[table_name] = []
            return column_map
        except Exception as e:
            logger.warning(f"[ColumnMap] Failed to get column map: {e}")
            return {}

    def find_similar_columns(self, bad_column: str, cutoff: float = 0.55) -> List[str]:
        """在所有表的列中找到与 bad_column 最相似的列名。
        
        参数:
            bad_column: 出错的列名
            cutoff: 相似度阈值（0~1），越高越严格，默认 0.55
            
        返回:
            相似列名列表，格式为 "table.column"，按相似度排序（最多8个）
        """
        column_map = self.get_column_map()
        # 构建 "table.column" 全限定列名列表
        all_qualified = [
            f"{tbl}.{col}"
            for tbl, cols in column_map.items()
            for col in cols
        ]
        # 搜索裸列名（去重后匹配，保持高效）
        all_bare = list(dict.fromkeys(col for cols in column_map.values() for col in cols))
        
        bare_matches = difflib.get_close_matches(bad_column, all_bare, n=5, cutoff=cutoff)
        
        # 把裸列名匹配结果展开为带表名的格式
        result: List[str] = []
        seen: set = set()
        for match in bare_matches:
            for tbl, cols in column_map.items():
                if match in cols:
                    qualified = f"{tbl}.{match}"
                    if qualified not in seen:
                        result.append(qualified)
                        seen.add(qualified)
        
        # 如果裸列名没有命中，再尝试全限定名匹配
        if not result:
            result = difflib.get_close_matches(bad_column, all_qualified, n=5, cutoff=cutoff)
        
        return result[:8]

    def close(self) -> None:
        """关闭数据库连接并清空缓存。"""
        if self._db is not None:
            try:
                self._db = None
                self._dialect = None
                if self.schema_cache:
                    self.schema_cache.clear()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
    
    def __enter__(self) -> "SQLDatabaseManager":
        """上下文管理器入口。"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口。"""
        self.close()

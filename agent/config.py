"""SQL Agent 的配置管理。

此模块提供集中式配置管理，支持环境变量、配置文件和默认值。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    """数据库配置设置。"""
    
    uri: str = "sqlite:///Chinook.db"  # 默认使用 SQLite（无需密码）
    max_query_results: int = 5
    timeout_seconds: int = 30
    
    def __post_init__(self) -> None:
        """验证数据库配置。"""
        if not self.uri:
            raise ValueError("Database URI cannot be empty")
        if self.max_query_results <= 0:
            raise ValueError("max_query_results must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class LLMConfig:
    """语言模型配置设置。"""
    
    provider: str = "tongyi"
    model: str = "qwen-plus"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    def __post_init__(self) -> None:
        """验证 LLM 配置。"""
        if not self.provider:
            raise ValueError("LLM provider cannot be empty")
        if not self.model:
            raise ValueError("LLM model cannot be empty")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass
class LoggingConfig:
    """日志配置设置。"""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class OutputConfig:
    """输出路径配置。"""
    
    report_dir: str = "report"    # 分析报告保存目录（相对项目根目录或绝对路径）
    chart_dir: str = "report/charts"  # 图表文件保存目录


@dataclass
class SecurityConfig:
    """SQL 安全护栏配置。
    
    四层防御：
    - Layer 1: 语句类型控制（只允许 SELECT）
    - Layer 2: 表访问控制（allowlist / denylist）
    - Layer 3: 查询复杂度限制（行数 / 长度）
    - Layer 4: 结果脱敏（敏感列名模糊匹配）
    """
    
    # Layer 1: 语句类型控制
    allowed_statements: List[str] = field(
        default_factory=lambda: ["SELECT"]
    )
    blocked_keywords: List[str] = field(
        default_factory=lambda: [
            "xp_cmdshell", "INTO OUTFILE", "LOAD DATA",
            "EXEC(", "EXECUTE(", "sp_executesql",
            "OPENROWSET", "BULK INSERT",
        ]
    )

    # Layer 2: 表访问控制
    table_allowlist: Optional[List[str]] = None   # None = 允许所有表
    table_denylist: List[str] = field(default_factory=list)

    # Layer 3: 复杂度限制
    max_rows: int = 1000          # 强制 LIMIT 上限
    max_query_length: int = 5000  # SQL 字符串长度上限

    # Layer 4: 结果脱敏
    sensitive_column_patterns: List[str] = field(
        default_factory=lambda: [
            r"password", r"passwd", r"secret", r"token",
            r"ssn", r"credit_card", r"phone", r"mobile",
            r"id_card", r"api_key", r"private_key",
        ]
    )
    mask_value: str = "***"

    # 审计日志
    enable_audit_log: bool = True
    audit_log_file: Optional[str] = None  # None = 只写 logger，不写文件

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """从环境变量读取安全配置。"""
        denylist_raw = os.getenv("SECURITY_TABLE_DENYLIST", "")
        denylist = [t.strip() for t in denylist_raw.split(",") if t.strip()]

        allowlist_raw = os.getenv("SECURITY_TABLE_ALLOWLIST", "")
        allowlist = [t.strip() for t in allowlist_raw.split(",") if t.strip()] or None

        return cls(
            max_rows=int(os.getenv("SECURITY_MAX_ROWS", "1000")),
            max_query_length=int(os.getenv("SECURITY_MAX_QUERY_LENGTH", "5000")),
            table_denylist=denylist,
            table_allowlist=allowlist,
            enable_audit_log=os.getenv("SECURITY_AUDIT_LOG", "true").lower() == "true",
            audit_log_file=os.getenv("SECURITY_AUDIT_LOG_FILE") or None,
        )


@dataclass
class AgentConfig:
    """SQL Agent 的主配置类。"""
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "AgentConfig":
        """从环境变量创建配置。
        
        参数:
            env_file: 要加载的 .env 文件的可选路径
            
        返回:
            包含环境变量值的 AgentConfig 实例
        """
        if env_file:
            load_dotenv(env_file)
        else:
            # 尝试从常见位置加载
            for env_path in [".env", "../.env", "../../.env"]:
                if Path(env_path).exists():
                    load_dotenv(env_path)
                    break
        
        # 数据库配置
        db_config = DatabaseConfig(
            uri=os.getenv("DB_URI") or DatabaseConfig.uri,
            max_query_results=int(os.getenv("DATABASE_MAX_QUERY_RESULTS", "5")),
            timeout_seconds=int(os.getenv("DATABASE_TIMEOUT_SECONDS", "30"))
        )
        
        # LLM 配置
        llm_config = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "tongyi"),
            model=os.getenv("LLM_MODEL", "qwen-plus"),
            api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS")) if os.getenv("LLM_MAX_TOKENS") else None
        )
        
        # 日志配置
        logging_config = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=os.getenv("LOG_FILE_PATH"),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
        
        # 输出路径配置
        output_config = OutputConfig(
            report_dir=os.getenv("REPORT_DIR", "report"),
            chart_dir=os.getenv("CHART_DIR", "report/charts"),
        )

        # 安全护栏配置
        security_config = SecurityConfig.from_env()
        
        return cls(
            database=db_config,
            llm=llm_config,
            logging=logging_config,
            output=output_config,
            security=security_config,
        )
    
    def validate(self) -> None:
        """验证整个配置。"""
        self.database.__post_init__()
        self.llm.__post_init__()
        
        if self.llm.api_key is None:
            raise ValueError("LLM API key is required")


def get_config() -> AgentConfig:
    """获取全局配置实例。
    
    返回:
        从环境加载的 AgentConfig 实例
    """
    config = AgentConfig.from_env()
    config.validate()
    return config

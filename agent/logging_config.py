"""SQL Agent 的日志配置。

本模块提供集中式日志配置，支持文件和控制台日志输出、
结构化格式化以及日志轮转功能。
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from .config import LoggingConfig


class ColoredFormatter(logging.Formatter):
    """控制台输出的彩色格式化器。"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 品红色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """使用颜色格式化日志记录。"""
        # 为日志级别名称添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        # 格式化消息
        formatted = super().format(record)
        
        # 为其他格式化器重置日志级别名称
        record.levelname = levelname
        
        return formatted


class StructuredFormatter(logging.Formatter):
    """文件输出的结构化格式化器。"""
    
    def format(self, record: logging.LogRecord) -> str:
        """使用结构化信息格式化日志记录。"""
        # 添加额外字段
        if not hasattr(record, 'component'):
            record.component = record.name.split('.')[-1] if '.' in record.name else record.name
        
        if not hasattr(record, 'function'):
            record.function = record.funcName
        
        if not hasattr(record, 'line'):
            record.line = record.lineno
        
        return super().format(record)


def setup_logging(config: LoggingConfig) -> None:
    """设置日志配置。
    
    参数：
        config: 日志配置
    """
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 带彩色输出的控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level.upper()))
    
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 带轮转的文件处理器（如果指定了文件路径）
    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, config.level.upper()))
        
        file_formatter = StructuredFormatter(
            fmt='%(asctime)s - %(component)s - %(levelname)s - %(function)s:%(line)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # 设置特定日志记录器的级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    # 记录配置信息
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {config.level}, File: {config.file_path}")


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器。
    
    参数：
        name: 日志记录器名称
        
    返回值：
        日志记录器实例
    """
    return logging.getLogger(name)
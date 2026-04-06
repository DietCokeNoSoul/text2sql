"""SQL Agent 图实现。

本模块提供 SQL Agent 的主图实现，
将所有组件集成到一个可工作的 LangGraph 应用中。
"""

import logging
import os
import uuid
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver

from .config import get_config
from .graph_builder import create_sql_agent_graph
from .logging_config import setup_logging


# 初始化配置
config = get_config()

# 设置日志
setup_logging(config.logging)
logger = logging.getLogger(__name__)

# 设置 API 密钥环境变量
if config.llm.api_key:
    # 通义千问使用 DASHSCOPE_API_KEY
    if config.llm.provider == "tongyi":
        os.environ["DASHSCOPE_API_KEY"] = config.llm.api_key
    else:
        os.environ[f"{config.llm.provider.upper()}_API_KEY"] = config.llm.api_key

# 初始化语言模型
try:
    if config.llm.provider == "tongyi":
        # 通义千问需要直接使用 ChatTongyi
        llm = ChatTongyi(
            model=config.llm.model,
            temperature=config.llm.temperature,
            dashscope_api_key=config.llm.api_key,
        )
        logger.info(f"Initialized Tongyi LLM: {config.llm.model}")
    else:
        # 其他支持的 provider 使用 init_chat_model
        llm = init_chat_model(f"{config.llm.provider}:{config.llm.model}")
        logger.info(f"Initialized LLM: {config.llm.provider}:{config.llm.model}")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise

# 创建内存检查点保存器（用于多轮对话记忆）
checkpointer = InMemorySaver()

# 创建图（带记忆功能）
try:
    graph = create_sql_agent_graph(config, llm, checkpointer=checkpointer)
    logger.info("SQL Agent graph created successfully with memory support")
except Exception as e:
    logger.error(f"Failed to create graph: {e}")
    raise


# 导出图供 LangGraph CLI 使用
__all__ = ["graph", "config", "run_query"]


def run_query(question: str, thread_id: str) -> None:
    """运行单个查询（带对话记忆）。
    
    参数:
        question: 用户的问题
        thread_id: 会话线程 ID，同一 ID 的对话共享记忆
    """
    logger.info(f"Running query (thread={thread_id}): {question}")

    # 配置 thread_id 以启用对话记忆
    config_with_thread = {"configurable": {"thread_id": thread_id}}

    try:
        for step in graph.stream(
            {"messages": [{"role": "user", "content": question}]},
            config_with_thread,
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
    except Exception as e:
        logger.error(f"Error running query: {e}")
        raise


def main() -> None:
    """交互式主循环，等待用户输入问题（支持多轮对话记忆）。"""
    # 为本次会话生成唯一的 thread_id
    thread_id = str(uuid.uuid4())
    
    print("=" * 50)
    print("SQL Agent 已启动 (支持多轮对话记忆)")
    print(f"数据库: {config.database.uri.split('@')[-1] if '@' in config.database.uri else config.database.uri}")
    print(f"模型: {config.llm.provider}:{config.llm.model}")
    print(f"会话ID: {thread_id[:8]}...")
    print("=" * 50)
    print("输入问题进行查询")
    print("输入 'new' 开始新会话，'quit' 或 'exit' 退出\n")

    while True:
        try:
            question = input("请输入您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ("quit", "exit", "q"):
                print("说不出分手！说好组一辈子的agent呢？")
                break
            
            if question.lower() == "new":
                thread_id = str(uuid.uuid4())
                print(f"\n已开始新会话，会话ID: {thread_id[:8]}...\n")
                continue
            
            print()
            run_query(question, thread_id)
            print()
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"发生错误: {e}\n")


if __name__ == "__main__":
    main()
"""SQL Agent 图实现。

本模块提供 SQL Agent 的主图实现，
将所有组件集成到一个可工作的 LangGraph 应用中。

流式输出说明:
  run_query()  — 同步流式输出（stream_mode="updates"），每个节点完成后打印。
  run_query_streaming() — 异步 token 级流式输出（astream_events），实时打印 LLM token。
"""

import asyncio
import logging
import os
import sys
import uuid
from typing import Any

# Windows UTF-8 console fix — allow emoji/CJK output without UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from langchain.chat_models import init_chat_model
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langgraph.checkpoint.memory import InMemorySaver

from .config import get_config
from .skill_graph_builder import create_skill_based_graph as create_sql_agent_graph
from .logging_config import setup_logging


# ── 节点名称→人类可读标签 ──────────────────────────────────────────────────
_NODE_LABELS: dict[str, str] = {
    "query_router":         "🔀 路由器",
    "list_tables":          "📋 列出表",
    "get_schema":           "🗂  获取结构",
    "dual_tower_retrieve":  "🔍 双塔检索",
    "plan":                 "🧠 生成计划",
    "execute_step":         "⚙️  执行步骤",
    "aggregate":            "📊 汇总结果",
    "judge":                "⚖️  判断完成",
    "understand_goal":      "🎯 理解目标",
    "explore_data":         "🔭 探索数据",
    "plan_analysis":        "📝 制定计划",
    "generate_queries":     "✍️  生成查询",
    "analyze_results":      "🔬 分析结果",
    "visualize":            "📈 可视化",
    "generate_report":      "📄 生成报告",
    "export_results":       "💾 导出结果",
    "generate_sql":         "✍️  生成 SQL",
    "execute_query":        "▶️  执行查询",
    "format_result":        "🖨  格式化输出",
    "error_correction":     "🔧 错误修复",
}


# 初始化配置
config = get_config()

# 设置日志
setup_logging(config.logging)
logger = logging.getLogger(__name__)

# 设置 API 密钥环境变量
if config.llm.api_key:
    if config.llm.provider == "tongyi":
        os.environ["DASHSCOPE_API_KEY"] = config.llm.api_key
    else:
        os.environ[f"{config.llm.provider.upper()}_API_KEY"] = config.llm.api_key

# 初始化语言模型
def _create_llm(cfg):
    """统一 LLM 初始化入口，按 provider 选择正确的后端。"""
    if cfg.provider == "tongyi":
        from langchain_community.chat_models import ChatTongyi
        kwargs = dict(
            model=cfg.model,
            temperature=cfg.temperature,
            dashscope_api_key=cfg.api_key,
        )
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        return ChatTongyi(**kwargs)
    extra = {"max_tokens": cfg.max_tokens} if cfg.max_tokens is not None else {}
    return init_chat_model(
        f"{cfg.provider}:{cfg.model}",
        temperature=cfg.temperature,
        **extra,
    )


try:
    _base_llm = _create_llm(config.llm)
    # B5: 自动重试 — 网络抖动时最多重试 3 次，指数退避 + jitter
    llm = _base_llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
    logger.info(f"Initialized LLM: {config.llm.provider}:{config.llm.model} (retry=3)")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise

# 初始化 LLM 响应缓存
if config.cache.enabled and config.cache.backend == "sqlite":
    try:
        set_llm_cache(SQLiteCache(database_path=config.cache.sqlite_path))
        logger.info(f"LLM SQLite cache enabled: {config.cache.sqlite_path}")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM cache: {e}")

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
__all__ = ["graph", "config", "run_query", "run_query_streaming"]


# ── 内部工具函数 ───────────────────────────────────────────────────────────

def _node_label(node_name: str) -> str:
    """返回节点的人类可读标签，未知节点保持原名。"""
    return _NODE_LABELS.get(node_name, f"▸ {node_name}")


def _print_node_update(node_name: str, update: dict) -> None:
    """将节点更新打印为格式化输出。"""
    label = _node_label(node_name)
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")

    messages = update.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content") and last_msg.content:
            content = last_msg.content
            # 截断超长输出
            if len(content) > 800:
                content = content[:800] + f"\n  … (共 {len(last_msg.content)} 字符)"
            print(content)
    elif update:
        # 打印非消息字段的摘要
        for k, v in update.items():
            if k == "messages":
                continue
            val_str = str(v)
            if len(val_str) > 120:
                val_str = val_str[:120] + "…"
            print(f"  {k}: {val_str}")


# ── 同步流式查询 ───────────────────────────────────────────────────────────

def run_query(question: str, thread_id: str) -> dict:
    """同步流式执行查询，每个节点完成后实时打印结果。

    使用 stream_mode="updates" 模式，每个 LangGraph 节点完成后
    立即输出该节点产生的状态变化，无需等待整个 graph 执行完毕。

    参数:
        question:  用户的自然语言问题
        thread_id: 会话线程 ID，同一 ID 的对话共享记忆

    返回:
        dict，包含:
            final_message  — 最后一个节点输出的文本内容
            nodes_visited  — 按顺序执行的节点名称列表
            export_files   — DataAnalysis 导出的文件路径列表（其他技能为空）
    """
    logger.info(f"Running query (thread={thread_id}): {question}")
    config_with_thread = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'═' * 55}")
    print(f"  🤖 SQL Agent 正在处理您的问题…")
    print(f"{'═' * 55}")

    nodes_visited: list[str] = []
    final_message: str = ""
    export_files: list[str] = []

    try:
        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": question}]},
            config_with_thread,
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                nodes_visited.append(node_name)
                _print_node_update(node_name, update)
                # 记录最后一个节点的消息作为最终回复
                msgs = update.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    if hasattr(last, "content") and last.content:
                        final_message = last.content
                # 收集导出文件
                if "export_files" in update:
                    export_files = update["export_files"]

        print(f"\n{'═' * 55}")
        print("  ✅ 执行完成")
        print(f"{'═' * 55}\n")

    except Exception as e:
        logger.error(f"Error running query: {e}")
        raise

    return {
        "final_message": final_message,
        "nodes_visited": nodes_visited,
        "export_files": export_files,
    }


# ── 异步 token 级流式查询 ──────────────────────────────────────────────────

async def run_query_streaming_async(question: str, thread_id: str) -> dict:
    """异步 token 级流式输出（astream_events）。

    通过 LangGraph 的 astream_events API 捕获每个 LLM chunk 事件，
    在 LLM 生成 token 时实时逐字打印，体验更流畅。

    参数:
        question:  用户的自然语言问题
        thread_id: 会话线程 ID

    返回:
        dict，包含 final_message / nodes_visited / export_files
    """
    logger.info(f"Streaming query (thread={thread_id}): {question}")
    config_with_thread = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'═' * 55}")
    print(f"  🤖 SQL Agent 流式输出模式…")
    print(f"{'═' * 55}\n")

    current_node = ""
    nodes_visited: list[str] = []
    token_buf: list[str] = []  # 累积当前节点 LLM 输出
    final_message: str = ""

    try:
        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": question}]},
            config_with_thread,
            version="v2",
        ):
            kind = event.get("event", "")
            metadata = event.get("metadata", {})

            # 节点开始
            if kind == "on_chain_start" and "langgraph_node" in metadata:
                node = metadata["langgraph_node"]
                if node != current_node:
                    current_node = node
                    nodes_visited.append(node)
                    label = _node_label(node)
                    print(f"\n{'─' * 55}")
                    print(f"  {label}")
                    print(f"{'─' * 55}")

            # LLM token 实时输出
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    print(chunk.content, end="", flush=True)
                    token_buf.append(chunk.content)

            # LLM 调用结束（换行，保存最终消息）
            elif kind == "on_chat_model_end":
                print()
                if token_buf:
                    final_message = "".join(token_buf)
                    token_buf = []

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise

    # Retrieve final graph state to extract export_files (not available in event stream)
    export_files: list[str] = []
    try:
        snapshot = await graph.aget_state(config_with_thread)
        if snapshot:
            export_files = snapshot.values.get("export_files", [])
    except Exception as e:
        logger.warning(f"Failed to retrieve graph state after streaming: {e}")

    print(f"\n{'═' * 55}")
    print("  ✅ 流式输出完成")
    return {
        "final_message": final_message,
        "nodes_visited": nodes_visited,
        "export_files": export_files,
    }


def run_query_streaming(question: str, thread_id: str) -> dict:
    """同步入口：调用异步 token 级流式输出。

    对外提供与 run_query() 相同签名的同步接口，内部使用 asyncio 运行异步版本。
    返回与 run_query() 相同的结构化 dict。
    """
    return asyncio.run(run_query_streaming_async(question, thread_id))


# ── 主循环 ─────────────────────────────────────────────────────────────────

def main() -> None:
    """交互式主循环，等待用户输入问题（支持多轮对话记忆）。

    流式模式选择:
        默认使用 run_query()（同步节点级流式）。
        输入 'stream' 切换至 token 级实时流式模式。
    """
    thread_id = str(uuid.uuid4())
    streaming_mode = False   # False = 节点级流式；True = token 级流式

    print("=" * 55)
    print("SQL Agent 已启动 (支持多轮对话记忆 + 流式输出)")
    print(f"数据库: {config.database.uri.split('@')[-1] if '@' in config.database.uri else config.database.uri}")
    print(f"模型: {config.llm.provider}:{config.llm.model}")
    print(f"会话ID: {thread_id[:8]}…")
    print("=" * 55)
    print("输入问题进行查询")
    print("输入 'stream'  切换 token 级流式模式（当前: 节点级）")
    print("输入 'new'     开始新会话")
    print("输入 'quit'    退出\n")

    while True:
        try:
            mode_tag = "[token流式]" if streaming_mode else "[节点流式]"
            question = input(f"{mode_tag} 请输入您的问题: ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("说不出分手！说好组一辈子的agent呢？")
                break

            if question.lower() == "new":
                thread_id = str(uuid.uuid4())
                print(f"\n已开始新会话，会话ID: {thread_id[:8]}…\n")
                continue

            if question.lower() == "stream":
                streaming_mode = not streaming_mode
                mode = "token 级流式" if streaming_mode else "节点级流式"
                print(f"\n已切换至 {mode} 模式\n")
                continue

            print()
            if streaming_mode:
                run_query_streaming(question, thread_id)
            else:
                run_query(question, thread_id)

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"发生错误: {e}\n")


if __name__ == "__main__":
    main()
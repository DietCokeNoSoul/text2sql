"""SQL 执行前的用户确认对话框及跳过原因分析。

当 SQL_CONFIRM_ENABLED=true 时，每条 SQL 在执行前都会暂停并等待
用户选择 [E]xecute（执行）或 [S]kip（跳过）。
若用户跳过，LLM 会分析该 SQL 的风险并建议下一步操作。

Web 模式扩展：
  register_web_hook(session_id, hook) 注册一个同步回调，替代终端 input()。
  set_web_session(session_id) 通过 ContextVar 传播当前会话 ID（自动随协程/线程继承）。
"""

import contextvars
import logging
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# 跳过信号标记，用于在消息中传递跳过事件
SKIP_SIGNAL_TAG = "[SKIP_SIGNAL]"

# ── Web 模式钩子 ────────────────────────────────────────────────────────────

# ContextVar：跨协程/线程自动继承当前 session_id（Python 3.7+ asyncio task + executor 均复制 context）
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sql_confirm_session", default=""
)

# session_id → 同步回调 hook(sql) → (action, reason)
_web_hooks: dict[str, Callable[[str], Tuple[str, str]]] = {}


def set_web_session(session_id: str) -> None:
    """设置当前上下文的会话 ID（供 SSE 端点在启动 graph 前调用）。"""
    _current_session_id.set(session_id)


def register_web_hook(session_id: str, hook: Callable[[str], Tuple[str, str]]) -> None:
    """注册 Web 模式的 SQL 确认回调，替代终端 input()。"""
    _web_hooks[session_id] = hook


def unregister_web_hook(session_id: str) -> None:
    """注销 Web 模式回调（请求结束后清理）。"""
    _web_hooks.pop(session_id, None)


def prompt_sql_confirmation(sql: str) -> Tuple[str, str]:
    """在控制台展示 SQL 并等待用户确认；Web 模式下调用已注册的 hook。

    返回:
        ("execute", "")   — 用户选择执行
        ("skip", reason)  — 用户选择跳过，reason 为可选的跳过原因
    """
    # Web 模式：查找当前 session 的 hook
    session_id = _current_session_id.get("")
    if session_id and session_id in _web_hooks:
        return _web_hooks[session_id](sql)

    # 终端模式（原逻辑）
    print(f"\n{'═' * 60}")
    print("  ⚠️  SQL 待执行确认")
    print(f"{'─' * 60}")
    # 对长 SQL 适当缩进换行
    for line in sql.strip().splitlines():
        print(f"  {line}")
    print(f"{'─' * 60}")
    print("  [E] Execute — 执行此 SQL")
    print("  [S] Skip    — 跳过此 SQL")
    print(f"{'═' * 60}")

    while True:
        try:
            choice = input("  请选择 [E/S]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # 非交互式环境默认执行
            print("\n  （非交互式环境，默认执行）")
            return "execute", ""

        if choice in ("e", "execute", "yes", "y", ""):
            logger.info("[SQLConfirm] User chose to execute SQL")
            return "execute", ""
        elif choice in ("s", "skip", "no", "n"):
            try:
                reason = input("  跳过原因（可选，直接回车跳过）: ").strip()
            except (EOFError, KeyboardInterrupt):
                reason = ""
            logger.info(f"[SQLConfirm] User chose to skip SQL. Reason: {reason!r}")
            return "skip", reason
        else:
            print("  请输入 E（执行）或 S（跳过）")


def build_skip_message(sql: str, reason: str) -> str:
    """构建带有 SKIP_SIGNAL 标记的消息，供 run_query() 识别跳过事件。"""
    reason_part = f" REASON={reason}" if reason else ""
    return f"⚠️ SQL已跳过 {SKIP_SIGNAL_TAG} SQL={sql}{reason_part}"


def parse_skip_signal(message: str) -> Optional[Tuple[str, str]]:
    """从消息中解析跳过信号。

    返回:
        (sql, reason) — 如果消息含有 SKIP_SIGNAL 标记
        None          — 否则
    """
    if SKIP_SIGNAL_TAG not in message:
        return None

    # 提取 SQL= 之后的内容
    sql_start = message.find("SQL=")
    if sql_start == -1:
        return None, ""

    sql_part = message[sql_start + 4:]

    # 尝试提取 REASON=
    reason = ""
    reason_start = sql_part.find(" REASON=")
    if reason_start != -1:
        reason = sql_part[reason_start + 8:]
        sql_part = sql_part[:reason_start]

    return sql_part.strip(), reason.strip()


def analyze_sql_skip(sql: str, reason: str, llm) -> str:
    """调用 LLM 分析被跳过的 SQL，给出风险评估和建议。

    参数:
        sql:    被跳过的 SQL 语句
        reason: 用户提供的跳过原因（可为空）
        llm:    BaseChatModel 实例

    返回:
        LLM 生成的分析文本
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    reason_line = f"用户跳过原因：{reason}" if reason else "用户未提供跳过原因"

    prompt = f"""用户刚刚选择跳过了以下 SQL 语句的执行：

```sql
{sql}
```

{reason_line}

请从以下几个角度分析：
1. **SQL 风险评估**：该 SQL 是否存在安全风险或数据修改风险？（只读查询 / 潜在风险）
2. **跳过原因判断**：用户跳过是因为 SQL 有问题，还是可能是误操作？
3. **建议**：
   - 如果 SQL 看起来安全且合理，建议用户重新确认是否误触跳过
   - 如果 SQL 存在问题，建议 LLM 重新生成一个更安全的版本
   - 如果无法判断，建议用户澄清需求

请用简洁的中文回答，不超过200字。"""

    messages = [
        SystemMessage(content="你是一位 SQL 安全专家，帮助用户评估 SQL 语句的风险。"),
        HumanMessage(content=prompt),
    ]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.warning(f"[SQLConfirm] Skip analysis failed: {e}")
        return f"无法完成分析（{e}）"

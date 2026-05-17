"""SQL 步骤事件发射器 — 技能执行每步 SQL 时向 Web 前端推送实时事件。

用法（技能内部）：
    from agent import sql_step_emitter
    sql_step_emitter.emit("1", "查询用户总数", "SELECT COUNT(*) FROM tb_user")

用法（server.py 注册）：
    sql_step_emitter.register_hook(session_id, lambda sid, label, sql: ...)
    sql_step_emitter.set_session(session_id)   # 在 graph 启动前的 async 上下文中调用
"""

import contextvars
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# ContextVar：Python 3.7+ asyncio task + executor 均会复制父上下文
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sql_step_session", default=""
)

# session_id → hook(step_id, label, sql, performance, optimization, elapsed_ms)
_hooks: dict[str, Callable[[str, str, str, dict | None, dict | None, int | None], None]] = {}


def set_session(session_id: str) -> None:
    """设置当前上下文的会话 ID（在启动 graph 前调用）。"""
    _current_session_id.set(session_id)


def register_hook(session_id: str, hook: Callable[[str, str, str, dict | None, dict | None, int | None], None]) -> None:
    """注册 SQL 步骤事件回调 hook(step_id, label, sql, performance, optimization, elapsed_ms)。"""
    _hooks[session_id] = hook


def unregister_hook(session_id: str) -> None:
    """注销 SQL 步骤事件回调（请求结束后清理）。"""
    _hooks.pop(session_id, None)


def emit(
    step_id: str,
    label: str,
    sql: str,
    performance: dict | None = None,
    optimization: dict | None = None,
    elapsed_ms: int | None = None,
) -> None:
    """向当前会话的前端发射一个 SQL 步骤执行事件。"""
    sid = _current_session_id.get("")
    if sid and sid in _hooks:
        try:
            _hooks[sid](step_id, label, sql, performance, optimization, elapsed_ms)
        except Exception as e:
            logger.warning(f"[SqlStepEmitter] hook error: {e}")

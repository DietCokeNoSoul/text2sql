"""
Shared SQL Error Correction Utility (v2 — category-aware)

Provides execute_with_correction(): run a SQL query with automatic retry
on error, using sql_errors.py to classify each exception and route it to
the most appropriate fix strategy.

Strategy routing
────────────────
  fail_fast              – re-raise immediately (connection / security / constraint)
  block_and_audit        – log and return failure without retry (security)
  inject_limit           – prepend LIMIT clause and retry once (performance)
  fuzzy_match_then_llm   – extract bad column, find similar, LLM fix (schema)
  llm_rewrite            – ask LLM to fix based on error text (syntax / type)
  llm_rewrite_with_type_hint – LLM fix augmented with type-specific guidance

Used by data_analysis and complex_query skills.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.sql_errors import (
    SQLErrorCategory,
    SQLErrorInfo,
    classify_sql_error,
)

logger = logging.getLogger(__name__)

# Maximum rows to inject when strategy is "inject_limit"
_AUTO_LIMIT = 1000


# ── Column-name helpers ──────────────────────────────────────────────────────

def _build_column_hint(error_info: SQLErrorInfo, db_manager: Any) -> str:
    """Build a human-readable column suggestion hint from a classified error."""
    bad_col = error_info.bad_token
    if not bad_col:
        return ""

    suggestions = db_manager.find_similar_columns(bad_col)
    if not suggestions:
        column_map = db_manager.get_column_map()
        lines = [f"  - {tbl}: {', '.join(cols)}" for tbl, cols in column_map.items() if cols]
        cols_text = "\n".join(lines) if lines else "（无法获取列信息）"
        return (
            f"**列名纠错提示**:\n"
            f"- 错误列名: `{bad_col}`\n"
            f"- 无法找到相似列名，以下是所有可用列，请从中选择最合适的：\n{cols_text}"
        )

    suggestions_text = "\n".join(f"  - `{s}`" for s in suggestions)
    return (
        f"**列名纠错提示**:\n"
        f"- 错误列名: `{bad_col}`\n"
        f"- 最相似的实际列名（优先选择这些）:\n{suggestions_text}"
    )


# ── LLM prompts ──────────────────────────────────────────────────────────────

_FIX_SYSTEM_PROMPT = """[Role & Policies]
您是一位 SQL 错误修复专家。只修复错误，不改变业务逻辑，不添加解释文字。

[Task]
根据提供的错误信息，输出修复后的 SQL 语句（纯文本，不要 Markdown 代码块）。

[Environment]
（错误信息、列名提示等由调用方在 HumanMessage 中提供）

[Evidence]
（错误列名建议和类型提示由调用方在 HumanMessage 中提供）

[Context]
（无）

[Output]
修复规则：
1. 列名错误 → 用建议列名替换，保留所有其他子句
2. 语法错误 → 修正语法，不改变业务逻辑
3. 类型错误 → 加入合适的 CAST/日期格式转换/NULLIF 防零除
4. GROUP BY 错误 → 把 SELECT 里的非聚合列加入 GROUP BY
5. 只返回修复后的 SQL，不要任何解释文字""".strip()

_TYPE_HINT_EXTRA = """
**类型修复提示**:
- 日期比较请使用 DATE()/STR_TO_DATE() 转换
- 数字比较请确保字段与字面量类型一致
- 防止除零请使用 NULLIF(分母, 0)
"""

_PERFORMANCE_HINT_EXTRA = """
**性能修复提示**:
- 本次执行超出行数/时间限制
- 优先考虑缩小查询范围（加 WHERE 条件 / 减少 JOIN）
- 若业务允许，使用 LIMIT 限制结果集
"""


def _llm_fix_sql(
    llm: BaseChatModel,
    bad_sql: str,
    error_info: SQLErrorInfo,
    column_hint: str = "",
    extra_hint: str = "",
) -> str:
    """Ask LLM to rewrite bad_sql; returns the fixed SQL string."""
    column_section = f"\n\n{column_hint}" if column_hint else ""
    extra_section  = f"\n\n{extra_hint}"  if extra_hint  else ""
    hints_section  = ""
    if error_info.hints:
        hints_section = "\n**额外提示**:\n" + "\n".join(f"- {h}" for h in error_info.hints)

    user_content = (
        f"原始 SQL:\n{bad_sql}\n\n"
        f"错误类型: {error_info.error_class}\n"
        f"错误信息:\n{error_info.raw_message}"
        f"{column_section}{extra_section}{hints_section}\n\n"
        "请输出修复后的 SQL（只输出 SQL，不要任何解释）："
    )
    messages = [
        SystemMessage(content=_FIX_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    fixed = response.content.strip()
    fixed = re.sub(r"^```(?:sql)?\s*", "", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\s*```$", "", fixed)
    return fixed.strip()


def _inject_limit(sql: str, limit: int = _AUTO_LIMIT) -> str:
    """Add or replace a LIMIT clause in a SELECT statement."""
    sql = sql.rstrip("; \t\n")
    if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        # Replace existing LIMIT value
        return re.sub(r"\bLIMIT\s+\d+\b", f"LIMIT {limit}", sql, flags=re.IGNORECASE)
    return f"{sql} LIMIT {limit}"


# ── Strategy dispatcher ───────────────────────────────────────────────────────

def _apply_fix(
    error_info: SQLErrorInfo,
    bad_sql: str,
    db_manager: Any,
    llm: BaseChatModel,
    label: str,
    attempt: int,
) -> Optional[str]:
    """
    Apply the appropriate fix for error_info.fix_strategy.

    Returns the fixed SQL string, or None if the strategy indicates we should
    stop retrying (fail_fast / block_and_audit).
    """
    strategy = error_info.fix_strategy
    category = error_info.category

    # ── fail-fast categories ─────────────────────────────────────────────────
    if strategy in ("fail_fast", "block_and_audit"):
        logger.warning(
            f"{label} Error category '{category.value}' ({error_info.error_class}) "
            "is non-retryable — aborting correction loop."
        )
        return None  # signal caller to stop

    # ── inject LIMIT ─────────────────────────────────────────────────────────
    if strategy == "inject_limit":
        fixed = _inject_limit(bad_sql)
        logger.info(f"{label} Auto-injected LIMIT {_AUTO_LIMIT}: {fixed[:200]}")
        return fixed

    # ── schema: fuzzy match + LLM ────────────────────────────────────────────
    if strategy == "fuzzy_match_then_llm":
        column_hint = _build_column_hint(error_info, db_manager)
        if column_hint:
            logger.info(f"{label} Column hint: {column_hint[:200]}")
        return _llm_fix_sql(llm, bad_sql, error_info, column_hint=column_hint)

    # ── type: LLM with type-specific guidance ────────────────────────────────
    if strategy == "llm_rewrite_with_type_hint":
        return _llm_fix_sql(llm, bad_sql, error_info, extra_hint=_TYPE_HINT_EXTRA.strip())

    # ── performance: LLM rewrite (cartesian / too many joins / timeout) ──────
    if category == SQLErrorCategory.PERFORMANCE:
        return _llm_fix_sql(llm, bad_sql, error_info, extra_hint=_PERFORMANCE_HINT_EXTRA.strip())

    # ── default: plain LLM rewrite ───────────────────────────────────────────
    return _llm_fix_sql(llm, bad_sql, error_info)


# ── Public API ────────────────────────────────────────────────────────────────

def execute_with_correction(
    sql: str,
    query_tool: Any,
    db_manager: Any,
    llm: BaseChatModel,
    max_retries: int = 2,
    context_label: str = "",
) -> Dict[str, Any]:
    """Execute a SQL query with category-aware automatic error correction.

    Parameters
    ----------
    sql:
        The SQL query to execute.
    query_tool:
        LangChain tool with a ``.invoke({"query": sql})`` interface.
    db_manager:
        ``SQLDatabaseManager`` instance (provides ``find_similar_columns``
        and ``get_column_map``).
    llm:
        ``BaseChatModel`` used for SQL rewriting on failure.
    max_retries:
        Maximum number of LLM-rewrite attempts (default 2).
        fail_fast / block_and_audit errors ignore this and stop immediately.
    context_label:
        Short label for log messages (e.g. ``"[DataAnalysis] step 2"``).

    Returns
    -------
    dict with keys:
        - ``result``      – query result string (empty string on total failure)
        - ``final_sql``   – the SQL that ultimately succeeded (or last tried)
        - ``success``     – bool
        - ``error``       – last error message (empty string on success)
        - ``retries``     – number of LLM-rewrite attempts made
        - ``error_class`` – SQLError subclass name (empty string on success)
        - ``error_category`` – SQLErrorCategory value (empty string on success)
    """
    label = context_label or "[SQLCorrection]"
    current_sql = sql
    last_error = ""
    last_error_info: Optional[SQLErrorInfo] = None

    for attempt in range(max_retries + 1):
        try:
            result = query_tool.invoke({"query": current_sql})
            if attempt > 0:
                logger.info(f"{label} SQL correction succeeded on attempt {attempt}")
            return {
                "result":         result,
                "final_sql":      current_sql,
                "success":        True,
                "error":          "",
                "retries":        attempt,
                "error_class":    "",
                "error_category": "",
            }
        except Exception as exc:
            last_error = str(exc)
            last_error_info = classify_sql_error(exc, sql=current_sql)
            logger.warning(
                f"{label} SQL failed (attempt {attempt}) "
                f"[{last_error_info.error_class}]: {last_error[:200]}"
            )

            if attempt >= max_retries:
                break

            # Try to produce a fixed SQL via category-aware strategy
            try:
                fixed_sql = _apply_fix(
                    last_error_info, current_sql, db_manager, llm, label, attempt
                )
            except Exception as llm_exc:
                logger.error(f"{label} Fix attempt failed: {llm_exc}")
                break

            if fixed_sql is None:
                # fail_fast / block_and_audit — stop immediately
                break

            logger.info(
                f"{label} [{last_error_info.fix_strategy}] fix proposed "
                f"(attempt {attempt + 1}): {fixed_sql[:200]}"
            )
            current_sql = fixed_sql

    error_info = last_error_info
    return {
        "result":         "",
        "final_sql":      current_sql,
        "success":        False,
        "error":          last_error,
        "retries":        max_retries,
        "error_class":    error_info.error_class    if error_info else "",
        "error_category": error_info.category.value if error_info else "",
    }

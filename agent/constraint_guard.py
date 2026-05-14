"""用户禁令守卫 — SQL 执行前的硬性约束检查。

工作原理
---------
1. 从每条禁令文本中提取关键词（SQL 标识符 / 英文单词 / 中文片段）。
2. SQL 执行前检查 SQL 文本是否包含这些关键词（大小写无关，词边界匹配）。
3. 命中后立即抛出 UserPolicyError（fix_strategy=fail_fast），跳过 LLM 重试。
"""

from __future__ import annotations

import logging
import re
from typing import List

from .sql_errors import (
    TableConstraintViolationError,
    ColumnConstraintViolationError,
    UserPolicyError,
)

logger = logging.getLogger(__name__)

# 停用词：这些词不当作匹配关键词
_STOPWORDS = {
    # 中文动词/副词/介词
    "不", "不得", "不能", "禁止", "不允许", "请勿", "无法", "拒绝",
    "查询", "访问", "使用", "操作", "执行",
    "的", "和", "或", "以及", "等",
    # 英文停用词
    "do", "not", "dont", "no", "never", "cannot", "cant", "should",
    "query", "access", "use", "select", "from", "where",
}

_MIN_KEYWORD_LEN = 3


def _extract_keywords(constraint: str) -> List[str]:
    """从禁令文本中提取用于 SQL 匹配的关键词列表（去重保序）。

    提取优先级:
    1. SQL 标识符（含下划线，如 tb_voucher）
    2. 纯英文单词（长度 >= 3，如 voucher、salary）
    3. 中文连续片段（2 汉字以上，如 优惠券）
    """
    keywords: List[str] = []

    # 1. SQL 标识符（含下划线）
    for m in re.finditer(r"\b[a-zA-Z][a-zA-Z0-9_]*_[a-zA-Z0-9_]+\b", constraint):
        kw = m.group().lower()
        if len(kw) >= _MIN_KEYWORD_LEN:
            keywords.append(kw)

    # 2. 纯英文单词
    for m in re.finditer(r"\b[a-zA-Z]{3,}\b", constraint):
        kw = m.group().lower()
        if kw not in _STOPWORDS and kw not in keywords:
            keywords.append(kw)

    # 3. 中文片段
    for m in re.finditer("[\u4e00-\u9fff]{2,}", constraint):
        kw = m.group()
        if kw not in _STOPWORDS and kw not in keywords:
            keywords.append(kw)

    return list(dict.fromkeys(keywords))


def _sql_contains_keyword(sql: str, keyword: str) -> bool:
    """大小写不敏感地检查 SQL 中是否包含 keyword。

    对英文标识符使用词边界匹配，避免 name 误命中 nick_name。
    对中文使用简单 in 运算符。
    """
    if not keyword:
        return False

    if re.search("[\u4e00-\u9fff]", keyword):
        return keyword in sql

    pattern = r"(?<![a-zA-Z0-9_])" + re.escape(keyword) + r"(?![a-zA-Z0-9_])"
    return bool(re.search(pattern, sql, re.IGNORECASE))


def check_constraints(sql: str, constraints: List[str]) -> None:
    """对即将执行的 SQL 进行用户禁令检查。

    参数:
        sql:         即将执行的 SQL 语句
        constraints: 已启用的禁令字符串列表（来自 state["constraints"]）

    异常:
        TableConstraintViolationError  — 匹配到表名级禁令
        ColumnConstraintViolationError — 匹配到字段名级禁令
    """
    if not constraints or not sql.strip():
        return

    for constraint in constraints:
        keywords = _extract_keywords(constraint)
        if not keywords:
            logger.debug("[ConstraintGuard] No keywords extracted from: %r", constraint)
            continue

        for kw in keywords:
            if _sql_contains_keyword(sql, kw):
                # 含下划线或以 tb 开头 → 更可能是表名
                is_table = "_" in kw or kw.lower().startswith("tb")
                exc_cls = (
                    TableConstraintViolationError if is_table
                    else ColumnConstraintViolationError
                )

                logger.warning(
                    "[ConstraintGuard] BLOCKED — constraint=%r  keyword=%r",
                    constraint, kw,
                )
                raise exc_cls(
                    message=(
                        "禁令拦截：该查询违反了用户禁令\u300c" + constraint + "\u300d"
                        "（匹配关键词：" + kw + "）"
                    ),
                    matched_constraint=constraint,
                    matched_keyword=kw,
                    sql=sql,
                )


def build_block_message(exc: UserPolicyError) -> str:
    """生成展示给用户的禁令拦截说明（Markdown 格式）。"""
    return (
        "\u26d4 **查询已被禁令拦截**\n\n"
        "- **触发禁令**\uff1a" + exc.matched_constraint + "\n"
        "- **匹配关键词**\uff1a`" + exc.matched_keyword + "`\n\n"
        "如需查询，请先在右侧抽屉的\u300c禁令\u300d面板中删除或禁用该条规则。"
    )

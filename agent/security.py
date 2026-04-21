"""SQL 安全护栏模块。

四层防御架构：
- Layer 1: 语句类型控制  —— 拒绝 SELECT 以外的所有 DML/DDL
- Layer 2: 表访问控制    —— allowlist / denylist 双向过滤
- Layer 3: 查询复杂度限制 —— 注入/降低 LIMIT，限制 SQL 长度
- Layer 4: 结果脱敏      —— 检测并标记包含敏感列的查询结果

典型用法::

    from agent.security import SQLSecurityGuard
    from agent.config import SecurityConfig

    guard = SQLSecurityGuard(SecurityConfig(), dialect="mysql")

    result = guard.validate("SELECT * FROM users")
    if not result.passed:
        raise SecurityViolationError(result.reason, result.layer)

    # 可能重写了 SQL（例如注入了 LIMIT）
    safe_sql = result.rewritten_sql

    # 执行查询后脱敏
    raw_output = db.run(safe_sql)
    safe_output = guard.sanitize_result(raw_output, safe_sql)
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import sqlglot
import sqlglot.expressions as exp

from .config import SecurityConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据类：验证结果
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """安全验证结果。"""

    passed: bool
    """True 表示通过所有检查。"""

    layer: Optional[str] = None
    """触发拦截的防御层名称，通过时为 None。"""

    reason: Optional[str] = None
    """拦截原因，通过时为 None。"""

    rewritten_sql: Optional[str] = None
    """Layer 3 可能重写的 SQL（注入/降低 LIMIT）；未重写时为 None。"""


# ---------------------------------------------------------------------------
# 核心护栏类
# ---------------------------------------------------------------------------

class SQLSecurityGuard:
    """SQL 安全护栏，四层防御架构。

    参数:
        config: :class:`~agent.config.SecurityConfig` 实例。
        dialect: sqlglot 方言字符串（``"mysql"``、``"sqlite"`` 等）；
                 空字符串表示自动检测。
    """

    # sqlglot Statement 类名 → 标准名映射
    _STMT_TYPE_MAP: Dict[str, str] = {
        "SELECT": "SELECT",
        "INSERT": "INSERT",
        "UPDATE": "UPDATE",
        "DELETE": "DELETE",
        "DROP": "DROP",
        "CREATE": "CREATE",
        "ALTER": "ALTER",
        "TRUNCATE": "TRUNCATE",
        "MERGE": "MERGE",
        "REPLACE": "REPLACE",
    }

    def __init__(self, config: SecurityConfig, dialect: str = "") -> None:
        self.config = config
        self.dialect = dialect
        self._audit_records: List[Dict[str, Any]] = []
        self._sensitive_re: List[re.Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in config.sensitive_column_patterns
        ]

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def validate(self, sql: str) -> ValidationResult:
        """对 SQL 语句执行全套安全检查。

        检查顺序：Layer 1 → Layer 2 → Layer 3。
        任意层不通过立即返回，不继续后续层。

        参数:
            sql: 待检查的 SQL 语句。

        返回:
            :class:`ValidationResult`；``passed=True`` 时 ``rewritten_sql``
            包含最终可安全执行的 SQL。
        """
        sql = sql.strip()

        for check in (
            self._check_statement_type,
            self._check_table_access,
            self._check_complexity,
        ):
            result = check(sql)
            if not result.passed:
                self._audit("BLOCKED", sql, result.layer, result.reason)
                return result
            # Layer 3 可能重写 SQL，后续操作基于重写后的版本
            if result.rewritten_sql:
                sql = result.rewritten_sql

        self._audit("ALLOWED", sql, None, None)
        return ValidationResult(passed=True, rewritten_sql=sql)

    def sanitize_result(self, result: str, sql: str) -> str:
        """Layer 4：检查查询是否涉及敏感列，并在日志中标记警告。

        当前策略为「警告式脱敏」：
        - 检测 SQL 中是否查询了敏感列（按正则匹配列名）。
        - 若检测到，在日志中输出警告，并在返回结果末尾追加脱敏提示行。
        - 不对结果内容做破坏性字符替换（避免损坏合法数据）。

        如需更严格的脱敏策略（如直接替换值），可在子类中覆写本方法。

        参数:
            result: 数据库返回的原始结果字符串。
            sql: 执行的 SQL 语句（用于提取列名）。

        返回:
            处理后的结果字符串。
        """
        sensitive_cols = self._extract_sensitive_columns(sql)
        if not sensitive_cols:
            return result

        for col in sensitive_cols:
            logger.warning(
                f"[SecurityGuard] ⚠️ Sensitive column queried: '{col}' — "
                "review result before exposing to end users."
            )

        notice = (
            f"\n[SecurityGuard] ⚠️ Warning: result contains sensitive "
            f"column(s): {sensitive_cols}. Handle with care."
        )
        return result + notice

    def is_sensitive_column(self, col_name: str) -> bool:
        """判断列名是否匹配任一敏感模式。"""
        return any(p.search(col_name) for p in self._sensitive_re)

    # ------------------------------------------------------------------
    # Layer 1：语句类型检查
    # ------------------------------------------------------------------

    def _check_statement_type(self, sql: str) -> ValidationResult:
        """仅允许 :attr:`SecurityConfig.allowed_statements` 中列出的语句类型。"""
        allowed_upper = {s.upper() for s in self.config.allowed_statements}

        # ① sqlglot AST 解析（主路径，最可靠）
        try:
            statements = sqlglot.parse(sql, dialect=self.dialect or None)
            for stmt in statements:
                if stmt is None:
                    continue
                stmt_type = type(stmt).__name__.upper()
                # sqlglot 类名：Select / Insert / Update / Delete / Drop / …
                # 去掉可能的 "INTO" 等后缀，取主体关键字
                canonical = self._STMT_TYPE_MAP.get(stmt_type, stmt_type)
                if canonical not in allowed_upper:
                    return ValidationResult(
                        passed=False,
                        layer="Layer1_StatementType",
                        reason=(
                            f"Statement type '{canonical}' is not allowed. "
                            f"Only {list(self.config.allowed_statements)} are permitted."
                        ),
                    )
        except Exception as parse_err:
            logger.warning(
                f"[SecurityGuard] sqlglot parse failed ({parse_err}), "
                "falling back to keyword check."
            )
            # ② 降级：首词正则检查
            first_word = (sql.split()[0].upper() if sql.split() else "")
            if first_word not in allowed_upper:
                return ValidationResult(
                    passed=False,
                    layer="Layer1_StatementType",
                    reason=(
                        f"Statement starts with '{first_word}', "
                        f"only {list(self.config.allowed_statements)} are permitted."
                    ),
                )

        # ③ 危险关键字扫描（双重保险，防混淆绕过）
        sql_upper = sql.upper()
        for kw in self.config.blocked_keywords:
            if kw.upper() in sql_upper:
                return ValidationResult(
                    passed=False,
                    layer="Layer1_BlockedKeyword",
                    reason=f"Blocked keyword detected: '{kw}'",
                )

        return ValidationResult(passed=True)

    # ------------------------------------------------------------------
    # Layer 2：表访问控制
    # ------------------------------------------------------------------

    def _check_table_access(self, sql: str) -> ValidationResult:
        """根据 denylist / allowlist 过滤表访问。"""
        try:
            tables = self._extract_table_names(sql)
        except Exception as e:
            logger.warning(f"[SecurityGuard] Table extraction failed: {e}")
            return ValidationResult(passed=True)  # 解析失败宽松放行

        if not tables:
            return ValidationResult(passed=True)

        deny_lower = {d.lower() for d in self.config.table_denylist}
        for table in tables:
            if table.lower() in deny_lower:
                return ValidationResult(
                    passed=False,
                    layer="Layer2_TableDenylist",
                    reason=f"Access to table '{table}' is denied.",
                )

        if self.config.table_allowlist is not None:
            allow_lower = {a.lower() for a in self.config.table_allowlist}
            for table in tables:
                if table.lower() not in allow_lower:
                    return ValidationResult(
                        passed=False,
                        layer="Layer2_TableAllowlist",
                        reason=(
                            f"Table '{table}' is not in the allowlist: "
                            f"{self.config.table_allowlist}"
                        ),
                    )

        return ValidationResult(passed=True)

    def _extract_table_names(self, sql: str) -> List[str]:
        """从 SQL AST 中提取所有引用的表名（去重）。"""
        tables: List[str] = []
        try:
            parsed = sqlglot.parse(sql, dialect=self.dialect or None)
            for stmt in parsed:
                if stmt is None:
                    continue
                for table_node in stmt.find_all(exp.Table):
                    if table_node.name:
                        tables.append(table_node.name)
        except Exception as e:
            raise RuntimeError(f"Failed to parse SQL for table extraction: {e}") from e
        return list(dict.fromkeys(tables))  # 保序去重

    # ------------------------------------------------------------------
    # Layer 3：复杂度限制
    # ------------------------------------------------------------------

    def _check_complexity(self, sql: str) -> ValidationResult:
        """检查 SQL 长度，并强制注入/降低 LIMIT。"""
        if len(sql) > self.config.max_query_length:
            return ValidationResult(
                passed=False,
                layer="Layer3_QueryLength",
                reason=(
                    f"Query length {len(sql)} exceeds maximum "
                    f"{self.config.max_query_length} characters."
                ),
            )

        rewritten = self._enforce_limit(sql)
        return ValidationResult(passed=True, rewritten_sql=rewritten)

    def _enforce_limit(self, sql: str) -> str:
        """如果 SELECT 无 LIMIT 或 LIMIT 超标，自动注入/降低 LIMIT。"""
        try:
            parsed = sqlglot.parse(sql, dialect=self.dialect or None)
            if not parsed or parsed[0] is None:
                return sql

            stmt = parsed[0]
            if not isinstance(stmt, exp.Select):
                return sql  # 非 SELECT 不处理（Layer1 已拦截非 SELECT）

            limit_node = stmt.find(exp.Limit)

            if limit_node is None:
                # 无 LIMIT → 注入
                stmt = stmt.limit(self.config.max_rows)
                rewritten = stmt.sql(dialect=self.dialect or None)
                logger.debug(
                    f"[SecurityGuard] Injected LIMIT {self.config.max_rows}"
                )
                return rewritten
            else:
                # 有 LIMIT → 检查是否超标
                try:
                    current_limit = int(limit_node.expression.this)
                    if current_limit > self.config.max_rows:
                        stmt = stmt.limit(self.config.max_rows)
                        rewritten = stmt.sql(dialect=self.dialect or None)
                        logger.debug(
                            f"[SecurityGuard] Reduced LIMIT "
                            f"{current_limit} → {self.config.max_rows}"
                        )
                        return rewritten
                except (ValueError, AttributeError):
                    pass

        except Exception as e:
            logger.warning(
                f"[SecurityGuard] LIMIT enforcement failed ({e}), "
                "returning original SQL."
            )

        return sql

    # ------------------------------------------------------------------
    # Layer 4：敏感列提取（辅助）
    # ------------------------------------------------------------------

    def _extract_sensitive_columns(self, sql: str) -> List[str]:
        """从 SQL AST 中找出被查询的敏感列名。"""
        sensitive: List[str] = []
        try:
            parsed = sqlglot.parse(sql, dialect=self.dialect or None)
            for stmt in parsed:
                if stmt is None:
                    continue
                for col_node in stmt.find_all(exp.Column):
                    col_name = col_node.name or ""
                    if col_name and self.is_sensitive_column(col_name):
                        sensitive.append(col_name)
        except Exception:
            pass
        return list(dict.fromkeys(sensitive))  # 保序去重

    # ------------------------------------------------------------------
    # 审计日志
    # ------------------------------------------------------------------

    def _audit(
        self,
        action: str,
        sql: str,
        layer: Optional[str],
        reason: Optional[str],
    ) -> None:
        """记录一条审计日志。"""
        if not self.config.enable_audit_log:
            return

        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "action": action,
            "layer": layer,
            "reason": reason,
            "sql_preview": sql[:200] + ("..." if len(sql) > 200 else ""),
        }
        self._audit_records.append(record)

        if action == "BLOCKED":
            logger.warning(
                f"[SecurityGuard] BLOCKED by {layer}: {reason} | "
                f"SQL: {record['sql_preview']}"
            )
        else:
            logger.debug(
                f"[SecurityGuard] ALLOWED | SQL: {record['sql_preview']}"
            )

        if self.config.audit_log_file:
            self._write_audit_file(record)

    def _write_audit_file(self, record: Dict[str, Any]) -> None:
        """将审计记录追加写入 JSONL 文件。"""
        try:
            with open(self.config.audit_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[SecurityGuard] Failed to write audit log: {e}")

    @property
    def audit_records(self) -> List[Dict[str, Any]]:
        """返回内存中全部审计记录的副本。"""
        return list(self._audit_records)

    def clear_audit(self) -> None:
        """清空内存审计记录。"""
        self._audit_records.clear()

    def audit_summary(self) -> Dict[str, int]:
        """返回审计统计摘要。"""
        allowed = sum(1 for r in self._audit_records if r["action"] == "ALLOWED")
        blocked = sum(1 for r in self._audit_records if r["action"] == "BLOCKED")
        return {"total": len(self._audit_records), "allowed": allowed, "blocked": blocked}

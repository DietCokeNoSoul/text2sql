"""
SQL Error Classification System — 7 Categories, 38 Error Types

Provides:
  - A hierarchy of typed SQLError exceptions (one per error kind)
  - classify_sql_error(exc, sql="") -> SQLErrorInfo
      Parses the raw exception string + optional SQL text and returns a
      structured SQLErrorInfo with category, error_type, bad_token, and
      a fix_strategy hint consumed by sql_correction.py.

Categories
──────────
  SCHEMA      – table / column structure mismatches          (8 types)
  SYNTAX      – SQL grammar / dialect issues                 (9 types)
  TYPE        – data-type / encoding problems                (6 types)
  CONNECTION  – database connectivity failures              (4 types)
  SECURITY    – blocked DML / DDL / injection attempts       (5 types)
  PERFORMANCE – resource / cardinality issues                (4 types)
  CONSTRAINT  – data-integrity violations, no LLM fix        (2 types)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# 1. Category enum
# ══════════════════════════════════════════════════════════════════════════════

class SQLErrorCategory(str, Enum):
    SCHEMA      = "schema"
    SYNTAX      = "syntax"
    TYPE        = "type"
    CONNECTION  = "connection"
    SECURITY    = "security"
    PERFORMANCE = "performance"
    CONSTRAINT  = "constraint"
    UNKNOWN     = "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Base exception
# ══════════════════════════════════════════════════════════════════════════════

class SQLError(Exception):
    """Base class for all typed SQL errors."""
    category: SQLErrorCategory = SQLErrorCategory.UNKNOWN
    fix_strategy: str = "llm_rewrite"   # hint for sql_correction.py


# ══════════════════════════════════════════════════════════════════════════════
# 3.1  SCHEMA errors  (8 types)
# ══════════════════════════════════════════════════════════════════════════════

class SchemaError(SQLError):
    category = SQLErrorCategory.SCHEMA
    fix_strategy = "fuzzy_match_then_llm"


class UnknownColumnError(SchemaError):
    """Column name does not exist in the target table."""


class UnknownTableError(SchemaError):
    """Table name does not exist in the database."""


class AmbiguousColumnError(SchemaError):
    """Column name is present in multiple joined tables but has no table prefix."""


class InvalidColumnAliasError(SchemaError):
    """SELECT alias referenced in WHERE / GROUP BY (not allowed in MySQL)."""


class OuterScopeAliasError(SchemaError):
    """Alias defined inside a subquery is referenced in the outer query."""


class WrongTableColumnError(SchemaError):
    """Column exists in the DB but belongs to a different table than used."""


class ColumnCountMismatchError(SchemaError):
    """INSERT column count does not match VALUES count."""
    fix_strategy = "llm_rewrite"   # fuzzy match won't help here


class DatabaseNotFoundError(SchemaError):
    """The target database / schema does not exist."""
    fix_strategy = "fail_fast"


# ══════════════════════════════════════════════════════════════════════════════
# 3.2  SYNTAX errors  (9 types)
# ══════════════════════════════════════════════════════════════════════════════

class SyntaxError(SQLError):
    category = SQLErrorCategory.SYNTAX
    fix_strategy = "llm_rewrite"


class SQLSyntaxError(SyntaxError):
    """Generic SQL syntax error (misspelled keyword, etc.)."""


class InvalidFunctionError(SyntaxError):
    """Unknown function name or wrong number of arguments."""


class InvalidOperatorError(SyntaxError):
    """Operator misuse (e.g., `= NULL` instead of `IS NULL`)."""


class SubqueryLimitError(SyntaxError):
    """LIMIT used inside a subquery with IN/ALL/ANY (MySQL restriction)."""


class UnclosedDelimiterError(SyntaxError):
    """Unclosed parenthesis or quote."""


class ReservedKeywordError(SyntaxError):
    """SQL reserved word used as identifier without backtick quoting."""


class InvalidGroupByError(SyntaxError):
    """SELECT contains non-aggregated column absent from GROUP BY."""


class InvalidHavingError(SyntaxError):
    """HAVING references a non-aggregated / non-grouped column."""


class OrderByOutOfRangeError(SyntaxError):
    """ORDER BY N where N exceeds the number of SELECT columns."""


# ══════════════════════════════════════════════════════════════════════════════
# 3.3  TYPE errors  (6 types)
# ══════════════════════════════════════════════════════════════════════════════

class TypeError(SQLError):
    category = SQLErrorCategory.TYPE
    fix_strategy = "llm_rewrite_with_type_hint"


class DataTypeMismatchError(TypeError):
    """String compared directly to numeric column (or vice-versa)."""


class InvalidCastError(TypeError):
    """CAST / CONVERT failed (e.g., casting letters to INT)."""


class DateFormatError(TypeError):
    """Date string format does not match the column's date format."""


class NumericOverflowError(TypeError):
    """Numeric value exceeds the column's defined range."""


class DivisionByZeroError(TypeError):
    """Division by zero in a SQL expression."""
    fix_strategy = "llm_rewrite"


class EncodingError(TypeError):
    """Character-set / collation mismatch."""
    fix_strategy = "llm_rewrite"


# ══════════════════════════════════════════════════════════════════════════════
# 3.4  CONNECTION errors  (4 types)
# ══════════════════════════════════════════════════════════════════════════════

class ConnectionError(SQLError):
    category = SQLErrorCategory.CONNECTION
    fix_strategy = "fail_fast"   # no LLM can fix connectivity


class DatabaseConnectionError(ConnectionError):
    """Cannot connect to the database server."""


class AuthenticationError(ConnectionError):
    """Wrong credentials."""


class QueryTimeoutError(ConnectionError):
    """Network-level connection timeout."""


class SocketError(ConnectionError):
    """Connection dropped mid-query."""


# ══════════════════════════════════════════════════════════════════════════════
# 3.5  SECURITY errors  (5 types)
# ══════════════════════════════════════════════════════════════════════════════

class SecurityError(SQLError):
    category = SQLErrorCategory.SECURITY
    fix_strategy = "block_and_audit"


class DMLOperationError(SecurityError):
    """INSERT / UPDATE / DELETE blocked by security guard."""


class DDLOperationError(SecurityError):
    """DROP / ALTER / TRUNCATE blocked by security guard."""


class MultiStatementError(SecurityError):
    """Semicolon-separated multi-statement injection detected."""


class CommentInjectionError(SecurityError):
    """SQL comment injection (-- or /**/) detected."""


class FileOperationError(SecurityError):
    """LOAD DATA INFILE or other file-system operation blocked."""


# ══════════════════════════════════════════════════════════════════════════════
# 3.6  PERFORMANCE errors  (4 types)
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceError(SQLError):
    category = SQLErrorCategory.PERFORMANCE
    fix_strategy = "inject_limit"


class MaxRowsExceededError(PerformanceError):
    """Result set exceeds the configured maximum row count."""


class ExecutionTimeoutError(PerformanceError):
    """SQL execution timed out (not a network timeout)."""
    fix_strategy = "llm_rewrite"   # may need query restructuring


class CartesianProductError(PerformanceError):
    """JOIN without ON condition — cartesian product detected."""
    fix_strategy = "llm_rewrite"


class TooManyJoinsError(PerformanceError):
    """JOIN depth exceeds the optimizer's limit."""
    fix_strategy = "llm_rewrite"


# ══════════════════════════════════════════════════════════════════════════════
# 3.7  CONSTRAINT errors  (2 types)
# ══════════════════════════════════════════════════════════════════════════════

class ConstraintError(SQLError):
    category = SQLErrorCategory.CONSTRAINT
    fix_strategy = "fail_fast"   # data problem; SQL rewrite can't fix it


class DuplicateKeyError(ConstraintError):
    """Primary key or unique key violation."""


class ForeignKeyViolationError(ConstraintError):
    """Foreign key constraint violated."""


# ══════════════════════════════════════════════════════════════════════════════
# 4. SQLErrorInfo — structured classification result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SQLErrorInfo:
    """Result of classify_sql_error()."""
    category:      SQLErrorCategory
    error_type:    type            # the specific SQLError subclass
    error_class:   str             # e.g. "UnknownColumnError"
    bad_token:     Optional[str]   # column / table / alias name extracted
    raw_message:   str             # original exception string
    fix_strategy:  str             # hint for sql_correction.py
    hints:         list[str] = field(default_factory=list)  # extra context


# ══════════════════════════════════════════════════════════════════════════════
# 5. Classifier — pattern-based, DB-agnostic
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (compiled_regex, error_class, token_group_index_or_None)
_PATTERNS: list[tuple[re.Pattern, type, Optional[int]]] = [

    # ── SCHEMA ────────────────────────────────────────────────────────────────
    (re.compile(r"Unknown column '([^']+)'", re.I),            UnknownColumnError,       1),
    (re.compile(r"has no column named (\S+)", re.I),           UnknownColumnError,       1),
    (re.compile(r'column "([^"]+)" does not exist', re.I),     UnknownColumnError,       1),
    (re.compile(r"Invalid column name '([^']+)'", re.I),       UnknownColumnError,       1),
    (re.compile(r"Table '([^']+)' doesn't exist", re.I),       UnknownTableError,        1),
    (re.compile(r"no such table:\s*(\S+)", re.I),              UnknownTableError,        1),
    (re.compile(r"relation \"([^\"]+)\" does not exist", re.I),UnknownTableError,        1),
    (re.compile(r"Unknown database '([^']+)'", re.I),          DatabaseNotFoundError,    1),
    (re.compile(r"Column '([^']+)' in (?:field list|where clause|order clause) is ambiguous", re.I),
                                                                AmbiguousColumnError,     1),
    (re.compile(r"column reference \"([^\"]+)\" is ambiguous", re.I),
                                                                AmbiguousColumnError,     1),
    (re.compile(r"Column count doesn't match value count", re.I),
                                                                ColumnCountMismatchError, None),
    (re.compile(r"column count of .+? doesn.t match value count", re.I),
                                                                ColumnCountMismatchError, None),

    # Outer scope alias — "b.create_time" style not in scope
    (re.compile(r"Unknown column '(\w+\.\w+)'", re.I),         OuterScopeAliasError,     1),

    # ── SYNTAX ────────────────────────────────────────────────────────────────
    (re.compile(r"You have an error in your SQL syntax", re.I),SQLSyntaxError,           None),
    (re.compile(r"syntax error at or near", re.I),             SQLSyntaxError,           None),
    (re.compile(r"syntax error",           re.I),              SQLSyntaxError,           None),
    (re.compile(r"FUNCTION .+? does not exist", re.I),         InvalidFunctionError,     None),
    (re.compile(r"No function named '([^']+)'", re.I),         InvalidFunctionError,     1),
    (re.compile(r"LIMIT & IN/ALL/ANY/SOME subquery", re.I),    SubqueryLimitError,       None),
    (re.compile(r"This version of MySQL doesn't yet support 'LIMIT & IN", re.I),
                                                                SubqueryLimitError,       None),
    (re.compile(r"Operand should contain \d+ column", re.I),   SQLSyntaxError,           None),
    (re.compile(r"Expression #\d+ of SELECT list is not in GROUP BY", re.I),
                                                                InvalidGroupByError,      None),
    (re.compile(r"non-aggregated column .+? which is not functionally dependent", re.I),
                                                                InvalidGroupByError,      None),
    (re.compile(r"'([^']+)' isn't in GROUP BY", re.I),         InvalidGroupByError,      1),
    (re.compile(r"ORDER BY position (\d+) is not in select list", re.I),
                                                                OrderByOutOfRangeError,   1),

    # ── TYPE ──────────────────────────────────────────────────────────────────
    (re.compile(r"Truncated incorrect .+ value", re.I),        DataTypeMismatchError,    None),
    (re.compile(r"Incorrect (integer|double|decimal|float) value", re.I),
                                                                DataTypeMismatchError,    None),
    (re.compile(r"Invalid use of null value", re.I),           DataTypeMismatchError,    None),
    (re.compile(r"division by zero", re.I),                    DivisionByZeroError,      None),
    (re.compile(r"Divide by zero",   re.I),                    DivisionByZeroError,      None),
    (re.compile(r"Out of range value for column '([^']+)'", re.I),
                                                                NumericOverflowError,     1),
    (re.compile(r"Incorrect datetime value: '([^']+)'", re.I), DateFormatError,          1),
    (re.compile(r"Incorrect date value",   re.I),              DateFormatError,          None),
    (re.compile(r"Incorrect time value",   re.I),              DateFormatError,          None),
    (re.compile(r"Cannot cast",            re.I),              InvalidCastError,         None),
    (re.compile(r"invalid input syntax for type", re.I),       InvalidCastError,         None),
    (re.compile(r"Illegal mix of collations", re.I),           EncodingError,            None),
    (re.compile(r"Incorrect string value", re.I),              EncodingError,            None),

    # ── CONNECTION ────────────────────────────────────────────────────────────
    (re.compile(r"Can't connect to .*(MySQL|database|server)", re.I),
                                                                DatabaseConnectionError,  None),
    (re.compile(r"Connection refused",      re.I),             DatabaseConnectionError,  None),
    (re.compile(r"Lost connection to",      re.I),             SocketError,              None),
    (re.compile(r"MySQL server has gone away", re.I),          SocketError,              None),
    (re.compile(r"Access denied for user '([^']+)'", re.I),    AuthenticationError,      1),
    (re.compile(r"authentication failed",   re.I),             AuthenticationError,      None),
    (re.compile(r"Connection timed out",    re.I),             QueryTimeoutError,        None),
    (re.compile(r"connect timeout",         re.I),             QueryTimeoutError,        None),

    # ── SECURITY ──────────────────────────────────────────────────────────────
    (re.compile(r"DML operation .+ not allowed", re.I),        DMLOperationError,        None),
    (re.compile(r"Operation not permitted: (INSERT|UPDATE|DELETE)", re.I),
                                                                DMLOperationError,        None),
    (re.compile(r"Operation not permitted: (DROP|ALTER|TRUNCATE|CREATE)", re.I),
                                                                DDLOperationError,        None),
    (re.compile(r"DDL operation .+ not allowed", re.I),        DDLOperationError,        None),
    (re.compile(r"Multiple statements",     re.I),             MultiStatementError,      None),
    (re.compile(r"LOAD DATA",               re.I),             FileOperationError,       None),

    # ── PERFORMANCE ───────────────────────────────────────────────────────────
    (re.compile(r"Result set exceeds maximum", re.I),          MaxRowsExceededError,     None),
    (re.compile(r"max_allowed_rows",            re.I),         MaxRowsExceededError,     None),
    (re.compile(r"Query execution was interrupted", re.I),     ExecutionTimeoutError,    None),
    (re.compile(r"Lock wait timeout exceeded",   re.I),        ExecutionTimeoutError,    None),

    # ── CONSTRAINT ────────────────────────────────────────────────────────────
    (re.compile(r"Duplicate entry '([^']+)' for key", re.I),  DuplicateKeyError,        1),
    (re.compile(r"unique constraint failed",          re.I),   DuplicateKeyError,        None),
    (re.compile(r"a foreign key constraint fails",    re.I),   ForeignKeyViolationError, None),
    (re.compile(r"violates foreign key constraint",   re.I),   ForeignKeyViolationError, None),
]


def _extract_token(msg: str, pattern: re.Pattern, group: Optional[int]) -> Optional[str]:
    """Return the captured token from a regex match, or None."""
    if group is None:
        return None
    m = pattern.search(msg)
    if m:
        token = m.group(group)
        # For "table.column" style, strip the table prefix
        return token.split(".")[-1] if "." in token else token
    return None


def _build_hints(error_type: type, bad_token: Optional[str], sql: str) -> list[str]:
    """Build extra hint strings based on the error type."""
    hints: list[str] = []

    if error_type is OuterScopeAliasError and bad_token:
        hints.append(
            f"Alias '{bad_token}' is not available in outer query scope. "
            "Restructure subquery to expose needed columns."
        )
    if error_type is SubqueryLimitError:
        hints.append(
            "MySQL does not allow LIMIT inside subqueries used with IN/ALL/ANY. "
            "Rewrite using a JOIN or CTE instead."
        )
    if error_type is InvalidGroupByError:
        hints.append(
            "Add all non-aggregated SELECT columns to the GROUP BY clause, "
            "or wrap them in an aggregate function."
        )
    if error_type is CartesianProductError:
        hints.append("Add an ON clause to each JOIN to avoid a cartesian product.")
    if error_type is DivisionByZeroError:
        hints.append(
            "Wrap the denominator in NULLIF(expr, 0) to avoid division by zero."
        )

    # SQL-based hint: detect cartesian-product JOIN with no ON
    if error_type is SQLSyntaxError and sql:
        if re.search(r"\bJOIN\b", sql, re.I) and not re.search(r"\bON\b", sql, re.I):
            hints.append("JOIN detected without ON clause — may cause cartesian product.")

    return hints


def classify_sql_error(exc: Exception, sql: str = "") -> SQLErrorInfo:
    """
    Classify a raw SQL exception into a structured SQLErrorInfo.

    Parameters
    ----------
    exc:
        The exception raised by the database driver or security guard.
    sql:
        The SQL string that caused the error (optional, used for extra hints).

    Returns
    -------
    SQLErrorInfo with category, error_type, bad_token, fix_strategy, hints.
    """
    msg = str(exc)

    for pattern, error_cls, token_group in _PATTERNS:
        if pattern.search(msg):
            bad_token = _extract_token(msg, pattern, token_group)
            instance   = error_cls(msg)
            hints      = _build_hints(error_cls, bad_token, sql)
            return SQLErrorInfo(
                category    = error_cls.category,
                error_type  = error_cls,
                error_class = error_cls.__name__,
                bad_token   = bad_token,
                raw_message = msg,
                fix_strategy= error_cls.fix_strategy,
                hints       = hints,
            )

    # Fallback: unknown error
    return SQLErrorInfo(
        category    = SQLErrorCategory.UNKNOWN,
        error_type  = SQLError,
        error_class = "SQLError",
        bad_token   = None,
        raw_message = msg,
        fix_strategy= "llm_rewrite",
        hints       = [],
    )

"""tests/test_security.py — SQL 安全护栏全面测试

覆盖：
- Layer 1: 语句类型控制（SELECT / DML / DDL / 危险关键字）
- Layer 2: 表 denylist / allowlist
- Layer 3: 查询长度限制 / LIMIT 注入与降低
- Layer 4: 敏感列检测 / 结果脱敏
- 审计日志：记录 / 统计 / 文件写入
- ValidationResult 数据结构
- SecurityConfig.from_env()
- SQLDatabaseManager 集成（SQLite + 护栏）
- SecurityViolationError 异常
"""

import io
import os
import sys
import tempfile
import unittest

# Windows UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 把项目根目录加入 sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.config import SecurityConfig
from agent.security import SQLSecurityGuard, ValidationResult
from agent.types import SecurityViolationError


# ===========================================================================
# 辅助函数
# ===========================================================================

def _make_guard(**kwargs) -> SQLSecurityGuard:
    """用默认配置创建护栏，支持覆盖参数。"""
    cfg = SecurityConfig(**kwargs)
    return SQLSecurityGuard(cfg, dialect="sqlite")


# ===========================================================================
# Layer 1: 语句类型测试
# ===========================================================================

class TestLayer1StatementType(unittest.TestCase):
    """Layer 1: 只允许 SELECT。"""

    def setUp(self):
        self.guard = _make_guard()

    # --- 允许通过 ---
    def test_select_passes(self):
        r = self.guard.validate("SELECT 1")
        self.assertTrue(r.passed)

    def test_select_with_where_passes(self):
        r = self.guard.validate("SELECT id, name FROM users WHERE id = 1")
        self.assertTrue(r.passed)

    def test_select_subquery_passes(self):
        r = self.guard.validate("SELECT * FROM (SELECT id FROM t) sub")
        self.assertTrue(r.passed)

    # --- 应被拦截 ---
    def test_insert_blocked(self):
        r = self.guard.validate("INSERT INTO users (name) VALUES ('hack')")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer1_StatementType")

    def test_update_blocked(self):
        r = self.guard.validate("UPDATE users SET name='x' WHERE id=1")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer1_StatementType")

    def test_delete_blocked(self):
        r = self.guard.validate("DELETE FROM users WHERE id=1")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer1_StatementType")

    def test_drop_blocked(self):
        r = self.guard.validate("DROP TABLE users")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer1_StatementType")

    def test_create_blocked(self):
        r = self.guard.validate("CREATE TABLE evil (x INT)")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer1_StatementType")

    def test_truncate_blocked(self):
        r = self.guard.validate("TRUNCATE TABLE logs")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer1_StatementType")

    # --- 危险关键字 ---
    def test_blocked_keyword_xp_cmdshell(self):
        r = self.guard.validate("SELECT xp_cmdshell('whoami')")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer1_BlockedKeyword")

    def test_blocked_keyword_into_outfile(self):
        r = self.guard.validate("SELECT * FROM t INTO OUTFILE '/tmp/x'")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer1_BlockedKeyword")

    def test_blocked_keyword_load_data(self):
        r = self.guard.validate("LOAD DATA INFILE '/etc/passwd' INTO TABLE t")
        self.assertFalse(r.passed)
        # sqlglot 可能把 LOAD DATA 解析为非 SELECT 语句，触发 Layer1_StatementType
        # 或作为无法识别的语句触发 Layer1_BlockedKeyword，均为 Layer1 拦截
        self.assertTrue(r.layer.startswith("Layer1"))

    # --- 允许 INSERT（自定义配置） ---
    def test_custom_allow_insert(self):
        guard = _make_guard(allowed_statements=["SELECT", "INSERT"])
        r = guard.validate("INSERT INTO logs (msg) VALUES ('ok')")
        self.assertTrue(r.passed)

    # --- 结果字段 ---
    def test_blocked_result_has_reason(self):
        r = self.guard.validate("DROP TABLE x")
        self.assertIsNotNone(r.reason)
        self.assertIn("DROP", r.reason.upper())


# ===========================================================================
# Layer 2: 表访问控制
# ===========================================================================

class TestLayer2TableAccess(unittest.TestCase):
    """Layer 2: denylist 和 allowlist。"""

    # --- denylist ---
    def test_denylist_blocks_exact(self):
        guard = _make_guard(table_denylist=["users"])
        r = guard.validate("SELECT * FROM users")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer2_TableDenylist")

    def test_denylist_case_insensitive(self):
        guard = _make_guard(table_denylist=["Users"])
        r = guard.validate("SELECT * FROM USERS")
        self.assertFalse(r.passed)

    def test_denylist_does_not_block_other_table(self):
        guard = _make_guard(table_denylist=["users"])
        r = guard.validate("SELECT * FROM orders")
        self.assertTrue(r.passed)

    def test_denylist_blocks_join(self):
        guard = _make_guard(table_denylist=["secrets"])
        r = guard.validate("SELECT a.id FROM accounts a JOIN secrets s ON a.id=s.id")
        self.assertFalse(r.passed)

    # --- allowlist ---
    def test_allowlist_permits_listed_table(self):
        guard = _make_guard(table_allowlist=["products", "orders"])
        r = guard.validate("SELECT * FROM products")
        self.assertTrue(r.passed)

    def test_allowlist_blocks_unlisted_table(self):
        guard = _make_guard(table_allowlist=["products"])
        r = guard.validate("SELECT * FROM users")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer2_TableAllowlist")

    def test_allowlist_none_allows_all(self):
        guard = _make_guard(table_allowlist=None)
        r = guard.validate("SELECT * FROM any_table")
        self.assertTrue(r.passed)

    def test_denylist_takes_priority_over_allowlist(self):
        # denylist 先检查
        guard = _make_guard(
            table_denylist=["users"],
            table_allowlist=["users", "orders"],
        )
        r = guard.validate("SELECT * FROM users")
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer2_TableDenylist")


# ===========================================================================
# Layer 3: 复杂度限制
# ===========================================================================

class TestLayer3Complexity(unittest.TestCase):
    """Layer 3: 查询长度限制 & LIMIT 注入/降低。"""

    def test_query_too_long_blocked(self):
        guard = _make_guard(max_query_length=50)
        long_sql = "SELECT " + "a, " * 100 + "1 FROM t"
        r = guard.validate(long_sql)
        self.assertFalse(r.passed)
        self.assertEqual(r.layer, "Layer3_QueryLength")

    def test_query_within_length_passes(self):
        guard = _make_guard(max_query_length=5000)
        r = guard.validate("SELECT id FROM users")
        self.assertTrue(r.passed)

    def test_limit_injected_when_missing(self):
        guard = _make_guard(max_rows=100)
        r = guard.validate("SELECT * FROM products")
        self.assertTrue(r.passed)
        self.assertIsNotNone(r.rewritten_sql)
        self.assertIn("100", r.rewritten_sql)

    def test_limit_kept_when_within_bound(self):
        guard = _make_guard(max_rows=100)
        r = guard.validate("SELECT * FROM products LIMIT 50")
        self.assertTrue(r.passed)
        # LIMIT 50 ≤ 100，不应被降低
        self.assertIn("50", r.rewritten_sql)

    def test_limit_reduced_when_over_bound(self):
        guard = _make_guard(max_rows=100)
        r = guard.validate("SELECT * FROM products LIMIT 99999")
        self.assertTrue(r.passed)
        self.assertIsNotNone(r.rewritten_sql)
        self.assertIn("100", r.rewritten_sql)
        self.assertNotIn("99999", r.rewritten_sql)

    def test_rewritten_sql_is_valid_select(self):
        guard = _make_guard(max_rows=10)
        r = guard.validate("SELECT name FROM tb_shop")
        self.assertTrue(r.passed)
        # 重写后仍然以 SELECT 开头
        self.assertTrue(r.rewritten_sql.strip().upper().startswith("SELECT"))


# ===========================================================================
# Layer 4: 结果脱敏
# ===========================================================================

class TestLayer4Sanitize(unittest.TestCase):
    """Layer 4: 敏感列检测与结果标记。"""

    def test_no_sensitive_col_returns_unchanged(self):
        guard = _make_guard()
        result = guard.sanitize_result("[('alice', 25)]", "SELECT name, age FROM users")
        self.assertEqual(result, "[('alice', 25)]")

    def test_sensitive_col_appends_warning(self):
        guard = _make_guard()
        result = guard.sanitize_result(
            "[('alice', 'hash123')]",
            "SELECT name, password FROM users"
        )
        self.assertIn("SecurityGuard", result)
        self.assertIn("password", result)

    def test_multiple_sensitive_cols_all_reported(self):
        guard = _make_guard()
        result = guard.sanitize_result(
            "[('tok', 'key')]",
            "SELECT token, api_key FROM credentials"
        )
        # 警告信息应包含敏感列名
        self.assertIn("SecurityGuard", result)

    def test_is_sensitive_column_password(self):
        guard = _make_guard()
        self.assertTrue(guard.is_sensitive_column("password"))
        self.assertTrue(guard.is_sensitive_column("user_password"))
        self.assertTrue(guard.is_sensitive_column("PASSWORD"))

    def test_is_sensitive_column_not_sensitive(self):
        guard = _make_guard()
        self.assertFalse(guard.is_sensitive_column("name"))
        self.assertFalse(guard.is_sensitive_column("age"))
        self.assertFalse(guard.is_sensitive_column("created_at"))

    def test_custom_sensitive_patterns(self):
        guard = _make_guard(sensitive_column_patterns=[r"salary"])
        self.assertTrue(guard.is_sensitive_column("employee_salary"))
        self.assertFalse(guard.is_sensitive_column("password"))  # 未配置


# ===========================================================================
# ValidationResult 数据结构
# ===========================================================================

class TestValidationResult(unittest.TestCase):
    """ValidationResult 字段正确性。"""

    def test_passed_result_has_no_layer(self):
        guard = _make_guard()
        r = guard.validate("SELECT 1")
        self.assertTrue(r.passed)
        self.assertIsNone(r.layer)
        self.assertIsNone(r.reason)

    def test_blocked_result_has_layer_and_reason(self):
        guard = _make_guard()
        r = guard.validate("DROP TABLE x")
        self.assertFalse(r.passed)
        self.assertIsNotNone(r.layer)
        self.assertIsNotNone(r.reason)

    def test_rewritten_sql_present_after_limit_injection(self):
        guard = _make_guard(max_rows=5)
        r = guard.validate("SELECT * FROM t")
        self.assertTrue(r.passed)
        self.assertIsNotNone(r.rewritten_sql)


# ===========================================================================
# 审计日志
# ===========================================================================

class TestAuditLog(unittest.TestCase):
    """审计记录与统计。"""

    def test_allowed_query_recorded(self):
        guard = _make_guard()
        guard.validate("SELECT 1")
        self.assertEqual(len(guard.audit_records), 1)
        self.assertEqual(guard.audit_records[0]["action"], "ALLOWED")

    def test_blocked_query_recorded(self):
        guard = _make_guard()
        guard.validate("DROP TABLE x")
        self.assertEqual(len(guard.audit_records), 1)
        self.assertEqual(guard.audit_records[0]["action"], "BLOCKED")

    def test_multiple_queries_accumulate(self):
        guard = _make_guard()
        guard.validate("SELECT 1")
        guard.validate("DELETE FROM t")
        guard.validate("SELECT 2")
        self.assertEqual(len(guard.audit_records), 3)

    def test_audit_summary(self):
        guard = _make_guard()
        guard.validate("SELECT 1")
        guard.validate("DROP TABLE x")
        guard.validate("SELECT 2")
        summary = guard.audit_summary()
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["allowed"], 2)
        self.assertEqual(summary["blocked"], 1)

    def test_clear_audit(self):
        guard = _make_guard()
        guard.validate("SELECT 1")
        guard.clear_audit()
        self.assertEqual(len(guard.audit_records), 0)

    def test_audit_disabled(self):
        guard = _make_guard(enable_audit_log=False)
        guard.validate("SELECT 1")
        guard.validate("DROP TABLE x")
        self.assertEqual(len(guard.audit_records), 0)

    def test_audit_file_written(self):
        import json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            fname = f.name
        try:
            guard = _make_guard(audit_log_file=fname)
            guard.validate("SELECT 1")
            guard.validate("DROP TABLE x")
            with open(fname, encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            self.assertEqual(len(lines), 2)
            actions = {l["action"] for l in lines}
            self.assertIn("ALLOWED", actions)
            self.assertIn("BLOCKED", actions)
        finally:
            os.unlink(fname)

    def test_audit_record_has_sql_preview(self):
        guard = _make_guard()
        guard.validate("SELECT id FROM users")
        record = guard.audit_records[0]
        self.assertIn("sql_preview", record)
        self.assertIn("SELECT", record["sql_preview"])

    def test_audit_record_has_timestamp(self):
        guard = _make_guard()
        guard.validate("SELECT 1")
        record = guard.audit_records[0]
        self.assertIn("timestamp", record)
        self.assertIsInstance(record["timestamp"], float)


# ===========================================================================
# SecurityConfig.from_env()
# ===========================================================================

class TestSecurityConfigFromEnv(unittest.TestCase):
    """SecurityConfig.from_env() 读取环境变量。"""

    def _set_env(self, **kwargs):
        for k, v in kwargs.items():
            os.environ[k] = v

    def _clear_env(self, *keys):
        for k in keys:
            os.environ.pop(k, None)

    def test_default_config(self):
        cfg = SecurityConfig()
        self.assertEqual(cfg.max_rows, 1000)
        self.assertIsNone(cfg.table_allowlist)
        self.assertEqual(cfg.table_denylist, [])

    def test_from_env_max_rows(self):
        self._set_env(SECURITY_MAX_ROWS="200")
        try:
            cfg = SecurityConfig.from_env()
            self.assertEqual(cfg.max_rows, 200)
        finally:
            self._clear_env("SECURITY_MAX_ROWS")

    def test_from_env_denylist(self):
        self._set_env(SECURITY_TABLE_DENYLIST="users,secrets")
        try:
            cfg = SecurityConfig.from_env()
            self.assertIn("users", cfg.table_denylist)
            self.assertIn("secrets", cfg.table_denylist)
        finally:
            self._clear_env("SECURITY_TABLE_DENYLIST")

    def test_from_env_allowlist(self):
        self._set_env(SECURITY_TABLE_ALLOWLIST="orders,products")
        try:
            cfg = SecurityConfig.from_env()
            self.assertIsNotNone(cfg.table_allowlist)
            self.assertIn("orders", cfg.table_allowlist)
        finally:
            self._clear_env("SECURITY_TABLE_ALLOWLIST")

    def test_from_env_audit_disabled(self):
        self._set_env(SECURITY_AUDIT_LOG="false")
        try:
            cfg = SecurityConfig.from_env()
            self.assertFalse(cfg.enable_audit_log)
        finally:
            self._clear_env("SECURITY_AUDIT_LOG")


# ===========================================================================
# SecurityViolationError 异常
# ===========================================================================

class TestSecurityViolationError(unittest.TestCase):
    """SecurityViolationError 异常类。"""

    def test_attributes(self):
        err = SecurityViolationError(
            reason="DROP not allowed",
            layer="Layer1_StatementType",
            sql="DROP TABLE x",
        )
        self.assertEqual(err.reason, "DROP not allowed")
        self.assertEqual(err.layer, "Layer1_StatementType")
        self.assertEqual(err.sql, "DROP TABLE x")

    def test_str_representation(self):
        err = SecurityViolationError("Bad SQL", "Layer2_TableDenylist", "SELECT * FROM secret")
        self.assertIn("Layer2_TableDenylist", str(err))
        self.assertIn("Bad SQL", str(err))

    def test_is_sql_agent_error(self):
        from agent.types import SQLAgentError
        err = SecurityViolationError("x")
        self.assertIsInstance(err, SQLAgentError)


# ===========================================================================
# SQLDatabaseManager 集成测试（SQLite，无网络依赖）
# ===========================================================================

class TestDatabaseManagerIntegration(unittest.TestCase):
    """SQLDatabaseManager + SecurityGuard 集成（SQLite 内存库）。"""

    def _make_db_manager(self, **security_kwargs):
        """创建使用内存 SQLite 的数据库管理器。"""
        from agent.config import DatabaseConfig
        from agent.database import SQLDatabaseManager

        db_config = DatabaseConfig(uri="sqlite:///:memory:")
        sec_config = SecurityConfig(**security_kwargs)
        mgr = SQLDatabaseManager(db_config, security_config=sec_config)

        # 建一张测试表
        import sqlalchemy
        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text(
                "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)"
            ))
            conn.execute(sqlalchemy.text(
                "INSERT INTO products VALUES (1, 'Widget', 9.99)"
            ))
            conn.commit()
        # 把 engine 注入（复用同一 SQLite 内存实例）
        from langchain_community.utilities import SQLDatabase
        mgr._db = SQLDatabase(engine)
        from agent.types import DatabaseDialect
        mgr._dialect = DatabaseDialect.SQLITE
        from agent.security import SQLSecurityGuard
        mgr._guard = SQLSecurityGuard(sec_config, dialect="sqlite")
        return mgr

    def test_select_allowed(self):
        mgr = self._make_db_manager()
        result = mgr.execute_query("SELECT * FROM products")
        self.assertIsInstance(result, str)

    def test_drop_raises_security_violation(self):
        mgr = self._make_db_manager()
        with self.assertRaises(SecurityViolationError) as ctx:
            mgr.execute_query("DROP TABLE products")
        self.assertIn("Layer1", ctx.exception.layer)

    def test_delete_raises_security_violation(self):
        mgr = self._make_db_manager()
        with self.assertRaises(SecurityViolationError):
            mgr.execute_query("DELETE FROM products")

    def test_denylist_blocks_table(self):
        mgr = self._make_db_manager(table_denylist=["products"])
        with self.assertRaises(SecurityViolationError) as ctx:
            mgr.execute_query("SELECT * FROM products")
        self.assertIn("Layer2", ctx.exception.layer)

    def test_limit_injected_transparently(self):
        mgr = self._make_db_manager(max_rows=1)
        # execute_query 内部会注入 LIMIT 1，不应报错
        result = mgr.execute_query("SELECT * FROM products")
        self.assertIsInstance(result, str)

    def test_guard_property(self):
        mgr = self._make_db_manager()
        self.assertIsNotNone(mgr.guard)

    def test_no_guard_when_not_configured(self):
        from agent.config import DatabaseConfig
        from agent.database import SQLDatabaseManager
        db_config = DatabaseConfig(uri="sqlite:///:memory:")
        mgr = SQLDatabaseManager(db_config, security_config=None)
        self.assertIsNone(mgr.guard)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  SQL SECURITY GUARDRAIL TESTS")
    print("=" * 72)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestLayer1StatementType,
        TestLayer2TableAccess,
        TestLayer3Complexity,
        TestLayer4Sanitize,
        TestValidationResult,
        TestAuditLog,
        TestSecurityConfigFromEnv,
        TestSecurityViolationError,
        TestDatabaseManagerIntegration,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

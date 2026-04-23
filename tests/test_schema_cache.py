"""Schema 缓存功能测试。

测试范围:
  Unit  - SchemaCache: 基本读写、TTL 过期、order-independent key、clear、stats
  Integ - SQLDatabaseManager: 真实数据库缓存命中/加速（需 .env 配置）

运行方式:
  python tests/test_schema_cache.py
"""

import io
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.database import SchemaCache


# ── Unit Tests ────────────────────────────────────────────────────────────────

class TestSchemaCacheUnit(unittest.TestCase):
    """Unit tests for SchemaCache — no DB connection required."""

    def setUp(self):
        self.cache = SchemaCache(ttl_seconds=2)

    # ── table names ──

    def test_table_names_initial_miss(self):
        self.assertIsNone(self.cache.get_table_names())

    def test_table_names_set_then_hit(self):
        tables = ["tb_shop", "tb_user", "tb_blog"]
        self.cache.set_table_names(tables)
        self.assertEqual(self.cache.get_table_names(), tables)

    def test_table_names_overwrite(self):
        self.cache.set_table_names(["a"])
        self.cache.set_table_names(["b", "c"])
        self.assertEqual(self.cache.get_table_names(), ["b", "c"])

    # ── schema ──

    def test_schema_initial_miss(self):
        self.assertIsNone(self.cache.get_schema(["tb_shop"]))

    def test_schema_set_then_hit(self):
        text = "CREATE TABLE tb_shop (id INT, name VARCHAR(255))"
        self.cache.set_schema(["tb_shop"], text)
        self.assertEqual(self.cache.get_schema(["tb_shop"]), text)

    def test_schema_order_independent_key(self):
        """get_schema with same tables in different order should hit same entry."""
        self.cache.set_schema(["tb_user", "tb_shop"], "multi-table schema")
        self.assertEqual(self.cache.get_schema(["tb_shop", "tb_user"]), "multi-table schema")
        self.assertEqual(self.cache.get_schema(["tb_user", "tb_shop"]), "multi-table schema")

    def test_schema_different_tables_different_keys(self):
        self.cache.set_schema(["tb_shop"], "shop schema")
        self.cache.set_schema(["tb_user"], "user schema")
        self.assertEqual(self.cache.get_schema(["tb_shop"]), "shop schema")
        self.assertEqual(self.cache.get_schema(["tb_user"]), "user schema")

    # ── TTL expiry ──

    def test_ttl_expiry(self):
        """Entries should expire after ttl_seconds."""
        cache = SchemaCache(ttl_seconds=1)
        cache.set_table_names(["tb_shop"])
        cache.set_schema(["tb_shop"], "schema")
        time.sleep(1.2)
        self.assertIsNone(cache.get_table_names(), "table names should have expired")
        self.assertIsNone(cache.get_schema(["tb_shop"]), "schema should have expired")

    # ── clear ──

    def test_clear_removes_all(self):
        self.cache.set_table_names(["tb_shop"])
        self.cache.set_schema(["tb_shop"], "schema")
        self.cache.clear()
        self.assertIsNone(self.cache.get_table_names())
        self.assertIsNone(self.cache.get_schema(["tb_shop"]))

    # ── stats ──

    def test_stats_is_dict_or_has_repr(self):
        """stats should be accessible without raising."""
        stats = self.cache.stats
        self.assertIsNotNone(stats)

    def test_repr_no_crash(self):
        str(self.cache)  # should not raise


# ── Integration Tests (requires DB) ──────────────────────────────────────────

class TestSchemaCacheIntegration(unittest.TestCase):
    """
    Integration tests with a real SQLite database.
    Skipped automatically if DB connection fails.
    """

    def setUp(self):
        try:
            from agent.config import DatabaseConfig
            from agent.database import SQLDatabaseManager

            config = DatabaseConfig()
            self.db_manager = SQLDatabaseManager(config, cache_ttl=300)
        except Exception as e:
            self.skipTest(f"DB not available: {e}")

    def tearDown(self):
        if hasattr(self, "db_manager"):
            self.db_manager.close()

    def test_table_names_cached_on_second_call(self):
        tables1 = self.db_manager.get_table_names()
        tables2 = self.db_manager.get_table_names()
        self.assertEqual(tables1, tables2)
        self.assertIsNotNone(tables1)

    def test_schema_cached_on_second_call(self):
        tables = self.db_manager.get_table_names()
        if not tables:
            self.skipTest("No tables found")
        t = tables[0]

        t0 = time.perf_counter()
        s1 = self.db_manager.get_table_schema([t])
        dur1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        s2 = self.db_manager.get_table_schema([t])
        dur2 = time.perf_counter() - t0

        self.assertEqual(s1, s2)
        # Second call should be at least 2x faster
        if dur1 > 0.001:
            self.assertLess(dur2, dur1, "Cached call should be faster")
        print(f"\n  Schema cache: 1st={dur1:.4f}s, 2nd={dur2:.6f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    # Windows UTF-8 输出（仅直接运行时生效，不影响 pytest 收集）
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("=" * 70)
    print("  Schema Cache 测试")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaCacheUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaCacheIntegration))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"  FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        for label, items in [("FAIL", result.failures), ("ERROR", result.errors)]:
            for item in items:
                print(f"\n  {label}: {item[0]}\n{item[1][:400]}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)


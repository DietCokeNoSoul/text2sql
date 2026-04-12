"""列名模糊匹配集成测试。

测试范围:
  Unit  - _extract_bad_column: 各数据库错误格式解析
  Unit  - _build_column_hint:  有匹配/无匹配/非列错误
  DB    - find_similar_columns: 对真实 SQLite 数据库的模糊匹配
  Mock  - _fix_query 注入验证: 确认 hint 出现在 Fix prompt 中

运行方式:
  python tests/test_column_fuzzy_match.py
"""

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import DatabaseConfig
from agent.database import SQLDatabaseManager
from skills.simple_query.skill import SimpleQuerySkill


# ── helpers ──────────────────────────────────────────────────────────────────

def make_skill(db_manager=None) -> SimpleQuerySkill:
    """Create a SimpleQuerySkill with mocked LLM and tool_manager."""
    llm = MagicMock()
    tool_manager = MagicMock()

    # tool_manager stubs so _build_graph() doesn't fail
    tool_manager.get_list_tables_tool.return_value = MagicMock()
    tool_manager.get_schema_tool.return_value = MagicMock()
    tool_manager.get_query_tool.return_value = MagicMock()

    if db_manager is None:
        db_manager = MagicMock()
        db_manager.get_dialect.return_value = MagicMock(value="MySQL")
        db_manager.config.max_query_results = 5

    skill = SimpleQuerySkill.__new__(SimpleQuerySkill)
    skill.llm = llm
    skill.tool_manager = tool_manager
    skill.db_manager = db_manager

    # Minimal BaseSkill attributes
    skill.name = "simple_query"
    skill.description = "test"

    # Build common nodes (needed for graph)
    from agent.nodes.common import CommonNodes
    skill.common = CommonNodes(tool_manager, llm)

    # Build graph (calls _build_graph)
    skill.graph = skill._build_graph()

    return skill


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestExtractBadColumn(unittest.TestCase):
    """Unit tests for _extract_bad_column."""

    def setUp(self):
        self.skill = make_skill()

    def test_mysql_format(self):
        err = "Unknown column 'shop_name' in 'field list'"
        self.assertEqual(self.skill._extract_bad_column(err), "shop_name")

    def test_mysql_table_qualified(self):
        err = "Unknown column 'tb_shop.shop_name' in 'field list'"
        # Should strip table prefix
        self.assertEqual(self.skill._extract_bad_column(err), "shop_name")

    def test_sqlite_format(self):
        err = "table tb_shop has no column named shop_name"
        self.assertEqual(self.skill._extract_bad_column(err), "shop_name")

    def test_postgresql_format(self):
        err = 'column "shop_name" does not exist'
        self.assertEqual(self.skill._extract_bad_column(err), "shop_name")

    def test_mssql_format(self):
        err = "Invalid column name 'shop_name'"
        self.assertEqual(self.skill._extract_bad_column(err), "shop_name")

    def test_syntax_error_returns_none(self):
        err = "Syntax error near FROM"
        self.assertIsNone(self.skill._extract_bad_column(err))

    def test_table_not_found_returns_none(self):
        err = "Table 'db.tb_xyz' doesn't exist"
        self.assertIsNone(self.skill._extract_bad_column(err))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.skill._extract_bad_column(""))


class TestBuildColumnHint(unittest.TestCase):
    """Unit tests for _build_column_hint with mocked db_manager."""

    def setUp(self):
        db_mock = MagicMock()
        db_mock.get_dialect.return_value = MagicMock(value="MySQL")
        db_mock.config.max_query_results = 5
        self.skill = make_skill(db_manager=db_mock)

    def test_non_column_error_returns_empty(self):
        hint = self.skill._build_column_hint("Syntax error near FROM")
        self.assertEqual(hint, "")

    def test_with_suggestions(self):
        self.skill.db_manager.find_similar_columns.return_value = [
            "tb_shop.name", "tb_shop.score"
        ]
        hint = self.skill._build_column_hint("Unknown column 'shop_name' in 'field list'")

        self.assertIn("shop_name", hint)
        self.assertIn("tb_shop.name", hint)
        self.assertIn("tb_shop.score", hint)
        self.assertIn("列名纠错提示", hint)
        print(f"\n  Hint (with suggestions):\n{hint}")

    def test_no_suggestions_shows_all_columns(self):
        self.skill.db_manager.find_similar_columns.return_value = []
        self.skill.db_manager.get_column_map.return_value = {
            "tb_shop": ["id", "name", "score"],
            "tb_user": ["id", "nick_name", "phone"],
        }
        hint = self.skill._build_column_hint("Unknown column 'xyz_abc' in 'field list'")

        self.assertIn("xyz_abc", hint)
        self.assertIn("tb_shop", hint)
        self.assertIn("tb_user", hint)
        print(f"\n  Hint (no suggestions):\n{hint}")

    def test_hint_contains_bad_col_name(self):
        self.skill.db_manager.find_similar_columns.return_value = ["tb_user.nick_name"]
        hint = self.skill._build_column_hint("Unknown column 'nickname' in 'field list'")
        self.assertIn("nickname", hint)
        self.assertIn("tb_user.nick_name", hint)


class TestFindSimilarColumnsOnRealDB(unittest.TestCase):
    """Integration tests against the real Chinook SQLite database."""

    CHINOOK_DB = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agent", "Chinook.db"
    )

    def setUp(self):
        if not os.path.exists(self.CHINOOK_DB):
            self.skipTest(f"Chinook.db not found at {self.CHINOOK_DB}")
        cfg = DatabaseConfig(uri=f"sqlite:///{self.CHINOOK_DB}")
        self.db = SQLDatabaseManager(cfg)

    def test_artist_name_typo(self):
        # ArtistName -> Artist.ArtistId  or  Artist.Name
        results = self.db.find_similar_columns("ArtistName")
        self.assertTrue(len(results) > 0, "Expected at least one match for 'ArtistName'")
        print(f"\n  ArtistName -> {results}")

    def test_milliseconds_typo(self):
        results = self.db.find_similar_columns("Millisec")
        self.assertTrue(
            any("Milliseconds" in r for r in results),
            f"Expected 'Milliseconds' in results, got {results}"
        )
        print(f"\n  Millisec -> {results}")

    def test_genre_name_typo(self):
        results = self.db.find_similar_columns("GenreName")
        self.assertTrue(len(results) > 0, f"Expected matches for 'GenreName', got {results}")
        print(f"\n  GenreName -> {results}")

    def test_no_match_returns_empty(self):
        results = self.db.find_similar_columns("xyz_qrst_uvw")
        self.assertEqual(results, [], f"Expected no matches, got {results}")
        print(f"\n  xyz_qrst_uvw -> {results} (expected empty)")

    def test_returns_table_qualified_names(self):
        results = self.db.find_similar_columns("TrackName")
        for r in results:
            self.assertIn(".", r, f"Result '{r}' should be in 'table.column' format")
        print(f"\n  TrackName -> {results}")


class TestFixQueryHintInjection(unittest.TestCase):
    """Verify that _fix_query injects column hint into the fix prompt sent to LLM."""

    def setUp(self):
        db_mock = MagicMock()
        db_mock.get_dialect.return_value = MagicMock(value="MySQL")
        db_mock.config.max_query_results = 5
        db_mock.find_similar_columns.return_value = ["tb_shop.name", "tb_shop.score"]
        self.skill = make_skill(db_manager=db_mock)

    def test_hint_injected_when_column_error(self):
        """_fix_query should inject column hint into the HumanMessage sent to LLM."""
        from langchain.messages import AIMessage

        # Mock LLM response (a tool call)
        mock_response = MagicMock(spec=AIMessage)
        mock_response.tool_calls = [{"id": "t1", "name": "sql_db_query", "args": {"query": "SELECT name FROM tb_shop"}}]
        mock_response.content = ""
        self.skill.llm.bind_tools.return_value.invoke.return_value = mock_response

        state = {
            "messages": [],
            "retry_count": 0,
            "last_error": "Unknown column 'shop_name' in 'field list'",
            "last_sql": "SELECT shop_name FROM tb_shop LIMIT 5",
        }

        result = self.skill._fix_query(state)

        # Find the HumanMessage that was added (fix prompt)
        from langchain.messages import HumanMessage
        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        self.assertTrue(len(human_messages) > 0, "No HumanMessage found in result")

        fix_prompt = human_messages[0].content
        print(f"\n  Fix prompt snippet:\n  {fix_prompt[:300]}")

        # Verify hint contents appear in the prompt
        self.assertIn("shop_name", fix_prompt)
        self.assertIn("tb_shop.name", fix_prompt)
        self.assertIn("列名纠错提示", fix_prompt)

    def test_no_hint_for_syntax_error(self):
        """_fix_query should NOT inject column hint for syntax errors."""
        from langchain.messages import AIMessage, HumanMessage

        mock_response = MagicMock(spec=AIMessage)
        mock_response.tool_calls = [{"id": "t1", "name": "sql_db_query", "args": {"query": "SELECT name FROM tb_shop"}}]
        mock_response.content = ""
        self.skill.llm.bind_tools.return_value.invoke.return_value = mock_response

        state = {
            "messages": [],
            "retry_count": 0,
            "last_error": "Syntax error near 'FORM'",
            "last_sql": "SELECT name FORM tb_shop",
        }

        result = self.skill._fix_query(state)

        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        fix_prompt = human_messages[0].content if human_messages else ""
        print(f"\n  Fix prompt (syntax error):\n  {fix_prompt[:200]}")

        self.assertNotIn("列名纠错提示", fix_prompt)

    def test_retry_count_incremented(self):
        """retry_count should be incremented after _fix_query."""
        from langchain.messages import AIMessage

        mock_response = MagicMock(spec=AIMessage)
        mock_response.tool_calls = [{"id": "t1", "name": "sql_db_query", "args": {"query": "SELECT x FROM y"}}]
        mock_response.content = ""
        self.skill.llm.bind_tools.return_value.invoke.return_value = mock_response

        state = {
            "messages": [],
            "retry_count": 1,
            "last_error": "Unknown column 'shop_name'",
            "last_sql": "SELECT shop_name FROM tb_shop",
        }

        result = self.skill._fix_query(state)
        self.assertEqual(result["retry_count"], 2)
        print(f"\n  retry_count: 1 -> {result['retry_count']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 70)
    print("  列名模糊匹配 集成测试")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestExtractBadColumn))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildColumnHint))
    suite.addTests(loader.loadTestsFromTestCase(TestFindSimilarColumnsOnRealDB))
    suite.addTests(loader.loadTestsFromTestCase(TestFixQueryHintInjection))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"  FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

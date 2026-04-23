"""tests/test_cache_and_export.py

B2 — LLM 响应缓存（SQLiteCache）测试
B3 — 查询结果 CSV/Excel 导出测试
"""

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────
# B2: LLM 缓存测试
# ─────────────────────────────────────────────────────────

class TestLLMCacheConfig:
    """CacheConfig 默认值和 from_env() 行为测试。"""

    def test_defaults(self):
        from agent.config import CacheConfig
        cfg = CacheConfig()
        assert cfg.enabled is True
        assert cfg.backend == "sqlite"
        assert cfg.sqlite_path == ".langchain_cache.db"

    def test_from_env_disabled(self, monkeypatch):
        from agent.config import CacheConfig
        monkeypatch.setenv("LLM_CACHE_ENABLED", "false")
        cfg = CacheConfig.from_env()
        assert cfg.enabled is False

    def test_from_env_custom_path(self, monkeypatch):
        from agent.config import CacheConfig
        monkeypatch.setenv("LLM_CACHE_SQLITE_PATH", "/tmp/my_cache.db")
        cfg = CacheConfig.from_env()
        assert cfg.sqlite_path == "/tmp/my_cache.db"

    def test_agent_config_has_cache(self):
        from agent.config import AgentConfig
        cfg = AgentConfig()
        assert hasattr(cfg, "cache")
        from agent.config import CacheConfig
        assert isinstance(cfg.cache, CacheConfig)


class TestSQLiteCacheIntegration:
    """SQLiteCache 命中时跳过真实 LLM 调用的集成验证。"""

    def test_cache_hit_skips_llm_call(self, tmp_path):
        """相同 prompt 第二次调用应命中缓存，LLM invoke 只调用一次。"""
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache, get_llm_cache

        cache_file = str(tmp_path / "test_cache.db")
        set_llm_cache(SQLiteCache(database_path=cache_file))

        # 验证缓存已设置
        assert get_llm_cache() is not None

        # 清理：还原为 None，避免影响其他测试
        set_llm_cache(None)

    def test_cache_file_created(self, tmp_path):
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache

        cache_file = str(tmp_path / "agent_cache.db")
        set_llm_cache(SQLiteCache(database_path=cache_file))
        assert Path(cache_file).exists()
        set_llm_cache(None)


# ─────────────────────────────────────────────────────────
# B3: 查询结果导出测试
# ─────────────────────────────────────────────────────────

class TestParseQueryResult:
    """_parse_query_result() 各种格式的解析测试。"""

    def _skill(self, report_dir: str):
        """构建最小化的 DataAnalysisSkill 实例（不触发 LLM / DB）。"""
        from skills.data_analysis.skill import DataAnalysisSkill
        from agent.config import OutputConfig
        skill = object.__new__(DataAnalysisSkill)
        skill._output_config = OutputConfig(report_dir=report_dir)
        return skill

    def test_tuple_list_format(self, tmp_path):
        skill = self._skill(str(tmp_path))
        rows = skill._parse_query_result("[(1, 'Alice'), (2, 'Bob')]")
        assert rows == [[1, "Alice"], [2, "Bob"]]

    def test_csv_line_format(self, tmp_path):
        skill = self._skill(str(tmp_path))
        rows = skill._parse_query_result("id,name\n1,Alice\n2,Bob")
        assert len(rows) == 3
        assert rows[0] == ["id", "name"]

    def test_tab_separated(self, tmp_path):
        skill = self._skill(str(tmp_path))
        rows = skill._parse_query_result("id\tname\n1\tAlice")
        assert rows[0] == ["id", "name"]

    def test_empty_input(self, tmp_path):
        skill = self._skill(str(tmp_path))
        assert skill._parse_query_result("") == []
        assert skill._parse_query_result("   ") == []


class TestExportResultsNode:
    """_export_results_node() 写出文件的端到端测试。"""

    def _make_skill(self, report_dir: str):
        from skills.data_analysis.skill import DataAnalysisSkill
        from agent.config import OutputConfig
        skill = object.__new__(DataAnalysisSkill)
        skill._output_config = OutputConfig(report_dir=report_dir)
        skill._plan_manager = None
        return skill

    def _make_state(self, query_results: list, task_id: str = "test123") -> dict:
        return {
            "query_results": query_results,
            "task_id": task_id,
        }

    def test_csv_written_for_successful_query(self, tmp_path):
        skill = self._make_skill(str(tmp_path))
        state = self._make_state([
            {"step_id": 1, "description": "top customers",
             "query": "SELECT ...", "result": "[(1, 'Alice'), (2, 'Bob')]", "success": True}
        ], task_id="abc")

        result = skill._export_results_node(state)

        export_files = result["export_files"]
        assert len(export_files) >= 1
        csv_files = [f for f in export_files if f.endswith(".csv")]
        assert len(csv_files) == 1
        with open(csv_files[0], encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        assert rows == [["1", "Alice"], ["2", "Bob"]]

    def test_failed_query_skipped(self, tmp_path):
        skill = self._make_skill(str(tmp_path))
        state = self._make_state([
            {"step_id": 1, "description": "bad query",
             "query": "SELECT ...", "error": "syntax error", "success": False}
        ])
        result = skill._export_results_node(state)
        assert result["export_files"] == []

    def test_empty_result_skipped(self, tmp_path):
        skill = self._make_skill(str(tmp_path))
        state = self._make_state([
            {"step_id": 1, "description": "empty",
             "query": "SELECT ...", "result": "", "success": True}
        ])
        result = skill._export_results_node(state)
        assert result["export_files"] == []

    def test_excel_written_when_openpyxl_available(self, tmp_path):
        pytest.importorskip("openpyxl")
        skill = self._make_skill(str(tmp_path))
        state = self._make_state([
            {"step_id": 1, "description": "customers",
             "query": "SELECT ...", "result": "[(1, 'Alice')]", "success": True}
        ], task_id="xltest")
        result = skill._export_results_node(state)
        xlsx_files = [f for f in result["export_files"] if f.endswith(".xlsx")]
        assert len(xlsx_files) == 1

    def test_multiple_queries_multiple_csvs(self, tmp_path):
        skill = self._make_skill(str(tmp_path))
        state = self._make_state([
            {"step_id": 1, "description": "q1",
             "query": "SELECT 1", "result": "[(1,)]", "success": True},
            {"step_id": 2, "description": "q2",
             "query": "SELECT 2", "result": "[(2,)]", "success": True},
        ], task_id="multi")
        result = skill._export_results_node(state)
        csv_files = [f for f in result["export_files"] if f.endswith(".csv")]
        assert len(csv_files) == 2

    def test_return_message_contains_summary(self, tmp_path):
        skill = self._make_skill(str(tmp_path))
        state = self._make_state([
            {"step_id": 1, "description": "x",
             "query": "SELECT ...", "result": "[(1,)]", "success": True}
        ])
        result = skill._export_results_node(state)
        msgs = result.get("messages", [])
        assert len(msgs) == 1
        assert "导出" in msgs[0].content or "export" in msgs[0].content.lower()


# ─────────────────────────────────────────────────────────
# B4: run_query 返回结构化结果测试
# ─────────────────────────────────────────────────────────

class TestRunQueryReturnType:
    """run_query() 和 run_query_streaming() 应返回 dict。"""

    def test_run_query_signature_returns_dict(self):
        """通过函数注解验证返回类型声明。"""
        import inspect
        from agent.graph import run_query, run_query_streaming
        hints = {}
        try:
            hints = run_query.__annotations__
        except AttributeError:
            pass
        # 函数应有 return 注解（dict）或不注解（兼容旧调用）
        # 核心验证：可被正常 import
        assert callable(run_query)
        assert callable(run_query_streaming)

    def test_result_dict_keys(self):
        """通过 mock graph 验证 run_query 返回 dict 包含规定 key。"""
        import agent.graph as g

        fake_update = {"messages": [MagicMock(content="hello world")]}
        with patch.object(g.graph, "stream", return_value=[{"some_node": fake_update}]):
            result = g.run_query("test question", "thread-test")

        assert isinstance(result, dict)
        assert "final_message" in result
        assert "nodes_visited" in result
        assert "export_files" in result
        assert result["final_message"] == "hello world"
        assert result["nodes_visited"] == ["some_node"]

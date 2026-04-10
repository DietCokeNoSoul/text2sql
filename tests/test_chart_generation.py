"""图表生成集成测试。

测试范围:
  Unit  - ChartSpec / ChartGenerator.render: 三种图表类型生成
  Unit  - _parse_query_result: SQL 结果字符串解析
  Unit  - from_query_result: 端到端从 SQL 结果生成图表文件
  Mock  - DataAnalysisSkill._visualize: 验证 _visualize 节点调用 ChartGenerator
          并将 chart_files 写入 state
  File  - 报告保存路径: 验证 report/ 目录和 chart/ 目录均正确创建

运行方式 (无需 LLM API Key):
  python tests/test_chart_generation.py
"""

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.data_analysis.chart_generator import (
    ChartGenerator,
    ChartSpec,
    _parse_query_result,
    _bar_chart,
    _pie_chart,
    _line_chart,
)


# ── Unit: _parse_query_result ─────────────────────────────────────────────────

class TestParseQueryResult(unittest.TestCase):
    """Tests for SQL result string → (labels, values) parsing."""

    def test_two_column_int_keys(self):
        labels, values = _parse_query_result("[(1, 9), (2, 5)]")
        self.assertEqual(labels, ["1", "2"])
        self.assertEqual(values, [9.0, 5.0])

    def test_two_column_str_keys(self):
        labels, values = _parse_query_result("[('TypeA', 4.5), ('TypeB', 3.8)]")
        self.assertEqual(labels, ["TypeA", "TypeB"])
        self.assertAlmostEqual(values[0], 4.5)

    def test_single_column(self):
        labels, values = _parse_query_result("[(14,)]")
        self.assertEqual(len(labels), 1)
        self.assertEqual(values, [14.0])

    def test_empty_result(self):
        labels, values = _parse_query_result("[]")
        self.assertEqual(labels, [])
        self.assertEqual(values, [])

    def test_unparseable_returns_empty(self):
        labels, values = _parse_query_result("no data here")
        self.assertEqual(labels, [])
        self.assertEqual(values, [])

    def test_multiple_rows(self):
        raw = "[(1, 9), (2, 5), (3, 3), (4, 1)]"
        labels, values = _parse_query_result(raw)
        self.assertEqual(len(labels), 4)
        self.assertEqual(len(values), 4)
        self.assertEqual(values[0], 9.0)


# ── Unit: SVG renderers ───────────────────────────────────────────────────────

class TestSVGRenderers(unittest.TestCase):
    """Tests that each renderer produces valid PNG bytes."""

    BASE_SPEC = dict(
        title="Test Chart",
        labels=["A", "B", "C"],
        values=[10.0, 20.0, 15.0],
        x_label="X",
        y_label="Y",
        width=700,
        height=420,
    )

    def _make_spec(self, chart_type: str) -> ChartSpec:
        return ChartSpec(chart_type=chart_type, **self.BASE_SPEC)

    def _assert_valid_png(self, data: bytes, chart_type: str):
        self.assertIsInstance(data, bytes, f"{chart_type}: should return bytes")
        self.assertGreater(len(data), 100, f"{chart_type}: PNG too small")
        # PNG magic bytes: \x89PNG
        self.assertTrue(data[:4] == b'\x89PNG', f"{chart_type}: not a PNG file")

    def test_bar_chart_produces_png(self):
        data = _bar_chart(self._make_spec("bar"))
        self._assert_valid_png(data, "bar")

    def test_pie_chart_produces_png(self):
        data = _pie_chart(self._make_spec("pie"))
        self._assert_valid_png(data, "pie")

    def test_line_chart_produces_png(self):
        data = _line_chart(self._make_spec("line"))
        self._assert_valid_png(data, "line")

    def test_labels_reflected_in_non_empty_output(self):
        data = _bar_chart(self._make_spec("bar"))
        self.assertGreater(len(data), 1000, "bar chart should produce substantial PNG")

    def test_unknown_type_falls_back_to_bar(self):
        spec = ChartSpec(chart_type="histogram", **self.BASE_SPEC)
        path = ChartGenerator.render(spec, tempfile.mkdtemp())
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))


# ── Unit: ChartGenerator.render ───────────────────────────────────────────────

class TestChartGeneratorRender(unittest.TestCase):
    """Tests for ChartGenerator.render() file output."""

    def setUp(self):
        self.outdir = tempfile.mkdtemp(prefix="chart_test_")

    def test_bar_chart_file_created(self):
        spec = ChartSpec("bar", "商店分布", ["A", "B"], [9.0, 5.0])
        path = ChartGenerator.render(spec, self.outdir)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path), f"File not found: {path}")
        self.assertGreater(os.path.getsize(path), 100)
        print(f"\n  Bar chart: {os.path.basename(path)} ({os.path.getsize(path)} bytes)")

    def test_pie_chart_file_created(self):
        spec = ChartSpec("pie", "占比分析", ["Type1", "Type2"], [9.0, 5.0])
        path = ChartGenerator.render(spec, self.outdir)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        print(f"\n  Pie chart: {os.path.basename(path)} ({os.path.getsize(path)} bytes)")

    def test_line_chart_file_created(self):
        spec = ChartSpec("line", "月度趋势", ["Jan", "Feb", "Mar"], [100.0, 150.0, 130.0])
        path = ChartGenerator.render(spec, self.outdir)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        print(f"\n  Line chart: {os.path.basename(path)} ({os.path.getsize(path)} bytes)")

    def test_empty_data_returns_none(self):
        spec = ChartSpec("bar", "Empty", [], [])
        path = ChartGenerator.render(spec, self.outdir)
        self.assertIsNone(path)

    def test_output_dir_created_automatically(self):
        new_dir = os.path.join(self.outdir, "subdir", "charts")
        self.assertFalse(os.path.exists(new_dir))
        spec = ChartSpec("bar", "Test", ["X"], [1.0])
        ChartGenerator.render(spec, new_dir)
        self.assertTrue(os.path.exists(new_dir))

    def test_png_content_valid(self):
        spec = ChartSpec("bar", "Valid PNG", ["A", "B"], [10.0, 20.0])
        path = ChartGenerator.render(spec, self.outdir)
        self.assertTrue(path.endswith(".png"))
        with open(path, "rb") as f:
            header = f.read(4)
        self.assertEqual(header, b'\x89PNG', "File should be a valid PNG")


# ── Unit: from_query_result ────────────────────────────────────────────────────

class TestFromQueryResult(unittest.TestCase):
    """End-to-end tests: SQL result string → chart file."""

    def setUp(self):
        self.outdir = tempfile.mkdtemp(prefix="chart_e2e_")

    def test_two_column_result(self):
        path = ChartGenerator.from_query_result(
            "[(1, 9), (2, 5)]", "bar", "类型分布",
            x_label="类型", y_label="数量", output_dir=self.outdir
        )
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        print(f"\n  from_query_result (bar): {os.path.basename(path)}")

    def test_single_value_result(self):
        path = ChartGenerator.from_query_result(
            "[(14,)]", "bar", "总数", output_dir=self.outdir
        )
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        print(f"\n  from_query_result (single): {os.path.basename(path)}")

    def test_string_labels(self):
        path = ChartGenerator.from_query_result(
            "[('TypeA', 4.5), ('TypeB', 3.8), ('TypeC', 4.2)]",
            "pie", "类型评分", output_dir=self.outdir
        )
        self.assertIsNotNone(path)
        print(f"\n  from_query_result (pie): {os.path.basename(path)}")

    def test_unparseable_returns_none(self):
        path = ChartGenerator.from_query_result(
            "no tabular data", "bar", "Empty", output_dir=self.outdir
        )
        self.assertIsNone(path)

    def test_line_chart_multiple_points(self):
        path = ChartGenerator.from_query_result(
            "[('Jan', 100), ('Feb', 150), ('Mar', 130), ('Apr', 200)]",
            "line", "月度趋势", x_label="月份", y_label="数量", output_dir=self.outdir
        )
        self.assertIsNotNone(path)
        print(f"\n  from_query_result (line): {os.path.basename(path)}")


# ── Integration: _visualize node with mocked LLM ──────────────────────────────

class TestVisualizeNode(unittest.TestCase):
    """
    Integration test for DataAnalysisSkill._visualize.
    Uses mocked LLM — no API key required.
    Verifies that chart_files are populated and files actually exist.
    """

    def _make_analysis_skill(self, outdir: str):
        """Build a DataAnalysisSkill with mocked LLM and config pointing to outdir."""
        from skills.data_analysis.skill import DataAnalysisSkill
        from agent.config import AgentConfig, OutputConfig

        llm = MagicMock()
        tool_manager = MagicMock()
        db_manager = MagicMock()
        db_manager.get_dialect.return_value = MagicMock(value="MySQL")
        db_manager.config.max_query_results = 5
        db_manager.config.uri = "sqlite:///fake.db"

        # Build minimal config with chart_dir pointing to our temp dir
        cfg = AgentConfig.__new__(AgentConfig)
        cfg.output = OutputConfig(report_dir=outdir, chart_dir=outdir)

        # Create skill without calling __init__ super (avoids graph build)
        skill = DataAnalysisSkill.__new__(DataAnalysisSkill)
        skill.llm = llm
        skill.tool_manager = tool_manager
        skill.db_manager = db_manager
        skill._output_config = cfg.output
        skill.name = "data_analysis"

        return skill

    def test_visualize_generates_chart_files(self):
        outdir = tempfile.mkdtemp(prefix="viz_test_")
        skill = self._make_analysis_skill(outdir)

        # Mock LLM to return a valid chart recommendation
        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"chart_type": "bar", "x_axis": "类型ID", "y_axis": "商店数", "title": "商店类型分布", "message": "Type 1 dominates"}'
        skill.llm.invoke.return_value = mock_llm_response

        state = {
            "query_results": [
                {
                    "step_id": 1,
                    "description": "Count shops by type",
                    "result": "[(1, 9), (2, 5)]",
                    "success": True,
                }
            ],
            "insights": [],
        }

        result = skill._visualize(state)

        # chart_files should be populated
        chart_files = result.get("chart_files", [])
        self.assertEqual(len(chart_files), 1, f"Expected 1 chart file, got {chart_files}")

        # File should actually exist
        self.assertTrue(os.path.exists(chart_files[0]), f"Chart file not found: {chart_files[0]}")
        self.assertGreater(os.path.getsize(chart_files[0]), 100)

        print(f"\n  Chart file generated: {chart_files[0]}")
        print(f"  File size: {os.path.getsize(chart_files[0])} bytes")

    def test_visualize_skips_failed_queries(self):
        outdir = tempfile.mkdtemp(prefix="viz_skip_")
        skill = self._make_analysis_skill(outdir)

        state = {
            "query_results": [
                {"step_id": 1, "description": "Failed query", "error": "table not found", "success": False},
            ],
            "insights": [],
        }

        result = skill._visualize(state)
        chart_files = result.get("chart_files", [])
        self.assertEqual(chart_files, [], "Failed queries should not generate charts")
        print(f"\n  Skipped failed query: chart_files={chart_files}")

    def test_visualize_multiple_queries(self):
        outdir = tempfile.mkdtemp(prefix="viz_multi_")
        skill = self._make_analysis_skill(outdir)

        responses = [
            '{"chart_type": "bar", "x_axis": "X", "y_axis": "Y", "title": "Chart 1", "message": ""}',
            '{"chart_type": "pie", "x_axis": "", "y_axis": "", "title": "Chart 2", "message": ""}',
        ]
        call_count = [0]

        def side_effect(*args, **kwargs):
            m = MagicMock()
            m.content = responses[call_count[0] % len(responses)]
            call_count[0] += 1
            return m

        skill.llm.invoke.side_effect = side_effect

        state = {
            "query_results": [
                {"step_id": 1, "description": "Step 1", "result": "[(1, 9), (2, 5)]", "success": True},
                {"step_id": 2, "description": "Step 2", "result": "[(1, 9), (2, 5)]", "success": True},
            ],
            "insights": [],
        }

        result = skill._visualize(state)
        chart_files = result.get("chart_files", [])
        self.assertEqual(len(chart_files), 2, f"Expected 2 charts, got {chart_files}")
        for p in chart_files:
            self.assertTrue(os.path.exists(p))
        print(f"\n  Generated {len(chart_files)} charts from 2 queries")

    def test_visualize_unparseable_result_no_file(self):
        """Unparseable SQL result should produce no chart file."""
        outdir = tempfile.mkdtemp(prefix="viz_unparse_")
        skill = self._make_analysis_skill(outdir)

        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"chart_type": "bar", "x_axis": "", "y_axis": "", "title": "Test", "message": ""}'
        skill.llm.invoke.return_value = mock_llm_response

        state = {
            "query_results": [
                {"step_id": 1, "description": "Text result", "result": "No rows found.", "success": True},
            ],
            "insights": [],
        }

        result = skill._visualize(state)
        chart_files = result.get("chart_files", [])
        self.assertEqual(chart_files, [], "Unparseable result should not generate chart")
        print(f"\n  Unparseable result: chart_files={chart_files} (expected empty)")


# ── Integration: report output directory ──────────────────────────────────────

class TestReportSaving(unittest.TestCase):
    """Tests for _save_report writing to configured directory."""

    def _make_skill_with_dir(self, report_dir: str):
        from skills.data_analysis.skill import DataAnalysisSkill
        from agent.config import OutputConfig

        skill = DataAnalysisSkill.__new__(DataAnalysisSkill)
        skill._output_config = OutputConfig(report_dir=report_dir, chart_dir=report_dir)
        return skill

    def test_report_saved_to_custom_dir(self):
        outdir = tempfile.mkdtemp(prefix="report_test_")
        skill = self._make_skill_with_dir(outdir)

        path = skill._save_report("# Test Report\n\nContent here.")
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.startswith(outdir))

        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("# Test Report", content)
        print(f"\n  Report saved: {path}")
        print(f"  Content preview: {content[:60]}")

    def test_report_filename_has_timestamp(self):
        outdir = tempfile.mkdtemp(prefix="report_ts_")
        skill = self._make_skill_with_dir(outdir)
        path = skill._save_report("test")
        basename = os.path.basename(path)
        self.assertTrue(basename.startswith("analysis_"), f"Expected 'analysis_' prefix, got {basename}")
        self.assertTrue(basename.endswith(".md"), f"Expected .md extension, got {basename}")
        print(f"\n  Report filename: {basename}")

    def test_report_dir_created_if_missing(self):
        base = tempfile.mkdtemp(prefix="report_mkdir_")
        new_dir = os.path.join(base, "reports", "2026")
        self.assertFalse(os.path.exists(new_dir))
        skill = self._make_skill_with_dir(new_dir)
        path = skill._save_report("content")
        self.assertTrue(os.path.exists(new_dir))
        self.assertIsNotNone(path)

    def test_default_report_dir_resolves_to_project_root(self):
        """Default 'report' resolves under project root, not cwd."""
        from skills.data_analysis.skill import DataAnalysisSkill
        from agent.config import OutputConfig

        skill = DataAnalysisSkill.__new__(DataAnalysisSkill)
        skill._output_config = OutputConfig(report_dir="report", chart_dir="report/charts")

        path = skill._save_report("# Default dir test")
        self.assertIsNotNone(path)

        # Should be under the project root, not a random temp dir
        # tests/ is one level below project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        expected_report_dir = os.path.join(project_root, "report")
        self.assertTrue(
            path.startswith(expected_report_dir),
            f"Expected path under {expected_report_dir}, got {path}"
        )
        # Clean up
        if os.path.exists(path):
            os.remove(path)
        print(f"\n  Default report dir: {expected_report_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 70)
    print("  图表生成集成测试")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestParseQueryResult))
    suite.addTests(loader.loadTestsFromTestCase(TestSVGRenderers))
    suite.addTests(loader.loadTestsFromTestCase(TestChartGeneratorRender))
    suite.addTests(loader.loadTestsFromTestCase(TestFromQueryResult))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizeNode))
    suite.addTests(loader.loadTestsFromTestCase(TestReportSaving))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"  FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        for f in result.failures:
            print(f"\n  FAIL: {f[0]}\n{f[1]}")
        for e in result.errors:
            print(f"\n  ERROR: {e[0]}\n{e[1]}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

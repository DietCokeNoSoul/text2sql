"""报告保存集成测试。

测试范围:
  Unit  - _save_report: 文件创建、时间戳格式、内容正确性、目录自动创建
  Unit  - 相对路径解析: 默认 'report' 路径锚定到项目根目录
  Unit  - Unicode 内容: 中文/特殊字符正确写入
  Unit  - 多次调用: 文件名唯一（时间戳不同）
  Unit  - 权限失败: 异常时返回 None 而非崩溃
  Integ - _generate_report 节点: mock LLM，验证报告写入磁盘
  Integ - 图表文件路径: 报告内容包含图表引用
  Integ - REPORT_DIR 环境变量: OutputConfig 读取 env var

运行方式 (无需 LLM API Key):
  python tests/test_report_saving.py
"""

import io
import os
import re
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import AgentConfig, OutputConfig
from skills.data_analysis.skill import DataAnalysisSkill


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_skill(report_dir: str, chart_dir: str = None) -> DataAnalysisSkill:
    """Create a DataAnalysisSkill stub with only _output_config set."""
    skill = DataAnalysisSkill.__new__(DataAnalysisSkill)
    skill._output_config = OutputConfig(
        report_dir=report_dir,
        chart_dir=chart_dir or report_dir,
    )
    skill.llm = MagicMock()
    skill.tool_manager = MagicMock()
    skill._plan_manager = None
    return skill


# ── Suite 1: _save_report 基础行为 ─────────────────────────────────────────────

class TestSaveReportBasic(unittest.TestCase):
    """Core _save_report behavior."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="rpt_basic_")

    # ── File creation ──

    def test_returns_path_string(self):
        skill = _make_skill(self.tmpdir)
        path = skill._save_report("# Hello")
        self.assertIsNotNone(path)
        self.assertIsInstance(path, str)

    def test_file_exists_after_save(self):
        skill = _make_skill(self.tmpdir)
        path = skill._save_report("# Report\n\ncontent.")
        self.assertTrue(os.path.exists(path), f"File not found: {path}")

    def test_file_has_md_extension(self):
        skill = _make_skill(self.tmpdir)
        path = skill._save_report("content")
        self.assertTrue(path.endswith(".md"), f"Expected .md, got: {path}")

    def test_file_in_configured_dir(self):
        skill = _make_skill(self.tmpdir)
        path = skill._save_report("content")
        self.assertEqual(os.path.dirname(path), self.tmpdir)

    def test_file_not_empty(self):
        skill = _make_skill(self.tmpdir)
        path = skill._save_report("# Report\n\n## Section\n\ncontent here.")
        self.assertGreater(os.path.getsize(path), 0)

    # ── Filename format ──

    def test_filename_starts_with_analysis(self):
        skill = _make_skill(self.tmpdir)
        path = skill._save_report("x")
        self.assertTrue(os.path.basename(path).startswith("analysis_"))

    def test_filename_has_timestamp_pattern(self):
        """Filename must match analysis_YYYYMMDD_HHMMSS.md"""
        skill = _make_skill(self.tmpdir)
        path = skill._save_report("x")
        name = os.path.basename(path)
        self.assertRegex(name, r"^analysis_\d{8}_\d{6}\.md$",
                         f"Filename '{name}' does not match expected pattern")

    def test_timestamp_in_valid_range(self):
        """Timestamp in filename should be a valid recent datetime."""
        import datetime
        skill = _make_skill(self.tmpdir)
        path = skill._save_report("x")
        name = os.path.basename(path)
        ts_str = name[len("analysis_"):len("analysis_") + 15]  # YYYYMMDD_HHMMSS
        ts = datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        now = datetime.datetime.now()
        delta = abs((now - ts).total_seconds())
        self.assertLess(delta, 10, f"Timestamp {ts_str} too far from now ({delta:.1f}s)")

    # ── Content integrity ──

    def test_content_written_exactly(self):
        content = "# 分析报告\n\n## 关键发现\n- 商店总数: 14\n- 平均评分: 4.2\n"
        skill = _make_skill(self.tmpdir)
        path = skill._save_report(content)
        with open(path, encoding="utf-8") as f:
            written = f.read()
        self.assertEqual(written, content)

    def test_unicode_content_preserved(self):
        content = "# 中文标题\n\n## 分析结果\n\n数据：商店类型分布分析完成。\n特殊字符：①②③④⑤\n"
        skill = _make_skill(self.tmpdir)
        path = skill._save_report(content)
        with open(path, encoding="utf-8") as f:
            written = f.read()
        self.assertIn("中文标题", written)
        self.assertIn("①②③④⑤", written)

    def test_multiline_content_preserved(self):
        content = "# Title\n\n## Section 1\n\nLine 1\nLine 2\nLine 3\n\n## Section 2\n\nEnd.\n"
        skill = _make_skill(self.tmpdir)
        path = skill._save_report(content)
        with open(path, encoding="utf-8") as f:
            written = f.read()
        self.assertEqual(written.count("\n"), content.count("\n"))

    def test_empty_content_still_creates_file(self):
        skill = _make_skill(self.tmpdir)
        path = skill._save_report("")
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.getsize(path), 0)


# ── Suite 2: 目录处理 ──────────────────────────────────────────────────────────

class TestSaveReportDirectories(unittest.TestCase):
    """Directory creation and path resolution."""

    def test_dir_created_if_missing(self):
        base = tempfile.mkdtemp(prefix="rpt_mkdir_")
        new_dir = os.path.join(base, "nested", "reports")
        self.assertFalse(os.path.exists(new_dir))
        skill = _make_skill(new_dir)
        skill._save_report("content")
        self.assertTrue(os.path.isdir(new_dir), f"Dir not created: {new_dir}")

    def test_deeply_nested_dir_created(self):
        base = tempfile.mkdtemp(prefix="rpt_deep_")
        deep = os.path.join(base, "a", "b", "c", "d", "reports")
        skill = _make_skill(deep)
        path = skill._save_report("deep")
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

    def test_existing_dir_no_error(self):
        """Calling _save_report twice should not fail."""
        skill = _make_skill(tempfile.mkdtemp())
        skill._save_report("first")
        path2 = skill._save_report("second")
        self.assertIsNotNone(path2)

    def test_relative_path_anchors_to_project_root(self):
        """report_dir='report' should resolve to <project_root>/report."""
        skill = _make_skill("report")
        path = skill._save_report("# relative path test")
        self.assertIsNotNone(path)

        # Compute expected project root from skill.py location
        skill_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "skills", "data_analysis", "skill.py")
        )
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(skill_file)))
        expected_dir = os.path.join(project_root, "report")

        self.assertTrue(
            path.startswith(expected_dir),
            f"Expected path under {expected_dir}, got {path}"
        )
        # Cleanup to keep repo clean
        if os.path.exists(path):
            os.remove(path)

    def test_absolute_path_used_directly(self):
        """Absolute report_dir should be used as-is (no project root joining)."""
        tmpdir = tempfile.mkdtemp(prefix="rpt_abs_")
        skill = _make_skill(tmpdir)
        path = skill._save_report("abs test")
        self.assertTrue(path.startswith(tmpdir),
                         f"Absolute path not used directly: {path}")


# ── Suite 3: 多次调用唯一性 ────────────────────────────────────────────────────

class TestSaveReportUniqueness(unittest.TestCase):
    """Each call should produce a distinct file."""

    def test_two_calls_same_second_share_name(self):
        """Two calls within the same second may get the same filename."""
        tmpdir = tempfile.mkdtemp(prefix="rpt_uniq_")
        skill = _make_skill(tmpdir)
        p1 = skill._save_report("report 1")
        p2 = skill._save_report("report 2")
        # Both should exist; if they collide, second overwrites first (known limitation)
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)

    def test_two_calls_different_seconds_distinct_files(self):
        """Calls at different seconds must produce different files."""
        tmpdir = tempfile.mkdtemp(prefix="rpt_ts_")
        skill = _make_skill(tmpdir)
        p1 = skill._save_report("report 1")
        time.sleep(1.1)
        p2 = skill._save_report("report 2")
        self.assertNotEqual(p1, p2, "Files should have different timestamps")
        # Both files should exist with their own content
        with open(p1, encoding="utf-8") as f:
            self.assertIn("report 1", f.read())
        with open(p2, encoding="utf-8") as f:
            self.assertIn("report 2", f.read())


# ── Suite 4: 错误处理 ──────────────────────────────────────────────────────────

class TestSaveReportErrorHandling(unittest.TestCase):
    """_save_report should never raise; return None on failure."""

    def test_returns_none_on_write_error(self):
        skill = _make_skill(tempfile.mkdtemp())
        # Patch open to raise an IOError
        with patch("builtins.open", side_effect=IOError("disk full")):
            path = skill._save_report("content")
        self.assertIsNone(path, "Should return None on write failure")

    def test_returns_none_on_makedirs_error(self):
        skill = _make_skill("/invalid/nonexistent/path/x/y/z")
        with patch("os.makedirs", side_effect=PermissionError("access denied")):
            path = skill._save_report("content")
        self.assertIsNone(path)


# ── Suite 5: OutputConfig 环境变量 ─────────────────────────────────────────────

class TestOutputConfigEnvVar(unittest.TestCase):
    """OutputConfig reads REPORT_DIR and CHART_DIR from environment."""

    def test_report_dir_from_env(self):
        with patch.dict(os.environ, {"REPORT_DIR": "/tmp/custom_reports"}):
            cfg = AgentConfig.from_env()
        self.assertEqual(cfg.output.report_dir, "/tmp/custom_reports")

    def test_chart_dir_from_env(self):
        with patch.dict(os.environ, {"CHART_DIR": "/tmp/custom_charts"}):
            cfg = AgentConfig.from_env()
        self.assertEqual(cfg.output.chart_dir, "/tmp/custom_charts")

    def test_default_report_dir(self):
        env = {k: v for k, v in os.environ.items() if k not in ("REPORT_DIR", "CHART_DIR")}
        with patch.dict(os.environ, env, clear=True):
            cfg = AgentConfig.from_env()
        self.assertEqual(cfg.output.report_dir, "report")

    def test_default_chart_dir(self):
        env = {k: v for k, v in os.environ.items() if k not in ("REPORT_DIR", "CHART_DIR")}
        with patch.dict(os.environ, env, clear=True):
            cfg = AgentConfig.from_env()
        self.assertEqual(cfg.output.chart_dir, "report/charts")

    def test_output_config_dataclass_fields(self):
        cfg = OutputConfig(report_dir="/a", chart_dir="/b")
        self.assertEqual(cfg.report_dir, "/a")
        self.assertEqual(cfg.chart_dir, "/b")


# ── Suite 6: _generate_report 节点集成 ────────────────────────────────────────

class TestGenerateReportNode(unittest.TestCase):
    """
    Integration test for DataAnalysisSkill._generate_report.
    Verifies that the node: calls LLM, saves file to disk, returns report content.
    Uses mocked LLM — no API key required.
    """

    SAMPLE_REPORT = (
        "# 数据分析报告\n\n## 摘要\n\n共14家商店。\n\n"
        "## 关键发现\n\n- 商店总数：14\n- 最高评分：5.0\n\n## 结论\n\n数据质量良好。\n"
    )

    def _make_skill_with_outdir(self, outdir: str):
        skill = DataAnalysisSkill.__new__(DataAnalysisSkill)
        skill._output_config = OutputConfig(
            report_dir=os.path.join(outdir, "report"),
            chart_dir=os.path.join(outdir, "charts"),
        )
        skill.llm = MagicMock()
        skill.llm.invoke.return_value = MagicMock(content=self.SAMPLE_REPORT)
        skill.tool_manager = MagicMock()
        skill._plan_manager = None
        return skill

    def _make_state(self, chart_files=None):
        return {
            "analysis_goal": "分析商店数据",
            "query_results": [
                {"step_id": 1, "description": "总数", "result": "[(14,)]", "success": True},
            ],
            "insights": [{"insight": "共14家商店"}],
            "visualizations": [{"step_id": 1, "chart_type": "bar", "title": "分布", "message": "ok"}],
            "chart_files": chart_files or [],
        }

    def test_report_file_created(self):
        tmpdir = tempfile.mkdtemp(prefix="gen_rpt_")
        skill = self._make_skill_with_outdir(tmpdir)
        result = skill._generate_report(self._make_state())

        report_dir = skill._output_config.report_dir
        files = os.listdir(report_dir) if os.path.exists(report_dir) else []
        md_files = [f for f in files if f.endswith(".md")]
        self.assertEqual(len(md_files), 1, f"Expected 1 .md file, got: {files}")
        print(f"\n  Report file: {md_files[0]}")

    def test_report_content_in_state(self):
        tmpdir = tempfile.mkdtemp(prefix="gen_rpt_content_")
        skill = self._make_skill_with_outdir(tmpdir)
        result = skill._generate_report(self._make_state())

        self.assertIn("report", result)
        report_text = result["report"]
        self.assertIn("数据分析报告", report_text)

    def test_report_contains_save_path(self):
        """report content should include the saved file path."""
        tmpdir = tempfile.mkdtemp(prefix="gen_rpt_path_")
        skill = self._make_skill_with_outdir(tmpdir)
        result = skill._generate_report(self._make_state())

        report_text = result["report"]
        self.assertIn("报告已保存至", report_text, "Report should reference saved path")

    def test_report_contains_chart_refs(self):
        """When chart_files are provided, report should reference chart section heading."""
        tmpdir = tempfile.mkdtemp(prefix="gen_rpt_charts_")
        skill = self._make_skill_with_outdir(tmpdir)

        chart_path = os.path.join(tmpdir, "charts", "bar_test.svg")
        result = skill._generate_report(self._make_state(chart_files=[chart_path]))

        report_text = result["report"]
        self.assertIn("bar_test", report_text)  # chart filename appears in img tag

    def test_report_file_content_matches_llm_output(self):
        """Content written to disk should include the LLM-generated text."""
        tmpdir = tempfile.mkdtemp(prefix="gen_rpt_match_")
        skill = self._make_skill_with_outdir(tmpdir)
        skill._generate_report(self._make_state())

        report_dir = skill._output_config.report_dir
        files = [f for f in os.listdir(report_dir) if f.endswith(".md")]
        self.assertEqual(len(files), 1)
        with open(os.path.join(report_dir, files[0]), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("关键发现", content)

    def test_messages_returned(self):
        """_generate_report should return 'messages' key for graph state."""
        tmpdir = tempfile.mkdtemp(prefix="gen_rpt_msg_")
        skill = self._make_skill_with_outdir(tmpdir)
        result = skill._generate_report(self._make_state())
        self.assertIn("messages", result)
        self.assertTrue(len(result["messages"]) > 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    # Windows UTF-8 输出（仅直接运行时生效，不影响 pytest 收集）
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("=" * 70)
    print("  报告保存集成测试")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [
        TestSaveReportBasic,
        TestSaveReportDirectories,
        TestSaveReportUniqueness,
        TestSaveReportErrorHandling,
        TestOutputConfigEnvVar,
        TestGenerateReportNode,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    total = result.testsRun
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"  ALL {total} TESTS PASSED")
    else:
        print(f"  FAILED: {len(result.failures)} failures, {len(result.errors)} errors / {total} tests")
        for label, items in [("FAIL", result.failures), ("ERROR", result.errors)]:
            for item in items:
                print(f"\n  {label}: {item[0]}\n{item[1][:500]}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

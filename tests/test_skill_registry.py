"""单元测试：_extract_skill_description 和 SkillRegistry.build_router_prompt。

测试范围:
  Unit - _extract_skill_description: 正常 SKILL.md、文件缺失、无目标章节、畸形内容
  Unit - SkillRegistry.build_router_prompt: 空注册中心、单 Skill、多 Skill

运行方式:
  pytest tests/test_skill_registry.py
  python tests/test_skill_registry.py
"""

import os
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.skills.base import _extract_skill_description
from agent.skills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_md(content: str) -> str:
    """Write a temp SKILL.md and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8", delete=False
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def _make_skill(name: str, description: str) -> MagicMock:
    skill = MagicMock()
    skill.name = name
    skill.description = description
    return skill


# ---------------------------------------------------------------------------
# Tests: _extract_skill_description
# ---------------------------------------------------------------------------

class TestExtractSkillDescription(unittest.TestCase):

    def test_extracts_purpose_and_scenarios(self):
        md = textwrap.dedent("""\
            # My Skill
            ## 目的
            处理简单查询。
            ## 适用场景
            - 单表查询
            - 过滤
            ## 不适用场景
            复杂分析。
        """)
        path = _write_md(md)
        try:
            result = _extract_skill_description(path)
        finally:
            os.unlink(path)

        self.assertIn("**目的**", result)
        self.assertIn("处理简单查询", result)
        self.assertIn("**适用场景**", result)
        self.assertIn("单表查询", result)
        # 不适用场景 should NOT be included
        self.assertNotIn("不适用场景", result)

    def test_missing_file_returns_empty_string(self):
        result = _extract_skill_description("/nonexistent/path/SKILL.md")
        self.assertEqual(result, "")

    def test_md_without_target_sections_returns_empty(self):
        md = textwrap.dedent("""\
            # My Skill
            ## 流程
            step1 → step2
            ## 不适用场景
            不用于分析。
        """)
        path = _write_md(md)
        try:
            result = _extract_skill_description(path)
        finally:
            os.unlink(path)

        self.assertEqual(result, "")

    def test_only_purpose_section_present(self):
        md = textwrap.dedent("""\
            # Skill
            ## 目的
            This is the purpose.
        """)
        path = _write_md(md)
        try:
            result = _extract_skill_description(path)
        finally:
            os.unlink(path)

        self.assertIn("**目的**", result)
        self.assertIn("This is the purpose.", result)
        self.assertNotIn("**适用场景**", result)

    def test_only_scenarios_section_present(self):
        md = textwrap.dedent("""\
            # Skill
            ## 适用场景
            - case A
            - case B
        """)
        path = _write_md(md)
        try:
            result = _extract_skill_description(path)
        finally:
            os.unlink(path)

        self.assertIn("**适用场景**", result)
        self.assertIn("case A", result)

    def test_real_simple_query_skill_md(self):
        """Smoke test against the actual SKILL.md shipped with the project."""
        real_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "skills", "simple_query", "SKILL.md",
        )
        if not os.path.exists(real_path):
            self.skipTest("skills/simple_query/SKILL.md not found")
        result = _extract_skill_description(real_path)
        self.assertIn("**目的**", result)
        self.assertIn("**适用场景**", result)

    def test_empty_file_returns_empty(self):
        path = _write_md("")
        try:
            result = _extract_skill_description(path)
        finally:
            os.unlink(path)
        self.assertEqual(result, "")

    def test_section_with_only_whitespace_is_excluded(self):
        md = textwrap.dedent("""\
            # Skill
            ## 目的

            ## 适用场景
            - item
        """)
        path = _write_md(md)
        try:
            result = _extract_skill_description(path)
        finally:
            os.unlink(path)
        # 目的 has no body → should not appear
        self.assertNotIn("**目的**", result)
        self.assertIn("**适用场景**", result)


# ---------------------------------------------------------------------------
# Tests: SkillRegistry.build_router_prompt
# ---------------------------------------------------------------------------

class TestBuildRouterPrompt(unittest.TestCase):

    def test_empty_registry_returns_empty_string(self):
        registry = SkillRegistry()
        result = registry.build_router_prompt()
        self.assertEqual(result, "")

    def test_single_skill_produces_block(self):
        registry = SkillRegistry()
        registry.register(_make_skill("simple_query", "**目的**\n简单查询。"))
        result = registry.build_router_prompt()
        self.assertIn("【simple_query】", result)
        self.assertIn("简单查询", result)

    def test_multiple_skills_all_present(self):
        registry = SkillRegistry()
        registry.register(_make_skill("simple_query", "simple desc"))
        registry.register(_make_skill("complex_query", "complex desc"))
        registry.register(_make_skill("data_analysis", "analysis desc"))
        result = registry.build_router_prompt()
        self.assertIn("【simple_query】", result)
        self.assertIn("【complex_query】", result)
        self.assertIn("【data_analysis】", result)
        self.assertIn("simple desc", result)
        self.assertIn("complex desc", result)
        self.assertIn("analysis desc", result)

    def test_skill_with_empty_description_uses_fallback(self):
        registry = SkillRegistry()
        registry.register(_make_skill("my_skill", ""))
        result = registry.build_router_prompt()
        self.assertIn("【my_skill】", result)
        self.assertIn("my_skill", result)  # fallback "Skill: my_skill"

    def test_skill_with_none_description_uses_fallback(self):
        registry = SkillRegistry()
        skill = _make_skill("none_skill", None)
        skill.description = None
        registry.register(skill)
        result = registry.build_router_prompt()
        self.assertIn("【none_skill】", result)

    def test_blocks_separated_by_double_newline(self):
        registry = SkillRegistry()
        registry.register(_make_skill("skill_a", "desc A"))
        registry.register(_make_skill("skill_b", "desc B"))
        result = registry.build_router_prompt()
        self.assertIn("\n\n", result)

    def test_unregister_removes_from_prompt(self):
        registry = SkillRegistry()
        registry.register(_make_skill("to_remove", "will be gone"))
        registry.register(_make_skill("keeper", "stays"))
        registry.unregister("to_remove")
        result = registry.build_router_prompt()
        self.assertNotIn("to_remove", result)
        self.assertIn("【keeper】", result)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)

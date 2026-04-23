"""
单元测试：ComplexQuerySkill._resolve_query_placeholders

测试占位符替换逻辑的各种场景：
- 正常替换（列表元组、列表标量）
- 字符串格式的结果（ast.literal_eval 路径）
- 无结果时降级为 (NULL)
- 多依赖链（step1 → step2 → step3）
- 无占位符时 SQL 不变
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


def _make_skill() -> "ComplexQuerySkill":  # noqa: F821
    """构造一个最小化 ComplexQuerySkill，仅初始化占位符替换所需属性。"""
    from skills.complex_query.skill import ComplexQuerySkill

    skill = ComplexQuerySkill.__new__(ComplexQuerySkill)
    skill._retriever = None
    skill._plan_manager = None
    skill.llm = MagicMock()
    skill.tool_manager = MagicMock()
    skill.tool_manager.get_tools.return_value = []
    skill.db_manager = MagicMock()
    skill.name = "complex_query"
    skill.description = "test"
    return skill


class TestResolvePlaceholders:

    def setup_method(self):
        self.skill = _make_skill()

    # ── 正常替换路径 ───────────────────────────────────────────────────────

    def test_tuple_list_extracts_first_element(self):
        """结果为元组列表时应提取每个元组的第一个元素。"""
        step_results = {
            1: {"result": [(10,), (20,), (30,)], "success": True}
        }
        query = "SELECT * FROM t WHERE id IN {step_1_results}"
        out = self.skill._resolve_query_placeholders(query, [1], step_results)
        assert "(10, 20, 30)" in out
        assert "{step_1_results}" not in out

    def test_scalar_list_used_directly(self):
        """结果为标量列表时直接用作 IN 列表。"""
        step_results = {
            1: {"result": [5, 6, 7], "success": True}
        }
        query = "SELECT name FROM shop WHERE type_id IN {step_1_results}"
        out = self.skill._resolve_query_placeholders(query, [1], step_results)
        assert "(5, 6, 7)" in out

    def test_string_result_parsed_via_literal_eval(self):
        """结果是字符串格式的列表时，应通过 ast.literal_eval 解析。"""
        step_results = {
            1: {"result": "[(1,), (2,), (3,)]", "success": True}
        }
        query = "SELECT * FROM orders WHERE shop_id IN {step_1_results}"
        out = self.skill._resolve_query_placeholders(query, [1], step_results)
        assert "(1, 2, 3)" in out

    # ── 降级路径 (NULL) ────────────────────────────────────────────────────

    def test_empty_result_list_becomes_null(self):
        """空结果列表应替换为 (NULL)。"""
        step_results = {1: {"result": [], "success": True}}
        query = "SELECT * FROM t WHERE id IN {step_1_results}"
        out = self.skill._resolve_query_placeholders(query, [1], step_results)
        assert "(NULL)" in out

    def test_missing_step_result_becomes_null(self):
        """依赖步骤结果缺失时应替换为 (NULL)。"""
        query = "SELECT * FROM t WHERE id IN {step_1_results}"
        out = self.skill._resolve_query_placeholders(query, [1], {})
        assert "(NULL)" in out

    def test_unparseable_string_becomes_null(self):
        """无法 literal_eval 的字符串结果应替换为 (NULL)。"""
        step_results = {1: {"result": "some plain text result", "success": True}}
        query = "SELECT * FROM t WHERE id IN {step_1_results}"
        out = self.skill._resolve_query_placeholders(query, [1], step_results)
        assert "(NULL)" in out

    # ── 无占位符 ───────────────────────────────────────────────────────────

    def test_no_placeholder_query_unchanged(self):
        """不含占位符的 SQL 应原样返回。"""
        query = "SELECT COUNT(*) FROM users WHERE status = 'active'"
        out = self.skill._resolve_query_placeholders(query, [], {})
        assert out == query

    def test_unrelated_placeholder_not_replaced(self):
        """depends_on 未包含的步骤占位符不应被替换。"""
        step_results = {
            1: {"result": [(10,)], "success": True},
            2: {"result": [(20,)], "success": True},
        }
        query = "SELECT * FROM t WHERE id IN {step_2_results}"
        # only depends_on=[1] is passed, step_2 should NOT be resolved
        out = self.skill._resolve_query_placeholders(query, [1], step_results)
        assert "{step_2_results}" in out  # unreplaced

    # ── 多依赖链 ───────────────────────────────────────────────────────────

    def test_multi_dependency_both_replaced(self):
        """多依赖时两个占位符都应被正确替换。"""
        step_results = {
            1: {"result": [(1,), (2,)], "success": True},
            2: {"result": [(100,), (200,)], "success": True},
        }
        query = (
            "SELECT * FROM t "
            "WHERE type_id IN {step_1_results} "
            "AND shop_id IN {step_2_results}"
        )
        out = self.skill._resolve_query_placeholders(query, [1, 2], step_results)
        assert "(1, 2)" in out
        assert "(100, 200)" in out
        assert "{step_1_results}" not in out
        assert "{step_2_results}" not in out

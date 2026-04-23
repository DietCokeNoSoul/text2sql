"""
集成测试：ComplexQuerySkill._dual_tower_retrieve_node

验证双塔检索节点能正确
  1. 将 DualTowerRetriever.retrieve() 的 pruned_schema 写入 table_schema 状态
  2. 将统计指标写入 retrieval_stats 状态
  3. 在检索失败时安全降级（保留原始 full schema，写入 error 键）
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ── 辅助工厂 ──────────────────────────────────────────────────────────────────

def _make_retrieval_result(pruned_schema: str = "TABLE A(id INT)") -> MagicMock:
    """构造一个符合 RetrievalResult 接口的 mock 对象。"""
    r = MagicMock()
    r.pruned_schema = pruned_schema
    r.full_schema_chars = 5000
    r.pruned_schema_chars = len(pruned_schema)
    r.char_saved = 5000 - len(pruned_schema)
    r.estimated_token_saved = (5000 - len(pruned_schema)) // 4
    r.reduction_pct = (5000 - len(pruned_schema)) / 5000 * 100
    r.join_path_tables = ["A"]
    r.retrieval_ms = 42.0
    r.summary.return_value = "DualTower: 已裁剪 80%"
    return r


def _make_skill(retriever: Any = None) -> "ComplexQuerySkill":  # noqa: F821
    """构造一个 ComplexQuerySkill 实例（LLM / tool_manager / db_manager 全部 mock）."""
    from skills.complex_query.skill import ComplexQuerySkill

    llm = MagicMock()
    tool_manager = MagicMock()
    tool_manager.get_tools.return_value = []
    db_manager = MagicMock()

    skill = ComplexQuerySkill.__new__(ComplexQuerySkill)
    # 直接初始化必要属性，绕过 _build_graph（避免实际 DB 连接）
    skill._retriever = retriever
    skill._plan_manager = None
    skill.db_manager = db_manager
    skill.llm = llm
    skill.tool_manager = tool_manager
    skill.name = "complex_query"
    skill.description = "测试"
    return skill


# ── 测试用例 ──────────────────────────────────────────────────────────────────

class TestDualTowerRetrieveNode:
    """_dual_tower_retrieve_node 节点的集成测试。"""

    def test_pruned_schema_replaces_table_schema(self):
        """成功检索时，pruned_schema 应覆盖 state['table_schema']。"""
        pruned = "TABLE Sales(id INT, amount DECIMAL)"
        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(pruned)

        skill = _make_skill(retriever=retriever)
        from langchain_core.messages import HumanMessage

        state: Dict[str, Any] = {
            "messages": [HumanMessage(content="每月的销售总额是多少？")],
            "table_schema": "TABLE A(...) TABLE B(...) TABLE C(...)",
        }

        result = skill._dual_tower_retrieve_node(state)

        assert result["table_schema"] == pruned
        retriever.retrieve.assert_called_once_with("每月的销售总额是多少？")

    def test_retrieval_stats_populated(self):
        """检索成功时，retrieval_stats 应包含预期的统计字段。"""
        pruned = "TABLE X(id INT)"
        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result(pruned)

        skill = _make_skill(retriever=retriever)
        from langchain_core.messages import HumanMessage

        state: Dict[str, Any] = {
            "messages": [HumanMessage(content="查询一下")],
            "table_schema": "FULL SCHEMA TEXT " * 100,
        }

        result = skill._dual_tower_retrieve_node(state)
        stats = result["retrieval_stats"]

        for key in ("full_chars", "pruned_chars", "saved_chars",
                    "saved_tokens_est", "reduction_pct",
                    "join_path_tables", "retrieval_ms"):
            assert key in stats, f"缺少统计字段: {key}"

        assert stats["pruned_chars"] == len(pruned)
        assert stats["reduction_pct"] >= 0

    def test_graceful_fallback_on_retriever_error(self):
        """检索抛异常时应降级使用原始 full schema，并在 retrieval_stats 写入 error 键。"""
        retriever = MagicMock()
        retriever.retrieve.side_effect = RuntimeError("Milvus连接失败")

        skill = _make_skill(retriever=retriever)
        from langchain_core.messages import HumanMessage

        full_schema = "TABLE Original(id INT)"
        state: Dict[str, Any] = {
            "messages": [HumanMessage(content="查一下")],
            "table_schema": full_schema,
        }

        result = skill._dual_tower_retrieve_node(state)

        # schema 不应被修改
        assert result["table_schema"] == full_schema
        # error 信息应记录在 retrieval_stats
        assert "error" in result["retrieval_stats"]
        assert "Milvus连接失败" in result["retrieval_stats"]["error"]

    def test_empty_messages_does_not_crash(self):
        """messages 为空列表时节点不应崩溃。"""
        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_result("TABLE Z(id INT)")

        skill = _make_skill(retriever=retriever)
        state: Dict[str, Any] = {
            "messages": [],
            "table_schema": "FULL",
        }

        result = skill._dual_tower_retrieve_node(state)
        # 空 question 时 retrieve 应以空字符串调用
        retriever.retrieve.assert_called_once_with("")
        assert "table_schema" in result

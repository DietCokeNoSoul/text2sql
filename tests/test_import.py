"""Smoke tests — verify key symbols can be imported without error.

运行方式:
  pytest tests/test_import.py
  python tests/test_import.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoreAgentImports(unittest.TestCase):
    """Smoke tests for agent core symbols."""

    def test_skill_graph_builder(self):
        from agent.skill_graph_builder import SkillBasedGraphBuilder
        self.assertTrue(callable(SkillBasedGraphBuilder))

    def test_base_skill(self):
        from agent.skills.base import BaseSkill
        self.assertTrue(callable(BaseSkill))

    def test_extract_skill_description(self):
        from agent.skills.base import _extract_skill_description
        self.assertTrue(callable(_extract_skill_description))

    def test_skill_registry(self):
        from agent.skills.registry import SkillRegistry
        self.assertTrue(callable(SkillRegistry))

    def test_simple_query_skill(self):
        from skills.simple_query.skill import SimpleQuerySkill
        self.assertTrue(callable(SimpleQuerySkill))

    def test_complex_query_skill(self):
        from skills.complex_query.skill import ComplexQuerySkill
        self.assertTrue(callable(ComplexQuerySkill))

    def test_data_analysis_skill(self):
        from skills.data_analysis.skill import DataAnalysisSkill
        self.assertTrue(callable(DataAnalysisSkill))

    def test_agent_config(self):
        from agent.config import AgentConfig, DatabaseConfig, OutputConfig
        self.assertTrue(callable(AgentConfig))
        self.assertTrue(callable(DatabaseConfig))
        self.assertTrue(callable(OutputConfig))

    def test_database_manager(self):
        from agent.database import SQLDatabaseManager, SchemaCache
        self.assertTrue(callable(SQLDatabaseManager))
        self.assertTrue(callable(SchemaCache))

    def test_sql_security_guard(self):
        from agent.database import SQLSecurityGuard
        self.assertTrue(callable(SQLSecurityGuard))

    def test_states(self):
        from agent.skills.states import MainGraphState
        self.assertTrue(callable(MainGraphState))


if __name__ == "__main__":
    unittest.main(verbosity=2)

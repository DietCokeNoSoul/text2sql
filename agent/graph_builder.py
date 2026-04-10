"""SQL Agent 图构建器（向后兼容层）。

新项目请直接使用 skill_graph_builder.py。
此文件作为兼容层保留，将调用转发到 Skill-based 实现。
"""

from .skill_graph_builder import create_skill_based_graph as create_sql_agent_graph, SkillBasedGraphBuilder as SQLAgentGraphBuilder

__all__ = ["create_sql_agent_graph", "SQLAgentGraphBuilder"]


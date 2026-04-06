"""
Complex Query Skill - Plan-Execute pattern with parallel execution

Handles multi-step queries that require breaking down into sub-queries.
Uses Send API to execute sub-queries in parallel.
"""

from .skill import ComplexQuerySkill

__all__ = ["ComplexQuerySkill"]

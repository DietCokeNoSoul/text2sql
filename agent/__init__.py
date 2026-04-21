"""SQL Agent with Skill-based Architecture

This module provides both legacy and skill-based graph implementations.

**Recommended**: Use skill-based architecture for new projects
**Legacy**: Old graph available for backward compatibility
"""

# Skill-based graph builder (recommended)
from .skill_graph_builder import (
    create_skill_based_graph,
    SkillBasedGraphBuilder
)

# Core components
from .config import get_config, AgentConfig, SecurityConfig
from .database import SQLDatabaseManager
from .tools import SQLToolManager
from .security import SQLSecurityGuard, ValidationResult
from .types import SecurityViolationError

# Legacy graph (backward compatibility) - lazy loading to avoid initialization on import
_legacy_graph = None

def _get_legacy_graph():
    """Lazy loading for legacy graph to avoid slow initialization on import."""
    global _legacy_graph
    if _legacy_graph is None:
        from .graph import graph
        _legacy_graph = graph
    return _legacy_graph

# Property-like access for backward compatibility
class _LegacyGraphProxy:
    """Proxy to provide lazy loading while maintaining attribute access."""
    def __getattr__(self, name):
        graph = _get_legacy_graph()
        return getattr(graph, name)
    
    def __call__(self, *args, **kwargs):
        graph = _get_legacy_graph()
        return graph(*args, **kwargs)

legacy_graph = _LegacyGraphProxy()

__all__ = [
    # Legacy
    "legacy_graph",
    
    # Skill-based (recommended)
    "create_skill_based_graph",
    "SkillBasedGraphBuilder",
    
    # Core components
    "get_config",
    "AgentConfig",
    "SecurityConfig",
    "SQLDatabaseManager",
    "SQLToolManager",
    
    # Security
    "SQLSecurityGuard",
    "ValidationResult",
    "SecurityViolationError",
]

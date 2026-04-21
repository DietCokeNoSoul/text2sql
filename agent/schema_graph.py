"""Schema Graph Builder using NetworkX + Steiner Tree.

构建数据库 Schema 的图结构，用于跨表 JOIN 路径规划：
- 节点 = 数据库表
- 边   = 外键关系 或 命名规则推断的关联（如 shop_id → tb_shop.id）
- Steiner Tree 求最小连通子树，给出候选表之间的最短 JOIN 路径
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class JoinEdge:
    """表示两张表之间的 JOIN 关系。"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    weight: float = 1.0          # 外键 = 1.0；命名推断 = 1.5（置信度稍低）
    source: str = "foreign_key"  # "foreign_key" | "name_pattern"

    def __repr__(self) -> str:
        return (f"{self.from_table}.{self.from_column} "
                f"--[{self.source}]--> {self.to_table}.{self.to_column}")


@dataclass
class JoinPath:
    """Steiner Tree 规划出的 JOIN 路径。"""
    tables: List[str]            # 涉及的表（有序）
    edges: List[JoinEdge]        # 涉及的 JOIN 条件
    sql_joins: List[str]         # 可直接插入 SQL 的 JOIN 子句
    total_weight: float = 0.0

    @property
    def join_hint(self) -> str:
        """返回给 LLM 的 JOIN 提示字符串。"""
        if not self.sql_joins:
            return ""
        lines = ["-- Recommended JOIN path (auto-planned):"]
        lines += [f"--   {j}" for j in self.sql_joins]
        return "\n".join(lines)


class SchemaGraph:
    """数据库 Schema 图，支持 Steiner Tree 路径规划。"""

    def __init__(self) -> None:
        try:
            import networkx as nx
            self._nx = nx
        except ImportError:
            raise ImportError("networkx is required: pip install networkx")

        self._graph = self._nx.Graph()
        self._edges: List[JoinEdge] = []
        self._column_map: Dict[str, List[str]] = {}  # table -> [col, ...]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_from_db(self, db_manager) -> "SchemaGraph":
        """从 SQLDatabaseManager 自动构建 Schema 图。

        步骤：
        1. 通过 SQLAlchemy inspector 获取显式外键
        2. 通过命名规则推断隐式外键（如 shop_id → tb_shop.id）
        """
        from sqlalchemy import inspect as sa_inspect

        self._graph.clear()
        self._edges.clear()
        self._column_map.clear()

        inspector = sa_inspect(db_manager.db._engine)
        tables = db_manager.get_table_names()

        # Add all tables as nodes
        for table in tables:
            self._graph.add_node(table)
            try:
                self._column_map[table] = [
                    col["name"] for col in inspector.get_columns(table)
                ]
            except Exception:
                self._column_map[table] = []

        # Layer 1: explicit foreign keys
        fk_count = 0
        for table in tables:
            try:
                fks = inspector.get_foreign_keys(table)
                for fk in fks:
                    ref_table = fk.get("referred_table")
                    if not ref_table or ref_table not in tables:
                        continue
                    from_cols = fk.get("constrained_columns", [])
                    to_cols = fk.get("referred_columns", [])
                    if not from_cols or not to_cols:
                        continue
                    edge = JoinEdge(
                        from_table=table,
                        from_column=from_cols[0],
                        to_table=ref_table,
                        to_column=to_cols[0],
                        weight=1.0,
                        source="foreign_key",
                    )
                    self._add_edge(edge)
                    fk_count += 1
            except Exception as e:
                logger.debug(f"[SchemaGraph] FK scan failed for {table}: {e}")

        # Layer 2: name-pattern inference (e.g. shop_id -> tb_shop.id)
        infer_count = self._infer_edges_by_naming(tables)

        logger.info(
            f"[SchemaGraph] Built graph: {len(tables)} nodes, "
            f"{fk_count} FK edges, {infer_count} inferred edges"
        )
        return self

    def _infer_edges_by_naming(self, tables: List[str]) -> int:
        """通过列名模式推断隐式外键。

        规则：
        - 若列名为 `{X}_id`，且存在名为 `tb_{X}` 或 `{X}` 的表，则推断关联
        - 权重 1.5（低于显式外键的 1.0）
        """
        count = 0
        table_set = set(tables)
        # build normalized lookup: "shop" -> "tb_shop", "shop_type" -> "tb_shop_type"
        norm_map: Dict[str, str] = {}
        for t in tables:
            # strip common prefixes: tb_, t_
            norm = re.sub(r'^(tb_|t_)', '', t)
            norm_map[norm] = t
            norm_map[t] = t  # exact match too

        for table, cols in self._column_map.items():
            for col in cols:
                m = re.match(r'^(.+)_id$', col)
                if not m:
                    continue
                ref_key = m.group(1)  # e.g. "shop", "shop_type"
                ref_table = norm_map.get(ref_key) or norm_map.get(f"tb_{ref_key}")
                if ref_table and ref_table != table:
                    # Check if reverse edge already exists (avoid duplicates)
                    if not self._graph.has_edge(table, ref_table):
                        # Verify ref_table has an 'id' column
                        ref_cols = self._column_map.get(ref_table, [])
                        to_col = "id" if "id" in ref_cols else (ref_cols[0] if ref_cols else "id")
                        edge = JoinEdge(
                            from_table=table,
                            from_column=col,
                            to_table=ref_table,
                            to_column=to_col,
                            weight=1.5,
                            source="name_pattern",
                        )
                        self._add_edge(edge)
                        count += 1
        return count

    def _add_edge(self, edge: JoinEdge) -> None:
        """Add edge to graph, keep lower-weight edge on duplicates."""
        existing = self._graph.get_edge_data(edge.from_table, edge.to_table)
        if existing is None or existing.get("weight", 99) > edge.weight:
            self._graph.add_edge(
                edge.from_table, edge.to_table,
                weight=edge.weight,
                join_edge=edge,
            )
            # Remove old entry if updating
            self._edges = [
                e for e in self._edges
                if not (e.from_table == edge.from_table and e.to_table == edge.to_table)
                and not (e.from_table == edge.to_table and e.to_table == edge.from_table)
            ]
            self._edges.append(edge)

    # ------------------------------------------------------------------
    # Steiner Tree path planning
    # ------------------------------------------------------------------

    def plan_join_path(self, required_tables: List[str]) -> Optional[JoinPath]:
        """对给定的候选表集合，用 Steiner Tree 算法规划最小连通 JOIN 路径。

        Args:
            required_tables: 向量检索命中的表名列表

        Returns:
            JoinPath（含 JOIN 子句）或 None（无法连通时）
        """
        # Filter to tables that exist in graph
        valid = [t for t in required_tables if self._graph.has_node(t)]
        if not valid:
            return None
        if len(valid) == 1:
            return JoinPath(tables=valid, edges=[], sql_joins=[], total_weight=0.0)

        try:
            # networkx Steiner tree approximation (Kou 2-approximation)
            tree = self._nx.algorithms.approximation.steiner_tree(
                self._graph, valid, weight="weight"
            )
        except Exception as e:
            logger.warning(f"[SchemaGraph] Steiner Tree failed: {e}, falling back to shortest path")
            tree = self._fallback_path(valid)
            if tree is None:
                return None

        # Extract edges from tree
        path_tables = list(tree.nodes())
        path_edges: List[JoinEdge] = []
        sql_joins: List[str] = []
        total_weight = 0.0

        for u, v, data in tree.edges(data=True):
            join_edge: Optional[JoinEdge] = data.get("join_edge")
            if join_edge is None:
                # Reconstruct from raw edge data
                join_edge = JoinEdge(from_table=u, from_column="id",
                                     to_table=v, to_column="id", weight=data.get("weight", 1.0))
            path_edges.append(join_edge)
            total_weight += join_edge.weight

            # Build SQL JOIN clause
            join_sql = (
                f"JOIN {join_edge.to_table} ON "
                f"{join_edge.from_table}.{join_edge.from_column} = "
                f"{join_edge.to_table}.{join_edge.to_column}"
            )
            sql_joins.append(join_sql)

        logger.info(
            f"[SchemaGraph] Steiner Tree: {len(path_tables)} tables, "
            f"{len(path_edges)} joins, weight={total_weight:.1f}"
        )

        return JoinPath(
            tables=path_tables,
            edges=path_edges,
            sql_joins=sql_joins,
            total_weight=total_weight,
        )

    def _fallback_path(self, tables: List[str]):
        """Fallback: shortest path tree between all required tables."""
        try:
            import networkx as nx
            subgraph_nodes: Set[str] = set()
            edges_data = {}

            for i, src in enumerate(tables):
                for dst in tables[i + 1:]:
                    try:
                        path = nx.shortest_path(self._graph, src, dst, weight="weight")
                        for j in range(len(path) - 1):
                            u, v = path[j], path[j + 1]
                            subgraph_nodes.add(u)
                            subgraph_nodes.add(v)
                            key = (min(u, v), max(u, v))
                            if key not in edges_data:
                                edges_data[key] = self._graph.get_edge_data(u, v, {})
                    except nx.NetworkXNoPath:
                        pass

            if not subgraph_nodes:
                return None

            sub = nx.Graph()
            sub.add_nodes_from(subgraph_nodes)
            for (u, v), data in edges_data.items():
                sub.add_edge(u, v, **data)
            return sub
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Schema pruning
    # ------------------------------------------------------------------

    def get_pruned_schema(self, db_manager, join_path: JoinPath) -> str:
        """只返回 JOIN 路径涉及表的 schema，而非全量 schema。"""
        if not join_path.tables:
            return db_manager.get_table_schema()
        return db_manager.get_table_schema(join_path.tables)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def describe(self) -> str:
        lines = [f"SchemaGraph: {self.node_count} tables, {self.edge_count} edges"]
        for u, v, data in self._graph.edges(data=True):
            je: Optional[JoinEdge] = data.get("join_edge")
            if je:
                lines.append(f"  {je}")
            else:
                lines.append(f"  {u} -- {v}")
        return "\n".join(lines)

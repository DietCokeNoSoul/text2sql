"""Dual-Tower Retrieval Coordinator.

双塔检索架构协调器：
- Tower 1: Milvus 列向量语义检索  → 定位相关列/表
- Tower 2: NetworkX Steiner Tree  → 规划跨表 JOIN 路径
- 输出: 剪枝后的 Schema 子集 + JOIN 提示 + Token 节省统计

用于在 ComplexQuerySkill 的 plan 节点之前自动剪枝 schema，
将 LLM 接收的 schema 从全量缩减为最小相关子集。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """双塔检索的完整输出。"""

    # 检索到的相关表
    relevant_tables: List[str] = field(default_factory=list)

    # Steiner Tree 规划的 JOIN 路径
    join_path_tables: List[str] = field(default_factory=list)
    join_hint: str = ""            # 给 LLM 的 JOIN 提示（SQL 注释格式）

    # 剪枝后的 Schema 字符串
    pruned_schema: str = ""
    full_schema: str = ""          # 原始全量 schema（用于对比）

    # Token 统计
    full_schema_chars: int = 0
    pruned_schema_chars: int = 0

    # 性能
    retrieval_ms: float = 0.0

    @property
    def char_saved(self) -> int:
        return max(0, self.full_schema_chars - self.pruned_schema_chars)

    @property
    def reduction_pct(self) -> float:
        if self.full_schema_chars == 0:
            return 0.0
        return self.char_saved / self.full_schema_chars * 100

    @property
    def estimated_token_saved(self) -> int:
        """粗估节省的 token 数（1 token ≈ 3.5 字符）。"""
        return int(self.char_saved / 3.5)

    def summary(self) -> str:
        return (
            f"[Retrieval] tables={len(self.join_path_tables)}/{self._total_tables()} | "
            f"schema {self.full_schema_chars}→{self.pruned_schema_chars} chars | "
            f"saved {self.char_saved} chars (~{self.estimated_token_saved} tokens, "
            f"{self.reduction_pct:.0f}%) | {self.retrieval_ms:.0f}ms"
        )

    def _total_tables(self) -> int:
        # rough estimate from full schema
        return self.full_schema.count("CREATE TABLE")


class DualTowerRetriever:
    """双塔检索协调器。

    Usage:
        retriever = DualTowerRetriever(db_manager)
        retriever.build_index()                        # 首次或 schema 变更时调用
        result = retriever.retrieve("查询每种类型店铺的平均评分")
        # result.pruned_schema  → 注入 plan prompt
        # result.join_hint      → 拼接到 schema 末尾
    """

    def __init__(
        self,
        db_manager,
        milvus_host: str = "127.0.0.1",
        milvus_port: int = 19530,
        top_k_columns: int = 15,
        max_candidate_tables: int = 6,
        score_threshold: float = 0.25,
        fallback_to_full_schema: bool = True,
    ) -> None:
        self._db_manager = db_manager
        self._top_k = top_k_columns
        self._max_tables = max_candidate_tables
        self._threshold = score_threshold
        self._fallback = fallback_to_full_schema

        # Lazy-initialized towers
        self._column_index: Optional[object] = None
        self._schema_graph: Optional[object] = None
        self._index_built: bool = False
        self._milvus_host = milvus_host
        self._milvus_port = milvus_port

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_index(self, force_rebuild: bool = False) -> int:
        """构建/更新 Milvus 列向量索引和 Schema 图。

        Returns:
            索引的列总数
        """
        from .column_index import ColumnIndex
        from .schema_graph import SchemaGraph

        t0 = time.time()

        # Tower 2: Schema Graph (fast, always rebuild)
        self._schema_graph = SchemaGraph()
        self._schema_graph.build_from_db(self._db_manager)
        logger.info(f"[DualTower] Schema graph: {self._schema_graph.describe().splitlines()[0]}")

        # Tower 1: Column Index (Milvus)
        self._column_index = ColumnIndex(
            milvus_host=self._milvus_host,
            milvus_port=self._milvus_port,
        )
        count = self._column_index.build_from_db(self._db_manager, force_rebuild=force_rebuild)

        elapsed = (time.time() - t0) * 1000
        self._index_built = True
        logger.info(f"[DualTower] Index ready: {count} cols, {elapsed:.0f}ms")
        return count

    def _ensure_index(self) -> None:
        if not self._index_built:
            logger.info("[DualTower] Auto-building index on first retrieve...")
            self.build_index()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """执行双塔检索，返回剪枝后的 schema 和 JOIN 提示。

        Args:
            query: 用户自然语言问题

        Returns:
            RetrievalResult（含剪枝 schema、JOIN 提示、token 统计）
        """
        self._ensure_index()
        t0 = time.time()

        result = RetrievalResult()

        # Get full schema for baseline measurement
        full_schema = self._db_manager.get_table_schema()
        result.full_schema = full_schema
        result.full_schema_chars = len(full_schema)

        try:
            # === Tower 1: Milvus semantic column search ===
            relevant_tables = self._column_index.get_relevant_tables(
                query,
                top_k=self._top_k,
                max_tables=self._max_tables,
                score_threshold=self._threshold,
            )
            result.relevant_tables = relevant_tables
            logger.info(f"[DualTower T1] Milvus hit tables: {relevant_tables}")

            if not relevant_tables:
                raise ValueError("Milvus returned no relevant tables")

            # === Tower 2: Steiner Tree JOIN path ===
            join_path = self._schema_graph.plan_join_path(relevant_tables)

            if join_path and join_path.tables:
                result.join_path_tables = join_path.tables
                result.join_hint = join_path.join_hint
                logger.info(f"[DualTower T2] Steiner Tree: {join_path.tables}")

                # Pruned schema = only Steiner Tree tables
                pruned = self._schema_graph.get_pruned_schema(
                    self._db_manager, join_path
                )
                result.pruned_schema = pruned
                if result.join_hint:
                    result.pruned_schema += f"\n\n{result.join_hint}"
            else:
                # JOIN path failed → fall back to Milvus tables only
                pruned = self._db_manager.get_table_schema(relevant_tables)
                result.pruned_schema = pruned
                result.join_path_tables = relevant_tables

        except Exception as e:
            logger.warning(f"[DualTower] Retrieval failed: {e}")
            if self._fallback:
                logger.info("[DualTower] Falling back to full schema")
                result.pruned_schema = full_schema
                result.join_path_tables = self._db_manager.get_table_names()
            else:
                raise

        result.pruned_schema_chars = len(result.pruned_schema)
        result.retrieval_ms = (time.time() - t0) * 1000

        logger.info(result.summary())
        return result

    def close(self) -> None:
        if self._column_index is not None:
            try:
                self._column_index.close()
            except Exception:
                pass

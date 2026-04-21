"""Milvus Column Vector Index for semantic column retrieval.

将数据库每列的特征（表名.列名 + 数据类型 + 样本值）编码为向量，
通过语义搜索定位与用户问题最相关的列和表。

连接地址: 127.0.0.1:19530 (默认 Milvus standalone)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

COLLECTION_NAME = "text2sql_columns"
EMBEDDING_DIM = 384          # paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class ColumnRecord:
    """一列的完整特征记录。"""
    table: str
    column: str
    data_type: str
    sample_values: List[str]
    text: str = ""           # 拼接后用于编码的文本

    def __post_init__(self) -> None:
        samples_str = ", ".join(self.sample_values[:5]) if self.sample_values else ""
        self.text = (
            f"table: {self.table}  "
            f"column: {self.column}  "
            f"type: {self.data_type}  "
            f"samples: {samples_str}"
        )

    @property
    def id(self) -> int:
        """Deterministic integer ID from table+column hash."""
        h = hashlib.md5(f"{self.table}.{self.column}".encode()).hexdigest()
        return int(h[:15], 16)


@dataclass
class ColumnSearchResult:
    """向量检索结果。"""
    table: str
    column: str
    score: float             # cosine similarity (0~1, higher=more relevant)
    data_type: str = ""


class ColumnIndex:
    """Milvus 列向量索引，支持语义检索与自动构建。"""

    def __init__(
        self,
        milvus_host: str = "127.0.0.1",
        milvus_port: int = 19530,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self._host = milvus_host
        self._port = milvus_port
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._encoder = None
        self._collection = None
        self._db_fingerprint: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            import os
            logger.info(f"[ColumnIndex] Loading embedding model: {self._embedding_model_name}")
            # Use local cache only — avoids network calls after first download
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            try:
                self._encoder = SentenceTransformer(
                    self._embedding_model_name, local_files_only=True
                )
            except Exception:
                # Model not cached yet — fall back to online download (first-time setup)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                os.environ.pop("HF_HUB_OFFLINE", None)
                logger.info("[ColumnIndex] Model not cached, downloading from HuggingFace...")
                self._encoder = SentenceTransformer(self._embedding_model_name)
            logger.info("[ColumnIndex] Embedding model loaded")
        return self._encoder

    def _connect_milvus(self):
        from pymilvus import connections, Collection, utility
        connections.connect("default", host=self._host, port=str(self._port))
        logger.info(f"[ColumnIndex] Connected to Milvus {self._host}:{self._port}")
        return Collection, utility

    def _get_or_create_collection(self):
        if self._collection is not None:
            return self._collection

        from pymilvus import (
            CollectionSchema, FieldSchema, DataType,
            Collection, utility, connections
        )
        connections.connect("default", host=self._host, port=str(self._port))

        if utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
            self._collection.load()
            logger.info(f"[ColumnIndex] Loaded existing collection '{self._collection_name}'")
        else:
            fields = [
                FieldSchema(name="id",         dtype=DataType.INT64,        is_primary=True, auto_id=False),
                FieldSchema(name="table_name", dtype=DataType.VARCHAR,      max_length=128),
                FieldSchema(name="col_name",   dtype=DataType.VARCHAR,      max_length=128),
                FieldSchema(name="data_type",  dtype=DataType.VARCHAR,      max_length=64),
                FieldSchema(name="text",       dtype=DataType.VARCHAR,      max_length=512),
                FieldSchema(name="embedding",  dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            ]
            schema = CollectionSchema(fields, description="Text2SQL column index")
            self._collection = Collection(self._collection_name, schema)

            # Create HNSW index for fast ANN search
            self._collection.create_index(
                "embedding",
                {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
            )
            self._collection.load()
            logger.info(f"[ColumnIndex] Created collection '{self._collection_name}'")

        return self._collection

    # ------------------------------------------------------------------
    # Build index from DB
    # ------------------------------------------------------------------

    def build_from_db(self, db_manager, force_rebuild: bool = False) -> int:
        """从数据库构建列向量索引，如果 schema 未变化则跳过。

        Args:
            db_manager: SQLDatabaseManager 实例
            force_rebuild: 强制重建索引

        Returns:
            索引的列总数
        """
        col_records = self._extract_column_records(db_manager)
        fingerprint = self._compute_fingerprint(col_records)

        collection = self._get_or_create_collection()

        # Skip rebuild if schema unchanged
        if not force_rebuild and fingerprint == self._db_fingerprint:
            count = collection.num_entities
            logger.info(f"[ColumnIndex] Schema unchanged, skipping rebuild ({count} cols)")
            return count

        # Clear and rebuild
        if collection.num_entities > 0:
            logger.info("[ColumnIndex] Dropping old collection for rebuild...")
            from pymilvus import utility
            utility.drop_collection(self._collection_name)
            self._collection = None
            # Recreate fresh collection
            collection = self._get_or_create_collection()

        # Encode all column texts
        encoder = self._get_encoder()
        texts = [r.text for r in col_records]
        logger.info(f"[ColumnIndex] Encoding {len(texts)} columns...")
        embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=False).tolist()

        # Insert into Milvus
        data = [
            [r.id for r in col_records],
            [r.table for r in col_records],
            [r.column for r in col_records],
            [r.data_type for r in col_records],
            [r.text[:512] for r in col_records],
            embeddings,
        ]
        collection.insert(data)
        collection.flush()

        self._db_fingerprint = fingerprint
        count = collection.num_entities
        logger.info(f"[ColumnIndex] Index built: {count} columns indexed")
        return count

    def _extract_column_records(self, db_manager) -> List[ColumnRecord]:
        """从数据库提取列特征记录，包含样本值。"""
        from sqlalchemy import inspect as sa_inspect, text as sa_text

        inspector = sa_inspect(db_manager.db._engine)
        records: List[ColumnRecord] = []

        for table in db_manager.get_table_names():
            try:
                cols = inspector.get_columns(table)
            except Exception:
                continue

            for col in cols:
                col_name = col["name"]
                data_type = str(col.get("type", "")).split("(")[0].upper()

                # Fetch sample values (up to 3 non-null distinct values)
                samples: List[str] = []
                try:
                    with db_manager.db._engine.connect() as conn:
                        sql = sa_text(
                            f"SELECT DISTINCT `{col_name}` FROM `{table}` "
                            f"WHERE `{col_name}` IS NOT NULL LIMIT 3"
                        )
                        rows = conn.execute(sql).fetchall()
                        samples = [str(r[0]) for r in rows]
                except Exception:
                    pass

                records.append(ColumnRecord(
                    table=table,
                    column=col_name,
                    data_type=data_type,
                    sample_values=samples,
                ))

        return records

    def _compute_fingerprint(self, records: List[ColumnRecord]) -> str:
        """Compute a fingerprint from table+column names to detect schema changes."""
        key = "|".join(sorted(f"{r.table}.{r.column}" for r in records))
        return hashlib.md5(key.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.25,
    ) -> List[ColumnSearchResult]:
        """语义搜索与 query 最相关的列。

        Args:
            query: 用户自然语言问题
            top_k: 返回的最大列数
            score_threshold: 最低相似度阈值（cosine，0~1）

        Returns:
            按相似度降序排列的 ColumnSearchResult 列表
        """
        collection = self._get_or_create_collection()
        if collection.num_entities == 0:
            logger.warning("[ColumnIndex] Collection is empty, returning no results")
            return []

        encoder = self._get_encoder()
        query_vec = encoder.encode([query], show_progress_bar=False).tolist()

        results = collection.search(
            data=query_vec,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["table_name", "col_name", "data_type"],
        )

        hits: List[ColumnSearchResult] = []
        for hit in results[0]:
            score = float(hit.score)
            if score < score_threshold:
                continue
            hits.append(ColumnSearchResult(
                table=hit.entity.get("table_name", ""),
                column=hit.entity.get("col_name", ""),
                score=score,
                data_type=hit.entity.get("data_type", ""),
            ))

        logger.info(
            f"[ColumnIndex] Search '{query[:50]}...' -> "
            f"{len(hits)} hits (threshold={score_threshold})"
        )
        return hits

    def get_relevant_tables(
        self,
        query: str,
        top_k: int = 10,
        max_tables: int = 6,
        score_threshold: float = 0.25,
    ) -> List[str]:
        """返回与 query 最相关的表名列表（去重，按最高列分降序）。"""
        hits = self.search(query, top_k=top_k, score_threshold=score_threshold)

        table_scores: Dict[str, float] = {}
        for hit in hits:
            # Keep max score per table
            prev = table_scores.get(hit.table, 0.0)
            if hit.score > prev:
                table_scores[hit.table] = hit.score

        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_tables[:max_tables]]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def drop_collection(self) -> None:
        """删除 Milvus 集合（用于完全重建）。"""
        from pymilvus import utility, connections
        connections.connect("default", host=self._host, port=str(self._port))
        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)
            self._collection = None
            self._db_fingerprint = None
            logger.info(f"[ColumnIndex] Dropped collection '{self._collection_name}'")

    def close(self) -> None:
        from pymilvus import connections
        try:
            connections.disconnect("default")
        except Exception:
            pass

    @property
    def indexed_count(self) -> int:
        """返回已索引的列数。"""
        try:
            col = self._get_or_create_collection()
            return col.num_entities
        except Exception:
            return 0

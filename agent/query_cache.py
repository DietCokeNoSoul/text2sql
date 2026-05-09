"""语义查询缓存 (Semantic Query Cache)

将历史查询的问题+答案持久化到 SQLite，新问题通过余弦相似度匹配历史问题。
若相似度超过阈值，直接返回缓存答案，不调用 LLM。

设计原则：
- 复用已有 sentence-transformers 模型（和 column_index.py 共享同一实例）
- 纯 Python + SQLite + numpy，无额外依赖
- 相似度阈值默认 0.92（可通过 env 变量 QUERY_CACHE_THRESHOLD 调节）
- 缓存条目默认保留 7 天（QUERY_CACHE_MAX_AGE_DAYS）
- 写操作失败时只记录警告，不中断主流程
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── 默认配置 ──────────────────────────────────────────────────────────────────
_DEFAULT_DB_PATH    = ".query_cache.db"
_DEFAULT_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_THRESHOLD  = 0.92   # 余弦相似度阈值
_DEFAULT_MAX_AGE    = 7      # 缓存最大保留天数

_DDL = """
CREATE TABLE IF NOT EXISTS query_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT    NOT NULL,
    answer_json TEXT    NOT NULL,
    embedding   BLOB    NOT NULL,   -- float32 numpy array, little-endian
    created_at  REAL    NOT NULL    -- Unix timestamp
);
"""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两向量余弦相似度，处理零向量情况。"""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


class SemanticQueryCache:
    """语义查询缓存，利用句向量余弦相似度匹配历史问题。

    使用示例：
        cache = SemanticQueryCache()
        hit = cache.lookup("每个专辑有多少首歌？")
        if hit:
            return hit          # dict，和 run_query 返回格式一致
        result = run_full_pipeline(question)
        cache.store("每个专辑有多少首歌？", result)
        return result
    """

    def __init__(
        self,
        db_path: str   = _DEFAULT_DB_PATH,
        model_name: str = _DEFAULT_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
        max_age_days: int = _DEFAULT_MAX_AGE,
    ) -> None:
        self.threshold   = threshold
        self.max_age_sec = max_age_days * 86400
        self._model_name = model_name
        self._encoder    = None   # 懒加载

        # 打开 SQLite（允许多线程共享同一连接）
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_DDL)
        self._conn.commit()
        logger.info(
            f"[QueryCache] Ready — db={db_path}, threshold={threshold}, "
            f"max_age={max_age_days}d"
        )

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def lookup(self, question: str) -> Optional[dict]:
        """查找语义相似的历史问题，命中则返回缓存答案，否则返回 None。"""
        try:
            q_vec = self._embed(question)
            rows  = self._load_valid_rows()
            if not rows:
                return None

            best_score = -1.0
            best_answer = None
            best_question = ""

            for (db_question, answer_json, emb_blob) in rows:
                db_vec = np.frombuffer(emb_blob, dtype=np.float32)
                score  = _cosine(q_vec, db_vec)
                if score > best_score:
                    best_score    = score
                    best_answer   = answer_json
                    best_question = db_question

            if best_score >= self.threshold:
                logger.info(
                    f"[QueryCache] HIT (score={best_score:.4f}) "
                    f"matched: '{best_question[:60]}'"
                )
                result = json.loads(best_answer)
                result["_cache_hit"]  = True
                result["_cache_score"] = round(best_score, 4)
                result["_cache_question"] = best_question
                return result

            logger.debug(f"[QueryCache] MISS (best_score={best_score:.4f})")
            return None

        except Exception as e:
            logger.warning(f"[QueryCache] lookup error (skipping cache): {e}")
            return None

    def store(self, question: str, result: dict) -> None:
        """将问题+答案写入缓存。写失败时只记录警告，不抛异常。"""
        try:
            # 不缓存失败的结果
            if not result.get("final_message"):
                return

            q_vec = self._embed(question)
            emb_blob = q_vec.astype(np.float32).tobytes()
            answer_json = json.dumps(result, ensure_ascii=False)

            self._conn.execute(
                "INSERT INTO query_cache (question, answer_json, embedding, created_at) "
                "VALUES (?, ?, ?, ?)",
                (question, answer_json, emb_blob, time.time()),
            )
            self._conn.commit()
            logger.debug(f"[QueryCache] Stored: '{question[:60]}'")

        except Exception as e:
            logger.warning(f"[QueryCache] store error: {e}")

    def clear(self) -> int:
        """手动清空全部缓存，返回删除条数。"""
        cur = self._conn.execute("DELETE FROM query_cache")
        self._conn.commit()
        n = cur.rowcount
        logger.info(f"[QueryCache] Cleared {n} entries")
        return n

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        cur = self._conn.execute(
            "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM query_cache"
        )
        total, oldest, newest = cur.fetchone()
        return {
            "total_entries": total or 0,
            "threshold":     self.threshold,
            "max_age_days":  self.max_age_sec // 86400,
            "oldest_entry":  oldest,
            "newest_entry":  newest,
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """将文本编码为 float32 向量，复用已缓存的 encoder 实例。"""
        if self._encoder is None:
            # 复用 column_index 的模型缓存，避免重复加载
            from agent.column_index import _get_cached_encoder
            self._encoder = _get_cached_encoder(self._model_name)
        vec = self._encoder.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def _load_valid_rows(self) -> list:
        """加载未过期的缓存行（顺便清理过期条目）。"""
        cutoff = time.time() - self.max_age_sec
        # 清理过期
        self._conn.execute("DELETE FROM query_cache WHERE created_at < ?", (cutoff,))
        self._conn.commit()
        # 加载有效行
        cur = self._conn.execute(
            "SELECT question, answer_json, embedding FROM query_cache "
            "WHERE created_at >= ?",
            (cutoff,),
        )
        return cur.fetchall()


# ── 模块级单例 ────────────────────────────────────────────────────────────────

_cache_instance: Optional[SemanticQueryCache] = None


def get_query_cache() -> Optional[SemanticQueryCache]:
    """返回模块级 SemanticQueryCache 单例（由 graph.py 初始化后调用）。"""
    return _cache_instance


def init_query_cache(
    db_path: str   = _DEFAULT_DB_PATH,
    model_name: str = _DEFAULT_MODEL,
    threshold: float | None = None,
    max_age_days: int | None = None,
) -> SemanticQueryCache:
    """初始化（或重置）模块级缓存单例。在 graph.py 启动时调用一次。"""
    global _cache_instance
    _threshold   = threshold   if threshold   is not None else float(os.getenv("QUERY_CACHE_THRESHOLD",  str(_DEFAULT_THRESHOLD)))
    _max_age     = max_age_days if max_age_days is not None else int(os.getenv("QUERY_CACHE_MAX_AGE_DAYS", str(_DEFAULT_MAX_AGE)))
    _cache_instance = SemanticQueryCache(
        db_path=db_path,
        model_name=model_name,
        threshold=_threshold,
        max_age_days=_max_age,
    )
    return _cache_instance

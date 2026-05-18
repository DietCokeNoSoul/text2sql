"""
统计数据库模块 — text2sql_stats (MySQL)

使用 pymysql + asyncio run_in_executor 实现异步写入，
避免阻塞 FastAPI event loop。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import pymysql
import pymysql.cursors

logger = logging.getLogger(__name__)

_DB_HOST = "127.0.0.1"
_DB_PORT = 3306
_DB_USER = "root"
_DB_PASSWORD = "1234"
_DB_NAME = "text2sql_stats"


def _get_conn() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=_DB_HOST,
        port=_DB_PORT,
        user=_DB_USER,
        password=_DB_PASSWORD,
        database=_DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


# ── 初始化 ────────────────────────────────────────────────────────────────────

def _sync_init_stats_db() -> None:
    """建库建表（幂等），在线程中执行。"""
    conn = pymysql.connect(
        host=_DB_HOST,
        port=_DB_PORT,
        user=_DB_USER,
        password=_DB_PASSWORD,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS `text2sql_stats` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cur.execute("USE `text2sql_stats`")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_events (
                    id                BIGINT        AUTO_INCREMENT PRIMARY KEY,
                    thread_id         VARCHAR(64)   NOT NULL,
                    session_name      VARCHAR(255)  NOT NULL DEFAULT '',
                    skill_type        VARCHAR(32)   NOT NULL DEFAULT 'unknown',
                    prompt_tokens     INT           NOT NULL DEFAULT 0,
                    completion_tokens INT           NOT NULL DEFAULT 0,
                    total_tokens      INT           NOT NULL DEFAULT 0,
                    sql_count         INT           NOT NULL DEFAULT 0,
                    success           TINYINT(1)    NOT NULL DEFAULT 1,
                    latency_ms        INT           NOT NULL DEFAULT 0,
                    created_at        DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_created_at (created_at),
                    INDEX idx_thread_id  (thread_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
        logger.info("stats_db: text2sql_stats.query_events ready")
    finally:
        conn.close()


async def init_stats_db() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_init_stats_db)


# ── 写入 ──────────────────────────────────────────────────────────────────────

def _sync_insert_query_event(
    thread_id: str,
    session_name: str,
    skill_type: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    sql_count: int,
    success: bool,
    latency_ms: int,
) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_events
                    (thread_id, session_name, skill_type,
                     prompt_tokens, completion_tokens, total_tokens,
                     sql_count, success, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    thread_id,
                    session_name,
                    skill_type,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    sql_count,
                    1 if success else 0,
                    latency_ms,
                ),
            )
        conn.commit()
    except Exception as e:
        logger.warning(f"stats_db insert failed: {e}")
    finally:
        conn.close()


async def insert_query_event(
    thread_id: str,
    session_name: str = "",
    skill_type: str = "unknown",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    sql_count: int = 0,
    success: bool = True,
    latency_ms: int = 0,
) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _sync_insert_query_event,
        thread_id,
        session_name,
        skill_type,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        sql_count,
        success,
        latency_ms,
    )


# ── 查询 ──────────────────────────────────────────────────────────────────────

def _sync_get_overview() -> dict:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            # 总量
            cur.execute("""
                SELECT
                    COUNT(*)                              AS total_queries,
                    COALESCE(SUM(total_tokens), 0)        AS total_tokens,
                    COALESCE(SUM(sql_count), 0)           AS total_sql,
                    ROUND(AVG(success) * 100, 1)          AS success_rate,
                    COALESCE(AVG(latency_ms), 0)          AS avg_latency_ms
                FROM query_events
            """)
            overview = cur.fetchone() or {}

            # 今日
            cur.execute("""
                SELECT COUNT(*) AS today_queries,
                       COALESCE(SUM(total_tokens), 0) AS today_tokens
                FROM query_events
                WHERE DATE(created_at) = CURDATE()
            """)
            today = cur.fetchone() or {}

            # 本月
            cur.execute("""
                SELECT COUNT(*) AS month_queries,
                       COALESCE(SUM(total_tokens), 0) AS month_tokens
                FROM query_events
                WHERE DATE_FORMAT(created_at, '%Y-%m') = DATE_FORMAT(NOW(), '%Y-%m')
            """)
            month = cur.fetchone() or {}

            # 技能分布
            cur.execute("""
                SELECT skill_type, COUNT(*) AS cnt
                FROM query_events
                GROUP BY skill_type
            """)
            skill_dist = cur.fetchall()

        return {
            **{k: (float(v) if v is not None else 0) for k, v in overview.items()},
            **{k: (float(v) if v is not None else 0) for k, v in today.items()},
            **{k: (float(v) if v is not None else 0) for k, v in month.items()},
            "skill_distribution": [
                {"skill_type": r["skill_type"], "count": int(r["cnt"])}
                for r in skill_dist
            ],
        }
    finally:
        conn.close()


async def get_overview() -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_get_overview)


def _sync_get_monthly_stats(months: int = 12) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    DATE_FORMAT(created_at, '%%Y-%%m')    AS month,
                    COUNT(*)                               AS query_count,
                    COALESCE(SUM(prompt_tokens), 0)        AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0)    AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0)         AS total_tokens,
                    COALESCE(SUM(sql_count), 0)            AS sql_count,
                    ROUND(AVG(success) * 100, 1)           AS success_rate,
                    ROUND(AVG(latency_ms), 0)              AS avg_latency_ms
                FROM query_events
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s MONTH)
                GROUP BY DATE_FORMAT(created_at, '%%Y-%%m')
                ORDER BY month ASC
                """,
                (months,),
            )
            rows = cur.fetchall()
        return [
            {
                "month": r["month"],
                "query_count": int(r["query_count"]),
                "prompt_tokens": int(r["prompt_tokens"]),
                "completion_tokens": int(r["completion_tokens"]),
                "total_tokens": int(r["total_tokens"]),
                "sql_count": int(r["sql_count"]),
                "success_rate": float(r["success_rate"] or 0),
                "avg_latency_ms": float(r["avg_latency_ms"] or 0),
            }
            for r in rows
        ]
    finally:
        conn.close()


async def get_monthly_stats(months: int = 12) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_get_monthly_stats, months)

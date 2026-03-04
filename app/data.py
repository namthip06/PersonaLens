"""
app/data.py
===========
Thin data-access layer that sits between the Streamlit UI and the SQLite
database.  All functions return pandas DataFrames (or plain lists/dicts)
so that the UI layer has zero SQL awareness.

Every function accepts an optional `db_path` argument (defaults to the
project-standard path) and opens its own short-lived connection so the
module can be used safely with Streamlit's multi-threaded rerun model.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "database" / "personalens.db"

logger.debug(f"Loading default DB from {_DEFAULT_DB}")


def _connect(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    logger.debug(f"Calling _connect(db_path={db_path})")
    path = str(db_path or _DEFAULT_DB)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Page 1 – Executive Dashboard
# ---------------------------------------------------------------------------


def get_sentiment_velocity(db_path=_DEFAULT_DB) -> pd.DataFrame:
    """
    Sentiment over time, one row per (date, category).
    Returns: [date, entity_category, average_sentiment]
    """
    logger.debug(f"Calling get_sentiment_velocity(db_path={db_path})")
    sql = """
        SELECT
            date(a.published_at)          AS date,
            e.category                    AS entity_category,
            avg(sr.final_score)           AS average_sentiment
        FROM   sentiment_results sr
        JOIN   articles  a ON a.article_id = sr.article_id
        JOIN   entities  e ON e.entity_id  = sr.entity_id
        WHERE  a.published_at IS NOT NULL
        GROUP  BY 1, 2
        ORDER  BY 1
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_top_mentioned(db_path=_DEFAULT_DB, limit: int = 20) -> pd.DataFrame:
    """
    Most-mentioned entities with label breakdown.
    Returns: [canonical_name, category, sentiment_label, count]
    """
    logger.debug(f"Calling get_top_mentioned(db_path={db_path}, limit={limit})")
    sql = """
        SELECT
            e.canonical_name,
            e.category,
            sr.sentiment_label,
            count(*)  AS count
        FROM   sentiment_results sr
        JOIN   entities e ON e.entity_id = sr.entity_id
        WHERE  e.canonical_name NOT LIKE '__unresolved__%'
        GROUP  BY e.entity_id, sr.sentiment_label
        ORDER  BY count(*) DESC
        LIMIT  ?
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=(limit * 4,))
    return df


def get_publisher_bias(
    db_path=_DEFAULT_DB, entity_ids: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Average sentiment per publisher (optionally filtered by entity ids).
    Returns: [publisher, average_sentiment]
    """
    logger.debug(
        f"Calling get_publisher_bias(db_path={db_path}, entity_ids={entity_ids})"
    )
    where = ""
    params: list = []
    if entity_ids:
        placeholders = ",".join("?" * len(entity_ids))
        where = f"AND sr.entity_id IN ({placeholders})"
        params = list(entity_ids)

    sql = f"""
        SELECT
            a.publisher,
            avg(sr.final_score) AS average_sentiment
        FROM   sentiment_results sr
        JOIN   articles  a ON a.article_id = sr.article_id
        WHERE  a.publisher IS NOT NULL
              {where}
        GROUP  BY a.publisher
        ORDER  BY count(*) DESC
        LIMIT  12
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return df


# ---------------------------------------------------------------------------
# Extra Dashboards
# ---------------------------------------------------------------------------


def get_sentiment_distribution(db_path=_DEFAULT_DB) -> pd.DataFrame:
    """Sentiment Distribution (Donut) สัดส่วน POSITIVE, NEGATIVE, NEUTRAL จากทั้งฐานข้อมูล"""
    logger.debug(f"Calling get_sentiment_distribution(db_path={db_path})")
    sql = """
        SELECT sentiment_label, count(*) AS count
        FROM sentiment_results
        GROUP BY sentiment_label
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    return df


def get_entity_cooccurrence(db_path=_DEFAULT_DB, limit: int = 50) -> pd.DataFrame:
    """Entity Correlation Network กราฟเครือข่ายแสดง Entity ที่มักปรากฏใน article_id เดียวกัน"""
    logger.debug(f"Calling get_entity_cooccurrence(db_path={db_path}, limit={limit})")
    sql = """
        SELECT
            e1.canonical_name AS source,
            e2.canonical_name AS target,
            COUNT(*) AS weight
        FROM sentiment_results sr1
        JOIN sentiment_results sr2 ON sr1.article_id = sr2.article_id AND sr1.entity_id < sr2.entity_id
        JOIN entities e1 ON e1.entity_id = sr1.entity_id
        JOIN entities e2 ON e2.entity_id = sr2.entity_id
        WHERE e1.canonical_name NOT LIKE '__unresolved__%' AND e2.canonical_name NOT LIKE '__unresolved__%'
        GROUP BY e1.entity_id, e2.entity_id, e1.canonical_name, e2.canonical_name
        ORDER BY weight DESC
        LIMIT ?
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=(limit,))
    return df


def get_daily_mention_volume(db_path=_DEFAULT_DB) -> pd.DataFrame:
    """Daily Mention Volume (Area) จำนวนบทความ (articles) ที่ถูกดูดเข้ามาในระบบรายวัน"""
    logger.debug(f"Calling get_daily_mention_volume(db_path={db_path})")
    sql = """
        SELECT date(published_at) AS date, COUNT(*) AS volume
        FROM articles
        WHERE published_at IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_conflict_support_index(db_path=_DEFAULT_DB, limit: int = 50) -> pd.DataFrame:
    """Conflict vs. Support Index กราฟ Scatter Plot ระหว่างจำนวนข่าว (Volume) กับความผันผวนของ Sentiment"""
    logger.debug(
        f"Calling get_conflict_support_index(db_path={db_path}, limit={limit})"
    )
    sql = """
        SELECT
            e.canonical_name,
            e.category,
            COUNT(sr.result_id) AS volume,
            AVG(sr.final_score * sr.final_score) - AVG(sr.final_score)*AVG(sr.final_score) AS volatility
        FROM sentiment_results sr
        JOIN entities e ON e.entity_id = sr.entity_id
        WHERE e.canonical_name NOT LIKE '__unresolved__%'
        GROUP BY e.entity_id, e.canonical_name, e.category
        HAVING volume >= 2
        ORDER BY volume DESC
        LIMIT ?
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=(limit,))
    # SQLite might return None for variance if there's precision issues, clean it up
    if not df.empty:
        df["volatility"] = df["volatility"].fillna(0).clip(lower=0)
    return df


def get_language_diversity(db_path=_DEFAULT_DB) -> pd.DataFrame:
    """Language Diversity (Pie) สัดส่วนภาษาของบทความ"""
    logger.debug(f"Calling get_language_diversity(db_path={db_path})")
    sql = """
        SELECT COALESCE(lang, 'unknown') as lang, count(*) AS count
        FROM articles
        GROUP BY lang
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    return df


# ---------------------------------------------------------------------------
# Page 2 – Entity Deep-Dive
# ---------------------------------------------------------------------------


def get_all_entities(db_path=_DEFAULT_DB) -> pd.DataFrame:
    """Return all non-placeholder entities for the selector."""
    logger.debug(f"Calling get_all_entities(db_path={db_path})")
    sql = """
        SELECT entity_id, canonical_name, category, lang, created_at
        FROM   entities
        WHERE  canonical_name NOT LIKE '__unresolved__%'
        ORDER  BY canonical_name
    """
    with _connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def get_entity_with_aliases(entity_id: str, db_path=_DEFAULT_DB) -> dict:
    """Return entity row + list of alias strings."""
    logger.debug(
        f"Calling get_entity_with_aliases(entity_id={entity_id}, db_path={db_path})"
    )
    with _connect(db_path) as conn:
        entity_row = conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        alias_rows = conn.execute(
            "SELECT alias_text, source_type FROM aliases WHERE entity_id = ?",
            (entity_id,),
        ).fetchall()
    if entity_row is None:
        return {}
    return {
        "entity": dict(entity_row),
        "aliases": [dict(r) for r in alias_rows],
    }


def get_entity_sentiment_summary(entity_id: str, db_path=_DEFAULT_DB) -> pd.DataFrame:
    """Sentiment label counts for a single entity."""
    logger.debug(
        f"Calling get_entity_sentiment_summary(entity_id={entity_id}, db_path={db_path})"
    )
    sql = """
        SELECT sentiment_label, count(*) AS count
        FROM   sentiment_results
        WHERE  entity_id = ?
        GROUP  BY sentiment_label
    """
    with _connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(entity_id,))


def get_analysis_details_for_entity(
    entity_id: str, db_path=_DEFAULT_DB, limit: int = 50
) -> list[dict]:
    """
    Fetch analysis_details rows for an entity, newest first.
    Each row has: sentence_text, is_headline, raw_sentiment,
                  reasoning (JSON string), confidence_score,
                  headline, source_url, published_at.
    """
    logger.debug(
        f"Calling get_analysis_details_for_entity(entity_id={entity_id}, db_path={db_path}, limit={limit})"
    )
    sql = """
        SELECT
            ad.detail_id,
            ad.sentence_text,
            ad.is_headline,
            ad.raw_sentiment,
            ad.reasoning,
            sr.confidence_score,
            sr.result_id,
            a.headline,
            a.source_url,
            a.published_at
        FROM   analysis_details ad
        JOIN   sentiment_results sr ON sr.result_id  = ad.result_id
        JOIN   articles          a  ON a.article_id  = sr.article_id
        WHERE  sr.entity_id = ?
        ORDER  BY a.published_at DESC NULLS LAST
        LIMIT  ?
    """
    with _connect(db_path) as conn:
        rows = conn.execute(sql, (entity_id, limit)).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        # Parse reasoning JSON safely
        try:
            d["reasoning_parsed"] = json.loads(d["reasoning"])
        except (json.JSONDecodeError, TypeError):
            d["reasoning_parsed"] = {}
        result.append(d)
    return result


def get_entity_timeline(entity_id: str, db_path=_DEFAULT_DB) -> pd.DataFrame:
    """Daily average sentiment for one entity."""
    logger.debug(
        f"Calling get_entity_timeline(entity_id={entity_id}, db_path={db_path})"
    )
    sql = """
        SELECT
            date(a.published_at) AS date,
            avg(sr.final_score)  AS avg_sentiment,
            count(*)             AS mention_count
        FROM   sentiment_results sr
        JOIN   articles a ON a.article_id = sr.article_id
        WHERE  sr.entity_id = ?
          AND  a.published_at IS NOT NULL
        GROUP  BY 1
        ORDER  BY 1
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=(entity_id,))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Page 2 – Entity Deep-Dive (Granular)
# ---------------------------------------------------------------------------


def get_entity_trajectory(entity_id: str, db_path=_DEFAULT_DB) -> pd.DataFrame:
    """Sentiment trajectory: final_score of each article for one entity."""
    logger.debug(
        f"Calling get_entity_trajectory(entity_id={entity_id}, db_path={db_path})"
    )
    sql = """
        SELECT
            a.published_at AS date,
            sr.final_score,
            COALESCE(a.headline, a.source_url) AS headline,
            a.source_url
        FROM sentiment_results sr
        JOIN articles a ON a.article_id = sr.article_id
        WHERE sr.entity_id = ? AND a.published_at IS NOT NULL
        ORDER BY a.published_at ASC
    """
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=(entity_id,))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_top_publishers_for_entity(
    entity_id: str, limit: int = 10, db_path=_DEFAULT_DB
) -> pd.DataFrame:
    """Publishers reporting on this entity."""
    logger.debug(
        f"Calling get_top_publishers_for_entity(entity_id={entity_id}, limit={limit}, db_path={db_path})"
    )
    sql = """
        SELECT
            a.publisher,
            count(*) AS article_count,
            avg(sr.final_score) AS avg_sentiment
        FROM sentiment_results sr
        JOIN articles a ON a.article_id = sr.article_id
        WHERE sr.entity_id = ? AND a.publisher IS NOT NULL
        GROUP BY a.publisher
        ORDER BY article_count DESC
        LIMIT ?
    """
    with _connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(entity_id, limit))


def get_confidence_distribution_for_entity(
    entity_id: str, db_path=_DEFAULT_DB
) -> pd.DataFrame:
    """Distribution of analysis confidence scores for an entity."""
    logger.debug(
        f"Calling get_confidence_distribution_for_entity(entity_id={entity_id}, db_path={db_path})"
    )
    sql = """
        SELECT confidence_score
        FROM sentiment_results
        WHERE entity_id = ?
    """
    with _connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(entity_id,))


def get_speaker_network_for_entity(
    entity_id: str, limit: int = 15, db_path=_DEFAULT_DB
) -> pd.DataFrame:
    """Speakers who mentioned this entity."""
    logger.debug(
        f"Calling get_speaker_network_for_entity(entity_id={entity_id}, limit={limit}, db_path={db_path})"
    )
    sql = """
        SELECT
            e.canonical_name AS speaker_name,
            COUNT(*) AS mention_count,
            AVG(sr.final_score) AS avg_sentiment
        FROM analysis_details ad
        JOIN sentiment_results sr ON sr.result_id = ad.result_id
        JOIN entities e ON e.entity_id = ad.speaker_id
        WHERE sr.entity_id = ? AND ad.speaker_id IS NOT NULL
        GROUP BY ad.speaker_id, e.canonical_name
        ORDER BY mention_count DESC
        LIMIT ?
    """
    with _connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(entity_id, limit))


# ---------------------------------------------------------------------------
# Page 3 – Admin / Pipeline
# ---------------------------------------------------------------------------


def get_pipeline_stats(db_path=_DEFAULT_DB) -> dict:
    """Quick counts for the admin page KPI cards."""
    logger.debug(f"Calling get_pipeline_stats(db_path={db_path})")
    with _connect(db_path) as conn:
        articles = conn.execute("SELECT count(*) FROM articles").fetchone()[0]
        entities = conn.execute(
            "SELECT count(*) FROM entities WHERE canonical_name NOT LIKE '__unresolved__%'"
        ).fetchone()[0]
        results = conn.execute("SELECT count(*) FROM sentiment_results").fetchone()[0]
        details = conn.execute("SELECT count(*) FROM analysis_details").fetchone()[0]
    return {
        "articles": articles,
        "entities": entities,
        "sentiment_results": results,
        "analysis_details": details,
    }


def get_recent_articles(db_path=_DEFAULT_DB, limit: int = 20) -> pd.DataFrame:
    logger.debug(f"Calling get_recent_articles(db_path={db_path}, limit={limit})")
    sql = """
        SELECT article_id, headline, publisher, published_at, source_url
        FROM   articles
        ORDER  BY published_at DESC NULLS LAST
        LIMIT  ?
    """
    with _connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(limit,))


def get_etl_metrics(db_path=_DEFAULT_DB) -> dict:
    """ETL KPIs placeholders and actual file sizes."""
    import os

    db_size = 0.0
    if os.path.exists(str(db_path)):
        db_size = os.path.getsize(str(db_path)) / (1024 * 1024)

    return {
        "pipeline_latency_sec": 12.4,  # Placeholder average latency
        "deduplication_rate_pct": 18.5,  # Placeholder dedup rate
        "cache_hit_rate_pct": 68.2,  # Placeholder alias DB cache hit rate
        "db_size_mb": round(db_size, 2),
    }


def get_resolution_accuracy(db_path=_DEFAULT_DB) -> pd.DataFrame:
    """Alias processing sources (manual vs slm vs api)."""
    sql = """
        SELECT source_type, count(*) as count
        FROM aliases
        GROUP BY source_type
    """
    with _connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def get_foreign_key_integrity(db_path=_DEFAULT_DB) -> pd.DataFrame:
    """Check for orphaned records."""
    sql = """
        SELECT 'Orphaned Sentiment Results' as issue, count(*) as count
        FROM sentiment_results 
        WHERE article_id NOT IN (SELECT article_id FROM articles) 
           OR entity_id NOT IN (SELECT entity_id FROM entities)
        UNION ALL
        SELECT 'Orphaned Analysis Details', count(*)
        FROM analysis_details
        WHERE result_id NOT IN (SELECT result_id FROM sentiment_results)
        UNION ALL
        SELECT 'Orphaned Aliases', count(*)
        FROM aliases
        WHERE entity_id NOT IN (SELECT entity_id FROM entities)
    """
    with _connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def get_failed_ingestion_logs() -> pd.DataFrame:
    """Mock failed ingestion logs table."""
    data = [
        {
            "timestamp": "2026-03-04 10:15:00",
            "url": "https://example.com/bad-article-1",
            "error": "Trafilatura scrape failed (404 Not Found)",
        },
        {
            "timestamp": "2026-03-04 11:20:00",
            "url": "https://example.com/paywall-news",
            "error": "Body length < 100 chars (Paywalled)",
        },
        {
            "timestamp": "2026-03-04 14:05:00",
            "url": "https://example.com/timeout",
            "error": "SLM Connection Timeout",
        },
    ]
    return pd.DataFrame(data)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    print(get_recent_articles())

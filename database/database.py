"""
database/database.py
=====================
SQLite persistence layer for PersonaLens.

Schema (mirrors the ERD in System Context.md)
---------------------------------------------
  entities          – unique personas (canonical_name, category, lang)
  aliases           – name variations linked to entities
  articles          – news article metadata
  sentiment_results – aggregated per-entity score per article
  analysis_details  – sentence-level granular ABSA output

Public API
----------
  Database(path)                     – open / create the database
  Database.upsert_entity(...)        – insert or ignore an entity row
  Database.upsert_alias(...)         – insert or ignore an alias row
  Database.upsert_article(...)       – insert or ignore an article row
  Database.save_analyzer_result(...) – persist one AnalyzerResult to DB
  Database.close()                   – close the connection
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# DDL – table definitions
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Master Data ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS entities (
    entity_id      TEXT    PRIMARY KEY,          -- UUID stored as TEXT
    canonical_name TEXT    NOT NULL UNIQUE,
    category       TEXT    NOT NULL,             -- PER | ORG | LOC | GPE
    lang           TEXT    NOT NULL DEFAULT 'th',
    created_at     TEXT    NOT NULL              -- ISO-8601 UTC timestamp
);

CREATE TABLE IF NOT EXISTS aliases (
    alias_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   TEXT    NOT NULL REFERENCES entities(entity_id)
                        ON DELETE CASCADE,
    alias_text  TEXT    NOT NULL,
    source_type TEXT    NOT NULL DEFAULT 'manual',  -- manual | slm | api
    UNIQUE (entity_id, alias_text)
);

-- ── Transactional Data ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS articles (
    article_id   TEXT    PRIMARY KEY,            -- UUID stored as TEXT
    headline     TEXT,
    source_url   TEXT    NOT NULL UNIQUE,
    publisher    TEXT,
    lang         TEXT    NOT NULL DEFAULT 'th',
    published_at TEXT                            -- ISO-8601 UTC timestamp (nullable)
);

CREATE TABLE IF NOT EXISTS sentiment_results (
    result_id       TEXT    PRIMARY KEY,         -- UUID stored as TEXT
    article_id      TEXT    NOT NULL REFERENCES articles(article_id)
                            ON DELETE RESTRICT,
    entity_id       TEXT    NOT NULL REFERENCES entities(entity_id)
                            ON DELETE RESTRICT,
    final_score     REAL    NOT NULL,            -- aggregated [-1, 1] or [0, 1] score
    sentiment_label TEXT    NOT NULL,            -- POSITIVE | NEGATIVE | NEUTRAL | MIXED
    confidence_score REAL   NOT NULL DEFAULT 1.0,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sr_entity_article
    ON sentiment_results (entity_id, article_id);

CREATE TABLE IF NOT EXISTS analysis_details (
    detail_id       TEXT    PRIMARY KEY,         -- UUID stored as TEXT
    result_id       TEXT    NOT NULL REFERENCES sentiment_results(result_id)
                            ON DELETE CASCADE,
    speaker_id      TEXT    REFERENCES entities(entity_id)
                            ON DELETE SET NULL,  -- NULL → journalist narration
    sentence_text   TEXT    NOT NULL,            -- the context_window sent to SLM
    is_headline     INTEGER NOT NULL DEFAULT 0,  -- 0 = false, 1 = true
    raw_sentiment   TEXT    NOT NULL,            -- ABSA SentimentLabel value
    reasoning       TEXT    NOT NULL             -- JSON blob with full ABSAOutput
);
"""

# ---------------------------------------------------------------------------
# Sentinel for "no article provided"
# ---------------------------------------------------------------------------

_SENTINEL_URL = "__internal__no_url__"


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class Database:
    """
    Thread-local SQLite wrapper for PersonaLens.

    Parameters
    ----------
    path : str | Path
        File-system path to the SQLite database file.
        Pass ``":memory:"`` for an in-memory database (testing only).
    """

    def __init__(self, path: str | Path = "database/personalens.db") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        """Apply DDL statements to create tables if they do not yet exist."""
        self._conn.executescript(_DDL)
        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_uuid() -> str:
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Master data – entities
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        canonical_name: str,
        category: str,
        lang: str = "th",
        entity_id: Optional[str] = None,
    ) -> str:
        """
        Insert an entity if it does not yet exist (matched on canonical_name).

        Returns
        -------
        str
            The entity_id (UUID string) of the existing or newly created row.
        """
        cur = self._conn.execute(
            "SELECT entity_id FROM entities WHERE canonical_name = ?",
            (canonical_name,),
        )
        row = cur.fetchone()
        if row:
            return row["entity_id"]

        eid = entity_id or self._new_uuid()
        self._conn.execute(
            """
            INSERT INTO entities (entity_id, canonical_name, category, lang, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (eid, canonical_name, category, lang, self._now()),
        )
        self._conn.commit()
        return eid

    def get_entity_by_name(self, canonical_name: str) -> Optional[sqlite3.Row]:
        """Return the entity row for *canonical_name*, or None."""
        cur = self._conn.execute(
            "SELECT * FROM entities WHERE canonical_name = ?", (canonical_name,)
        )
        return cur.fetchone()

    def get_entity_by_id(self, entity_id: str) -> Optional[sqlite3.Row]:
        """Return the entity row for *entity_id*, or None."""
        cur = self._conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
        )
        return cur.fetchone()

    # ------------------------------------------------------------------
    # Master data – aliases
    # ------------------------------------------------------------------

    def upsert_alias(
        self,
        entity_id: str,
        alias_text: str,
        source_type: str = "manual",
    ) -> int:
        """
        Insert an alias row if it does not already exist.

        Returns
        -------
        int
            The alias_id (ROWID) of the existing or newly created alias.
        """
        cur = self._conn.execute(
            "SELECT alias_id FROM aliases WHERE entity_id = ? AND alias_text = ?",
            (entity_id, alias_text),
        )
        row = cur.fetchone()
        if row:
            return row["alias_id"]

        cur = self._conn.execute(
            """
            INSERT INTO aliases (entity_id, alias_text, source_type)
            VALUES (?, ?, ?)
            """,
            (entity_id, alias_text, source_type),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_aliases(self, entity_id: str) -> list[sqlite3.Row]:
        """Return all alias rows for the given entity."""
        cur = self._conn.execute(
            "SELECT * FROM aliases WHERE entity_id = ?", (entity_id,)
        )
        return cur.fetchall()

    def find_alias_exact(self, surface_form: str) -> Optional[sqlite3.Row]:
        """
        Case-insensitive exact lookup of *surface_form* in the ``aliases`` table.

        Returns a ``sqlite3.Row`` with columns
        ``(entity_id, canonical_name, category, lang)`` on match, else ``None``.
        Used by ``alias_resolver.lookup_alias_exact()``.
        """
        cur = self._conn.execute(
            """
            SELECT a.entity_id, e.canonical_name, e.category, e.lang
            FROM   aliases  a
            JOIN   entities e ON e.entity_id = a.entity_id
            WHERE  lower(a.alias_text) = lower(?)
            LIMIT  1
            """,
            (surface_form.strip(),),
        )
        return cur.fetchone()

    def find_all_aliases_with_entities(self) -> list[sqlite3.Row]:
        """
        Return every alias row joined to its parent entity.

        Columns: ``(alias_text, entity_id, canonical_name, category, lang)``.
        Used by ``alias_resolver.lookup_alias_fuzzy()`` which loads all aliases
        into memory and ranks them with rapidfuzz.
        """
        cur = self._conn.execute(
            """
            SELECT a.alias_text, a.entity_id, e.canonical_name, e.category, e.lang
            FROM   aliases  a
            JOIN   entities e ON e.entity_id = a.entity_id
            ORDER  BY e.canonical_name
            """
        )
        return cur.fetchall()

    # ------------------------------------------------------------------
    # Transactional data – articles
    # ------------------------------------------------------------------

    def upsert_article(
        self,
        source_url: str,
        headline: Optional[str] = None,
        publisher: Optional[str] = None,
        lang: str = "th",
        published_at: Optional[str] = None,
        article_id: Optional[str] = None,
    ) -> str:
        """
        Insert an article by its URL (unique constraint).

        Returns
        -------
        str
            The article_id (UUID string) of the existing or newly created row.
        """
        cur = self._conn.execute(
            "SELECT article_id FROM articles WHERE source_url = ?", (source_url,)
        )
        row = cur.fetchone()
        if row:
            return row["article_id"]

        aid = article_id or self._new_uuid()
        self._conn.execute(
            """
            INSERT INTO articles (article_id, headline, source_url, publisher, lang, published_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (aid, headline, source_url, publisher, lang, published_at),
        )
        self._conn.commit()
        return aid

    # ------------------------------------------------------------------
    # Core persistence – AnalyzerResult
    # ------------------------------------------------------------------

    def save_analyzer_result(
        self,
        result,  # src.schemas.inference.AnalyzerResult
        article_id: Optional[str] = None,
        source_url: Optional[str] = None,
        headline: Optional[str] = None,
        publisher: Optional[str] = None,
        lang: str = "th",
        published_at: Optional[str] = None,
        is_headline: bool = False,
    ) -> str:
        """
        Persist one ``AnalyzerResult`` to the database.

        This is the main write entry-point for the analysis pipeline.

        Parameters
        ----------
        result       : AnalyzerResult instance.
        article_id   : Pre-existing article UUID.  Supply either this OR
                       ``source_url`` so the method can upsert the article.
        source_url   : URL of the article (used for upsert if article_id is absent).
        headline     : Optional headline text.
        publisher    : Optional publisher name.
        lang         : Language code (default ``"th"``).
        published_at : ISO-8601 publish timestamp of the article.
        is_headline  : Whether this context_window came from the headline.

        Returns
        -------
        str
            The ``result_id`` UUID of the newly written ``sentiment_results`` row.
        """
        absa = result.absa

        # ── 1. Resolve article ────────────────────────────────────────────
        if article_id is None:
            url = source_url or _SENTINEL_URL
            article_id = self.upsert_article(
                source_url=url,
                headline=headline,
                publisher=publisher,
                lang=lang,
                published_at=published_at,
            )

        # ── 2. Resolve entity ─────────────────────────────────────────────
        if result.global_id is not None:
            entity_id = str(result.global_id)
        elif result.canonical_name:
            entity_id = self.upsert_entity(
                canonical_name=result.canonical_name,
                category="PER",  # default; caller can update afterwards
                lang=lang,
            )
        else:
            # Completely unresolved – store under the surface form as a
            # placeholder entity so we never lose the data.
            entity_id = self.upsert_entity(
                canonical_name=f"__unresolved__{result.surface_form}",
                category="PER",
                lang=lang,
            )

        # ── 3. Map sentiment to a numeric score ───────────────────────────
        _score_map = {
            "POSITIVE": 1.0,
            "NEGATIVE": -1.0,
            "NEUTRAL": 0.0,
            "MIXED": 0.0,
        }
        final_score = _score_map.get(absa.sentiment.value, 0.0)
        sentiment_label = absa.sentiment.value

        # ── 4. Resolve speaker entity (for QUOTE rows) ────────────────────
        speaker_id: Optional[str] = None
        if absa.speaker_type.value == "QUOTE" and absa.speaker_name:
            speaker_id = self.upsert_entity(
                canonical_name=absa.speaker_name,
                category="PER",
                lang=lang,
            )

        # ── 5. Write sentiment_results row ────────────────────────────────
        result_id = self._new_uuid()
        self._conn.execute(
            """
            INSERT INTO sentiment_results
                (result_id, article_id, entity_id, final_score, sentiment_label,
                 confidence_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                article_id,
                entity_id,
                final_score,
                sentiment_label,
                1.0,  # confidence placeholder – extend as needed
                self._now(),
            ),
        )

        # ── 6. Write analysis_details row ─────────────────────────────────
        detail_id = self._new_uuid()
        reasoning_json = absa.model_dump_json()  # full ABSAOutput as JSON
        self._conn.execute(
            """
            INSERT INTO analysis_details
                (detail_id, result_id, speaker_id, sentence_text,
                 is_headline, raw_sentiment, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detail_id,
                result_id,
                speaker_id,
                result.context_window,
                int(is_headline),
                sentiment_label,
                reasoning_json,
            ),
        )

        self._conn.commit()
        return result_id

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def get_sentiment_results(
        self, entity_id: str, limit: int = 100
    ) -> list[sqlite3.Row]:
        """
        Return the most recent sentiment results for an entity.
        Joins ``articles`` to include ``published_at`` and ``source_url``.
        """
        cur = self._conn.execute(
            """
            SELECT sr.*, a.headline, a.source_url, a.published_at
            FROM   sentiment_results sr
            JOIN   articles a ON sr.article_id = a.article_id
            WHERE  sr.entity_id = ?
            ORDER  BY a.published_at DESC NULLS LAST
            LIMIT  ?
            """,
            (entity_id, limit),
        )
        return cur.fetchall()

    def get_analysis_details(self, result_id: str) -> list[sqlite3.Row]:
        """Return all analysis_details rows for a given sentiment_results row."""
        cur = self._conn.execute(
            "SELECT * FROM analysis_details WHERE result_id = ?", (result_id,)
        )
        return cur.fetchall()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_) -> None:
        self.close()

"""
database/init_db.py
===================
Database initialization script for PersonaLens.

Run this script once to:
  1. Verify the connection to PostgreSQL.
  2. Create all tables (if they do not already exist).
  3. Apply recommended performance indexes on top of what SQLAlchemy creates.

Usage
-----
    python -m database.init_db            # normal run
    python -m database.init_db --drop     # ⚠ DROP all tables first then recreate
"""

import argparse
import logging
import sys

from sqlalchemy import Index, inspect, text

from database.connection import get_engine, ping
from database.models import (
    Alias,
    AnalysisDetail,
    Article,
    Base,
    Entity,
    SentimentResult,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("personalens.init_db")


# ---------------------------------------------------------------------------
# Extra performance indexes (not expressible purely in the model definitions)
# ---------------------------------------------------------------------------


def _define_extra_indexes() -> list[Index]:
    """
    Returns additional SQLAlchemy Index objects that should be created after
    the tables exist.

    These are defined here (not on the models) because they reference columns
    from multiple tables or use PostgreSQL-specific index options that are more
    readable when kept separate from the ORM model declarations.
    """
    return [
        # ----------------------------------------------------------------
        # Senior's advice §6: "Composite Index on (entity_id, published_at)"
        # Most dashboards query: "entity X's sentiment over time period Y."
        # Because published_at lives on articles, we create an index on
        # sentiment_results.entity_id + articles.published_at via a
        # functional / expression index approach.
        #
        # Here we add a plain composite index on sentiment_results alone
        # (entity_id, article_id).  A covering index that joins published_at
        # requires a materialized view or a denormalised column; a migration
        # can add that once query profiling confirms the bottleneck.
        # ----------------------------------------------------------------
        Index(
            "ix_sentiment_results_entity_article",
            SentimentResult.entity_id,
            SentimentResult.article_id,
        ),
        # Index to support alias lookup during entity-linking
        Index(
            "ix_aliases_alias_text",
            Alias.alias_text,
            postgresql_ops={"alias_text": "text_pattern_ops"},  # supports LIKE lookups
        ),
        # Fast lookup of all analysis details containing a quoted speaker
        Index(
            "ix_analysis_details_speaker",
            AnalysisDetail.speaker_id,
        ),
    ]


# ---------------------------------------------------------------------------
# Core init logic
# ---------------------------------------------------------------------------


def drop_all(engine) -> None:
    """Drop every table managed by this project's Base (irreversible!)."""
    logger.warning("⚠  Dropping ALL PersonaLens tables …")
    Base.metadata.drop_all(bind=engine)
    logger.warning("⚠  All tables dropped.")


def create_all(engine) -> None:
    """Create tables that do not exist yet (safe to run multiple times)."""
    logger.info("Creating tables (checkfirst=True) …")
    Base.metadata.create_all(bind=engine, checkfirst=True)
    logger.info("Tables created successfully.")


def apply_extra_indexes(engine) -> None:
    """Create the extra performance indexes defined above."""
    inspector = inspect(engine)
    for idx in _define_extra_indexes():
        table_name = (
            idx.table.name
            if hasattr(idx, "table") and idx.table is not None
            else "unknown"
        )
        existing = (
            {i["name"] for i in inspector.get_indexes(table_name)}
            if table_name != "unknown"
            else set()
        )
        if idx.name in existing:
            logger.info("Index '%s' already exists – skipping.", idx.name)
        else:
            try:
                idx.create(bind=engine)
                logger.info("Created index '%s' on '%s'.", idx.name, table_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create index '%s': %s", idx.name, exc)


def report_schema(engine) -> None:
    """Log the list of tables and their columns for confirmation."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    logger.info("─" * 60)
    logger.info("Schema summary (%d tables):", len(table_names))
    for tname in sorted(table_names):
        cols = [c["name"] for c in inspector.get_columns(tname)]
        idxs = [i["name"] for i in inspector.get_indexes(tname)]
        logger.info("  %-30s cols=%s", tname, cols)
        if idxs:
            logger.info("  %-30s idx =%s", "", idxs)
    logger.info("─" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PersonaLens – PostgreSQL initialization script"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="⚠ DROP all tables before recreating them (data will be lost).",
    )
    args = parser.parse_args()

    engine = get_engine()

    # 1. Verify connectivity
    logger.info("Pinging database …")
    if not ping(engine):
        logger.error(
            "Cannot connect to the database. "
            "Check your .env file and ensure PostgreSQL is running."
        )
        sys.exit(1)
    logger.info("Database connection OK.")

    # 2. Optional destructive reset
    if args.drop:
        confirm = input(
            "\n⚠  WARNING: This will DELETE all tables and data.\n"
            "Type 'yes' to continue: "
        ).strip()
        if confirm.lower() != "yes":
            logger.info("Aborted.")
            sys.exit(0)
        drop_all(engine)

    # 3. Create tables
    create_all(engine)

    # 4. Extra performance indexes
    apply_extra_indexes(engine)

    # 5. Print summary
    report_schema(engine)

    logger.info("✅  PersonaLens database initialization complete.")


if __name__ == "__main__":
    main()

"""
database/connection.py
======================
Builds the SQLAlchemy Engine and Session factory from environment variables
loaded via python-dotenv.

Usage
-----
    from database.connection import get_engine, get_session

    engine  = get_engine()
    Session = get_session()
    with Session() as session:
        ...
"""

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Load .env (or .env file passed to load_dotenv).
# This is a no-op if the variables are already set in the process environment,
# which is safe for production deployments that inject secrets via the OS.
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_database_url() -> str:
    """
    Construct the PostgreSQL connection URL from individual env vars.
    Using individual vars (instead of a single DATABASE_URL) is safer because
    it avoids embedding the password in a single string that might be logged.

    Raises
    ------
    EnvironmentError
        If any required variable is missing.
    """
    required = {
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT", "5432"),
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
    }

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Please check your .env file."
        )

    return (
        "postgresql+psycopg2://"
        f"{required['DB_USER']}:{required['DB_PASSWORD']}"
        f"@{required['DB_HOST']}:{required['DB_PORT']}"
        f"/{required['DB_NAME']}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Return a singleton SQLAlchemy Engine.

    Pool settings are read from optional env vars with sensible defaults:
        DB_POOL_SIZE    (default 5)
        DB_MAX_OVERFLOW (default 10)
        DB_POOL_TIMEOUT (default 30 seconds)
    """
    url = _build_database_url()
    engine = create_engine(
        url,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
        pool_pre_ping=True,  # discard stale connections before handing them out
        echo=os.getenv("DB_ECHO", "false").lower() == "true",
    )

    # Log only a sanitised URL (password masked)
    safe_url = url.split("@")[-1]  # everything after the credentials
    logger.info("Database engine created → postgresql://***@%s", safe_url)

    return engine


def get_session() -> sessionmaker:
    """
    Return a configured sessionmaker bound to the singleton engine.
    Each call returns THE SAME factory (engine is cached via lru_cache).

    Example
    -------
        Session = get_session()
        with Session() as session:
            results = session.execute(text("SELECT 1")).all()
    """
    engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db_session() -> Session:
    """
    Convenience generator for dependency-injection patterns (e.g. FastAPI).

    Example
    -------
        for session in get_db_session():
            ...
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def ping(engine: Engine | None = None) -> bool:
    """
    Execute a trivial query to verify the connection is alive.

    Returns True if the database is reachable, False otherwise.
    """
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Database ping failed: %s", exc)
        return False

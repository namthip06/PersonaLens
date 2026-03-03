"""
main.py
=======
PersonaLens – entry point.

Quick-start
-----------
1. Copy .env.example → .env and fill in your PostgreSQL credentials.
2. Run the database initializer:
       python -m database.init_db
3. Then run this script:
       python main.py
"""

import logging

from database.connection import get_engine, get_session, ping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    engine = get_engine()

    if ping(engine):
        logger.info("✅  Connected to PostgreSQL successfully.")
    else:
        logger.error("❌  Could not reach the database. Check your .env file.")
        return

    # Example: open a session and run a quick health query
    Session = get_session()
    with Session() as session:
        from sqlalchemy import text

        result = session.execute(
            text("SELECT current_database(), version()")
        ).fetchone()
        logger.info("Database : %s", result[0])
        logger.info("Version  : %s", result[1].split(",")[0])


if __name__ == "__main__":
    main()

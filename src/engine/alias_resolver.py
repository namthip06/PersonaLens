"""
src/engine/alias_resolver.py
============================
Step 2.2: Semantic Alias Resolution — The "Alias Bridge"

Resolves entity surface forms to canonical identities using the local
`aliases` DB table before escalating to external APIs.

Resolution strategy (in order of preference)
--------------------------------------------
1. Exact match  – case-insensitive lookup in `aliases.alias_text`
2. Fuzzy match  – Levenshtein ratio ≥ threshold via rapidfuzz

Public functions
----------------
    lookup_alias_exact(surface_form, session)  → tuple | None
    lookup_alias_fuzzy(surface_form, session, threshold)  → tuple | None
    resolve_from_db(entity, session)  → ResolvedEntity | None

Returns None when no match is found (caller should escalate to Step 2.3).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.schemas.inference import (  # noqa: E402
    ExtractedEntity,
    ResolvedEntity,
)

if TYPE_CHECKING:
    import uuid
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fuzzy matching threshold (Levenshtein ratio, 0.0 – 1.0)
# ---------------------------------------------------------------------------
DEFAULT_FUZZY_THRESHOLD: float = 0.88

# Trigger prefixes / patterns for Thai political nicknames
# If a surface_form starts with any of these, it hints at an alias/nickname.
NICKNAME_PREFIXES: tuple[str, ...] = (
    "เสี่ย",  # "Sia" – Thai honorific for wealthy/influential person
    "หมอ",  # "Mor" – Doctor (as nickname)
    "บิ๊ก",  # "Big"
    "เฮีย",  # "Hia" – older brother / influential figure
    "ป้า",  # "Pa" – aunt
    "ลุง",  # "Lung" – uncle
    "Sia",
    "Khun",
    "นาย",  # "Nai" – Mister (formal)
    "ดร.",  # "Dr."
)

LOC_GPE_KEYWORDS: tuple[str, ...] = (
    "จังหวัด",  # Province
    "อำเภอ",  # District
    "ภาค",  # Region (e.g. ภาคเหนือ = North)
    "แขวง",  # Sub-district
    "ประเทศ",  # Country
    "กรุง",  # Capital city prefix
)


def is_likely_alias(surface_form: str) -> bool:
    """
    Heuristic: returns True if the surface_form looks like a nickname
    or informal alias that warrants a deeper DB lookup log message.

    Used only for verbosity control – all entities still go through the
    full lookup regardless of this flag.
    """
    sf = surface_form.strip()
    return any(sf.startswith(p) for p in NICKNAME_PREFIXES)


# ---------------------------------------------------------------------------
# Step 2.2a – Exact-match lookup
# ---------------------------------------------------------------------------


def lookup_alias_exact(
    surface_form: str,
    session: "Session",
) -> tuple["uuid.UUID", str] | None:
    """
    Query the `aliases` table for an exact (case-insensitive) match.

    Parameters
    ----------
    surface_form : The entity text span to look up.
    session      : Active SQLAlchemy session connected to the DB.

    Returns
    -------
    (entity_id, canonical_name) tuple on match, else None.
    """
    try:
        from database.models import Alias, Entity  # lazy import to avoid circular
        from sqlalchemy import func

        row = (
            session.query(Alias.entity_id, Entity.canonical_name)
            .join(Entity, Entity.entity_id == Alias.entity_id)
            .filter(func.lower(Alias.alias_text) == func.lower(surface_form.strip()))
            .first()
        )
        if row:
            logger.debug(
                "alias_exact hit: '%s' → entity %s", surface_form, row.entity_id
            )
            return row.entity_id, row.canonical_name
        return None
    except Exception as exc:
        logger.warning("alias_exact lookup failed for '%s': %s", surface_form, exc)
        return None


# ---------------------------------------------------------------------------
# Step 2.2b – Fuzzy-match lookup
# ---------------------------------------------------------------------------


def lookup_alias_fuzzy(
    surface_form: str,
    session: "Session",
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> tuple["uuid.UUID", str, float] | None:
    """
    Load all alias texts from the DB and find the closest match using
    Levenshtein ratio (rapidfuzz).  Only returns a hit if the score meets
    the threshold.

    Parameters
    ----------
    surface_form : Entity text span to compare.
    session      : Active SQLAlchemy session.
    threshold    : Minimum similarity ratio (default 0.88).

    Returns
    -------
    (entity_id, canonical_name, score) on match, else None.
    """
    try:
        from rapidfuzz import fuzz, process
        from database.models import Alias, Entity

        # Fetch all aliases with their entity details in one query
        rows = (
            session.query(Alias.alias_text, Alias.entity_id, Entity.canonical_name)
            .join(Entity, Entity.entity_id == Alias.entity_id)
            .all()
        )

        if not rows:
            return None

        choices = {row.alias_text: (row.entity_id, row.canonical_name) for row in rows}

        # Use token_sort_ratio for better handling of word-order differences
        match = process.extractOne(
            surface_form.strip(),
            choices.keys(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold * 100,  # rapidfuzz uses 0–100 scale
        )

        if match:
            matched_text, score, _ = match
            entity_id, canonical_name = choices[matched_text]
            normalized_score = score / 100.0
            logger.debug(
                "alias_fuzzy hit: '%s' → '%s' (score=%.3f)",
                surface_form,
                matched_text,
                normalized_score,
            )
            return entity_id, canonical_name, normalized_score

        return None
    except ImportError:
        logger.error("rapidfuzz not installed – fuzzy matching unavailable")
        return None
    except Exception as exc:
        logger.warning("alias_fuzzy lookup failed for '%s': %s", surface_form, exc)
        return None


# ---------------------------------------------------------------------------
# Step 2.2 – Main resolver
# ---------------------------------------------------------------------------


def resolve_from_db(
    entity: ExtractedEntity,
    session: "Session",
) -> ResolvedEntity | None:
    """
    Orchestrates the two-stage DB alias resolution for a single extracted entity.

    Resolution order
    ----------------
    1. Exact case-insensitive match  → confidence 1.0, method "alias_exact"
    2. Fuzzy (Levenshtein) match     → confidence = similarity score, method "alias_fuzzy"
    3. No match                      → return None (caller escalates to Step 2.3)

    Parameters
    ----------
    entity  : ExtractedEntity from the NER step (Step 2.1).
    session : Active SQLAlchemy DB session.

    Returns
    -------
    ResolvedEntity if a match was found, else None.
    """
    sf = entity.surface_form

    if is_likely_alias(sf):
        logger.info("'%s' flagged as likely alias – triggering deep alias check", sf)

    # --- Stage 1: exact match ---
    exact = lookup_alias_exact(sf, session)
    if exact:
        entity_id, canonical_name = exact
        return ResolvedEntity(
            surface_form=sf,
            entity_type=entity.entity_type,
            global_id=entity_id,
            canonical_name=canonical_name,
            confidence_score=1.0,
            resolution_method="alias_exact",
        )

    # --- Stage 2: fuzzy match ---
    fuzzy = lookup_alias_fuzzy(sf, session)
    if fuzzy:
        entity_id, canonical_name, score = fuzzy
        return ResolvedEntity(
            surface_form=sf,
            entity_type=entity.entity_type,
            global_id=entity_id,
            canonical_name=canonical_name,
            confidence_score=round(score, 4),
            resolution_method="alias_fuzzy",
        )

    logger.info("No DB alias found for '%s' – escalating to external validation", sf)
    return None


# ---------------------------------------------------------------------------
# Standalone smoke test – `uv run src/engine/alias_resolver.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    print("=" * 60)
    print("PersonaLens – Alias Resolver Smoke Test")
    print("=" * 60)

    # Test the heuristic trigger
    test_cases = [
        "เสี่ยหนู",  # should flag as likely alias
        "Anutin Charnvirakul",  # formal name, no flag
        "ภาคเหนือ",  # LOC – no flag (not a nickname)
        "พรรคภูมิใจไทย",  # ORG – no flag
    ]
    print("\nHeuristic trigger check (is_likely_alias):")
    for tc in test_cases:
        flag = is_likely_alias(tc)
        print(f"  {'✓' if flag else '✗'}  '{tc}'")

    print(
        "\nNote: DB-dependent lookups require a live PostgreSQL connection.\n"
        "Run with a real session to test lookup_alias_exact / lookup_alias_fuzzy."
    )

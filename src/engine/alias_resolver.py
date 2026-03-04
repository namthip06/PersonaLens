"""
src/engine/alias_resolver.py
============================
Step 2.2: Semantic Alias Resolution — The "Alias Bridge"

Resolves entity surface forms (from ``extract_entities_with_slm``) to their
canonical identities using the local SQLite ``entities`` / ``aliases`` tables
before escalating to external APIs.

Resolution strategy (in order of preference)
--------------------------------------------
1. Exact match  – case-insensitive lookup on ``aliases.alias_text``
2. Fuzzy match  – Levenshtein ratio ≥ threshold via rapidfuzz

Both lookups call ``Database`` methods directly — no SQLAlchemy, no ORM.

Public API
----------
    lookup_alias_exact(surface_form, db)            → tuple | None
    lookup_alias_fuzzy(surface_form, db, threshold) → tuple | None
    resolve_from_db(entity, db)                     → ResolvedEntity | None

Returns ``None`` when no match is found (caller should escalate to Step 2.3).
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database import Database  # noqa: E402
from src.schemas.inference import (  # noqa: E402
    ExtractedEntity,
    ResolvedEntity,
)

if TYPE_CHECKING:
    pass

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
    Heuristic: returns True if *surface_form* looks like a nickname or informal
    alias that warrants a deeper DB lookup log message.

    Used only for verbosity control – all entities still go through the full
    lookup regardless of this flag.
    """
    sf = surface_form.strip()
    return any(sf.startswith(p) for p in NICKNAME_PREFIXES)


# ---------------------------------------------------------------------------
# Step 2.2a – Exact-match lookup
# ---------------------------------------------------------------------------


def lookup_alias_exact(
    surface_form: str,
    db: Database,
) -> tuple[uuid.UUID, str] | None:
    """
    Query the ``aliases`` table for an exact (case-insensitive) match.

    Delegates to ``Database.find_alias_exact()`` which executes:

        SELECT a.entity_id, e.canonical_name
        FROM   aliases a JOIN entities e USING (entity_id)
        WHERE  lower(a.alias_text) = lower(?)

    Parameters
    ----------
    surface_form : The entity text span to look up (e.g. ``"อนุทิน"``).
    db           : An open ``Database`` instance.

    Returns
    -------
    ``(entity_id, canonical_name)`` UUID + string tuple on match, else ``None``.
    """
    try:
        row = db.find_alias_exact(surface_form)
        if row:
            entity_uuid = uuid.UUID(row["entity_id"])
            logger.debug(
                "alias_exact hit: '%s' → entity_id=%s  canonical='%s'",
                surface_form,
                entity_uuid,
                row["canonical_name"],
            )
            return entity_uuid, row["canonical_name"]
        return None
    except Exception as exc:
        logger.warning("alias_exact lookup failed for '%s': %s", surface_form, exc)
        return None


# ---------------------------------------------------------------------------
# Step 2.2b – Fuzzy-match lookup
# ---------------------------------------------------------------------------


def lookup_alias_fuzzy(
    surface_form: str,
    db: Database,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> tuple[uuid.UUID, str, float] | None:
    """
    Load all alias texts from the DB and find the closest match using
    Levenshtein ratio (rapidfuzz).  Returns a hit only if the score meets
    *threshold*.

    Strategy
    --------
    * Calls ``Database.find_all_aliases_with_entities()`` to load the full
      alias table in one query.  For typical Thai political corpora this is
      O(hundreds) of rows — fast in SQLite, acceptable to keep in memory.
    * Uses ``rapidfuzz.process.extractOne`` with ``token_sort_ratio`` for
      better handling of word-order differences common in Thai.

    Parameters
    ----------
    surface_form : Entity text span to compare.
    db           : An open ``Database`` instance.
    threshold    : Minimum similarity ratio (default 0.88, i.e. 88 / 100).

    Returns
    -------
    ``(entity_id, canonical_name, score)`` on match, else ``None``.
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        logger.error("rapidfuzz not installed – fuzzy matching unavailable")
        return None

    try:
        rows = db.find_all_aliases_with_entities()
        if not rows:
            return None

        # Build lookup dict: alias_text → (entity_id UUID, canonical_name)
        choices: dict[str, tuple[uuid.UUID, str]] = {}
        for row in rows:
            choices[row["alias_text"]] = (
                uuid.UUID(row["entity_id"]),
                row["canonical_name"],
            )

        # token_sort_ratio handles word-order differences well for Thai names
        match = process.extractOne(
            surface_form.strip(),
            choices.keys(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold * 100,  # rapidfuzz uses 0–100 scale
        )

        if match:
            matched_text, score, _ = match
            entity_uuid, canonical_name = choices[matched_text]
            normalized_score = round(score / 100.0, 4)
            logger.debug(
                "alias_fuzzy hit: '%s' → '%s'  canonical='%s'  score=%.4f",
                surface_form,
                matched_text,
                canonical_name,
                normalized_score,
            )
            return entity_uuid, canonical_name, normalized_score

        return None
    except Exception as exc:
        logger.warning("alias_fuzzy lookup failed for '%s': %s", surface_form, exc)
        return None


# ---------------------------------------------------------------------------
# Step 2.2 – Main resolver
# ---------------------------------------------------------------------------


def resolve_from_db(
    entity: ExtractedEntity,
    db: Database,
) -> ResolvedEntity | None:
    """
    Orchestrate the two-stage DB alias resolution for a single extracted entity.

    Resolution order
    ----------------
    1. Exact case-insensitive match  → confidence 1.0,   method ``"alias_exact"``
    2. Fuzzy (Levenshtein) match     → confidence = score, method ``"alias_fuzzy"``
    3. No match                      → return ``None`` (caller escalates to Step 2.3)

    Parameters
    ----------
    entity : ``ExtractedEntity`` from the NER step (``extract_entities_with_slm``).
    db     : An open ``Database`` instance.

    Returns
    -------
    ``ResolvedEntity`` if a match was found, else ``None``.
    """
    sf = entity.surface_form

    if is_likely_alias(sf):
        logger.info("'%s' flagged as likely alias – triggering deep alias check", sf)

    # ── Stage 1: exact match ──────────────────────────────────────────────────
    exact = lookup_alias_exact(sf, db)
    if exact:
        entity_id, canonical_name = exact
        logger.info(
            "  ✓ alias_exact: '%s' → '%s'  (entity_id=%s)",
            sf,
            canonical_name,
            entity_id,
        )
        return ResolvedEntity(
            surface_form=sf,
            entity_type=entity.entity_type,
            global_id=entity_id,
            canonical_name=canonical_name,
            confidence_score=1.0,
            resolution_method="alias_exact",
        )

    # ── Stage 2: fuzzy match ──────────────────────────────────────────────────
    fuzzy = lookup_alias_fuzzy(sf, db)
    if fuzzy:
        entity_id, canonical_name, score = fuzzy
        logger.info(
            "  ~ alias_fuzzy: '%s' → '%s'  score=%.4f  (entity_id=%s)",
            sf,
            canonical_name,
            score,
            entity_id,
        )
        return ResolvedEntity(
            surface_form=sf,
            entity_type=entity.entity_type,
            global_id=entity_id,
            canonical_name=canonical_name,
            confidence_score=score,
            resolution_method="alias_fuzzy",
        )

    logger.info("  ✗ no DB alias for '%s' – escalating to external validation", sf)
    return None


# ---------------------------------------------------------------------------
# Smoke test – `uv run src/engine/alias_resolver.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging
    from src.schemas.inference import EntityType, ExtractedEntity

    # 1. Setup Logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    )

    DB_PATH = project_root / "database" / "personalens.db"

    print("\n" + "=" * 80)
    print(f"{'PersonaLens Alias Resolver Pipeline Test':^80}")
    print("=" * 80)

    # ── Heuristic check ───────────────────────────────────────────────────────
    test_cases = [
        "เสี่ยหนู",  # should flag as likely alias
        "Anutin Charnvirakul",  # formal name – no flag
        "ภาคเหนือ",  # LOC – no flag
        "พรรคภูมิใจไทย",  # ORG – no flag
    ]
    print("\n🚀 Running Heuristic trigger check (is_likely_alias)...")
    for tc in test_cases:
        flag = is_likely_alias(tc)
        print(f"  {'✅' if flag else '❌'}  [{tc}]")

    print("\n🚀 Running [Step 2.2]: DB-backed Entity Resolution...")

    with Database(str(DB_PATH)) as db:
        # Seed a known entity + several aliases for testing
        eid = db.upsert_entity("Anutin Charnvirakul", "PER", "th")
        for alias in ("อนุทิน", "เสี่ยหนู", "Anutin", "อนุทิน ชาญวีรกูล"):
            db.upsert_alias(eid, alias, source_type="manual")

        beid = db.upsert_entity("Bhumjaithai Party", "ORG", "th")
        for alias in ("พรรคภูมิใจไทย", "ภูมิใจไทย", "BJT"):
            db.upsert_alias(beid, alias, source_type="manual")

        # Test surfaces to resolve
        surfaces = [
            ("อนุทิน", EntityType.PER),  # exact hit expected
            ("เสี่ยหนู", EntityType.PER),  # exact hit (alias)
            ("Anutn", EntityType.PER),  # fuzzy hit expected (typo)
            ("พรรคภูมิใจไทย", EntityType.ORG),  # exact hit
            ("ภูมิใจไท", EntityType.ORG),  # fuzzy hit (truncated)
            ("สมศักดิ์", EntityType.PER),  # no match → None expected
        ]

        print("\n" + "-" * 80)
        print(f"{'FINAL RESOLVED ENTITIES (Step 2.2)':^80}")
        print("-" * 80)

        for idx, (sf, etype) in enumerate(surfaces, 1):
            entity = ExtractedEntity(
                surface_form=sf,
                entity_type=etype,
                context_clue="smoke test",
            )
            result = resolve_from_db(entity, db)
            if result:
                status = "✅" if result.is_resolved else "❓"
                print(
                    f"{idx}. {status} [{result.surface_form}] -> {result.canonical_name or 'Unresolved'}"
                )
                print(
                    f"   Method: {result.resolution_method} | Confidence: {result.confidence_score:.2f}"
                )
                print(f"   ID:     {result.global_id}\n")
            else:
                print(f"{idx}. ❓ [{sf}] -> Unresolved (Not found in DB)")
                print("   Escalating to Step 2.3...\n")

        print("\n" + "=" * 80)
        print("Smoke test complete.")

"""
src/engine/disambiguator.py
===========================
Step 2.4: Disambiguation & Global ID Assignment

This module is the final arbiter in the entity resolution pipeline.
It orchestrates Steps 2.2 → 2.3 sequentially, and handles homonym
disambiguation (two entities with identical names) via keyword scoring.

Public function
---------------
    assign_global_id(entity, article_context, session, slm_client, article_date)
        → ResolvedEntity

Pipeline
--------
  ExtractedEntity
      │
      ▼
  resolve_from_db()      ← Step 2.2: alias exact + fuzzy (alias_resolver.py)
      │   ─ Hit: return with method="alias_exact" or "alias_fuzzy"
      │   ─ Miss: ↓
      ▼
  validate_entity_external()  ← Step 2.3: DDG + SLM (external_validator.py)
      │   ─ Hit: return with method="external_api"
      │   ─ Miss or homonyms: ↓
      ▼
  score_candidates()     ← Step 2.4: keyword overlap scoring (this file)
      │
      ▼
  ResolvedEntity (global_id assigned or method="unresolved")
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
from src.engine.alias_resolver import resolve_from_db  # noqa: E402
from src.engine.external_validator import validate_entity_external  # noqa: E402

if TYPE_CHECKING:
    import uuid
    from sqlalchemy.orm import Session
    from src.engine.slm_client import SLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context keyword sets for Thai politics (used in homonym scoring)
# ---------------------------------------------------------------------------

# These keywords shift the probability toward a specific known entity.
# Format: { context_keyword_substring: entity_canonical_name_fragment }
CONTEXT_WEIGHT_HINTS: list[tuple[str, str]] = [
    ("ภูมิใจไทย", "Anutin"),  # Bhumjaithai Party → Anutin
    ("Bhumjaithai", "Anutin"),
    ("บุรีรัมย์", "Anutin"),  # Buriram Province → Anutin
    ("มหาดไทย", "Anutin"),  # Interior Ministry → Anutin
    ("เพื่อไทย", "Thaksin"),  # Pheu Thai Party → Thaksin family
    ("ประชาธิปัตย์", "Abhisit"),  # Democrat Party → Abhisit
    ("รวมไทยสร้างชาติ", "Prayuth"),  # UTNP → Prayuth
]


# ---------------------------------------------------------------------------
# Step 2.4a – Candidate scoring for homonyms
# ---------------------------------------------------------------------------


def score_candidates(
    candidates: list[tuple["uuid.UUID", str]],
    context_text: str,
) -> list[tuple["uuid.UUID", str, float]]:
    """
    Score a list of candidate entities against the article context.
    Used when multiple entities share the same canonical name (homonyms).

    Scoring strategy
    ----------------
    Each candidate starts with a base score of 0.5.
    For every CONTEXT_WEIGHT_HINTS match found in `context_text`, the
    matching candidate gains +0.2 (capped at 1.0).
    Non-matching candidates lose -0.1 per conflicting hint.

    Parameters
    ----------
    candidates   : List of (entity_id, canonical_name) tuples to rank.
    context_text : Full article text or context clue for keyword scanning.

    Returns
    -------
    Sorted list of (entity_id, canonical_name, score) descending by score.
    """
    scores: dict[str, float] = {eid_name[1]: 0.5 for eid_name in candidates}
    id_map: dict[str, "uuid.UUID"] = {name: eid for eid, name in candidates}

    for keyword, name_hint in CONTEXT_WEIGHT_HINTS:
        if keyword in context_text:
            for name in scores:
                if name_hint.lower() in name.lower():
                    scores[name] = min(1.0, scores[name] + 0.2)
                    logger.debug(
                        "Context keyword '%s' → boosting '%s' score", keyword, name
                    )

    result = [(id_map[name], name, round(score, 4)) for name, score in scores.items()]
    result.sort(key=lambda x: x[2], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Step 2.4b – Fetch homonym candidates from DB
# ---------------------------------------------------------------------------


def _fetch_homonym_candidates(
    canonical_name: str,
    session: "Session",
) -> list[tuple["uuid.UUID", str]]:
    """
    Query the entities table for all rows with the same canonical_name.
    Returns a list of (entity_id, canonical_name) tuples.
    """
    try:
        from database.models import Entity

        rows = (
            session.query(Entity.entity_id, Entity.canonical_name)
            .filter(Entity.canonical_name == canonical_name)
            .all()
        )
        return [(row.entity_id, row.canonical_name) for row in rows]
    except Exception as exc:
        logger.warning("Homonym candidate fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Step 2.4 – Main orchestrator: assign_global_id
# ---------------------------------------------------------------------------


def assign_global_id(
    entity: ExtractedEntity,
    article_context: str = "",
    session: "Session | None" = None,
    slm_client: "SLMClient | None" = None,
    article_date: str = "",
) -> ResolvedEntity:
    """
    Full Step 2.2 → 2.3 → 2.4 orchestration for a single entity.

    Tries each resolution strategy in order:
    1. DB alias exact match (fast)
    2. DB alias fuzzy match (slightly slower)
    3. DuckDuckGo + SLM external validation (slow, network call)
    4. Contextual scoring for homonyms if Step 3 returned ambiguous results
    5. Return "unresolved" if all strategies fail

    Parameters
    ----------
    entity          : ExtractedEntity from NER Step 2.1.
    article_context : Full article body (used for homonym keyword scoring).
    session         : SQLAlchemy session (required for DB-backed steps).
    slm_client      : SLMClient instance (required for Steps 2.3).
    article_date    : ISO date string for ROLE time-sensitive resolution.

    Returns
    -------
    ResolvedEntity – always returns a value; method="unresolved" if no match.
    """
    sf = entity.surface_form
    logger.info("assign_global_id: processing '%s' (%s)", sf, entity.entity_type.value)

    # ── Step 2.2: DB alias resolution ────────────────────────────────────────
    if session is not None:
        db_result = resolve_from_db(entity, session)
        if db_result is not None:
            logger.info(
                "'%s' resolved via %s → '%s' (id=%s)",
                sf,
                db_result.resolution_method,
                db_result.canonical_name,
                db_result.global_id,
            )
            return db_result
    else:
        logger.warning("No DB session provided – skipping Step 2.2 alias lookup")

    # ── Step 2.3: External DuckDuckGo + SLM validation ───────────────────────
    if slm_client is not None:
        ext_result = validate_entity_external(
            entity=entity,
            slm_client=slm_client,
            session=session,
            article_date=article_date,
        )

        if ext_result is not None:
            # ── Step 2.4: Homonym disambiguation ─────────────────────────────
            # If multiple entities in DB share the same canonical_name, use
            # context keyword scoring to pick the most probable one.
            if session is not None and ext_result.canonical_name:
                candidates = _fetch_homonym_candidates(
                    ext_result.canonical_name, session
                )
                if len(candidates) > 1:
                    logger.info(
                        "Homonym detected for '%s': %d candidates – scoring",
                        ext_result.canonical_name,
                        len(candidates),
                    )
                    scored = score_candidates(candidates, article_context)
                    best_id, best_name, best_score = scored[0]
                    logger.info(
                        "Homonym resolved: '%s' (id=%s, score=%.3f)",
                        best_name,
                        best_id,
                        best_score,
                    )
                    return ResolvedEntity(
                        surface_form=sf,
                        entity_type=entity.entity_type,
                        global_id=best_id,
                        canonical_name=best_name,
                        confidence_score=round(
                            ext_result.confidence_score * best_score, 4
                        ),
                        resolution_method="external_api",
                    )

            return ext_result
    else:
        logger.warning("No SLMClient provided – skipping Step 2.3 external validation")

    # ── Unresolved fallback ───────────────────────────────────────────────────
    logger.warning(
        "'%s' (%s) could not be resolved – returning unresolved",
        sf,
        entity.entity_type.value,
    )
    return ResolvedEntity(
        surface_form=sf,
        entity_type=entity.entity_type,
        global_id=None,
        canonical_name=None,
        confidence_score=0.0,
        resolution_method="unresolved",
    )


# ---------------------------------------------------------------------------
# Standalone smoke test – `uv run src/engine/disambiguator.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    print("=" * 60)
    print("PersonaLens – Disambiguator Smoke Test")
    print("=" * 60)

    # --- Test homonym scoring -----------------------------------------------------------
    import uuid as _uuid

    mock_candidates = [
        (_uuid.UUID("00000000-0000-0000-0000-000000000001"), "Anutin Charnvirakul"),
        (
            _uuid.UUID("00000000-0000-0000-0000-000000000002"),
            "Anutin Somchai",
        ),  # fictional
    ]
    context_with_bhumjaithai = "เสี่ยหนู หัวหน้าพรรคภูมิใจไทย ลงพื้นที่บุรีรัมย์"
    context_without = "เสี่ยหนู ลงพื้นที่"

    print("\nHomonym scoring test:")
    print(f"  Context: '{context_with_bhumjaithai}'")
    scored = score_candidates(mock_candidates, context_with_bhumjaithai)
    for eid, name, score in scored:
        print(f"    {score:.3f}  {name}")

    print(f"\n  Context: '{context_without}'")
    scored = score_candidates(mock_candidates, context_without)
    for eid, name, score in scored:
        print(f"    {score:.3f}  {name}")

    print(
        "\nNote: Full resolution pipeline requires a DB session + Ollama.\n"
        "Run `uv run src/engine/entity_linker.py` for the end-to-end smoke test."
    )

"""
src/engine/external_validator.py
================================
Step 2.3: External Identity Validation

When the local alias DB has no match, this module queries DuckDuckGo (via
DDGS text search) and feeds the top snippets to the SLM for resolution.

Public function
---------------
    validate_entity_external(entity, model_name, session=None, article_date=None)
        → ResolvedEntity | None

Pipeline (per entity)
---------------------
  ExtractedEntity (unresolved)
      │
      ▼
  build_query(surface_form, entity_type)  ← category-aware, hard-coded English query
      │
      ▼
  search_ddgs(query)  ← DDGS text search (duckduckgo-search library)
      │
      ▼
  validate_with_slm(snippets, ..., model_name)  ← SLM reads snippets, returns canonical name
      │   schema: ExternalResolutionOutput
      ▼
  write new Alias row to DB (if session provided and confidence ≥ threshold)
      │
      ▼
  ResolvedEntity (method="external_api")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from ddgs import DDGS
from langdetect import detect

import uuid

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.schemas.inference import (  # noqa: E402
    ExtractedEntity,
    EntityType,
    ExternalResolutionOutput,
    ResolvedEntity,
)
from src.utils.prompts import load_prompt  # noqa: E402
from src.engine.slm_client import SLMClient  # noqa: E402
from database.models import Alias, Entity  # noqa: E402

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_RESULTS = 5
CONFIDENCE_WRITE_THRESHOLD = 0.75  # min confidence to persist a new alias to DB

# Mapping: ISO 639-1 language code → DDGS region code
_LANG_TO_REGION: dict[str, str] = {
    "en": "us-en",
    "th": "th-th",
    "zh-cn": "cn-zh",
    "zh-tw": "tw-zh",
    "ja": "jp-ja",
    "ko": "kr-ko",
    "de": "de-de",
    "fr": "fr-fr",
    "es": "es-es",
    "pt": "br-pt",
    "ru": "ru-ru",
    "ar": "xa-ar",
    "id": "id-id",
    "vi": "vn-vi",
}

# Mapping: entity_type → prompt template name (matches YAML filenames)
PROMPT_MAP: dict[str, str] = {
    EntityType.PER: "snippet_analysis_per",
    EntityType.ORG: "snippet_analysis_org",
    EntityType.LOC: "snippet_analysis_loc",
    EntityType.GPE: "snippet_analysis_gpe",
}


# ---------------------------------------------------------------------------
# Step 2.3a – Category-aware search query builder
# ---------------------------------------------------------------------------


def build_query(
    surface_form: str,
    entity_type: EntityType,
) -> str:
    """
    Build a globalized search query to find the canonical name/identity.

    Strategy: Focus on "identity definition" rather than geographic location.

    Parameters
    ----------
    surface_form : The raw text span from the article.
    entity_type  : PER | ORG | LOC | GPE

    Returns
    -------
    A concise English search query string for that entity category.
    """
    sf = surface_form.strip()

    if entity_type == EntityType.PER:
        return f"who is {sf} full name and identity"
    elif entity_type == EntityType.ORG:
        return f"what is {sf} official name and organization details"
    elif entity_type == EntityType.LOC or entity_type == EntityType.GPE:
        return f"where is {sf} and its full official name"

    return f"what is {sf} and its actual name"


# ---------------------------------------------------------------------------
# Internal helper – detect DDGS region from query text
# ---------------------------------------------------------------------------


def _detect_region(text: str) -> str:
    """
    Use langdetect to pick the best DDGS region code for the given text.
    Falls back to 'wt-wt' (worldwide) if detection fails.
    """
    try:
        lang = detect(text)
        return _LANG_TO_REGION.get(lang, "wt-wt")
    except Exception as exc:
        logger.debug("langdetect failed for %r: %s – defaulting to wt-wt", text, exc)
        return "wt-wt"


# ---------------------------------------------------------------------------
# Step 2.3b – DDGS text search
# ---------------------------------------------------------------------------


def search_ddgs(
    query: str,
    surface_form: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[str]:
    """
    Query DDGS text search and return a list of text snippets.

    The region is auto-detected from the surface_form language using langdetect.

    Parameters
    ----------
    query        : Search query string.
    surface_form : Original entity text span (used for region detection).
    max_results  : Maximum number of snippet strings to return.

    Returns
    -------
    List of snippet strings (may be empty if no results or request fails).
    """
    try:
        region = _detect_region(surface_form)
        logger.info("DDGS search: %r  region=%s", query, region)

        results = DDGS().text(
            query,
            region=region,
            safesearch="moderate",
            timelimit=None,
            max_results=max_results,
            page=1,
            backend=[
                "wikipedia",
                "google",
                "duckduckgo",
                "brave",
            ],
        )

        snippets: list[str] = []
        for r in results or []:
            body = r.get("body", "").strip()
            if body:
                snippets.append(body)

        logger.info("DDGS returned %d snippet(s)", len(snippets))
        return snippets

    except Exception as exc:
        logger.warning("DDGS search failed for %r: %s", query, exc)
        return []


# ---------------------------------------------------------------------------
# Step 2.3c – SLM snippet analysis
# ---------------------------------------------------------------------------


def validate_with_slm(
    snippets: list[str],
    surface_form: str,
    entity_type: EntityType,
    context_clue: str,
    model_name: str,
    article_date: str = "",
) -> ExternalResolutionOutput | None:
    """
    Send search snippets to the SLM for identity confirmation.
    Uses a category-specific prompt template (per / org / loc / gpe).

    A fresh SLMClient is instantiated inside this function using `model_name`.

    Parameters
    ----------
    snippets      : Text snippets from DDGS.
    surface_form  : Original entity text span.
    entity_type   : PER | ORG | LOC | GPE – selects the right prompt template.
    context_clue  : Short context clue from NER.
    model_name    : Ollama model tag, e.g. "qwen2.5:7b".
    article_date  : Optional ISO date (passed to prompt but not used by LOC/GPE).

    Returns
    -------
    ExternalResolutionOutput with canonical_name + confidence, or None on error.
    """

    if not snippets:
        logger.warning(
            "No snippets to analyse for '%s' – skipping SLM call", surface_form
        )
        return None

    prompt_id = PROMPT_MAP.get(entity_type, "snippet_analysis_per")
    print("prompt_id :", prompt_id)

    try:
        prompt_data = load_prompt(prompt_id)
    except FileNotFoundError as exc:
        logger.error("Prompt template not found: %s", exc)
        return None

    system_prompt: str = prompt_data["templates"]["system"]

    user_template: str = prompt_data["templates"]["user"]
    snippets_text = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets))

    try:
        user_prompt = user_template.format(
            surface_form=surface_form,
            context_clue=context_clue,
            snippets=snippets_text,
            article_date=article_date or "unknown",
        )
    except KeyError as exc:
        logger.error("Prompt template format error – missing key %s", exc)
        return None

    slm_client = SLMClient(model=model_name)

    try:
        result, elapsed_ms = slm_client.chat_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=ExternalResolutionOutput,
        )
        logger.info(
            "SLM snippet analysis: '%s' → '%s' (confidence=%.2f, latency=%dms)",
            surface_form,
            result.canonical_name,
            result.confidence,
            elapsed_ms,
        )
        return result

    except Exception as exc:
        logger.error("SLM snippet analysis failed for '%s': %s", surface_form, exc)
        return None


# ---------------------------------------------------------------------------
# Step 2.3 – Top-level external validator
# ---------------------------------------------------------------------------


def validate_entity_external(
    entity: ExtractedEntity,
    model_name: str,
    session: "Session | None" = None,
    article_date: str = "",
) -> ResolvedEntity | None:
    """
    Full Step 2.3 pipeline for a single entity:
    build query → DDGS search → SLM analysis → optional DB write.

    If resolution confidence meets CONFIDENCE_WRITE_THRESHOLD and a DB
    session is provided, a new `Alias` row is written so subsequent
    occurrences hit the faster alias_exact path (Step 2.2).

    Parameters
    ----------
    entity        : Unresolved ExtractedEntity from Step 2.1.
    model_name    : Ollama model tag (e.g. "qwen2.5:7b").
    session       : Optional SQLAlchemy session for DB writes.
    article_date  : ISO-format date string (forwarded to the prompt template).

    Returns
    -------
    ResolvedEntity (method="external_api") on success, None if unresolvable.
    """
    sf = entity.surface_form
    query = build_query(sf, entity.entity_type)
    snippets = search_ddgs(query, sf)

    resolution = validate_with_slm(
        snippets=snippets,
        surface_form=sf,
        entity_type=entity.entity_type,
        context_clue=entity.context_clue,
        model_name=model_name,
        article_date=article_date,
    )

    if resolution is None or resolution.confidence < 0.5:
        logger.warning(
            "External validation inconclusive for '%s' (confidence=%.2f)",
            sf,
            resolution.confidence if resolution else 0.0,
        )
        return None

    # Attempt to find or create the entity in the DB
    global_id = None
    if session is not None:
        global_id = _upsert_entity_and_alias(
            session=session,
            canonical_name=resolution.canonical_name,
            wikidata_id=resolution.wikidata_id,
            entity_type=entity.entity_type,
            surface_form=sf,
            confidence=resolution.confidence,
        )

    return ResolvedEntity(
        surface_form=sf,
        entity_type=entity.entity_type,
        global_id=global_id,
        canonical_name=resolution.canonical_name,
        confidence_score=round(resolution.confidence, 4),
        resolution_method="external_api",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _upsert_entity_and_alias(
    session: "Session",
    canonical_name: str,
    wikidata_id: str | None,
    entity_type: EntityType,
    surface_form: str,
    confidence: float,
) -> "uuid.UUID | None":
    """
    Find or create the Entity row and write a new Alias row for the surface form.
    Returns the entity_id (Global ID) on success, None on DB error.
    """
    try:
        # --- Find existing entity ---
        entity_row = None

        if wikidata_id:
            entity_row = (
                session.query(Entity).filter(Entity.wikidata_id == wikidata_id).first()
            )

        if entity_row is None:
            entity_row = (
                session.query(Entity)
                .filter(Entity.canonical_name == canonical_name)
                .first()
            )

        # --- Create if not found ---
        if entity_row is None:
            cat_map = {
                EntityType.PER: "politician",
                EntityType.ORG: "organization",
                EntityType.LOC: "location",
                EntityType.GPE: "geopolitical",
            }
            entity_row = Entity(
                entity_id=uuid.uuid4(),
                canonical_name=canonical_name,
                category=cat_map.get(entity_type, "unknown"),
                wikidata_id=wikidata_id or None,
                lang="th",
            )
            session.add(entity_row)
            session.flush()  # get entity_id without committing
            logger.info(
                "Created new entity: '%s' (id=%s)", canonical_name, entity_row.entity_id
            )

        # --- Write alias row if confidence is sufficient ---
        if confidence >= CONFIDENCE_WRITE_THRESHOLD:
            existing_alias = (
                session.query(Alias)
                .filter(
                    Alias.entity_id == entity_row.entity_id,
                    Alias.alias_text == surface_form,
                )
                .first()
            )
            if existing_alias is None:
                new_alias = Alias(
                    entity_id=entity_row.entity_id,
                    alias_text=surface_form,
                    source_type="external_api",
                )
                session.add(new_alias)
                session.flush()
                logger.info(
                    "Persisted new alias: '%s' → entity '%s'",
                    surface_form,
                    canonical_name,
                )

        session.commit()
        return entity_row.entity_id

    except Exception as exc:
        logger.error("DB upsert failed for '%s': %s", canonical_name, exc)
        if session:
            session.rollback()
        return None


# ---------------------------------------------------------------------------
# Standalone smoke test – `uv run src/engine/external_validator.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    MODEL = "qwen2.5:7b"

    # ------------------------------------------------------------------
    # validate_with_slm – mock entity data
    # ------------------------------------------------------------------
    print("\n[3] validate_with_slm with mock entities (requires Ollama running):")

    mock_entities_raw = [
        {
            "surface_form": "อนุทิน ชาญวีรกูล",
            "entity_type": "PER",
            "context_clue": "นายอนุทิน ชาญวีรกูล",
        },
        {
            "surface_form": "เสี่ยหนู",
            "entity_type": "PER",
            "context_clue": "หรือ เสี่ยหนู",
        },
        {
            "surface_form": "พรรคภูมิใจไทย",
            "entity_type": "ORG",
            "context_clue": "พรรคภูมิใจไทย",
        },
        {
            "surface_form": "ภาคเหนือ",
            "entity_type": "LOC",
            "context_clue": "ลงพื้นที่ตรวจสอบสถานการณ์น้ำท่วมในภาคเหนือ",
        },
    ]

    mock_entities: list[ExtractedEntity] = [
        ExtractedEntity(**e) for e in mock_entities_raw
    ]

    for entity in mock_entities:
        print(f"\n  ── Entity: '{entity.surface_form}' [{entity.entity_type.value}]")
        query = build_query(entity.surface_form, entity.entity_type)
        print(f"     Query   : {query!r}")

        snippets = search_ddgs(query, entity.surface_form)
        print(f"     Snippets: {len(snippets)} retrieved")

        if not snippets:
            print("     Result  : skipped (no snippets)")
            continue
        else:
            for i, snippet in enumerate(snippets, 1):
                print(f"- Snippet {i}: {snippet}")

        result = validate_with_slm(
            snippets=snippets,
            surface_form=entity.surface_form,
            entity_type=entity.entity_type,
            context_clue=entity.context_clue,
            model_name=MODEL,
            article_date="2025-10-01",
        )

        if result:
            print(f"     Result  : {result}")
        else:
            print("     Result  : None (SLM unavailable or inconclusive)")

    print("\n" + "=" * 60)
    print("Smoke test complete.")

"""
src/engine/entity_linker.py
===========================
Step 2: Entity Intelligence – Alias Resolution & Mapping

Public functions
----------------
    extract_entities_with_slm(text, model_name, prompt_id)
        → NERInferenceResult | None

    resolve_all_entities(ner_result, article_context, session, slm_client)
        → EntityLinkerResult

Pipeline
--------
  Article text
      │
      ▼
  [Step 2.1] extract_entities_with_slm()
      │   load_prompt → SLMClient.chat_structured() → NERInferenceResult
      │
      ▼
  [Step 2.2] alias_resolver.resolve_from_db()    ← exact + fuzzy DB lookup
      │
      ▼
  [Step 2.3] external_validator.validate_entity_external()  ← DDG + SLM
      │
      ▼
  EntityLinkerResult  ← envelope: list[ResolvedEntity] + metadata
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.engine.slm_client import SLMClient, SLMInferenceError  # noqa: E402
from src.schemas.inference import (  # noqa: E402
    EntityLinkerResult,
    InferenceMetadata,
    NERInferenceResult,
    NEROutput,
    ResolvedEntity,
)
from src.utils.prompts import load_prompt  # noqa: E402
from src.engine import alias_resolver, external_validator  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


def extract_entities_with_slm(
    text: str,
    model_name: str = "qwen2.5:7b",
    prompt_id: str = "ner_v1",
) -> NERInferenceResult | None:
    """
    Send article text to the SLM via Ollama and return a validated
    NERInferenceResult containing all extracted entities.

    Parameters
    ----------
    text       : Raw article body (or headline) to analyse.
    model_name : Ollama model tag (must be pulled locally).
    prompt_id  : Filename (without .yaml) of the prompt template in
                 src/utils/prompts/.

    Returns
    -------
    NERInferenceResult on success, None on unrecoverable error.

    Output shape
    ------------
    {
      "data": {
        "entities": [
          { "surface_form": "เสี่ยนู",        "entity_type": "PER",  "context_clue": "..." },
          { "surface_form": "พรรคภูมิใจไทย", "entity_type": "ORG",  "context_clue": "..." }
        ]
      },
      "metadata": {
        "prompt_id":   "ner-v1.0-qwen2.5:7b",
        "model":       "qwen2.5:7b",
        "duration_ms": 1240
      }
    }
    """
    # 1. Load the YAML prompt template
    try:
        prompt_data = load_prompt(prompt_id)
    except FileNotFoundError as exc:
        logger.error("Prompt template not found: %s", exc)
        return None

    system_prompt: str = prompt_data["templates"]["system"]
    user_prompt: str = prompt_data["templates"]["user"].format(text=text)
    prompt_yaml_id: str = prompt_data["id"]

    # 2. Call Ollama with structured output (Pydantic schema constrains format)
    client = SLMClient(model=model_name)

    try:
        ner_output, elapsed_ms = client.chat_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=NEROutput,  # ← constrain + validate output
        )
    except SLMInferenceError as exc:
        logger.error("NER inference failed: %s", exc)
        return None

    # 3. Wrap in the result envelope with provenance metadata
    result = NERInferenceResult(
        data=ner_output,
        metadata=InferenceMetadata(
            prompt_id=prompt_yaml_id,
            model=model_name,
            duration_ms=elapsed_ms,
        ),
    )

    logger.info(
        "Extracted %d entities from text (len=%d) in %dms",
        len(result.data.entities),
        len(text),
        elapsed_ms,
    )
    return result


# ---------------------------------------------------------------------------
# Step 2.2 → 2.4: resolve_all_entities
# ---------------------------------------------------------------------------


def resolve_all_entities(
    ner_result: NERInferenceResult,
    article_context: str = "",
    session=None,
    model_name: str = "qwen2.5:7b",
    article_date: str = "",
) -> EntityLinkerResult:
    """
    Run Steps 2.2 → 2.4 on every entity extracted by the NER step.

    For each ExtractedEntity in `ner_result`, calls:
      1. alias_resolver.resolve_from_db()            (Step 2.2)
      2. external_validator.validate_entity_external() (Step 2.3)

    Parameters
    ----------
    ner_result       : Output of `extract_entities_with_slm()` (Step 2.1).
    article_context  : Full article body – used for homonym keyword scoring.
    session          : SQLAlchemy Session for DB alias lookup & writes.
                       Pass None to skip DB steps (useful for unit tests).
    model_name       : Ollama model tag used to instantiate SLMClient internally
                       for Step 2.3 external validation (default: "qwen2.5:7b").
    article_date     : ISO-format date string for ROLE time-sensitive lookups.

    Returns
    -------
    EntityLinkerResult containing a list[ResolvedEntity] + provenance metadata.

    Output shape
    ------------
    {
      "resolved": [
        {
          "surface_form": "เสี่ยหนู",
          "entity_type": "PER",
          "global_id": "a1b2c3d4-...",
          "canonical_name": "Anutin Charnvirakul",
          "confidence_score": 0.92,
          "resolution_method": "external_api"
        },
        ...
      ],
      "metadata": { "prompt_id": "ner-v1.0-qwen2.5:7b", "model": "qwen2.5:7b", ... }
    }
    """
    import time

    t0 = time.monotonic()

    resolved: list[ResolvedEntity] = []
    entities = ner_result.data.entities

    logger.info(
        "resolve_all_entities: resolving %d entities (DB=%s, model=%s)",
        len(entities),
        "yes" if session else "no",
        model_name,
    )

    for entity in entities:
        res = None

        # Step 2.2: resolve_from_db
        if session:
            res = alias_resolver.resolve_from_db(entity, db=session)

        # Step 2.3: validate_entity_external
        if not res:
            res = external_validator.validate_entity_external(
                entity=entity,
                model_name=model_name,
                db=session,
                article_date=article_date,
            )

        if not res:
            res = ResolvedEntity(
                surface_form=entity.surface_form,
                entity_type=entity.entity_type,
                global_id=None,
                canonical_name=None,
                confidence_score=0.0,
                resolution_method="unresolved",
            )

        resolved.append(res)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Resolution complete: %d/%d resolved in %dms",
        sum(1 for r in resolved if r.is_resolved),
        len(resolved),
        elapsed_ms,
    )

    return EntityLinkerResult(
        resolved=resolved,
        metadata=ner_result.metadata,
    )


# ---------------------------------------------------------------------------
# Test execution – `uv run src/engine/entity_linker.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import logging as _logging
    from database import Database

    # 1. Setup Logging to see the Step 2.1 -> 2.3 transitions
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    )

    DB_PATH = project_root / "database" / "personalens.db"

    # 2. Input Data
    SAMPLE_TEXT = (
        "นายอนุทิน ชาญวีรกูล หรือ เสี่ยหนู รองนายกรัฐมนตรีและรัฐมนตรีว่าการกระทรวงมหาดไทย "
        "ในฐานะหัวหน้าพรรคภูมิใจไทย ลงพื้นที่ตรวจสอบสถานการณ์น้ำท่วมในภาคเหนือ"
    )
    CURRENT_DATE = "2024-05-20"  # Important for Step 2.3 validation
    MODEL = "qwen2.5:7b"

    print("\n" + "=" * 80)
    print(f"{'PersonaLens Entity Linking Pipeline Test':^80}")
    print("=" * 80)
    print(f"Input Text: {SAMPLE_TEXT}\n")

    # ── [STEP 2.1] NER EXTRACTION ──
    print(f"🚀 Running [Step 2.1]: NER Extraction using {MODEL}...")
    ner_result = extract_entities_with_slm(
        text=SAMPLE_TEXT, model_name=MODEL, prompt_id="ner_v1"
    )

    if not ner_result:
        print("❌ NER Step failed. Ensure Ollama is running and model is pulled.")
    else:
        print(f"✅ Found {len(ner_result.data.entities)} entities.")

        # ── [STEP 2.2 & 2.3] ENTITY RESOLUTION ──
        print(f"\n🚀 Running [Step 2.2 -> 2.4]: Entity Resolution...")
        # Note: session=None will skip Step 2.2 (DB) and go straight to Step 2.3 (External/AI)
        with Database(str(DB_PATH)) as db_session:
            final_result = resolve_all_entities(
                ner_result=ner_result,
                article_context=SAMPLE_TEXT,
                session=db_session,
                model_name=MODEL,
                article_date=CURRENT_DATE,
            )

        # ── DISPLAY FINAL RESULTS ──
        print("\n" + "-" * 80)
        print(f"{'FINAL RESOLVED ENTITIES':^80}")
        print("-" * 80)

        for idx, res in enumerate(final_result.resolved, 1):
            status = "✅" if res.is_resolved else "❓"
            print(
                f"{idx}. {status} [{res.surface_form}] -> {res.canonical_name or 'Unresolved'}"
            )
            print(
                f"   Method: {res.resolution_method} | Confidence: {res.confidence_score:.2f}"
            )
            print(f"   ID:     {res.global_id}\n")

        print("Full JSON Output Payload:")
        print(
            json.dumps(
                final_result.model_dump(mode="json"), indent=2, ensure_ascii=False
            )
        )

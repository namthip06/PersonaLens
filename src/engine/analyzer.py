"""
src/engine/analyzer.py
======================
Phase II: Pre-Inference & Unified SLM Analysis

Consumes the output of entity_linker.resolve_all_entities() and performs
Aspect-Based Sentiment Analysis (ABSA) for every resolved entity using a
single, multi-task SLM inference pass per entity.

Pipeline
--------
  EntityLinkerResult (list[ResolvedEntity])
      │
      ▼
  [Step 3.1] chunk_context_window()
      │   Segment article into sentences (Thai: PyThaiNLP, EN: spaCy).
      │   Return the sentence containing the target + the i±1 neighbours.
      │
      ▼
  [Step 3.2] build_anchored_snippet()
      │   Wrap every occurrence of the surface form with <target>…</target>.
      │   Prepend a one-line entity anchor describing the target.
      │
      ▼
  [Step 3.3] analyze_entity()
      │   Load YAML prompt → format user template → SLMClient.chat_structured()
      │   schema=ABSAOutput  (Task A + B + C in one pass).
      │
      ▼
  AnalyzerResult  ← per-entity envelope with validated ABSAOutput + metadata

Public API
----------
    chunk_context_window(text, target_surface, lang) → str
    build_anchored_snippet(window, target_surface, target_description) → str
    analyze_entity(resolved_entity, article_text, slm_client,
                   lang, prompt_id, model) → AnalyzerResult | None
    run_analysis(linker_result, article_text, lang,
                 model_name, prompt_id) → list[AnalyzerResult]
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.engine.slm_client import SLMClient, SLMInferenceError  # noqa: E402
from src.schemas.inference import (  # noqa: E402
    ABSAOutput,
    AnalyzerResult,
    EntityLinkerResult,
    InferenceMetadata,
    ResolvedEntity,
)
from src.utils.prompts import load_prompt  # noqa: E402
from database.database import Database  # noqa: E402

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons for sentence segmenters (loaded only on first use)
# ---------------------------------------------------------------------------

_thai_segmentor = None  # pythainlp.tokenize.thaisumcut.ThaiSentenceSegmentor
_spacy_nlp = None  # spacy Language (en_core_web_sm)


def _get_thai_segmentor():
    """Return (and cache) a ThaiSentenceSegmentor instance."""
    global _thai_segmentor
    if _thai_segmentor is None:
        try:
            from pythainlp.tokenize.thaisumcut import ThaiSentenceSegmentor

            _thai_segmentor = ThaiSentenceSegmentor()
            logger.info("ThaiSentenceSegmentor loaded.")
        except ImportError as exc:
            raise ImportError(
                "PyThaiNLP is required for Thai sentence segmentation. "
                "Install it with: pip install pythainlp"
            ) from exc
    return _thai_segmentor


def _get_spacy_nlp():
    """
    Return (and cache) a spaCy English pipeline (en_core_web_sm).

    On the first call, if ``en_core_web_sm`` is not yet installed the function
    automatically downloads it via ``spacy.cli.download`` (no manual step
    required).  Subsequent calls reuse the cached model.
    """
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for English sentence segmentation. "
                "Install it with: uv pip install spacy"
            ) from exc

        def _load():
            return spacy.load("en_core_web_sm")

        try:
            _spacy_nlp = _load()
            logger.info("spaCy en_core_web_sm loaded.")
        except OSError:
            # Model not present – download it once, then reload
            logger.info(
                "spaCy model 'en_core_web_sm' not found – downloading (first run only) …"
            )
            from spacy.cli import download as spacy_download

            spacy_download("en_core_web_sm")
            _spacy_nlp = _load()
            logger.info("spaCy en_core_web_sm downloaded and loaded.")
    return _spacy_nlp


# ---------------------------------------------------------------------------
# Step 3.1: Context Windowing
# ---------------------------------------------------------------------------


def chunk_context_window(
    text: str,
    target_surface: str,
    lang: str = "th",
    window: int = 1,
) -> str:
    """
    Segment *text* into sentences and return the sentence that contains
    *target_surface* together with *window* sentences on each side.

    Parameters
    ----------
    text           : Full article body (or any multi-sentence text).
    target_surface : The exact surface form to locate (e.g. "อนุทิน").
    lang           : Language code. "th" uses PyThaiNLP; anything else uses spaCy.
    window         : Number of surrounding sentences to include (default 1 → i±1).

    Returns
    -------
    A single string containing the target sentence plus its neighbours,
    joined by a space.  If the target is not found, the full text is returned
    as a fallback (with a warning).

    Notes
    -----
    Thai sentence boundary detection uses PyThaiNLP's ThaiSentenceSegmentor
    (crfcut / thaisumcut engine).  English uses spaCy's en_core_web_sm model.
    Both segmenters are loaded lazily and cached for the lifetime of the process.
    """
    if not text or not target_surface:
        return text

    # ── Sentence segmentation ────────────────────────────────────────────────
    if lang == "th":
        segmentor = _get_thai_segmentor()
        sentences: list[str] = segmentor.split_into_sentences(text)
    else:
        nlp = _get_spacy_nlp()
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        logger.warning("Sentence segmentation returned no sentences; using full text.")
        return text

    # ── Find the target sentence ─────────────────────────────────────────────
    target_idx: int | None = None
    for idx, sent in enumerate(sentences):
        if target_surface in sent:
            target_idx = idx
            break

    if target_idx is None:
        logger.warning(
            "Target '%s' not found in any sentence; returning full text as fallback.",
            target_surface,
        )
        return text

    # ── Extract i±window slice ───────────────────────────────────────────────
    start = max(0, target_idx - window)
    end = min(len(sentences), target_idx + window + 1)
    context_sentences = sentences[start:end]

    result = " ".join(s.strip() for s in context_sentences if s.strip())
    logger.debug(
        "Context window for '%s': sentences[%d:%d] → %d chars",
        target_surface,
        start,
        end,
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Step 3.2: Entity Anchor + Target Tagging
# ---------------------------------------------------------------------------


def build_anchored_snippet(
    window: str,
    target_surface: str,
    target_description: str,
) -> tuple[str, str]:
    """
    Wrap every occurrence of *target_surface* with ``<target>…</target>`` tags
    and compose a brief entity-anchor header for injection into the prompt.

    Parameters
    ----------
    window              : The context window string from chunk_context_window().
    target_surface      : The surface form to tag (e.g. "อนุทิน").
    target_description  : A short description of the target entity, used as the
                          entity anchor in the prompt to prevent confusion when
                          similar names or overlapping roles exist.

    Returns
    -------
    (tagged_text, target_label)
    tagged_text   : The window text with <target>…</target> wrapping applied.
    target_label  : The decorated label string (e.g. "<target>อนุทิน</target>")
                    ready to be passed as the {target} placeholder in the prompt.

    Notes
    -----
    Uses a simple string replacement (not regex) to avoid escaping issues with
    Thai Unicode.  Replacement is done after escaping the surface form so that
    any special regex characters in entity names do not cause errors.
    """
    if not window or not target_surface:
        return window, target_surface

    tagged_surface = f"<target>{target_surface}</target>"
    # Use regex word-boundary-free replacement (Thai has no word spaces)
    escaped = re.escape(target_surface)
    tagged_text = re.sub(escaped, tagged_surface, window)

    target_label = tagged_surface
    logger.debug("Tagged '%s' in snippet (%d chars).", target_surface, len(tagged_text))
    return tagged_text, target_label


# ---------------------------------------------------------------------------
# Step 3.3: Single-entity SLM inference
# ---------------------------------------------------------------------------


def analyze_entity(
    resolved_entity: ResolvedEntity,
    article_text: str,
    slm_client: SLMClient,
    *,
    lang: str = "th",
    prompt_id: str = "absa_analysis",
    window: int = 1,
) -> AnalyzerResult | None:
    """
    Run the full Phase II analysis pipeline for a single resolved entity.

    Steps performed internally:
      1. chunk_context_window()   – extract i±1 sentence window
      2. build_anchored_snippet() – tag the target + build entity anchor
      3. SLMClient.chat_structured(schema=ABSAOutput) – unified Task A+B+C pass

    Parameters
    ----------
    resolved_entity : A ResolvedEntity from the entity linker output.
    article_text    : Full article body (used for context windowing).
    slm_client      : A pre-initialised SLMClient instance.
    lang            : Language code ("th" or "en").
    prompt_id       : Filename (without .yaml) of the ABSA prompt template.
    window          : Sentence window radius (default 1 → i±1).

    Returns
    -------
    AnalyzerResult on success, None on unrecoverable error.
    """
    surface = resolved_entity.surface_form
    canonical = resolved_entity.canonical_name or surface
    target_description = (
        f"Canonical name: {canonical}. "
        f"Entity type: {resolved_entity.entity_type.value}."
    )

    # 3.1 – Chunk
    context_window = chunk_context_window(
        article_text, surface, lang=lang, window=window
    )

    # 3.2 – Anchor + tag
    tagged_window, target_label = build_anchored_snippet(
        context_window, surface, target_description
    )

    # Load prompt template
    try:
        prompt_data = load_prompt(prompt_id)
    except FileNotFoundError as exc:
        logger.error("ABSA prompt template not found: %s", exc)
        return None

    prompt_yaml_id: str = prompt_data["id"]
    system_prompt: str = prompt_data["templates"]["system"]
    user_prompt: str = prompt_data["templates"]["user"].format(
        target=target_label,
        target_description=target_description,
        context_window=tagged_window,
    )

    # 3.3 – Inference (Task A + B + C in one pass)
    logger.info(
        "Analyzing entity '%s' (canonical='%s') | window=%d chars",
        surface,
        canonical,
        len(tagged_window),
    )

    try:
        absa_output, elapsed_ms = slm_client.chat_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=ABSAOutput,
        )
    except SLMInferenceError as exc:
        logger.error("ABSA inference failed for '%s': %s", surface, exc)
        return None

    result = AnalyzerResult(
        surface_form=surface,
        canonical_name=resolved_entity.canonical_name,
        global_id=resolved_entity.global_id,
        context_window=tagged_window,
        absa=absa_output,
        metadata=InferenceMetadata(
            prompt_id=prompt_yaml_id,
            model=slm_client.model,
            duration_ms=elapsed_ms,
        ),
    )

    logger.info(
        "  → %s | aimed=%s | speaker=%s | sentiment=%s | latency=%dms",
        surface,
        absa_output.is_aimed_at_target,
        absa_output.speaker_type.value,
        absa_output.sentiment.value,
        elapsed_ms,
    )
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_analysis(
    linker_result: EntityLinkerResult,
    article_text: str,
    *,
    lang: str = "th",
    model_name: str = "qwen2.5:7b",
    prompt_id: str = "absa_analysis",
    window: int = 1,
    source_url: str | None = None,
    headline: str | None = None,
    publisher: str | None = None,
    published_at: str | None = None,
    save_to_db: bool = True,
    db_path: str = "database/personalens.db",
) -> list[AnalyzerResult]:
    """
    Run Phase II analysis over *all* resolved entities from the entity linker.

    Skips unresolved entities (those with resolution_method == "unresolved")
    with a warning, but still analyses them if a canonical name is missing –
    the surface form is used as a fallback anchor.

    If save_to_db is True, the AnalyzerResult will be persisted to the SQLite database.

    Parameters
    ----------
    linker_result : Output of entity_linker.resolve_all_entities().
    article_text  : Full article body string.
    lang          : Language code ("th" or "en").
    model_name    : Ollama model tag (e.g. "qwen2.5:7b").
    prompt_id     : ABSA prompt YAML filename (without extension).
    window        : Sentence window radius (default 1 → i±1).
    source_url    : Metadata for Database (URL of article).
    headline      : Metadata for Database (Headline of article).
    publisher     : Metadata for Database (Publisher of article).
    published_at  : Metadata for Database (Publication date).
    save_to_db    : Whether to save the result to sqlite database.
    db_path       : Path to the SQLite DB.

    Returns
    -------
    list[AnalyzerResult] – one entry per entity (None results are dropped).
    """
    t0 = time.monotonic()
    client = SLMClient(model=model_name)
    results: list[AnalyzerResult] = []

    db = Database(path=db_path) if save_to_db else None

    entities = linker_result.resolved
    logger.info(
        "run_analysis: %d entities | lang=%s | model=%s | prompt=%s",
        len(entities),
        lang,
        model_name,
        prompt_id,
    )

    for entity in entities:
        if entity.resolution_method == "unresolved":
            logger.warning(
                "Entity '%s' is unresolved; ABSA may be less reliable.",
                entity.surface_form,
            )

        result = analyze_entity(
            resolved_entity=entity,
            article_text=article_text,
            slm_client=client,
            lang=lang,
            prompt_id=prompt_id,
            window=window,
        )
        if result is not None:
            results.append(result)
            if db is not None:
                try:
                    db.save_analyzer_result(
                        result=result,
                        source_url=source_url,
                        headline=headline,
                        publisher=publisher,
                        lang=lang,
                        published_at=published_at,
                    )
                    logger.debug(
                        "Saved AnalyzerResult for '%s' to database.",
                        entity.surface_form,
                    )
                except Exception as exc:
                    logger.error("Failed to save AnalyzerResult to database: %s", exc)

    if db is not None:
        db.close()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Phase II complete: %d/%d entities analysed in %dms",
        len(results),
        len(entities),
        elapsed_ms,
    )
    return results


# ---------------------------------------------------------------------------
# Smoke test – `uv run src/engine/analyzer.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import logging as _logging
    import uuid

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    from src.schemas.inference import (
        EntityLinkerResult,
        EntityType,
        InferenceMetadata,
        ResolvedEntity,
    )

    # ── Mock article ─────────────────────────────────────────────────────────
    SAMPLE_TEXT = (
        "นายอนุทิน ชาญวีรกูล หรือ เสี่ยหนู รองนายกรัฐมนตรีและรัฐมนตรีว่าการกระทรวงมหาดไทย "
        "ในฐานะหัวหน้าพรรคภูมิใจไทย ลงพื้นที่ตรวจสอบสถานการณ์น้ำท่วมในภาคเหนือ "
        "นักวิเคราะห์ระบุว่านโยบายของอนุทินล้มเหลวในการแก้ปัญหาน้ำท่วมอย่างยั่งยืน "
        "พร้อมสั่งการให้หน่วยงานที่เกี่ยวข้องเร่งแก้ไขปัญหาโดยด่วน"
    )

    # ── Mock EntityLinkerResult (two entities) ───────────────────────────────
    mock_linker_result = EntityLinkerResult(
        resolved=[
            ResolvedEntity(
                surface_form="อนุทิน",
                entity_type=EntityType.PER,
                global_id=uuid.UUID("a1b2c3d4-0000-0000-0000-000000000001"),
                canonical_name="Anutin Charnvirakul",
                confidence_score=0.95,
                resolution_method="alias_exact",
            ),
            ResolvedEntity(
                surface_form="พรรคภูมิใจไทย",
                entity_type=EntityType.ORG,
                global_id=uuid.UUID("a1b2c3d4-0000-0000-0000-000000000002"),
                canonical_name="Bhumjaithai Party",
                confidence_score=0.92,
                resolution_method="alias_exact",
            ),
        ],
        metadata=InferenceMetadata(
            prompt_id="ner-v1.0-qwen2.5:7b",
            model="qwen2.5:7b",
            duration_ms=800,
        ),
    )

    # ── Chunk demo (no SLM needed) ───────────────────────────────────────────
    print("=" * 60)
    print("PersonaLens – Phase II Analyzer Smoke Test")
    print("=" * 60)
    print(f"\nArticle ({len(SAMPLE_TEXT)} chars):\n  {SAMPLE_TEXT}\n")

    for entity in mock_linker_result.resolved:
        window = chunk_context_window(SAMPLE_TEXT, entity.surface_form, lang="th")
        tagged, label = build_anchored_snippet(
            window, entity.surface_form, entity.canonical_name or entity.surface_form
        )
        print(f"[{entity.surface_form}] context window:")
        print(f"  {tagged}\n")

    # ── Full SLM run (requires Ollama) ───────────────────────────────────────
    client = SLMClient(model="qwen2.5:7b")
    if not client.ping():
        print("\nOllama not reachable – skipping SLM inference step.")
    else:
        print("\nRunning full Phase II analysis...\n")
        analysis_results = run_analysis(
            linker_result=mock_linker_result,
            article_text=SAMPLE_TEXT,
            lang="th",
            save_to_db=False,  # Skip DB write in smoke test
        )
        for ar in analysis_results:
            print(json.dumps(ar.model_dump(mode="json"), indent=2, ensure_ascii=False))
            print()

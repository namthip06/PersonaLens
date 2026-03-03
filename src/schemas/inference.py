"""
src/schemas/inference.py
========================
Pydantic v2 schemas for validating SLM (Ollama) inference outputs.

These schemas serve two purposes:
  1. Pass `.model_json_schema()` to Ollama's `format=` parameter so the model
     is CONSTRAINED to return valid JSON matching the schema.
  2. Validate and deserialize the raw response string back into typed Python
     objects via `.model_validate_json()`.

Schema hierarchy
----------------
  ExtractedEntity          ← a single entity mention in a sentence
  NEROutput                ← full NER response: list[ExtractedEntity]
  InferenceMetadata        ← provenance tracking (prompt_id, model, timing)
  NERInferenceResult       ← final envelope: data + metadata

  # Phase II – ABSA Analysis
  SentimentLabel           ← POSITIVE | NEGATIVE | NEUTRAL | MIXED
  SpeakerType              ← REPORTER | QUOTE | UNKNOWN
  ABSAOutput               ← unified Task A + B + C output (SLM-constrained)
  AnalyzerResult           ← final envelope per resolved entity
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Entity type enumeration
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    """
    Controlled vocabulary for entity categories produced by the NER prompt.
    Pydantic will reject any value not in this list → avoids hallucinated types.
    """

    PER = "PER"  # Person
    ORG = "ORG"  # Organisation / political party
    LOC = "LOC"  # Location (physical place: city, province, river, etc.)
    GPE = "GPE"  # Geopolitical Entity (country, state, administrative region)


# ---------------------------------------------------------------------------
# Core NER output schemas
# ---------------------------------------------------------------------------


class ExtractedEntity(BaseModel):
    """
    Represents a single entity mention extracted by the SLM.

    Fields
    ------
    surface_form  : The exact text span as it appears in the source article.
                    (e.g. "เสี่ยนู", "พรรคภูมิใจไทย")
    entity_type   : One of PER | ORG | LOC | GPE (validated against EntityType enum)
    context_clue  : A short phrase explaining WHY the model identified this as
                    an entity.  Used for transparency and debugging alias resolution.
    """

    surface_form: str = Field(
        ..., description="Exact text span from the source article"
    )
    entity_type: EntityType = Field(
        ..., description="Entity category: PER, ORG, LOC, or GPE"
    )
    context_clue: str = Field(
        ..., description="Short reasoning / evidence from the text"
    )


class NEROutput(BaseModel):
    """
    The structured payload that Ollama must return for every NER call.
    Passing `NEROutput.model_json_schema()` to `format=` constrains the model.

    Fields
    ------
    entities  : Ordered list of entities found in the article.
                Empty list is valid (article may mention no tracked persona).
    """

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="All entity mentions extracted from the text",
    )


# ---------------------------------------------------------------------------
# Provenance / metadata schema
# ---------------------------------------------------------------------------


class InferenceMetadata(BaseModel):
    """
    Tracks how a particular inference call was made – essential for
    reproducibility and prompt-version auditing.

    Fields
    ------
    prompt_id     : YAML prompt file identifier (e.g. "ner-v1.0-llama3")
    model         : Ollama model tag used for this call (e.g. "llama3")
    duration_ms   : End-to-end latency in milliseconds (optional, filled by client)
    """

    prompt_id: str = Field(..., description="Prompt YAML id field")
    model: str = Field(..., description="Ollama model name")
    duration_ms: Optional[int] = Field(None, description="Inference latency in ms")


# ---------------------------------------------------------------------------
# Top-level result envelope
# ---------------------------------------------------------------------------


class NERInferenceResult(BaseModel):
    """
    Final result object returned by `entity_linker.extract_entities_with_slm()`.

    Mirrors the JSON structure documented in the Action Plan:
    {
      "data":     { "entities": [ ... ] },
      "metadata": { "prompt_id": ..., "model": ... }
    }
    """

    data: NEROutput = Field(..., description="Validated NER payload from the SLM")
    metadata: InferenceMetadata = Field(
        ..., description="Provenance: which model + prompt produced this"
    )

    def to_entity_list(self) -> list[ExtractedEntity]:
        """Convenience accessor – returns the flat entity list."""
        return self.data.entities


# ---------------------------------------------------------------------------
# Resolution output schemas (Steps 2.2 – 2.4)
# ---------------------------------------------------------------------------


ResolutionMethod = Literal[
    "alias_exact",  # matched an exact row in the aliases table
    "alias_fuzzy",  # matched via fuzzy/Levenshtein similarity
    "external_api",  # confirmed via DuckDuckGo + SLM snippet analysis
    "unresolved",  # no match found; identity unknown
]


class ExternalResolutionOutput(BaseModel):
    """
    Structured payload the SLM must return when analysing DuckDuckGo snippets
    during Step 2.3 external validation.

    Fields
    ------
    canonical_name  : The resolved official name (e.g. "Anutin Charnvirakul")
    confidence      : Model confidence in the resolution [0.0 – 1.0]
    """

    canonical_name: str = Field(
        ..., description="Resolved canonical name of the entity"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score [0, 1]"
    )


class ResolvedEntity(BaseModel):
    """
    The final, resolved representation of a single entity mention after
    running the full Steps 2.2 → 2.4 pipeline.

    Fields
    ------
    surface_form        : Original text span from the article (verbatim)
    entity_type         : PER | ORG | LOC | GPE category from Step 2.1
    global_id           : UUID primary key from the `entities` table.
                          None if the entity could not be resolved.
    canonical_name      : Official name from the DB / API. None if unresolved.
    confidence_score    : Overall confidence of the resolution [0.0 – 1.0].
                          1.0 for exact alias matches, lower for fuzzy / API.
    resolution_method   : How the entity was resolved (see ResolutionMethod).
    """

    surface_form: str = Field(..., description="Original text span from the article")
    entity_type: EntityType = Field(
        ..., description="NER category: PER, ORG, LOC, or GPE"
    )
    global_id: Optional[uuid.UUID] = Field(
        None, description="UUID primary key from the entities table"
    )
    canonical_name: Optional[str] = Field(
        None, description="Canonical name from DB or external API"
    )
    confidence_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Resolution confidence [0, 1]"
    )
    resolution_method: ResolutionMethod = Field(
        "unresolved", description="Which step resolved this entity"
    )

    @property
    def is_resolved(self) -> bool:
        """True when a Global ID has been assigned."""
        return self.global_id is not None


class EntityLinkerResult(BaseModel):
    """
    Final result envelope returned by `entity_linker.resolve_all_entities()`.

    Mirrors the NERInferenceResult pattern but for the resolution pipeline:
    {
      "resolved": [ { "surface_form": ..., "global_id": ..., ... }, ... ],
      "metadata": { "prompt_id": ..., "model": ..., "duration_ms": ... }
    }
    """

    resolved: list[ResolvedEntity] = Field(
        default_factory=list,
        description="All resolved entity mentions from the article",
    )
    metadata: InferenceMetadata = Field(
        ..., description="Provenance metadata from the NER inference step"
    )

    @property
    def resolved_count(self) -> int:
        """Number of entities that were successfully resolved to a Global ID."""
        return sum(1 for e in self.resolved if e.is_resolved)


# ---------------------------------------------------------------------------
# Phase II – ABSA schemas (analyzer.py)
# ---------------------------------------------------------------------------


class SentimentLabel(str, Enum):
    """
    Overall sentiment orientation of the text snippet toward the target entity.
    MIXED is used when strongly positive and negative aspects coexist.
    """

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class SpeakerType(str, Enum):
    """
    Task A: Who produced the sentiment-bearing text?
    REPORTER  = written by the journalist as narrative.
    QUOTE     = a direct quote attributed to a named speaker.
    UNKNOWN   = cannot determine from the snippet.
    """

    REPORTER = "REPORTER"
    QUOTE = "QUOTE"
    UNKNOWN = "UNKNOWN"


class ABSAOutput(BaseModel):
    """
    The structured payload the SLM must return for every ABSA call.
    Covers three tasks in a single inference pass:

    Task A – Speaker identification
    --------------------------------
    speaker_type  : Who authored the sentiment (REPORTER / QUOTE / UNKNOWN).
    speaker_name  : Name of the speaker when speaker_type == QUOTE; else None.

    Task B – Target relation (Quotation-Contamination guard)
    ---------------------------------------------------------
    is_aimed_at_target  : True if the sentiment is directed *at* the target.
    targeting_keywords  : Key words / phrases that justify the decision
                          (explainability / Task B+ check).

    Task C – Aspect-Based Sentiment Analysis
    -----------------------------------------
    sentiment  : POSITIVE | NEGATIVE | NEUTRAL | MIXED toward the target.
    aspects    : Aspect terms (policy, performance, integrity, …) covered.
    rationale  : One-sentence explanation of the sentiment judgment.
    """

    # Task A
    speaker_type: SpeakerType = Field(
        ..., description="Who produced the text: REPORTER, QUOTE, or UNKNOWN"
    )
    speaker_name: Optional[str] = Field(
        None, description="Speaker name when speaker_type is QUOTE, else null"
    )

    # Task B
    is_aimed_at_target: bool = Field(
        ..., description="True if the sentiment is directed at the target entity"
    )
    targeting_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords/phrases evidencing the target relation (explainability)",
    )

    # Task C
    sentiment: SentimentLabel = Field(
        ..., description="Overall sentiment toward the target entity"
    )
    aspects: list[str] = Field(
        default_factory=list,
        description="Aspect terms mentioned (e.g. 'policy', 'leadership', 'integrity')",
    )
    rationale: str = Field(
        ..., description="One-sentence reasoning for the sentiment judgment"
    )


class AnalyzerResult(BaseModel):
    """
    Final result envelope for a single resolved entity after Phase II analysis.

    Fields
    ------
    surface_form    : Original text span from the article.
    canonical_name  : Resolved canonical name (from entity linker).
    global_id       : UUID from the entities table (None if unresolved).
    context_window  : The chunked text snippet (i±1 sentences) sent to the SLM.
    absa            : Validated ABSAOutput from the SLM.
    metadata        : Provenance (prompt id, model, latency).
    """

    surface_form: str = Field(..., description="Original text span from the article")
    canonical_name: Optional[str] = Field(
        None, description="Canonical name from entity linker"
    )
    global_id: Optional[uuid.UUID] = Field(
        None, description="UUID from the entities table"
    )
    context_window: str = Field(
        ..., description="Chunked i±1 sentence window sent to the SLM"
    )
    absa: ABSAOutput = Field(
        ..., description="Validated multi-task ABSA output from the SLM"
    )
    metadata: InferenceMetadata = Field(
        ..., description="Provenance: which model + prompt produced this result"
    )

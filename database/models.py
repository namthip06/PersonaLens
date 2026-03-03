"""
database/models.py
==================
SQLAlchemy ORM models for PersonaLens.
Maps directly to the schema defined in System Context §6.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# Category 1 – Master Data (Identity)
# ---------------------------------------------------------------------------


class Entity(Base):
    """
    Stores the unique persona (canonical identity) for each public figure or
    organisation tracked by PersonaLens.

    Columns
    -------
    entity_id       : UUID primary key (auto-generated)
    canonical_name  : The definitive, official name of the entity
    category        : High-level type – e.g. 'politician', 'company', 'ngo'
    wikidata_id     : Optional Wikidata QID for external knowledge linkage
    lang            : Primary language of the entity (ISO 639-1, e.g. 'th')
    created_at      : UTC timestamp when this record was first inserted

    Relationships
    -------------
    aliases             : 1-to-many → Alias
    sentiment_results   : 1-to-many → SentimentResult
    quoted_details      : 1-to-many → AnalysisDetail (as speaker)
    """

    __tablename__ = "entities"

    entity_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    canonical_name: str = Column(String(255), nullable=False, index=True)
    category: str = Column(String(100), nullable=True)
    wikidata_id: str = Column(String(50), nullable=True, unique=True)
    lang: str = Column(String(10), nullable=False, default="th")
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    # -- Relationships -------------------------------------------------------
    aliases = relationship(
        "Alias",
        back_populates="entity",
        cascade="all, delete-orphan",  # ON DELETE CASCADE
        passive_deletes=True,
    )
    sentiment_results = relationship(
        "SentimentResult",
        back_populates="entity",
        passive_deletes=True,  # ON DELETE RESTRICT enforced at DB level
    )
    quoted_details = relationship(
        "AnalysisDetail",
        back_populates="speaker",
        foreign_keys="AnalysisDetail.speaker_id",
    )

    def __repr__(self) -> str:
        return f"<Entity id={self.entity_id} name='{self.canonical_name}'>"


class Alias(Base):
    """
    Stores name variations for an entity (e.g. nicknames, transliterations).
    Critical for robust entity-linking so every mention maps to one canonical
    record even when writers spell a name differently.

    Columns
    -------
    alias_id    : Serial (auto-increment) primary key
    entity_id   : FK → entities (ON DELETE CASCADE)
    alias_text  : The surface form of the name variant
    source_type : Where this alias originates – 'manual', 'wikipedia', 'ner', …
    """

    __tablename__ = "aliases"

    alias_id: int = Column(Integer, primary_key=True, autoincrement=True)
    entity_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        ForeignKey("entities.entity_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    alias_text: str = Column(String(255), nullable=False, index=True)
    source_type: str = Column(String(50), nullable=True)  # e.g. 'manual', 'wikipedia'

    __table_args__ = (
        UniqueConstraint("entity_id", "alias_text", name="uq_alias_entity_text"),
    )

    # -- Relationships -------------------------------------------------------
    entity = relationship("Entity", back_populates="aliases")

    def __repr__(self) -> str:
        return f"<Alias '{self.alias_text}' → entity {self.entity_id}>"


# ---------------------------------------------------------------------------
# Category 2 – Transactional Data (Analysis)
# ---------------------------------------------------------------------------


class Article(Base):
    """
    Stores metadata for a news article ingested into the pipeline.
    The raw text is NOT stored here to keep this table lightweight and
    query-friendly for time-series analytics.

    Columns
    -------
    article_id   : UUID primary key
    headline     : Article title / headline text
    source_url   : Canonical URL of the article (unique)
    publisher    : News outlet identifier (e.g. 'Bangkok Post')
    lang         : Article language (ISO 639-1)
    published_at : Original publication timestamp (UTC)
    """

    __tablename__ = "articles"

    article_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    headline: str = Column(Text, nullable=False)
    source_url: str = Column(Text, nullable=False, unique=True)
    publisher: str = Column(String(255), nullable=True)
    lang: str = Column(String(10), nullable=False, default="th")
    published_at: datetime = Column(DateTime(timezone=True), nullable=True, index=True)

    # -- Relationships -------------------------------------------------------
    sentiment_results = relationship(
        "SentimentResult",
        back_populates="article",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Article id={self.article_id} headline='{self.headline[:40]}…'>"


class SentimentResult(Base):
    """
    Stores the FINAL, aggregated sentiment score for one entity in one article.
    This is the primary table for dashboard queries.

    Performance note
    ----------------
    A composite index on (entity_id, published_at via article join) is the
    most common dashboard access pattern.  The index is approximated here as
    (entity_id, article_id).  See migration notes for the recommended partial
    index joining against articles.published_at.

    Columns
    -------
    result_id         : UUID primary key
    article_id        : FK → articles (ON DELETE RESTRICT – preserves history)
    entity_id         : FK → entities (ON DELETE RESTRICT – preserves history)
    final_score       : Aggregated sentiment score in [-1, 1]
    sentiment_label   : Human-readable label ('positive', 'negative', 'neutral')
    confidence_score  : Model confidence [0, 1]
    """

    __tablename__ = "sentiment_results"

    result_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    article_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        ForeignKey("articles.article_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    entity_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        ForeignKey("entities.entity_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    final_score: float = Column(Float, nullable=True)
    sentiment_label: str = Column(
        String(20), nullable=True
    )  # positive / negative / neutral
    confidence_score: float = Column(Float, nullable=True)

    __table_args__ = (
        # Composite index for the most common dashboard query pattern:
        # "Give me entity X's results ordered by time."
        # Maps to: WHERE entity_id = ? ORDER BY article.published_at
        # The index below covers the FK side; a covering index on published_at
        # can be added via a migration after join profiling.
        UniqueConstraint("article_id", "entity_id", name="uq_result_article_entity"),
        {
            "comment": "Composite index (entity_id, article_id) supports time-series dashboard queries."
        },
    )

    # -- Relationships -------------------------------------------------------
    article = relationship("Article", back_populates="sentiment_results")
    entity = relationship("Entity", back_populates="sentiment_results")
    analysis_details = relationship(
        "AnalysisDetail",
        back_populates="sentiment_result",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<SentimentResult entity={self.entity_id} "
            f"article={self.article_id} label='{self.sentiment_label}'>"
        )


class AnalysisDetail(Base):
    """
    Granular, sentence-level evidence that explains HOW a final sentiment score
    was derived.  Provides full transparency and enables audit / debugging of
    the SLM pipeline.

    Columns
    -------
    detail_id     : UUID primary key
    result_id     : FK → sentiment_results (ON DELETE CASCADE)
    speaker_id    : FK → entities (NULLABLE) – who said this sentence.
                    NULL means the sentence is reporter narration.
    sentence_text : The exact sentence extracted from the article
    is_headline   : True if this row represents the article headline
    raw_sentiment : Per-sentence raw label from the SLM
    reasoning     : JSONB blob – stores arbitrary SLM metadata such as
                    token_usage, attention_score, chain-of-thought, etc.
                    Using JSONB avoids schema migrations as the SLM evolves.
    """

    __tablename__ = "analysis_details"

    detail_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    result_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        ForeignKey("sentiment_results.result_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    speaker_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        ForeignKey("entities.entity_id", ondelete="SET NULL"),
        nullable=True,  # NULL → journalist narration, not a quote
        index=True,
    )
    sentence_text: str = Column(Text, nullable=False)
    is_headline: bool = Column(Boolean, nullable=False, default=False)
    raw_sentiment: str = Column(String(20), nullable=True)
    reasoning: dict = Column(JSONB, nullable=True)  # flexible SLM metadata

    # -- Relationships -------------------------------------------------------
    sentiment_result = relationship(
        "SentimentResult", back_populates="analysis_details"
    )
    speaker = relationship(
        "Entity",
        back_populates="quoted_details",
        foreign_keys=[speaker_id],
    )

    def __repr__(self) -> str:
        preview = self.sentence_text[:50] if self.sentence_text else ""
        return (
            f"<AnalysisDetail result={self.result_id} "
            f"is_headline={self.is_headline} sentiment='{self.raw_sentiment}' "
            f"sentence='{preview}…'>"
        )

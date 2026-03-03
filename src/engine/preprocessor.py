"""
src/engine/preprocessor.py
===========================
Step 1 – Ingestion, Cleaning, Normalisation, and Deduplication.

Receives raw news articles (plain dict or URL), cleans the body text,
applies NLP normalisation, deduplicates against the database via MinHash
LSH, and persists clean records to the `articles` SQLite table.

Pipeline
--------
  NewsArticle (raw dict / URL)
      │
      ▼
  [Step 1.1] fetch_and_extract()
      │   Trafilatura: HTTP fetch → extract main body, headline, publisher,
      │   published_at. Falls back to provided body text if URL fetch fails.
      │
      ▼
  [Step 1.2] clean_text()
      │   Unicode normalise (NFKC), collapse whitespace, strip HTML artifacts.
      │   Optionally replace dates → <DATE> and numbers → <NUM>.
      │
      ▼
  [Step 1.3] is_duplicate()
      │   MinHash (128 permutations, 5-gram shingles) + LSH index.
      │   Threshold: Jaccard ≥ 0.8  →  treated as duplicate.
      │
      ▼
  [Step 1.4] ingest_article()
      │   Persist headline, source_url, publisher, lang, published_at
      │   to the `articles` table via Database.upsert_article().
      │
      ▼
  Returns (article_id, cleaned_body)  ← ready for entity linker

Public API
----------
    NewsArticle                             – typed input dataclass
    ArticlePreprocessor(db, ...)            – stateful preprocessor
    ArticlePreprocessor.ingest(article)     – main entry: run full pipeline
    ArticlePreprocessor.ingest_url(url)     – convenience: fetch URL first
    fetch_and_extract(url)                  – low-level Trafilatura wrapper
    clean_text(text, ...)                   – NLP cleaning / normalisation
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database import Database  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & config
# ---------------------------------------------------------------------------

# MinHash parameters
_MINHASH_NUM_PERM: int = 128  # permutations (higher → more accurate)
_MINHASH_SHINGLE_SIZE: int = 5  # character n-gram length
_DEDUP_THRESHOLD: float = 0.80  # Jaccard similarity threshold

# NLP normalisation tokens
_TOKEN_DATE: str = "<DATE>"
_TOKEN_NUM: str = "<NUM>"

# Thai / mixed digit date pattern  (e.g. "5 มีนาคม 2567", "03/03/2025")
_RE_DATE = re.compile(
    r"""
    \b(?:
        \d{1,2}\s*(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|
                    กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)\s*\d{4}
        | \d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}     # 03/03/2025 or 3-3-67
        | \d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}        # ISO  2025-03-03
    )\b
    """,
    re.VERBOSE,
)

# General number pattern (integers, decimals, comma-separated — not inside words)
_RE_NUM = re.compile(r"(?<!\w)\d[\d,\.]*\d*(?!\w)")

# Collapse excessive whitespace / newlines
_RE_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Input dataclass
# ---------------------------------------------------------------------------


@dataclass
class NewsArticle:
    """
    Raw article payload.  Provide either *source_url* or *body* (or both).

    Fields
    ------
    source_url   : Canonical URL of the article (required for dedup + storage).
    body         : Pre-fetched article body text.  If None, Trafilatura will
                   fetch and extract it from source_url.
    headline     : Article headline / title (optional; Trafilatura can extract).
    publisher    : Publication name (optional).
    lang         : Language code — ``"th"`` (Thai) or ``"en"`` (English).
    published_at : ISO-8601 string (optional; Trafilatura can extract).
    """

    source_url: str
    body: Optional[str] = None
    headline: Optional[str] = None
    publisher: Optional[str] = None
    lang: str = "th"
    published_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Step 1.1 – Trafilatura fetching & extraction
# ---------------------------------------------------------------------------


def fetch_and_extract(url: str) -> dict:
    """
    Fetch an article by URL and extract its main content via Trafilatura.

    Returns
    -------
    dict with keys: body, headline, publisher, published_at
    All values can be None if extraction fails for that field.
    """
    try:
        import trafilatura
    except ImportError as exc:
        raise ImportError(
            "trafilatura is required for URL fetching. "
            "Install with: uv pip install trafilatura"
        ) from exc

    result: dict = {
        "body": None,
        "headline": None,
        "publisher": None,
        "published_at": None,
    }

    logger.info("Fetching URL: %s", url)
    raw_html = trafilatura.fetch_url(url)
    if not raw_html:
        logger.warning("Trafilatura could not fetch URL: %s", url)
        return result

    # Extract as JSON metadata + body
    extracted = trafilatura.extract(
        raw_html,
        include_comments=False,
        include_tables=False,
        with_metadata=True,
        output_format="json",
    )

    if not extracted:
        logger.warning("Trafilatura extraction returned nothing for: %s", url)
        return result

    import json as _json

    data = _json.loads(extracted)
    result["body"] = data.get("text") or data.get("raw_text")
    result["headline"] = data.get("title")
    result["publisher"] = data.get("sitename") or data.get("author")
    result["published_at"] = data.get("date")
    return result


# ---------------------------------------------------------------------------
# Step 1.2 – NLP text cleaning & normalisation
# ---------------------------------------------------------------------------


def clean_text(
    text: str,
    *,
    replace_dates: bool = True,
    replace_numbers: bool = False,
    lowercase: bool = False,
) -> str:
    """
    Clean and normalise raw article body text for downstream NLP tasks.

    Processing order
    ----------------
    1. Unicode NFKC normalisation  – standardise full-width / composed chars.
    2. Strip HTML/XML entities & leftover tags.
    3. Date tokenisation           – optionally replace with ``<DATE>``.
    4. Number tokenisation         – optionally replace with ``<NUM>``.
    5. Lowercase                   – optionally fold to lower case.
    6. Collapse whitespace         – reduce all runs of \\s to a single space.

    Parameters
    ----------
    text           : Raw input string.
    replace_dates  : If True, replace date patterns with ``<DATE>``.
                     Recommended for sentiment / NER tasks where dates are noise.
    replace_numbers: If True, replace numeric patterns with ``<NUM>``.
                     Set True for sentiment tasks; False if amounts matter
                     (e.g. financial analysis).
    lowercase      : Fold to lower case (useful for BoW / TF-IDF; leave False
                     for transformer models that preserve case).

    Returns
    -------
    str – cleaned, normalised text ready for tokenisation.
    """
    if not text:
        return ""

    # 1. Unicode normalise
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip any residual HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&quot;", '"')
    )

    # 3. Date normalisation (before number pass so "5 มีนาคม 2567" is caught first)
    if replace_dates:
        text = _RE_DATE.sub(_TOKEN_DATE, text)

    # 4. Number normalisation
    if replace_numbers:
        text = _RE_NUM.sub(_TOKEN_NUM, text)

    # 5. Lowercase
    if lowercase:
        text = text.lower()

    # 6. Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Step 1.3 – MinHash deduplication
# ---------------------------------------------------------------------------


def _shingle(text: str, k: int = _MINHASH_SHINGLE_SIZE) -> set[str]:
    """Return the set of character k-grams from *text*."""
    text = text.replace(" ", "")  # Thai: spaces are not word boundaries
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _compute_minhash(text: str, num_perm: int = _MINHASH_NUM_PERM):
    """
    Return a ``datasketch.MinHash`` object for *text*.

    The MinHash is built from character 5-gram shingles so that slight
    reformulations (word order, punctuation) do not cause false misses.
    """
    try:
        from datasketch import MinHash
    except ImportError as exc:
        raise ImportError(
            "datasketch is required for deduplication. "
            "Install with: uv pip install datasketch"
        ) from exc

    mh = MinHash(num_perm=num_perm)
    for shingle in _shingle(text):
        mh.update(shingle.encode("utf-8"))
    return mh


class _LSHIndex:
    """
    Thin wrapper around ``datasketch.MinHashLSH`` that keeps the index in memory
    for the duration of a preprocessor session.

    The LSH index is initialised once and reused across all ``ingest()`` calls
    so that cross-article deduplication works within a single session.
    """

    def __init__(
        self,
        threshold: float = _DEDUP_THRESHOLD,
        num_perm: int = _MINHASH_NUM_PERM,
    ) -> None:
        try:
            from datasketch import MinHashLSH
        except ImportError as exc:
            raise ImportError("datasketch is required for deduplication.") from exc
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._num_perm = num_perm
        self._count: int = 0

    def _key(self, url: str) -> str:
        """Stable LSH key derived from the article URL."""
        return hashlib.sha1(url.encode()).hexdigest()[:16]

    def is_duplicate(self, text: str, url: str) -> bool:
        """
        Check whether *text* is a near-duplicate of any document already in
        the index.  Returns True if a duplicate is found.
        """
        mh = _compute_minhash(text, num_perm=self._num_perm)
        neighbours = self._lsh.query(mh)
        return len(neighbours) > 0

    def add(self, text: str, url: str) -> None:
        """Add *text* to the LSH index, keyed by *url*."""
        mh = _compute_minhash(text, num_perm=self._num_perm)
        key = self._key(url)
        try:
            self._lsh.insert(key, mh)
            self._count += 1
        except ValueError:
            # Key already present – idempotent
            pass

    @property
    def size(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Step 1.4 – Main ArticlePreprocessor
# ---------------------------------------------------------------------------


class ArticlePreprocessor:
    """
    Stateful article ingestion pipeline for PersonaLens.

    Each instance owns:
    - A reference to the SQLite ``Database``.
    - A session-scoped ``_LSHIndex`` for in-memory deduplication.

    Parameters
    ----------
    db             : An open ``Database`` instance.
    replace_dates  : Pass ``True`` (default) to normalise date patterns.
    replace_numbers: Pass ``True`` to normalise number patterns (default False).
    lowercase      : Pass ``True`` to lowercase body text (default False).
    dedup_threshold: Jaccard similarity threshold for near-duplicate detection.
                     Articles above this threshold are skipped (default 0.8).
    """

    def __init__(
        self,
        db: Database,
        *,
        replace_dates: bool = True,
        replace_numbers: bool = False,
        lowercase: bool = False,
        dedup_threshold: float = _DEDUP_THRESHOLD,
    ) -> None:
        self._db = db
        self._replace_dates = replace_dates
        self._replace_numbers = replace_numbers
        self._lowercase = lowercase
        self._lsh = _LSHIndex(threshold=dedup_threshold)

        # Stats for the session
        self._stats = {
            "ingested": 0,
            "duplicates_skipped": 0,
            "fetch_errors": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, article: NewsArticle) -> tuple[str, str] | None:
        """
        Run the full pipeline for *article*.

        Returns
        -------
        (article_id, cleaned_body)  on success.
        None                        if the article was a duplicate or had no body.
        """
        # ── 1.1 Fetch + extract (if no body provided) ─────────────────────
        body = article.body
        headline = article.headline
        publisher = article.publisher
        published_at = article.published_at

        if not body and article.source_url:
            fetched = fetch_and_extract(article.source_url)
            body = fetched["body"]
            headline = headline or fetched["headline"]
            publisher = publisher or fetched["publisher"]
            published_at = published_at or fetched["published_at"]

        if not body:
            logger.warning(
                "No body text available for article '%s'. Skipping.",
                article.source_url,
            )
            self._stats["fetch_errors"] += 1
            return None

        # ── 1.2 Clean & normalise ─────────────────────────────────────────
        cleaned = clean_text(
            body,
            replace_dates=self._replace_dates,
            replace_numbers=self._replace_numbers,
            lowercase=self._lowercase,
        )

        if not cleaned:
            logger.warning(
                "Body reduced to empty after cleaning for '%s'. Skipping.",
                article.source_url,
            )
            return None

        # ── 1.3 Deduplication ─────────────────────────────────────────────
        if self._lsh.is_duplicate(cleaned, article.source_url):
            logger.info(
                "Near-duplicate detected → skipping '%s'  (LSH index size: %d)",
                article.source_url,
                self._lsh.size,
            )
            self._stats["duplicates_skipped"] += 1
            return None

        # Add to the in-process LSH index so future articles are checked against it
        self._lsh.add(cleaned, article.source_url)

        # ── 1.4 Persist to SQLite ─────────────────────────────────────────
        article_id = self._db.upsert_article(
            source_url=article.source_url,
            headline=headline,
            publisher=publisher,
            lang=article.lang,
            published_at=published_at,
        )

        self._stats["ingested"] += 1
        logger.info(
            "Ingested article '%s' → article_id=%s  (session total: %d)",
            article.source_url,
            article_id,
            self._stats["ingested"],
        )
        return article_id, cleaned

    def ingest_url(self, url: str, lang: str = "th") -> tuple[str, str] | None:
        """
        Convenience wrapper: build a ``NewsArticle`` from a bare URL and ingest.

        Returns
        -------
        (article_id, cleaned_body)  on success, or None on failure / duplicate.
        """
        return self.ingest(NewsArticle(source_url=url, lang=lang))

    def ingest_batch(self, articles: list[NewsArticle]) -> list[tuple[str, str]]:
        """
        Run :meth:`ingest` for every article in *articles*.

        Duplicates and fetch failures are silently skipped (logged).

        Returns
        -------
        List of (article_id, cleaned_body) tuples — one per accepted article.
        """
        results = []
        for art in articles:
            out = self.ingest(art)
            if out is not None:
                results.append(out)
        return results

    @property
    def stats(self) -> dict:
        """Return a copy of the current session statistics."""
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Smoke test – `uv run src/engine/preprocessor.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    DB_PATH = project_root / "database" / "personalens.db"
    db = Database(str(DB_PATH))
    preprocessor = ArticlePreprocessor(db, replace_dates=True, replace_numbers=False)

    # ── Test 1: plain body ────────────────────────────────────────────────────
    sample = NewsArticle(
        source_url="https://example.com/news/1",
        body=(
            "อนุทิน ชาญวีรกูล เดินทางเยี่ยมชมพื้นที่น้ำท่วมภาคเหนือเมื่อวันที่ 5 มีนาคม 2567 "
            "โดยสั่งการให้หน่วยงานที่เกี่ยวข้องเร่งแก้ไขปัญหาน้ำท่วมอย่างเร่งด่วน "
            "พร้อมอนุมัติงบประมาณกว่า 500 ล้านบาทเพื่อฟื้นฟูพื้นที่ได้รับผลกระทบ"
        ),
        headline="อนุทินเยี่ยมพื้นที่น้ำท่วม",
        publisher="Thai News Agency",
        lang="th",
    )
    result = preprocessor.ingest(sample)
    print(f"\nTest 1 – ingest plain body: {result}")

    # ── Test 2: near-duplicate (same body, different URL) ─────────────────────
    duplicate = NewsArticle(
        source_url="https://mirror.example.com/news/1",
        body=sample.body,  # identical body → should be flagged as duplicate
        lang="th",
    )
    result2 = preprocessor.ingest(duplicate)
    print(f"Test 2 – near-duplicate detection: {result2}  (should be None)")

    # ── Test 3: clean_text demo ───────────────────────────────────────────────
    raw = "ราคาน้ำมันดิบอยู่ที่ 85.50 USD/barrel วันที่ 03/03/2025  ลดลง 2.3%"
    cleaned_with_num = clean_text(raw, replace_dates=True, replace_numbers=True)
    cleaned_no_num = clean_text(raw, replace_dates=True, replace_numbers=False)
    print("\nTest 3 – clean_text:")
    print(f"  raw            : {raw}")
    print(f"  dates+nums     : {cleaned_with_num}")
    print(f"  dates only     : {cleaned_no_num}")

    print("\nSession stats:", preprocessor.stats)
    db.close()

"""
pages/3_⚙️_Admin_Pipeline.py
==============================
Administrative control panel:
  A. Raw Ingestion Form  – submit URL or body text → Orchestrator pipeline
  B. Processing Log      – real-time log output in a terminal-style window
  C. Database KPIs       – quick stats on current database state
"""

import contextlib
import json
import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='font-size:1.9rem; font-weight:700; margin-bottom:4px;'>
        ⚙️ Admin &amp; Pipeline Control
    </h1>
    <p style='color:#64748b; margin-top:0;'>
        Ingest new articles, trigger the SLM pipeline, and monitor processing logs.
    </p>
    <hr style='border-color:#334155; margin-bottom:24px;'>
    """,
    unsafe_allow_html=True,
)

# ── Import pipeline modules (graceful fallback) ───────────────────────────────
try:
    from database.database import Database
    from src.engine.analyzer import (
        analyze_entity,
        build_anchored_snippet,
        chunk_context_window,
    )
    from src.engine.entity_linker import extract_entities_with_slm, resolve_all_entities
    from src.engine.preprocessor import (
        ArticlePreprocessor,
        NewsArticle,
        clean_text,
        fetch_and_extract,
    )
    from src.engine.slm_client import SLMClient

    _PIPELINE_AVAILABLE = True
except ImportError as _e:
    _PIPELINE_AVAILABLE = False
    _IMPORT_ERROR = str(_e)

# ─────────────────────────────────────────────────────────────────────────────
# C. Database & ETL Quality KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🗄️ System & Pipeline Health (ETL)")

try:
    from app.data import (
        get_pipeline_stats,
        get_recent_articles,
        get_etl_metrics,
        get_resolution_accuracy,
        get_foreign_key_integrity,
        get_failed_ingestion_logs,
    )
    import plotly.graph_objects as go
    import plotly.express as px

    stats = get_pipeline_stats()
    etl = get_etl_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📰 Articles", stats["articles"])
    c2.metric("🧑 Entities", stats["entities"])
    c3.metric("📊 Sentiment Results", stats["sentiment_results"])
    c4.metric("🔬 Analysis Details", stats["analysis_details"])

    # Additional System Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    sys1, sys2, sys3, sys4 = st.columns(4)
    sys1.metric("💾 Database Size", f"{etl['db_size_mb']} MB")
    sys2.metric(
        "🛑 Deduplication Rate",
        f"{etl['deduplication_rate_pct']}%",
        delta="MinHash LSH",
        delta_color="off",
    )
    sys3.metric(
        "⚡ Cache Hit Rate",
        f"{etl['cache_hit_rate_pct']}%",
        delta="Alias DB",
        delta_color="off",
    )
    sys4.metric("⚙️ System Status", "qwen2.5:7b", delta="Online", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    chart1, chart2 = st.columns([1, 1])

    with chart1:
        st.markdown("**Pipeline Latency (Avg SEC)**")
        # Gauge chart using Plotly
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=etl["pipeline_latency_sec"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Seconds per Article", "font": {"size": 14}},
                gauge={
                    "axis": {
                        "range": [None, 30],
                        "tickwidth": 1,
                        "tickcolor": "darkblue",
                    },
                    "bar": {"color": "#3b82f6"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 10], "color": "#dcfce7"},
                        {"range": [10, 20], "color": "#fef08a"},
                        {"range": [20, 30], "color": "#fecaca"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="white",
            font=dict(color="#1e293b"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with chart2:
        st.markdown("**Entity Resolution Accuracy**")
        acc_df = get_resolution_accuracy()
        if not acc_df.empty:
            fig_acc = px.pie(
                acc_df,
                values="count",
                names="source_type",
                hole=0.4,
                color_discrete_sequence=["#3b82f6", "#8b5cf6", "#10b981"],
            )
            fig_acc.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="white",
                font=dict(color="#1e293b"),
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("No alias data yet.")

    st.markdown("<br>", unsafe_allow_html=True)
    col_logs, col_orphans = st.columns([2, 1])
    with col_logs:
        st.markdown("**Failed Ingestion Logs**")
        fails_df = get_failed_ingestion_logs()
        st.dataframe(fails_df, hide_index=True, use_container_width=True)
    with col_orphans:
        st.markdown("**Foreign Key Integrity Status**")
        orphans_df = get_foreign_key_integrity()
        st.dataframe(orphans_df, hide_index=True, use_container_width=True)

except Exception as e:
    st.warning(f"Could not fetch pipeline stats: {e}")

st.markdown("<hr style='border-color:#1e293b; margin:24px 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# A. Raw Ingestion Form
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Raw Ingestion Form")

if not _PIPELINE_AVAILABLE:
    st.error(
        f"⚠️ Pipeline modules could not be imported. "
        f"Make sure the project is installed correctly and SLM dependencies are available.\n\n"
        f"Import error: `{_IMPORT_ERROR}`"
    )
    st.info(
        "You can still explore existing data on the Dashboard and Deep-Dive pages. "
        "The ingestion form will become active once the engine dependencies are resolved."
    )
else:
    tab_url, tab_body = st.tabs(["🔗 Ingest by URL", "📝 Ingest Raw Text"])

    with tab_url:
        with st.form("ingest_url_form", clear_on_submit=False):
            st.markdown(
                "<p style='color:#94a3b8; font-size:0.85rem;'>Paste a news article URL. "
                "Trafilatura will fetch and extract the content automatically.</p>",
                unsafe_allow_html=True,
            )
            url_input = st.text_input(
                "Article URL",
                placeholder="https://example.com/news/article",
                key="url_input",
            )
            publisher_url = st.text_input(
                "Publisher (optional)",
                placeholder="e.g. Bangkok Post",
                key="publisher_url",
            )
            submitted_url = st.form_submit_button(
                "🚀 Run Pipeline", type="primary", use_container_width=True
            )

    with tab_body:
        with st.form("ingest_body_form", clear_on_submit=False):
            st.markdown(
                "<p style='color:#94a3b8; font-size:0.85rem;'>Paste the article body text "
                "directly. A synthetic URL will be generated for deduplication.</p>",
                unsafe_allow_html=True,
            )
            body_input = st.text_area(
                "Article Body",
                placeholder="Paste the full article text here…",
                height=180,
                key="body_input",
            )
            headline_input = st.text_input(
                "Headline (optional)",
                placeholder="Article headline",
                key="headline_input",
            )
            publisher_body = st.text_input(
                "Publisher (optional)",
                placeholder="e.g. Matichon",
                key="publisher_body",
            )
            col_lang, col_pad = st.columns([1, 3])
            with col_lang:
                lang_input = st.selectbox("Language", ["th", "en"], key="lang_input")
            submitted_body = st.form_submit_button(
                "🚀 Run Pipeline", type="primary", use_container_width=True
            )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _fmt_result(obj) -> str:
        """
        Serialise a Pydantic model, dict, or list to an indented JSON string.
        """
        try:
            if hasattr(obj, "model_dump"):
                raw = obj.model_dump(mode="json")
            elif isinstance(obj, list) and obj and hasattr(obj[0], "model_dump"):
                raw = [item.model_dump(mode="json") for item in obj]
            else:
                raw = obj
            return json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
        except Exception as exc:
            return f"(could not serialise result: {exc})\n"

    def _render_log(buf: list[str]) -> str:
        """Join log lines and wrap in a code block for st.empty display."""
        return "```\n" + "".join(buf) + "\n```"

    # ── Log capture ───────────────────────────────────────────────────────────
    class _ListHandler(logging.Handler):
        """Logging handler that appends formatted records to a list and refreshes the UI."""

        def __init__(self, buffer: list[str], refresh_fn=None) -> None:
            super().__init__()
            self._buf = buffer
            self._refresh = refresh_fn or (lambda: None)

        def emit(self, record: logging.LogRecord) -> None:
            level = record.levelname
            prefix = {
                "DEBUG": "🔍",
                "INFO": "ℹ️ ",
                "WARNING": "⚠️ ",
                "ERROR": "❌",
                "CRITICAL": "🔥",
            }.get(level, "  ")
            name_parts = record.name.split(".")
            short_name = (
                ".".join(name_parts[-2:]) if len(name_parts) >= 2 else record.name
            )
            self._buf.append(f"{prefix} [{short_name}] {self.format(record)}\n")
            self._refresh()

    @contextlib.contextmanager
    def _engine_log_capture(buffer: list[str], refresh_fn=None):
        """
        Attach a ListHandler to the 'src.engine' logger for the duration of
        the pipeline call, then remove it.  Every emitted log record will call
        *refresh_fn* so the UI updates in real time.
        """
        handler = _ListHandler(buffer, refresh_fn=refresh_fn)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))
        engine_logger = logging.getLogger("src.engine")
        engine_logger.setLevel(logging.DEBUG)
        engine_logger.addHandler(handler)
        try:
            yield
        finally:
            engine_logger.removeHandler(handler)

    # ── Pipeline execution ────────────────────────────────────────────────────
    def _run_pipeline(
        article: "NewsArticle",
        log_buffer: list[str],
        log_placeholder=None,
    ) -> str | None:
        """
        Execute the full PersonaLens pipeline for one article, explicitly
        calling each documented sub-step function and logging its result.
        *log_placeholder* is an `st.empty()` that re-renders on every append.
        Returns the article_id on success, None on failure/duplicate.
        """
        SEP = "━" * 50 + "\n"

        def _refresh():
            if log_placeholder is not None:
                log_placeholder.markdown(_render_log(log_buffer))

        def _banner(title: str):
            log_buffer.append(SEP)
            log_buffer.append(f"🔵 {title}\n")
            log_buffer.append(SEP)
            _refresh()

        db = Database()
        try:
            with _engine_log_capture(log_buffer, refresh_fn=_refresh):
                # ── Step 1.1 – fetch_and_extract() (URL articles only) ────────
                body = article.body
                headline = article.headline
                publisher = article.publisher
                published_at = article.published_at

                if (
                    not body
                    and article.source_url
                    and not article.source_url.startswith("__manual__")
                ):
                    _banner("STEP 1.1 — fetch_and_extract()")
                    fetched = fetch_and_extract(article.source_url)
                    body = fetched["body"]
                    headline = headline or fetched["headline"]
                    publisher = publisher or fetched["publisher"]
                    published_at = published_at or fetched["published_at"]
                    log_buffer.append("📦 Extracted metadata:\n")
                    log_buffer.append(
                        _fmt_result(
                            {
                                "headline": headline,
                                "publisher": publisher,
                                "published_at": published_at,
                                "body_length": len(body) if body else 0,
                            }
                        )
                    )
                    _refresh()
                    if not body:
                        log_buffer.append(
                            "❌ [Step 1.1] Could not fetch body — aborting.\n"
                        )
                        _refresh()
                        return None

                # ── Step 1.2 – clean_text() ───────────────────────────────────
                _banner("STEP 1.2 — clean_text()")
                clean_body = clean_text(
                    body or "", replace_dates=True, replace_numbers=False
                )
                raw_len = len(body) if body else 0
                log_buffer.append(
                    f"✅ [Step 1.2] Cleaned body: {raw_len} → {len(clean_body)} chars\n"
                )
                _refresh()

                # ── Step 1.3 – Deduplication (MinHash LSH) ────────────────────
                _banner("STEP 1.3 — is_duplicate() [MinHash LSH, Jaccard ≥ 0.8]")
                # Rebuild article with pre-fetched body so ingest() skips its
                # own fetch and goes directly to dedup → persist.
                article_prefetched = NewsArticle(
                    source_url=article.source_url,
                    body=body,
                    headline=headline,
                    publisher=publisher,
                    lang=article.lang,
                    published_at=published_at,
                )
                prep = ArticlePreprocessor(db)
                result = prep.ingest(article_prefetched)

                if result is None:
                    log_buffer.append(
                        "🟡 [Step 1.3] Near-duplicate detected — article skipped.\n"
                    )
                    _refresh()
                    return None

                article_id, clean_body = result  # use preprocessor's cleaned copy
                log_buffer.append("✅ [Step 1.3] Unique article confirmed.\n")
                _refresh()

                # ── Step 1.4 – upsert_article() ──────────────────────────────
                _banner("STEP 1.4 — ingest_article() → upsert_article()")
                log_buffer.append(
                    f"✅ [Step 1.4] Article persisted → article_id = {article_id}\n"
                )
                log_buffer.append("📦 Stored metadata:\n")
                log_buffer.append(
                    _fmt_result(
                        {
                            "article_id": article_id,
                            "source_url": article.source_url,
                            "headline": headline,
                            "publisher": publisher,
                            "lang": article.lang,
                            "published_at": published_at,
                            "clean_body_chars": len(clean_body),
                        }
                    )
                )
                _refresh()

                # ── Step 2.1 – extract_entities_with_slm() ───────────────────
                _banner("STEP 2.1 — extract_entities_with_slm()")
                ner_result = extract_entities_with_slm(text=clean_body)
                if ner_result is None:
                    log_buffer.append("❌ [Step 2.1] NER returned None — aborting.\n")
                    _refresh()
                    return None
                entity_count = len(ner_result.data.entities)
                log_buffer.append(
                    f"✅ [Step 2.1] Extracted {entity_count} raw entities.\n"
                )
                log_buffer.append("📦 NER Result:\n")
                log_buffer.append(_fmt_result(ner_result))
                _refresh()

                # ── Step 2.2 – alias_resolver.resolve_from_db() ──────────────
                _banner("STEP 2.2 — alias_resolver.resolve_from_db() [Exact + Fuzzy]")
                _refresh()

                # ── Step 2.3 – external_validator.validate_entity_external() ──
                _banner(
                    "STEP 2.3 — external_validator.validate_entity_external() [DDG + SLM]"
                )
                _refresh()
                # Both 2.2 & 2.3 execute inside resolve_all_entities;
                # their engine logger calls flow through _engine_log_capture.
                linker_result = resolve_all_entities(ner_result, clean_body, session=db)
                resolved_count = len(linker_result.resolved)
                log_buffer.append(
                    f"✅ [Step 2.2–2.3] Resolved {resolved_count} entities.\n"
                )
                log_buffer.append("📦 Entity Linker Result:\n")
                log_buffer.append(_fmt_result(linker_result))
                _refresh()

                # ── Step 3 – ABSA per entity ──────────────────────────────────
                _banner("STEP 3 — ABSA Sentiment Analysis (per entity)")
                slm_client = SLMClient(model="qwen2.5:7b")
                absa_results = []

                for entity in linker_result.resolved:
                    surface = entity.surface_form
                    canonical = entity.canonical_name or surface

                    log_buffer.append(
                        f"\n  ▸ Entity: {surface!r}  (canonical: {canonical})\n"
                    )
                    _refresh()

                    # Step 3.1 – chunk_context_window()
                    log_buffer.append("  [Step 3.1] chunk_context_window()\n")
                    _refresh()
                    context_window = chunk_context_window(
                        clean_body, surface, lang=article.lang
                    )
                    log_buffer.append(
                        f"  📦 context window ({len(context_window)} chars):\n"
                    )
                    log_buffer.append(_fmt_result({"context_window": context_window}))
                    _refresh()

                    # Step 3.2 – build_anchored_snippet()
                    log_buffer.append("  [Step 3.2] build_anchored_snippet()\n")
                    _refresh()
                    target_description = (
                        f"Canonical name: {canonical}. "
                        f"Entity type: {entity.entity_type.value}."
                    )
                    tagged_window, target_label = build_anchored_snippet(
                        context_window, surface, target_description
                    )
                    log_buffer.append("  📦 anchored snippet:\n")
                    log_buffer.append(
                        _fmt_result(
                            {
                                "tagged_window": tagged_window,
                                "target_label": target_label,
                            }
                        )
                    )
                    _refresh()

                    # Step 3.3 – analyze_entity()
                    log_buffer.append("  [Step 3.3] analyze_entity()\n")
                    _refresh()
                    result_absa = analyze_entity(
                        resolved_entity=entity,
                        article_text=clean_body,
                        slm_client=slm_client,
                        lang=article.lang,
                    )
                    if result_absa is None:
                        log_buffer.append(
                            f"  ⚠️  analyze_entity returned None for {surface!r}\n"
                        )
                        _refresh()
                        continue

                    # Step 3.4 - Save to DB
                    try:
                        db.save_analyzer_result(
                            result=result_absa,
                            article_id=article_id,
                            source_url=article.source_url,
                            headline=article.headline,
                            publisher=article.publisher,
                            lang=article.lang,
                            published_at=published_at,
                        )
                        log_buffer.append(
                            f"  ✅ Saved AnalyzerResult for {surface!r} to database.\n"
                        )
                    except Exception as exc:
                        log_buffer.append(
                            f"  ❌ Failed to save AnalyzerResult to database: {exc}\n"
                        )

                    log_buffer.append(
                        f"  ✅ [Step 3.3] ABSA done in {result_absa.metadata.duration_ms}ms\n"
                    )
                    log_buffer.append("  📦 AnalyzerResult:\n")
                    log_buffer.append(_fmt_result(result_absa))
                    _refresh()
                    absa_results.append(result_absa)

                log_buffer.append(
                    f"\n✅ [Step 3] Analysis complete. {len(absa_results)} results.\n"
                )
                _refresh()

            log_buffer.append(SEP)
            log_buffer.append("🏁 Pipeline complete.\n")
            _refresh()
            return article_id

        except Exception as exc:
            log_buffer.append(f"❌ Pipeline error: {exc}\n")
            _refresh()
            return None
        finally:
            db.close()

    # Trigger from URL form
    if "submitted_url" in dir() and submitted_url and url_input.strip():
        article_obj = NewsArticle(
            source_url=url_input.strip(),
            publisher=publisher_url.strip() or None,
        )
        log_lines: list[str] = []
        live_ph = st.empty()
        live_ph.markdown(_render_log(["Starting pipeline…\n"]))
        aid = _run_pipeline(article_obj, log_lines, log_placeholder=live_ph)

        if aid:
            st.toast(f"✅ Article ingested! article_id: {aid[:8]}…", icon="🎉")
        else:
            st.toast("Article was a duplicate or an error occurred.", icon="⚠️")

        st.session_state["last_log"] = log_lines
        live_ph.empty()

    # Trigger from body form
    if "submitted_body" in dir() and submitted_body and body_input.strip():
        import hashlib

        synthetic_url = f"__manual__{hashlib.md5(body_input.encode()).hexdigest()[:12]}"
        article_obj = NewsArticle(
            source_url=synthetic_url,
            body=body_input.strip(),
            headline=headline_input.strip() or None,
            publisher=publisher_body.strip() or None,
            lang=lang_input,
        )
        log_lines2: list[str] = []
        live_ph2 = st.empty()
        live_ph2.markdown(_render_log(["Starting pipeline…\n"]))
        aid2 = _run_pipeline(article_obj, log_lines2, log_placeholder=live_ph2)

        if aid2:
            st.toast(f"✅ Article ingested! article_id: {aid2[:8]}…", icon="🎉")
        else:
            st.toast("Article was a duplicate or an error occurred.", icon="⚠️")

        st.session_state["last_log"] = log_lines2
        live_ph2.empty()

st.markdown("<hr style='border-color:#1e293b; margin:24px 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# B. Processing Log
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Processing Log")

log_data = st.session_state.get("last_log", [])

if log_data:
    st.code("".join(log_data), language="")  # monospace, no syntax highlight
else:
    st.info(
        "No log data yet. Submit an article above to see real-time processing steps."
    )

if st.button("🗑️ Clear Log", key="clear_log"):
    st.session_state["last_log"] = []
    st.rerun()

st.markdown("<hr style='border-color:#1e293b; margin:24px 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Recent Articles Table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Recent Articles")

try:
    recent_df = get_recent_articles(limit=20)
    if recent_df.empty:
        st.info("No articles in the database yet.")
    else:
        # Make source_url a clickable link via column_config
        st.dataframe(
            recent_df,
            column_config={
                "article_id": st.column_config.TextColumn("ID", width="small"),
                "headline": st.column_config.TextColumn("Headline", width="large"),
                "publisher": st.column_config.TextColumn("Publisher"),
                "published_at": st.column_config.TextColumn("Published At"),
                "source_url": st.column_config.LinkColumn("URL", width="medium"),
            },
            hide_index=True,
            use_container_width=True,
        )
except Exception as e:
    st.warning(f"Could not load recent articles: {e}")

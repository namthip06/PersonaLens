"""
pages/2_🔍_Entity_Deep_Dive.py
================================
Granular entity view: identity cards, alias cloud, CoT reasoning explorer,
confidence bars, and per-entity sentiment timeline.
"""

import json
import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.data import (
    get_all_entities,
    get_analysis_details_for_entity,
    get_entity_sentiment_summary,
    get_entity_timeline,
    get_entity_with_aliases,
    get_entity_trajectory,
    get_top_publishers_for_entity,
    get_confidence_distribution_for_entity,
    get_speaker_network_for_entity,
)

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='font-size:1.9rem; font-weight:700; margin-bottom:4px;'>
        🔍 Entity Deep-Dive
    </h1>
    <p style='color:#64748b; margin-top:0;'>
        Investigate a single entity — identity, aliases, reasoning chains &amp; sentiment history.
    </p>
    <hr style='border-color:#334155; margin-bottom:24px;'>
    """,
    unsafe_allow_html=True,
)

_DARK = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font_color="#e2e8f0",
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Entity selector
# ─────────────────────────────────────────────────────────────────────────────
entities_df = get_all_entities()

if entities_df.empty:
    st.warning(
        "No entities in the database yet. Use the Admin page to ingest articles."
    )
    st.stop()

entity_options = dict(zip(entities_df["canonical_name"], entities_df["entity_id"]))
selected_name = st.selectbox(
    "Select an entity", list(entity_options.keys()), key="entity_sel"
)
entity_id = entity_options[selected_name]

# ─────────────────────────────────────────────────────────────────────────────
# A. Canonical Identity & Aliases
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Canonical Identity & Aliases")

entity_data = get_entity_with_aliases(entity_id)
ent = entity_data.get("entity", {})
aliases = entity_data.get("aliases", [])

# Category badge colour
cat_colors = {"PER": "#3b82f6", "ORG": "#8b5cf6", "LOC": "#10b981", "GPE": "#f59e0b"}
cat = ent.get("category", "?")
cat_color = cat_colors.get(cat, "#64748b")

col_id, col_stats = st.columns([3, 2])

with col_id:
    st.markdown(
        f"""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:16px; padding:24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:16px;'>
                <span style='font-size:2rem;'>{"🧑" if cat == "PER" else "🏢" if cat == "ORG" else "🌍"}</span>
                <div>
                    <h2 style='margin:0; padding: 5px 0; font-size:1.4rem; color:#1e293b;'>{ent.get("canonical_name", "—")}</h2>
                    <span style='background:{cat_color}15; color:{cat_color}; border:1px solid {cat_color}33;
                                border-radius:999px; padding:2px 10px; font-size:0.72rem; font-weight:600;'>
                        {cat}
                    </span>
                </div>
            </div>
            <p style='color:#94a3b8; font-size:0.8rem; margin:0; font-family: sans-serif;'>
                Entity ID: {entity_id}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_stats:
    sentiment_df = get_entity_sentiment_summary(entity_id)
    if not sentiment_df.empty:
        label_colors = {
            "POSITIVE": "#22c55e",
            "NEGATIVE": "#ef4444",
            "NEUTRAL": "#94a3b8",
            "MIXED": "#f59e0b",
        }
        fig_pie = go.Figure(
            go.Pie(
                labels=sentiment_df["sentiment_label"],
                values=sentiment_df["count"],
                hole=0.55,
                marker_colors=[
                    label_colors.get(l, "#64748b")
                    for l in sentiment_df["sentiment_label"]
                ],
                textinfo="percent+label",
                textfont=dict(color="#1e293b", size=11),
            )
        )
        fig_pie.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_pie, width="stretch")

# Aliases tag cloud
st.markdown(
    "<p style='font-size:0.85rem; font-weight:600; color:#94a3b8; margin-top:16px;'>Known Aliases</p>",
    unsafe_allow_html=True,
)
if aliases:
    tags_html = ", ".join(
        f"<span class='tag-pill {a['source_type']}'>{a['alias_text']}</span>"
        for a in aliases
    )
    st.markdown(tags_html, unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.72rem; color:#475569; margin-top:6px;'>"
        "🔵 manual &nbsp; 🟣 slm &nbsp; 🟠 api</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<span style='color:#475569; font-size:0.85rem;'>No aliases registered yet.</span>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border-color:#1e293b; margin:24px 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Timeline
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📈 Sentiment Timeline")

timeline_df = get_entity_timeline(entity_id)
if not timeline_df.empty:
    fig_tl = go.Figure()

    # 1. ปรับสี Bar เป็นสีเทาเข้ม/ฟ้าทึบ เพื่อให้เด่นบนพื้นขาว
    fig_tl.add_trace(
        go.Bar(
            x=timeline_df["date"],
            y=timeline_df["mention_count"],
            name="Mentions",
            marker_color="#94a3b8",  # สีเทา slate อ่อน (เพื่อให้ดูสะอาดตา)
            yaxis="y2",
            hovertemplate="Date: %{x}<br>Mentions: %{y}<extra></extra>",
        )
    )

    # 2. ปรับสีเส้น Scatter เป็นสีน้ำเงินที่เข้มขึ้นเล็กน้อยเพื่อให้ตัดกับพื้นขาว
    fig_tl.add_trace(
        go.Scatter(
            x=timeline_df["date"],
            y=timeline_df["avg_sentiment"],
            mode="lines+markers",
            name="Avg Sentiment",
            line=dict(color="#2563eb", width=2.5),  # สีน้ำเงินเด่นชัด
            marker=dict(size=7, color="#2563eb"),
            hovertemplate="Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>",
        )
    )

    # 3. เส้นแบ่ง Sentiment 0 (Zero Line)
    fig_tl.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")

    fig_tl.update_layout(
        height=280,
        paper_bgcolor="white",  # พื้นหลังด้านนอกเป็นสีขาว
        plot_bgcolor="white",  # พื้นหลังกราฟเป็นสีขาว
        font=dict(color="#1e293b"),  # ตัวอักษรสีน้ำเงินเข้มเกือบดำ
        yaxis=dict(
            title="Avg Sentiment",
            range=[-1.1, 1.1],
            gridcolor="#f1f5f9",  # เส้นกริดสีเทาจางมากๆ
            zerolinecolor="#f1f5f9",
        ),
        yaxis2=dict(
            title="Mentions",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            y=1.1,
            bgcolor="rgba(255,255,255,0)",
            font=dict(color="#1e293b"),
        ),
        margin=dict(l=0, r=0, t=12, b=0),
        xaxis=dict(
            gridcolor="#f1f5f9",  # เส้นกริดสีเทาจางๆ
            linecolor="#cbd5e1",  # เส้นขอบแกน X
        ),
    )
    st.plotly_chart(fig_tl, use_container_width=True)
else:
    st.info("No timeline data available for this entity.")

st.markdown("<hr style='border-color:#1e293b; margin:24px 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# B. Granular Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🔬 Granular Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sentiment Trajectory (Article Level)**")
    traj_df = get_entity_trajectory(entity_id)
    if not traj_df.empty:
        # text wrapper to truncate long headlines inside hover
        traj_df["headline_short"] = traj_df["headline"].apply(
            lambda x: (str(x)[:60] + "...") if len(str(x)) > 60 else x
        )
        fig_traj = px.line(
            traj_df,
            x="date",
            y="final_score",
            markers=True,
            hover_data={
                "date": True,
                "final_score": True,
                "headline_short": True,
                "headline": False,
            },
        )
        fig_traj.update_traces(
            marker=dict(size=6, color="#2563eb"), line=dict(color="#94a3b8", width=1.5)
        )
        fig_traj.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
        fig_traj.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#1e293b"),
            xaxis=dict(gridcolor="#f1f5f9", title=""),
            yaxis=dict(gridcolor="#f1f5f9", title="Final Score", range=[-1.1, 1.1]),
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    else:
        st.info("No trajectory data available.")

with col2:
    st.markdown("**Top Publishers for Entity**")
    pub_df = get_top_publishers_for_entity(entity_id)
    if not pub_df.empty:
        # reverse order so the highest is at the top in horizontal bar
        pub_df = pub_df.sort_values(by="article_count", ascending=True)
        fig_pub = px.bar(
            pub_df,
            x="article_count",
            y="publisher",
            orientation="h",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            range_color=[-1, 1],
            hover_data=["avg_sentiment", "article_count"],
        )
        fig_pub.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#1e293b"),
            xaxis=dict(gridcolor="#f1f5f9", title="Article Count"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_pub, use_container_width=True)
    else:
        st.info("No publisher data available.")

st.markdown("<br>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    st.markdown("**Sentiment Confidence Histogram**")
    conf_df = get_confidence_distribution_for_entity(entity_id)
    if not conf_df.empty:
        fig_conf = px.histogram(
            conf_df, x="confidence_score", nbins=10, color_discrete_sequence=["#8b5cf6"]
        )
        fig_conf.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#1e293b"),
            xaxis=dict(title="Confidence Score"),
            yaxis=dict(gridcolor="#f1f5f9", title="Count"),
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("No confidence data available.")

with col4:
    st.markdown("**Speaker Network (Who talks about them)**")
    speaker_df = get_speaker_network_for_entity(entity_id)
    if not speaker_df.empty:
        fig_speak = px.scatter(
            speaker_df,
            x="avg_sentiment",
            y="mention_count",
            text="speaker_name",
            size="mention_count",
            size_max=40,
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            range_color=[-1, 1],
            hover_data={
                "speaker_name": True,
                "mention_count": True,
                "avg_sentiment": True,
            },
        )
        fig_speak.update_traces(
            textposition="top center", textfont=dict(color="#1e293b")
        )
        fig_speak.add_vline(x=0, line_dash="dot", line_color="#cbd5e1")
        fig_speak.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#1e293b"),
            xaxis=dict(gridcolor="#f1f5f9", title="Avg Sentiment", range=[-1.2, 1.2]),
            yaxis=dict(gridcolor="#f1f5f9", title="Mention Count"),
        )
        st.plotly_chart(fig_speak, use_container_width=True)
    else:
        st.info("No speaker network data available.")

st.markdown("<hr style='border-color:#1e293b; margin:24px 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# C. CoT Explorer & Contextual Snippets
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### C. CoT Explorer & Contextual Snippets")

details = get_analysis_details_for_entity(entity_id, limit=10)

if not details:
    st.info("No analysis details found for this entity.")
else:
    sent_badge = {
        "POSITIVE": "🟢 POSITIVE",
        "NEGATIVE": "🔴 NEGATIVE",
        "NEUTRAL": "⚪ NEUTRAL",
        "MIXED": "🟡 MIXED",
    }

    for d in details:
        label = d.get("raw_sentiment", "NEUTRAL")
        badge = sent_badge.get(label, f"⚪ {label}")
        headline_tag = d.get("is_headline") or ""
        pub_at = d.get("published_at", "")[:10] if d.get("published_at") else "—"
        source = d.get("headline") or d.get("source_url") or "Unknown source"
        conf = d.get("confidence_score", 1.0) or 1.0

        with st.expander(
            f"{badge} - {headline_tag}  {pub_at}  -  {source[:80]}",
            expanded=False,
        ):
            # Snippet with <target> highlighted
            raw_snippet = d.get("sentence_text", "")
            highlighted = raw_snippet.replace(
                "<target>",
                "<mark style='background:#1d4ed8; color:#e0f2fe; border-radius:4px; padding:1px 4px;'>",
            ).replace("</target>", "</mark>")

            st.markdown("**📌 Context Snippet**")
            st.markdown(
                f"<div style='background:#152035; border-left:3px solid #3b82f6; "
                f"padding:12px 16px; border-radius:0 8px 8px 0; font-size:0.9rem; "
                f"line-height:1.7; color:#cbd5e1;'>{highlighted}</div>",
                unsafe_allow_html=True,
            )

            # Confidence bar
            st.markdown(
                f"<p style='font-size:0.8rem; color:#64748b; margin:12px 0 4px;'>"
                f"Confidence Score : <b>{conf:.2%}</b></p>",
                unsafe_allow_html=True,
            )
            st.progress(float(conf))

            # Reasoning / Chain-of-Thought
            st.markdown("**🧠 SLM Reasoning (Chain-of-Thought)**")
            reasoning = d.get("reasoning_parsed", {})
            if reasoning:
                # Display key CoT fields from ABSAOutput schema
                cot_fields = {
                    "target_relevance": "Target Relevance",
                    "speaker_type": "Speaker Type",
                    "speaker_name": "Speaker Name",
                    "sentiment": "Sentiment",
                    "reasoning": "Reasoning",
                    "aspects": "Aspects",
                }
                for key, label_text in cot_fields.items():
                    val = reasoning.get(key)
                    if val is None:
                        continue
                    if isinstance(val, dict):
                        val = val.get("value", val)
                    if isinstance(val, list):
                        val = ", ".join(
                            str(v.get("value", v) if isinstance(v, dict) else v)
                            for v in val
                        )
                    st.markdown(
                        f"<div style='display:flex; gap:12px; margin-bottom:6px;'>"
                        f"<span style='color:#64748b; font-size:0.8rem; min-width:130px;'>"
                        f"{label_text}</span>"
                        f"<span style='font-size:0.85rem;'>{val}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Full JSON toggle
                with st.expander("Raw JSON output", expanded=False):
                    st.json(reasoning, expanded=False)
            else:
                st.code(d.get("reasoning", "—"), language="json")

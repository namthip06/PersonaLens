"""
pages/1_📊_Executive_Dashboard.py
===================================
Global view: Sentiment Velocity · Top-Mentioned Leaderboard · Publisher Bias Radar
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.data import (
    get_publisher_bias,
    get_sentiment_velocity,
    get_top_mentioned,
    get_daily_mention_volume,
    get_language_diversity,
    get_sentiment_distribution,
    get_conflict_support_index,
    get_entity_cooccurrence,
)

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='font-size:1.9rem; font-weight:700; margin-bottom:4px;'>
        📊 Executive Dashboard
    </h1>
    <p style='color:#64748b; margin-top:0;'>
        Macro-level sentiment signals across all tracked entities and publishers.
    </p>
    <hr style='border-color:#e2e8f0; margin-bottom:24px;'>
    """,
    unsafe_allow_html=True,
)

# ── Plotly base template ──────────────────────────────────────────────────────
_LIGHT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_color="#1e293b",
    xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#cbd5e1"),
    yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#cbd5e1"),
)

# ─────────────────────────────────────────────────────────────────────────────
# 0. System Overview
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### System Overview")
col_vol, col_lang = st.columns([2, 1])

with col_vol:
    vol_df = get_daily_mention_volume()
    if vol_df.empty:
        st.info("No volume data yet.")
    else:
        fig_vol = px.area(
            vol_df,
            x="date",
            y="volume",
            labels={"date": "Date", "volume": "Number of Articles"},
            title="Daily Mention Volume",
        )
        fig_vol.update_traces(line_color="#2563eb", fillcolor="rgba(37,99,235,0.2)")
        fig_vol.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=320, **_LIGHT)
        st.plotly_chart(fig_vol, use_container_width=True)

with col_lang:
    lang_df = get_language_diversity()
    if lang_df.empty:
        st.info("No language data yet.")
    else:
        fig_lang = px.pie(
            lang_df,
            names="lang",
            values="count",
            title="Language Diversity",
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        fig_lang.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=320, **_LIGHT)
        st.plotly_chart(fig_lang, use_container_width=True)

st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# A. Sentiment Velocity Chart
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Sentiment Velocity")

col_ctrl1, col_ctrl2 = st.columns([1, 3])
with col_ctrl1:
    resample = st.selectbox(
        "Resample by",
        ["Day", "Week", "Month"],
        key="velocity_resample",
    )

velocity_df = get_sentiment_velocity()

if velocity_df.empty:
    st.info("No sentiment data yet. Ingest some articles via the Admin page.")
else:
    # Resample
    freq_map = {"Day": "D", "Week": "W", "Month": "ME"}
    freq = freq_map[resample]

    frames = []
    for cat, grp in velocity_df.groupby("entity_category"):
        grp = (
            grp.set_index("date")
            .resample(freq)["average_sentiment"]
            .mean()
            .reset_index()
        )
        grp["entity_category"] = cat
        frames.append(grp)

    if frames:
        resampled = pd.concat(frames, ignore_index=True)

        color_map = {
            "PER": "#2563eb",
            "ORG": "#7c3aed",
            "LOC": "#059669",
            "GPE": "#d97706",
        }

        fig_vel = px.line(
            resampled,
            x="date",
            y="average_sentiment",
            color="entity_category",
            color_discrete_map=color_map,
            markers=True,
            labels={
                "date": "Date",
                "average_sentiment": "Avg. Sentiment Score",
                "entity_category": "Category",
            },
        )
        fig_vel.add_hline(
            y=0,
            line_dash="dot",
            line_color="#cbd5e1",
            annotation_text="Neutral",
            annotation_font_color="#64748b",
        )
        fig_vel.update_traces(line=dict(width=2.5), marker=dict(size=6))
        fig_vel.update_layout(
            height=360,
            margin=dict(l=0, r=0, t=12, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
            ),
            **_LIGHT,
        )
        st.plotly_chart(fig_vel, use_container_width=True)

st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# B. Sentiment Distribution & Conflict Index
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Sentiment Distribution & Conflict Index")
col_dist, col_conf = st.columns([1, 2])

with col_dist:
    dist_df = get_sentiment_distribution()
    if dist_df.empty:
        st.info("No sentiment data yet.")
    else:
        fig_donut = px.pie(
            dist_df,
            names="sentiment_label",
            values="count",
            hole=0.5,
            title="Sentiment Distribution",
            color="sentiment_label",
            color_discrete_map={
                "POSITIVE": "#22c55e",
                "NEGATIVE": "#ef4444",
                "NEUTRAL": "#94a3b8",
                "MIXED": "#f59e0b",
            },
        )
        fig_donut.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=380, **_LIGHT)
        st.plotly_chart(fig_donut, use_container_width=True)

with col_conf:
    conf_df = get_conflict_support_index(limit=60)
    if conf_df.empty:
        st.info("No conflict index data yet.")
    else:
        fig_scatter = px.scatter(
            conf_df,
            x="volume",
            y="volatility",
            size="volume",
            color="category",
            hover_name="canonical_name",
            labels={
                "volume": "Total Mentions",
                "volatility": "Sentiment Volatility (Variance)",
            },
            title="Conflict vs. Support Index",
        )
        fig_scatter.update_layout(
            margin=dict(l=0, r=0, t=40, b=0), height=380, **_LIGHT
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# C. Top-Mentioned Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Top-Mentioned Leaderboard")

top_n = 10

top_df = get_top_mentioned(limit=top_n)

if top_df.empty:
    st.info("No mention data yet.")
else:
    # Pivot: entity × sentiment_label → count
    pivoted = top_df.pivot_table(
        index="canonical_name",
        columns="sentiment_label",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivoted["_total"] = pivoted.drop(columns="canonical_name").sum(axis=1)
    pivoted = pivoted.nlargest(top_n, "_total").sort_values("_total")

    label_colors = {
        "POSITIVE": "#22c55e",
        "NEGATIVE": "#ef4444",
        "NEUTRAL": "#94a3b8",
        "MIXED": "#f59e0b",
    }

    fig_bar = go.Figure()
    for label, color in label_colors.items():
        if label in pivoted.columns:
            fig_bar.add_trace(
                go.Bar(
                    name=label,
                    y=pivoted["canonical_name"],
                    x=pivoted[label],
                    orientation="h",
                    marker_color=color,
                    hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x}}<extra></extra>",
                )
            )

    fig_bar.update_layout(
        barmode="stack",
        height=max(320, top_n * 28),
        margin=dict(l=0, r=0, t=12, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_title="Total Mentions",
        **_LIGHT,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# D. Publisher Bias Radar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Publisher Bias Radar")

bias_df = get_publisher_bias()

if bias_df.empty or len(bias_df) < 3:
    st.info(
        "Not enough publisher data to render the radar chart. "
        "Ingest more articles from multiple publishers."
    )
else:
    publishers = bias_df["publisher"].tolist()
    scores = bias_df["average_sentiment"].tolist()

    # Close the polygon
    publishers_closed = publishers + [publishers[0]]
    scores_closed = scores + [scores[0]]

    # Normalise to [0, 1] for radar (score ∈ [-1, 1])
    scores_norm = [(s + 1) / 2 for s in scores_closed]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=scores_norm,
            theta=publishers_closed,
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line=dict(color="#2563eb", width=2),
            marker=dict(color="#2563eb", size=7),
            name="Avg. Sentiment",
            hovertemplate="<b>%{theta}</b><br>Raw score: "
            + "<br>".join([f"{p}: {s:.3f}" for p, s in zip(publishers, scores)]).split(
                "<br>"
            )[0]
            + "<extra></extra>",
        )
    )

    # Add neutral ring at 0.5
    neutral_r = [0.5] * len(publishers_closed)
    fig_radar.add_trace(
        go.Scatterpolar(
            r=neutral_r,
            theta=publishers_closed,
            mode="lines",
            line=dict(color="#cbd5e1", width=1, dash="dot"),
            name="Neutral",
            hoverinfo="skip",
        )
    )

    fig_radar.update_layout(
        polar=dict(
            bgcolor="#f8fafc",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor="#e2e8f0",
                linecolor="#e2e8f0",
            ),
            angularaxis=dict(
                gridcolor="#e2e8f0",
                linecolor="#e2e8f0",
                tickfont=dict(color="#64748b", size=11),
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
        ),
        height=480,
        margin=dict(l=60, r=60, t=40, b=40),
        **{k: v for k, v in _LIGHT.items() if k in ("paper_bgcolor", "font_color")},
    )

    col_r, col_key = st.columns([3, 1])
    with col_r:
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_key:
        st.markdown(
            """
            <div style='background:#f8fafc; border:1px solid #e2e8f0;
                        border-radius:12px; padding:16px; margin-top:24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                <p style='font-weight:600; font-size:0.85rem; margin-bottom:8px; color:#1e293b;'>Score Key</p>
                <p style='font-size:0.8rem; color:#475569; margin:2px 0;'>
                    <span style='color:#22c55e;'>▲ Outer ring</span> = Positive bias
                </p>
                <p style='font-size:0.8rem; color:#475569; margin:2px 0;'>
                    <span style='color:#94a3b8;'>— Dashed</span> = Neutral
                </p>
                <p style='font-size:0.8rem; color:#475569; margin:2px 0;'>
                    <span style='color:#ef4444;'>▼ Centre</span> = Negative bias
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Table of raw scores
        st.markdown(
            "<p style='font-size:0.8rem; font-weight:600; margin-top:16px;'>"
            "Raw averages</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            bias_df.rename(
                columns={"publisher": "Publisher", "average_sentiment": "Score"}
            ).style.format({"Score": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# E. Entity Correlation Network
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Entity Correlation Network")
st.caption(
    "Shows entities that frequently co-occur within the same articles. Larger nodes imply higher total co-occurrence."
)
co_df = get_entity_cooccurrence(limit=40)

if co_df.empty:
    st.info("No co-occurrence data currently available.")
else:
    # Basic circular layout calculation for the network graph
    nodes = list(set(co_df["source"]).union(set(co_df["target"])))
    N = len(nodes)
    node_pos = {
        n: (np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N))
        for i, n in enumerate(nodes)
    }

    edge_x = []
    edge_y = []
    for _, row in co_df.iterrows():
        x0, y0 = node_pos[row["source"]]
        x1, y1 = node_pos[row["target"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig_net = go.Figure()

    # Edges
    fig_net.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#cbd5e1"),
            hoverinfo="none",
            mode="lines",
        )
    )

    # Calculate degree
    degree = {n: 0 for n in nodes}
    for _, row in co_df.iterrows():
        degree[row["source"]] += row["weight"]
        degree[row["target"]] += row["weight"]

    node_sizes = [min(35, max(12, degree[n] * 2)) for n in nodes]
    node_x = [node_pos[n][0] for n in nodes]
    node_y = [node_pos[n][1] for n in nodes]

    # Nodes
    fig_net.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=node_sizes, color="#2563eb", line=dict(width=1.5, color="white")
            ),
            textfont=dict(size=11, color="#475569"),
        )
    )

    fig_net.update_layout(**_LIGHT)

    fig_net.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20),
        height=500,
    )
    st.plotly_chart(fig_net, use_container_width=True)

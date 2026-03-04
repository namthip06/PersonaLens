"""
app/app.py
==========
PersonaLens – Streamlit analytics dashboard.

Run with:
    streamlit run app/app.py
"""

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PersonaLens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Light sidebar ── */
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }

    /* ── Main background ── */
    .stApp {
        background: #ffffff;
        color: #1e293b;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #1e293b !important; font-size: 1.6rem !important; }

    /* ── Section headers ── */
    h1, h2, h3 { color: #0f172a !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #f1f5f9 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    .streamlit-expanderContent {
        background: #ffffff !important;
        border-radius: 0 0 8px 8px !important;
        border: 1px solid #e2e8f0;
        border-top: none;
    }

    /* ── Divider ── */
    hr { border-color: #e2e8f0; }

    /* ── Tag pills ── */
    .tag-pill {
        display: inline-block;
        background: #1d4ed8;
        color: #e0f2fe;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px 3px;
        font-weight: 500;
    }
    .tag-pill.manual  { background: #065f46; color: #d1fae5; }
    .tag-pill.slm     { background: #7c3aed; color: #ede9fe; }
    .tag-pill.api     { background: #b45309; color: #fef3c7; }

    /* ── Sentiment badge ── */
    .badge-pos { color: #22c55e; font-weight: 600; }
    .badge-neg { color: #ef4444; font-weight: 600; }
    .badge-neu { color: #64748b; font-weight: 600; }
    .badge-mix { color: #f59e0b; font-weight: 600; }

    /* ── Log window ── */
    .log-window {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #1e293b;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #f8fafc; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar branding ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 24px 0 8px 0;">
            <span style="font-size:2.5rem;">🔭</span>
            <h2 style="margin:4px 0 0 0; font-size:1.4rem; font-weight:700;
                       background: linear-gradient(90deg,#2563eb,#7c3aed);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                PersonaLens
            </h2>
            <p style="font-size:0.72rem; color:#64748b; margin:2px 0 0 0;">
                News Sentiment Intelligence
            </p>
        </div>
        <hr style="border-color:#e2e8f0; margin: 12px 0;">
        """,
        unsafe_allow_html=True,
    )

# ── Home landing ─────────────────────────────────────────────────────────────
home_html = (
    "<div style='padding: 20px 0 24px 0;'>"
    "<h1 style='font-size: 3rem; font-weight: 800; "
    "background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777); "
    "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
    "margin-bottom: 8px;'>PersonaLens Dashboard</h1>"
    "<p style='color: #475569; font-size: 1.2rem; max-width: 800px; margin-top: 0; margin-bottom: 32px; line-height: 1.6;'>"
    "AI-powered information extraction and sentiment intelligence for public figures. "
    "Transform unstructured news text into structured, actionable insights using Small Language Models (SLMs)."
    "</p>"
    "<div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 32px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);'>"
    "<h2 style='font-size: 1.6rem; color: #0f172a; margin-top: 0; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;'>"
    "<span>🎯</span> Purpose</h2>"
    "<p style='color: #475569; font-size: 1.05rem; line-height: 1.6; margin: 0;'>"
    "PersonaLens bridges the gap between raw news and actionable analytics. By reading news articles "
    "and extracting Named Entities with a specialized focus on public figures, the system dynamically resolves "
    "local aliases to canonical identities. It then performs highly targeted Aspect-Based Sentiment Analysis."
    "</p>"
    "</div>"
    "<div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 32px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);'>"
    "<h2 style='font-size: 1.6rem; color: #0f172a; margin-top: 0; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;'>"
    "<span>⚙️</span> How the Pipeline Works</h2>"
    "<div style='display: flex; flex-direction: column; gap: 16px;'>"
    "<div style='background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px;'>"
    "<h3 style='margin: 0 0 8px 0; font-size: 1.15rem; color: #2563eb;'>1. Ingestion & Preprocessing</h3>"
    "<p style='margin: 0; color: #475569; font-size: 0.95rem;'>Fetches raw articles and processes text through a MinHash LSH index to block near-duplicates.</p>"
    "</div>"
    "<div style='background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px;'>"
    "<h3 style='margin: 0 0 8px 0; font-size: 1.15rem; color: #7c3aed;'>2. Entity Intelligence & Alias Resolution</h3>"
    "<p style='margin: 0; color: #475569; font-size: 0.95rem;'>Extracts people and organizations via SLM. Unrecognized nicknames are automatically resolved via DB or web-search loop.</p>"
    "</div>"
    "<div style='background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px;'>"
    "<h3 style='margin: 0 0 8px 0; font-size: 1.15rem; color: #db2777;'>3. Aspect-Based Sentiment Analysis (ABSA)</h3>"
    "<p style='margin: 0; color: #475569; font-size: 0.95rem;'>Evaluates the speaker, confirms target relevance, and assigns precise sentiment polarity with reasoning.</p>"
    "</div>"
    "</div>"
    "</div>"
    "<div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 32px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);'>"
    "<h2 style='font-size: 1.6rem; color: #0f172a; margin-top: 0; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;'>"
    "<span>🧭</span> Usage Guide</h2>"
    "<div style='display: flex; flex-wrap: wrap; gap: 24px;'>"
    "<div style='flex: 1; min-width: 200px; border-left: 4px solid #3b82f6; padding-left: 16px;'>"
    "<h4 style='margin: 0 0 6px 0; color: #0f172a;'>📊 Dashboard</h4>"
    "<p style='margin: 0; color: #64748b; font-size: 0.9rem;'>View macro trends and sentiment velocity.</p>"
    "</div>"
    "<div style='flex: 1; min-width: 200px; border-left: 4px solid #8b5cf6; padding-left: 16px;'>"
    "<h4 style='margin: 0 0 6px 0; color: #0f172a;'>🔍 Deep-Dive</h4>"
    "<p style='margin: 0; color: #64748b; font-size: 0.9rem;'>Inspect individual canonical profiles.</p>"
    "</div>"
    "</div>"
    "</div>"
    "</div>"
)

st.markdown(home_html, unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:0.95rem; margin-top:32px;'>"
    "👈 Use the sidebar navigation to get started."
    "</p>",
    unsafe_allow_html=True,
)

# Streamlit Pages (`app/pages/`)

## Purpose (LLM Context)
This README acts as the definitive specification for the `app/pages/` directory. It enables LLM agents to fully understand the application's graphic user interface (GUI) and its routing logic without needing to read the raw Python files.

The directory houses the individual multipage application paths used by **Streamlit**. Each file in this directory sequentially appears in the main dashboard's sidebar navigation. The logic bridges the gap between the project's SQLite Database (`app/data.py` data-fetching layer) and the user's browser, constructing responsive tables, Plotly charts, and administrative pipeline controls.

---

## Folder Structure
Because Streamlit natively uses the file name to dictate the sidebar routing title and icon, the naming convention here is extremely rigid.

```text
app/pages/
├── 1_📊_Executive_Dashboard.py    # High-level aggregate visualizations
├── 2_🔍_Entity_Deep_Dive.py       # Granular tracing for specific entities
└── 3_⚙️_Admin_Pipeline.py         # ETL interactions and console logging
```

---

## How the Pipeline Works (Execution Flow)
Streamlit executes these files top-to-bottom every time a user interacts with a widget on that specific page.

1. **`3_⚙️_Admin_Pipeline.py` (Ingestion & ETL)** 
   - Acts as an operational control terminal.
   - The user inputs raw unstructured text or a URL and submits a form.
   - A custom logging handler aggressively captures all logs emitted from the Engine (`src/engine/`).
   - It bypasses data caching, orchestrating Trafilatura URL fetching, LSH deduplication, and strictly typed SLM extraction before triggering the `Database` persistence commands.
   - Shows live system health metrics (like resolution accuracy, average latency).

2. **`1_📊_Executive_Dashboard.py` (Macro Analysis)**
   - Acts as the default landing view showcasing BI aggregates.
   - Pulls widespread Database metrics through Pandas (e.g., Sentiment Velocity, Top-Mentioned leaderboards, and Publisher Bias radials).
   - Generates interactive Plotly (`px.line`, `px.scatterpolar`, etc.) views for decision-makers.

3. **`2_🔍_Entity_Deep_Dive.py` (Micro Investigation)**
   - Acts as the forensic audit page.
   - User queries a dropdown for a specific `canonical_name` from the Local DB.
   - The view populates UUID metadata, known linguistic aliases, and a granular sentence-level trajectory chart.
   - Most importantly, it visualizes the SLMs exact Chain-of-Thought (CoT) reasoning behind specific scores mapped to individual articles, explaining *why* an entity received a specific label.

---

## Public Functions
As these are structural GUI endpoints for Streamlit rather than importable service modules, **they expose no traditional public functions** for other Python files to invoke. 

Any underlying data parsing or SQL extraction occurs in the isolated `app/data.py` wrapper, which these pages import.

---

## Usage
These pages are entirely driven by the Streamlit orchestrator daemon. You do not execute these files directly with `python`.

Instead, invoke the main entrypoint file (`app/app.py`), which will automatically detect and mount this `pages/` directory into the frontend.

```bash
uv run streamlit run app/app.py
```
After executing this command, open your browser to `http://localhost:8501`. The files mapped in this directory will immediately become clickable tabs within the application's sidebar.

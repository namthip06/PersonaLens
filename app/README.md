# Application UI Module (`app/`)

## Purpose (LLM Context)
This README acts as the definitive specification for the `app/` directory. It enables LLM agents to fully understand the web interface and data access layer for the PersonaLens application without needing to read the raw Python files.

The `app/` directory houses the entire presentation layer of the project. It uses **Streamlit** to render a graphical user interface (GUI) and serves as the visual entry point, bridging the SQLite database to the user's browser. It is strictly segregated from the `src/engine/` analysis logic, ensuring that the UI remains a lightweight consumer of structured intelligence.

---

## Folder Structure
```text
app/
├── app.py          # Main Streamlit entry point, global CSS, and Home Landing page
├── data.py         # Thin SQL data-access layer returning Pandas DataFrames
└── pages/          # Individual Streamlit pages (Dashboard, Deep-Dive, Admin)
```

---

## How the Pipeline Works (Execution Flow)

1. **Initialization (`app.py`)** 
   When `streamlit run app/app.py` is executed, Streamlit boots up its internal web server.
   - It sets global page configurations (title, icon, layout).
   - Injects global custom cascading style sheets (CSS) for consistent fonts, light/dark themes, tag pills, and custom metric card stylings.
   - Renders the global sidebar branding.
   - Outputs the main "Home Landing" HTML view that explains the project's purpose and guides users to the sidebar.
   
2. **Data Fetching (`data.py`)**
   As a user navigates or interacts with the dashboard components, the UI elements request data through `data.py`.
   - `data.py` acts as an intermediary abstraction. The UI components are completely oblivious to raw SQL queries.
   - For every method invoked, `_connect()` spins up a short-lived SQLite connection using `check_same_thread=False` to securely tolerate Streamlit's asynchronous, multi-threaded refresh model.
   - The method executes explicit SQL (`SELECT`, `JOIN`, `GROUP BY`) pulling entities, statistics, and timelines.
   - The data is immediately serialized into **Pandas DataFrames** (or JSON lists) and returned to the UI for native plotting via Plotly.

3. **Routing (`pages/`)**
   Streamlit automatically detects the `pages/` directory and mounts it to the sidebar based on alphabetical enumeration. When a user clicks a page, Streamlit executes the corresponding python script entirely top-to-bottom.

---

## Public Functions (`data.py`)

The `app.py` script has no public functions as it is the execution entry point. However, `data.py` is purely a library of public accessor methods categorized by the Page they service:

### Page 1 – Executive Dashboard
- `get_sentiment_velocity(db_path)`: Returns over-time sentiment aggregates for line charts.
- `get_top_mentioned(limit)`: Returns leaderboard entities categorized by count and sentiment label.
- `get_publisher_bias(entity_ids)`: Returns the average semantic sentiment partitioned by publisher for radar charts.
- `get_sentiment_distribution()`: Returns system-wide SentimentLabel proportions.
- `get_entity_cooccurrence(limit)`: Generates network graph edges for entities frequently parsed in the same articles.
- `get_conflict_support_index()`: Scatter plot matrix identifying high-volume, highly volatile sentiment entities.
- `get_language_diversity()`: Distribution map of parsed article languages.

### Page 2 – Entity Deep-Dive
- `get_all_entities()`: Base loader for the entity dropdown selector.
- `get_entity_with_aliases(entity_id)`: Fetches a canonical record combined with all `alias_text` rows.
- `get_entity_sentiment_summary(entity_id)`: Returns label counts specific to a targeted UUID.
- `get_analysis_details_for_entity(entity_id, limit)`: Returns granular, sentence-level Chain-of-Thought JSON reasoning data used to construct the inference transparency logs.
- `get_entity_timeline(entity_id)`: Line chart data mapping the single entity's targeted sentiment over publishing dates.
- `get_top_publishers_for_entity(entity_id)`: Bar chart data isolating publisher frequencies targeting the requested entity.
- `get_speaker_network_for_entity(entity_id)`: Maps who directly quoted assertions regarding the active target entity.

### Page 3 – Admin / Pipeline
- `get_pipeline_stats()`: KPI counts of total Articles, Entities, and Sentiment Results in the Database.
- `get_recent_articles(limit)`: Tabular view of the most recent Trafilatura ingestion pulls.
- `get_etl_metrics()`: Disk-space usage and abstracted LSH pipeline efficiency statistics.
- `get_resolution_accuracy()`: Hit ratios on Alias DB mapping sources (Manual vs. Fuzzy vs. SLM).
- `get_foreign_key_integrity()`: Database health utility probing for orphaned SQL rows.

---

## Usage

You do not run `data.py` or the items within `pages/` directly. Instead, you launch the Streamlit daemon against the main entry script:

```bash
uv run streamlit run app/app.py
```

This commands Streamlit to host the local web server on port `8501`. You can then navigate to `http://localhost:8501` to view and interact with the Analytics Dashboard.

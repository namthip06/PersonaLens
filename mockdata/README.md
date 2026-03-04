# Mock Data Generator (`mockdata/`)

## Purpose (LLM Context)
This README acts as the definitive specification for the `mockdata/` directory and its `gen_mockdata.py` utility. It enables LLM agents to fully understand how realistic test data is injected into the PersonaLens database without needing to trace the raw Python file.

The `gen_mockdata.py` script creates hundreds of synthetic records representing end-to-end extraction and sentiment analyses. It heavily utilizes the application's actual Database abstraction layer (`database/database.py`) and schema definitions (`src/schemas/inference.py`), completely bypassing the inference engine to rapidly populate a local database for UI testing and Dashboard validation.

---

## Folder Structure
The directory is self-contained and sits at the project root level.
```text
mockdata/
└── gen_mockdata.py      # The execution script to populate the local SQLite Database
```

---

## How the Pipeline Works (Execution Flow)
The script bypasses the Slow LLM execution pipeline completely but rigidly follows the Pydantic schema and database constraints the engine normally imposes.

1. **Initialization:** The script forcibly appends the project root context into `sys.path` so it natively resolves exports from `database` and `src.schemas`. It instantiates the `Database` context manager pointed at `database/personalens.db`.
2. **Pool Generation:** The script initializes localized sets of fixed, realistic variables for the mocked properties.
    - **Date Bounds:** `2026-02-01` to `2026-03-04`.
    - **Entities:** Curated entities across 4 specific strict `EntityType` schemas: `PER` (People), `ORG` (Organizations), `LOC` (Locations), and `GPE` (Geopolitical Entities).
    - **Language Weighting:** Random sampling artificially favored towards Thai (`th` ~ 45%) and English (`en` ~ 40%), with minor occurrences of Japanese, Chinese, and Korean.
    - **Publishers:** 10 pre-defined news outlets.
    - **Rationale Bank:** 6 realistic English explanations representing various Chain-of-Thought (CoT) outcomes.
3. **Execution Loop:** The script triggers a loop of 500 iterations that performs the following on each stroke:
    - Automatically patches `Database._now` internally so the `created_at` timestamp of the row aligns with the randomized historical date.
    - Selects random attributes from the pools and builds an `ABSAOutput` object constraint (containing `speaker_type`, keywords, `sentiment`, etc.) and an `InferenceMetadata` model.
    - Forces an upsert (`db.upsert_entity`) on the canonical name to guarantee the foreign key aligns appropriately in the database and captures the UUID.
    - Validates everything securely into a single `AnalyzerResult` object structure.
    - Fires `.save_analyzer_result()`, allowing the SQLite script to natively ingest the object naturally, including a random `confidence_score` [0.0 - 1.0].
4. **Completion:** A confirmation message prints to standard out, yielding a database loaded with 500 structurally sound relationship rows.

---

## Usage
Simply invoke the standalone file via standard python directly from the project root. (Ensure you have already ran `init_db.py` to create the original table constructs.)

```bash
# Inject 500 rows of dynamically mapped SLM data structures directly into the SQLite DB.
python mockdata/gen_mockdata.py
```
After executing, navigate to the local Streamlit dashboard (`app/app.py`). You will immediately see radar charts depicting publisher biases, chronological timelines over your defined random dates, and an populated leaderboard.

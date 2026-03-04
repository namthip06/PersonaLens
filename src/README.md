# Source Directory (`src/`)

## Purpose
This README provides an overview of the `src/` directory, which houses the primary execution logic, configurations, and internal utilities for **PersonaLens**. 

The `src` module isolates the core inference, parsing, schema validation, and database abstraction layers of the application from frontend logic (located in `app/`) and database persistence scripts (located in `database/`).

---

## Folder Structure

The `src/` directory follows a modularized, domain-driven structure.

```text
src/
├── __init__.py          # Marks the directory as a Python package
├── engine/              # Inference Engine: core NLP pipelines
│   ├── preprocessor.py  # Cleans articles and blocks duplicates
│   ├── entity_linker.py # Orchestrates NER extraction
│   ├── alias_resolver.py# Fast DB exact & fuzzy lookup
│   ├── external_validator.py # DuckDuckGo validation
│   ├── analyzer.py      # Multi-task Aspect-Based Sentiment Analysis
│   └── slm_client.py    # Structured Ollama SDK wrapper
├── schemas/             # Strict Data Constraints & Models
│   └── inference.py     # Pydantic v2 classes/enums (e.g. NEROutput, ABSAOutput)
└── utils/               # Helpers & static configurations
    ├── prompts.py       # Central module to load YAML templates securely
    └── prompts/         # Static templates (e.g. ner_v1.yaml, absa_analysis.yaml)
```

---

## Component Breakdown

### 1. Engine (`src/engine/`)
The `engine` is the operational backbone of PersonaLens. It runs the overarching Multi-Step Natural Language Processing pipeline:
1. **Preprocessing & Ingestion:** Fetches text via Trafilatura, strips invalid characters, and uses MinHash Locality-Sensitive Hashing to deduplicate articles instantly.
2. **Entity Linking & Extraction:** Leverages SLM (via `slm_client.py`) to systematically extract named targets (PER, ORG, LOC, GPE) using zero-shot prompts. It features a robust fallback strategy to map nicknames/aliases securely into local UUID identities via Exact Lookups, Levenshtein Fuzzy matching (rapidfuzz), or internet-crawled fallback validation.
3. **Sentiment Engine (ABSA):** Builds localized character-window contexts, injects `<target>` tags, and performs simultaneous Multi-Task querying to retrieve speaker types, direct relevance, sentiment metrics, and granular rationales from the model.

[Read the detailed Engine Documentation here](./engine/README.md)

### 2. Schemas (`src/schemas/`)
To combat generative AI hallucination, the `schemas` directory enforces rigid data structures across the application. Utilizing **Pydantic v2**:
- It supplies `.model_json_schema()` to constrain the parameter space of Ollama's `format=` argument directly at inference.
- It parses returning JSON blobs back into safe, strongly-typed Python objects.
- Contains essential enumerations like `EntityType` (PER | ORG | LOC | GPE) and `SentimentLabel` (POSITIVE | NEGATIVE | NEUTRAL | MIXED) to govern analysis thresholds rigidly.

[Read the detailed Schemas Documentation here](./schemas/README.md)

### 3. Utils (`src/utils/`)
The `utils` module hosts detached supporting infrastructure. Primarily, it contains the Prompt Utility logic:
- Prompts are detached from pure Python source code and stored as static, standalone `.yaml` files in the `src/utils/prompts/` sub-directory.
- Using `load_prompt()`, the internal orchestrators fetch dynamic runtime-ready text templates and model target data safely and on the fly without risking arbitrary code manipulation.

[Read the detailed Utils Documentation here](./utils/README.md)

# Inference Schemas (`src/schemas/inference.py`)

## Purpose (LLM Context)
This README acts as the definitive specification for `src/schemas/inference.py`. It enables LLM agents to fully understand the input/output models used for SLM (Small Language Model - Ollama) inference pipelines without needing to read the raw Python file.

The file leverages **Pydantic v2** to serve two critical roles:
1. Generate JSON Schemas (`.model_json_schema()`) to physically constrain Ollama generation ensuring it returns strictly structured JSON.
2. Validate and deserialize the raw JSON strings back into strongly typed Python objects (`.model_validate_json()`).

---

## Folder Structure
The file sits in the schemas directory, isolated from the execution engine.
```text
src/schemas/
└── inference.py    # Centralized Pydantic models & Enums for data validation
```

---

## How the Pipeline Works (Execution Flow)
The schema acts as the boundary between the unstructured text generation of an SLM and the strict operational logic of Python.
1. **Schema Generation:** Before making an API call to Ollama, the pipeline calls `.model_json_schema()` on a target Pydantic model (e.g., `NEROutput` or `ABSAOutput`).
2. **SLM Constraining:** The generated JSON schema is passed into Ollama's `format=` parameter. This forcibly limits the model's token probabilities so it can only generate JSON matching the schema fields and Enum limits.
3. **Response Validation & Deserialization:** The raw output string returned by Ollama is passed into the `.model_validate_json()` function matching the target class.
4. **Safety Verification:** 
   - If the model deviated (e.g., missed a required key, output `"ALIEN"` instead of `"ORG"` for entity type), Pydantic catches it and raises a `ValidationError`.
   - If successful, it produces a reliable Python object (e.g., `NEROutput` instance) that the downstream code can safely interact with without nested dictionary lookups.

---

## Public Models (Public Functions / Classes)

Since this is a schema file, the public API consists of **Pydantic Models** and **Enums** instead of standalone functions. They are broken down into three phases:

### 1. Core NER schemas (Phase I: Extraction)
- **`EntityType` (Enum):** Constrains the entity categories to `PER` (Person), `ORG` (Organisation), `LOC` (Location), and `GPE` (Geopolitical Entity). Prevents hallucinations.
- **`ExtractedEntity` (BaseModel):** Represents a single entity mention.
  - `surface_form` (`str`): Exact text span.
  - `entity_type` (`EntityType`): Restricted category.
  - `context_clue` (`str`): Evidence/reasoning for the extraction.
- **`NEROutput` (BaseModel):** The payload format strictly enforced onto Ollama for extraction.
  - `entities` (`list[ExtractedEntity]`): An ordered list of entities found.
- **`InferenceMetadata` (BaseModel):** Tracks provenance (i.e. model used, prompt ID, duration).
- **`NERInferenceResult` (BaseModel):** The top-level wrapper returned to the client combining `data` (`NEROutput`) and `metadata`. Includes a `.to_entity_list()` helper method.

### 2. Resolution schemas (Entity Linker)
- **`ResolutionMethod` (Literal):** Allowed strings: `"alias_exact"`, `"alias_fuzzy"`, `"external_api"`, `"unresolved"`.
- **`ExternalResolutionOutput` (BaseModel):** Constrained payload for external DuckDuckGo snippet analysis. Contains `canonical_name` and `confidence` score `[0.0-1.0]`.
- **`ResolvedEntity` (BaseModel):** Represents an entity that has gone through resolution. Contains standard entity data + `global_id` (UUID), `canonical_name`, and `confidence_score`. Includes an `.is_resolved` property.
- **`EntityLinkerResult` (BaseModel):** Top-level wrapper returning `resolved` (list of `ResolvedEntity`) and `metadata`. Includes a `.resolved_count` property.

### 3. Phase II – ABSA Analysis schemas (Analyzer)
- **`SentimentLabel` (Enum):** Constrained to `POSITIVE`, `NEGATIVE`, `NEUTRAL`, `MIXED`.
- **`SpeakerType` (Enum):** Constrained to `REPORTER`, `QUOTE`, `UNKNOWN`.
- **`ABSAOutput` (BaseModel):** The heavily constrained payload combining Task A, B, and C for analysis.
  - **Task A:** `speaker_type`, `speaker_name`.
  - **Task B:** `is_aimed_at_target` (`bool`), `targeting_keywords` (`list[str]`).
  - **Task C:** `sentiment` (`SentimentLabel`), `aspects` (`list[str]`), `rationale` (`str`).
- **`AnalyzerResult` (BaseModel):** Final top-level envelope summarizing an entity and its full ABSA results alongside `metadata` and the original `context_window`.

---

## Private Functions
*(There are no private functions or classes in this module. It exclusively contains exposed declarations to act as rigid data contracts.)*

---

## Usage

Typical usage pattern when executing an inference call against Ollama:

```python
import requests
from src.schemas.inference import NEROutput

def execute_ner_extraction(text: str):
    # 1. Pass the Schema to Ollama via `format` to strictly constrain JSON output
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": f"Extract entities based on system instructions for text: {text}",
        "format": NEROutput.model_json_schema(),
        "stream": False
    })
    
    # 2. Get Raw JSON String
    raw_json_str = response.json().get("response", "{}")
    
    # 3. Validate and Deserialize back to a strongly typed Python Object
    try:
        validated_data = NEROutput.model_validate_json(raw_json_str)
        
        # 4. Use Object-Oriented Auto-complete safely
        for entity in validated_data.entities:
            print(f"[{entity.entity_type.value}] {entity.surface_form}")
            print(f"Reasoning: {entity.context_clue}")
            
    except Exception as e:
        print(f"Validation failed due to SLM hallucination: {e}")
```

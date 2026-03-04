# Prompt Utility Module (`src/utils/prompts.py`)

## Purpose (LLM Context)
This README is intended to serve as the definitive specification for `src/utils/prompts.py`. It provides LLM agents with a complete understanding of how the module works so that **the actual code file does not need to be read**. 

The module acts as the isolated central utility for loading LLM prompt templates, configurations, and metadata from YAML files, decoupling static prompt text from Python code.

---

## Folder Structure
The script relies heavily on its relative placement in the directory structure.
```text
src/utils/
├── prompts.py                 # The script containing the load logic
└── prompts/                   # Static directory for YAML templates
    ├── absa_analysis.yaml
    ├── ner_v1.yaml
    ├── snippet_analysis_gpe.yaml
    ├── snippet_analysis_loc.yaml
    ├── snippet_analysis_org.yaml
    └── snippet_analysis_per.yaml
```

---

## How the Pipeline Works (Execution Flow)
1. **Target Identification:** A caller requests a template by string name (e.g., `"ner_v1"`).
2. **Contextual Path Resolution:** Using `Path(__file__).parent`, the script intrinsically resolves the absolute path of its own directory (`src/utils/`).
3. **Path Construction:** It appends the `prompts/` subdirectory and `{prompt_name}.yaml` to form the full target path.
4. **Validation Check:** 
   - Checks if `file_path.exists()`.
   - Halts pipeline and raises a formatted `FileNotFoundError` if absent.
5. **Ingestion & Deserialization:** The file is opened securely using `utf-8` encoding. It processes the YAML stream using `yaml.safe_load(f)`, which converts the YAML structure into a native Python dictionary while preventing arbitrary code execution.
6. **Delivery:** The resulting dictionary is returned directly to the caller. **Note:** There is no internal caching; each call performs disk I/O.

---

## 🛠️ Public Functions

### `load_prompt(prompt_name: str) -> dict`
This is the **only** public interface exposed by the module.

**Description:**
Safely loads and parses a target YAML file from the local `prompts/` directory into a Python dictionary.

**Parameters:**
- `prompt_name` (`str`): The exact base name of the desired YAML template without the extension (e.g., `"ner_v1"`).

**Returns:**
- `dict`: The full deserialized YAML configuration. The expected schema typically involves:
  - `id`: Unique identifier for the prompt.
  - `version`: Version tracking.
  - `model`: Target LLM model name (e.g., gpt-4, claude-3).
  - `templates`: Nested dictionary holding internal text blocks (e.g., `system`, `user`).

**Exceptions:**
- Raises `FileNotFoundError` if the constructed file path does not point to a valid file.

---

## Private Functions
*(There are no private functions in this module. The pipeline is simple and fully contained within the `load_prompt` function block.)*

---

## Usage
Typical implementation when working with an external LLM function involves requesting the prompt dictionary, then interpolating it with runtime variables.

```python
from src.utils.prompts import load_prompt

def test_inference_pipeline():
    # 1. Ingest Configuration
    prompt_data = load_prompt("ner_v1")
    
    # 2. Extract Data 
    target_model = prompt_data.get("model")
    system_prompt = prompt_data["templates"]["system"]
    user_prompt_template = prompt_data["templates"]["user"]
    
    # 3. Format & Execute
    formatted_user_prompt = user_prompt_template.format(input_text="Sample entity data.")
    
    # (Pseudo-code) Use extracted info in LLM calls
    # return llm_client.generate(model=target_model, prompts=[system_prompt, formatted_user_prompt])
```

**Testing the Script Directly:**
A `__main__` block is included at the bottom of the script for direct verification and local debugging. It attempts to load `ner_v1` and print key values. Run it via:
```bash
uv run src/utils/prompts.py
```

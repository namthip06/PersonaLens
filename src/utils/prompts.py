# Load Prompt Template and Prompt_ID from YAML files

import yaml
from pathlib import Path


def load_prompt(prompt_name: str):
    # Find the path of the YAML file by referencing the position of this file
    current_dir = Path(__file__).parent
    file_path = current_dir / "prompts" / f"{prompt_name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prompt file {prompt_name}.yaml not found at {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# For testing (uv run src/utils/prompts.py)
if __name__ == "__main__":
    try:
        data = load_prompt("ner_v1")
        print(f"Successfully loaded: {data['id']}")
        print(data["version"])
        print(data["model"])
        print(data["templates"]["system"])
        print(data["templates"]["user"])
    except Exception as e:
        print(f"Error: {e}")

"""
src/engine/slm_client.py
========================
Low-level Ollama client for PersonaLens.

Wraps the official `ollama` Python SDK and adds:
  - Structured JSON output via Pydantic schema (format= parameter)
  - Automatic response validation with `.model_validate_json()`
  - Timing instrumentation written back into InferenceMetadata
  - A thin retry / error-handling surface

Usage
-----
    from src.engine.slm_client import SLMClient
    from src.schemas.inference import NEROutput

    client = SLMClient(model="qwen2.5:7b")
    result = client.chat_structured(
        system_prompt="You are an expert NER engine ...",
        user_prompt="Text: ณัฐวุฒิ ใสยเกื้อ ...",
        schema=NEROutput,
    )
    # result is a validated NEROutput instance (or raises on bad response)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Type, TypeVar

from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Generic type variable so the return type mirrors the schema passed in
T = TypeVar("T", bound=BaseModel)

# Default Ollama host – can be overridden via the constructor
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class SLMInferenceError(Exception):
    """Raised when the SLM call fails or returns an invalid response."""


class SLMClient:
    """
    Stateless wrapper around the Ollama Python SDK.

    Parameters
    ----------
    model   : Ollama model tag to use by default (e.g. "qwen2.5:7b", "mistral")
    host    : Base URL of the Ollama server
    options : Extra model options forwarded to every call (e.g. temperature)
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = DEFAULT_OLLAMA_HOST,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.options = options or {"temperature": 0}  # deterministic by default
        self._client = Client(host=host)
        logger.info("SLMClient initialised → model=%s host=%s", model, host)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        model: str | None = None,
    ) -> T:
        """
        Send a chat request to Ollama and return a VALIDATED Pydantic object.

        How it works
        ------------
        1. Pass `schema.model_json_schema()` to `format=` so Ollama is
           constrained to emit JSON matching the schema structure.
        2. Parse the raw response string into the Pydantic model via
           `schema.model_validate_json()`.  This raises `ValidationError` if
           the SLM hallucinated a field name or wrong type.

        Parameters
        ----------
        system_prompt : Instructions / persona for the model
        user_prompt   : The main user turn (article text, etc.)
        schema        : A Pydantic BaseModel subclass whose JSON schema will
                        constrain the model's output format
        model         : Override the instance-default model for this call only

        Returns
        -------
        An instance of `schema` populated with the model's response.

        Raises
        ------
        SLMInferenceError
            If the Ollama API call fails OR the response fails Pydantic validation.
        """
        effective_model = model or self.model
        t0 = time.monotonic()

        try:
            response = self._client.chat(
                model=effective_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=schema.model_json_schema(),  # ← constrains output structure
                options=self.options,
            )
        except ResponseError as exc:
            raise SLMInferenceError(
                f"Ollama API error (model={effective_model}): {exc}"
            ) from exc
        except Exception as exc:
            raise SLMInferenceError(f"Unexpected error calling Ollama: {exc}") from exc

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        raw_content: str = response.message.content

        logger.debug(
            "SLM raw response [%dms] model=%s:\n%s",
            elapsed_ms,
            effective_model,
            raw_content,
        )

        # Validate + deserialise into the requested Pydantic schema
        try:
            parsed = schema.model_validate_json(raw_content)
        except ValidationError as exc:
            raise SLMInferenceError(
                f"Response from Ollama did not match schema {schema.__name__}:\n"
                f"  raw={raw_content!r}\n"
                f"  errors={exc.errors()}"
            ) from exc

        logger.info(
            "SLM inference OK → schema=%s model=%s latency=%dms",
            schema.__name__,
            effective_model,
            elapsed_ms,
        )
        return parsed, elapsed_ms

    def ping(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            self._client.list()  # lightweight endpoint
            return True
        except Exception:
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import json
    import sys
    from pathlib import Path

    root_path = Path(__file__).parent.parent.parent
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))

    try:
        from src.schemas.inference import (
            NERInferenceResult,
            NEROutput,
            ExtractedEntity,
            EntityType,
            InferenceMetadata,
        )

        print("All imports OK")
    except ImportError as e:
        print(f"Import Error: {e}")
        sys.exit(1)

    client = SLMClient(model="qwen2.5:7b")

    if not client.ping():
        print("Error: Cannot connect to Ollama server")
    else:
        mock_article = """
        อนุทิน ชาญวีรกูล หรือ เสี่ยนู หัวหน้าพรรคภูมิใจไทย 
        ลงพื้นที่ตรวจเยี่ยมสถานการณ์น้ำท่วมที่จังหวัดเชียงราย 
        ท่ามกลางการต้อนรับของข้าราชการท้องถิ่น
        """

        system_instructions = (
            "You are an expert Thai NER engine. "
            "Extract People (PER), Organizations (ORG), and Roles (ROLE). "
            "Return the result in strictly JSON format matching the schema."
        )

        try:
            ner_data, latency = client.chat_structured(
                system_prompt=system_instructions,
                user_prompt=f"Analyze this text: {mock_article}",
                schema=NEROutput,
            )

            final_result = NERInferenceResult(
                data=ner_data,
                metadata=InferenceMetadata(
                    prompt_id="ner-v1.0-test", model=client.model, duration_ms=latency
                ),
            )

            print(f"SLM Result (Latency: {latency}ms):")
            print(json.dumps(final_result.model_dump(), indent=2, ensure_ascii=False))

        except SLMInferenceError as e:
            print(f"SLM Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

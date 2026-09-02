import json

import httpx
from pydantic import ValidationError

from job_bot.llm.base import LLMProvider, SchemaT

DEFAULT_TIMEOUT = 120.0


class OllamaProviderError(RuntimeError):
    """Raised when the local Ollama server is unreachable or returns bad output."""


class OllamaProvider(LLMProvider):
    """Generic structured-output client for any model pulled into Ollama.

    Works for DeepSeek, Llama, GLM, Qwen, Mistral, or any other model the user
    runs `ollama pull <model>` for - the model name is just config, there is no
    per-model code here.
    """

    def __init__(self, model: str, base_url: str):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "format": schema.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0.2},
        }

        try:
            resp = httpx.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
        except httpx.ConnectError as e:
            raise OllamaProviderError(
                f"Could not reach Ollama at {self._base_url}. Is it running? "
                "Try `ollama serve` in another terminal."
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaProviderError("Ollama request timed out.") from e

        if resp.status_code == 404:
            raise OllamaProviderError(
                f"Model '{self._model}' is not pulled. Run `ollama pull {self._model}`."
            )
        if resp.status_code != 200:
            raise OllamaProviderError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

        try:
            body = resp.json()
            content = body["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise OllamaProviderError("Unexpected response shape from Ollama.") from e

        try:
            return schema.model_validate_json(content)
        except (ValidationError, ValueError) as e:
            raise OllamaProviderError(f"Model '{self._model}' did not return schema-valid JSON: {e}") from e

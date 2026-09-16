import json

import httpx
from pydantic import ValidationError

from job_bot.llm.base import LLMProvider, SchemaT

DEFAULT_TIMEOUT = 120.0

# A local model occasionally emits truncated/malformed JSON for no
# structural reason (seen live: qwen3:30b cutting a CoverLetter response off
# mid-string at ~1800 chars, well under any context/output limit) - a bare
# retry of the same request resolves it almost every time, the same
# reasoning as linkedin_adapter.py's _goto_with_retry() for a flaky page
# load. Not retried: ConnectError (Ollama isn't reachable at all - retrying
# without the user fixing that first can't help) and a 404 (the model isn't
# pulled - same reasoning).
MAX_GENERATION_ATTEMPTS = 3


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
            # We only ever want the structured answer, never a reasoning
            # trace - on a thinking model (e.g. qwen3) this skips the hidden
            # <think> pass entirely, which is most of the latency. Ollama
            # silently ignores this on models that don't support thinking.
            "think": False,
        }

        last_error: Exception | None = None
        for _ in range(MAX_GENERATION_ATTEMPTS):
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
                last_error = e
                continue

            if resp.status_code == 404:
                raise OllamaProviderError(
                    f"Model '{self._model}' is not pulled. Run `ollama pull {self._model}`."
                )
            if resp.status_code != 200:
                last_error = OllamaProviderError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                continue

            try:
                body = resp.json()
                content = body["message"]["content"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                last_error = e
                continue

            try:
                return schema.model_validate_json(content)
            except (ValidationError, ValueError) as e:
                last_error = e
                continue

        raise OllamaProviderError(
            f"Model '{self._model}' did not return schema-valid JSON after "
            f"{MAX_GENERATION_ATTEMPTS} attempts: {last_error}"
        ) from last_error

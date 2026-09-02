import anthropic

from job_bot.llm.base import LLMProvider, SchemaT

MAX_TOKENS = 4096


class ClaudeProviderError(RuntimeError):
    """Raised when the Claude API call fails after retries, or is misconfigured."""


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise ClaudeProviderError(
                "ANTHROPIC_API_KEY is not set. Add it to app/.env, or switch "
                "LLM_PROVIDER=ollama to use a local model instead."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
            )
        except anthropic.AuthenticationError as e:
            raise ClaudeProviderError("Invalid ANTHROPIC_API_KEY.") from e
        except anthropic.PermissionDeniedError as e:
            raise ClaudeProviderError("API key lacks permission for this request.") from e
        except anthropic.NotFoundError as e:
            raise ClaudeProviderError(f"Model '{self._model}' not found.") from e
        except anthropic.RateLimitError as e:
            raise ClaudeProviderError("Rate limited by the Claude API; try again shortly.") from e
        except anthropic.APIConnectionError as e:
            raise ClaudeProviderError("Network error reaching the Claude API.") from e
        except anthropic.APIStatusError as e:
            raise ClaudeProviderError(f"Claude API error ({e.status_code}): {e.message}") from e

        if response.stop_reason == "refusal":
            raise ClaudeProviderError("Claude declined to answer this request.")

        parsed = response.parsed_output
        if parsed is None:
            raise ClaudeProviderError("Claude did not return a schema-valid response.")
        return parsed

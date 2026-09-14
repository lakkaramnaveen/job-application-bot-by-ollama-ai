"""Picks the LLMProvider implementation for settings.llm_provider - the only
place in the app that knows Claude and Ollama exist; everything else just
holds an LLMProvider and calls generate_structured() (see llm/base.py).
"""

from job_bot.config import Settings
from job_bot.llm.base import LLMProvider
from job_bot.llm.claude_provider import ClaudeProvider
from job_bot.llm.ollama_provider import OllamaProvider


def get_provider(settings: Settings) -> LLMProvider:
    """settings.llm_provider is itself constrained to "claude"/"ollama" by a
    pydantic pattern (see Settings), so the ValueError below is unreachable
    through normal config - it's a guard against constructing an invalid
    Settings by hand (as a test might) rather than a real runtime path.
    """
    if settings.llm_provider == "claude":
        return ClaudeProvider(api_key=settings.anthropic_api_key, model=settings.claude_model)
    if settings.llm_provider == "ollama":
        return OllamaProvider(model=settings.ollama_model, base_url=settings.ollama_base_url)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")

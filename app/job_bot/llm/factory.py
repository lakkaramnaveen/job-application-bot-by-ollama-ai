from job_bot.config import Settings
from job_bot.llm.base import LLMProvider
from job_bot.llm.claude_provider import ClaudeProvider
from job_bot.llm.ollama_provider import OllamaProvider


def get_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "claude":
        return ClaudeProvider(api_key=settings.anthropic_api_key, model=settings.claude_model)
    if settings.llm_provider == "ollama":
        return OllamaProvider(model=settings.ollama_model, base_url=settings.ollama_base_url)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")

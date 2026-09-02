import pytest

from job_bot.config import Settings
from job_bot.llm.claude_provider import ClaudeProvider
from job_bot.llm.factory import get_provider
from job_bot.llm.ollama_provider import OllamaProvider


def test_factory_returns_claude_provider(tmp_path):
    settings = Settings(_env_file=None, llm_provider="claude", anthropic_api_key="sk-ant-fake")
    provider = get_provider(settings)
    assert isinstance(provider, ClaudeProvider)


def test_factory_returns_ollama_provider(tmp_path):
    settings = Settings(_env_file=None, llm_provider="ollama")
    provider = get_provider(settings)
    assert isinstance(provider, OllamaProvider)


def test_factory_rejects_unknown_provider():
    settings = Settings(_env_file=None, llm_provider="claude")
    settings.llm_provider = "not-a-real-provider"
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_provider(settings)

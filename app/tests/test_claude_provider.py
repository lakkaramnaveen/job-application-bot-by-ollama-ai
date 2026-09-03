from types import SimpleNamespace

import anthropic
import httpx
import pytest

from job_bot.llm.claude_provider import ClaudeProvider, ClaudeProviderError
from job_bot.models.schemas import JobMatchScore


def make_provider() -> ClaudeProvider:
    return ClaudeProvider(api_key="sk-ant-fake-key", model="claude-opus-5")


def test_missing_api_key_raises_clear_error():
    with pytest.raises(ClaudeProviderError, match="ANTHROPIC_API_KEY"):
        ClaudeProvider(api_key=None, model="claude-opus-5")


def test_generate_structured_returns_parsed_output(monkeypatch):
    provider = make_provider()
    expected = JobMatchScore(
        eligibility="pass",
        technical_fit=88,
        experience_fit=88,
        culture_fit=88,
        score=88,
        reasoning="Good fit",
        should_apply=True,
        missing_qualifications=[],
    )

    def fake_parse(**kwargs):
        assert kwargs["output_format"] is JobMatchScore
        return SimpleNamespace(stop_reason="end_turn", parsed_output=expected)

    monkeypatch.setattr(provider._client.messages, "parse", fake_parse)

    result = provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)

    assert result is expected


def test_refusal_stop_reason_raises(monkeypatch):
    provider = make_provider()

    def fake_parse(**kwargs):
        return SimpleNamespace(stop_reason="refusal", parsed_output=None)

    monkeypatch.setattr(provider._client.messages, "parse", fake_parse)

    with pytest.raises(ClaudeProviderError, match="declined"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)


def test_none_parsed_output_raises(monkeypatch):
    provider = make_provider()

    def fake_parse(**kwargs):
        return SimpleNamespace(stop_reason="end_turn", parsed_output=None)

    monkeypatch.setattr(provider._client.messages, "parse", fake_parse)

    with pytest.raises(ClaudeProviderError, match="schema-valid"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)


def test_authentication_error_wrapped(monkeypatch):
    provider = make_provider()

    def fake_parse(**kwargs):
        response = httpx.Response(
            401,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            json={"error": {"message": "bad key"}},
        )
        raise anthropic.AuthenticationError("bad key", response=response, body=None)

    monkeypatch.setattr(provider._client.messages, "parse", fake_parse)

    with pytest.raises(ClaudeProviderError, match="Invalid ANTHROPIC_API_KEY"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)

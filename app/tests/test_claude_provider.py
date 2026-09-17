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


def _fake_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        json={"error": {"message": "details"}},
    )


def test_permission_denied_error_wrapped(monkeypatch):
    provider = make_provider()

    def fake_parse(**kwargs):
        raise anthropic.PermissionDeniedError("nope", response=_fake_response(403), body=None)

    monkeypatch.setattr(provider._client.messages, "parse", fake_parse)

    with pytest.raises(ClaudeProviderError, match="lacks permission"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)


def test_not_found_error_wrapped(monkeypatch):
    provider = make_provider()

    def fake_parse(**kwargs):
        raise anthropic.NotFoundError("no such model", response=_fake_response(404), body=None)

    monkeypatch.setattr(provider._client.messages, "parse", fake_parse)

    with pytest.raises(ClaudeProviderError, match="claude-opus-5.*not found"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)


def test_rate_limit_error_wrapped(monkeypatch):
    provider = make_provider()

    def fake_parse(**kwargs):
        raise anthropic.RateLimitError("slow down", response=_fake_response(429), body=None)

    monkeypatch.setattr(provider._client.messages, "parse", fake_parse)

    with pytest.raises(ClaudeProviderError, match="Rate limited"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)


def test_connection_error_wrapped(monkeypatch):
    provider = make_provider()

    def fake_parse(**kwargs):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.APIConnectionError(request=request)

    monkeypatch.setattr(provider._client.messages, "parse", fake_parse)

    with pytest.raises(ClaudeProviderError, match="Network error"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)


def test_generic_status_error_wrapped_with_status_code(monkeypatch):
    """Anything not covered by the specific subclasses above (a 5xx, say)
    still surfaces the actual status code and message rather than a blank
    "something went wrong" - useful when a brand-new Claude API error type
    shows up that this module doesn't special-case yet.
    """
    provider = make_provider()

    def fake_parse(**kwargs):
        raise anthropic.APIStatusError("server exploded", response=_fake_response(500), body=None)

    monkeypatch.setattr(provider._client.messages, "parse", fake_parse)

    with pytest.raises(ClaudeProviderError, match=r"Claude API error \(500\): server exploded"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)

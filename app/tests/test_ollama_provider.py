import subprocess

import httpx
import pytest
import respx

from job_bot.llm.ollama_provider import OllamaProvider, OllamaProviderError, quit_ollama
from job_bot.models.schemas import JobMatchScore

BASE_URL = "http://localhost:11434"


def make_provider() -> OllamaProvider:
    return OllamaProvider(model="deepseek-r1:8b", base_url=BASE_URL)


@respx.mock
def test_generate_structured_parses_valid_json():
    provider = make_provider()
    content = (
        '{"eligibility": "pass", "technical_fit": 75, "experience_fit": 75, "culture_fit": 75, '
        '"score": 75, "reasoning": "decent", "should_apply": true, "missing_qualifications": []}'
    )
    payload = {"message": {"role": "assistant", "content": content}}
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(200, json=payload))

    result = provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)

    assert isinstance(result, JobMatchScore)
    assert result.score == 75
    assert result.should_apply is True


@respx.mock
def test_model_not_pulled_raises_helpful_error():
    provider = make_provider()
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(404, text="model not found"))

    with pytest.raises(OllamaProviderError, match="ollama pull"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)


@respx.mock
def test_connection_error_raises_helpful_message():
    provider = make_provider()
    respx.post(f"{BASE_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(OllamaProviderError, match="Is it running"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)


@respx.mock
def test_invalid_json_raises_helpful_error():
    provider = make_provider()
    payload = {"message": {"role": "assistant", "content": "not json at all"}}
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(200, json=payload))

    with pytest.raises(OllamaProviderError, match="schema-valid"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)


@respx.mock
def test_retries_once_after_truncated_json_then_succeeds():
    """Real failure this guards against: qwen3:30b cut a CoverLetter
    response off mid-string with no structural cause (well under any
    context/output limit) - seen live. A bare retry of the same request
    resolved it, so generate_structured() must retry instead of failing the
    whole application prep on one flaky generation.
    """
    provider = make_provider()
    bad_payload = {"message": {"role": "assistant", "content": '{"body": "Dear hiring team, I am'}}
    good_content = '{"body": "Dear hiring team, I am excited to apply."}'
    good_payload = {"message": {"role": "assistant", "content": good_content}}
    route = respx.post(f"{BASE_URL}/api/chat").mock(
        side_effect=[httpx.Response(200, json=bad_payload), httpx.Response(200, json=good_payload)]
    )

    from job_bot.models.schemas import CoverLetter

    result = provider.generate_structured(system="sys", prompt="prompt", schema=CoverLetter)

    assert result.body == "Dear hiring team, I am excited to apply."
    assert route.call_count == 2


@respx.mock
def test_gives_up_after_max_attempts_of_truncated_json():
    provider = make_provider()
    payload = {"message": {"role": "assistant", "content": "not json at all"}}
    route = respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(200, json=payload))

    with pytest.raises(OllamaProviderError, match="after 3 attempts"):
        provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)

    assert route.call_count == 3


@respx.mock
def test_timeout_retries_then_succeeds():
    """A slow/stuck generation (httpx.TimeoutException, distinct from
    ConnectError - the server is reachable but didn't respond in time)
    should be retried like any other transient failure, not raised
    immediately - Ollama under local GPU contention can legitimately be
    slow on one attempt and fine on the next.
    """
    provider = make_provider()
    content = (
        '{"eligibility": "pass", "technical_fit": 75, "experience_fit": 75, "culture_fit": 75, '
        '"score": 75, "reasoning": "decent", "should_apply": true, "missing_qualifications": []}'
    )
    good_payload = {"message": {"role": "assistant", "content": content}}
    route = respx.post(f"{BASE_URL}/api/chat").mock(
        side_effect=[httpx.TimeoutException("timed out"), httpx.Response(200, json=good_payload)]
    )

    result = provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)

    assert result.score == 75
    assert route.call_count == 2


@respx.mock
def test_non_404_error_status_retries_then_succeeds():
    """A 5xx (or any non-200, non-404) response is a transient server-side
    failure, not "model isn't pulled" (404, raised immediately) or bad
    output (retried after a 200) - it gets the same retry treatment.
    """
    provider = make_provider()
    content = (
        '{"eligibility": "pass", "technical_fit": 75, "experience_fit": 75, "culture_fit": 75, '
        '"score": 75, "reasoning": "decent", "should_apply": true, "missing_qualifications": []}'
    )
    good_payload = {"message": {"role": "assistant", "content": content}}
    route = respx.post(f"{BASE_URL}/api/chat").mock(
        side_effect=[httpx.Response(500, text="internal error"), httpx.Response(200, json=good_payload)]
    )

    result = provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)

    assert result.score == 75
    assert route.call_count == 2


@respx.mock
def test_response_body_missing_message_content_retries_then_succeeds():
    """A 200 whose body doesn't even have the expected {"message":
    {"content": ...}} shape (an Ollama API change, or a malformed
    streaming remnant) must be retried like a bad completion, not crash
    with an unhandled KeyError.
    """
    provider = make_provider()
    content = (
        '{"eligibility": "pass", "technical_fit": 75, "experience_fit": 75, "culture_fit": 75, '
        '"score": 75, "reasoning": "decent", "should_apply": true, "missing_qualifications": []}'
    )
    good_payload = {"message": {"role": "assistant", "content": content}}
    route = respx.post(f"{BASE_URL}/api/chat").mock(
        side_effect=[httpx.Response(200, json={"unexpected": "shape"}), httpx.Response(200, json=good_payload)]
    )

    result = provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)

    assert result.score == 75
    assert route.call_count == 2


@respx.mock
def test_request_uses_json_schema_format():
    provider = make_provider()
    content = (
        '{"eligibility": "pass", "technical_fit": 50, "experience_fit": 50, "culture_fit": 50, '
        '"score": 50, "reasoning": "ok", "should_apply": false, "missing_qualifications": ["x"]}'
    )
    payload = {"message": {"role": "assistant", "content": content}}
    route = respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(200, json=payload))

    provider.generate_structured(system="sys", prompt="prompt", schema=JobMatchScore)

    sent_body = route.calls.last.request.content
    assert b'"format"' in sent_body
    assert b"deepseek-r1:8b" in sent_body


def test_quit_ollama_returns_true_when_a_command_succeeds(monkeypatch):
    monkeypatch.setattr("job_bot.llm.ollama_provider.platform.system", lambda: "Darwin")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        returncode = 0 if command[0] == "pkill" else 1
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr("job_bot.llm.ollama_provider.subprocess.run", fake_run)

    assert quit_ollama() is True
    # Both the AppleScript "quit app" attempt and the pkill fallback are
    # tried, regardless of the first one's outcome - either may be the one
    # that actually applies depending on how Ollama was started.
    assert len(calls) == 2


def test_quit_ollama_returns_false_when_nothing_was_running(monkeypatch):
    """Ollama already stopped (or never running) is a normal, harmless
    outcome - every command just reports a nonzero exit, not an exception.
    """
    monkeypatch.setattr("job_bot.llm.ollama_provider.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "job_bot.llm.ollama_provider.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1),
    )

    assert quit_ollama() is False


def test_quit_ollama_survives_a_missing_command(monkeypatch):
    """A command this function shells out to (osascript/pkill/taskkill) not
    existing on the current machine must never surface as a crash - this is
    a courtesy cleanup after a run that already finished its real work.
    """
    monkeypatch.setattr("job_bot.llm.ollama_provider.platform.system", lambda: "Linux")

    def raise_missing(command, **kwargs):
        raise FileNotFoundError("pkill not found")

    monkeypatch.setattr("job_bot.llm.ollama_provider.subprocess.run", raise_missing)

    assert quit_ollama() is False

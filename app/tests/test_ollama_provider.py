import httpx
import pytest
import respx

from job_bot.llm.ollama_provider import OllamaProvider, OllamaProviderError
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

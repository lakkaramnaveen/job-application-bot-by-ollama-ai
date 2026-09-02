import base64

import pytest

from job_bot.integrations.gmail_client import GmailClient, GmailClientError


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


class FakeExecutable:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def execute(self):
        if self._error:
            raise self._error
        return self._result


class FakeMessagesResource:
    def __init__(self, list_result=None, get_results: dict | None = None, list_error=None):
        self._list_result = list_result or {}
        self._get_results = get_results or {}
        self._list_error = list_error
        self.list_calls = []
        self.get_calls = []

    def list(self, userId, q, maxResults):  # noqa: N803 - matches googleapiclient's camelCase kwargs
        self.list_calls.append({"userId": userId, "q": q, "maxResults": maxResults})
        return FakeExecutable(self._list_result, error=self._list_error)

    def get(self, userId, id, format):  # noqa: N803, A002
        self.get_calls.append(id)
        return FakeExecutable(self._get_results.get(id))


class FakeUsersResource:
    def __init__(self, messages_resource):
        self._messages_resource = messages_resource

    def messages(self):
        return self._messages_resource


class FakeService:
    def __init__(self, messages_resource):
        self._users = FakeUsersResource(messages_resource)

    def users(self):
        return self._users


def make_message(msg_id: str, subject: str, sender: str, body_parts: list[dict]) -> dict:
    return {
        "id": msg_id,
        "snippet": "a snippet",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
                {"name": "Date", "value": "Mon, 1 Jan 2026 10:00:00 -0800"},
            ],
            "parts": body_parts,
        },
    }


def test_search_messages_parses_headers_and_prefers_plain_text_body(tmp_path):
    message = make_message(
        "m1",
        "Interview invite",
        "hr@acme.com",
        [
            {"mimeType": "text/html", "body": {"data": b64("<p>html</p>")}},
            {"mimeType": "text/plain", "body": {"data": b64("plain text body")}},
        ],
    )
    messages_resource = FakeMessagesResource(
        list_result={"messages": [{"id": "m1"}]},
        get_results={"m1": message},
    )
    client = GmailClient(
        credentials_path=tmp_path / "creds.json",
        token_path=tmp_path / "token.json",
        service=FakeService(messages_resource),
    )

    results = client.search_messages("query", max_results=10)

    assert len(results) == 1
    email = results[0]
    assert email.id == "m1"
    assert email.subject == "Interview invite"
    assert email.sender == "hr@acme.com"
    assert email.body_text == "plain text body"


def test_search_messages_walks_nested_multipart_for_plain_text(tmp_path):
    message = make_message(
        "m1",
        "Nested",
        "hr@acme.com",
        [
            {
                "mimeType": "multipart/mixed",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": b64("nested plain text")}},
                ],
            }
        ],
    )
    messages_resource = FakeMessagesResource(
        list_result={"messages": [{"id": "m1"}]}, get_results={"m1": message}
    )
    client = GmailClient(tmp_path / "c.json", tmp_path / "t.json", service=FakeService(messages_resource))

    results = client.search_messages("q")

    assert results[0].body_text == "nested plain text"


def test_search_messages_empty_result_returns_empty_list(tmp_path):
    messages_resource = FakeMessagesResource(list_result={})
    client = GmailClient(tmp_path / "c.json", tmp_path / "t.json", service=FakeService(messages_resource))

    assert client.search_messages("q") == []


def test_search_messages_wraps_api_errors(tmp_path):
    messages_resource = FakeMessagesResource(list_error=RuntimeError("quota exceeded"))
    client = GmailClient(tmp_path / "c.json", tmp_path / "t.json", service=FakeService(messages_resource))

    with pytest.raises(GmailClientError, match="Gmail search failed"):
        client.search_messages("q")


def test_missing_credentials_file_raises_clear_error(tmp_path):
    client = GmailClient(
        credentials_path=tmp_path / "does-not-exist.json",
        token_path=tmp_path / "token.json",
    )

    with pytest.raises(GmailClientError, match="Gmail OAuth client file not found"):
        client._load_credentials()

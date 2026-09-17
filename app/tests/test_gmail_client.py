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
    def __init__(
        self,
        list_result=None,
        get_results: dict | None = None,
        list_error=None,
        get_error=None,
    ):
        self._list_result = list_result or {}
        self._get_results = get_results or {}
        self._list_error = list_error
        self._get_error = get_error
        self.list_calls = []
        self.get_calls = []

    def list(self, userId, q, maxResults):  # noqa: N803 - matches googleapiclient's camelCase kwargs
        self.list_calls.append({"userId": userId, "q": q, "maxResults": maxResults})
        return FakeExecutable(self._list_result, error=self._list_error)

    def get(self, userId, id, format):  # noqa: N803, A002
        self.get_calls.append(id)
        if self._get_error:
            return FakeExecutable(error=self._get_error)
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


def test_fetch_message_wraps_api_errors(tmp_path):
    messages_resource = FakeMessagesResource(
        list_result={"messages": [{"id": "m1"}]}, get_error=RuntimeError("not found")
    )
    client = GmailClient(tmp_path / "c.json", tmp_path / "t.json", service=FakeService(messages_resource))

    with pytest.raises(GmailClientError, match="Failed to fetch message m1"):
        client.search_messages("q")


def test_search_messages_returns_empty_body_for_undecodable_base64(tmp_path):
    """A malformed body.data (truncated/corrupted upstream, or a Gmail API
    quirk) must degrade to an empty body_text rather than raising and
    losing the whole email - see _decode_body's except clause.
    """
    message = make_message(
        "m1", "Subj", "a@b.com", [{"mimeType": "text/plain", "body": {"data": "a"}}]
    )
    messages_resource = FakeMessagesResource(
        list_result={"messages": [{"id": "m1"}]}, get_results={"m1": message}
    )
    client = GmailClient(tmp_path / "c.json", tmp_path / "t.json", service=FakeService(messages_resource))

    results = client.search_messages("q")

    assert results[0].body_text == ""


class FakeCreds:
    def __init__(self, valid=False, expired=False, refresh_token=None):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refreshed = False

    def refresh(self, request):
        self.refreshed = True
        self.valid = True

    def to_json(self):
        return '{"fake": "creds"}'


class FakeFlow:
    def __init__(self, creds):
        self._creds = creds

    def run_local_server(self, port):
        return self._creds


def test_load_credentials_returns_existing_valid_token(tmp_path, monkeypatch):
    token_path = tmp_path / "token.json"
    token_path.write_text("{}", encoding="utf-8")
    fake_creds = FakeCreds(valid=True)
    monkeypatch.setattr(
        "job_bot.integrations.gmail_client.Credentials.from_authorized_user_file",
        lambda path, scopes: fake_creds,
    )
    client = GmailClient(tmp_path / "creds.json", token_path)

    assert client._load_credentials() is fake_creds


def test_load_credentials_refreshes_an_expired_token(tmp_path, monkeypatch):
    token_path = tmp_path / "token.json"
    token_path.write_text("{}", encoding="utf-8")
    fake_creds = FakeCreds(valid=False, expired=True, refresh_token="rt")
    monkeypatch.setattr(
        "job_bot.integrations.gmail_client.Credentials.from_authorized_user_file",
        lambda path, scopes: fake_creds,
    )
    monkeypatch.setattr("job_bot.integrations.gmail_client.Request", lambda: object())
    client = GmailClient(tmp_path / "creds.json", token_path)

    result = client._load_credentials()

    assert result is fake_creds
    assert fake_creds.refreshed
    assert token_path.read_text(encoding="utf-8") == '{"fake": "creds"}'


def test_load_credentials_runs_oauth_flow_when_no_token_exists(tmp_path, monkeypatch):
    creds_path = tmp_path / "creds.json"
    creds_path.write_text("{}", encoding="utf-8")
    token_path = tmp_path / "token.json"
    fake_creds = FakeCreds(valid=True)
    monkeypatch.setattr(
        "job_bot.integrations.gmail_client.InstalledAppFlow.from_client_secrets_file",
        lambda path, scopes: FakeFlow(fake_creds),
    )
    client = GmailClient(creds_path, token_path)

    result = client._load_credentials()

    assert result is fake_creds
    assert token_path.read_text(encoding="utf-8") == '{"fake": "creds"}'


def test_ensure_service_builds_and_caches_the_client(tmp_path, monkeypatch):
    fake_service = object()
    built_with = {}

    def fake_build(name, version, credentials):
        built_with.update(name=name, version=version, credentials=credentials)
        return fake_service

    monkeypatch.setattr("job_bot.integrations.gmail_client.build", fake_build)
    client = GmailClient(tmp_path / "c.json", tmp_path / "t.json")
    monkeypatch.setattr(client, "_load_credentials", lambda: "fake-creds")

    service = client._ensure_service()

    assert service is fake_service
    assert built_with == {"name": "gmail", "version": "v1", "credentials": "fake-creds"}
    # A second call must reuse the cached service rather than calling
    # build() (and therefore _load_credentials()) again on every request.
    assert client._ensure_service() is fake_service

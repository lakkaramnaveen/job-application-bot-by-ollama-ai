"""Thin, testable wrapper around the Gmail API.

Read-only by design (SCOPES below) - this project never sends, deletes, or
modifies email, only reads it to help populate the application tracker. See
SECURITY.md for the full data-handling posture.

The Google client libraries (google-auth, google-auth-oauthlib,
google-api-python-client) are real dependencies (declared in pyproject.toml),
so this module imports them at module load time rather than lazily - if
they're missing, the ImportError should surface immediately and clearly
rather than only when gmail-sync is actually invoked.
"""

import base64
import dataclasses
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Read-only: this project never sends, deletes, labels, or modifies mail.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailClientError(RuntimeError):
    pass


@dataclasses.dataclass
class EmailMessage:
    id: str
    subject: str
    sender: str
    date: str
    snippet: str
    body_text: str


class GmailClient:
    def __init__(
        self,
        credentials_path: Path,
        token_path: Path,
        service: Any = None,
    ):
        """`service` is an injection point for tests - pass a fake object
        implementing the same `.users().messages().list/get(...).execute()`
        chain the real googleapiclient Resource exposes, and no OAuth flow
        or network call happens.
        """
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._service = service

    def _ensure_service(self) -> Any:
        if self._service is None:
            self._service = build("gmail", "v1", credentials=self._load_credentials())
        return self._service

    def _load_credentials(self) -> Credentials:
        creds: Credentials | None = None
        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self._token_path), SCOPES)

        if creds and creds.valid:
            return creds

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not self._credentials_path.exists():
                raise GmailClientError(
                    f"Gmail OAuth client file not found at {self._credentials_path}. "
                    "See the Gmail sync section of README.md to create one in Google "
                    "Cloud Console and download it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(self._credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)

        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        self._token_path.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def search_messages(self, query: str, max_results: int = 50) -> list[EmailMessage]:
        service = self._ensure_service()
        try:
            resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        except Exception as e:  # googleapiclient raises its own HttpError subclass
            raise GmailClientError(f"Gmail search failed: {e}") from e

        return [self._fetch_message(service, m["id"]) for m in resp.get("messages", [])]

    def _fetch_message(self, service: Any, message_id: str) -> EmailMessage:
        try:
            raw = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        except Exception as e:
            raise GmailClientError(f"Failed to fetch message {message_id}: {e}") from e

        payload = raw.get("payload", {})
        headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
        return EmailMessage(
            id=message_id,
            subject=headers.get("subject", ""),
            sender=headers.get("from", ""),
            date=headers.get("date", ""),
            snippet=raw.get("snippet", ""),
            body_text=_extract_body_text(payload),
        )


def _extract_body_text(payload: dict[str, Any]) -> str:
    """Depth-first search for the first text/plain part. Gmail message
    bodies are base64url-encoded and may be nested arbitrarily under
    multipart/* parts (e.g. multipart/alternative inside multipart/mixed).
    """
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data")
        if data:
            return _decode_body(data)

    for part in payload.get("parts", []) or []:
        text = _extract_body_text(part)
        if text:
            return text

    return ""


def _decode_body(data: str) -> str:
    padding = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="replace")
    except (ValueError, UnicodeDecodeError):
        return ""

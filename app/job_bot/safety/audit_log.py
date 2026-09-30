import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Matches common secret shapes so they can never end up in the audit log even
# if a caller accidentally passes one through in a detail value.
_SECRET_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9\-_]+|Bearer\s+[A-Za-z0-9\-_.]+")
_REDACTED = "[REDACTED]"


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET_PATTERN.sub(_REDACTED, value)
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class AuditLogger:
    """Append-only, secret-redacted log of every action the bot takes.

    Intentionally records metadata only (action, job id, company, outcome) -
    never full resume text or raw LLM prompts, which may contain PII.
    """

    def __init__(self, log_path: Path):
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, action: str, **details: Any) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            "details": _redact(details),
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def read_entries(self, *, search: str | None = None, action: str | None = None) -> list[dict[str, Any]]:
        """Every logged entry, most recent first - the read counterpart to
        log() above, for `job-bot audit-log` to browse the trail log()
        writes on every action cmd_run/gmail_sync take. `job-bot doctor`'s
        own "Audit log writable" check proves this file CAN be written to;
        until this method existed there was no way to read it back short
        of grepping the raw JSONL file by hand.

        Each line is one independent JSON object (see log()) - a single
        truncated or corrupted line (e.g. a killed process mid-write) is
        skipped rather than failing the read of every other, valid line.
        This is different from the JSON *stores* elsewhere (CompanyBlacklist,
        AnswerGapStore, ResumeStore.faq_answers()), where one corrupted file
        means the whole thing degrades to empty: those are each a single
        JSON document, but this is an append-only log of independent
        entries, so a partial read is both possible and the right behavior
        - losing one bad line shouldn't hide every entry logged before or
        after it. A non-UTF-8 file (the one corruption shape a single bad
        line can't explain, since it usually affects the whole file) still
        degrades to no entries, the same as those JSON stores.

        `action` is an exact match (e.g. "applied", "skip_blacklisted",
        "gmail_sync_update" - see every audit.log(...)/failure_log.log(...)
        call site in cli.py/gmail_sync.py for the full vocabulary);
        `search` is a case-insensitive substring match against the whole
        entry (action name and every detail value) serialized to text, so
        a company name, job_id, or error message anywhere in an entry can
        be found without knowing which action logged it.

        Every returned entry's "details" is guaranteed to be a real dict,
        never missing, null, or some other JSON type a hand-edited or
        externally-authored line might carry - normalized to {} otherwise -
        so every caller (cmd_audit_log, render_audit_log_html) can safely
        call .items() on it without its own defensive check.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except (OSError, UnicodeDecodeError):
            # OSError beyond a missing file (permission denied, a directory
            # where the file should be) - unreadable is the same as empty
            # for every reader here (`job-bot audit-log`, the dashboard's
            # Audit Log view, doctor), the same fallback the JSON stores
            # use; doctor's "Audit log writable" check reports it.
            return []

        entries = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            if not isinstance(entry.get("details"), dict):
                # log() always writes "details" as a dict (the redacted
                # **details kwargs) - but this reads back whatever's
                # actually on disk, including a hand-edited or externally-
                # authored line where "details" is null, a string, or
                # anything else JSON allows. Normalizing it here means
                # every caller (cmd_audit_log's text/JSON output,
                # render_audit_log_html) can trust entry["details"] is
                # always a real dict and safely call .items() on it,
                # without each needing its own defensive check - the same
                # single-source-of-truth reasoning Tracker.missing_
                # qualifications_counts() already gets for being shared
                # by cli.py and dashboard/server.py instead of each
                # reimplementing the same logic.
                entry["details"] = {}
            if action is not None and entry.get("action") != action:
                continue
            if search is not None and search.lower() not in json.dumps(entry, default=str).lower():
                continue
            entries.append(entry)
        entries.reverse()
        return entries

    def last_search_signed_out_at(self) -> str | None:
        """The timestamp of the most recent search outcome here if that
        search found LinkedIn signed out (cmd_run logs search_error with
        signed_out=True for LinkedInSignedOut), else None. Only the latest
        search counts: a successful search after `job-bot login` clears it,
        and a search_error that wasn't a sign-out says nothing about the
        session. Shared by `job-bot doctor` and the dashboard's banner -
        neither contacts LinkedIn itself.
        """
        for entry in self.read_entries():  # most recent first
            if entry.get("action") == "search":
                return None
            if entry.get("action") == "search_error":
                return str(entry.get("timestamp", "")) if entry["details"].get("signed_out") is True else None
        return None

    def failure_kinds(self) -> list[dict[str, Any]]:
        """This log's entries grouped by kind of failure - "{action}:
        {first line of the error}", with every digit run replaced by "N" so
        the same failure on different job ids, timeouts, or line/column
        numbers groups together - most common first (ties alphabetical),
        each with its count and most recent timestamp. For `job-bot report
        --by-failure` on the failed-applications log: which failure types
        actually dominate, without reading hundreds of entries one by one.
        """
        kinds: dict[str, dict[str, Any]] = {}
        for entry in self.read_entries():  # most recent first
            action = str(entry.get("action", ""))
            error = str(entry["details"].get("error", "")).strip()
            first_line = error.splitlines()[0] if error else ""
            normalized = re.sub(r"\d+", "N", " ".join(first_line.split()))[:120]
            kind = f"{action}: {normalized}" if normalized else action
            if kind not in kinds:
                kinds[kind] = {"kind": kind, "count": 0, "last_seen": str(entry.get("timestamp", ""))}
            kinds[kind]["count"] += 1
        return sorted(kinds.values(), key=lambda k: (-k["count"], k["kind"].casefold()))

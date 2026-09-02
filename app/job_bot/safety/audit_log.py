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

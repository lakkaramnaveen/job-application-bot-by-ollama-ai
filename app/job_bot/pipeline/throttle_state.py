"""LinkedIn's refusal streak, remembered across `job-bot run` restarts.

The --loop driver backs off exponentially while LinkedIn keeps refusing
page loads (throttle_backoff_minutes()), but that streak used to live only
in the running process. Restarting the bot - which is what anyone does
when a run seems stuck - reset it, and the new run searched straight away.
Real case (2026-10-04): a run refused at 10:29 was stopped and restarted
at 12:43, and the restart's first search was refused again (HTTP 429).

The state is tiny (a streak and a timestamp), local to this machine, and
best effort: a missing or unreadable file just means "not throttled".
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from job_bot.pipeline.failures import throttle_backoff_minutes


class ThrottleState:
    def __init__(self, path: Path, *, now: Callable[[], datetime] = datetime.now):
        self._path = path
        self._now = now

    def streak(self) -> int:
        return self._read()[0]

    def resume_at(self, base_minutes: float) -> datetime | None:
        """When searching may resume, if that's still in the future - the
        last refusal plus the back-off its streak earned."""
        streak, last = self._read()
        if streak <= 0 or last is None:
            return None
        resume = last + timedelta(minutes=throttle_backoff_minutes(base_minutes, streak))
        return resume if resume > self._now() else None

    def last_refused_at(self) -> datetime | None:
        return self._read()[1]

    def record_refusal(self) -> int:
        """One more refused cycle; returns the new streak."""
        streak = self.streak() + 1
        with contextlib.suppress(OSError):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({"streak": streak, "last_refused_at": self._now().isoformat()}))
        return streak

    def clear(self) -> None:
        """A search went through - the streak is over."""
        with contextlib.suppress(OSError):
            self._path.unlink(missing_ok=True)

    def _read(self) -> tuple[int, datetime | None]:
        try:
            data = json.loads(self._path.read_text())
            return int(data["streak"]), datetime.fromisoformat(data["last_refused_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0, None

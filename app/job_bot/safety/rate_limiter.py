import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

from job_bot.config import HARD_DAILY_APPLICATION_CEILING


class DailyCapReached(RuntimeError):
    pass


class RateLimiter:
    """Persisted daily application counter.

    The configured cap is always clamped to HARD_DAILY_APPLICATION_CEILING -
    no config value, however large, can bypass that ceiling.
    """

    def __init__(self, db_path: Path, daily_cap: int):
        self._db_path = db_path
        self._daily_cap = min(daily_cap, HARD_DAILY_APPLICATION_CEILING)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS application_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    applied_on TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def count_today(self) -> int:
        today = date.today().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM application_events WHERE applied_on = ?",
                (today,),
            ).fetchone()
        return row[0] if row else 0

    def remaining_today(self) -> int:
        return max(0, self._daily_cap - self.count_today())

    def record_application(self) -> None:
        """Record one application submission. Raises DailyCapReached if over cap."""
        if self.remaining_today() <= 0:
            raise DailyCapReached(f"Daily application cap ({self._daily_cap}) reached. Try again tomorrow.")
        today = date.today().isoformat()
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO application_events (applied_on, created_at) VALUES (?, ?)",
                (today, now),
            )

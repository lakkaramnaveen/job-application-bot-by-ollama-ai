import sqlite3
from datetime import UTC, datetime
from pathlib import Path

# The single authoritative set of values the `jobs.status` column may hold.
# `seen`/`applied`/`skipped` are written by `job_bot run` itself; the rest
# are outcomes you record by hand later (`job-bot status <job_id> <status>`)
# as an application progresses - this tracker doesn't learn outcomes on its
# own, since nothing in the auto-apply flow observes them.
TRACKER_STATUSES = frozenset(
    {
        "seen",
        "applied",
        "skipped",
        "interviewing",
        "offer",
        "rejected",
        "withdrawn",
        "no_response",
    }
)


class InvalidStatus(ValueError):
    pass


class Tracker:
    """SQLite-backed record of jobs seen and applications submitted."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    url TEXT NOT NULL,
                    match_score INTEGER,
                    status TEXT NOT NULL DEFAULT 'seen',
                    first_seen_at TEXT NOT NULL,
                    applied_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def upsert_job(
        self, job_id: str, title: str, company: str, url: str, match_score: int | None = None
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, title, company, url, match_score, first_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET match_score = excluded.match_score
                """,
                (job_id, title, company, url, match_score, now),
            )

    def mark_applied(self, job_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'applied', applied_at = ? WHERE job_id = ?",
                (now, job_id),
            )

    def mark_skipped(self, job_id: str) -> None:
        self.update_status(job_id, "skipped")

    def update_status(self, job_id: str, status: str) -> None:
        """Record a status outside the automated seen/applied/skipped flow -
        e.g. `interviewing`, `offer`, `rejected` - as you hear back on an
        application. Raises InvalidStatus for anything not in
        TRACKER_STATUSES, and ValueError if job_id isn't tracked yet.
        """
        if status not in TRACKER_STATUSES:
            raise InvalidStatus(
                f"Unknown status {status!r}. Valid statuses: {', '.join(sorted(TRACKER_STATUSES))}"
            )
        with self._connect() as conn:
            cursor = conn.execute("UPDATE jobs SET status = ? WHERE job_id = ?", (status, job_id))
            if cursor.rowcount == 0:
                raise ValueError(f"No tracked job with id {job_id!r}")

    def has_applied(self, job_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return bool(row) and row[0] == "applied"

    def status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
        return dict(rows)

    def record_qa(self, job_id: str, question: str, answer: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO qa_history (job_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                (job_id, question, answer, now),
            )

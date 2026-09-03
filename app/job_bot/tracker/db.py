import contextlib
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


class InvalidSort(ValueError):
    pass


# Whitelisted so `sort` (which reaches list_jobs()/count_jobs() as a query
# string from the dashboard) can never be interpolated into SQL as anything
# other than one of these exact, known-safe column names.
SORTABLE_COLUMNS = frozenset(
    {"first_seen_at", "applied_at", "title", "company", "match_score", "status"}
)


class Tracker:
    """SQLite-backed record of jobs seen and applications submitted."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """A connection that's both transactional (commits on success, rolls
        back on exception, like `with self._connect() as conn:`) and
        actually closed afterward - sqlite3.Connection's own context manager
        only handles the transaction, never closes the connection, so a
        `with self._connect() as conn:` at every call site (as this used to
        be) leaks one open connection per call. The dashboard in particular
        constructs a fresh Tracker per HTTP request and calls several of
        these methods per request, so that leak compounds quickly under
        sustained polling.
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._transaction() as conn:
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
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, title, company, url, match_score, first_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET match_score = excluded.match_score
                """,
                (job_id, title, company, url, match_score, now),
            )

    def mark_applied(self, job_id: str) -> None:
        """Raises ValueError if job_id isn't tracked yet - silently no-oping
        here (as this used to) would let a real LinkedIn submission go
        unrecorded with no error, which is exactly the "lost from the
        tracker" failure a real submission must never suffer (see
        record_application()'s own defense-in-depth comment in
        safety/rate_limiter.py and cli.py's cmd_run).
        """
        now = datetime.now(UTC).isoformat()
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'applied', applied_at = ? WHERE job_id = ?",
                (now, job_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"No tracked job with id {job_id!r}")

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
        with self._transaction() as conn:
            cursor = conn.execute("UPDATE jobs SET status = ? WHERE job_id = ?", (status, job_id))
            if cursor.rowcount == 0:
                raise ValueError(f"No tracked job with id {job_id!r}")

    def has_applied(self, job_id: str) -> bool:
        """True once mark_applied() has ever run for this job_id - checked
        against `applied_at` rather than the current `status`, since status
        can legitimately move on afterward (`interviewing`, `offer`, ...) or
        be corrected by hand (`job-bot status`, the dashboard's inline
        status control). Keying this off `status == "applied"` instead would
        make has_applied() flip back to False the moment status changes to
        anything else, and the run loop's `if tracker.has_applied(...):
        continue` dedup check would then let a real second application go
        through for a job already applied to - reopening the exact
        duplicate-submission risk fixed in commit a853e9b.
        """
        with self._transaction() as conn:
            row = conn.execute("SELECT applied_at FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return bool(row) and row[0] is not None

    def status_counts(self) -> dict[str, int]:
        with self._transaction() as conn:
            rows = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
        return dict(rows)

    @staticmethod
    def _where_clause(status: str | None, search: str | None) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if search:
            # Escape LIKE wildcards in user input so e.g. a search for "50%"
            # matches literally rather than acting as a wildcard.
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("(title LIKE ? ESCAPE '\\' OR company LIKE ? ESCAPE '\\')")
            like = f"%{escaped}%"
            params.extend([like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def list_jobs(
        self,
        status: str | None = None,
        search: str | None = None,
        sort: str = "first_seen_at",
        direction: str = "desc",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Tracked jobs, optionally filtered by status and/or a title/company
        substring search, sorted, and paginated. Used by the dashboard and by
        gmail_sync's company matching (which relies on the no-filter default
        returning every tracked job).

        `sort` must be one of SORTABLE_COLUMNS and `direction` one of
        "asc"/"desc" - both are validated here (raising InvalidSort) rather
        than interpolated as-is, since they come from a query string.
        """
        if sort not in SORTABLE_COLUMNS:
            raise InvalidSort(f"Unknown sort column {sort!r}. Valid: {', '.join(sorted(SORTABLE_COLUMNS))}")
        if direction not in ("asc", "desc"):
            raise InvalidSort(f"Unknown sort direction {direction!r}. Valid: asc, desc")

        where, params = self._where_clause(status, search)
        query = (
            f"SELECT * FROM jobs {where} "
            f"ORDER BY {sort} {direction.upper()}, job_id {direction.upper()}"
        )
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = [*params, limit, offset]

        with self._transaction() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def count_jobs(self, status: str | None = None, search: str | None = None) -> int:
        where, params = self._where_clause(status, search)
        with self._transaction() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM jobs {where}", params).fetchone()
        return int(row[0])

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._transaction() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def record_qa(self, job_id: str, question: str, answer: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO qa_history (job_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                (job_id, question, answer, now),
            )

    def list_qa(self, job_id: str) -> list[dict[str, Any]]:
        """A job's answered-question transcript, oldest first."""
        with self._transaction() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT question, answer, created_at FROM qa_history WHERE job_id = ? ORDER BY id ASC",
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

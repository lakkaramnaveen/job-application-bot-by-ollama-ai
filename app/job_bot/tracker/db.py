"""SQLite-backed persistence for every job the bot has seen, applied to, or
been told about by hand - the single source of truth cli.py's cmd_run,
cmd_report/export/status, gmail_sync.py, and the dashboard all read and
write. See Tracker below for the schema and every read/write method; the
`_transaction()` helper each of them uses is worth reading first.
"""

import contextlib
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Outcome statuses that count as a genuine positive signal for a past
# tailored resume - see best_resume_examples().
_POSITIVE_OUTCOME_STATUSES = ("interviewing", "offer")

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    skills_json TEXT NOT NULL,
                    bullets_json TEXT NOT NULL,
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

    def record_score(
        self, job_id: str, title: str, company: str, url: str, score: int, should_apply: bool
    ) -> None:
        """Upsert a job together with its LLM match score and the resulting
        seen/skipped status, in one transaction - as opposed to an
        upsert_job() call followed by a separate mark_skipped(), which would
        leave a crash between the two calls able to record a score without
        ever recording the "don't apply" decision that went with it. cmd_run
        relies on that atomicity: on a later run, a tracked job with
        status="seen" and a non-null match_score is trusted to mean "already
        scored, and the LLM said apply" without needing to re-check should_apply.
        """
        status = "seen" if should_apply else "skipped"
        now = datetime.now(UTC).isoformat()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, title, company, url, match_score, status, first_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET match_score = excluded.match_score, status = excluded.status
                """,
                (job_id, title, company, url, score, status, now),
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

        A transition to "applied" here (gmail_sync's application_confirmation
        match, or a manual `job-bot status <id> applied`) is just as real a
        signal that an application went out as mark_applied() is - so it
        stamps applied_at too, via COALESCE so an existing timestamp (set by
        mark_applied() itself) is never overwritten. Leaving applied_at unset
        on this path would silently break has_applied()'s dedup check in the
        run loop (which keys off applied_at, not status) and `report
        --stale-days`'s follow-up nudge (which requires applied_at) for any
        job whose "applied" status came from here instead.
        """
        if status not in TRACKER_STATUSES:
            raise InvalidStatus(
                f"Unknown status {status!r}. Valid statuses: {', '.join(sorted(TRACKER_STATUSES))}"
            )
        with self._transaction() as conn:
            if status == "applied":
                cursor = conn.execute(
                    "UPDATE jobs SET status = ?, applied_at = COALESCE(applied_at, ?) WHERE job_id = ?",
                    (status, datetime.now(UTC).isoformat(), job_id),
                )
            else:
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

    def recent_qa_pairs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Up to `limit` most-recently-answered *unique* questions across
        every job, most recent first - fed into qa_answerer.py's prompt as
        informal context so answering the same or a similarly-phrased
        question on a different posting benefits from how it (or a
        previous run) answered it before, not just the curated subset that
        made it into FAQ_PATH (see resume/store.py's save_faq_answer(),
        which only promotes high-confidence, resume-grounded answers).

        Deliberately not filtered by confidence - a low-confidence or
        since-superseded answer is still useful *reference* for how a
        similar question was approached, the same way a human remembers
        their own past attempts even the ones that didn't go well. The
        caller (qa_answerer.py) is told this is informal history, not
        verified fact like FAQ_PATH, and the LLM is instructed accordingly.
        Deduplicated by question text (most recent answer per question
        wins) so this can't be dominated by one question asked on many
        postings crowding out everything else.
        """
        with self._transaction() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT question, answer FROM qa_history
                WHERE id IN (SELECT MAX(id) FROM qa_history GROUP BY question)
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_resume_generation(
        self, job_id: str, title: str, company: str, summary: str, skills: list[str], bullets: list[str]
    ) -> None:
        """Log one tailor_resume() output, so a later best_resume_examples()
        call can feed it back as a few-shot example - the closest thing to
        "learning from previous responses" a local model we never fine-tune
        can actually do (see resume_tailor.py's tailor_resume() docstring).
        """
        now = datetime.now(UTC).isoformat()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO resume_generations
                    (job_id, title, company, summary, skills_json, bullets_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, title, company, summary, json.dumps(skills), json.dumps(bullets), now),
            )

    def best_resume_examples(self, limit: int = 3) -> list[dict[str, Any]]:
        """Up to `limit` past tailored-resume generations to use as few-shot
        style/quality reference for a new one, ranked with generations tied
        to a job whose status has since become "interviewing" or "offer"
        first (a real, human-confirmed positive outcome for that resume),
        falling back to the most recent generations otherwise - e.g. on a
        fresh install where nothing has an outcome yet. Each dict has keys
        job_id, title, company, summary, skills (list[str]), bullets
        (list[str]), created_at.

        Statuses are recorded by hand via `job-bot status <job_id> <status>`
        (see TRACKER_STATUSES) - there is nothing automatic connecting an
        interview back to the resume that helped land it beyond that.
        """
        placeholders = ", ".join("?" for _ in _POSITIVE_OUTCOME_STATUSES)
        with self._transaction() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT rg.job_id, rg.title, rg.company, rg.summary, rg.skills_json, rg.bullets_json,
                       rg.created_at
                FROM resume_generations rg
                LEFT JOIN jobs j ON j.job_id = rg.job_id
                ORDER BY
                    CASE WHEN j.status IN ({placeholders}) THEN 0 ELSE 1 END,
                    rg.created_at DESC
                LIMIT ?
                """,
                (*_POSITIVE_OUTCOME_STATUSES, limit),
            ).fetchall()
        return [
            {
                "job_id": row["job_id"],
                "title": row["title"],
                "company": row["company"],
                "summary": row["summary"],
                "skills": json.loads(row["skills_json"]),
                "bullets": json.loads(row["bullets_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

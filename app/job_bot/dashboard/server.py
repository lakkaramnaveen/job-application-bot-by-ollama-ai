"""One-page tracker dashboard: a local, no-auth HTTP server bound to
localhost only (never 0.0.0.0) since it serves your application data with
no access control. See render.py for the HTML/escaping logic this wraps.

The dashboard also accepts six state-changing requests: POST status
update, POST blacklist (add), POST blacklist/remove, POST note, POST
answer-gaps/dismiss, and POST faq/remove. Because the server has no auth,
any page open in the same browser could in principle try to trigger one
(a "drive-by localhost" request) - _is_same_origin (all six) plus the
browser's own CORS preflight (triggered by the JSON Content-Type each of
them requires) are what stand in for auth here. See _is_same_origin,
_handle_status_update, _handle_blacklist, _handle_blacklist_remove,
_handle_note_set, _handle_answer_gaps_dismiss, and _handle_faq_remove
below.
"""

import io
import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, unquote, urlparse

from job_bot.dashboard.render import (
    PAGE_SIZE,
    render_answer_gaps_html,
    render_audit_log_html,
    render_blacklist_html,
    render_faq_html,
    render_missing_qualifications_html,
    render_page_html,
    render_qa_history_html,
    render_qa_html,
    render_resume_html,
    render_rows_html,
    render_stats_html,
    render_weekly_activity_html,
)
from job_bot.data_files import CorruptDataFile
from job_bot.resume.store import ResumeStore
from job_bot.safety.answer_gaps import AnswerGapStore
from job_bot.safety.audit_log import AuditLogger
from job_bot.safety.blacklist import CompanyBlacklist
from job_bot.tracker.db import (
    InvalidSort,
    InvalidStatus,
    Tracker,
    write_export_csv,
    write_export_json,
)

DASHBOARD_HOST = "127.0.0.1"

# A POST body larger than this is rejected outright - the only legitimate
# body is a tiny {"status": "..."} JSON object, so anything bigger is either
# a bug or abuse.
MAX_BODY_BYTES = 4096


class DashboardPortInUse(RuntimeError):
    """Raised when the configured dashboard port is already bound - most
    commonly a previous `job-bot dashboard` invocation still running (e.g.
    a forgotten terminal tab), or another process using it. Without this,
    ThreadingHTTPServer's own OSError ("Address already in use") propagated
    straight out as a raw traceback instead of a clean, actionable message -
    see cli.py's EXPECTED_ERRORS, which catches this the same way every
    other user-facing configuration/input problem is caught.
    """

_DEFAULT_SORT = "first_seen_at"
_DEFAULT_DIRECTION = "desc"
# Same reasoning as `job-bot report --missing-qualifications-limit`: exact-
# string counting means most distinct phrases end up at count=1 once
# enough jobs are tracked, so an unlimited panel would mostly show one-off
# noise. The dashboard has no equivalent flag to override this with, so a
# fixed cap keeps the panel readable in the common case instead of adding
# a control for a value nobody's likely to need to change interactively.
_MISSING_QUALIFICATIONS_LIMIT = 20

# Same reasoning as _MISSING_QUALIFICATIONS_LIMIT: no dashboard control to
# page or otherwise narrow this down, so a fixed cap keeps the Audit Log
# panel readable rather than dumping a run's entire, potentially very long
# history into one modal. Higher than the missing-qualifications cap since
# an audit log entry is one line, not a paragraph, and `job-bot audit-log`
# (no cap at all) is right there for anyone who needs the full history.
_AUDIT_LOG_LIMIT = 50

# Same reasoning as _AUDIT_LOG_LIMIT, for the same reason: Tracker.
# search_qa() itself has no limit (matching `job-bot qa-history`, which is
# the tool for anyone who needs the full, unbounded history), but the
# dashboard's own modal has no paging control, so a fixed cap keeps it from
# dumping every question ever recorded into one unbounded list.
_QA_HISTORY_LIMIT = 50


def _parse_list_params(query: dict[str, list[str]]) -> dict:
    """Shared query-string parsing for `/` and `/api/rows` - keeps the two
    handlers' filter/sort/page behavior identical.
    """
    status = (query.get("status", [""])[0] or "").strip()
    eligibility = (query.get("eligibility", [""])[0] or "").strip()
    search = (query.get("q", [""])[0] or "").strip()
    sort = (query.get("sort", [_DEFAULT_SORT])[0] or _DEFAULT_SORT).strip()
    direction = (query.get("dir", [_DEFAULT_DIRECTION])[0] or _DEFAULT_DIRECTION).strip()
    try:
        page = max(1, int(query.get("page", ["1"])[0]))
    except ValueError:
        page = 1
    return {
        "status": status or None,
        "eligibility": eligibility or None,
        "search": search or None,
        "sort": sort,
        "direction": direction,
        "page": page,
    }


def make_handler(
    db_path: Path,
    blacklist_path: Path,
    audit_log_path: Path,
    failed_applications_log_path: Path,
    answer_gaps_path: Path,
    resume_path: Path,
    faq_path: Path,
    *,
    stale_after_days: int = 14,
) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus | int, content_type: str, body: bytes, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, status: HTTPStatus | int, text: str) -> None:
            self._send(status, "text/plain; charset=utf-8", text.encode("utf-8"))

        def _send_json(self, status: HTTPStatus | int, payload: dict) -> None:
            self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

        def _is_same_origin(self) -> bool:
            """Reject a POST unless its Origin header matches this
            dashboard's own host:port. Real browsers attach Origin to every
            POST/PUT/DELETE fetch, same-origin or cross-origin alike, so the
            dashboard's own JS (which only ever calls fetch() from the page
            it's served from) always has one - a missing Origin header means
            the request didn't come from a browser honoring same-origin
            semantics at all (e.g. a bare HTTP client), which is exactly the
            case this check exists to reject, not a case to wave through.
            """
            origin = self.headers.get("Origin")
            if origin is None:
                return False
            port = cast(ThreadingHTTPServer, self.server).server_port
            expected = {f"http://{DASHBOARD_HOST}:{port}", f"http://localhost:{port}"}
            return origin in expected

        def _job_id_from_path(self, prefix: str, suffix: str) -> str | None:
            """Extract `{job_id}` from a path shaped like
            f"{prefix}{{job_id}}{suffix}", or None if it doesn't match.
            job_id is only ever used as a parameterized SQL value, so no
            further sanitization is needed here beyond "non-empty" - but it
            does need percent-decoding first: the frontend sends it via
            encodeURIComponent (render.py's qa-button/status-select handlers)
            so a job_id containing e.g. a space or '#' arrives as raw
            "%20"/"%23" in self.path, which must be decoded back to the
            literal value actually stored in the database before use.
            """
            path = urlparse(self.path).path
            if not (path.startswith(prefix) and path.endswith(suffix)):
                return None
            job_id = unquote(path[len(prefix) : len(path) - len(suffix)])
            return job_id or None

        def do_GET(self) -> None:  # noqa: N802 - required name for BaseHTTPRequestHandler
            parsed = urlparse(self.path)
            tracker = Tracker(db_path)

            if parsed.path == "/":
                self._handle_index(tracker, parse_qs(parsed.query))
            elif parsed.path == "/api/rows":
                self._handle_rows(tracker, parse_qs(parsed.query))
            elif parsed.path == "/api/stats":
                self._handle_stats(tracker, parse_qs(parsed.query))
            elif parsed.path == "/api/export.csv":
                self._handle_export(tracker, parse_qs(parsed.query), fmt="csv")
            elif parsed.path == "/api/export.json":
                self._handle_export(tracker, parse_qs(parsed.query), fmt="json")
            elif parsed.path == "/api/jobs":
                jobs = tracker.list_jobs()
                self._send(200, "application/json", json.dumps(jobs, default=str).encode("utf-8"))
            elif (job_id := self._job_id_from_path("/api/jobs/", "/qa")) is not None:
                self._handle_qa(tracker, job_id)
            elif (job_id := self._job_id_from_path("/api/jobs/", "/resume")) is not None:
                self._handle_resume(tracker, job_id)
            elif (job_id := self._job_id_from_path("/api/jobs/", "/note")) is not None:
                self._handle_note_get(tracker, job_id)
            elif parsed.path == "/api/blacklist":
                self._handle_blacklist_list()
            elif parsed.path == "/api/missing-qualifications":
                self._handle_missing_qualifications(tracker)
            elif parsed.path == "/api/weekly-activity":
                self._handle_weekly_activity(tracker)
            elif parsed.path == "/api/answer-gaps":
                self._handle_answer_gaps()
            elif parsed.path == "/api/faq":
                self._handle_faq_list()
            elif parsed.path == "/api/qa-history":
                self._handle_qa_history(parse_qs(parsed.query))
            elif parsed.path == "/api/audit-log":
                self._handle_audit_log(parse_qs(parsed.query))
            else:
                self._send_text(404, "Not found")

        def do_POST(self) -> None:  # noqa: N802 - required name for BaseHTTPRequestHandler
            try:
                self._dispatch_post()
            except CorruptDataFile as e:
                # The blacklist file exists but is unreadable, so the write
                # was refused rather than replacing it - tell the page why
                # instead of dropping the connection mid-request.
                self._send_text(500, str(e))

        def _dispatch_post(self) -> None:
            if (job_id := self._job_id_from_path("/api/jobs/", "/status")) is not None:
                self._handle_status_update(Tracker(db_path), job_id)
            elif (job_id := self._job_id_from_path("/api/jobs/", "/blacklist")) is not None:
                self._handle_blacklist(Tracker(db_path), job_id)
            elif (job_id := self._job_id_from_path("/api/jobs/", "/note")) is not None:
                self._handle_note_set(Tracker(db_path), job_id)
            elif urlparse(self.path).path == "/api/blacklist/remove":
                self._handle_blacklist_remove()
            elif urlparse(self.path).path == "/api/answer-gaps/dismiss":
                self._handle_answer_gaps_dismiss()
            elif urlparse(self.path).path == "/api/faq/remove":
                self._handle_faq_remove()
            else:
                self._send_text(404, "Not found")

        def _handle_index(self, tracker: Tracker, query: dict[str, list[str]]) -> None:
            params = _parse_list_params(query)
            try:
                total = tracker.count_jobs(
                    status=params["status"], search=params["search"], eligibility=params["eligibility"]
                )
                jobs = tracker.list_jobs(
                    status=params["status"],
                    search=params["search"],
                    eligibility=params["eligibility"],
                    sort=params["sort"],
                    direction=params["direction"],
                    limit=PAGE_SIZE,
                    offset=(params["page"] - 1) * PAGE_SIZE,
                )
            except InvalidSort:
                params["sort"], params["direction"] = _DEFAULT_SORT, _DEFAULT_DIRECTION
                total = tracker.count_jobs(
                    status=params["status"], search=params["search"], eligibility=params["eligibility"]
                )
                jobs = tracker.list_jobs(
                    status=params["status"],
                    search=params["search"],
                    eligibility=params["eligibility"],
                    limit=PAGE_SIZE,
                    offset=0,
                )
                params["page"] = 1
            body = render_page_html(
                jobs,
                total=total,
                page=params["page"],
                page_size=PAGE_SIZE,
                status=params["status"] or "",
                eligibility=params["eligibility"] or "",
                search=params["search"] or "",
                sort=params["sort"],
                direction=params["direction"],
                counts=tracker.status_counts(search=params["search"], eligibility=params["eligibility"]),
                stale_after_days=stale_after_days,
                signed_out_at=AuditLogger(audit_log_path).last_search_signed_out_at(),
            ).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_stats(self, tracker: Tracker, query: dict[str, list[str]]) -> None:
            params = _parse_list_params(query)
            counts = tracker.status_counts(search=params["search"], eligibility=params["eligibility"])
            body = render_stats_html(counts, params["status"] or "").encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_export(self, tracker: Tracker, query: dict[str, list[str]], *, fmt: str) -> None:
            """Same CSV/JSON shape as `job-bot export --format ...` (see
            tracker/db.py's write_export_csv/write_export_json, shared by
            both) - respects the dashboard's current status/eligibility
            filters and search box, but always exports every matching job,
            not just the currently-visible page.
            """
            params = _parse_list_params(query)
            jobs = tracker.list_jobs(
                status=params["status"],
                search=params["search"],
                eligibility=params["eligibility"],
                sort="first_seen_at",
                direction="asc",
            )
            buffer = io.StringIO()
            if fmt == "json":
                write_export_json(buffer, jobs)
                content_type = "application/json"
            else:
                write_export_csv(buffer, jobs)
                content_type = "text/csv; charset=utf-8"
            body = buffer.getvalue().encode("utf-8")
            self._send(
                200,
                content_type,
                body,
                headers={"Content-Disposition": f'attachment; filename="job_bot_export.{fmt}"'},
            )

        def _handle_rows(self, tracker: Tracker, query: dict[str, list[str]]) -> None:
            params = _parse_list_params(query)
            try:
                total = tracker.count_jobs(
                    status=params["status"], search=params["search"], eligibility=params["eligibility"]
                )
                jobs = tracker.list_jobs(
                    status=params["status"],
                    search=params["search"],
                    eligibility=params["eligibility"],
                    sort=params["sort"],
                    direction=params["direction"],
                    limit=PAGE_SIZE,
                    offset=(params["page"] - 1) * PAGE_SIZE,
                )
            except InvalidSort as e:
                self._send_text(400, str(e))
                return
            body = render_rows_html(jobs, stale_after_days=stale_after_days).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body, headers={"X-Total-Jobs": str(total)})

        def _handle_qa(self, tracker: Tracker, job_id: str) -> None:
            if tracker.get_job(job_id) is None:
                self._send_text(404, "Job not found")
                return
            body = render_qa_html(tracker.list_qa(job_id)).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_resume(self, tracker: Tracker, job_id: str) -> None:
            """The tailored resume generated for this job (summary/skills/
            bullets, same shape `job-bot status <job_id>` already prints),
            for the dashboard's Resume modal - the one piece of per-job
            detail `job-bot status` already surfaced that the dashboard had
            no equivalent for, unlike Q&A/notes/blacklist which all already
            have their own modal here.
            """
            if tracker.get_job(job_id) is None:
                self._send_text(404, "Job not found")
                return
            body = render_resume_html(tracker.get_resume_generation(job_id)).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_note_get(self, tracker: Tracker, job_id: str) -> None:
            """The job's current note (possibly empty), for the note edit
            dialog to pre-fill before the user starts typing - read-only,
            no same-origin check needed (see _handle_status_update's own
            reasoning: GETs here never change state).
            """
            job = tracker.get_job(job_id)
            if job is None:
                self._send_text(404, "Job not found")
                return
            self._send_json(200, {"note": job.get("notes") or ""})

        def _handle_note_set(self, tracker: Tracker, job_id: str) -> None:
            if not self._is_same_origin():
                self._send_text(403, "Cross-origin request rejected")
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._send_text(400, "Content-Type must be application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send_text(400, "Invalid Content-Length header")
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_text(400, "Request body missing or too large")
                return
            raw_body = self.rfile.read(length)

            try:
                payload = json.loads(raw_body)
                note = payload["note"]
                if not isinstance(note, str):
                    raise ValueError("note must be a string")
            except (json.JSONDecodeError, KeyError, ValueError):
                self._send_text(400, "Body must be JSON: {\"note\": \"<text>\"}")
                return

            try:
                tracker.set_note(job_id, note)
            except ValueError as e:
                self._send_text(404, str(e))
                return
            self._send_json(200, {"ok": True, "job_id": job_id, "note": note})

        def _handle_status_update(self, tracker: Tracker, job_id: str) -> None:
            if not self._is_same_origin():
                self._send_text(403, "Cross-origin request rejected")
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._send_text(400, "Content-Type must be application/json")
                return

            try:
                # A client-supplied header, so it isn't necessarily a number
                # at all - int() raising here used to escape do_POST entirely,
                # dumping a traceback and dropping the connection with no
                # HTTP response rather than answering 400.
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send_text(400, "Invalid Content-Length header")
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_text(400, "Request body missing or too large")
                return
            raw_body = self.rfile.read(length)

            try:
                payload = json.loads(raw_body)
                new_status = payload["status"]
                if not isinstance(new_status, str):
                    raise ValueError("status must be a string")
            except (json.JSONDecodeError, KeyError, ValueError):
                self._send_text(400, "Body must be JSON: {\"status\": \"<status>\"}")
                return

            try:
                tracker.update_status(job_id, new_status)
            except InvalidStatus as e:
                self._send_text(400, str(e))
                return
            except ValueError as e:
                self._send_text(404, str(e))
                return
            self._send_json(200, {"ok": True, "job_id": job_id, "status": new_status})

        def _handle_blacklist(self, tracker: Tracker, job_id: str) -> None:
            """Blacklists this job's company (safety/blacklist.py) - the
            same effect `job-bot blacklist add "<company>"` has from the
            CLI, one click away from a row instead of leaving the
            dashboard. Only ever adds to the blacklist; doesn't touch this
            or any other job's own tracked status, exactly like the CLI
            command it mirrors - including the same in-progress-application
            warning cmd_blacklist's add branch gives (via the shared
            Tracker.in_progress_jobs_at_company(), so the two can't
            silently diverge): blacklisting only stops future applications,
            so a company you're still actively interviewing at deserves a
            heads-up here too, not just from the CLI.

            Deliberately doesn't take a --reason equivalent the way
            `job-bot blacklist add --reason` does: a client-side prompt()
            before every click would turn "one click away from a row" into
            a two-step interaction for what's an optional, low-stakes
            field, for a button whole point is being the fast path.
            `add()` already updates a company's reason on a re-add (a
            second `add` call for an already-blacklisted company overwrites
            its stored reason the same way it already overwrites its
            display-name casing), so attaching one after the fact is still
            just `job-bot blacklist add "<company>" --reason "..."` - the
            reason still shows up here afterward,
            via render_blacklist_html's own reason display, next time the
            Manage Blacklist modal is opened.
            """
            if not self._is_same_origin():
                self._send_text(403, "Cross-origin request rejected")
                return
            job = tracker.get_job(job_id)
            if job is None:
                self._send_text(404, f"No tracked job with id {job_id!r}.")
                return
            company = job["company"]
            CompanyBlacklist(blacklist_path).add(company)
            in_progress = tracker.in_progress_jobs_at_company(company)
            warning = None
            if in_progress:
                statuses = ", ".join(sorted({j["status"] for j in in_progress}))
                warning = (
                    f"{len(in_progress)} tracked application(s) at {company} are still in progress "
                    f"({statuses}) - blacklisting only stops future applications, these aren't affected."
                )
            self._send_json(200, {"ok": True, "job_id": job_id, "company": company, "warning": warning})

        def _handle_blacklist_list(self) -> None:
            """Every blacklisted company and its reason (if any), for the
            dashboard's Manage Blacklist modal - the read-only counterpart
            to _handle_blacklist above and _handle_blacklist_remove below,
            together giving the dashboard the same add/view/remove
            blacklist actions `job-bot blacklist` already has on the CLI.
            """
            entries = CompanyBlacklist(blacklist_path).list_entries()
            body = render_blacklist_html(entries).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_weekly_activity(self, tracker: Tracker) -> None:
            """Applications sent per week, for the dashboard's Weekly
            Activity modal - the dashboard counterpart to `job-bot report
            --by-week`, read-only the same way _handle_missing_qualifications
            is. Uncapped: one row per week with at least one application
            stays small for any realistic job search.
            """
            body = render_weekly_activity_html(tracker.applications_by_week()).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_missing_qualifications(self, tracker: Tracker) -> None:
            """Which specific gaps the LLM scorer flags most often across
            tracked postings, for the dashboard's Missing Qualifications
            modal - the dashboard counterpart to `job-bot report
            --by-missing-qualifications`, read-only the same way
            _handle_blacklist_list is.
            """
            breakdown = tracker.missing_qualifications_counts(limit=_MISSING_QUALIFICATIONS_LIMIT)
            body = render_missing_qualifications_html(breakdown).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_answer_gaps(self) -> None:
            """The dashboard counterpart to `job-bot review-answers` - read-
            only the same way _handle_blacklist_list/
            _handle_missing_qualifications are: actually answering a gap
            still needs review-answers' interactive prompt, but seeing what's
            piling up (and how often each one has come up) no longer needs a
            separate terminal just to check.
            """
            gaps = AnswerGapStore(answer_gaps_path).list_unanswered()
            body = render_answer_gaps_html(gaps).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_faq_list(self) -> None:
            """The dashboard counterpart to `job-bot faq list` - read-only
            the same way _handle_blacklist_list/_handle_answer_gaps are.
            """
            faq_answers = ResumeStore(resume_path, faq_path).faq_answers()
            body = render_faq_html(faq_answers).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_qa_history(self, query: dict[str, list[str]]) -> None:
            """The dashboard counterpart to `job-bot qa-history` - read-only
            the same way _handle_audit_log is, including the same `q` query
            param name/semantics (case-insensitive substring match against
            question or answer text) and the same fixed-cap-instead-of-
            paging tradeoff (_QA_HISTORY_LIMIT), for the same reason.
            """
            search = (query.get("q") or [""])[0].strip() or None
            pairs = Tracker(db_path).search_qa(search=search)[:_QA_HISTORY_LIMIT]
            body = render_qa_history_html(pairs).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_audit_log(self, query: dict[str, list[str]]) -> None:
            """The dashboard counterpart to `job-bot audit-log` - read-only
            the same way _handle_blacklist_list/_handle_missing_qualifications
            are. `q` (the same query param name the main search box already
            uses) matches AuditLogger.read_entries()'s own `search` - the
            whole entry (action and every detail value), not just the action
            name - capped to the _AUDIT_LOG_LIMIT most recent matching
            entries; `job-bot audit-log` itself has no such cap for anyone
            who needs the full history. `failures=1` reads
            FAILED_APPLICATIONS_LOG_PATH instead of AUDIT_LOG_PATH, the same
            switch `job-bot audit-log --failures` makes on the CLI.
            """
            search = (query.get("q") or [""])[0].strip() or None
            show_failures = (query.get("failures") or [""])[0] == "1"
            path = failed_applications_log_path if show_failures else audit_log_path
            entries = AuditLogger(path).read_entries(search=search)[:_AUDIT_LOG_LIMIT]
            body = render_audit_log_html(entries).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def _handle_blacklist_remove(self) -> None:
            if not self._is_same_origin():
                self._send_text(403, "Cross-origin request rejected")
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._send_text(400, "Content-Type must be application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send_text(400, "Invalid Content-Length header")
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_text(400, "Request body missing or too large")
                return
            raw_body = self.rfile.read(length)

            try:
                payload = json.loads(raw_body)
                company = payload["company"]
                if not isinstance(company, str):
                    raise ValueError("company must be a string")
            except (json.JSONDecodeError, KeyError, ValueError):
                self._send_text(400, "Body must be JSON: {\"company\": \"<company>\"}")
                return

            removed = CompanyBlacklist(blacklist_path).remove(company)
            self._send_json(200, {"ok": True, "company": company, "removed": removed})

        def _handle_answer_gaps_dismiss(self) -> None:
            """The dashboard counterpart to `job-bot review-answers
            --dismiss` - same shape (same-origin check, Content-Type/body-
            size validation, JSON body) as _handle_blacklist_remove above,
            for the Dismiss button render_answer_gaps_html() gives each
            gap.
            """
            if not self._is_same_origin():
                self._send_text(403, "Cross-origin request rejected")
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._send_text(400, "Content-Type must be application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send_text(400, "Invalid Content-Length header")
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_text(400, "Request body missing or too large")
                return
            raw_body = self.rfile.read(length)

            try:
                payload = json.loads(raw_body)
                question = payload["question"]
                if not isinstance(question, str):
                    raise ValueError("question must be a string")
            except (json.JSONDecodeError, KeyError, ValueError):
                self._send_text(400, "Body must be JSON: {\"question\": \"<question>\"}")
                return

            dismissed = AnswerGapStore(answer_gaps_path).resolve(question)
            self._send_json(200, {"ok": True, "question": question, "dismissed": dismissed})

        def _handle_faq_remove(self) -> None:
            """The dashboard counterpart to `job-bot faq remove` - same
            shape (same-origin check, Content-Type/body-size validation,
            JSON body) as _handle_blacklist_remove/_handle_answer_gaps_
            dismiss above, for the Remove button render_faq_html() gives
            each cached FAQ answer.
            """
            if not self._is_same_origin():
                self._send_text(403, "Cross-origin request rejected")
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._send_text(400, "Content-Type must be application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send_text(400, "Invalid Content-Length header")
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_text(400, "Request body missing or too large")
                return
            raw_body = self.rfile.read(length)

            try:
                payload = json.loads(raw_body)
                question = payload["question"]
                if not isinstance(question, str):
                    raise ValueError("question must be a string")
            except (json.JSONDecodeError, KeyError, ValueError):
                self._send_text(400, "Body must be JSON: {\"question\": \"<question>\"}")
                return

            removed = ResumeStore(resume_path, faq_path).remove_faq_answer(question)
            self._send_json(200, {"ok": True, "question": question, "removed": removed})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # quiet by default; the CLI prints the one line that matters

    return DashboardHandler


def run_dashboard(
    db_path: Path,
    blacklist_path: Path,
    audit_log_path: Path,
    failed_applications_log_path: Path,
    answer_gaps_path: Path,
    resume_path: Path,
    faq_path: Path,
    port: int = 8765,
    open_browser: bool = True,
    *,
    stale_after_days: int = 14,
) -> None:
    handler = make_handler(
        db_path,
        blacklist_path,
        audit_log_path,
        failed_applications_log_path,
        answer_gaps_path,
        resume_path,
        faq_path,
        stale_after_days=stale_after_days,
    )
    try:
        server = ThreadingHTTPServer((DASHBOARD_HOST, port), handler)
    except (OSError, OverflowError) as e:
        # OSError: most commonly "Address already in use" (errno 48/98) -
        # a previous `job-bot dashboard` still running is by far the most
        # likely real cause. OverflowError: `--port`/DASHBOARD_PORT set to
        # a value outside 0-65535 - argparse's type=int has no range check
        # of its own, so a typo here previously reached socket.bind()
        # itself before failing.
        raise DashboardPortInUse(
            f"Could not start the dashboard on port {port}: {e}. Is `job-bot dashboard` already "
            "running? Use --port to pick a different one."
        ) from e
    url = f"http://{DASHBOARD_HOST}:{server.server_port}/"
    print(f"Dashboard running at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

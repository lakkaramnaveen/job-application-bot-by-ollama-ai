"""Pure HTML-rendering functions for the one-page tracker dashboard - no
server or I/O here, so the output is directly unit-testable. See server.py
for the HTTP layer that calls this.
"""

import html
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from job_bot.tracker.db import TRACKER_STATUSES

# job.url ultimately comes from a scraped LinkedIn anchor href (see
# linkedin_adapter.py's search()) - untrusted data. html.escape() alone
# neutralizes attribute breakout but not a javascript: URI, which would
# execute in the dashboard's origin on click. Only ever render an <a href>
# for these two schemes; anything else renders as plain, unlinked text.
_SAFE_URL_SCHEMES = frozenset({"http", "https"})

STATUS_COLORS = {
    "seen": "#9ca3af",
    "applied": "#3b82f6",
    "skipped": "#9ca3af",
    "interviewing": "#f59e0b",
    "offer": "#22c55e",
    "rejected": "#ef4444",
    "withdrawn": "#6b7280",
    "no_response": "#6b7280",
}

PAGE_TITLE = "job_bot tracker"
PAGE_SIZE = 25

# (value, label) pairs for the sort dropdown - value is "column:direction",
# validated server-side against tracker.db.SORTABLE_COLUMNS before use.
SORT_OPTIONS = [
    ("first_seen_at:desc", "Newest first"),
    ("first_seen_at:asc", "Oldest first"),
    ("applied_at:desc", "Recently applied"),
    ("match_score:desc", "Score: high to low"),
    ("match_score:asc", "Score: low to high"),
    ("company:asc", "Company: A-Z"),
    ("title:asc", "Title: A-Z"),
]


def _badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#6b7280")
    label = html.escape(status.replace("_", " "))
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:999px;font-size:12px;white-space:nowrap">{label}</span>'
    )


def _title_cell(title: str, raw_url: str) -> str:
    """Link the title to its posting only when the URL is http(s) - any
    other scheme (javascript:, data:, ...) renders as plain text instead of
    a clickable link, since raw_url is untrusted scraped/emailed data.
    """
    try:
        scheme = urlparse(raw_url).scheme.lower()
    except ValueError:
        scheme = ""
    if scheme not in _SAFE_URL_SCHEMES:
        return html.escape(title)
    safe_url = html.escape(raw_url, quote=True)
    return f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{html.escape(title)}</a>'


def _status_select(job_id: str, current_status: str) -> str:
    """A per-row status control. Options come from TRACKER_STATUSES so an
    invalid status can never be submitted client-side; the server still
    re-validates on write since this is untrusted input regardless.

    Uses a `data-job-id` attribute rather than an inline `onchange="..."`
    with interpolated data, so a crafted job_id/status can never break out
    into a JS-string context - the static <script> at the bottom reads
    dataset.jobId and the select's own value via the DOM instead.
    """
    options = list(TRACKER_STATUSES)
    if current_status not in TRACKER_STATUSES:
        options.append(current_status)
    opts_html = "\n".join(
        f'<option value="{html.escape(s, quote=True)}"{" selected" if s == current_status else ""}>'
        f"{html.escape(s.replace('_', ' '))}</option>"
        for s in sorted(options)
    )
    safe_job_id = html.escape(job_id, quote=True)
    return (
        f'<select class="status-select" data-job-id="{safe_job_id}" '
        f'aria-label="Update status">{opts_html}</select>'
    )


def _stat_pill(status: str, label: str, count: int, active: bool) -> str:
    """No inline --pill-color at all for "All" or an unrecognized status
    (rather than hardcoding a light-mode gray) so the CSS's own
    var(--pill-color, var(--pill-default)) fallback picks the right neutral
    for whichever color scheme is active - a literal inline color would
    always win over that fallback and never adapt to dark mode.
    """
    cls = "stat-pill active" if active else "stat-pill"
    safe_status = html.escape(status, quote=True)
    color = STATUS_COLORS.get(status)
    style_attr = f' style="--pill-color:{color}"' if color else ""
    return (
        f'<button type="button" class="{cls}" data-status="{safe_status}"{style_attr}>'
        f"{html.escape(label)} <span class=\"count\">{count}</span></button>"
    )


def render_stats_html(counts: dict[str, int], selected_status: str) -> str:
    """Clickable per-status count pills above the table - reused by both
    the full page and the polling refresh (see server.py's /api/stats) so
    the counts stay live as statuses change, the same pattern render_rows_html
    already uses for the table body. `counts` should come from
    Tracker.status_counts(search=...) so the pills reflect the current
    search box; a status filter is never baked into `counts` itself, or
    every pill but the selected one would show zero.

    A zero-count status is hidden unless it's the one currently selected -
    otherwise picking a status that has since emptied out (e.g. the last
    "interviewing" job was just marked "offer") would leave no way to
    click back off of it.
    """
    total = sum(counts.values())
    pills = [_stat_pill("", "All", total, selected_status == "")]
    # Known statuses first in their usual order, then any unrecognized ones
    # `counts` turned up (e.g. legacy data predating a TRACKER_STATUSES
    # change) - same defensive stance render_rows_html's status <select>
    # already takes: an unknown status still needs a way to filter to it.
    unknown = sorted(s for s in counts if s not in STATUS_COLORS)
    for status in (*STATUS_COLORS, *unknown):
        count = counts.get(status, 0)
        if count == 0 and status != selected_status:
            continue
        pills.append(_stat_pill(status, status.replace("_", " ").title(), count, status == selected_status))
    return "\n".join(pills)


def render_rows_html(jobs: list[dict[str, Any]], *, stale_after_days: int | None = None) -> str:
    """The <tbody> contents only - reused by both the full page and the
    polling/filtering endpoint that refreshes just the table body.

    `stale_after_days` (Settings.stale_after_days) marks an applied job
    with no reply past that many days - the same "worth a follow-up"
    signal `job-bot report --stale-days` already surfaces, but previously
    only in a separate CLI command someone had to remember to run; now
    visible directly on the row people are already scanning. None (the
    default) omits the marker entirely, for callers - tests included -
    that don't have a threshold to compare against.
    """
    if not jobs:
        return '<tr><td colspan="7" class="empty">No jobs tracked yet - run `job-bot run` first.</td></tr>'

    stale_cutoff = (
        (datetime.now(UTC) - timedelta(days=stale_after_days)).isoformat()
        if stale_after_days is not None
        else None
    )
    rows = []
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        title_cell = _title_cell(str(job.get("title", "")), str(job.get("url", "")))
        company = html.escape(str(job.get("company", "")))
        score = job.get("match_score")
        score_text = str(score) if score is not None else "-"
        reasoning = job.get("match_reasoning")
        eligibility = job.get("eligibility")
        # A categorical eligibility-gate rejection (e.g. a citizenship
        # requirement) or an ambiguous "flag" verdict the model itself was
        # uncertain about is worth calling out at a glance in the table,
        # not just readable one click away in `job-bot status <job_id>` -
        # same information, surfaced where someone scanning the dashboard
        # actually looks first.
        flagged_eligibility = eligibility in ("fail", "flag")
        score_text = f"⚠️ {score_text}" if flagged_eligibility else score_text
        # A native title attribute rather than a button/modal like Note or
        # Q&A get: this is the LLM's own read-only explanation of the score
        # it already gave (see Tracker.record_score()'s reasoning/
        # eligibility params) - nothing to edit, so a hover tooltip on the
        # very cell it explains is enough, without another click needed
        # just to read a sentence.
        tooltip_parts = []
        if flagged_eligibility:
            note = job.get("eligibility_note")
            tooltip_parts.append(f"Eligibility: {eligibility}" + (f" - {note}" if note else ""))
        if reasoning:
            tooltip_parts.append(str(reasoning))
        # Stored as a JSON array (see Tracker.record_score()'s docstring) -
        # None for a job scored before this column existed, or never
        # scored at all.
        raw_missing_quals = job.get("missing_qualifications")
        missing_quals = json.loads(raw_missing_quals) if raw_missing_quals else []
        if missing_quals:
            tooltip_parts.append("Missing: " + ", ".join(str(q) for q in missing_quals))
        score_title_attr = (
            f' title="{html.escape(chr(10).join(tooltip_parts), quote=True)}"' if tooltip_parts else ""
        )
        status = str(job.get("status", ""))
        applied_at_raw = job.get("applied_at")
        applied_at = html.escape(str(applied_at_raw or "-"))
        is_stale = (
            stale_cutoff is not None
            and status == "applied"
            and applied_at_raw is not None
            and str(applied_at_raw) < stale_cutoff
        )
        applied_display = f"⏰ {applied_at}" if is_stale else applied_at
        applied_title_attr = (
            f' title="No reply after {stale_after_days}+ days - job-bot report --stale-days shows all of these"'
            if is_stale
            else ""
        )
        safe_job_id = html.escape(job_id, quote=True)
        has_note = bool(job.get("notes"))
        note_class = "note-button has-note" if has_note else "note-button"
        note_title = "Edit note" if has_note else "Add a note"
        rows.append(
            "<tr>"
            f"<td>{title_cell}</td>"
            f"<td>{company}</td>"
            f"<td{score_title_attr}>{score_text}</td>"
            f"<td>{_badge(status)}</td>"
            f"<td{applied_title_attr}>{applied_display}</td>"
            f'<td class="jobid">{html.escape(job_id)}</td>'
            "<td class=\"actions\">"
            f"{_status_select(job_id, str(job.get('status', '')))}"
            f'<button type="button" class="qa-button" data-job-id="{safe_job_id}">Q&amp;A</button>'
            f'<button type="button" class="resume-button" data-job-id="{safe_job_id}">Resume</button>'
            f'<button type="button" class="{note_class}" data-job-id="{safe_job_id}" '
            f'title="{note_title}">Note</button>'
            f'<button type="button" class="blacklist-button" data-job-id="{safe_job_id}" '
            f'title="Never apply to {company} again">Blacklist</button>'
            "</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_qa_html(qa: list[dict[str, Any]]) -> str:
    """A job's Q&A transcript as an HTML fragment, for the dashboard's Q&A
    modal. Untrusted end-to-end: questions/answers ultimately came from a
    scraped application form and LLM output, so every field is escaped.
    """
    if not qa:
        return '<p class="empty">No answered questions recorded for this job.</p>'
    items = []
    for entry in qa:
        question = html.escape(str(entry.get("question", "")))
        answer = html.escape(str(entry.get("answer", "")))
        items.append(f"<dt>{question}</dt><dd>{answer}</dd>")
    return f'<dl class="qa-list">{"".join(items)}</dl>'


def render_resume_html(generation: dict[str, Any] | None) -> str:
    """The tailored resume generated for this job (summary/skills/bullets -
    same shape Tracker.get_resume_generation() returns, and `job-bot status
    <job_id>` already prints) as an HTML fragment, for the dashboard's
    Resume modal. `generation` is None when the job never cleared the fit
    gate or otherwise was never tailored - not an error, just nothing to
    show. LLM-generated content, so every field is escaped, same as
    render_qa_html above.
    """
    if generation is None:
        return '<p class="empty">No tailored resume generated for this job.</p>'
    summary = html.escape(str(generation.get("summary", "")))
    skills = ", ".join(html.escape(str(s)) for s in generation.get("skills", []))
    bullets = "".join(f"<li>{html.escape(str(b))}</li>" for b in generation.get("bullets", []))
    bullets_html = f"<ul>{bullets}</ul>" if bullets else ""
    return (
        f"<p><strong>Summary:</strong> {summary}</p>"
        f"<p><strong>Skills:</strong> {skills or '-'}</p>"
        f"{bullets_html}"
    )


def render_blacklist_html(entries: list[dict[str, str]]) -> str:
    """The blacklisted companies (each {"name", "reason"} - see
    CompanyBlacklist.list_entries()) as an HTML fragment, for the
    dashboard's Manage Blacklist modal - same shape as render_qa_html
    above (a server-rendered fragment the client just drops into the
    dialog), with one remove button per company wired the same
    data-attribute way status-select/qa-button/blacklist-button already
    are, rather than an inline onclick with interpolated data. The reason
    (when set - via `job-bot blacklist add --reason`, the only way to set
    one; see _handle_blacklist's own docstring for why the dashboard's
    one-click blacklist button deliberately doesn't prompt for one) prints
    dimmed after the name, the same treatment render_rows_html already
    gives less-important secondary text.
    """
    if not entries:
        return '<p class="empty">Blacklist is empty.</p>'
    items = []
    for entry in entries:
        name = entry["name"]
        safe_company = html.escape(name, quote=True)
        reason_html = f' <span class="blacklist-reason">- {html.escape(entry["reason"])}</span>' if entry["reason"] else ""
        items.append(
            f"<li><span>{html.escape(name)}{reason_html}</span> "
            f'<button type="button" class="blacklist-remove-button" data-company="{safe_company}">'
            "Remove</button></li>"
        )
    return f'<ul class="blacklist-list">{"".join(items)}</ul>'


def render_missing_qualifications_html(breakdown: dict[str, int]) -> str:
    """The missing-qualifications breakdown (Tracker.missing_qualifications_
    counts()) as an HTML fragment, for the dashboard's Missing Qualifications
    modal - same read-only-list shape as render_blacklist_html above, minus
    the per-item remove button since there's nothing here to manage, just
    read. `job-bot report --by-missing-qualifications` is the CLI
    counterpart to this same data.
    """
    if not breakdown:
        return '<p class="empty">No missing qualifications recorded yet.</p>'
    ordered = sorted(breakdown, key=lambda qual: (-breakdown[qual], qual.casefold()))
    items = "".join(
        f'<li><span class="mq-count">{breakdown[qual]}</span> {html.escape(qual)}</li>' for qual in ordered
    )
    return f'<ul class="mq-list">{items}</ul>'


def _options_html(options: list[tuple[str, str]], selected: str) -> str:
    return "\n".join(
        f'<option value="{html.escape(value, quote=True)}"{" selected" if value == selected else ""}>'
        f"{html.escape(label)}</option>"
        for value, label in options
    )


def render_page_html(
    jobs: list[dict[str, Any]],
    *,
    total: int = 0,
    page: int = 1,
    page_size: int = PAGE_SIZE,
    status: str = "",
    eligibility: str = "",
    search: str = "",
    sort: str = "first_seen_at",
    direction: str = "desc",
    refresh_seconds: int = 5,
    counts: dict[str, int] | None = None,
    stale_after_days: int | None = None,
) -> str:
    rows_html = render_rows_html(jobs, stale_after_days=stale_after_days)
    stats_html = render_stats_html(counts or {}, status)
    status_options = [("", "All statuses"), *((s, s.replace("_", " ")) for s in sorted(TRACKER_STATUSES))]
    eligibility_options = [
        ("", "Any eligibility"),
        ("pass", "Eligibility: pass"),
        ("flag", "Eligibility: flag"),
        ("fail", "Eligibility: fail"),
    ]
    sort_value = f"{sort}:{direction}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(PAGE_TITLE)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg: #f9fafb; --fg: #111827; --muted: #6b7280; --border: #d1d5db;
    --border-light: #e5e7eb; --surface: #fff; --header-bg: #f3f4f6;
    --link: #2563eb; --shadow: rgba(0,0,0,0.06); --backdrop: rgba(0,0,0,0.35);
    --dialog-shadow: rgba(0,0,0,0.15); --pill-default: #374151;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0f172a; --fg: #e5e7eb; --muted: #9ca3af; --border: #374151;
      --border-light: #1f2937; --surface: #1e293b; --header-bg: #172033;
      --link: #60a5fa; --shadow: rgba(0,0,0,0.4); --backdrop: rgba(0,0,0,0.6);
      --dialog-shadow: rgba(0,0,0,0.5); --pill-default: #9ca3af;
    }}
  }}
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem;
          background: var(--bg); color: var(--fg); }}
  h1 {{ font-size: 1.25rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 1.25rem; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }}
  .stat-pill {{ display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.8rem;
              padding: 0.3rem 0.75rem; border-radius: 999px; cursor: pointer;
              background: var(--surface); color: var(--fg);
              border: 1px solid var(--pill-color, var(--border)); }}
  .stat-pill:hover {{ border-color: var(--pill-color, var(--muted)); }}
  .stat-pill.active {{ background: var(--pill-color, var(--pill-default)); color: #fff;
              border-color: var(--pill-color, var(--pill-default)); }}
  .stat-pill .count {{ font-weight: 600; }}
  .toolbar {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
              margin-bottom: 1rem; }}
  .toolbar input, .toolbar select {{ font-size: 0.85rem; padding: 0.4rem 0.6rem;
              border: 1px solid var(--border); border-radius: 6px;
              background: var(--surface); color: var(--fg); }}
  .toolbar input[type="search"] {{ min-width: 220px; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; min-width: 640px; background: var(--surface);
           box-shadow: 0 1px 2px var(--shadow); }}
  th, td {{ text-align: left; padding: 0.6rem 0.9rem; border-bottom: 1px solid var(--border-light);
            font-size: 0.9rem; vertical-align: middle; }}
  th {{ background: var(--header-bg); font-weight: 600; }}
  td.jobid {{ color: var(--muted); font-size: 0.75rem; }}
  td.empty {{ color: var(--muted); text-align: center; padding: 2rem; }}
  td[title] {{ cursor: help; border-bottom: 1px dotted var(--muted); }}
  td.actions {{ display: flex; gap: 0.4rem; align-items: center; white-space: nowrap; }}
  .status-select {{ font-size: 0.8rem; padding: 0.25rem 0.4rem; border-radius: 4px;
              border: 1px solid var(--border); background: var(--surface); color: var(--fg); }}
  .qa-button, .resume-button, .blacklist-button, .note-button {{ font-size: 0.8rem; padding: 0.25rem 0.6rem;
              border-radius: 4px; border: 1px solid var(--border); background: var(--surface);
              color: var(--fg); cursor: pointer; }}
  .qa-button:hover, .resume-button:hover {{ background: var(--header-bg); }}
  .blacklist-button:hover {{ border-color: #ef4444; color: #ef4444; }}
  .note-button:hover {{ background: var(--header-bg); }}
  .note-button.has-note {{ border-color: #3b82f6; color: #3b82f6; }}
  .export-link {{ font-size: 0.85rem; padding: 0.4rem 0.75rem; border-radius: 6px;
              border: 1px solid var(--border); background: var(--surface); color: var(--fg);
              text-decoration: none; margin-left: auto; }}
  .export-link:hover {{ background: var(--header-bg); }}
  a {{ color: var(--link); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .pager {{ display: flex; gap: 0.75rem; align-items: center; margin-top: 1rem;
            font-size: 0.85rem; color: var(--muted); }}
  .pager button {{ font-size: 0.8rem; padding: 0.3rem 0.7rem; border-radius: 6px;
              border: 1px solid var(--border); background: var(--surface); color: var(--fg);
              cursor: pointer; }}
  .pager button:disabled {{ opacity: 0.5; cursor: default; }}
  dialog {{ border: none; border-radius: 10px; padding: 1.25rem 1.5rem; max-width: 32rem;
            background: var(--surface); color: var(--fg); box-shadow: 0 10px 30px var(--dialog-shadow); }}
  dialog::backdrop {{ background: var(--backdrop); }}
  .qa-list dt {{ font-weight: 600; margin-top: 0.75rem; }}
  .qa-list dd {{ margin: 0.25rem 0 0; color: var(--fg); }}
  #qaClose, #blacklistClose, #noteSave, #noteCancel {{ margin-top: 1rem; font-size: 0.85rem;
              padding: 0.35rem 0.8rem; border-radius: 6px; border: 1px solid var(--border);
              background: var(--surface); color: var(--fg); cursor: pointer; }}
  #noteSave {{ background: #3b82f6; border-color: #3b82f6; color: #fff; }}
  .dialog-actions {{ display: flex; gap: 0.5rem; }}
  #noteTextarea {{ width: 100%; box-sizing: border-box; font: inherit; padding: 0.5rem;
              border-radius: 6px; border: 1px solid var(--border); background: var(--surface);
              color: var(--fg); resize: vertical; }}
  .blacklist-list {{ list-style: none; margin: 0; padding: 0; }}
  .blacklist-list li {{ display: flex; justify-content: space-between; align-items: center;
              gap: 0.75rem; padding: 0.4rem 0; border-bottom: 1px solid var(--border); }}
  .blacklist-list li:last-child {{ border-bottom: none; }}
  .blacklist-remove-button {{ font-size: 0.8rem; padding: 0.2rem 0.5rem; border-radius: 4px;
              border: 1px solid var(--border); background: var(--surface); color: var(--fg); cursor: pointer; }}
  .blacklist-remove-button:hover {{ border-color: #ef4444; color: #ef4444; }}
  .blacklist-reason {{ color: var(--muted); font-size: 0.85em; }}
  .mq-list {{ list-style: none; margin: 0; padding: 0; }}
  .mq-list li {{ padding: 0.4rem 0; border-bottom: 1px solid var(--border); }}
  .mq-list li:last-child {{ border-bottom: none; }}
  .mq-count {{ display: inline-block; min-width: 1.75rem; font-weight: 600; color: var(--muted); }}
</style>
</head>
<body>
<h1>{html.escape(PAGE_TITLE)}</h1>
<p class="subtitle">Auto-refreshes every {refresh_seconds}s. Update a status inline below, or via
`job-bot status` / `job-bot run` / `job-bot gmail-sync`.</p>

<div class="stats" id="stats">
{stats_html}
</div>

<form class="toolbar" id="filters">
  <input type="search" id="q" name="q" placeholder="Search title, company, notes, or scoring details..."
         value="{html.escape(search, quote=True)}">
  <select id="status" name="status">
    {_options_html(status_options, status)}
  </select>
  <select id="eligibility" name="eligibility">
    {_options_html(eligibility_options, eligibility)}
  </select>
  <select id="sort" name="sort">
    {_options_html(SORT_OPTIONS, sort_value)}
  </select>
  <a id="exportCsv" class="export-link" href="/api/export.csv">Export CSV</a>
  <a id="exportJson" class="export-link" href="/api/export.json">Export JSON</a>
  <button type="button" id="manageBlacklist" class="export-link">Manage Blacklist</button>
  <button type="button" id="showMissingQualifications" class="export-link">Missing Qualifications</button>
</form>

<div class="table-wrap">
<table>
  <thead>
    <tr><th>Title</th><th>Company</th><th>Score</th><th>Status</th><th>Applied</th><th>Job ID</th><th>Actions</th></tr>
  </thead>
  <tbody id="rows">
{rows_html}
  </tbody>
</table>
</div>

<div class="pager">
  <button type="button" id="prevPage">&laquo; Prev</button>
  <span id="pageInfo"></span>
  <button type="button" id="nextPage">Next &raquo;</button>
</div>

<dialog id="qaDialog">
  <h2>Q&amp;A history</h2>
  <div id="qaContent"></div>
  <button type="button" id="qaClose">Close</button>
</dialog>

<dialog id="resumeDialog">
  <h2>Tailored resume</h2>
  <div id="resumeContent"></div>
  <button type="button" id="resumeClose">Close</button>
</dialog>

<dialog id="blacklistDialog">
  <h2>Blacklisted companies</h2>
  <div id="blacklistContent"></div>
  <button type="button" id="blacklistClose">Close</button>
</dialog>

<dialog id="missingQualificationsDialog">
  <h2>Most common missing qualifications</h2>
  <div id="missingQualificationsContent"></div>
  <button type="button" id="missingQualificationsClose">Close</button>
</dialog>

<dialog id="noteDialog">
  <h2>Note</h2>
  <textarea id="noteTextarea" rows="6" placeholder="Salary info, referral, anything worth remembering..."></textarea>
  <div class="dialog-actions">
    <button type="button" id="noteSave">Save</button>
    <button type="button" id="noteCancel">Cancel</button>
  </div>
</dialog>

<script>
// Initial filter/sort/page state is read from the DOM (already HTML-escaped
// by the server) rather than interpolated as string literals here, so a
// crafted search term can't break out of this block via the closing script
// tag - see render_page_html's docstring-adjacent note in render.py.
const state = {{
  q: document.getElementById('q').value,
  status: document.getElementById('status').value,
  eligibility: document.getElementById('eligibility').value,
  sort: document.getElementById('sort').value.split(':')[0],
  dir: document.getElementById('sort').value.split(':')[1],
  page: {page},
}};
const PAGE_SIZE = {page_size};

function buildQuery() {{
  const params = new URLSearchParams();
  if (state.q) params.set('q', state.q);
  if (state.status) params.set('status', state.status);
  if (state.eligibility) params.set('eligibility', state.eligibility);
  params.set('sort', state.sort);
  params.set('dir', state.dir);
  params.set('page', String(state.page));
  return params.toString();
}}

async function refresh() {{
  const query = buildQuery();
  document.getElementById('exportCsv').href = '/api/export.csv?' + query;
  document.getElementById('exportJson').href = '/api/export.json?' + query;
  try {{
    const res = await fetch('/api/rows?' + query);
    if (!res.ok) return;
    document.getElementById('rows').innerHTML = await res.text();
    const total = parseInt(res.headers.get('X-Total-Jobs') || '0', 10);
    const start = total === 0 ? 0 : (state.page - 1) * PAGE_SIZE + 1;
    const end = Math.min(state.page * PAGE_SIZE, total);
    document.getElementById('pageInfo').textContent = total === 0 ? 'No results' : `${{start}}-${{end}} of ${{total}}`;
    document.getElementById('prevPage').disabled = state.page <= 1;
    document.getElementById('nextPage').disabled = end >= total;
  }} catch (e) {{
    // Network hiccup on a local server - next tick will retry.
  }}
  try {{
    const statsRes = await fetch('/api/stats?' + query);
    if (statsRes.ok) document.getElementById('stats').innerHTML = await statsRes.text();
  }} catch (e) {{
    // Same as above - leave the stat pills as they were until the next tick.
  }}
}}

let searchTimer = null;
document.getElementById('q').addEventListener('input', (e) => {{
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {{
    state.q = e.target.value;
    state.page = 1;
    refresh();
  }}, 300);
}});
document.getElementById('status').addEventListener('change', (e) => {{
  state.status = e.target.value;
  state.page = 1;
  refresh();
}});
document.getElementById('eligibility').addEventListener('change', (e) => {{
  state.eligibility = e.target.value;
  state.page = 1;
  refresh();
}});
// Delegated to the container, not bound to individual pill buttons, since
// #stats' contents are replaced wholesale on every refresh() - a listener
// on the old buttons would stop firing the moment they're replaced.
document.getElementById('stats').addEventListener('click', (e) => {{
  const pill = e.target.closest('.stat-pill');
  if (!pill) return;
  state.status = pill.dataset.status;
  document.getElementById('status').value = state.status;
  state.page = 1;
  refresh();
}});
document.getElementById('sort').addEventListener('change', (e) => {{
  const [sort, dir] = e.target.value.split(':');
  state.sort = sort;
  state.dir = dir;
  refresh();
}});
document.getElementById('prevPage').addEventListener('click', () => {{
  if (state.page > 1) {{ state.page -= 1; refresh(); }}
}});
document.getElementById('nextPage').addEventListener('click', () => {{
  state.page += 1;
  refresh();
}});

document.getElementById('rows').addEventListener('change', async (e) => {{
  if (!e.target.classList.contains('status-select')) return;
  const jobId = e.target.dataset.jobId;
  const newStatus = e.target.value;
  try {{
    const res = await fetch(`/api/jobs/${{encodeURIComponent(jobId)}}/status`, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ status: newStatus }}),
    }});
    if (!res.ok) {{
      alert('Could not update status: ' + (await res.text()));
    }}
  }} catch (err) {{
    alert('Could not update status (network error).');
  }} finally {{
    refresh();
  }}
}});

const qaDialog = document.getElementById('qaDialog');
document.getElementById('rows').addEventListener('click', async (e) => {{
  if (!e.target.classList.contains('qa-button')) return;
  const jobId = e.target.dataset.jobId;
  document.getElementById('qaContent').innerHTML = 'Loading...';
  qaDialog.showModal();
  try {{
    const res = await fetch(`/api/jobs/${{encodeURIComponent(jobId)}}/qa`);
    document.getElementById('qaContent').innerHTML = res.ok ? await res.text() : 'Could not load Q&A.';
  }} catch (err) {{
    document.getElementById('qaContent').innerHTML = 'Could not load Q&A (network error).';
  }}
}});
document.getElementById('qaClose').addEventListener('click', () => qaDialog.close());

const resumeDialog = document.getElementById('resumeDialog');
document.getElementById('rows').addEventListener('click', async (e) => {{
  if (!e.target.classList.contains('resume-button')) return;
  const jobId = e.target.dataset.jobId;
  document.getElementById('resumeContent').innerHTML = 'Loading...';
  resumeDialog.showModal();
  try {{
    const res = await fetch(`/api/jobs/${{encodeURIComponent(jobId)}}/resume`);
    document.getElementById('resumeContent').innerHTML = res.ok ? await res.text() : 'Could not load resume.';
  }} catch (err) {{
    document.getElementById('resumeContent').innerHTML = 'Could not load resume (network error).';
  }}
}});
document.getElementById('resumeClose').addEventListener('click', () => resumeDialog.close());

const noteDialog = document.getElementById('noteDialog');
const noteTextarea = document.getElementById('noteTextarea');
let noteJobId = null;
document.getElementById('rows').addEventListener('click', async (e) => {{
  if (!e.target.classList.contains('note-button')) return;
  noteJobId = e.target.dataset.jobId;
  noteTextarea.value = 'Loading...';
  noteDialog.showModal();
  try {{
    const res = await fetch(`/api/jobs/${{encodeURIComponent(noteJobId)}}/note`);
    noteTextarea.value = res.ok ? (await res.json()).note : '';
  }} catch (err) {{
    noteTextarea.value = '';
  }}
  noteTextarea.focus();
}});
document.getElementById('noteCancel').addEventListener('click', () => noteDialog.close());
document.getElementById('noteSave').addEventListener('click', async () => {{
  try {{
    const res = await fetch(`/api/jobs/${{encodeURIComponent(noteJobId)}}/note`, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ note: noteTextarea.value }}),
    }});
    if (!res.ok) {{
      alert('Could not save note: ' + (await res.text()));
      return;
    }}
  }} catch (err) {{
    alert('Could not save note (network error).');
    return;
  }}
  noteDialog.close();
  refresh();
}});

document.getElementById('rows').addEventListener('click', async (e) => {{
  if (!e.target.classList.contains('blacklist-button')) return;
  const jobId = e.target.dataset.jobId;
  try {{
    const res = await fetch(`/api/jobs/${{encodeURIComponent(jobId)}}/blacklist`, {{ method: 'POST' }});
    if (res.ok) {{
      const body = await res.json();
      let msg = `Blacklisted ${{body.company}} - job-bot run will always skip it from now on.`;
      if (body.warning) msg += `\n\n${{body.warning}}`;
      alert(msg);
    }} else {{
      alert('Could not blacklist: ' + (await res.text()));
    }}
  }} catch (err) {{
    alert('Could not blacklist (network error).');
  }}
}});

const blacklistDialog = document.getElementById('blacklistDialog');
async function loadBlacklist() {{
  const content = document.getElementById('blacklistContent');
  content.innerHTML = 'Loading...';
  try {{
    const res = await fetch('/api/blacklist');
    content.innerHTML = res.ok ? await res.text() : 'Could not load the blacklist.';
  }} catch (err) {{
    content.innerHTML = 'Could not load the blacklist (network error).';
  }}
}}
document.getElementById('manageBlacklist').addEventListener('click', () => {{
  blacklistDialog.showModal();
  loadBlacklist();
}});
document.getElementById('blacklistClose').addEventListener('click', () => blacklistDialog.close());
document.getElementById('blacklistContent').addEventListener('click', async (e) => {{
  if (!e.target.classList.contains('blacklist-remove-button')) return;
  const company = e.target.dataset.company;
  try {{
    const res = await fetch('/api/blacklist/remove', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ company }}),
    }});
    if (!res.ok) alert('Could not remove: ' + (await res.text()));
  }} catch (err) {{
    alert('Could not remove (network error).');
  }} finally {{
    loadBlacklist();
  }}
}});

const missingQualificationsDialog = document.getElementById('missingQualificationsDialog');
document.getElementById('showMissingQualifications').addEventListener('click', async () => {{
  const content = document.getElementById('missingQualificationsContent');
  content.innerHTML = 'Loading...';
  missingQualificationsDialog.showModal();
  try {{
    const res = await fetch('/api/missing-qualifications');
    content.innerHTML = res.ok ? await res.text() : 'Could not load missing qualifications.';
  }} catch (err) {{
    content.innerHTML = 'Could not load missing qualifications (network error).';
  }}
}});
document.getElementById('missingQualificationsClose').addEventListener(
  'click', () => missingQualificationsDialog.close()
);

// The periodic refresh below replaces the whole <tbody>, which would
// otherwise yank a status <select> out from under a user mid-interaction
// (open dropdown, or just focused) and silently drop their in-progress
// choice. Skip auto-refresh ticks - but not the explicit refresh() calls
// triggered by the user's own actions above - while a status select has
// focus.
let statusSelectFocused = false;
document.getElementById('rows').addEventListener('focusin', (e) => {{
  if (e.target.classList.contains('status-select')) statusSelectFocused = true;
}});
document.getElementById('rows').addEventListener('focusout', (e) => {{
  if (e.target.classList.contains('status-select')) statusSelectFocused = false;
}});

refresh();
setInterval(() => {{ if (!statusSelectFocused) refresh(); }}, {refresh_seconds * 1000});
</script>
</body>
</html>
"""

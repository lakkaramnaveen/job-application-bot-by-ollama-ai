"""Pure HTML-rendering functions for the one-page tracker dashboard - no
server or I/O here, so the output is directly unit-testable. See server.py
for the HTTP layer that calls this.
"""

import html
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


def render_rows_html(jobs: list[dict[str, Any]]) -> str:
    """The <tbody> contents only - reused by both the full page and the
    polling/filtering endpoint that refreshes just the table body.
    """
    if not jobs:
        return '<tr><td colspan="7" class="empty">No jobs tracked yet - run `job-bot run` first.</td></tr>'

    rows = []
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        title_cell = _title_cell(str(job.get("title", "")), str(job.get("url", "")))
        company = html.escape(str(job.get("company", "")))
        score = job.get("match_score")
        score_text = str(score) if score is not None else "-"
        status = html.escape(str(job.get("status", "")))
        applied_at = html.escape(str(job.get("applied_at") or "-"))
        safe_job_id = html.escape(job_id, quote=True)
        rows.append(
            "<tr>"
            f"<td>{title_cell}</td>"
            f"<td>{company}</td>"
            f"<td>{score_text}</td>"
            f"<td>{_badge(status)}</td>"
            f"<td>{applied_at}</td>"
            f'<td class="jobid">{html.escape(job_id)}</td>'
            "<td class=\"actions\">"
            f"{_status_select(job_id, str(job.get('status', '')))}"
            f'<button type="button" class="qa-button" data-job-id="{safe_job_id}">Q&amp;A</button>'
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
    search: str = "",
    sort: str = "first_seen_at",
    direction: str = "desc",
    refresh_seconds: int = 5,
) -> str:
    rows_html = render_rows_html(jobs)
    status_options = [("", "All statuses"), *((s, s.replace("_", " ")) for s in sorted(TRACKER_STATUSES))]
    sort_value = f"{sort}:{direction}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(PAGE_TITLE)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem;
          background: #f9fafb; color: #111827; }}
  h1 {{ font-size: 1.25rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 1.25rem; }}
  .toolbar {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
              margin-bottom: 1rem; }}
  .toolbar input, .toolbar select {{ font-size: 0.85rem; padding: 0.4rem 0.6rem;
              border: 1px solid #d1d5db; border-radius: 6px; background: #fff; }}
  .toolbar input[type="search"] {{ min-width: 220px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
  th, td {{ text-align: left; padding: 0.6rem 0.9rem; border-bottom: 1px solid #e5e7eb;
            font-size: 0.9rem; vertical-align: middle; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  td.jobid {{ color: #9ca3af; font-size: 0.75rem; }}
  td.empty {{ color: #6b7280; text-align: center; padding: 2rem; }}
  td.actions {{ display: flex; gap: 0.4rem; align-items: center; white-space: nowrap; }}
  .status-select {{ font-size: 0.8rem; padding: 0.25rem 0.4rem; border-radius: 4px;
              border: 1px solid #d1d5db; }}
  .qa-button {{ font-size: 0.8rem; padding: 0.25rem 0.6rem; border-radius: 4px;
              border: 1px solid #d1d5db; background: #fff; cursor: pointer; }}
  .qa-button:hover {{ background: #f3f4f6; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .pager {{ display: flex; gap: 0.75rem; align-items: center; margin-top: 1rem;
            font-size: 0.85rem; color: #4b5563; }}
  .pager button {{ font-size: 0.8rem; padding: 0.3rem 0.7rem; border-radius: 6px;
              border: 1px solid #d1d5db; background: #fff; cursor: pointer; }}
  .pager button:disabled {{ opacity: 0.5; cursor: default; }}
  dialog {{ border: none; border-radius: 10px; padding: 1.25rem 1.5rem; max-width: 32rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15); }}
  dialog::backdrop {{ background: rgba(0,0,0,0.35); }}
  .qa-list dt {{ font-weight: 600; margin-top: 0.75rem; }}
  .qa-list dd {{ margin: 0.25rem 0 0; color: #374151; }}
  #qaClose {{ margin-top: 1rem; }}
</style>
</head>
<body>
<h1>{html.escape(PAGE_TITLE)}</h1>
<p class="subtitle">Auto-refreshes every {refresh_seconds}s. Update a status inline below, or via
`job-bot status` / `job-bot run` / `job-bot gmail-sync`.</p>

<form class="toolbar" id="filters">
  <input type="search" id="q" name="q" placeholder="Search title or company..."
         value="{html.escape(search, quote=True)}">
  <select id="status" name="status">
    {_options_html(status_options, status)}
  </select>
  <select id="sort" name="sort">
    {_options_html(SORT_OPTIONS, sort_value)}
  </select>
</form>

<table>
  <thead>
    <tr><th>Title</th><th>Company</th><th>Score</th><th>Status</th><th>Applied</th><th>Job ID</th><th>Actions</th></tr>
  </thead>
  <tbody id="rows">
{rows_html}
  </tbody>
</table>

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

<script>
// Initial filter/sort/page state is read from the DOM (already HTML-escaped
// by the server) rather than interpolated as string literals here, so a
// crafted search term can't break out of this block via the closing script
// tag - see render_page_html's docstring-adjacent note in render.py.
const state = {{
  q: document.getElementById('q').value,
  status: document.getElementById('status').value,
  sort: document.getElementById('sort').value.split(':')[0],
  dir: document.getElementById('sort').value.split(':')[1],
  page: {page},
}};
const PAGE_SIZE = {page_size};

function buildQuery() {{
  const params = new URLSearchParams();
  if (state.q) params.set('q', state.q);
  if (state.status) params.set('status', state.status);
  params.set('sort', state.sort);
  params.set('dir', state.dir);
  params.set('page', String(state.page));
  return params.toString();
}}

async function refresh() {{
  try {{
    const res = await fetch('/api/rows?' + buildQuery());
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

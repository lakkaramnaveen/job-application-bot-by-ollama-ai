"""Pure HTML-rendering functions for the one-page tracker dashboard - no
server or I/O here, so the output is directly unit-testable. See server.py
for the HTTP layer that calls this.
"""

import html
from typing import Any

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


def _badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#6b7280")
    label = html.escape(status.replace("_", " "))
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:999px;font-size:12px;white-space:nowrap">{label}</span>'
    )


def render_rows_html(jobs: list[dict[str, Any]]) -> str:
    """The <tbody> contents only - reused by both the full page and the
    polling endpoint that refreshes just the table body.
    """
    if not jobs:
        return '<tr><td colspan="6" class="empty">No jobs tracked yet - run `job-bot run` first.</td></tr>'

    rows = []
    for job in jobs:
        title = html.escape(str(job.get("title", "")))
        company = html.escape(str(job.get("company", "")))
        url = html.escape(str(job.get("url", "")), quote=True)
        score = job.get("match_score")
        score_text = str(score) if score is not None else "-"
        status = html.escape(str(job.get("status", "")))
        applied_at = html.escape(str(job.get("applied_at") or "-"))
        rows.append(
            "<tr>"
            f'<td><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a></td>'
            f"<td>{company}</td>"
            f"<td>{score_text}</td>"
            f"<td>{_badge(status)}</td>"
            f"<td>{applied_at}</td>"
            f'<td class="jobid">{html.escape(str(job.get("job_id", "")))}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def render_page_html(jobs: list[dict[str, Any]], refresh_seconds: int = 5) -> str:
    rows_html = render_rows_html(jobs)
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
  .subtitle {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
  th, td {{ text-align: left; padding: 0.6rem 0.9rem; border-bottom: 1px solid #e5e7eb;
            font-size: 0.9rem; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  td.jobid {{ color: #9ca3af; font-size: 0.75rem; }}
  td.empty {{ color: #6b7280; text-align: center; padding: 2rem; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>{html.escape(PAGE_TITLE)}</h1>
<p class="subtitle">Auto-refreshes every {refresh_seconds}s. Statuses update via
`job-bot run` and `job-bot gmail-sync`, or set them by hand with `job-bot status`.</p>
<table>
  <thead>
    <tr><th>Title</th><th>Company</th><th>Score</th><th>Status</th><th>Applied</th><th>Job ID</th></tr>
  </thead>
  <tbody id="rows">
{rows_html}
  </tbody>
</table>
<script>
async function refresh() {{
  try {{
    const res = await fetch('/api/rows');
    if (res.ok) {{
      document.getElementById('rows').innerHTML = await res.text();
    }}
  }} catch (e) {{
    // Network hiccup on a local server - next tick will retry.
  }}
}}
setInterval(refresh, {refresh_seconds * 1000});
</script>
</body>
</html>
"""

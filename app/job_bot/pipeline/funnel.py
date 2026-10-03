"""Where postings drop out between being found and being submitted - for
`job-bot report --funnel`.

docs/scaling.md's measured conclusion: most scored postings are good fits
(87% clear the bar), so the cost that matters is model work on fits that
then never get submitted. This turns the audit log into that funnel -
searches, postings considered, fits, materials written, applied - plus
what was skipped (by reason) and what failed (by FailureClass, see
pipeline/failures.py), so the biggest leak is visible at a glance.

Pure: reads audit-log entries, returns a value; cli.py prints it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_FAILURE_ACTIONS = frozenset({"search_error", "prep_error", "apply_error"})


@dataclass(frozen=True)
class Funnel:
    searches: int = 0
    considered: int = 0  # scored fresh, or an earlier score reused
    fits: int = 0  # cleared the bar
    materials: int = 0  # tailored resume + cover letter written
    applied: int = 0
    skipped: dict[str, int] = field(default_factory=dict)  # by reason, before any model call
    failures: dict[str, int] = field(default_factory=dict)  # by FailureClass value


def build_funnel(entries: Iterable[dict[str, Any]], *, since: datetime) -> Funnel:
    """Aggregate audit-log entries logged at or after `since`. A failure
    logged before failures carried a class counts as "unclassified"."""
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    fits = 0
    for entry in entries:
        when = _parse(entry.get("timestamp"))
        if when is None or when < since:
            continue
        action = str(entry.get("action", ""))
        details = entry.get("details") or {}
        if action == "scored":
            counts["considered"] += 1
            fits += bool(details.get("should_apply"))
        elif action == "reused_score":
            counts["considered"] += 1
            fits += 1  # only a score still clearing today's bar is reused
        elif action.startswith("skip_"):
            skipped[action.removeprefix("skip_")] += 1
        elif action in _FAILURE_ACTIONS:
            failures[str(details.get("failure_class") or "unclassified")] += 1
        else:
            counts[action] += 1
    return Funnel(
        searches=counts["search"],
        considered=counts["considered"],
        fits=fits,
        materials=counts["generated_materials"],
        applied=counts["applied"],
        skipped=dict(skipped.most_common()),
        failures=dict(failures.most_common()),
    )


def format_funnel(funnel: Funnel, *, days: float) -> list[str]:
    def pct(part: int, whole: int) -> str:
        return f"{100 * part / whole:.0f}%" if whole else "-"

    lines = [
        f"Funnel, last {days:g} day(s):",
        f"  searches              {funnel.searches}",
        f"  postings considered   {funnel.considered}",
        f"  fits (cleared bar)    {funnel.fits}  ({pct(funnel.fits, funnel.considered)} of considered)",
        f"  materials written     {funnel.materials}  ({pct(funnel.materials, funnel.fits)} of fits)",
        f"  applied               {funnel.applied}  ({pct(funnel.applied, funnel.fits)} of fits)",
    ]
    if funnel.skipped:
        lines.append("  skipped before scoring: " + ", ".join(f"{k} {v}" for k, v in funnel.skipped.items()))
    if funnel.failures:
        lines.append("  failures by class: " + ", ".join(f"{k} {v}" for k, v in funnel.failures.items()))
    return lines


def _parse(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

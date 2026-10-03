"""build_funnel(): where postings drop out between found and submitted -
see job_bot/pipeline/funnel.py."""

from datetime import UTC, datetime

from job_bot.pipeline.funnel import Funnel, build_funnel, format_funnel

SINCE = datetime(2026, 10, 2, 0, 0, tzinfo=UTC)


def e(action, **details):
    return {"timestamp": "2026-10-02T15:00:00+00:00", "action": action, "details": details}


def test_counts_each_stage_skips_by_reason_and_failures_by_class():
    entries = [
        e("search"),
        e("scored", should_apply=True),
        e("scored", should_apply=False),
        e("reused_score"),
        e("generated_materials"),
        e("generated_materials"),
        e("applied"),
        e("skip_company_limit"),
        e("skip_company_limit"),
        e("skip_blacklisted"),
        e("apply_error", failure_class="posting"),
        e("prep_error", failure_class="transient"),
        e("search_error"),  # logged before failures carried a class
        {"timestamp": "2026-10-01T09:00:00+00:00", "action": "applied", "details": {}},  # before the window
        {"timestamp": "garbage", "action": "applied", "details": {}},
    ]

    funnel = build_funnel(entries, since=SINCE)

    assert funnel == Funnel(
        searches=1,
        considered=3,
        fits=2,
        materials=2,
        applied=1,
        skipped={"company_limit": 2, "blacklisted": 1},
        failures={"posting": 1, "transient": 1, "unclassified": 1},
    )


def test_format_shows_conversion_and_handles_an_empty_window():
    lines = format_funnel(Funnel(considered=4, fits=2, materials=2, applied=1), days=1)
    assert "  fits (cleared bar)    2  (50% of considered)" in lines
    assert "  applied               1  (50% of fits)" in lines
    assert "  applied               0  (- of fits)" in format_funnel(Funnel(), days=1)

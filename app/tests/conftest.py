import time

import pytest


@pytest.fixture
def chicago_tz(monkeypatch):
    """Pin the local timezone (a UTC-negative zone, where UTC and local
    calendar dates disagree every evening) so local-time bucketing and
    display are deterministic regardless of the machine running the suite.
    """
    monkeypatch.setenv("TZ", "America/Chicago")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture(autouse=True)
def _no_search_pacing(monkeypatch):
    """The search's human-paced waits (between result pages, while scrolling
    the results list) are real seconds - zero in tests, which check what's
    loaded, not how slowly."""
    monkeypatch.setattr("job_bot.browser.linkedin_adapter.SEARCH_PAGE_PAUSE_SECONDS", 0)
    monkeypatch.setattr("job_bot.browser.linkedin_adapter.RESULTS_SCROLL_WAIT_SECONDS", 0)

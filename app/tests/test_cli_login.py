"""cmd_login had no test coverage at all before this - these fake out the
browser context/page the same way test_cli_run.py fakes the adapter, so the
wait-for-navigation logic (added to replace a blocking input() that crashed
with EOFError in any environment without a live interactive terminal
attached to the process) is verified without a real browser or network call.
"""

import contextlib

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from job_bot.cli import LOGIN_WAIT_TIMEOUT_MS, _login_finished, cmd_login
from job_bot.config import Settings


def make_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="claude",
        anthropic_api_key="sk-ant-fake",
        resume_path=tmp_path / "resume.txt",
        faq_path=tmp_path / "faq.json",
        blacklist_path=tmp_path / "blacklist.json",
        db_path=tmp_path / "db.sqlite3",
        browser_profile_dir=tmp_path / "profile",
        audit_log_path=tmp_path / "audit.log",
        applications_dir=tmp_path / "applications",
    )


class FakePage:
    def __init__(self, wait_for_url_effect: Exception | None = None):
        self.goto_calls: list[str] = []
        self.wait_for_url_calls: list[tuple] = []
        self._wait_for_url_effect = wait_for_url_effect

    def goto(self, url, **kwargs):
        self.goto_calls.append(url)

    def wait_for_url(self, predicate, *, timeout=None):
        self.wait_for_url_calls.append((predicate, timeout))
        if self._wait_for_url_effect is not None:
            raise self._wait_for_url_effect


class FakeContext:
    def __init__(self, page: FakePage):
        self._page = page

    def new_page(self) -> FakePage:
        return self._page


def fake_browser_session_factory(page: FakePage):
    @contextlib.contextmanager
    def _fake_browser_session(profile_dir, headless=False, cdp_url=None):
        yield FakeContext(page)

    return _fake_browser_session


def test_login_waits_for_navigation_and_reports_success(tmp_path, monkeypatch, capsys):
    page = FakePage()
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session_factory(page))

    cmd_login(make_settings(tmp_path))

    assert page.goto_calls == ["https://www.linkedin.com/login"]
    assert len(page.wait_for_url_calls) == 1
    predicate, timeout = page.wait_for_url_calls[0]
    assert predicate is _login_finished
    assert timeout == LOGIN_WAIT_TIMEOUT_MS
    assert "Session saved to" in capsys.readouterr().out


def test_login_timeout_reports_clearly_without_crashing_or_false_success(tmp_path, monkeypatch, capsys):
    page = FakePage(wait_for_url_effect=PlaywrightTimeoutError("timed out"))
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session_factory(page))

    cmd_login(make_settings(tmp_path))  # must not raise

    out = capsys.readouterr().out
    assert "Still on the login page" in out
    assert "Session saved to" not in out


def test_login_reports_cleanly_when_the_browser_closes_mid_wait(tmp_path, monkeypatch, capsys):
    """Distinct from the timeout above: the browser/tab itself closed (the
    user closed the window, it crashed, or the host process was killed)
    before login finished - not a bug to surface as a raw traceback.
    """
    page = FakePage(wait_for_url_effect=PlaywrightError("Target page, context or browser has been closed"))
    monkeypatch.setattr("job_bot.cli.browser_session", fake_browser_session_factory(page))

    cmd_login(make_settings(tmp_path))  # must not raise

    out = capsys.readouterr().out
    assert "Browser closed before login finished" in out
    assert "Session saved to" not in out


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.linkedin.com/login", False),
        ("https://www.linkedin.com/login/authwall?trk=x", False),
        ("https://www.linkedin.com/checkpoint/challenge/", False),
        ("https://www.linkedin.com/checkpoint/lg/login-submit", False),
        ("https://www.linkedin.com/feed/", True),
        ("https://www.linkedin.com/in/someone/", True),
    ],
)
def test_login_finished_predicate(url, expected):
    assert _login_finished(url) is expected

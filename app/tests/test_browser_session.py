"""browser_session() had no test coverage at all before this. These fake out
sync_playwright() itself (rather than driving a real browser) to verify the
two modes' distinct contracts - most importantly that CDP-attach mode never
closes the context it's handed, since that context is the user's own,
already-running Chrome rather than a browser job-bot launched itself.
"""

import job_bot.browser.session as session_module
from job_bot.browser.session import browser_session


class FakeContext:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self, contexts):
        self.contexts = contexts
        self.new_context_called = False

    def new_context(self):
        self.new_context_called = True
        ctx = FakeContext()
        self.contexts.append(ctx)
        return ctx


class FakeChromium:
    def __init__(self):
        self.connect_over_cdp_calls: list[str] = []
        self.launch_persistent_context_calls: list[dict] = []
        self.cdp_browser: FakeBrowser | None = None
        self.persistent_context: FakeContext | None = None

    def connect_over_cdp(self, endpoint_url, **kwargs):
        self.connect_over_cdp_calls.append(endpoint_url)
        return self.cdp_browser

    def launch_persistent_context(self, **kwargs):
        self.launch_persistent_context_calls.append(kwargs)
        return self.persistent_context


class FakePlaywrightCM:
    def __init__(self, chromium: FakeChromium):
        self.chromium = chromium

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_cdp_mode_attaches_and_reuses_the_existing_context_without_closing_it(monkeypatch, tmp_path):
    existing_context = FakeContext()
    chromium = FakeChromium()
    chromium.cdp_browser = FakeBrowser(contexts=[existing_context])
    monkeypatch.setattr(session_module, "sync_playwright", lambda: FakePlaywrightCM(chromium))

    with browser_session(tmp_path / "profile", cdp_url="http://localhost:9222") as context:
        assert context is existing_context

    assert chromium.connect_over_cdp_calls == ["http://localhost:9222"]
    assert chromium.cdp_browser.new_context_called is False
    # The defining property of CDP-attach mode: this is the user's actual,
    # already-running browser - closing it here would close their real
    # Chrome window, not just clean up something job-bot launched.
    assert existing_context.closed is False


def test_cdp_mode_creates_a_context_when_the_browser_has_none_yet(monkeypatch, tmp_path):
    chromium = FakeChromium()
    chromium.cdp_browser = FakeBrowser(contexts=[])
    monkeypatch.setattr(session_module, "sync_playwright", lambda: FakePlaywrightCM(chromium))

    with browser_session(tmp_path / "profile", cdp_url="http://localhost:9222") as context:
        assert chromium.cdp_browser.new_context_called is True
        assert context in chromium.cdp_browser.contexts

    assert context.closed is False


def test_default_mode_launches_an_isolated_profile_and_closes_it_on_exit(monkeypatch, tmp_path):
    chromium = FakeChromium()
    chromium.persistent_context = FakeContext()
    monkeypatch.setattr(session_module, "sync_playwright", lambda: FakePlaywrightCM(chromium))
    profile_dir = tmp_path / "profile"

    with browser_session(profile_dir, headless=True) as context:
        assert context is chromium.persistent_context
        assert context.closed is False

    assert chromium.persistent_context.closed is True
    assert profile_dir.exists()
    assert chromium.connect_over_cdp_calls == []
    [call_kwargs] = chromium.launch_persistent_context_calls
    assert call_kwargs == {"user_data_dir": str(profile_dir), "headless": True}

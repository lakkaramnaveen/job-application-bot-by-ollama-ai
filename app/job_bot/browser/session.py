"""Provides the one BrowserContext every browser-driving command (cmd_login,
cmd_run) runs against. Two distinct modes - see browser_session()'s
docstring for the tradeoff between them, and README.md's "Browser profile"
section for the user-facing explanation.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import BrowserContext, sync_playwright


@contextmanager
def browser_session(
    profile_dir: Path, headless: bool = False, cdp_url: str | None = None
) -> Iterator[BrowserContext]:
    """Yield a BrowserContext to drive, in one of two modes:

    - Default (cdp_url is None): launches Chromium with its own persistent,
      isolated profile under profile_dir. Login happens once, manually, in
      the visible window this opens - the bot never sees or stores the
      platform password, and this profile only ever holds whatever cookies
      LinkedIn (or another job board) sets, never the user's everyday
      browsing sessions (email, banking, ...). See SECURITY.md.

    - cdp_url set: attaches to an already-running Chrome via the Chrome
      DevTools Protocol (started separately with
      --remote-debugging-port=<port>) and reuses its first existing context
      - the user's actual, already-logged-in profile - instead of a
      separate one. The context is deliberately never closed here: it's the
      user's real browser, and closing it (or the underlying connection's
      browser handle) would close their actual Chrome windows, not just the
      tab this opened. This trades the isolation above for not having to
      log in a second time - a real security tradeoff (that Chrome's
      debugging port gives any local process full control over it and read
      access to every cookie in it, not just this app), not merely a
      convenience toggle.
    """
    if cdp_url:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(cdp_url)
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            yield context
        return

    profile_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=headless,
        )
        try:
            yield context
        finally:
            context.close()

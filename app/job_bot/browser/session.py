from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import BrowserContext, sync_playwright


@contextmanager
def browser_session(profile_dir: Path, headless: bool = False) -> Iterator[BrowserContext]:
    """Launch Chromium with a persistent profile so the user's login survives
    across runs. Login happens once, manually, in the visible window this
    opens - the bot never sees or stores the platform password.
    """
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

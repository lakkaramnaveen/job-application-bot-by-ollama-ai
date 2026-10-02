"""What a failure means for the run - one classifier instead of the same
"is this fatal?" if-chain repeated after every step that can fail.

Before this module, cli.py's _run_apply_cycle() checked the same three
run-ending conditions (Ollama unreachable, Claude misconfigured, the
browser gone) in two copy-pasted chains - after preparing materials and
after applying - plus a third copy on the search path. Every new
run-ending condition had to be added in each place, and one copy
drifting from another was a real bug class (a duplicated check shipped
and was caught in review on 2026-10-02). classify_failure() is now the
single answer to "after this exception, does the run continue?".
"""

from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page

from job_bot.browser.linkedin_adapter import NavigationFailed
from job_bot.llm.claude_provider import ClaudeProviderError
from job_bot.llm.ollama_provider import OllamaProviderError


@dataclass(frozen=True)
class FailureVerdict:
    """How one posting's failure affects the rest of the run.

    fatal: every remaining posting (and every later --loop cycle) would
        fail the same way - stop the run, printing `message`.
    navigation_refused: LinkedIn refused to load the page - one data point
        for the caller's consecutive-refusal (rate limiting) streak.
    """

    fatal: bool = False
    message: str = ""
    navigation_refused: bool = False


POSTING_ONLY = FailureVerdict()


def classify_failure(e: Exception, page: Page) -> FailureVerdict:
    """The verdict for an exception raised while preparing or applying to
    one posting. Order matters only for messages: each fatal condition is
    independent and any one of them stops the run."""
    if _is_ollama_unreachable(e):
        return FailureVerdict(
            fatal=True,
            message="Ollama is unreachable - stopping the run instead of repeating this for every posting.",
        )
    if _is_claude_misconfigured(e):
        return FailureVerdict(
            fatal=True,
            message=f"Claude provider is misconfigured ({e}) - stopping the run instead of repeating this "
            "for every posting.",
        )
    if _browser_is_gone(e, page):
        return FailureVerdict(fatal=True, message=BROWSER_GONE_MESSAGE)
    if isinstance(e, NavigationFailed):
        return FailureVerdict(navigation_refused=True)
    return POSTING_ONLY


def _browser_is_gone(e: Exception, page: Page) -> bool:
    """True when the bot's own browser can't be used any more - its window
    was closed, it crashed, or the Playwright driver behind it died. Every
    remaining posting (and every later --loop search) would fail the same
    way, so the run stops instead.
    """
    if isinstance(e, PlaywrightError) and (
        "Target page, context or browser has been closed" in str(e)
        or "Connection closed while reading from the driver" in str(e)
    ):
        return True
    if "Connection closed while reading from the driver" in str(e):
        return True  # raised as a bare Exception by Playwright's sync layer
    try:
        return page.is_closed()
    except Exception:  # noqa: BLE001 - a page we can't even ask is gone
        return True


BROWSER_GONE_MESSAGE = (
    "The bot's browser window was closed or crashed - stopping the run. Start it again with `job-bot run`."
)


def _is_ollama_unreachable(e: Exception) -> bool:
    """True for the specific OllamaProviderError raised when the local
    server can't be connected to at all (see ollama_provider.py's
    httpx.ConnectError handling) - deliberately narrower than "any
    OllamaProviderError", since the other two cases it covers (a malformed
    JSON response after retries, a 404 for an unpulled model) aren't
    necessarily going to fail identically on every remaining posting the
    way a fully unreachable server is. Checked by message rather than a
    dedicated exception subclass since that string is this project's own
    and unlikely to drift without both sides being updated together.
    """
    return isinstance(e, OllamaProviderError) and "Could not reach Ollama" in str(e)


# The three claude_provider.py ClaudeProviderError messages that mean the
# provider is fundamentally misconfigured - not a one-off request failure
# (rate limit, network blip) - and will therefore raise the exact same
# error on every remaining posting in the batch too. Same reasoning as
# _is_ollama_unreachable() above, and the same fix: recognize it after the
# first failure instead of repeating an identical, guaranteed-to-fail LLM
# call (and an identical printed error) once per remaining posting.
_CLAUDE_CONFIG_ERROR_MARKERS = (
    "Invalid ANTHROPIC_API_KEY",
    "API key lacks permission",
    "not found.",  # ClaudeProviderError(f"Model '{model}' not found.")
)


def _is_claude_misconfigured(e: Exception) -> bool:
    return isinstance(e, ClaudeProviderError) and any(
        marker in str(e) for marker in _CLAUDE_CONFIG_ERROR_MARKERS
    )

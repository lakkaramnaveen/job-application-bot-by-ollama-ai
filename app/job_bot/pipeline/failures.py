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
from enum import Enum

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page

from job_bot.browser.linkedin_adapter import (
    FieldsRejected,
    LinkedInSignedOut,
    NavigationFailed,
    UnansweredRequiredQuestion,
)
from job_bot.llm.circuit_breaker import ModelUnavailable
from job_bot.llm.claude_provider import ClaudeProviderError
from job_bot.llm.ollama_provider import OLLAMA_UNAUTHORIZED, OllamaProviderError


class FailureClass(Enum):
    """How a failure should be handled anywhere in the system - retry,
    back off, move on, ask the user, or stop. See docs/scaling.md
    ("Error handling as a system"); logged with every failure so failures
    can be counted and alerted on by class rather than by error text."""

    TRANSIENT = "transient"  # likely to succeed if retried soon
    THROTTLED = "throttled"  # the other side is rate-limiting: back off
    POSTING = "posting"  # this posting can't be done; move on
    USER_ACTION = "user_action"  # only the user can fix it
    FATAL = "fatal"  # the run can't continue


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
    failure_class: FailureClass = FailureClass.POSTING
    # End this cycle as throttled (--loop backs off) - the model's circuit
    # breaker is open, so every remaining posting would fail the same way.
    throttle_cycle: bool = False


POSTING_ONLY = FailureVerdict()

# Longest wait between --loop cycles while LinkedIn keeps refusing page
# loads - see throttle_backoff_minutes().
MAX_THROTTLE_BACKOFF_MINUTES = 120


def throttle_backoff_minutes(base_minutes: float, streak: int) -> float:
    """How long --loop waits after `streak` throttled cycles in a row:
    base, then doubling each time (20 -> 40 -> 80 -> 120 with the default
    20-minute interval), capped at MAX_THROTTLE_BACKOFF_MINUTES (or `base`,
    if that's already longer).

    Real case (2026-10-02): once LinkedIn started refusing page loads it
    kept refusing for hours, and a fixed 20-minute retry meant three probes
    an hour against a session already being rate-limited - the request
    pattern most likely to get an automated account restricted.
    """
    if streak <= 1:
        return base_minutes
    return min(base_minutes * 2 ** (streak - 1), max(MAX_THROTTLE_BACKOFF_MINUTES, base_minutes))


def classify_failure(e: Exception, page: Page) -> FailureVerdict:
    """The verdict for an exception raised while preparing or applying to
    one posting. Order matters only for messages: each fatal condition is
    independent and any one of them stops the run. Every verdict carries a
    FailureClass; `fatal` and `navigation_refused` are what this run acts on."""
    if _is_ollama_unreachable(e):
        return FailureVerdict(
            fatal=True,
            message="Ollama is unreachable - stopping the run instead of repeating this for every posting.",
            failure_class=FailureClass.FATAL,
        )
    if isinstance(e, OllamaProviderError) and str(e).startswith(OLLAMA_UNAUTHORIZED):
        return FailureVerdict(
            fatal=True,
            message=f"{e} Stopping the run - every call would be refused the same way.",
            failure_class=FailureClass.USER_ACTION,
        )
    if _is_claude_misconfigured(e):
        return FailureVerdict(
            fatal=True,
            message=f"Claude provider is misconfigured ({e}) - stopping the run instead of repeating this "
            "for every posting.",
            failure_class=FailureClass.USER_ACTION,  # a bad key or model name - only the user can fix it
        )
    if _browser_is_gone(e, page):
        return FailureVerdict(fatal=True, message=BROWSER_GONE_MESSAGE, failure_class=FailureClass.FATAL)
    if isinstance(e, NavigationFailed):
        return FailureVerdict(navigation_refused=True, failure_class=FailureClass.THROTTLED)
    if isinstance(e, ModelUnavailable):
        return FailureVerdict(
            throttle_cycle=True,
            message=f"{e} Ending this cycle instead of failing every remaining posting.",
            failure_class=FailureClass.THROTTLED,
        )
    return FailureVerdict(failure_class=failure_class_of(e))


def failure_class_of(e: Exception) -> FailureClass:
    """The FailureClass of an exception that doesn't end the run on its own -
    used for the per-posting verdict above, and for logging any failure."""
    if isinstance(e, NavigationFailed | ModelUnavailable):
        return FailureClass.THROTTLED
    if isinstance(e, LinkedInSignedOut | UnansweredRequiredQuestion | FieldsRejected):
        # Signed out: `job-bot login`. A question the bot can't answer, or an
        # answer LinkedIn refused: `job-bot review-answers`.
        return FailureClass.USER_ACTION
    if _is_ollama_unreachable(e):
        return FailureClass.FATAL
    if _is_claude_misconfigured(e):
        return FailureClass.USER_ACTION
    if isinstance(e, OllamaProviderError | ClaudeProviderError):
        # Malformed JSON after retries, a generation stopped at its deadline,
        # a model-side abort - or the provider rate-limiting us.
        return FailureClass.THROTTLED if "rate limit" in str(e).casefold() else FailureClass.TRANSIENT
    return FailureClass.POSTING


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

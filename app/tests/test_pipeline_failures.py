"""classify_failure(): the single answer to "after this exception, does
the run continue?" - see job_bot/pipeline/failures.py."""

import pytest
from playwright.sync_api import Error as PlaywrightError

from job_bot.browser.linkedin_adapter import (
    FieldsRejected,
    LinkedInSignedOut,
    NavigationFailed,
    UnansweredRequiredQuestion,
)
from job_bot.llm.claude_provider import ClaudeProviderError
from job_bot.llm.ollama_provider import OllamaProviderError
from job_bot.pipeline.failures import BROWSER_GONE_MESSAGE, POSTING_ONLY, FailureClass, classify_failure


class Page:
    def __init__(self, closed=False, broken=False):
        self._closed = closed
        self._broken = broken

    def is_closed(self):
        if self._broken:
            raise RuntimeError("driver gone")
        return self._closed


@pytest.mark.parametrize(
    ("error", "message_start"),
    [
        (OllamaProviderError("Could not reach Ollama at http://localhost:11434."), "Ollama is unreachable"),
        (ClaudeProviderError("Invalid ANTHROPIC_API_KEY."), "Claude provider is misconfigured"),
        (PlaywrightError("Page.goto: Target page, context or browser has been closed"), BROWSER_GONE_MESSAGE),
        (Exception("Page.title: Connection closed while reading from the driver"), BROWSER_GONE_MESSAGE),
    ],
)
def test_run_ending_failures_are_fatal_with_their_message(error, message_start):
    verdict = classify_failure(error, Page())
    assert verdict.fatal
    assert verdict.message.startswith(message_start)


def test_a_closed_or_unreachable_page_is_fatal_whatever_the_error():
    assert classify_failure(RuntimeError("anything"), Page(closed=True)).fatal
    assert classify_failure(RuntimeError("anything"), Page(broken=True)).fatal


def test_a_refused_page_load_counts_toward_the_throttling_streak_but_is_not_fatal():
    verdict = classify_failure(NavigationFailed("Failed to load https://x/jobs/view/1/ after 3 attempts"), Page())
    assert verdict.navigation_refused and not verdict.fatal


@pytest.mark.parametrize(
    ("error", "failure_class"),
    [
        (RuntimeError("Could not complete the Easy Apply form for job 1 (stuck on a step ...)"), FailureClass.POSTING),
        (OllamaProviderError("Model 'm' did not return schema-valid JSON after 3 attempts"), FailureClass.TRANSIENT),
        (ClaudeProviderError("Rate limited, try again later"), FailureClass.THROTTLED),
        (UnansweredRequiredQuestion("1", "Security clearance level?", "No answer"), FailureClass.USER_ACTION),
        (FieldsRejected("1", [("Q?", "Invalid input")], "stuck"), FailureClass.USER_ACTION),
        (LinkedInSignedOut("signed out"), FailureClass.USER_ACTION),
    ],
)
def test_everything_else_costs_only_this_posting_with_its_class(error, failure_class):
    verdict = classify_failure(error, Page())
    assert not verdict.fatal and not verdict.navigation_refused  # the run carries on
    assert verdict.failure_class is failure_class


@pytest.mark.parametrize(
    ("error", "failure_class"),
    [
        (OllamaProviderError("Could not reach Ollama at http://localhost:11434."), FailureClass.FATAL),
        (ClaudeProviderError("Invalid ANTHROPIC_API_KEY."), FailureClass.USER_ACTION),
        (PlaywrightError("Page.goto: Target page, context or browser has been closed"), FailureClass.FATAL),
        (NavigationFailed("Failed to load https://x/jobs/view/1/ after 3 attempts"), FailureClass.THROTTLED),
    ],
)
def test_run_level_failures_carry_their_class(error, failure_class):
    assert classify_failure(error, Page()).failure_class is failure_class


def test_posting_only_is_the_plain_posting_class():
    assert POSTING_ONLY.failure_class is FailureClass.POSTING


@pytest.mark.parametrize(
    ("streak", "minutes"),
    [(0, 20), (1, 20), (2, 40), (3, 80), (4, 120), (10, 120)],
)
def test_throttle_backoff_doubles_per_throttled_cycle_and_caps(streak, minutes):
    from job_bot.pipeline.failures import throttle_backoff_minutes

    assert throttle_backoff_minutes(20, streak) == minutes


def test_throttle_backoff_never_shortens_a_long_base_interval():
    from job_bot.pipeline.failures import throttle_backoff_minutes

    assert throttle_backoff_minutes(180, 3) == 180


def test_a_rejected_model_key_ends_the_run_as_user_action():
    error = OllamaProviderError("Ollama rejected the request (HTTP 401) - check OLLAMA_API_KEY in .env.")
    verdict = classify_failure(error, Page())
    assert verdict.fatal and verdict.failure_class is FailureClass.USER_ACTION
    assert "every call would be refused" in verdict.message

"""configure_logging() had no test coverage at all.

These tests intercept logging.basicConfig() itself rather than asserting
on logging.getLogger().level afterward: pytest's own log-capturing handler
gets attached to the root logger right as each test body starts (after
fixture setup runs), so by the time configure_logging() -> basicConfig()
executes, the root logger already has handlers - and basicConfig() is a
documented no-op whenever the root logger already has any. Asserting on
the real root logger's level would just be asserting pytest didn't touch
it, not that configure_logging() computed the right level.
"""

import logging

from job_bot.logging_setup import configure_logging


def test_defaults_to_info_level_when_log_level_unset(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    captured = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))

    configure_logging()

    assert captured["level"] == logging.INFO


def test_respects_a_valid_log_level_env_var(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    captured = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))

    configure_logging()

    assert captured["level"] == logging.DEBUG


def test_log_level_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "warning")
    captured = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))

    configure_logging()

    assert captured["level"] == logging.WARNING


def test_falls_back_to_info_for_an_unrecognized_log_level(monkeypatch):
    """A typo'd LOG_LEVEL (e.g. "INF0") must not crash startup - it should
    silently fall back to INFO rather than raising or leaving level unset.
    """
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_REAL_LEVEL")
    captured = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))

    configure_logging()

    assert captured["level"] == logging.INFO


def test_quiets_the_noisy_playwright_logger(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)

    configure_logging()

    assert logging.getLogger("playwright").level == logging.WARNING


def test_httpx_request_lines_are_silenced_at_the_default_level(monkeypatch):
    """One "HTTP Request: POST .../api/chat" INFO line per LLM call buried
    the run's own output (seen in a live run). Default level hides them."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logging.getLogger("httpx").setLevel(logging.NOTSET)

    configure_logging()

    # .level, not getEffectiveLevel(): the root logger is often already at
    # WARNING under pytest, which would make an inherited level pass even
    # if configure_logging() never set anything.
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_log_level_debug_keeps_httpx_request_lines(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    logging.getLogger("httpx").setLevel(logging.NOTSET)
    logging.getLogger("httpcore").setLevel(logging.NOTSET)
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.DEBUG)

    configure_logging()

    assert logging.getLogger("httpx").level == logging.NOTSET  # left alone - inherits DEBUG

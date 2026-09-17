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

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

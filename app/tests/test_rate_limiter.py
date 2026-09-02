import pytest

from job_bot.config import HARD_DAILY_APPLICATION_CEILING
from job_bot.safety.rate_limiter import DailyCapReached, RateLimiter


def test_allows_applications_up_to_cap(tmp_path):
    limiter = RateLimiter(tmp_path / "db.sqlite3", daily_cap=3)
    for _ in range(3):
        limiter.record_application()
    assert limiter.remaining_today() == 0


def test_raises_once_cap_reached(tmp_path):
    limiter = RateLimiter(tmp_path / "db.sqlite3", daily_cap=1)
    limiter.record_application()
    with pytest.raises(DailyCapReached):
        limiter.record_application()


def test_config_cannot_exceed_hard_ceiling(tmp_path):
    limiter = RateLimiter(tmp_path / "db.sqlite3", daily_cap=HARD_DAILY_APPLICATION_CEILING + 1000)
    for _ in range(HARD_DAILY_APPLICATION_CEILING):
        limiter.record_application()
    assert limiter.remaining_today() == 0
    with pytest.raises(DailyCapReached):
        limiter.record_application()


def test_count_persists_across_instances(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    RateLimiter(db_path, daily_cap=5).record_application()
    limiter2 = RateLimiter(db_path, daily_cap=5)
    assert limiter2.count_today() == 1

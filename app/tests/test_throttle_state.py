from datetime import datetime, timedelta

from job_bot.pipeline.throttle_state import ThrottleState

T0 = datetime(2026, 10, 4, 10, 29)


class Clock:
    def __init__(self, at):
        self.at = at

    def __call__(self):
        return self.at


def test_the_back_off_survives_a_restart_and_grows_with_the_streak(tmp_path):
    clock = Clock(T0)
    path = tmp_path / "linkedin_throttle.json"
    assert ThrottleState(path, now=clock).resume_at(20) is None

    assert ThrottleState(path, now=clock).record_refusal() == 1
    # A new ThrottleState - a restarted run - still sees it.
    assert ThrottleState(path, now=clock).resume_at(20) == T0 + timedelta(minutes=20)

    assert ThrottleState(path, now=clock).record_refusal() == 2
    assert ThrottleState(path, now=clock).resume_at(20) == T0 + timedelta(minutes=40)

    clock.at = T0 + timedelta(minutes=41)
    assert ThrottleState(path, now=clock).resume_at(20) is None


def test_a_search_that_goes_through_ends_the_streak(tmp_path):
    state = ThrottleState(tmp_path / "linkedin_throttle.json", now=Clock(T0))
    state.record_refusal()
    state.clear()
    assert state.streak() == 0 and state.resume_at(20) is None
    state.clear()  # nothing to clear is fine


def test_a_damaged_state_file_means_not_throttled(tmp_path):
    path = tmp_path / "linkedin_throttle.json"
    for content in ("", "{not json", '{"streak": "x"}', '{"streak": 2}'):
        path.write_text(content)
        assert ThrottleState(path, now=Clock(T0)).resume_at(20) is None

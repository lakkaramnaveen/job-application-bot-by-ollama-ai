"""CircuitBreakerProvider: stop calling a model that keeps failing, then
try again after a cooldown - see job_bot/llm/circuit_breaker.py."""

import pytest

from job_bot.llm.circuit_breaker import CircuitBreakerProvider, ModelUnavailable
from job_bot.models.schemas import CoverLetter


class FlakyModel:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.stats = "model stats"

    def generate_structured(self, *, system, prompt, schema):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def call(breaker):
    return breaker.generate_structured(system="s", prompt="p", schema=CoverLetter)


def test_opens_after_the_threshold_and_fails_fast_during_the_cooldown():
    boom = RuntimeError("prediction aborted, token repeat limit reached")
    model, clock = FlakyModel([boom, boom, boom]), Clock()
    breaker = CircuitBreakerProvider(model, failure_threshold=3, cooldown_seconds=300, clock=clock)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            call(breaker)
    assert breaker.is_open

    clock.now = 100
    with pytest.raises(ModelUnavailable, match="failed 3 calls in a row .*token repeat limit.* 200s"):
        call(breaker)
    assert model.calls == 3  # the open circuit didn't call the model


def test_a_success_resets_the_failure_count():
    ok = CoverLetter(body="Hi.")
    boom = RuntimeError("bad json")
    model = FlakyModel([boom, boom, ok, boom, boom])
    breaker = CircuitBreakerProvider(model, failure_threshold=3, clock=Clock())

    for expected in (RuntimeError, RuntimeError, None, RuntimeError, RuntimeError):
        if expected is None:
            assert call(breaker) == ok
        else:
            with pytest.raises(expected):
                call(breaker)
    assert not breaker.is_open  # never 3 failures in a row


def test_after_the_cooldown_one_trial_call_closes_or_reopens_it():
    ok = CoverLetter(body="Hi.")
    boom = RuntimeError("bad json")
    model, clock = FlakyModel([boom, boom, boom, boom, ok]), Clock()
    breaker = CircuitBreakerProvider(model, failure_threshold=3, cooldown_seconds=300, clock=clock)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            call(breaker)

    clock.now = 301  # half-open: the trial fails -> open again
    with pytest.raises(RuntimeError):
        call(breaker)
    assert breaker.is_open

    clock.now = 700  # half-open again: the trial succeeds -> closed
    assert call(breaker) == ok
    assert not breaker.is_open


def test_stats_pass_through_to_the_wrapped_model():
    assert CircuitBreakerProvider(FlakyModel([])).stats == "model stats"

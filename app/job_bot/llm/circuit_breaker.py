"""A circuit breaker around any LLMProvider - step 3 of docs/scaling.md.

Each provider call already retries internally (the Ollama provider up to 3
attempts, each capped at MAX_GENERATION_SECONDS). When the model itself is
unhealthy - aborting with "token repeat limit reached", returning JSON that
won't validate, timing out - every remaining posting in the cycle still paid
for its own full set of attempts, minutes each. A model failure during form
filling also counted against that posting's MAX_APPLY_ATTEMPTS, blaming the
posting for the model's problem.

After `failure_threshold` failed calls in a row the circuit OPENS: calls fail
immediately with ModelUnavailable for `cooldown_seconds`. Then it's
HALF-OPEN: one trial call goes through, and its result closes the circuit
(success) or re-opens it (failure). pipeline/failures.py classifies
ModelUnavailable as THROTTLED and ends the cycle, so --loop's back-off takes
over instead of every posting failing in turn.

An unreachable Ollama server is still a run-ending error on its own (see
pipeline/failures.py) - the breaker is for a model that's up but failing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from job_bot.llm.base import LLMProvider, SchemaT

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 300.0


class ModelUnavailable(RuntimeError):
    """The circuit is open: recent model calls kept failing, so this one
    wasn't attempted."""


class CircuitBreakerProvider(LLMProvider):
    def __init__(
        self,
        inner: LLMProvider,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._inner = inner
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._last_error = ""

    @property
    def inner(self) -> LLMProvider:
        return self._inner

    @property
    def stats(self) -> Any:
        """The wrapped provider's GenerationStats, if it keeps any - so the
        per-cycle "Model:" line still reports through the wrapper."""
        return getattr(self._inner, "stats", None)

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None and self._clock() - self._opened_at < self._cooldown

    def generate_structured(self, *, system: str, prompt: str, schema: type[SchemaT]) -> SchemaT:
        if self.is_open:
            remaining = self._cooldown - (self._clock() - (self._opened_at or 0.0))
            raise ModelUnavailable(
                f"The model failed {self._consecutive_failures} calls in a row (last: {self._last_error}) - "
                f"not calling it again for {remaining:.0f}s."
            )
        # Closed, or half-open after the cooldown: this call is the trial.
        try:
            result = self._inner.generate_structured(system=system, prompt=prompt, schema=schema)
        except Exception as e:
            self._consecutive_failures += 1
            self._last_error = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
            if self._consecutive_failures >= self._threshold:
                self._opened_at = self._clock()
            raise
        self._consecutive_failures = 0
        self._opened_at = None
        return result

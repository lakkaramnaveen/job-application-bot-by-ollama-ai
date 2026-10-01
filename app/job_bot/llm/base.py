import dataclasses
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclasses.dataclass
class GenerationStats:
    """Running totals of a provider's LLM calls since the last reset() - the
    per-cycle performance line `job-bot run` prints (see cli.py's
    _print_cycle_summary()). Tokens and load time are whatever the backend
    reports; 0 when it doesn't.
    """

    calls: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    load_seconds: float = 0.0

    def record(self, *, seconds: float, prompt_tokens: int = 0, output_tokens: int = 0, load_seconds: float = 0.0) -> None:
        self.calls += 1
        self.seconds += seconds
        self.prompt_tokens += prompt_tokens
        self.output_tokens += output_tokens
        self.load_seconds += load_seconds

    def summary(self) -> str:
        avg_prompt = self.prompt_tokens // self.calls if self.calls else 0
        rate = self.output_tokens / self.seconds if self.seconds else 0.0
        return (
            f"Model: {self.calls} call(s), {self.seconds:.1f}s total ({self.seconds / max(self.calls, 1):.1f}s avg), "
            f"avg prompt {avg_prompt:,} tokens, {self.output_tokens:,} tokens generated ({rate:.0f}/s), "
            f"{self.load_seconds:.1f}s loading the model."
        )

    def reset(self) -> None:
        self.calls, self.seconds, self.prompt_tokens, self.output_tokens, self.load_seconds = 0, 0.0, 0, 0, 0.0


class LLMProvider(ABC):
    """Common interface every model backend (Claude, Ollama, ...) implements.

    Every call site in job_bot uses this method and a Pydantic schema, never a
    raw text completion - this keeps provider-specific SDK/response quirks out
    of the application logic and guarantees callers get validated data back.
    """

    @abstractmethod
    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        """Run one structured-output call and return a validated `schema` instance.

        `system` must contain only fixed, trusted instructions - never
        interpolate untrusted data (e.g. scraped job postings) into it. Put
        untrusted data in `prompt` instead, where it is inert data rather than
        instructions the model follows.
        """
        raise NotImplementedError

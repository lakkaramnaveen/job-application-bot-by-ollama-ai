from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


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

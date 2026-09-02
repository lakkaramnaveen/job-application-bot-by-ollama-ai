from collections.abc import Callable


class SubmitConfirmer:
    """Human-in-the-loop gate before a real application is submitted.

    Defaults to prompting on the terminal. Pass a custom `ask` function (e.g.
    to drive from tests, or a future GUI/queue) instead of the default input().
    """

    def __init__(self, required: bool, ask: Callable[[str], bool] | None = None):
        self._required = required
        self._ask = ask or self._default_ask

    @staticmethod
    def _default_ask(prompt: str) -> bool:
        reply = input(f"{prompt} [y/N] ").strip().lower()
        return reply in ("y", "yes")

    def confirm(self, summary: str) -> bool:
        """Returns True if the submission should proceed."""
        if not self._required:
            return True
        return self._ask(summary)

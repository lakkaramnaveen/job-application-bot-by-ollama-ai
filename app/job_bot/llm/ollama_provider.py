import json
import platform
import re
import subprocess
import time

import httpx
from pydantic import ValidationError

from job_bot.llm.base import GenerationStats, LLMProvider, SchemaT

DEFAULT_TIMEOUT = 120.0

# Two bounds on a single generation, added after a real hang: with
# streaming (see _read_chat_stream()), DEFAULT_TIMEOUT applies per chunk,
# not to the whole response - a generation that never stops never times
# out. Seen live (2026-10-01): one cover-letter generation kept a run stuck
# for 28+ minutes with the model server at full CPU. MAX_OUTPUT_TOKENS caps
# it server-side (the longest real output - a cover letter - is ~600
# tokens); MAX_GENERATION_SECONDS caps the wall-clock read of the stream,
# after which the client disconnects, which makes Ollama stop generating.
MAX_OUTPUT_TOKENS = 4096
MAX_GENERATION_SECONDS = 240.0

# Context window requested per call (num_ctx). Ollama reserves KV-cache
# memory for the whole window up front: on qwen3:30b (48 layers x 4 KV heads
# x 128 dims x 2 x fp16) that's ~98 KB per token, so its 32,768 default cost
# ~3.2 GB of RAM. Measured on real data, the largest prompt is now ~5k tokens
# (resume tailoring), plus up to MAX_OUTPUT_TOKENS of output - 12,288 fits
# that at ~1.2 GB. _context_window() grows it for an unusually large prompt
# rather than let Ollama silently truncate the input.
CONTEXT_TOKENS = 12288
_CONTEXT_STEP = 4096
# Deliberately pessimistic chars-per-token estimate (English averages ~4), so
# a prompt is never underestimated into a window it overflows.
_CHARS_PER_TOKEN_ESTIMATE = 3


def _context_window(system: str, prompt: str) -> int:
    """num_ctx for a request: CONTEXT_TOKENS, or the next multiple of
    _CONTEXT_STEP that fits a larger prompt plus MAX_OUTPUT_TOKENS."""
    needed = (len(system) + len(prompt)) // _CHARS_PER_TOKEN_ESTIMATE + MAX_OUTPUT_TOKENS
    if needed <= CONTEXT_TOKENS:
        return CONTEXT_TOKENS
    return -(-needed // _CONTEXT_STEP) * _CONTEXT_STEP

# A local model occasionally emits truncated/malformed JSON for no
# structural reason (seen live: qwen3:30b cutting a CoverLetter response off
# mid-string at ~1800 chars, well under any context/output limit) - a bare
# retry of the same request resolves it almost every time, the same
# reasoning as linkedin_adapter.py's _goto_with_retry() for a flaky page
# load. Not retried: ConnectError (Ollama isn't reachable at all - retrying
# without the user fixing that first can't help) and a 404 (the model isn't
# pulled - same reasoning).
MAX_GENERATION_ATTEMPTS = 3

# Matches a long run of literal "\n" (backslash-n, two characters each)
# escape sequences at the very end of an otherwise-unterminated JSON string
# value - see _repair_truncated_json_string()'s docstring for what this
# guards against. 3+ in a row is deliberately conservative: a real letter
# legitimately has isolated "\n\n" between paragraphs, never a long
# uninterrupted run of them.
_TRAILING_BLANK_LINE_PADDING = re.compile(r"(?:\\n){3,}$")

def _count_unescaped_quotes(text: str) -> int:
    """Counts '"' characters that delimit real JSON string boundaries,
    not escaped \\" ones inside a string.

    A quote is escaped only when it's preceded by an ODD number of
    consecutive backslashes (an unpaired one that escapes the quote) - an
    EVEN run is itself a complete sequence of escaped literal backslashes
    (\\\\, \\\\\\\\, ...) that leaves the quote after it unescaped. A naive
    "is the single preceding character a backslash" check gets this wrong
    for any even run of 2+ (e.g. a string value ending in a literal
    backslash, encoded as \\\\ right before the closing quote) - it
    miscounts that quote as escaped and skips it, throwing off the parity
    this function's caller relies on to tell whether a string is still
    open.
    """
    count = 0
    backslash_run = 0
    for ch in text:
        if ch == "\\":
            backslash_run += 1
            continue
        if ch == '"' and backslash_run % 2 == 0:
            count += 1
        backslash_run = 0
    return count


def _repair_truncated_json_string(content: str) -> str | None:
    """Attempts to recover a response cut off mid-string, specifically the
    pattern observed live with qwen3:30b: it finishes writing a complete,
    coherent CoverLetter body (ending naturally, e.g. "Sincerely, <name>")
    and then keeps the JSON string open for hundreds to thousands more
    characters of pure "\\n" padding before generation is cut off with no
    closing quote/brace at all - 10 failed applications in this project's
    own audit log from this exact shape alone. Not a context-window or
    output-length-limit issue (reproduced identically regardless of prompt
    length or num_predict/num_ctx), so nothing in the request options can
    tune it away - see CoverLetter.body's max_length for the complementary
    fix that bounds how much padding can accumulate before Ollama's own
    JSON-schema-constrained decoding is forced to close the string, which
    doesn't by itself cover generations that stall before ever reaching
    that cap.

    Deliberately narrow, to only ever trim clearly-inert trailing padding
    off content that already looks finished - never fabricates or alters
    real content. Returns the repaired JSON text if all of these hold,
    else None (the caller then falls through to a full retry as before):
    - the tail matches the padding pattern above (3+ consecutive "\\n"
      escapes) at the very end of the response,
    - what remains after trimming that ends inside an open string (an odd
      number of unescaped quote characters), and
    - exactly one top-level object is still open (true for every schema
      this project uses - see models/schemas.py, all flat/single-level).
    Anything else - cut off mid-word with no trailing padding, more than
    one unclosed brace, an escape sequence split mid-pair - is structurally
    different from the one pattern this was built for, and is left to the
    normal retry path rather than guessed at.
    """
    match = _TRAILING_BLANK_LINE_PADDING.search(content)
    if not match:
        return None
    trimmed = content[: match.start()]
    if not trimmed or trimmed.endswith("\\"):
        return None
    if _count_unescaped_quotes(trimmed) % 2 == 0:
        return None
    open_braces = trimmed.count("{") - trimmed.count("}")
    if open_braces != 1:
        return None
    return trimmed + '"}'


class _MalformedStream(ValueError):
    """A 200 response whose chunks don't have Ollama's chat shape - retried
    like any other bad completion."""


def _read_chat_stream(
    resp: httpx.Response, *, deadline: float | None = None, metrics: dict | None = None
) -> tuple[str, str | None]:
    """Accumulates a streamed /api/chat response: (content, error).

    Real failure this exists for (2026-09-30, Ollama 0.34): when qwen3:30b
    falls into the trailing-padding loop docs/qwen_notes.md §2 describes,
    Ollama now aborts the generation itself - "prediction aborted, token
    repeat limit reached" - and a non-streamed request gets only that HTTP
    500, with none of the content. 7 of 25 postings in one live run failed
    this way, every retry identically. Streamed, the complete letter
    arrives first, followed by the padding and then a final {"error": ...}
    chunk, so _repair_truncated_json_string() gets to trim the padding and
    close the JSON exactly as it was built to (confirmed live on two
    postings that failed every time non-streamed).

    Every chunk must carry either message.content or an error; anything
    else raises _MalformedStream (an API change, or a garbled body).

    Stops reading at `deadline` (time.monotonic()) and reports it as the
    error, keeping what arrived so far - the same repair then gets its
    chance, exactly as for an Ollama-side abort. Leaving the `with` block
    closes the connection, which makes Ollama stop generating.

    If `metrics` is given, it's filled from Ollama's final "done" chunk
    (prompt_eval_count, eval_count, load_duration in ns, ...) - the numbers
    behind the per-cycle performance line.
    """
    parts: list[str] = []
    error: str | None = None
    for line in resp.iter_lines():
        if deadline is not None and time.monotonic() > deadline:
            error = f"generation exceeded {MAX_GENERATION_SECONDS:.0f}s and was stopped"
            break
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
            if "error" in chunk:
                error = str(chunk["error"])
                continue
            parts.append(chunk["message"]["content"])
            if metrics is not None and chunk.get("done"):
                metrics.update({k: v for k, v in chunk.items() if k.endswith(("_count", "_duration"))})
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise _MalformedStream(f"Unexpected Ollama response chunk: {line[:200]!r}") from e
    return "".join(parts), error


class OllamaProviderError(RuntimeError):
    """Raised when the local Ollama server is unreachable or returns bad output."""


class OllamaProvider(LLMProvider):
    """Generic structured-output client for any model pulled into Ollama.

    Works for DeepSeek, Llama, GLM, Qwen, Mistral, or any other model the user
    runs `ollama pull <model>` for - the model name is just config, there is no
    per-model code here.
    """

    def __init__(self, model: str, base_url: str):
        self._model = model
        self._base_url = base_url.rstrip("/")
        # Performance metrics for every request this provider makes - see
        # GenerationStats and cli.py's per-cycle summary line.
        self.stats = GenerationStats()

    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "format": schema.model_json_schema(),
            # Streamed so that a generation Ollama aborts midway still
            # yields the text produced before the abort - see
            # _read_chat_stream().
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": MAX_OUTPUT_TOKENS,
                "num_ctx": _context_window(system, prompt),
            },
            # We only ever want the structured answer, never a reasoning
            # trace - on a thinking model (e.g. qwen3) this skips the hidden
            # <think> pass entirely, which is most of the latency. Ollama
            # silently ignores this on models that don't support thinking.
            "think": False,
        }

        last_error: Exception | None = None
        for _ in range(MAX_GENERATION_ATTEMPTS):
            try:
                with httpx.stream(
                    "POST", f"{self._base_url}/api/chat", json=payload, timeout=DEFAULT_TIMEOUT
                ) as resp:
                    if resp.status_code == 404:
                        raise OllamaProviderError(
                            f"Model '{self._model}' is not pulled. Run `ollama pull {self._model}`."
                        )
                    if resp.status_code != 200:
                        resp.read()
                        last_error = OllamaProviderError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                        continue
                    started = time.monotonic()
                    metrics: dict = {}
                    try:
                        content, stream_error = _read_chat_stream(
                            resp, deadline=started + MAX_GENERATION_SECONDS, metrics=metrics
                        )
                    finally:
                        self.stats.record(
                            seconds=time.monotonic() - started,
                            prompt_tokens=int(metrics.get("prompt_eval_count") or 0),
                            output_tokens=int(metrics.get("eval_count") or 0),
                            load_seconds=(metrics.get("load_duration") or 0) / 1e9,
                        )
            except httpx.ConnectError as e:
                raise OllamaProviderError(
                    f"Could not reach Ollama at {self._base_url}. Is it running? "
                    "Try `ollama serve` in another terminal."
                ) from e
            except httpx.TimeoutException as e:
                last_error = e
                continue
            except _MalformedStream as e:
                last_error = e
                continue

            try:
                return schema.model_validate_json(content)
            except (ValidationError, ValueError) as e:
                repaired = _repair_truncated_json_string(content)
                if repaired is not None:
                    try:
                        return schema.model_validate_json(repaired)
                    except (ValidationError, ValueError):
                        pass
                last_error = OllamaProviderError(f"Ollama aborted generation: {stream_error}") if stream_error else e
                continue

        raise OllamaProviderError(
            f"Model '{self._model}' did not return schema-valid JSON after "
            f"{MAX_GENERATION_ATTEMPTS} attempts: {last_error}"
        ) from last_error


def quit_ollama() -> bool:
    """Best-effort shutdown of the locally running Ollama server/app - see
    Settings.quit_ollama_when_done, which cli.py checks before calling this
    once a run stops drawing on Ollama for the day. There is no HTTP
    endpoint to ask Ollama to shut down (only to unload one loaded model,
    which isn't the same as quitting the app/server), so this reaches for
    the OS process directly instead.

    On macOS this first asks the menu-bar app to quit via AppleScript (the
    common install path - quitting the app also stops the server process it
    launched), then falls back to killing a bare `ollama serve` process by
    name in case that's how it was actually started; either, both, or
    neither may apply on a given machine, so every command is attempted and
    a failure in one doesn't skip the rest. Never raises - this is a
    courtesy cleanup after a run that has already finished its real work,
    not something that should fail the run itself, and "nothing to quit"
    (Ollama already stopped, or was never running) is a normal, harmless
    outcome each command reports as a plain nonzero exit rather than an
    exception.
    """
    system = platform.system()
    if system == "Darwin":
        commands = [["osascript", "-e", 'quit app "Ollama"'], ["pkill", "-x", "ollama"]]
    elif system == "Windows":
        commands = [["taskkill", "/IM", "ollama app.exe", "/F"], ["taskkill", "/IM", "ollama.exe", "/F"]]
    else:
        commands = [["pkill", "-x", "ollama"]]

    quit_ok = False
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, timeout=10, check=False)
            quit_ok = quit_ok or result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            continue
    return quit_ok

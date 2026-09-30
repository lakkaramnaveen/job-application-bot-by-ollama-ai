"""Guards for the small JSON data files (FAQ_PATH, the company blacklist,
answer_gaps.json) that are read with a deliberate graceful fallback - a
corrupt file loads as empty rather than crashing every command, and `job-bot
doctor` is what reports it. The catch: each store's save path rebuilds the
whole file from what it loaded, so the first write after corruption
(caching one FAQ answer, blacklisting one company, recording one gap)
silently replaced the unreadable file - and everything still recoverable in
it - with just that one entry. assert_safe_to_overwrite() closes that: a
file that exists but can't be read is left untouched and the write fails
loudly instead.
"""

import json
from pathlib import Path


class CorruptDataFile(RuntimeError):
    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(
            f"{path} exists but could not be read ({reason}) - refusing to overwrite it and lose "
            "its contents. Fix it by hand or move it aside, then retry (`job-bot doctor` checks it)."
        )


def assert_safe_to_overwrite(path: Path, expected_type: type) -> None:
    """Raises CorruptDataFile if `path` exists but isn't valid UTF-8 JSON of
    `expected_type` (dict or list - whatever that store's own loader
    accepts). A missing file is always safe to create.
    """
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        # OSError too: a permission problem or a directory where the file
        # should be is just as unreadable, and just as unsafe to replace.
        raise CorruptDataFile(path, str(e)) from e
    if not isinstance(data, expected_type):
        raise CorruptDataFile(path, f"expected a JSON {expected_type.__name__}, found {type(data).__name__}")

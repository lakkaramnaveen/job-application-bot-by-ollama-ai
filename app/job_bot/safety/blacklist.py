import json
from pathlib import Path
from typing import Any

from job_bot.data_files import assert_safe_to_overwrite
from job_bot.text_utils import normalize_company_name


class CompanyBlacklist:
    """Companies to never apply to. Defaults to an empty list plus whatever the
    user configures (e.g. past employers) in company_blacklist.json.

    Keyed internally by normalize_company_name() so is_blocked()/add()/
    remove() agree with gmail_sync.py on which spellings mean the same
    company, but the *display* casing the user actually typed is kept
    alongside each key (not the normalized/casefolded form) - list_companies()
    (and `job-bot blacklist list`) previously showed every company permanently
    lowercased ("acme corp"), which is what got stored as the dict key,
    because that key was also the only copy of the name kept.

    Each entry also carries an optional `reason` (e.g. "no H1B
    sponsorship", "bad interview experience") - `job-bot blacklist add
    --reason` sets it, `job-bot blacklist list` shows it. On disk, an entry
    with no reason is still stored as a plain string, exactly the format
    this file already used before reasons existed; only an entry that
    actually has one becomes a `{"name": ..., "reason": ...}` object - so a
    blacklist where nobody has bothered with a reason round-trips through
    _save() byte-for-byte the same shape as before this feature, and
    company_blacklist.json stays readable by any external tool that only
    ever expected a plain list of strings.
    """

    def __init__(self, blacklist_path: Path):
        self._path = blacklist_path
        self._companies = self._load()  # normalized name -> {"name": ..., "reason": ...}

    def _load(self) -> dict[str, dict[str, str]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            # UnicodeDecodeError alongside JSONDecodeError - a
            # company_blacklist.json saved with a non-UTF-8 encoding is
            # corruption exactly the same way invalid JSON already is, and
            # gets the same graceful fallback here rather than crashing
            # every command that touches the blacklist. cli.py's own
            # _blacklist_check() doctor check is what actually surfaces
            # this to the user, instead of it silently looking like
            # "nothing blacklisted".
            return {}
        if not isinstance(data, list):
            return {}
        entries: dict[str, dict[str, str]] = {}
        for item in data:
            if isinstance(item, str):
                name, reason = item.strip(), ""
            elif isinstance(item, dict):
                # The reason-carrying shape _save() writes, and also what a
                # hand-edited or externally-authored file might use -
                # tolerate a missing/non-string "name"/"reason" the same
                # way a plain non-string list item is already skipped below,
                # rather than crashing on one malformed entry.
                name = str(item.get("name", "")).strip()
                reason = str(item.get("reason", "")).strip()
            else:
                continue
            if not name:
                continue
            entries[self._normalize(name)] = {"name": name, "reason": reason}
        return entries

    @staticmethod
    def _normalize(name: str) -> str:
        return normalize_company_name(name)

    def is_blocked(self, company_name: str) -> bool:
        return self._normalize(company_name) in self._companies

    def get_entry(self, company_name: str) -> dict[str, str] | None:
        """The {"name", "reason"} entry for `company_name` (matched the
        same normalized way is_blocked() already does), or None if it
        isn't blocked - `job-bot blacklist check` needs this to show not
        just whether a company is blocked but why, without exposing the
        whole list the way list_entries() does. Returns a fresh dict, not
        a reference into internal state, so a caller can't accidentally
        mutate this entry out from under _save()'s next call.
        """
        entry = self._companies.get(self._normalize(company_name))
        return dict(entry) if entry is not None else None

    def add(self, company_name: str, *, reason: str = "") -> None:
        self._companies[self._normalize(company_name)] = {
            "name": company_name.strip(),
            "reason": reason.strip(),
        }
        self._save()

    def remove(self, company_name: str) -> bool:
        """Returns True if the company was on the list (and is now removed),
        False if it wasn't there to begin with.
        """
        normalized = self._normalize(company_name)
        if normalized not in self._companies:
            return False
        del self._companies[normalized]
        self._save()
        return True

    def list_companies(self) -> list[str]:
        return [entry["name"] for entry in self._sorted_entries()]

    def list_entries(self) -> list[dict[str, str]]:
        """Every blacklisted company as {"name": ..., "reason": ...}
        (reason is "" when none was given) - the counterpart to
        list_companies() above for a caller that needs the reason too:
        `job-bot blacklist list` and its `--format json`. list_companies()
        itself stays name-only and unchanged so every existing caller
        (is_blocked(), the dashboard's blacklist-list/remove flow) keeps
        working exactly as before this feature.
        """
        return [{"name": entry["name"], "reason": entry["reason"]} for entry in self._sorted_entries()]

    def _sorted_entries(self) -> list[dict[str, str]]:
        return sorted(self._companies.values(), key=lambda entry: entry["name"].casefold())

    def _save(self) -> None:
        assert_safe_to_overwrite(self._path, list)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data: list[Any] = [
            {"name": entry["name"], "reason": entry["reason"]} if entry["reason"] else entry["name"]
            for entry in self._sorted_entries()
        ]
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

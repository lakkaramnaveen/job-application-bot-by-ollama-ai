import json
from pathlib import Path

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
    """

    def __init__(self, blacklist_path: Path):
        self._path = blacklist_path
        self._companies = self._load()  # normalized name -> display name

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, list):
            return {}
        return {
            self._normalize(c): c.strip() for c in data if isinstance(c, str) and c.strip()
        }

    @staticmethod
    def _normalize(name: str) -> str:
        return normalize_company_name(name)

    def is_blocked(self, company_name: str) -> bool:
        return self._normalize(company_name) in self._companies

    def add(self, company_name: str) -> None:
        self._companies[self._normalize(company_name)] = company_name.strip()
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
        return sorted(self._companies.values(), key=str.casefold)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self.list_companies(), indent=2), encoding="utf-8")

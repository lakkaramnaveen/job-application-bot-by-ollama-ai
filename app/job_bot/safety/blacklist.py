import json
from pathlib import Path

from job_bot.text_utils import normalize_company_name


class CompanyBlacklist:
    """Companies to never apply to. Defaults to an empty list plus whatever the
    user configures (e.g. past employers) in company_blacklist.json.
    """

    def __init__(self, blacklist_path: Path):
        self._path = blacklist_path
        self._companies = self._load()

    def _load(self) -> set[str]:
        if not self._path.exists():
            return set()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return set()
        if not isinstance(data, list):
            return set()
        return {self._normalize(c) for c in data if isinstance(c, str)}

    @staticmethod
    def _normalize(name: str) -> str:
        return normalize_company_name(name)

    def is_blocked(self, company_name: str) -> bool:
        return self._normalize(company_name) in self._companies

    def add(self, company_name: str) -> None:
        self._companies.add(self._normalize(company_name))
        self._save()

    def remove(self, company_name: str) -> bool:
        """Returns True if the company was on the list (and is now removed),
        False if it wasn't there to begin with.
        """
        normalized = self._normalize(company_name)
        if normalized not in self._companies:
            return False
        self._companies.discard(normalized)
        self._save()
        return True

    def list_companies(self) -> list[str]:
        return sorted(self._companies)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(sorted(self._companies), indent=2), encoding="utf-8")

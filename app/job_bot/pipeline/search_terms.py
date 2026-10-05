"""Several job titles to search, one per cycle.

SEARCH_KEYWORDS (or --keywords) may list several titles, comma-separated -
"java full stack, mern stack, junior software engineer". Searching all of
them every cycle would multiply LinkedIn requests by the number of titles,
and request volume is what got searches rate limited (HTTP 429, 2026-10-03
to 05). So each cycle searches the next title in turn: the request rate
stays what it was with one title, and every title is covered within a few
cycles.
"""

from __future__ import annotations


def split_search_terms(keywords: str) -> list[str]:
    """The titles in a comma-separated SEARCH_KEYWORDS, trimmed, in order,
    without blanks or (case-insensitive) repeats. A single title is a list
    of one."""
    terms: list[str] = []
    seen: set[str] = set()
    for term in keywords.split(","):
        term = " ".join(term.split())
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)
    return terms


class SearchRotation:
    """Hands out one title per cycle, round robin."""

    def __init__(self, keywords: str):
        self.terms = split_search_terms(keywords) or [keywords]
        self._next = 0

    def next_term(self) -> str:
        term = self.terms[self._next % len(self.terms)]
        self._next += 1
        return term

    def position(self, term: str) -> str:
        """ " (3 of 18)" for a rotation, "" for a single title."""
        if len(self.terms) == 1:
            return ""
        return f" ({self.terms.index(term) + 1} of {len(self.terms)})"

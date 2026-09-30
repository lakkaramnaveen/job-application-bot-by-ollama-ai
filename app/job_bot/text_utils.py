from datetime import datetime


def normalize_company_name(name: str) -> str:
    """Canonical form for comparing company names across the codebase -
    trimmed, casefolded, and with internal whitespace collapsed to single
    spaces. Used by both safety/blacklist.py (is a company blocked?) and
    integrations/gmail_sync.py (does an email's company guess match a
    tracked job?) so the two can never silently disagree on whether two
    differently-whitespaced spellings of the same company are "the same"
    company.
    """
    return " ".join(name.strip().casefold().split())


def is_echoed_question(question: str, answer: str) -> bool:
    """True if `answer` is just `question` repeated back (ignoring case,
    whitespace, and a trailing "?"/":"/"*") - not an answer at all.

    Confirmed in real data: for bare form-field labels with no question in
    them ("Phone country code", "Year of From", "LinkedIn"), the model
    sometimes returns the label itself as the answer. Such an answer never
    matches a select/radio option, and in a free-text field it types the
    label into the form. Worse, it was cached to FAQ_PATH, and cli.py's
    exact-match FAQ shortcut then replayed it on every later posting that
    asked the same thing - docs/qwen_notes.md §3's "the cache doesn't
    validate the shape of what it caches" pattern.
    """

    def normalize(text: str) -> str:
        return " ".join(text.strip().rstrip("?:*").casefold().split())

    normalized_answer = normalize(answer)
    return bool(normalized_answer) and normalized_answer == normalize(question)


# LinkedIn Easy Apply's own work-history block labels, confirmed verbatim in
# data/failed_applications.log (7 different companies' forms) - one set per
# position, each with its own dates. Exact labels only, not a pattern: a
# "From"/"To" question elsewhere can be perfectly answerable.
_PER_POSITION_FIELD_LABELS = frozenset({"month of from", "year of from", "month of to", "year of to"})


def is_per_position_field(label: str) -> bool:
    """True for a work-history date field whose right answer differs per
    position. The LLM only ever sees the bare label, so it can't know which
    position is meant, and a single cached FAQ answer would put the same date
    on every position of every future application - wrong data, submitted
    under the user's name. Such a field is left unanswered instead.
    """
    return " ".join(label.strip().rstrip("?:*").casefold().split()) in _PER_POSITION_FIELD_LABELS


def format_local_timestamp(value: str) -> str:
    """A stored UTC ISO timestamp (applied_at and friends) as "YYYY-MM-DD
    HH:MM" in the user's local time - the same local-time convention
    Tracker.applications_by_week() already uses, so an application sent on a
    Sunday evening in a UTC-negative zone shows as that Sunday everywhere,
    not as Monday. A value that doesn't parse is returned unchanged rather
    than raising: this is display-only, and a hand-edited row shouldn't take
    down a whole report or dashboard page.
    """
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return value

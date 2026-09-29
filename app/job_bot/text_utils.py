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

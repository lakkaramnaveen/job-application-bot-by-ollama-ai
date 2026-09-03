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

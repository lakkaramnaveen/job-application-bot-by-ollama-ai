"""Guards against accidentally weakening the personal-data .gitignore rules
(resume, API keys, browser session, tracker DB all live under app/data/ and
app/.env - see SECURITY.md). Adapted from the same idea in ai-job-search's
tools/security_guards.py: make an accidental widening loud in review rather
than silently shipping a way to commit someone's personal data.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_RULES = ["app/.env", "app/data/"]


def gitignore_lines() -> list[str]:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()]


def test_gitignore_file_exists():
    assert (REPO_ROOT / ".gitignore").exists()


def test_personal_data_rules_are_present():
    lines = gitignore_lines()
    for rule in REQUIRED_RULES:
        assert rule in lines, f".gitignore is missing the required rule: {rule!r}"


def test_no_negation_reintroduces_personal_data_paths():
    """A `!pattern` line re-includes a path an earlier rule excluded. Catch
    any negation that would silently undo one of the required rules above.
    """
    negations = [line for line in gitignore_lines() if line.startswith("!")]
    for negation in negations:
        pattern = negation.lstrip("!")
        for rule in REQUIRED_RULES:
            assert not pattern.startswith(rule.rstrip("/")), (
                f".gitignore negation {negation!r} would re-include the "
                f"personal-data path protected by {rule!r}"
            )

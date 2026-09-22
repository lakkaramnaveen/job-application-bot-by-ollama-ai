from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

# Label/placeholder substrings that mean "never fill this field, no matter
# what answer_question might return for it" - these ask for exactly the
# categories this project will never enter into a form, on either adapter.
# Shared here (rather than duplicated per-adapter, the way e.g.
# _numeric_value()/_best_match_index() are) because this is a safety
# boundary, not an independent per-site quirk - the two copies silently
# drifting apart is exactly the failure mode to avoid for a list like this
# one. Originally external_apply_adapter.py-only (arbitrary third-party
# employer sites being the obvious risk), until it became clear LinkedIn
# Easy Apply forms carry the identical risk: employers attach their own
# custom screening questions to Easy Apply, free-text and no less able to
# ask for a Social Security Number than a field on an external site would.
# A required field matching this list is always left blank, which the
# caller's required-field check turns into a clear, specific error rather
# than either silently skipping it or guessing.
SENSITIVE_FIELD_MARKERS = (
    "social security",
    "ssn",
    "passport number",
    "driver's license number",
    "driver license number",
    "national id",
    "bank account",
    "routing number",
    "credit card",
    "debit card",
    "cvv",
    "tax id",
    "ein",
)

# A resume upload step commonly has more than one file input (resume, cover
# letter, portfolio, ...) - these decide which one is the actual resume
# field without ever guessing. Shared here for the exact same reason
# SENSITIVE_FIELD_MARKERS is: a real drift was found between the two
# adapters' independent copies before this fix -
# external_apply_adapter.py's _NON_RESUME_FILE_LABEL_MARKERS was missing
# "certificate"/"license", both present in linkedin_adapter.py's list from
# the start. That gap meant a single file field on an external site
# labeled e.g. "Upload your teaching certificate" was treated as the
# (unlabeled-for-anything-else) resume field by default and had the
# user's resume uploaded into it - the exact mistake this allowlist/
# denylist pair exists to prevent, just silently un-prevented on one of
# the two adapters.
RESUME_FILE_LABEL_MARKERS = ("resume", "cv")
NON_RESUME_FILE_LABEL_MARKERS = (
    "cover letter",
    "portfolio",
    "writing sample",
    "transcript",
    "certificate",
    "license",
)


@dataclass
class JobPosting:
    job_id: str
    title: str
    company: str
    url: str
    description: str
    # True (the default) for a posting the board itself can submit in-page
    # (LinkedIn Easy Apply). False marks a posting whose application is
    # handled entirely off the board, on the employer's own site - see
    # LinkedInAdapter.open_external_application() and
    # browser/external_apply_adapter.py. Only ever False when search() was
    # called with include_external=True; existing callers/tests that never
    # pass that flag see every posting as easy_apply=True, unchanged.
    easy_apply: bool = True


class JobBoardAdapter(ABC):
    """Interface a job board integration implements. v1 ships LinkedInAdapter
    only; new boards (Indeed, ZipRecruiter, ...) plug in by implementing this
    same interface without changing anything else in the app.
    """

    @abstractmethod
    def search(
        self,
        keywords: str,
        location: str,
        max_results: int = 25,
        experience_levels: list[str] | None = None,
        include_external: bool = False,
    ) -> list[JobPosting]:
        """Return up to max_results postings, paging through search results
        and skipping postings already marked Applied. By default (
        include_external=False, the long-standing behavior) every posting
        returned is Easy-Apply-eligible (easy_apply=True). With
        include_external=True, postings whose application is handled off
        the board entirely (on the employer's own site) are included too,
        marked easy_apply=False - callers that don't know how to handle
        those (see JobPosting.easy_apply) should leave this at the default.

        experience_levels, when given, restricts results to those seniority
        levels at the search level (values are board-specific - see the
        implementing adapter). None means no restriction: every level the
        board returns for these keywords/location.
        """
        raise NotImplementedError

    @abstractmethod
    def fill_and_submit(
        self,
        posting: JobPosting,
        *,
        answer_question: Callable[[str], str],
        resume_path: str | None,
        cover_letter_text: str | None,
        dry_run: bool,
    ) -> bool:
        """Open the application form, fill every field, and submit unless
        dry_run is True (in which case it stops right before the submit
        click and returns False). `answer_question` is called for each
        free-text question the adapter can't fill deterministically.
        Returns True if the application was actually submitted.
        """
        raise NotImplementedError

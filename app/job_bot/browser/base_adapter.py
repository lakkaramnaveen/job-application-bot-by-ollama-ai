from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


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

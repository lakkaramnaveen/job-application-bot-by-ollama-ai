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


class JobBoardAdapter(ABC):
    """Interface a job board integration implements. v1 ships LinkedInAdapter
    only; new boards (Indeed, ZipRecruiter, ...) plug in by implementing this
    same interface without changing anything else in the app.
    """

    @abstractmethod
    def search(self, keywords: str, location: str, max_results: int = 25) -> list[JobPosting]:
        """Return up to max_results Easy-Apply-eligible postings, paging
        through search results and skipping postings already marked Applied.
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

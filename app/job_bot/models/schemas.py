from typing import Literal

from pydantic import BaseModel, Field

# "fail" means the posting explicitly requires something the resume gives no
# indication the candidate holds (citizenship, permanent residency, an
# existing security clearance) - a categorical exclusion, not a fit question.
# "flag" means the posting is silent or ambiguous on work authorization -
# not disqualifying, but worth a human's attention before applying.
EligibilityVerdict = Literal["pass", "fail", "flag"]


class JobMatchScore(BaseModel):
    """LLM's assessment of how well a candidate's resume fits a job posting.

    Eligibility is evaluated as a gate, separate from the fit dimensions:
    scorer.score_job_match() forces should_apply=False whenever eligibility
    is "fail", regardless of what the model itself set should_apply to - a
    categorical exclusion (e.g. "must be a US citizen") is not something a
    high fit score should be able to override.
    """

    eligibility: EligibilityVerdict = Field(
        description=(
            "'fail' if the posting explicitly requires citizenship, permanent "
            "residency, or an existing security clearance with no indication "
            "the candidate holds it. 'flag' if the posting is silent or "
            "ambiguous on work authorization. 'pass' otherwise."
        )
    )
    eligibility_note: str = Field(
        default="",
        description="The specific posting wording driving the eligibility verdict, or empty if none found",
    )
    technical_fit: int = Field(ge=0, le=100, description="How well required/preferred skills match")
    experience_fit: int = Field(ge=0, le=100, description="How well work history matches what's sought")
    culture_fit: int = Field(
        ge=0, le=100, description="Likely culture/working-style fit, from posting tone/values"
    )
    score: int = Field(ge=0, le=100, description="Overall fit score, 0 (no match) to 100 (perfect match)")
    reasoning: str = Field(description="Brief explanation of the score")
    should_apply: bool = Field(description="Whether this job clears the bar to apply to")
    missing_qualifications: list[str] = Field(default_factory=list)


class TailoredResume(BaseModel):
    """A resume rewritten/reordered to emphasize fit for one specific job."""

    summary: str = Field(description="2-3 sentence professional summary tailored to this job")
    highlighted_skills: list[str] = Field(
        description="Skills from the resume most relevant to this job, ordered by relevance"
    )
    bullet_points: list[str] = Field(description="Tailored, ATS-friendly experience bullet points")


class CoverLetter(BaseModel):
    """A cover letter generated for one specific job application."""

    body: str = Field(description="Full cover letter body text, 3-4 short paragraphs")


class ApplicationAnswer(BaseModel):
    """An answer to one free-text or short-answer application question."""

    answer: str = Field(description="The answer to fill into the form field")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model's confidence this answer is correct/appropriate"
    )
    based_on_resume: bool = Field(
        description="Whether this answer is directly supported by resume/FAQ content"
    )


# What kind of job-application-related email this is, if any. "other" covers
# both "not job related" and "job related but doesn't fit another bucket" -
# gmail_sync only acts on the four specific categories below.
EmailCategory = Literal["interview_invite", "rejection", "offer", "application_confirmation", "other"]


class EmailClassification(BaseModel):
    """LLM's read of one email, for matching it back to a tracked application.

    company_guess/role_guess are free-text extractions from the email, not
    validated against the tracker - job_bot.integrations.gmail_sync does that
    matching itself and never updates a tracked job on a guess it can't
    confidently tie to exactly one row.
    """

    is_job_related: bool = Field(description="Whether this email concerns a specific job application")
    category: EmailCategory = Field(description="Best-fit category if is_job_related, else 'other'")
    company_guess: str = Field(default="", description="Company name this email appears to be from/about")
    role_guess: str = Field(default="", description="Job title this email appears to reference, if any")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in is_job_related and category")

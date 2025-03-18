"""
LLM integration utilities for structured data extraction and analysis.

This module provides functions for interacting with Anthropic's Claude API:
- Job post detection and validation
- CV-to-job matching with detailed scoring
- Structured information extraction with retry logic
- Error handling and input validation
"""

import json
import logging
import time
from typing import Any, TypeVar

import instructor
from pydantic import BaseModel

from _production import LLM_BASE_MODEL
from _production.config.config import ANTHROPIC_CLIENT
from _production.utils.exceptions import (
    LLMError,
    LLMInputError,
    LLMRateLimitError,
    LLMResponseError,
)
from _production.utils.prompts import (
    CLEAN_JOB_POST_PROMPT,
    CV_MATCHING_PROMPT,
    JOB_POST_DETECTION_PROMPT,
    JOB_POST_PARSING_PROMPT,
    SINGLE_JOB_POST_DETECTION_PROMPT,
)

T = TypeVar("T", bound=BaseModel)

MAX_TEXT_LENGTH = 64000

# Initialize instructor-wrapped client
ANTHROPIC_CLIENT_STRUCTURED = instructor.from_anthropic(ANTHROPIC_CLIENT)


class CleanJobPost(BaseModel):
    """
    Structured representation of a cleaned job post with standardized fields.

    Contains normalized values for job details including title, seniority, location,
    and other key attributes needed for matching and analysis.
    """

    job_title: str | None
    seniority_level: str | None
    location: str | None
    remote_status: str | None
    relocation_support: bool | None
    visa_sponsorship: bool | None
    salary_range: str | None
    company_name: str | None
    description: str | None


def validate_text_input(
    text: str, field_name: str, max_length: int = MAX_TEXT_LENGTH
) -> None:
    """Validate text input.

    Args:
        text: Input text to validate
        field_name: Name of the field for error messages
        max_length: Maximum allowed length

    Raises:
        LLMInputError: If validation fails
    """
    if not text or not text.strip():
        raise LLMInputError(f"{field_name} cannot be empty")
    if len(text) > max_length:
        raise LLMInputError(
            f"{field_name} exceeds maximum length of {max_length} characters"
        )


def _make_llm_call(
    messages: list[dict[str, str]],
    response_format: type[T],
    max_retries: int = 3,
    sleep_time: int = 10,
) -> T | None:
    """Make Anthropic API calls with retry logic."""
    system_message = next(
        (msg["content"] for msg in messages if msg["role"] == "system"), ""
    )
    user_message = next(
        (msg["content"] for msg in messages if msg["role"] == "user"), ""
    )

    for attempt in range(max_retries):
        try:
            response = ANTHROPIC_CLIENT_STRUCTURED.messages.create(
                model=LLM_BASE_MODEL,
                system=system_message,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=1024,
                temperature=0.0,
                response_model=response_format,
            )

            if attempt > 0:  # Only log if we had to retry
                logging.info(f"Successful LLM call after {attempt + 1} attempts")
            return response

        except Exception as error:
            if "rate_limit" in str(error).lower():
                logging.warning(
                    f"Rate limit hit. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
                if attempt == max_retries - 1:
                    raise LLMRateLimitError(
                        "Rate limit exceeded after max retries"
                    ) from error
            else:
                logging.error(f"LLM call failed: {error!s}")
                raise LLMError(f"LLM call failed: {error!s}") from error

    return None


def job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains any job postings."""

    class JobPost(BaseModel):
        is_job_description: bool

    messages = [
        {
            "role": "system",
            "content": JOB_POST_DETECTION_PROMPT,
        },
        {
            "role": "user",
            "content": f"Does this text contain any job postings?\n\n{post}",
        },
    ]

    result = _make_llm_call(messages, JobPost, max_retries, sleep_time)
    return result.is_job_description if result else None


def single_job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains exactly one job posting."""

    class SingleJobPost(BaseModel):
        is_single_post: bool

    messages = [
        {
            "role": "system",
            "content": SINGLE_JOB_POST_DETECTION_PROMPT,
        },
        {
            "role": "user",
            "content": f"Does this text contain exactly one job posting?\n\n{post}",
        },
    ]

    result = _make_llm_call(messages, SingleJobPost, max_retries, sleep_time)
    return result.is_single_post if result else None


def match_cv_with_job(
    cv_text: str, post: str, max_retries: int = 3, sleep_time: int = 10
) -> float | None:
    """Evaluates match between CV and job post using a comprehensive single-call approach."""
    try:
        validate_text_input(cv_text, "CV text")
        validate_text_input(post, "Job post")

        # Define a more structured response model that captures all three areas
        class CVMatchDetailed(BaseModel):
            experience_score: float
            skills_score: float
            soft_skills_score: float
            final_score: float

        messages = [
            {
                "role": "system",
                "content": CV_MATCHING_PROMPT,
            },
            {
                "role": "user",
                "content": f"""Analyze this CV and job post for compatibility.

                First, evaluate these three key areas separately:
                1. Experience Match (years of experience, domain knowledge, project scale)
                2. Skills Match (technical skills, education, tools, certifications)
                3. Soft Skills Match (communication, teamwork, problem-solving, cultural fit)

                Then calculate the final weighted score using:
                - Skills Match (45%)
                - Experience Match (40%)
                - Soft Skills Match (15%)

                CV:
                {cv_text}

                Job Post:
                {post}

                Provide your response in this exact JSON format:
                {{
                    "experience_score": 85,
                    "skills_score": 90,
                    "soft_skills_score": 88,
                    "final_score": 87
                }}""",
            },
        ]

        result = _make_llm_call(messages, CVMatchDetailed, max_retries, sleep_time)
        if result:
            # Verify the final score calculation is correct
            calculated_score = int(
                (result.skills_score * 0.45)
                + (result.experience_score * 0.40)
                + (result.soft_skills_score * 0.15)
            )

            # If there's a significant discrepancy, use our calculation
            if abs(calculated_score - result.final_score) > 2:
                logging.warning(
                    f"Correcting LLM score calculation: {result.final_score} -> {calculated_score}"
                )
                return calculated_score

            return result.final_score
        return None
    except LLMInputError as error:
        logging.error(f"Input validation failed: {error!s}")
        return None
    except (LLMError, LLMResponseError, LLMRateLimitError) as error:
        logging.error(f"LLM service error during CV matching: {error!s}")
        return None
    except (ValueError, TypeError, KeyError) as error:
        logging.error(f"Data processing error during CV matching: {error!s}")
        return None
    except Exception as error:
        logging.error(f"Unexpected error during CV matching: {error!s}", exc_info=True)
        return None


def job_post_parsing(
    post: str, max_retries: int = 3, sleep_time: int = 10
) -> dict[str, Any] | None:
    """Parse job posting into structured format."""
    try:
        validate_text_input(post, "Job post")

        class JobPostStructure(BaseModel):
            job_title: str
            seniority_level: str
            location: str
            remote_status: str
            relocation_support: bool | None
            visa_sponsorship: bool | None
            salary_range: str | None
            company_name: str
            description: str

        messages = [
            {
                "role": "system",
                "content": JOB_POST_PARSING_PROMPT,
            },
            {
                "role": "user",
                "content": f"Parse this job posting into a structured format:\n\n{post}",
            },
        ]

        result = _make_llm_call(messages, JobPostStructure, max_retries, sleep_time)
        if not result:
            return None

        # Get initial parsed response
        response = result.model_dump()

        # Get cleaned response
        cleaned_response = clean_job_post_values(response)

        cleaned_response_filtered = {
            key: value
            for key, value in cleaned_response.items()
            if value and isinstance(value, str)
        }
        if (
            not cleaned_response_filtered
            or "job_title" not in cleaned_response_filtered
        ):
            return None

        # Single strip operation only on string values
        return cleaned_response_filtered
    except LLMInputError as error:
        logging.error(f"Input validation failed: {error!s}")
        return None


def clean_job_post_values(response: dict[str, Any]) -> dict[str, Any]:
    """Clean and standardize job post values.

    Args:
        response: Raw job post data dictionary

    Returns:
        Dictionary with cleaned values

    Raises:
        LLMResponseError: If cleaning fails
    """
    messages = [
        {
            "role": "system",
            "content": CLEAN_JOB_POST_PROMPT,
        },
        {
            "role": "user",
            "content": f"Please clean and standardize this job posting data: {json.dumps(response)}",
        },
    ]

    try:
        result = _make_llm_call(
            messages=messages, response_format=CleanJobPost, max_retries=3, sleep_time=1
        )
        if not result:
            raise LLMResponseError("Failed to get valid response from LLM")
        return result.model_dump()
    except Exception as error:
        logging.error(f"Failed to clean job post values: {error!s}")
        return CleanJobPost(
            job_title=None,
            seniority_level=None,
            location=None,
            remote_status=None,
            relocation_support=None,
            visa_sponsorship=None,
            salary_range=None,
            company_name=None,
            description=None,
        ).model_dump()


if __name__ == "__main__":
    # Test data
    sample_cv = """
    John Doe
    Senior Software Engineer

    Experience:
    - 7 years Python development
    - Lead backend architect for high-scale systems
    - Team leadership experience managing 5-person teams
    - Microservices architecture design and implementation

    Technical Skills:
    - Languages: Python, Go, TypeScript
    - Frameworks: Django, FastAPI, React
    - Databases: PostgreSQL, MongoDB, Redis
    - Cloud: AWS (ECS, Lambda, S3), Docker, Kubernetes
    - Tools: Git, CI/CD, Prometheus, Grafana

    Certifications:
    - AWS Solutions Architect Professional
    - Kubernetes CKA
    """

    sample_job_post = """
    Senior Backend Engineer

    Company: TechCorp Solutions
    Location: Berlin, Germany (Hybrid)
    Department: Engineering

    About the Role:
    We're seeking an experienced Backend Engineer to join our platform team.

    Required Skills:
    - 5+ years Python development experience
    - Strong expertise in Django and FastAPI
    - Advanced PostgreSQL and Redis knowledge
    - Experience with Docker and Kubernetes
    - Microservices architecture design
    - AWS cloud services (ECS, Lambda)

    Nice to Have:
    - Go programming experience
    - MongoDB experience
    - Prometheus/Grafana monitoring
    - TypeScript/React for full-stack contributions
    - AWS certifications

    Benefits:
    - Competitive salary: €85K-120K
    - Flexible hybrid work arrangement
    - Professional development budget
    """

    # Test all functions
    print("\n=== Testing job_post_detection ===")
    result = job_post_detection(sample_job_post)
    print(f"Is job post: {result}")

    print("\n=== Testing single_job_post_detection ===")
    result = single_job_post_detection(sample_job_post)
    print(f"Is single job post: {result}")

    print("\n=== Testing match_cv_with_job ===")
    match_score = match_cv_with_job(sample_cv, sample_job_post)
    print(f"CV match score: {match_score}")

    print("\n=== Testing job_post_parsing ===")
    parsed_job = job_post_parsing(sample_job_post)
    print("Parsed job post:")
    if parsed_job:
        for key, value in parsed_job.items():
            print(f"{key}: {value}")
    else:
        print("Failed to parse job post")

    print("\n=== Testing clean_job_post_values ===")
    sample_response = {
        "job_title": "Senior Backend Engineer",
        "seniority_level": "Senior",
        "location": "Berlin, Germany (Hybrid)",
        "salary_range": "€85K-120K",
        "company_name": "TechCorp Solutions",
        "description": "We're seeking an experienced Backend Engineer to join our platform team.",
        "required_skills": [
            "Python",
            "Django",
            "FastAPI",
            "PostgreSQL",
            "Redis",
            "Docker",
            "Kubernetes",
            "AWS",
            "Microservices",
        ],
        "preferred_skills": [
            "Go",
            "MongoDB",
            "Prometheus",
            "Grafana",
            "TypeScript",
            "React",
        ],
    }
    cleaned = clean_job_post_values(sample_response)
    print("Cleaned job post:")
    for key, value in cleaned.items():
        print(f"{key}: {value}")

"""
LLM integration utilities for structured data extraction and analysis.

This module provides functions for interacting with LLM models:
- Job post detection and validation
- CV-to-job matching with detailed scoring
- Structured information extraction with retry logic
- Error handling and input validation

Supports custom local model server based on configuration.
"""

import json
import logging
import time
from typing import Any, Literal, TypeVar

import requests
from pydantic import BaseModel, Field

from _production import LLM_INSTANCE_URL
from _production.utils.custom_model import get_custom_model_client
from _production.utils.exceptions import (
    LLMError,
    LLMInputError,
    LLMRateLimitError,
    LLMResponseError,
)
from _production.utils.instructor_wrapper import from_custom_model
from _production.utils.prompts import (
    CLEAN_JOB_POST_PROMPT,
    EXPERIENCE_MATCHING_PROMPT,
    JOB_POST_DETECTION_PROMPT,
    SINGLE_JOB_POST_DETECTION_PROMPT,
    SKILLS_MATCHING_PROMPT,
    SOFT_SKILLS_MATCHING_PROMPT,
)

T = TypeVar("T", bound=BaseModel)

MAX_TEXT_LENGTH = 64000

# Initialize LLM clients
logging.info("Using custom model client for LLM operations")
CUSTOM_MODEL_CLIENT = get_custom_model_client()
LLM_STRUCTURED_CLIENT = from_custom_model()


class CleanJobPost(BaseModel):
    """
    Structured representation of a cleaned job post with standardized fields.

    Contains normalized values for job details including title, seniority, location,
    and other key attributes needed for matching and analysis.
    """

    job_title: str | None = Field(
        default=None,
        description="Standardized job title",
        examples=["Senior Software Engineer", "Data Scientist"],
    )
    seniority_level: str | None = Field(
        default=None,
        description="Standardized seniority level",
        examples=["Senior", "Mid-level", "Junior"],
    )
    location: str | None = Field(
        default=None,
        description="Standardized location format",
        examples=["Berlin, Germany", "Remote"],
    )
    salary_range: str | None = Field(
        default=None,
        description="Standardized salary range",
        examples=["€85K-120K", "$100K-150K"],
    )
    company_name: str | None = Field(
        default=None,
        description="Standardized company name",
        examples=["TechCorp Solutions", "Startup Inc."],
    )
    skills: str | None = Field(
        default=None,
        description="Combined technical and professional skills, ordered by importance",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_title": "Senior Software Engineer",
                    "seniority_level": "Senior",
                    "location": "Berlin, Germany",
                    "salary_range": "€85K-120K",
                    "company_name": "TechCorp Solutions",
                    "skills": "Python, Django, FastAPI, PostgreSQL, Redis, Docker, Kubernetes",
                }
            ]
        }
    }


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
    max_retries: int = 5,
    sleep_time: int = 10,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> T | None:
    """Make LLM API calls with retry logic.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        response_format: Pydantic model class for structured response
        max_retries: Maximum number of retry attempts
        sleep_time: Initial sleep time between retries
        max_tokens: Maximum tokens to generate (must be positive)
        temperature: Sampling temperature (must be between 0.0 and 1.0)

    Returns:
        T | None: Parsed response or None if all retries fail

    Raises:
        LLMInputError: If input validation fails
    """
    # Validate messages
    if not messages:
        raise LLMInputError("Messages list cannot be empty")

    for msg in messages:
        if not isinstance(msg, dict):
            raise LLMInputError("Each message must be a dictionary")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in msg.items()):
            raise LLMInputError("Message keys and values must be strings")
        if "role" not in msg or "content" not in msg:
            raise LLMInputError("Each message must have 'role' and 'content' keys")

    # Validate temperature
    if not 0.0 <= temperature <= 1.0:
        raise LLMInputError("Temperature must be between 0.0 and 1.0")

    # Validate max_tokens
    if max_tokens <= 0:
        raise LLMInputError("max_tokens must be positive")

    start_time = time.time()

    system_message = next(
        (msg["content"] for msg in messages if msg["role"] == "system"), ""
    )
    user_message = next(
        (msg["content"] for msg in messages if msg["role"] == "user"), ""
    )

    call_type = str(response_format.__name__)

    system_tokens = len(system_message) // 4  # Approximate 4 chars per token
    user_tokens = len(user_message) // 4
    total_input_tokens = system_tokens + user_tokens

    logging.info(f"Using character-based token estimation for {call_type}")
    logging.info(f"Total input size (estimated): {total_input_tokens} tokens")
    logging.info(
        f"System tokens (estimated): {system_tokens}, User tokens (estimated): {user_tokens}"
    )

    # Adjust timeout based on input tokens and model type
    base_timeout = 60  # Increased base timeout to 60s
    size_based_timeout = min(
        300,  # Increased max timeout to 300s
        base_timeout + (total_input_tokens / 10),  # 1s per 10 tokens
    )
    logging.info(f"Using timeout of {size_based_timeout:.1f}s for {call_type}")

    # Check server health before making request
    def check_server_health():
        try:
            status_response = requests.get(f"{LLM_INSTANCE_URL}/status", timeout=5)
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data.get("status") == "loaded":
                    cache_size = status_data.get("cache_size", 0)
                    logging.info(
                        f"Server is healthy (status: loaded, cache_size: {cache_size})"
                    )
                    # If cache is empty, wait a bit to let it warm up
                    if cache_size == 0 and attempt == 0:
                        logging.info("Cache is empty, waiting 5s for warm-up...")
                        time.sleep(5)
                    return True
                else:
                    logging.warning(
                        f"Server not ready (status: {status_data.get('status', 'unknown')}, "
                        f"cache_size: {status_data.get('cache_size', 0)})"
                    )
                    return False
            logging.warning(
                f"Server status check failed with status {status_response.status_code}"
            )
            return False
        except Exception as error:
            logging.warning(f"Server status check failed: {error}")
            return False

    for attempt in range(max_retries):
        try:
            # Check server health before each attempt
            if not check_server_health():
                if attempt < max_retries - 1:
                    backoff_time = sleep_time * (2**attempt)
                    logging.warning(
                        f"Server not ready, waiting {backoff_time}s before retry..."
                    )
                    time.sleep(backoff_time)
                    continue
                else:
                    raise LLMError("Server not ready after max retries")

            logging.info(
                f"Starting LLM call for {call_type} (attempt {attempt + 1}/{max_retries})"
            )
            attempt_start = time.time()

            # Add exponential backoff between retries
            if attempt > 0:
                backoff_time = sleep_time * (2 ** (attempt - 1))
                logging.info(f"Waiting {backoff_time}s before retry...")
                time.sleep(backoff_time)

            response = LLM_STRUCTURED_CLIENT.messages.create(
                model="",  # Model name is ignored by our custom client
                system=system_message,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=max_tokens,
                temperature=temperature,
                response_model=response_format,
                timeout=size_based_timeout,  # Use size-based timeout
            )

            attempt_duration = time.time() - attempt_start
            logging.info(
                f"LLM call for {call_type} completed in {attempt_duration:.2f}s (attempt {attempt + 1})"
            )

            if attempt > 0:  # Only log if we had to retry
                logging.info(f"Successful LLM call after {attempt + 1} attempts")

            total_duration = time.time() - start_time
            logging.info(
                f"Total processing time for {call_type}: {total_duration:.2f}s"
            )

            return response_format.model_validate(response)

        except requests.exceptions.Timeout:
            attempt_duration = time.time() - attempt_start
            logging.warning(
                f"LLM call attempt {attempt + 1} timed out after {attempt_duration:.2f}s"
            )
            if attempt == max_retries - 1:
                total_duration = time.time() - start_time
                logging.error(
                    f"LLM call for {call_type} timed out after {total_duration:.2f}s total processing time"
                )
                raise LLMError("Request to model server timed out")
            continue

        except requests.exceptions.HTTPError as error:
            if error.response.status_code == 500:
                attempt_duration = time.time() - attempt_start
                logging.warning(
                    f"Server error (500) on attempt {attempt + 1} after {attempt_duration:.2f}s"
                )
                if attempt == max_retries - 1:
                    total_duration = time.time() - start_time
                    logging.error(
                        f"LLM call for {call_type} failed after {total_duration:.2f}s total processing time: {error!s}"
                    )
                    raise LLMError(f"Failed to connect to model server: {error!s}")
                continue
            else:
                total_duration = time.time() - start_time
                logging.error(
                    f"LLM call for {call_type} failed after {total_duration:.2f}s total processing time: {error!s}"
                )
                raise LLMError(f"Failed to connect to model server: {error!s}")

        except Exception as error:
            attempt_duration = time.time() - attempt_start
            logging.warning(
                f"LLM call attempt {attempt + 1} failed after {attempt_duration:.2f}s"
            )

            if "rate_limit" in str(error).lower():
                logging.warning(
                    f"Rate limit hit. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
                if attempt == max_retries - 1:
                    total_duration = time.time() - start_time
                    logging.error(
                        f"Rate limit error for {call_type} after {total_duration:.2f}s total processing time"
                    )
                    raise LLMRateLimitError(
                        "Rate limit exceeded after max retries"
                    ) from error
            else:
                total_duration = time.time() - start_time
                logging.error(
                    f"LLM call for {call_type} failed after {total_duration:.2f}s total processing time: {error!s}"
                )
                raise LLMError(f"LLM call failed: {error!s}") from error

    return None


def job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains any job postings."""
    start_time = time.time()

    class JobPost(BaseModel):
        """Model for job post detection response.

        This model represents the binary classification result for whether a text contains a job posting.
        A job posting must include a specific job title and at least two of the following:
        - Job responsibilities/requirements
        - Application instructions
        - Employment terms (salary, location, work type)
        - Company hiring information
        - Recruiter or hiring manager contacts
        """

        is_job_post: Literal[True, False] = Field(
            ...,
            description="Whether the text contains a job posting",
        )

        model_config = {
            "json_schema_extra": {
                "examples": [{"is_job_post": True}, {"is_job_post": False}]
            }
        }

    messages = [
        {
            "role": "system",
            "content": JOB_POST_DETECTION_PROMPT,
        },
        {
            "role": "user",
            "content": f"Text to classify:\n\n{post}",
        },
    ]

    result = _make_llm_call(messages, JobPost, max_retries, sleep_time)

    processing_time = time.time() - start_time
    logging.info(f"Job post detection completed in {processing_time:.2f}s")

    return result.is_job_post if result else None


def single_job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains exactly one job posting."""
    start_time = time.time()

    class SingleJobPost(BaseModel):
        """Response model for single job post detection."""

        is_single_post: bool = Field(
            description="Whether the text contains exactly one job posting",
            examples=[True, False],
        )

        model_config = {
            "json_schema_extra": {
                "examples": [{"is_single_post": True}, {"is_single_post": False}]
            }
        }

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

    processing_time = time.time() - start_time
    logging.info(f"Single job post detection completed in {processing_time:.2f}s")

    return result.is_single_post if result else None


def match_cv_with_job(
    cv_text: str, post: str, max_retries: int = 5, sleep_time: int = 10
) -> float | None:
    """Evaluates match between CV and job post using separate prompts for each category.

    Args:
        cv_text: The candidate's CV text
        post: The job posting text
        max_retries: Maximum number of retry attempts
        sleep_time: Initial sleep time between retries

    Returns:
        float | None: The final matching score (0-100) or None if evaluation fails
    """
    start_time = time.time()

    try:
        validate_text_input(cv_text, "CV text")
        validate_text_input(post, "Job post")

        max_input_length = 32000
        cv_text = cv_text[:max_input_length]
        post = post[:max_input_length]

        class CategoryMatchScore(BaseModel):
            """Response model for category match score."""

            score: int = Field(
                ge=1,
                le=3,
                description="Match score between 1-3 where 1=excellent, 2=good, 3=poor",
            )

            model_config = {
                "json_schema_extra": {
                    "examples": [{"score": 1}, {"score": 2}, {"score": 3}]
                }
            }

        def evaluate_category(prompt: str, category_name: str) -> int | None:
            messages = [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": f"""
                    CV:
                    {cv_text}

                    Job Post:
                    {post}

                    Provide your response in this exact JSON format:
                    {{
                        "score": <your calculated score>
                    }}

                    IMPORTANT:
                    - Score should be 1, 2, or 3
                    - 1 = Exceeds or meets all requirements
                    - 2 = Meets most requirements
                    - 3 = Misses critical requirements
                    - Calculate score based on actual content analysis
                    - Do not return fixed or default values
                    """,
                },
            ]

            result = _make_llm_call(
                messages,
                CategoryMatchScore,
                max_retries,
                sleep_time,
                max_tokens=512,
                temperature=0.0,
            )

            if result:
                raw_score = result.score
                logging.info(f"{category_name} raw score: {raw_score}")
                return raw_score
            return 2

        def normalize_score(score) -> float:
            """Normalize score to integer in range 1-3."""
            if score is None:
                return 2.0  # Default to worst case for missing scores

            # Handle float values by rounding to nearest integer
            if isinstance(score, float):
                score = round(score)

            # Ensure score is within valid range
            if score < 1:
                return 1
            elif score > 3:
                return 3

            return float(score)

        # Evaluate each category
        experience_score = evaluate_category(EXPERIENCE_MATCHING_PROMPT, "Experience")
        skills_score = evaluate_category(SKILLS_MATCHING_PROMPT, "Skills")
        soft_skills_score = evaluate_category(
            SOFT_SKILLS_MATCHING_PROMPT, "Soft Skills"
        )

        # Check if we got valid scores for at least one category
        if any(
            score is not None
            for score in [experience_score, skills_score, soft_skills_score]
        ):
            # Normalize all scores to valid range
            experience_norm = normalize_score(experience_score)
            skills_norm = normalize_score(skills_score)
            soft_skills_norm = normalize_score(soft_skills_score)

            logging.info(
                f"Normalized scores - Experience: {experience_norm}, Skills: {skills_norm}, Soft Skills: {soft_skills_norm}"
            )

            # Calculate weighted average
            final_score = (
                (skills_norm * 0.45)
                + (experience_norm * 0.40)
                + (soft_skills_norm * 0.15)
            )

            processing_time = time.time() - start_time
            logging.info(
                f"CV matching completed in {processing_time:.2f}s with score: {final_score}"
            )
            return final_score

        return None

    except Exception as error:
        processing_time = time.time() - start_time
        logging.error(
            f"Error during CV matching in {processing_time:.2f}s: {error!s}",
            exc_info=True,
        )
        return None


def job_post_parsing(
    post: str, max_retries: int = 3, sleep_time: int = 10
) -> dict[str, Any] | None:
    """Parse job posting into structured format."""
    start_time = time.time()

    try:
        validate_text_input(post, "Job post")

        messages = [
            {
                "role": "system",
                "content": CLEAN_JOB_POST_PROMPT,
            },
            {
                "role": "user",
                "content": f"Parse this job posting into a structured format:\n\n{post}",
            },
        ]

        try:
            result = _make_llm_call(messages, CleanJobPost, max_retries, sleep_time)
            if not result:
                processing_time = time.time() - start_time
                logging.info(
                    f"Job post parsing completed in {processing_time:.2f}s with no result"
                )
                return None

            # Get initial parsed response
            response = result.model_dump()

            # Log the successful response for debugging
            logging.debug(f"Successfully parsed structured data: {response}")

        except LLMResponseError as error:
            # Attempt manual parsing from raw response if available
            logging.warning(f"Failed to parse structured response: {error}")
            return None

        response_filtered = {
            key: value.strip()
            for key, value in response.items()
            if value and isinstance(value, str)
        }
        if not response_filtered:
            processing_time = time.time() - start_time
            logging.info(
                f"Job post parsing completed in {processing_time:.2f}s but produced invalid result"
            )
            return None

        # Single strip operation only on string values
        processing_time = time.time() - start_time
        logging.info(
            f"Job post parsing completed in {processing_time:.2f}s for job: {response_filtered}"
        )

        return response_filtered

    except LLMInputError as error:
        processing_time = time.time() - start_time
        logging.error(
            f"Input validation failed during job parsing in {processing_time:.2f}s: {error!s}"
        )
        return None
    except Exception as error:
        processing_time = time.time() - start_time
        logging.error(f"Job post parsing failed in {processing_time:.2f}s: {error!s}")
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
    start_time = time.time()

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
            processing_time = time.time() - start_time
            logging.error(
                f"Failed to get valid response from LLM in {processing_time:.2f}s"
            )
            raise LLMResponseError("Failed to get valid response from LLM")

        processing_time = time.time() - start_time
        logging.info(f"Job post cleaning completed in {processing_time:.2f}s")
        return result.model_dump()

    except Exception as error:
        processing_time = time.time() - start_time
        logging.error(
            f"Failed to clean job post values in {processing_time:.2f}s: {error!s}"
        )
        return CleanJobPost(
            job_title=None,
            seniority_level=None,
            location=None,
            salary_range=None,
            company_name=None,
            skills=None,
        ).model_dump()


# For direct text generation without structured output
def generate_text(prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """Generate text using the LLM.

    Args:
        prompt: Text prompt
        max_tokens: Maximum tokens to generate
        temperature: Temperature parameter

    Returns:
        Generated text
    """
    try:
        return CUSTOM_MODEL_CLIENT.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as error:
        logging.error(f"Text generation failed: {error}")
        return ""


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
        "skills": "Python, Django, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS, Microservices, Go, MongoDB, Prometheus, Grafana, TypeScript, React",
    }
    cleaned = clean_job_post_values(sample_response)
    print("Cleaned job post:")
    for key, value in cleaned.items():
        print(f"{key}: {value}")

import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from _production import LLM_BASE_MODEL
from _production.config.config import OPENAI_CLIENT
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


class CleanJobPost(BaseModel):
    job_title: Optional[str]
    seniority_level: Optional[str]
    location: Optional[str]
    remote_status: Optional[str]
    relocation_support: Optional[bool]
    visa_sponsorship: Optional[bool]
    salary_range: Optional[str]
    company_name: Optional[str]
    description: Optional[str]


def validate_text_input(
    text: str, field_name: str, max_length: int = MAX_TEXT_LENGTH
) -> None:
    """Validate text input

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
    messages: List[Dict[str, str]],
    response_format: Type[T],
    max_retries: int = 3,
    sleep_time: int = 10,
) -> Optional[T]:
    """Make OpenAI API calls with retry logic"""
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
                    ChatCompletionSystemMessageParam(
                        role="system", content=msg["content"]
                    )
                    if msg["role"] == "system"
                    else ChatCompletionUserMessageParam(
                        role="user", content=msg["content"]
                    )
                    for msg in messages
                ],
                temperature=0.0,
                response_format=response_format,
            )
            logging.info(f"Successful LLM call after {attempt + 1} attempts")
            return response.choices[0].message.parsed

        except Exception as error:
            if "Too Many Requests" in str(error):
                logging.warning(
                    f"Rate limit hit. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
                if attempt == max_retries - 1:
                    raise LLMRateLimitError("Rate limit exceeded after max retries")
            else:
                logging.error("Error in LLM call", exc_info=True)
                raise LLMError(f"LLM call failed: {str(error)}")

    return None


def job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains any job postings"""

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
    """Determines if the text contains exactly one job posting"""

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
) -> Optional[float]:
    """Evaluates match between CV and job post"""
    try:
        validate_text_input(cv_text, "CV text")
        validate_text_input(post, "Job post")

        class CVMatch(BaseModel):
            score: float

        messages = [
            {
                "role": "system",
                "content": CV_MATCHING_PROMPT,
            },
            {
                "role": "user",
                "content": f"CV:\n{cv_text}\n\nJob Post:\n{post}",
            },
        ]

        result = _make_llm_call(messages, CVMatch, max_retries, sleep_time)
        return result.score if result else None
    except LLMInputError as e:
        logging.error(f"Input validation failed: {str(e)}")
        return None


def job_post_parsing(
    post: str, max_retries: int = 3, sleep_time: int = 10
) -> Optional[Dict[str, Any]]:
    """Parse job posting into structured format"""
    try:
        validate_text_input(post, "Job post")

        class JobPostStructure(BaseModel):
            job_title: str
            seniority_level: str
            location: str
            remote_status: str
            relocation_support: Optional[bool]
            visa_sponsorship: Optional[bool]
            salary_range: Optional[str]
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
        if not cleaned_response or "job_title" not in cleaned_response:
            return None

        # Single strip operation only on string values
        return {
            key: value.strip()
            for key, value in cleaned_response.items()
            if isinstance(value, str)
        }
    except LLMInputError as error:
        logging.error(f"Input validation failed: {str(error)}")
        return None


def clean_job_post_values(response: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and standardize job post values

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
    except Exception as e:
        logging.error(f"Failed to clean job post values: {str(e)}")
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

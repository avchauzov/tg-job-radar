import logging
import time
from pydantic import BaseModel
from typing import Optional
import json

from _production import LLM_BASE_MODEL
from _production.config.config import OPENAI_CLIENT
from _production.utils.prompts import (
    CLEAN_JOB_POST_PROMPT,
    CV_MATCHING_PROMPT,
    JOB_POST_DETECTION_PROMPT,
    JOB_POST_PARSING_PROMPT,
    SINGLE_JOB_POST_DETECTION_PROMPT,
)


def _make_llm_call(messages, response_format, max_retries=3, sleep_time=10):
    """Helper function to make OpenAI API calls with retry logic"""
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=messages,
                temperature=0.0,
                response_format=response_format,
            )
            return response.choices[0].message.parsed

        except Exception as error:
            if "Too Many Requests" in str(error):
                logging.warning(
                    f"Received 429 error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
            else:
                logging.error("Error in LLM call", exc_info=True)
                raise

    logging.error(f"Failed after {max_retries} attempts")
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


def match_cv_with_job(cv_text: str, post: str, max_retries=3, sleep_time=10):
    """Evaluates match between CV and job post, returns score 0-100"""

    class CVMatch(BaseModel):
        score: float

    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": CV_MATCHING_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"CV:\n{cv_text}\n\nJob Post:\n{post}",
                    },
                ],
                temperature=0.0,
                response_format=CVMatch,
            )

            return response.choices[0].message.parsed.score

        except Exception as error:
            if "Too Many Requests" in str(error):
                logging.warning(
                    f"Received 429 error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)

            else:
                logging.error("Error matching CV", exc_info=True)
                raise

    logging.error(f"Failed to match CV after {max_retries} attempts\n{post}")
    return None


def clean_job_post_values(response: dict) -> dict:
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
        return result.model_dump() if result else None
    except Exception as e:
        logging.error(f"Failed to clean job post values: {str(e)}")
        # Return a valid empty structure matching CleanJobPost schema
        return {
            "job_title": None,
            "seniority_level": None,
            "location": None,
            "remote_status": None,
            "relocation_support": None,
            "visa_sponsorship": None,
            "salary_range": None,
            "company_name": None,
            "description": None,
        }


def job_post_parsing(post, max_retries=3, sleep_time=10):
    """Parse job posting into structured format with cleaned values"""
    """Clean and standardize job post values using LLM"""

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
        key: value.strip() if isinstance(value, str) else value
        for key, value in cleaned_response.items()
    }

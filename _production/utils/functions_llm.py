import logging
import time

from pydantic import BaseModel
from typing import Optional

from _production import LLM_BASE_MODEL
from _production.config.config import OPENAI_CLIENT


class JobPost(BaseModel):
    is_job_description: bool


class SingleJobPost(BaseModel):
    is_single_post: bool


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


class JobURLResult(BaseModel):
    url: list[str]
    is_direct_job_description: list[int]


class CVMatch(BaseModel):
    score: float


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
    messages = [
        {
            "role": "system",
            "content": """You are an expert at analyzing job-related content. 
            Determine if a text contains ANY job postings.

            A job posting MUST include:
                - Specific job title(s)
            
            AND at least one of:
                - Job responsibilities/requirements
                - Application instructions
                - Employment terms
                - Company hiring information
                - recruiter or hiring manager contacts
            
            Do NOT classify as job postings:
                - General career advice
                - Industry news
                - Company updates without hiring intent
                - Educational content
                - Network/community building posts
            
            Respond only with "True" or "False".""",
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
    messages = [
        {
            "role": "system",
            "content": """You are an expert at analyzing job postings. 
            Determine if a text contains EXACTLY ONE job posting.

            Indicators of a single job posting:
                - One clear job title
                - Consistent requirements for one role
                - Single set of qualifications
            
            Indicators of multiple job postings:
                - Multiple distinct job titles
                - Different sets of requirements
                - "Multiple positions available"
                - Lists of different roles
                - Separate sections for different positions
            
            Respond only with "True" for single job posts or "False" for multiple job posts.""",
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
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an experienced and strict technical recruiter. 
						Evaluate how well a candidate's CV matches a job posting requirements.
						
						Consider:
                            - Required and desired technical skills match
                            - Required and desired domain knowledge
                            - Required seniority level match
                            - Required experience match
                            - Years of experience match
                            - Education requirements if specified
						
						Return a score from 0-100 where:
						- 90-100: Perfect match of required and desired
						- 70-89: Strong match, meets most required
						- 50-69: Moderate match, meets some key requirements
						- 0-49: Weak match, missing critical requirements
						
						Be strict and objective in your evaluation.""",
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


def job_post_parsing(post, max_retries=3, sleep_time=10):
    """Parse job posting into structured format"""
    messages = [
        {
            "role": "system",
            "content": """You are an expert at parsing job descriptions. 
            Extract and structure job posting information accurately.

            Rules:
                - If information is not provided, use None
                - Normalize seniority levels to: Junior, Mid-Level, Senior, Lead, Principal, or Executive
                - For remote_status, use only: "Remote", "Hybrid", "On-site"
                - Keep the description concise but include all important details and required skills
                - Extract salary range if mentioned, standardize format""",
        },
        {
            "role": "user",
            "content": f"Parse this job posting into a structured format:\n\n{post}",
        },
    ]

    result = _make_llm_call(messages, JobPostStructure, max_retries, sleep_time)
    if not result:
        return None

    response = result.model_dump()
    response = {
        key: value.strip()
        for key, value in response.items()
        if isinstance(value, str)
        and any(char.isalnum() for char in value)
        and value.strip()
        and value.strip().lower() not in ["none", "null", "не указано"]
    }
    return response

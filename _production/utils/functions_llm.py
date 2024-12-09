import logging
import time

from pydantic import BaseModel
from typing import Optional, List

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
    remote_status: str  # "Remote" / "Hybrid" / "On-site"
    relocation_support: Optional[bool]
    visa_sponsorship: Optional[bool]
    salary_range: Optional[str]
    company_name: str
    description: str


class JobURLResult(BaseModel):
    url: list[str]
    is_direct_job_description: list[int]


class CVMatch(BaseModel):
    score: float  # 0-100 score


def job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains any job postings"""
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing job-related content. Determine if a text contains ANY job postings.

A job posting MUST include:
- Specific job title(s)
AND at least one of:
- Job responsibilities/requirements
- Application instructions
- Employment terms
- Company hiring information

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
                ],
                temperature=0.0,
                response_format=JobPost,
            )

            return (
                True if response.choices[0].message.parsed.is_job_description else False
            )

        except Exception as error:
            if "Too Many Requests" in str(error):
                logging.warning(
                    f"Received 429 error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
            else:
                logging.error(f"Error detecting job post: {error}")
                return True  # Err on the side of caution

    logging.error(f"Failed to detect job post after {max_retries} attempts")
    return True


def single_job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains exactly one job posting"""
    # If it's not a job post at all, return False
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing job postings. Determine if a text contains EXACTLY ONE job posting.

Indicators of a single job posting:
- One clear job title
- Consistent requirements for one role
- Single set of qualifications
- One location/department

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
                ],
                temperature=0.0,
                response_format=SingleJobPost,
            )

            return response.choices[0].message.parsed.is_single_post

        except Exception as error:
            if "Too Many Requests" in str(error):
                logging.warning(
                    f"Received 429 error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
            else:
                logging.error(f"Error detecting single job post: {error}")
                return False  # Err on the side of caution

    logging.error(f"Failed to detect single job post after {max_retries} attempts")
    return False


def reformat_post(post, max_retries=3, sleep_time=10):
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at parsing job descriptions. Extract and structure job posting information accurately.
						
Rules:
- If information is not provided, use None
- Normalize seniority levels to: Junior, Mid-Level, Senior, Lead, Principal, or Executive
- For remote_status, use only: "Remote", "Hybrid", "On-site"
- Keep the description concise but include all important details
- Extract salary range if mentioned, standardize format""",
                    },
                    {
                        "role": "user",
                        "content": f"Parse this job posting into a structured format:\n\n{post}",
                    },
                ],
                temperature=0.0,
                response_format=JobPostStructure,
            )
            logging.info(response.choices[0].message.parsed.model_dump())
            return response.choices[0].message.parsed.model_dump()

        except Exception as error:
            if "Too Many Requests" in str(error):
                logging.warning(
                    f"Received 429 error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
            else:
                logging.error(f"Error parsing job post: {error}")
                return create_empty_job_structure()

    logging.error(f"Failed to parse job post after {max_retries} attempts")
    return create_empty_job_structure()


def match_cv_with_job(cv_text: str, job_post: str, max_retries=3, sleep_time=10):
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
						- Required technical skills match
						- Years of experience match
						- Required seniority level match
						- Required domain knowledge
						- Education requirements if specified
						
						Return a score from 0-100 where:
						- 90-100: Perfect match, exceeds requirements
						- 70-89: Strong match, meets most requirements
						- 50-69: Moderate match, meets some key requirements
						- 0-49: Weak match, missing critical requirements
						
						Be strict and objective in your evaluation.""",
                    },
                    {
                        "role": "user",
                        "content": f"CV:\n{cv_text}\n\nJob Post:\n{job_post}",
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
                logging.error(f"Error matching CV: {error}")
                return 0  # Return 0 score on error

    logging.error(f"Failed to match CV after {max_retries} attempts")
    return 0


def filter_job_urls(description, url_list, max_retries=3, sleep_time=10):
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Task:\n"
                            "For each URL in the provided list, determine if it likely leads to a direct job description (1) "
                            "or not (0). A direct job description URL should contain specific job information and an option "
                            "to apply directly. Exclude URLs leading to general company pages, career listing pages, or "
                            '"other jobs" pages. Return 1 for direct job descriptions and 0 otherwise. '
                            "Ensure the output order matches the input URL order."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Job post: {description}\n\nURLs:\n{url_list}",
                    },
                ],
                temperature=0.0,
                response_format=JobURLResult,
            )

            filtered_urls = [
                url
                for url, mask in zip(
                    url_list,
                    response.choices[0].message.parsed.is_direct_job_description,
                )
                if mask
            ]
            return filtered_urls

        except Exception as error:
            if "Too Many Requests" in str(error):
                logging.warning(
                    f"Received 429 error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)

            else:
                logging.error(f"Error filtering job URLs: {error}")
                return url_list

    logging.error(f"Failed to filter URLs after {max_retries} attempts")
    return url_list


def create_empty_job_structure():
    """Creates an empty job structure with default values for error cases"""
    return JobPostStructure(
        job_title="Unknown",
        seniority_level="Not specified",
        location="Not specified",
        remote_status="Not specified",
        relocation_support=None,
        visa_sponsorship=None,
        salary_range=None,
        company_name="Unknown",
        description="Original post parsing failed",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    job_description = "We're hiring a Software Engineer to join our team. Apply now!"
    test_urls = [
        "https://example.com/job/software-engineer-123",
        "https://example.com/careers",
        "https://example.com/company/about",
        "https://example.com/jobs/engineering/software-engineer",
    ]

    # filtered_urls = filter_job_urls(job_description, test_urls)
    # print(filtered_urls)

    job_post = job_post_detection(job_description)
    print(job_post)

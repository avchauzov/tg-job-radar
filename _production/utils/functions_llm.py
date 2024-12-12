import json
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


def job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains any job postings"""
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
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
                ],
                temperature=0.0,
                response_format=JobPost,
            )

            return response.choices[0].message.parsed.is_job_description

        except Exception as error:
            if "Too Many Requests" in str(error):
                logging.warning(
                    f"Received 429 error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)

            else:
                logging.error(f"Error detecting job post: {error}\n{post}")
                raise Exception(f"Error detecting job post: {error}\n{post}")

    logging.error(f"Failed to detect job post after {max_retries} attempts\n{post}")
    return False


def single_job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains exactly one job posting"""
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
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
                logging.error(f"Error detecting single job post: {error}\n{post}")
                raise Exception(f"Error detecting single job post: {error}\n{post}")

    logging.error(
        f"Failed to detect single job post after {max_retries} attempts\n{post}"
    )
    return False


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
                logging.error(f"Error matching CV: {error}\n{post}")
                raise Exception(f"Error matching CV: {error}\n{post}")

    logging.error(f"Failed to match CV after {max_retries} attempts\n{post}")
    return 0


def job_post_parsing(post, max_retries=3, sleep_time=10):
    for attempt in range(max_retries):
        try:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model=LLM_BASE_MODEL,
                messages=[
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
                ],
                temperature=0.0,
                response_format=JobPostStructure,
            )

            response = response.choices[0].message.parsed.model_dump()
            response = {
                key: value.strip()
                for key, value in response.items()
                if isinstance(value, str)
                and any(
                    char.isalnum() for char in value
                )  # Ensure at least one alphanumeric character
                and value.strip()  # Ensure the stripped value is not empty
            }
            return response

        except Exception as error:
            if "Too Many Requests" in str(error):
                logging.warning(
                    f"Received 429 error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
                continue

            logging.error(f"Error parsing job post: {error}\n{post}")
            raise Exception(f"Error parsing job post: {error}\n{post}")

    logging.error(f"Failed to parse job post after {max_retries} attempts\n{post}")
    return json.dumps({})


# TODO: ask to revise the script
# TODO: create separate test cases and put into CircleCI
